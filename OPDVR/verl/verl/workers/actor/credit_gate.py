# -*- coding: utf-8 -*-
"""E030 credit gate: Adam-aligned per-token mask for sampled-token OPD.

For token t of response i (state s_t, sampled token o_t), with the teacher push
    ell_t = log pi_T(o_t|s_t) - log pi_theta(o_t|s_t)          (= rm_scores in the OPDVR fork)
the credit is the first-order change of the group-relative RL objective along the push,

    c_t = ell_t * < grad_theta log pi_theta(o_t|s_t), u_s >,

where u_s is the "GRPO direction" estimated from the shadow Adam moments of the Dr.GRPO
tangent G_s = sum_i (A_i/m_i) sum_t grad log pi(o_{i,t}|s_{i,t}),  A_i = R_i - mean_group(R):

    m_s = b1 m_{s-1} + (1-b1) G_s,   v_s = b2 v_{s-1} + (1-b2) G_s^2   (bias-corrected, bf16 storage)
    u_s = mhat_{s-1} / (sqrt(vhat_{s-1}) + adam_eps)                    (PRE-update: G_s is not in u_s)

The directional derivative is a central finite difference on the served weights,

    < grad log pi(o_t|s_t), u > ~= [log pi_{theta+eps u}(o_t|s_t) - log pi_{theta-eps u}(o_t|s_t)] / (2 eps),

two ordinary forwards; eps is fixed by the pre-flight calibration (protocol_e030 section 4).
Mask: w_t = 1[c_t > 0]. Step 0 (no moments yet) -> w == 1.

Everything here operates on the LOCALLY OWNED parameter storage (FSDP1 flat-param shards,
FSDP2 DTensor locals, or plain parameters); dot products are all-reduced, so the returned
scalars are global. Perturbation is applied to the storage dtype: with fp32 master weights
under FSDP mixed precision the forward casts (theta + eps u) to bf16 -- exactly the
"perturb the master, then cast" path the calibration measured.
"""
import math

import torch
import torch.distributed as dist


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


class ShardView:
    """Locally-owned parameter storage + gradient, uniformly for FSDP1 / FSDP2 / plain modules.

    FSDP1 with use_orig_params=False registers one FlatParameter per FSDP unit as a regular parameter
    (`..._flat_param`), so `module.parameters()` already yields the local shards and their sharded grads;
    FSDP2 parameters are DTensors (use `.to_local()`); plain modules are plain."""

    def __init__(self, module):
        self.module = module
        self.kind = "plain"
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if isinstance(module, FSDP):
                self.kind = "fsdp1"
        except Exception:  # noqa: BLE001
            pass
        self._params = list(module.parameters())          # no requires_grad filter: the FD copy has none
        try:
            from torch.distributed.tensor import DTensor
            if any(isinstance(p, DTensor) for p in self._params):
                self.kind = "fsdp2"
        except Exception:  # noqa: BLE001
            pass
        assert self._params, "ShardView: module has no parameters"

    def _raw(self):
        return self._params

    def params(self):
        """Local storage tensors (views; in-place ops hit the real weights)."""
        for p in self._raw():
            if self.kind == "fsdp2" and hasattr(p, "to_local"):
                yield p.to_local()
            else:
                yield p.data
        return

    def grads(self):
        for p in self._raw():
            g = p.grad
            if g is None:
                yield None
            elif self.kind == "fsdp2" and hasattr(g, "to_local"):
                yield g.to_local()
            else:
                yield g

    def zero_grads(self):
        for p in self._raw():
            p.grad = None

    def numel_local(self):
        return sum(p.numel() for p in self.params())


class CreditGate:
    def __init__(self, module, eps=1e-6, beta1=0.9, beta2=0.999, adam_eps=1e-8,
                 moments_dtype=torch.bfloat16, split_half=True, lowp_fracs=(0.1, 0.2), fd_module=None):
        self.view = ShardView(module)
        # optional fp32 copy of `module` with an identical shard layout: the finite difference is evaluated
        # there (calibration 09-10: bf16/TF32 round eps=1e-6 away; fp32 passes with sign agreement 0.986)
        self.view_fd = ShardView(fd_module) if fd_module is not None else None
        if self.view_fd is not None:
            na = [p.numel() for p in self.view.params()]
            nf = [p.numel() for p in self.view_fd.params()]
            assert na == nf, "fp32 FD copy does not match the actor's shard layout: %s vs %s" % (na[:3], nf[:3])
        self.eps, self.b1, self.b2, self.adam_eps = float(eps), float(beta1), float(beta2), float(adam_eps)
        self.split_half = bool(split_half)
        self.lowp_fracs = tuple(float(f) for f in lowp_fracs)
        self.m = [torch.zeros_like(p, dtype=moments_dtype) for p in self.view.params()]
        self.v = [torch.zeros_like(p, dtype=moments_dtype) for p in self.view.params()]
        self.n_upd = 0
        self.numel = _all_reduce_sum(torch.tensor(float(self.view.numel_local()),
                                                  device=self.m[0].device if self.m else "cpu")).item()

    # ------------------------------------------------------------------ vector ops (global)
    def dot(self, A, B):
        dev = A[0].device
        acc = torch.zeros((), dtype=torch.float64, device=dev)
        for a, b in zip(A, B):
            acc += (a.double() * b.double()).sum()
        return float(_all_reduce_sum(acc).item())

    def norm(self, A):
        return math.sqrt(max(self.dot(A, A), 0.0))

    def cos(self, A, B):
        return self.dot(A, B) / max(self.norm(A) * self.norm(B), 1e-30)

    # ------------------------------------------------------------------ grads
    def snapshot_grads(self):
        out = []
        for p, g in zip(self.view.params(), self.view.grads()):
            # Sign convention: the upstream loss is -(w*lp*mask).sum(), so .grad = -G;
            # G is defined as the ascent direction sum_i (A_i/m_i) sum_t grad log pi (see the header), hence the negation here.
            out.append(torch.zeros_like(p, dtype=torch.float32) if g is None else g.detach().to(torch.float32).neg())
        return out

    def zero_grads(self):
        self.view.zero_grads()

    # ------------------------------------------------------------------ moments / direction
    @torch.no_grad()
    def direction(self):
        """u = mhat/(sqrt(vhat)+eps) from the moments accumulated so far (None before the first update)."""
        if self.n_upd == 0:
            return None
        bc1 = 1.0 - self.b1 ** self.n_upd
        bc2 = 1.0 - self.b2 ** self.n_upd
        return [((m.float() / bc1) / ((v.float() / bc2).sqrt() + self.adam_eps)) for m, v in zip(self.m, self.v)]

    @torch.no_grad()
    def direction_from_optimizer(self, optimizer, beta2=None):
        """E031: u = m_hat / (sqrt(v_hat^L) + adam_eps), with v^L the actor AdamW's OWN second moment
        (same metric as the update; bias-corrected with the optimizer's step count). None before the
        first optimizer step or before the first moment update -> caller trains plain (w == 1)."""
        if self.n_upd == 0 or optimizer is None:
            return None
        bc1 = 1.0 - self.b1 ** self.n_upd
        u = []
        for p, m in zip(self.view._raw(), self.m):
            st = optimizer.state.get(p, None)
            if st is None or "exp_avg_sq" not in st:
                return None
            v = st["exp_avg_sq"]
            v = v.to_local() if hasattr(v, "to_local") else v
            step = st.get("step", None)
            step = float(step.item() if torch.is_tensor(step) else (step or 1))
            b2 = float(beta2 if beta2 is not None else optimizer.param_groups[0]["betas"][1])
            vhat = v.float() / max(1.0 - b2 ** step, 1e-12)
            u.append((m.float() / bc1) / (vhat.sqrt() + self.adam_eps))
        return u

    @torch.no_grad()
    def update_moments(self, G):
        for m, v, g in zip(self.m, self.v, G):
            gf = g.float()
            m.copy_((m.float() * self.b1 + gf * (1.0 - self.b1)).to(m.dtype))
            v.copy_((v.float() * self.b2 + gf * gf * (1.0 - self.b2)).to(v.dtype))
        self.n_upd += 1

    @torch.no_grad()
    def sync_fd(self):
        """Copy the actor's local shards into the fp32 FD copy (no communication: same layout)."""
        if self.view_fd is None:
            return
        for pa, pf in zip(self.view.params(), self.view_fd.params()):
            pf.copy_(pa.to(pf.dtype))

    @torch.no_grad()
    def perturb(self, u, alpha):
        """theta <- theta + alpha * u on the FD copy if present, else on the module itself."""
        view = self.view_fd if self.view_fd is not None else self.view
        for p, ui in zip(view.params(), u):
            p.add_(ui.to(p.dtype), alpha=float(alpha))

    # ------------------------------------------------------------------ mask + telemetry
    @staticmethod
    def signed_share(c, mask):
        """sum_t c_t / sum_t |c_t| over mask, global (all-reduced)."""
        num = (c * mask).sum().double()
        den = (c.abs() * mask).sum().double()
        num, den = _all_reduce_sum(num), _all_reduce_sum(den)
        return float(num / den.clamp_min(1e-30))

    @staticmethod
    def global_quantile(x, mask, q):
        """q-quantile of x over mask, taken over ALL ranks (all_gather; equal shapes per rank)."""
        vals = x[mask.bool()]
        if _is_dist():
            n = torch.tensor([vals.numel()], device=x.device)
            ns = [torch.zeros_like(n) for _ in range(dist.get_world_size())]
            dist.all_gather(ns, n)
            nmax = int(max(int(t.item()) for t in ns))
            buf = torch.full((nmax,), float("nan"), device=x.device, dtype=torch.float32)
            buf[:vals.numel()] = vals.float()
            bufs = [torch.zeros_like(buf) for _ in range(dist.get_world_size())]
            dist.all_gather(bufs, buf)
            allv = torch.cat([b[:int(k.item())] for b, k in zip(bufs, ns)])
        else:
            allv = vals.float()
        if allv.numel() == 0:
            return float("nan")
        # kthvalue, not torch.quantile: quantile refuses inputs above 16M elements (1024 x 16K tokens can exceed it)
        k = int(max(1, min(allv.numel(), round(q * allv.numel()))))
        return float(allv.kthvalue(k).values.item())


# ----------------------------------------------------------------------------- E031: influence-weighted gate
@torch.no_grad()
def iw_gate(iota, mask, d, p_y, active_frac=0.2, w_max=3.0, f_hi=0.05, f_lo=0.15, lam_cap=4.0, last_tok=None, lam_fixed=0.0, w_min=0.0):
    """w_t = clip(1 + lam * z_t, w_min, w_max) on the active set S = {d != 0, p(y_t) <= q_active(p | d != 0)};
    w_t = 1 outside S.  z_t = iota_t / RMS_S(iota).  lam = min(lam_hi, lam_lo) from two-sided clip budgets:
    lam_hi = (w_max - 1) / Q_z(1 - f_hi)  (share capped <= f_hi),  lam_lo = -1 / Q_z(f_lo)  (share zeroed <= f_lo).
    All quantiles / sums are global (all-reduced / all-gathered). Returns (w, stats)."""
    valid = mask.bool() & (d != 0)
    p_thr = CreditGate.global_quantile(p_y, valid, active_frac)
    S = valid & (p_y <= p_thr)
    Sf = S.float()
    nS = _all_reduce_sum(Sf.sum().double())
    sig = torch.sqrt(_all_reduce_sum((iota.double() ** 2 * Sf).sum()) / nS.clamp_min(1)).float()
    z = iota / sig.clamp_min(1e-30)
    q = {f: CreditGate.global_quantile(z, S, f) for f in (0.05, 0.10, 0.50, 0.90, 0.95, 1.0 - f_hi, f_lo)}
    lam_hi = (w_max - 1.0) / q[1.0 - f_hi] if q[1.0 - f_hi] > 0 else float("inf")
    lam_lo = -1.0 / q[f_lo] if q[f_lo] < 0 else float("inf")
    lam = float(lam_fixed) if lam_fixed and lam_fixed > 0 else min(lam_hi, lam_lo, lam_cap)   # E031: fixed lambda (user 09-13) or budgets
    w = torch.where(S, (1.0 + lam * z).clamp(float(w_min), w_max), torch.ones_like(z))
    w = w * mask
    capped = S & (w >= w_max - 1e-6)
    zeroed = S & (w <= float(w_min) + 1e-9)   # tokens pushed to the lower clip (which is not necessarily 0)
    stats = {"credit/lam": lam, "credit/lam_hi": min(lam_hi, 1e6), "credit/lam_lo": min(lam_lo, 1e6),
             "credit/w_min": float(w_min), "credit/sigma_S": float(sig), "credit/p_thr_active": p_thr, "credit/n_active": float(nS),
             "credit/frac_active": float(nS / _all_reduce_sum(mask.sum().double()).clamp_min(1)),
             "credit/frac_w0_S": float(_all_reduce_sum(zeroed.sum().double()) / nS.clamp_min(1)),
             "credit/frac_wmax_S": float(_all_reduce_sum(capped.sum().double()) / nS.clamp_min(1)),
             "credit/w_mean_S": float(_all_reduce_sum((w * Sf).sum().double()) / nS.clamp_min(1)),
             "credit/z_q05": q[0.05], "credit/z_q10": q[0.10], "credit/z_q50": q[0.50], "credit/z_q90": q[0.90], "credit/z_q95": q[0.95]}
    # tail telemetry (matters when the cap is loose): how much weight mass sits far from 1
    nSc = nS.clamp_min(1)
    stats["credit/frac_w_gt15_S"] = float(_all_reduce_sum((S & (w > 1.5)).sum().double()) / nSc)
    stats["credit/frac_w_gt2_S"] = float(_all_reduce_sum((S & (w > 2.0)).sum().double()) / nSc)
    wmax_local = torch.tensor(float(w.max()) if w.numel() else 0.0, device=w.device, dtype=torch.float64)
    if _is_dist():
        dist.all_reduce(wmax_local, op=dist.ReduceOp.MAX)
    stats["credit/w_max_obs"] = float(wmax_local)
    if last_tok is not None:      # share of capped tokens that are their response's last token (length-axis proxy)
        nc = _all_reduce_sum(capped.sum().double())
        stats["credit/capped_last_tok_share"] = float(_all_reduce_sum((capped & last_tok.bool()).sum().double()) / nc.clamp_min(1))
        hi = S & (w > 1.5)
        stats["credit/w_gt15_last_tok_share"] = float(_all_reduce_sum((hi & last_tok.bool()).sum().double()) / _all_reduce_sum(hi.sum().double()).clamp_min(1))
    return w, stats
