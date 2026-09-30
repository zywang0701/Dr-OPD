# -*- coding: utf-8 -*-
"""JVP token credits: the per-token influence D_t = <u, grad_theta log pi(y_t|s_t)> from one
forward-mode pass on a bf16, NON-FSDP replica of the student, instead of two fp32 finite-difference
forwards on the sharded copy.

Why this is allowed in bf16 while the finite difference is not: forward-mode AD carries the tangent
as its own tensor (dual numbers), so each tensor only needs ~0.4 % relative precision of ITSELF;
the finite difference needs the 1e-6-relative perturbation to survive inside the primal, which bf16
and TF32 round away.

Pieces:
  * replica        plain HF model on every GPU, bf16, eval, no grad, chunked eager attention
                   (math-identical to eager, memory linear in L, JVP-traversable; flash/SDPA kernels have
                   no forward-AD rule)
  * sync_weights   FULL_STATE_DICT gather from the FSDP actor -> replica params (bf16)
  * gather_tangent the direction u lives sharded like the flat params; swap u into the flat params,
                   run the same FULL_STATE_DICT gather, swap back -> per-parameter full u (bf16)
  * influence      per sequence (trimmed to its valid span), torch.func.jvp through functional_call,
                   output = log-softmax at the sampled tokens, so the tangent output IS D_t

Costs per step (4B, 8 GPUs): two 16 GB all-gathers (weights, u), one forward-mode pass (~3 bf16
forward-equivalents with eager attention at mean length ~3K). Resident: replica 8 GB + u 8 GB.

The CPU test (tests/test_jvp_influence_cpu.py) covers the influence math, chunked attention
registration and trimming against an fp32 finite difference on a plain model. It does not
cover FSDP synchronization or long-sequence GPU memory requirements.
"""
import torch
import torch.nn.functional as F

_QBLOCK = 1024


# ----------------------------------------------------------------------------- attention (JVP-safe)
def _chunked_eager_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers' eager_attention_forward computed in query blocks: scores are [B,H,qblock,L]
    instead of [B,H,L,L]. Plain ops only, so torch.func.jvp can traverse it."""
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    if scaling is None:
        scaling = module.head_dim ** -0.5
    L = query.shape[2]
    Lk = key_states.shape[-2]
    outs = []
    for i in range(0, L, _QBLOCK):
        jq = min(i + _QBLOCK, L)
        kend = Lk - L + jq      # speedup: under causal attention no row of this block can see keys after kend, so they are cut (numerically identical, about half the attention work)
        q = query[:, :, i:jq]
        w = torch.matmul(q, key_states[:, :, :kend].transpose(2, 3)) * scaling
        if attention_mask is not None:
            mblk = attention_mask[:, :, i:jq, :kend]
            w = w.masked_fill(~mblk, torch.finfo(w.dtype).min) if mblk.dtype == torch.bool else w + mblk
        else:                                   # no mask handed in: enforce causality ourselves
            causal = torch.ones(jq - i, kend, dtype=torch.bool, device=w.device).tril(diagonal=Lk - L + i)
            w = w.masked_fill(~causal, torch.finfo(w.dtype).min)
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        outs.append(torch.matmul(w, value_states[:, :, :kend]))
    out = torch.cat(outs, dim=2).transpose(1, 2).contiguous()
    return out, None


def register_chunked_eager():
    """Register the attention AND its mask builder. transformers picks the mask by implementation name:
    an unregistered name falls back to the sdpa mask (boolean, or None when there is no padding), which
    would make this attention non-causal / add 0-1 instead of -inf. The eager float mask is what we need."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS._global_mapping.setdefault("opd_chunked_eager", _chunked_eager_attention)
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.setdefault("opd_chunked_eager", eager_mask)
    except ImportError:      # older transformers without the mask registry: the eager 4-D float mask is passed through
        pass


# ----------------------------------------------------------------------------- the module
class JVPInfluence:
    def __init__(self, model_path, device, dtype=torch.bfloat16, trust_remote_code=False, qblock=1024):
        global _QBLOCK
        _QBLOCK = int(__import__("os").environ.get("CREDIT_JVP_QBLOCK", max(int(qblock), 2048)))   # query block size, 2048 by default
        from transformers import AutoModelForCausalLM
        register_chunked_eager()
        self.replica = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="eager", trust_remote_code=trust_remote_code).to(device)
        self.replica.config._attn_implementation = "opd_chunked_eager"
        self.replica.eval()
        for p in self.replica.parameters():
            p.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.names = [n for n, _ in self.replica.named_parameters()]
        self.u_full = None

    # ---- weights: FSDP actor -> replica
    @torch.no_grad()
    def sync_weights(self, actor_module_fsdp):
        sd = _full_state_dict(actor_module_fsdp)
        params = dict(self.replica.named_parameters())
        missing = [n for n in self.names if n not in sd]
        assert not missing, "replica params missing from actor state dict, e.g. %s" % missing[:3]
        for n, p in params.items():
            p.copy_(sd[n].to(p.dtype))
        del sd
        torch.cuda.empty_cache()

    # ---- tangent: sharded (flat-param layout) -> per-parameter full tensors on this rank
    @torch.no_grad()
    def gather_tangent(self, shard_view, u_shards):
        """Swap u into the actor's flat-param shards, run the FULL_STATE_DICT gather, swap back.
        Relies on the gather reading whatever is in the local shards (it does: it all-gathers them)."""
        saved = []
        for p, ui in zip(shard_view.params(), u_shards):
            saved.append(p.detach().clone())
            p.copy_(ui.to(p.dtype))
        try:
            sd = _full_state_dict(shard_view.module)
        finally:
            for p, s in zip(shard_view.params(), saved):
                p.copy_(s)
            del saved
        self.u_full = {n: sd[n].to(self.dtype) for n in self.names}
        del sd
        torch.cuda.empty_cache()
        return self.u_full

    # ---- influence
    @torch.no_grad()
    def influence(self, input_ids, attention_mask, position_ids, responses, u_full=None, need=None):
        """Returns D [B, R] (fp32): <u, grad log pi(y_t|s_t)> per response token, 0 on padding.
        Layout = verl's: [prompt left-padded | response right-padded], attention_mask on valid tokens."""
        from torch.func import functional_call, jvp
        u_full = self.u_full if u_full is None else u_full
        params = {n: p for n, p in self.replica.named_parameters()}
        buffers = {n: b for n, b in self.replica.named_buffers()}
        B, R = responses.shape
        out = torch.zeros(B, R, device=input_ids.device, dtype=torch.float32)
        for b in range(B):
            am = attention_mask[b]
            valid = am.nonzero(as_tuple=True)[0]
            if valid.numel() == 0:
                continue
            lo, hi = int(valid[0]), int(valid[-1]) + 1
            n_resp = int(am[-R:].sum())
            if n_resp == 0:
                continue
            # speedup: when need is given, only the tokens the gate uses (the active set) are computed. The response is truncated after the last needed token
            # (causal, so earlier tokens keep their influence) and lm_head runs only on the needed rows. Everything else stays 0.
            if need is not None:
                idx = need[b, :n_resp].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                cut = int(idx[-1]) + 1
            else:
                idx, cut = None, n_resp
            hi2 = hi - (n_resp - cut)
            ids = input_ids[b:b + 1, lo:hi2]
            pos = position_ids[b:b + 1, lo:hi2] if position_ids.dim() == 2 else position_ids[:, b:b + 1, lo:hi2]
            y = responses[b, :cut] if idx is None else responses[b, idx]
            s = (hi - lo) - n_resp - 1            # row that predicts response token 0

            def f(p):
                h = functional_call(self.replica.model, {**{k[len("model."):]: v for k, v in p.items() if k.startswith("model.")},
                                                        **{k[len("model."):]: v for k, v in buffers.items() if k.startswith("model.")}},
                                    (ids,), {"position_ids": pos, "use_cache": False}).last_hidden_state[0, s:s + cut]
                if idx is not None:
                    h = h[idx]
                W = p["lm_head.weight"] if "lm_head.weight" in p else p["model.embed_tokens.weight"]
                lps = []
                for i in range(0, h.shape[0], _QBLOCK):                # chunk the head: [qblock, V] live at a time
                    z = h[i:i + _QBLOCK] @ W.t()
                    lps.append(torch.log_softmax(z.float(), -1).gather(-1, y[i:i + _QBLOCK, None])[:, 0])
                return torch.cat(lps)

            tang = {n: u_full[n] for n in params}
            _, D = jvp(f, (params,), (tang,))
            if idx is None:
                out[b, :n_resp] = D.float()
            else:
                out[b, idx] = D.float()
        return out


def _full_state_dict(fsdp_module):
    """Per-parameter full tensors on every rank (same path verl uses to feed vLLM), keys without the
    FSDP wrapper prefix. For a plain module this is just state_dict()."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
        if isinstance(fsdp_module, FSDP):
            with FSDP.state_dict_type(fsdp_module, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
                sd = fsdp_module.state_dict()
            return {k.replace("_fsdp_wrapped_module.", ""): v for k, v in sd.items()}
    except ImportError:
        pass
    return {k: v for k, v in fsdp_module.state_dict().items()}
