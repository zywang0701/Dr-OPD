# -*- coding: utf-8 -*-
"""ExOPD baseline: G-OPD with reward extrapolation (Yang et al. 2026, arXiv 2602.12125, github.com/RUCBM/G-OPD).

Per sampled token, used directly as the advantage (token_reward_direct), no group baseline / normalisation / clipping:
    r_t = lam * (logT - logRef) - (logS - logRef)
        = (logT - logS) + (lam - 1) * (logT - logRef)
  logT   teacher log-prob of the sampled token
  logS   student old_log_probs (the tensor the teacher worker subtracted; detached)
  logRef reference model = the frozen initial student (verl ref worker, loads actor_rollout_ref.model.path)
Same as G-OPD's single-teacher branch (verl/workers/actor/dp_actor.py):
    reverse_kl = (old_log_prob - base_log_prob) - lam * (teacher_log_prob - base_log_prob); advantages = -reverse_kl
In this fork rm_scores = logT - logS (fsdp_workers.compute_rm_score, top_k=0), so logT = rm_scores + logS.
lam = 1 is plain OPD (logRef cancels). Paper: lam = 1.25 for every ExOPD run, reference = student's initial model.

torch only (no verl imports) so the CPU test can load it with importlib.
"""
import torch


def exopd_rewards(rm_scores, old_log_probs, ref_log_prob, response_mask, lam):
    """Returns (rewards (B, T) in rm_scores.dtype, zero on padding; dict of scalar metrics)."""
    assert rm_scores.dim() == 2, "ExOPD needs sampled-token OPD (log_prob_top_k=0): rm_scores must be (B, T)"
    assert rm_scores.shape == old_log_probs.shape == ref_log_prob.shape == response_mask.shape
    m = response_mask.float()
    opd = rm_scores.float()                                  # logT - logS
    log_t = opd + old_log_probs.float()
    extra = (lam - 1.0) * (log_t - ref_log_prob.float())     # (lam - 1) * (logT - logRef)
    r = (opd + extra) * m
    n = m.sum().clamp_min(1.0)
    stats = {
        "exopd/lambda": float(lam),
        "exopd/opd_mean": float((opd * m).sum() / n),        # the plain-OPD part
        "exopd/extra_mean": float((extra * m).sum() / n),    # the extrapolation part
        "exopd/reward_mean": float(r.sum() / n),
        # |logS - logRef|: ~0 at step 1 (ref == initial student), then grows as the student moves
        "exopd/ref_gap_absmean": float(((old_log_probs.float() - ref_log_prob.float()).abs() * m).sum() / n),
    }
    return r.to(rm_scores.dtype), stats
