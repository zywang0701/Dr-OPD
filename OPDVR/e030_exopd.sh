#!/bin/bash
# E030 baseline arm: ExOPD (G-OPD with reward extrapolation, Yang et al. 2026, arXiv 2602.12125; github.com/RUCBM/G-OPD).
#   r_t = lam*(log pi_T - log pi_ref) - (log pi_S - log pi_ref) on the sampled token, used directly as the advantage
#   pi_ref = frozen initial student (verl ref worker on actor_rollout_ref.model.path; main_ppo creates it when exopd_lambda != 1)
#   lam = EXOPD_LAM (paper: 1.25 for every ExOPD run, incl. Qwen3-30B-A3B-Instruct-2507 -> Qwen3-1.7B). See verl/trainer/ppo/exopd_reward.py.
# Everything else = e030_plain.sh.
set -eu; cd "$(dirname "$0")"; source ./e030_common.sh
export EXOPD_LAM=${EXOPD_LAM:-1.25}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-e030_exopd_lam${EXOPD_LAM}_$(date +%m%d_%H%M)}
export CORRECTNESS_GATED=False GRPO_SCALED=False
bash opd_baseline.sh "${E030_EXTRA[@]}" +actor_rollout_ref.rollout.exopd_lambda=$EXOPD_LAM "$@"
