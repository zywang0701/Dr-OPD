#!/bin/bash
# E030 baseline arm: OPD+GRPO (OPDVR paper App. B.2 "add a GRPO-based policy gradient loss", after MiMo-V2-Flash MOPD).
#   A_t = (log pi_T - log pi_S)(o_t) + w * A_GRPO(i),   A_GRPO = (R_i - mean_g R) / (std_g R + 1e-6),  R_i in {0,1}
# = OPDVR's own estimator token_reward_direct_plus_grpo (verl/trainer/ppo/core_algos.py), w = algorithm.grpo_outcome_weight.
# w = OPDGRPO_W (paper: equal weighting 1:1 -> 1.0); /std = OPDVR's default (algorithm.norm_adv_by_std_in_grpo=True).
# Everything else = e030_plain.sh. opd_baseline.sh hard-sets ADV_ESTIMATOR, so the estimator is overridden here (last value wins).
set -eu; cd "$(dirname "$0")"; source ./e030_common.sh
export OPDGRPO_W=${OPDGRPO_W:-1.0}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-e030_opdgrpo_w${OPDGRPO_W}_$(date +%m%d_%H%M)}
export CORRECTNESS_GATED=False GRPO_SCALED=False
bash opd_baseline.sh "${E030_EXTRA[@]}" \
  algorithm.adv_estimator=token_reward_direct_plus_grpo \
  algorithm.grpo_outcome_weight=$OPDGRPO_W \
  algorithm.norm_adv_by_std_in_grpo=True "$@"
