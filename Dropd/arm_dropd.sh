#!/bin/bash
# Dr. OPD: JVP token credits with influence-weighted distillation by default.
# See verl/workers/actor/credit_gate.py and jvp_influence.py for credit and gate definitions.
set -eu; cd "$(dirname "$0")"; source ./common.sh
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dropd_$(date +%m%d_%H%M)}
export CORRECTNESS_GATED=False GRPO_SCALED=False
export VAL_N=${VAL_N:-16} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}   # evaluation options; root preset may override
export CREDIT_EPS=${CREDIT_EPS:-1e-6}   # step size for the optional fp32 finite-difference path
bash opd_baseline.sh "${ARM_EXTRA[@]}" \
  +actor_rollout_ref.rollout.credit_gated=True \
  +actor_rollout_ref.rollout.credit_eps=$CREDIT_EPS \
  +actor_rollout_ref.rollout.credit_beta1=${CREDIT_BETA1:-0.9} \
  +actor_rollout_ref.rollout.credit_beta2=${CREDIT_BETA2:-0.999} \
  +actor_rollout_ref.rollout.credit_split_half=${CREDIT_SPLIT_HALF:-True} \
  "+actor_rollout_ref.rollout.credit_lowp_fracs=[0.1,0.2]" \
  +actor_rollout_ref.rollout.credit_mode=${CREDIT_MODE:-jvp} \
  +actor_rollout_ref.rollout.credit_gate_kind=${CREDIT_GATE:-iw} \
  +actor_rollout_ref.rollout.credit_adv_norm=${CREDIT_ADV_NORM:-grpo} \
  +actor_rollout_ref.rollout.credit_len_norm=${CREDIT_LEN_NORM:-True} \
  +actor_rollout_ref.rollout.credit_active_frac=${CREDIT_ACTIVE:-0.2} \
  +actor_rollout_ref.rollout.credit_w_max=${CREDIT_WMAX:-3.0} \
  +actor_rollout_ref.rollout.credit_w_min=${CREDIT_WMIN:-0.0} \
  +actor_rollout_ref.rollout.credit_lam_fixed=${CREDIT_LAM:-0.2} \
  +actor_rollout_ref.rollout.credit_f_hi=${CREDIT_F_HI:-0.05} \
  +actor_rollout_ref.rollout.credit_f_lo=${CREDIT_F_LO:-0.15} \
  +actor_rollout_ref.rollout.credit_jvp_check=${CREDIT_JVP_CHECK:-0} "$@"
