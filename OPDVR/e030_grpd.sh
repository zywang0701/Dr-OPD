#!/bin/bash
# GRPD: correctness-gated ReLU + |group-relative advantage| scaling.
# Dr.GRPO scaling (GRPO_NORM_BY_STD=False) unless overridden.
set -eu; cd "$(dirname "$0")"; source ./e030_common.sh
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-e030_grpd_$(date +%m%d_%H%M)}
export CORRECTNESS_GATED=True GRPO_SCALED=True GRPO_SCALE_BASELINE=${GRPO_SCALE_BASELINE:-0.0}
bash grpd.sh "${E030_EXTRA[@]}" "$@"
