#!/bin/bash
# Vanilla sampled-token OPD (mask == 1).
set -eu; cd "$(dirname "$0")"; source ./e030_common.sh
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-e030_plain_$(date +%m%d_%H%M)}
export CORRECTNESS_GATED=False GRPO_SCALED=False
bash opd_baseline.sh "${E030_EXTRA[@]}" "$@"
