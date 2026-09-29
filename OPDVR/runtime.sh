#!/usr/bin/env bash
# Dr. OPD packaging: use the root launcher's environment, never stop other Ray jobs.
# Sourced by legacy entry points; no change to algorithm/configuration defaults.
set -euo pipefail
if [ -n "${VENV:-}" ]; then
  export PATH="$VENV/bin:$PATH"
fi
# main_ppo.py owns ray.init(). RAY_ADDRESS may select an existing allocation.
# Do not activate a machine-specific Conda environment or run ray stop --force.
