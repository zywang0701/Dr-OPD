#!/usr/bin/env bash
# Shared runtime: use the root launcher's environment, never stop other Ray jobs.
# Sourced by the method entry points.
set -euo pipefail
if [ -n "${VENV:-}" ]; then
  export PATH="$VENV/bin:$PATH"
fi
# main_ppo.py owns ray.init(). RAY_ADDRESS may select an existing allocation.
# Do not activate a machine-specific Conda environment or run ray stop --force.
