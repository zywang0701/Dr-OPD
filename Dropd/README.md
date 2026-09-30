# Dr. OPD implementation

This directory holds the method implementation and the per-method launchers. `verl/` is the vendored training framework; `opd_baseline.sh`, `grpd.sh` and `opdvr.sh` are the upstream entry points and keep their original names.

Start from the [project homepage](../README.md) and use `bash run.sh` from the repository root. The `arm_*.sh` files hold the per-method configurations and all share `common.sh`.

| Entry point | Method |
| --- | --- |
| `arm_dropd.sh` | Dr. OPD |
| `arm_plain.sh` | Vanilla OPD |
| `arm_grpd.sh` | GRPD |
| `arm_opdgrpo.sh` | OPD + GRPO |
| `arm_exopd.sh` | ExOPD |

Direct invocation of these internal scripts has different fallback defaults from the root launcher; use the root launcher for the documented configuration. See [reproduction notes](../docs/reproduction.md), [method notes](../docs/method.md) and [upstream provenance](../UPSTREAM.md).
