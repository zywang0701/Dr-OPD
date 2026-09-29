# Dr. OPD implementation

This directory preserves the original implementation layout. `OPDVR` is an inherited directory name, not the name of the method presented by this repository.

Start from the [project homepage](../README.md) and use `bash run.sh` from the repository root. The `e030_*.sh` files retain the original method configurations; `verl/` contains the training framework and Dr. OPD implementation.

| Entry point | Method |
| --- | --- |
| `e030_ours.sh` | Dr. OPD |
| `e030_plain.sh` | Vanilla OPD |
| `e030_grpd.sh` | GRPD |
| `e030_opdgrpo.sh` | OPD + GRPO |
| `e030_exopd.sh` | ExOPD |

Direct invocation of these internal scripts has different fallback defaults from the root launcher; use the root launcher for the documented configuration. See [reproduction notes](../docs/reproduction.md), [method notes](../docs/method.md) and [upstream provenance](../UPSTREAM.md).
