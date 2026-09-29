# Reproduction guide

[← Project homepage](../README.md)

## Scope

The repository provides math training with Dr. OPD and four baselines. The default setting is Qwen3-4B → Qwen3-1.7B, non-thinking mode, 50 training steps.

## Evaluation

The paper's evaluation protocol and the root launcher's defaults differ as follows:

| Item | Paper protocol | Launcher defaults |
| --- | --- | --- |
| Math average | AIME24, AIME25, AMC, Minerva, OlympiadBench | The prepared evaluation file also includes MATH500; exclude it when computing the paper's five-benchmark average |
| Evaluation schedule | Every 10 optimization steps; best evaluated checkpoint reported | `test_freq=1000` plus the final step; at 50 training steps, evaluation runs at the final step |
| Checkpoints | Best evaluated checkpoint | Checkpoints saved at steps 30, 40 and 50 via `SAVE_STEPS` |

`EVAL_STEPS` controls checkpoint saving and exports `TEST_STEPS`, which the trainer does not consume. It does **not** change evaluation timing. A final-step score is not directly comparable to the paper's best-of-checkpoints score; match both the evaluation schedule and aggregation when reproducing the tables.

The Base-student settings require different data bands and prompt templates, not just a different `STUDENT` model ID. Dedicated code-generation and same-size presets are not included.

## Environment

The reference configuration is one node with 8 × H100 80 GB GPUs, Linux, Python 3.11 and CUDA 12.x. A lower minimum GPU configuration has not been validated. Model downloads and installation need network access; GPU kernels are checked during installation.

```bash
PYTHON=python3.11 bash run.sh setup
```

The installer creates `work/venv`, installs the vendored framework, and downloads Qwen3-1.7B and Qwen3-4B. Set `SKIP_MODELS=1` to skip downloads and use your own model paths. Major pinned packages include vLLM 0.10.2, Transformers 4.57.6, FlashAttention 2.8.3 and datasets 5.0.1. Transitive dependencies are not fully pinned; see [the installer](../scripts/install.sh) for version constraints.

If models were downloaded by the installer, explicitly select those local copies to avoid resolving their IDs into a different cache:

```bash
export STUDENT="$PWD/work/models/Qwen3-1.7B"
export TEACHER="$PWD/work/models/Qwen3-4B"
```

Use absolute paths for `WORK`, `VENV`, `RESULTS_DIR`, `FINAL_CKPT_DIR` and local model paths when overriding them. Data paths are normalized before the internal entry point changes directory.

## Data

Dataset binaries are not included. See [data.md](data.md) for the required files, schema and checksums.

```bash
export TRAIN_DATASET=/path/to/deepmath-level6-train.parquet
export TEST_DATASET=/path/to/valid_final_unique.parquet
bash run.sh data
```

The check requires PyArrow, confirms the expected row counts and required top-level columns, and exits nonzero on mismatches. It does not prove semantic equivalence to the source data; file hashes are provided separately.

## Launchers

```bash
bash run.sh train ours --dry-run  # CPU-safe launch plan, not a full Hydra configuration dump
bash run.sh train ours 0.4 2      # Dr. OPD: lambda=0.4, data seed=2
bash run.sh train plain          # Vanilla sampled-token OPD
bash run.sh train grpd           # GRPD
bash run.sh train opdgrpo        # OPD + GRPO
bash run.sh train exopd          # ExOPD
```

The root launcher supplies overrides that differ from some internal scripts' fallback defaults. Do not use `e030_ours.sh` directly as a substitute for the root preset.

| Setting | Default | Override |
| --- | --- | --- |
| Student / teacher | Qwen3-1.7B / Qwen3-4B | `STUDENT`, `TEACHER` |
| Training steps | 50 | `STEPS` |
| Checkpoint step list | 30,40,50 | `EVAL_STEPS` (inherited name; see caveat above) |
| Prompts × rollouts | 128 × 8 | Internal shared configuration |
| Prompt / response limits | 1,024 / 12,288 | `RESP_LEN`, `VAL_LEN` for response limits |
| Learning rate | 1e-5, constant | `ACTOR_LR` |
| Gradient clipping | 1.0 | `GRAD_CLIP` |
| Weighting strength | 0.4 | Third positional argument for `train ours` |
| Weight bounds | 0.001 to 3.0 | `CREDIT_WMIN`, `CREDIT_WMAX` |
| Active fraction | 1.0 | `CREDIT_ACTIVE` |
| Data seed | 2 | Fourth positional argument |
| GPUs | 8 | `NGPU`; changing it requires validation |

The positional seed sets `DATA_SEED`; it is not a unified seed for every random number generator. The original rollout default is seed 0. Reproduce RNG settings explicitly rather than inferring them from a run name.

`CUDA_VISIBLE_DEVICES` is preserved, and the entry points use the environment selected by `VENV`. The trainer calls `ray.init()`; if using a pre-existing allocation, configure `RAY_ADDRESS` appropriately. The launchers do not stop other Ray jobs.

## Outputs

Console logs go to `results/<run-name>.log`. The launcher refuses to overwrite an existing log; use a new `RESULTS_DIR` for a retry. Checkpoints default to `work/checkpoints/<run-name>`. The original estimate is approximately 22 GB per 1.7B checkpoint, in addition to model/environment storage. Weights & Biases is offline by default.

## CPU-only checks

```bash
python3 -m unittest discover -s tests -v
```

The tests validate documentation, argument handling, dry-run behavior and launcher wiring using harmless stubs. They do not load models, start Ray, or access GPUs.
