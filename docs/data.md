# Data preparation

[← Project homepage](../README.md)

## Required files

Supply the prepared math files below via `TRAIN_DATASET` and `TEST_DATASET`. Dataset binaries and an automatic download command for these prepared files are not included in this repository. The checksums identify the reference files.

| File | Rows | Use |
| --- | --- | --- |
| `deepmath-level6-train.parquet` | 57,046 | Default Qwen3-1.7B math training, DeepMath difficulty ≥ 6 |
| `valid_final_unique.parquet` | 1,590 | Six-benchmark math evaluation bundle |

Reference SHA-256 checksums:

```text
de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597  deepmath-level6-train.parquet
1ab214dd731759843da8aeb1c8e899394cc05c527780333ba3b7bdc8f6807c19  valid_final_unique.parquet
```

## Expected input

Set `TRAIN_DATASET` and `TEST_DATASET` to your prepared files and run `bash run.sh data`. The original data schema includes:

- `prompt`: chat messages, including a user problem and the math-answer suffix.
- `reward_model`: verifier metadata, including `ground_truth` and `style`.
- `data_source`: benchmark/dataset identifier used in evaluation logging.
- Additional source fields such as `ability` and `extra_info` may be retained.

The default suffix is:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

## Benchmark accounting

The evaluation bundle combines AIME24 (30), AIME25 (30), AMC (83), MATH500 (500), Minerva (272), and OlympiadBench (675), totaling 1,590 prompts. The manuscript's reported math average includes the five benchmarks **other than MATH500**. Average the per-benchmark scores according to the paper protocol, not all 1,590 prompts as one undifferentiated pool.

## Other settings

The manuscript trains Qwen3-1.7B-Base on the difficulty `[3,4)` band, with a separate four-shot prompt. Other math students use `[6,10]`. The helper `scripts/make_deepmath_level.py` creates a single difficulty band and requires the level-6 reference file. Consult its arguments and set `DM_REF` explicitly; its default paths use `/tmp`.

Code-generation data, code execution/evaluation infrastructure, and the Base-student prompting recipe are described in the [paper](../assets/paper.pdf) but are not packaged as runnable presets here. Dataset and model terms remain those of their original providers.
