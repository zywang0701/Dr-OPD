# Data preparation

[← Project homepage](../README.md)

## Distribution status

This repository contains no dataset binaries. The original author-provided archive contains the prepared math files below; their hashes are recorded for exact identification. A public, version-pinned download location and redistribution review are still pending. **If you do not already have these files, the default training example is not yet self-contained.** This is a release item, not a reason to substitute a different dataset silently.

| File | Rows | Use |
| --- | --- | --- |
| `deepmath-level6-train.parquet` | 57,046 | Default Qwen3-1.7B math training, DeepMath difficulty ≥ 6 |
| `valid_final_unique.parquet` | 1,590 | Six-set evaluation bundle from the original snapshot |

SHA-256 of the supplied snapshot files:

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

The manuscript trains Qwen3-1.7B-Base on the difficulty `[3,4)` band, with a separate four-shot prompt. Other math students use `[6,10]`. The retained `scripts/make_deepmath_level.py` is an original helper for creating a single difficulty band; it requires the original level-6 reference file and has legacy `/tmp` defaults. It is not a complete, version-pinned public data-preparation pipeline. Consult its arguments and set `DM_REF` explicitly before using it.

Code-generation data, code execution/evaluation infrastructure, and the Base-student prompting recipe are described in the [paper](../assets/paper.pdf) but are not packaged as runnable presets here. Dataset and model terms remain those of their original providers.
