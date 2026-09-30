# Results reported in the paper

[← Project homepage](../README.md)

The following results are reported in the [paper](https://arxiv.org/abs/2609.38025). Math uses avg@16 across AIME24, AIME25, AMC, Minerva and OlympiadBench. Code uses avg@4 across HumanEval+, MBPP+ and LiveCodeBench. The paper reports the best evaluated checkpoint per method.

## Strong-to-weak distillation

Teacher: Qwen3-4B. Entries are task averages in percentage points; the manuscript contains per-benchmark results.

| Method | Qwen3-1.7B · Math | Qwen3-1.7B · Code | Qwen3-1.7B-Base · Math | Qwen3-1.7B-Base · Code |
| --- | ---: | ---: | ---: | ---: |
| Initial student | 22.6 | 41.4 | 5.2 | 3.7 |
| OPD | 25.3 | 43.4 | 16.8 | 40.2 |
| ExOPD | 25.7 | 44.3 | 16.8 | 43.4 |
| OPD + GRPO | 25.6 | 43.7 | 17.5 | 44.4 |
| GRPD | 24.7 | 43.6 | 16.9 | 40.9 |
| **Dr. OPD** | **35.0** | **48.4** | **18.6** | **46.6** |
| Teacher reference | 34.1 | 52.7 | 34.1 | 52.7 |

## Same-size distillation

Each student uses its corresponding RL-trained teacher. The table reports math results.

| Method | Qwen3-4B | Qwen3-4B-Base |
| --- | ---: | ---: |
| Initial student | 34.1 | 18.9 |
| OPD | 55.6 | 33.0 |
| ExOPD | 55.3 | 34.3 |
| OPD + GRPO | 55.8 | 34.0 |
| GRPD | 56.2 | 33.5 |
| **Dr. OPD** | **57.8** | **36.0** |
| Teacher reference | 55.2 | 34.1 |

![Figure 2: Empirical gains reported in the manuscript](../assets/figure2.png)

For training coverage, checkpoint selection and benchmark aggregation, see the [reproduction guide](reproduction.md#evaluation).
