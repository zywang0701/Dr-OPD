# Acknowledgements and source layout

Dr. OPD builds on [OPD](https://github.com/thunlp/OPD) and [verl](https://github.com/volcengine/verl), with contributions from the OPDVR/GRPD codebase.

## Implementation

`Dropd/verl/` contains the training framework and Dr. OPD implementation. It is included directly, not as a Git submodule. The upstream entry points `opd_baseline.sh`, `grpd.sh` and `opdvr.sh` keep their original names so that they can be compared against the upstream code; the per-method launchers are `Dropd/arm_*.sh`.

Upstream license text and source notices are preserved. See [third-party licenses](LICENSE.md).

## Paper and presentation

The [paper](assets/paper.pdf), [Figure 1](assets/figure1.png) and [Figure 2](assets/figure2.png) accompany the method and results described on the homepage.

The README layout was inspired by [On-Policy Self-Adaptation](https://github.com/DripNowhy/On-Policy-Self-Adaptation).
