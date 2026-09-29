# Source provenance

This repository was prepared from Tianze's author-provided `iclr2027_math_code_anonymous.zip`, without importing the separate token-interpretability experiment changes.

Archive SHA-256:

```text
de0f25b47991d2d023e95fcc5a53cf18a6a398cba74dcb169fba840c26437bdc
```

## Inherited implementation

The source archive describes an OPD/OPDVR/GRPD code lineage built on [verl](https://github.com/volcengine/verl). The original inner README identifies verl v0.7.0 and links to [OPD](https://github.com/thunlp/OPD). The archive has no Git history, so an exact public upstream commit has not been verified and is not invented here.

`OPDVR/verl/` is an ordinary source directory, not a Git submodule. All framework Python files are copied unchanged. Existing license text and source notices are preserved. The internal `OPDVR` and `e030` names remain to avoid restructuring the scientific implementation.

## Packaging changes

- New bilingual project homepages, paper figures/PDF, code navigation, citation metadata, and focused documentation.
- Root launcher: CPU-safe `--dry-run`, argument checks, preserved `CUDA_VISIBLE_DEVICES`, data-path normalization, non-overwriting logs, explicit checkpoint output location, and failure propagation through `tee`.
- Legacy shell entry points: use the selected environment instead of a fixed Conda installation; let the existing trainer initialize Ray instead of forcibly stopping shared Ray jobs; replace machine-local fallback model paths with the same public model IDs.
- Installer: require Python 3.11, propagate installation failures, and correct the next-step instructions.
- New data-file validator and CPU-only repository checks. Training algorithms, method hyperparameters and framework Python source are unchanged.
- Excluded the machine-specific `e030_env.sh`, dataset binaries, caches, experiment outputs, credentials and unrelated internal materials.

## Paper assets

`assets/figure1.png` and `assets/figure2.png` are unmodified copies of the manuscript's Figures 1 and 2. `assets/paper.pdf` is a snapshot of the authors' preprint; no arXiv identifier or venue acceptance is asserted.

PDF SHA-256 at packaging:

```text
7d9419ed913f6c0736b9b29f91cb397001dbd74e91356e4a356231a7242f2549
```

Paper source SHA-256 at packaging:

```text
e7be6256c4f8324e283bc931135f79248e340dc40d1a2b6a11331df1a67a3907
```

README information architecture was inspired by [On-Policy Self-Adaptation](https://github.com/DripNowhy/On-Policy-Self-Adaptation); its artwork and scientific descriptions were not copied.

## Release status

See [the release checklist](docs/release-checklist.md) for unresolved protocol, data-distribution and licensing decisions. This snapshot makes no new repository-wide license grant; see [LICENSE.md](LICENSE.md).
