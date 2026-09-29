# Before announcing a public release

This page tracks decisions and evidence, not promised future results.

- [x] Follow the manuscript's Figure 1 narrative: dense teacher supervision, adaptive weights, Eq. (1), iterative solver.
- [x] Reuse original Figures 1 and 2; provide English and Chinese homepages.
- [x] Preserve the original framework Python implementation and upstream license.
- [x] Separate paper-reported results from code validation and packaged experiment coverage.
- [x] Provide a CPU-only dry run; do not start GPU experiments during packaging.
- [x] Exclude datasets, weights, private logs, server reports and credentials from Git.
- [x] Pass 13 CPU-only packaging tests and compare all 322 framework Python files with the source snapshot.
- [x] Render and visually review the English README, including Eq. (1) and both paper figures.
- [ ] Confirm the GitHub owner, repository name and visibility with the authors.
- [ ] Select a repository-wide license for the project additions and paper assets.
- [ ] Add the final public paper URL and identifier when available.
- [ ] Provide reviewed, version-pinned data downloads/preparation with checksums.
- [ ] Reconcile evaluation timing: the original trainer does not consume `TEST_STEPS`.
- [ ] Confirm paper aggregation excludes MATH500 and document checkpoint selection from the original records.
- [ ] Verify the Figure 2 same-size +2.3 annotation against unrounded results (displayed scores are 55.6 and 57.8).
- [ ] Supply the remaining Base-student, same-size and code-generation presets before claiming complete table reproduction.
- [ ] Freeze and validate a complete dependency/model-revision manifest on healthy hardware.
- [ ] Run end-to-end GPU validation after experiment suspension is lifted.

Outstanding items are explicit: the existing implementation can be reviewed without treating this preparation snapshot as a fully verified public reproduction release.
