# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-28

Initial release.

### Added

- `jsd_matrix`, `jsd_pdist`, `jsd_pairwise`, `jsd` and `to_similarity`, with
  an explicit logarithm base. Base 2 is the default, which bounds the distance
  in `[0, 1]` and makes `1 - d` a usable similarity.
- Five kernels behind `backend="auto"`: tiled dense and column-scatter sparse
  in pure NumPy, parallel dense and intersection-merge sparse in Numba, and an
  8-wide vectorised dense kernel when Intel SVML is available. Dispatch is by
  input density; `available_backends()` reports what an installation can use.
- A sparse path built on the support-intersection identity, so a pair
  evaluates logarithms only where both rows have mass. Exact, not an
  approximation.
- An occupancy bitmap that skips pairs whose supports cannot intersect, and a
  bitmap rank-select kernel that replaces the merge where the merge would
  otherwise spend most of its time stepping past non-matching entries.
- `well_conditioned=`, selecting a rearrangement of the per-coordinate summand
  that removes the cancellation affecting near-coincident distributions.
- Exactness guarantees documented with proofs in `docs/MATH.md`, including
  exactly `0.0` for identical rows, and pinned by the test suite.
- `bench/`: benchmark harnesses, baselines including the published sparse
  form, two real membership fixtures, and an A/A control that measures the
  noise floor of the machine before any ratio is reported.
- A `workflow_dispatch` cross-machine benchmark matrix spanning Linux x86-64,
  aarch64, Windows and macOS, with an aggregation step that reports on the
  legs that arrived and names those that did not.
