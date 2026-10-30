# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-25

### Added

- `backend="numba-f32"`, an opt-in dense kernel that computes in float32 and
  returns float32 (it implies `dtype=np.float32`; `auto` never selects it).
  The pair loop stays in float32, for twice the SIMD width, with an inlined
  division-free polynomial logarithm (at most 1.06 ULP) so that LLVM can
  vectorize the row, and with the cancellation-free series on coordinates
  where `|u| <= 1/2`. Identical rows still give exactly 0.0, disjoint
  supports exactly 1.0 in base 2 -- through a float64 pass for near-maximal
  pairs, which also clamps the distance to 1.0. On x86-64: about `1e-7`
  absolute error on the distance against float64 arithmetic on the same rows
  (at most 1.6e-7 over the regimes `bench/f32.py` sweeps, `k` up to 4096;
  each pair is one float32 sum, so 3.4e-6 at `k = 262144`) and `1e-7`
  relative on near duplicates; on one 4-core AVX-512 machine, float32 input,
  2.3x-2.5x faster than `numba-svml` and 5.6x-7.5x faster than `numba`, and
  2.1x-6.4x faster than `numba` on GitHub's x86-64 runners. On ARM (Linux
  aarch64, Apple Silicon) LLVM leaves the loop scalar, so the float32 sum is
  sequential: 0.7x-1.1x the speed of `numba` on GitHub's runners, and up to
  2e-6 over the same regimes. Use `numba` there. Plain Numba: no compiler,
  runtime library or new dependency. `docs/MATH.md` ("The float32 kernel")
  has the arithmetic; `bench/f32.py` measures accuracy, exactness and speed.

### Changed

- Cross matrices with fewer rows in `P` than Numba threads -- one query
  against many -- are scheduled over pairs instead of rows on the dense and
  sparse Numba kernels, so every thread gets work. Results are bitwise
  unchanged (`tests/test_schedule.py`); the dense kernel runs up to 3.7x
  faster at 4 threads on such shapes.
- The cross-machine benchmark covers `numba-f32` (leg schema
  `fastjsd-crossmachine/2`). Every leg times it against `numba` on the dense
  cases, on the same float32 rows, and checks its exact 0.0 and its accuracy
  against the truth of its float32-rounded input; every leg also checks that
  disjoint supports come back exactly 1.0. `bench/aggregate.py` counts the
  ratio over boxes and prints any loss in the headline.
  `tests/test_crossmachine.py` covers both halves, and the `test` extra now
  includes mpmath.
- CI runs the `numba-svml` tests, with libsvml from PyPI's
  `intel-cmplr-lib-rt`, on a runner with AVX-512; until now no CI job ran
  them. Every job lists its skipped tests with the reason, and the `oldest`
  job also runs the mpmath accuracy test.
- The license is declared as an SPDX expression (`license = "MIT"`, PEP 639)
  instead of a table and a classifier, which setuptools deprecates. Building
  from source now needs setuptools 77 or newer; installed packages are
  unaffected.

### Fixed

- On NumPy 1.x with Numba installed, the sparse Numba backend -- which `auto`
  picks for sparse input -- raised `AttributeError: module 'numpy' has no
  attribute 'bitwise_count'`, although the package declares `numpy>=1.24`:
  it counted bits with `np.bitwise_count`, which NumPy added in 2.0. It now
  falls back to a byte-table count with identical results, and CI tests the
  declared floors, NumPy 1.24 with Numba 0.58.
- `jsd_pairwise` and `jsd` could return a base-2 distance above 1.0 for
  disjoint supports, so `to_similarity` went slightly negative: they
  evaluated the general summand on one-sided coordinates and summed with
  `np.sum`, which missed 1.0 on 15 % of disjoint pairs, by up to two ULP
  either way. They now use the NumPy kernel's arithmetic -- matched
  coordinates only, `ln 2` times each row's one-sided mass totalled left to
  right -- so disjoint supports give exactly 1.0, identical rows exactly 0.0,
  and `jsd(p, q)` is bitwise `jsd(q, p)`. Other values change by under
  1e-15.
- The sdist's test suite could not run: it shipped `tests/` without
  `tests/conftest.py`, which most test modules import, and without the
  `bench/` modules `tests/test_crossmachine.py` imports. It ships both now,
  and CI runs the sdist's suite against the built wheel.
- `docs/MATH.md` G6 described an engine from before the release: it said
  disjoint supports give exactly 1.0 only by coincidence, with 4-27 % of
  pairs 1-2 ULP low. 0.1.0 snaps row totals and keeps one accumulator per
  row, so `jsd_matrix` and `jsd_pdist` return exactly 1.0 there for any
  weights (float64, `normalize=True`). G6 now gives the proof, its side
  conditions and where it does not apply -- `normalize=False`, and float32 on
  the NumPy kernels. The two tests that encoded the old statement now assert
  the exact value.

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
