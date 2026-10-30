# fastjsd

[![PyPI version](https://img.shields.io/pypi/v/fastjsd.svg)](https://pypi.org/project/fastjsd/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastjsd.svg)](https://pypi.org/project/fastjsd/)
[![Tests](https://github.com/briankeithn/fastjsd/actions/workflows/ci.yml/badge.svg)](https://github.com/briankeithn/fastjsd/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast, exact, pairwise Jensen-Shannon distance matrices for Python.

`fastjsd` computes the full `n x n` matrix of Jensen-Shannon distances between
rows of a stack of discrete distributions. It is a drop-in replacement for
`scipy.spatial.distance.pdist(P, "jensenshannon")` that adds a base argument,
parallel kernels, and a sparse path that skips coordinates neither
distribution occupies.

```python
import numpy as np
from fastjsd import jsd_matrix

P = np.random.default_rng(0).dirichlet(np.full(32, 0.1), size=500)
D = jsd_matrix(P)          # (500, 500), base 2, distances in [0, 1]
```

## Install

```bash
pip install fastjsd            # NumPy backends only
pip install "fastjsd[numba]"   # adds the parallel dense and sparse kernels
```

Python 3.10 or newer. NumPy is the only hard dependency; Numba is optional and
enables the compiled kernels. SciPy is needed only by the test suite and the
benchmark harnesses, and comes in with the `test` and `bench` extras.

## Why it is fast

For distributions `p, q` over `k` outcomes with `m = (p + q) / 2`,

```
JS(p, q) = (1/2) * sum_c [ p_c ln p_c + q_c ln q_c - 2 m_c ln m_c ]
```

Two properties of that arrangement do the work.

**One logarithm per pair-coordinate.** `p ln p` depends on a single row, so it
is tabulated once per entry and the inner loop evaluates exactly one
logarithm. SciPy's kernel groups the same quantity as `a ln(a/m) + b ln(b/m)`,
where `m` depends on the pair, so neither logarithm can be hoisted and it
pays two.

**Logarithms only on the support intersection.** Where exactly one side has
mass, the summand is exactly `a ln 2` (Lee, 1997). A pair therefore needs
logarithms only where both rows are non-zero, and everything else is a
constant times the mass outside the intersection. For rows of density `d` with
independent supports the expected intersection is `k d^2` rather than `k`.
This is exact, not an approximation, and it is what makes thresholded
membership and topic-mixture data cheap.

The sparse kernels add an occupancy bitmap so pairs whose supports cannot
intersect are skipped without walking either row.

## Exactness

The guarantees, with proofs and the measurements that support them, are in
[`docs/MATH.md`](https://github.com/briankeithn/fastjsd/blob/main/docs/MATH.md). In summary:

- Every per-coordinate summand is non-negative, so the sum over coordinates
  has condition number exactly 1.
- Identical rows return exactly `0.0`, not float dust that `sqrt` amplifies.
  This is a property of keeping the constant inside the sum; forms that fold
  `ln 2` outside it lose the exact zero.
- Rows with disjoint supports return exactly `1.0` in base 2, for any
  weights.
- Results agree with SciPy to within a few times `1e-13` across dense, sparse,
  cross and condensed forms.
- The individual summand is ill-conditioned when `p` and `q` nearly coincide.
  That is a property of the grouping, it is documented rather than hidden, and
  `well_conditioned=True` selects a rearrangement that removes it.

## Benchmarks

`bench/` contains the harnesses, the baselines and two real membership
fixtures. Performance depends heavily on the machine, so this package ships
the instruments and one worked example rather than a headline number. The
harnesses live in the repository rather than in the wheel, so run them from a
checkout:

```bash
pip install -e ".[bench]"
python bench/run_all.py            # fingerprints the machine, then measures
```

`run_all.py` records the CPU, core counts, library versions and thread
settings before timing anything, then runs the three timing harnesses that
have to be read together -- `aa_control.py`, `bits_vs_merge.py` and
`lee_control.py` -- in dependency order, into one fingerprinted results
directory. The first of them, `bench/aa_control.py`, duels the implementation
against *itself*, so the true ratio is 1.000 by construction and everything it
reports is the measurement noise floor. Read that number first: a speedup
smaller than the noise floor of the machine that produced it is not a result.

### One machine's numbers

[`docs/measurements/`](https://github.com/briankeithn/fastjsd/blob/main/docs/measurements/README.md) holds a complete run with
its fingerprint and noise floor attached. Single-threaded on a 4-core Xeon at
2.10 GHz, which compares kernels rather than thread counts since SciPy's
`pdist` is single-threaded regardless:

| against | synthetic grid | real fixtures |
|---|---|---|
| SciPy | 2.19x - 33.17x | 2.14x - 8.93x |
| Lee 1997, transcribed literally | 1.86x - 5.06x | 1.52x - 2.10x |
| Lee 1997, competently implemented | **9 of 14 cells win, 1.19x - 3.61x; 5 lose, up to 1.25x** | |

Read the third row, not the second. The two differ in what the baseline is
allowed to do, and the gap between them is large: the literal transcription
evaluates `a ln(a/s) + b ln(b/s)` as printed, paying two logarithms and two
divisions per matched coordinate, while the same identity factors as
`a ln a + b ln b - s ln s` and accepts the same tabulated `x ln x` this package
uses. Giving the baseline that tabulation and a 1984 occupancy prefilter --
neither of which has anything to do with where `ln 2` sits -- makes it 1.1x to
2.6x faster, and that difference is most of the apparent margin in row two.

So the honest comparison is the third row, and it includes five losses,
covering both real fixtures under the stricter cleanup. That is expected, as 
the folded arrangement absorbs the one-sided mass into a constant and needs
no per-row matched-mass accumulators at all. What the unfolded arrangement
buys is the exact zero, not speed --

```
fastjsd          exactly 0.0 in all 96 duplicate-row trials
Lee-1997 sparse  NONZERO in 30/96, max 3.58e-08
scipy            exactly 0.0 in all 96
```

-- and SciPy is at parity on that, so the guarantee is against the
constant-folded published arrangement, not against SciPy.

The SciPy spread tracks density, which is the support-intersection identity
doing the work: about 2.2x on dense input, above 30x at 1.6 % non-zero.

Treat those figures as one data point on one host, not a specification.

Every harness also runs on its own:

| script | question |
|---|---|
| `aa_control.py` | what is the smallest ratio this machine can resolve? |
| `bench_honest.py` | how does this compare to SciPy and to the published sparse form? |
| `lee_control.py` | how much of any margin is the arrangement, and how much the implementation? |
| `bits_vs_merge.py` | where is the crossover between the two sparse kernels? |
| `ratio_vs_n.py` | does the ratio to the baselines hold across `n`, or only at one `n`? |
| `accuracy.py` | true error against 50-digit arithmetic, by regime |
| `conditioning.py` | cancellation in each grouping of the summand |
| `lee_arrangement.py` | which arrangements return exactly zero for identical rows |
| `scale.py` | do the fitted exponents match the cost model? |
| `gpu_baseline.py` | would a GPU make this irrelevant? |
| `distributions.py` | do the synthetic inputs look like the real ones where it matters? |
| `fixture_shape.py` | how much of the fixtures' shape does the thresholding step add? |
| `f32.py` | how accurate is the float32 kernel, and (`--time`) how fast? |

`accuracy.py`, `conditioning.py`, `lee_arrangement.py`, `distributions.py`,
`fixture_shape.py` and `f32.py` without `--time` start no timer -- they report
errors and counts -- so they are the ones that are safe to run on a busy
machine. The rest are timing harnesses and want an idle one.

`bench_honest.py` times whatever else is installed alongside the two main
baselines, so it also needs `pynndescent` and `torch`; `bench/portable.py`,
described below, is the variant that records a missing implementation instead
of failing on it. `gpu_baseline.py` needs `torch` and should be run on its own,
since it loads the CPU too.

### Across many machines

A single machine cannot settle a performance claim: the crossover between the
two sparse kernels is sensitive to the branch predictor, so two hosts can
disagree about which is faster on identical input. Two more programs cover
that. `bench/portable.py` runs one *leg* -- one box, one self-describing JSON
document, every implementation probed before it is timed and recorded as
absent rather than fatal when it cannot run. `bench/aggregate.py` combines
legs into a claim and refuses to combine them wrongly; in particular it never
divides a time measured on one box by a time measured on another.

The `cross-machine benchmark (manual)` workflow
(`.github/workflows/cross-machine.yml`) runs one leg per runner across Linux
x86-64 (three draws from the same pool, for a variance estimate), older glibc,
aarch64, Windows and macOS, then aggregates them. It is `workflow_dispatch`
only. What it establishes is the *direction* of each claim across genuinely
different silicon, libm and toolchains, and the exactness guarantees, which
are deterministic and travel unchanged. `bench/CROSS_MACHINE.md` is the
operator's guide, and says which figures may be quoted and which may not.

## Backends

`backend="auto"` picks by input density and available extensions. Force one
with `backend=`:

- `numpy` and `numpy-sparse`: no compiled dependency.
- `numba` and `numba-sparse`: parallel dense and intersection kernels.
- `numba-svml`: an 8-wide vectorised dense kernel when Intel SVML is present.
- `numba-f32`: an opt-in dense kernel that computes in float32 (below).

`available_backends()` lists what this installation can use.

### The float32 kernel

`backend="numba-f32"` computes in float32 and returns float32 -- it implies
`dtype=np.float32` -- trading the last digits for throughput. The other
kernels compute in float64 whatever the input dtype. This one keeps the hot
loop in float32, which doubles the SIMD width, and replaces the libm logarithm
with an inlined polynomial, so that LLVM can vectorize the whole row -- which
it does on x86-64, and does not on ARM. Close coordinates take the
cancellation-free series of `docs/MATH.md` instead of a logarithm. It is plain
Numba -- no compiler or runtime library beyond what the `numba` backend
needs -- so it runs wherever that backend does. `auto` never selects it.

- **Accuracy**, against float64 arithmetic on the same float32 rows: each
  pair is one float32 sum over its `k` coordinates, so the error grows with
  `k`. On x86-64, where LLVM splits that sum over its vector lanes, about
  `1e-7` on the distance -- at most `1.6e-7` over the regimes `bench/f32.py`
  sweeps, `k` up to 4096 -- rising to `8.7e-7` at `k = 65536` and `3.4e-6`
  at `k = 262144`, and about `1e-7` *relative* on close pairs. On ARM the sum
  is sequential: the cross-machine benchmark measured up to `3.2e-7` on its
  fixtures there (`k <= 64`), and `bench/f32.py` run in that order reaches
  `2e-6` over its regimes with `k` up to 4096 and `9e-5` at `k = 262144`
  (`docs/MATH.md`). The float64 kernels fed float32 input can be `1e-4` off
  on near duplicates, because they round `a + b` to float32 first.
- **Exactness**: identical rows give exactly `0.0` and disjoint supports
  exactly `1.0` in base 2, as on the other backends; the base-2 distance never
  exceeds `1.0`. Results agree across machines to float32 rounding, not
  bitwise.
- **Speed**, one machine (4-core Xeon at 2.8 GHz, AVX-512, float32 input for
  every kernel, `python bench/f32.py --time`): 2.3x-2.5x faster than
  `numba-svml` and 5.6x-7.5x faster than `numba` on dense matrices of 192 to
  1024 rows, single-threaded and on 4 threads. On GitHub's runners the
  cross-machine workflow measured it 2.1x-6.4x faster than `numba` on x86-64,
  but 0.7x-1.1x on ARM (Linux aarch64 and Apple Silicon), where LLVM leaves
  its loop scalar: there it is no faster and less accurate, so use `numba`.
  Measure your own.

## API

```python
jsd_matrix(P, Q=None, *, base=2.0, squared=False, normalize=True,
           check=True, dtype=None, backend="auto", out=None,
           threads=None, well_conditioned=None)   # (n, n) or (n, m)

jsd_pdist(P, ...)       # condensed upper triangle, SciPy pdist order
jsd_pairwise(P, Q, ...) # row-aligned distances between two stacks
jsd(p, q, ...)          # scalar
to_similarity(D)        # 1 - D, meaningful in base 2
```

`base=2.0` bounds the distance in `[0, 1]`, which is why it is the default and
why `to_similarity` is well defined. SciPy's compiled pairwise path is base
`e` only, under which the distance caps at `sqrt(ln 2)` and `1 - d` never
reaches zero for disjoint supports.

## Documentation

- [`docs/MATH.md`](https://github.com/briankeithn/fastjsd/blob/main/docs/MATH.md): the identity, its consequences, the
  guarantees and their proofs.
- [`docs/PRIOR_ART.md`](https://github.com/briankeithn/fastjsd/blob/main/docs/PRIOR_ART.md): what other libraries do, and
  where this one differs.

## Acknowledgement

Claude Code was used to assist the development of this code as part of a
bigger project that required sparse JSD computations at scale.

## References

- Lin, J. (1991). Divergence measures based on the Shannon entropy.
  *IEEE Transactions on Information Theory* 37(1), 145-151.
- Lee, L. (1997). *Similarity-Based Approaches to Natural Language
  Processing*. Harvard University Technical Report TR-11-97.
- Endres, D. M. and Schindelin, J. E. (2003). A new metric for probability
  distributions. *IEEE Transactions on Information Theory* 49(7), 1858-1860.
- Österreicher, F. and Vajda, I. (2003). A new class of metric divergences on
  probability spaces and its applicability in statistics. *Annals of the
  Institute of Statistical Mathematics* 55(3), 639-653.

## License

MIT. See [LICENSE](https://github.com/briankeithn/fastjsd/blob/main/LICENSE).
