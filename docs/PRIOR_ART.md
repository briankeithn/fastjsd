# Prior art and baselines

The alternatives this package is measured against are shipped rather than
quoted. `fastjsd/reference.py` keeps the implementations it replaces, in the
shape people actually run them, and the test suite checks agreement against
them. `bench/baselines.py` adds SciPy, the published sparse form written out,
and the third-party kernels that are installed in a typical environment; each
declares the agreement it can hold, and `bench/bench_honest.py` verifies every
implementation against SciPy before timing it. "Faster than what?" is the only
question that makes a speedup claim mean anything, so this file describes what
each alternative computes and where its cost goes, and the harnesses in
`bench/` measure the magnitudes on the machine running them.

Two conclusions shape the design and recur below: a NumPy broadcast over the
pairwise mixture is bandwidth-bound where a compiled scalar loop is
compute-bound, and no widely installed Python implementation exploits the
support-intersection identity even though it has been published since 1997.

## Per-pair Python callable (`python_loop_matrix`)

```python
clust_sim = distance.cdist(membership, membership,
                           lambda u, v: distance.jensenshannon(u, v, base=2.0))
```

`cdist` with a Python callable calls back into the interpreter once per
pair: `n^2` Python calls, each allocating and reducing `k`-length arrays.
It also computes the full square matrix for a symmetric metric, doing every
pair twice. The interpreter round trip, not the arithmetic inside it,
dominates the runtime.

## Chunked NumPy broadcast (`broadcast_matrix`)

```python
def broadcast_matrix(P, chunk=128):
    H_P = entropy(P)
    out = np.empty((n, n))
    for lo in range(0, n, chunk):
        M = 0.5 * (P[lo:hi, None, :] + P[None, :, :])       # (chunk, n, k)
        divergence = entropy(M) - 0.5 * (H_P[lo:hi, None] + H_P[None, :])
        out[lo:hi] = np.sqrt(np.clip(divergence, 0, None))
```

Precomputing the per-row entropy and chunking to bound memory both help, and
this is far faster than a per-pair Python callable. It is nonetheless slower
than SciPy's compiled metric once `k` is large enough for the `(chunk, n, k)`
temporary to leave cache, and the reason is structural.

### Why the broadcast loses to a compiled scalar loop

**The `(chunk, n, k)` tensor is the whole cost.** At `n=5408, k=192` one
128-row chunk is `128 x 5408 x 192 x 8 B` = **1.06 GB**, and the kernel
walks it about five times over (`M`, two `np.where`, the `log`, the
product, the reduction). SciPy's C loop keeps two `k`-length rows in L1 and
never allocates. Chunking bounds peak memory but not memory *traffic*,
which is the actual bottleneck -- so the kernel is bandwidth-bound where
SciPy is compute-bound.

Three smaller costs compound it:

- **The full square matrix** is computed for a symmetric metric -- every
  pair twice. `pdist` + `squareform` removes that half.
- **`np.where(M > 0, M * np.log2(np.where(M > 0, M, 1)), 0.0)`** evaluates
  the product everywhere and materializes three extra tensors to mask it,
  where `np.log(M, out=L, where=M > 0)` writes one.
- **Density is ignored.** Membership vectors thresholded at `1/k` before use
  are 12-24 % nonzero, and a broadcast does the same work regardless.

The disadvantage widens with both `n` and `k`, which is the signature of a
bandwidth problem rather than a fixed constant factor, and it widens again on
sparse input, where a compiled scalar metric skips zero terms and a broadcast
cannot. The broadcast and SciPy's `pdist` are both in `fastjsd/reference.py`,
so the comparison can be reproduced directly on any machine rather than taken
on trust.

## SciPy `pdist` + `squareform`

```python
topic_sim = 1 - distance.squareform(distance.pdist(cluster_probs,
                                                   metric='jensenshannon'))
```

This keeps SciPy's compiled metric and removes the redundant half of the work.
For a change that adds no dependency it is the right call. Its ceiling is
SciPy's: single-threaded, one pair at a time, base `e` only, and no use of
sparsity. Its *accuracy* is fine -- the grouping SciPy uses returns exactly
`0.0` for identical rows.

## What a broadcast cannot reach

Each of the following is available to a compiled per-pair loop and to none of
the forms above. They are the mechanisms this package is built on;
`bench/bench_honest.py` times the package against SciPy, the published sparse
form and whichever third-party kernels are installed, and its `--threads 1`
mode separates kernel quality from parallelism.

- **One logarithm per pair-coordinate, and no temporaries.** `p ln p` depends
  on a single row, so it is tabulated once per entry and only the mixture term
  needs a logarithm inside the pair loop. Arithmetic, not vectorization
  (`docs/MATH.md`).
- **Threading over rows.** The problem is embarrassingly parallel and none of
  the forms above exploit it.
- **Skipping non-intersecting coordinates on sparse rows**, exactly, via the
  one-sided-mass identity (Lee 1997, below).
- **`float32`**, where ~1e-4 error is acceptable.
- **Duplicate rows at exactly `0.0`**, which is parity with SciPy, since it
  also returns `0.0`; the gain is only against sparse forms that fold the
  constant out of the sum.

None of it is available from a NumPy broadcast, which is why further effort on
that axis does not pay.

## Lineage and licence

The per-pair callable form is from `briankeithn/narrative-maps`
(`packages/solve_LP.py`, Keith & Mitra 2021; SoftwareX 2025), MIT,
(c) Brian Keith Norambuena. The broadcast and the `pdist` + `squareform` forms
are reimplementations of the same computation that grew around it in
downstream application code; both are kept because they are what the
alternatives look like in practice.

The benchmark fixtures in `bench/fixtures/` are real HDBSCAN soft-membership
matrices, stripped to bare numeric arrays; `bench/fixtures/README.md` gives
their shapes, what a row contains and why synthetic draws do not substitute
for them.

## The sparse identity is published prior art

The support-intersection identity the sparse kernels rest on is not original
to this package. It was published in 1997 and has been restated several times
since:

- **Lillian Lee (1997)**, *Similarity-Based Approaches to Natural Language
  Processing*, Harvard TR-11-97 (PhD thesis), section 2.3.2 Eq. (2.8) --
  <https://arxiv.org/abs/cmp-lg/9708011>. Splits the coordinates into
  `Both = {y : q(y)>0, r(y)>0}` and `Justq`, shows the `Justq` sum collapses,
  and states the motivation: *"Equation (2.8) is computationally
  convenient, for it involves sums only over elements of Both, as opposed to
  over all the elements in Y."*
- **Dagan, Lee & Pereira (1997)**, ACL/EACL, <https://aclanthology.org/P97-1008/>
  -- the operational claim in one sentence: *"the calculation ... requires
  summing only over those w2 for which both are non-zero, which, for sparse
  data, makes the computation quite fast."*
- **Dagan, Lee & Pereira (1999)**, *Machine Learning* 34(1-3):43-69.
- **Connor, Cardillo, Moss & Rabitti (2013)**, *Evaluation of Jensen-Shannon
  Distance over Sparse Data*, SISAP 2013, LNCS 8199 --
  <https://doi.org/10.1007/978-3-642-41062-8_16>. The same column-oriented
  sparse evaluation, benchmarked over 1e10 distance evaluations, plus an
  optional Jensen-inequality early-abandon this package does not have.
- **Luo, Yao & Xu (2024)**, ICML, Appendix C "Sparse Jensen-Shannon
  Divergence" -- <https://arxiv.org/abs/2406.02343>.

One difference remains. Those papers all fold the one-sided-mass term into a
leading `log 2 +` (or `1 - (1/2) * SUM`) constant. Keeping it unfolded, as this
package does, is what preserves the exact zero on duplicate rows. That is a
small implementation delta, not a new identity.

## No surveyed Python library exploits it

The identity being long published raises the question of what a package adds.
Surveyed, on the installed versions:

- **SciPy 1.17.1** -- the compiled pairwise kernel is
  `pdist_jensenshannon_double_wrap`. It groups each coordinate as
  `a ln(a/m) + b ln(b/m)`; confirmed by reimplementing both groupings in
  float64 and comparing bitwise against `pdist(X, "jensenshannon")` over 500
  random pairs -- that grouping matches 359/500 (the rest being 1-ULP `log`
  differences), while `a ln a + b ln b - 2 m ln m` matches **0/500**. It is
  dense: all `k` coordinates are visited whatever the support, and since `m`
  depends on the pair, neither logarithm can be hoisted out of the pair loop.
- **scikit-learn 1.9.0** -- no compiled JS metric exists.
  `metric='jensenshannon'` is rejected by the pairwise-distances whitelist,
  so the only route is a per-pair Python callable, which pays an interpreter
  round trip per pair exactly as `python_loop_matrix` does above.
- **pynndescent 0.6** -- has a Numba kernel, and a nominally sparse variant
  that does not use sparsity at all:

        # Because of the EPS values and the need to normalize after adding
        # them (and then average those for jensen_shannon) it seems like we
        # might as well just take the dense union (dense vectors supported
        # on the union of indices) and call the dense distance functions
        @numba.njit()
        def sparse_jensen_shannon_divergence(ind1, data1, ind2, data2):
            dense_data1, dense_data2 = dense_union(ind1, data1, ind2, data2)
            return jensen_shannon_divergence(dense_data1, dense_data2)

  It also adds `FLOAT32_EPS` (1.19e-07) to *every* coordinate before
  normalizing, so it computes a **smoothed** divergence: measured deviation
  6.0e-06 on dense rows, 1.2e-04 on 3 %-dense thresholded ones, growing with
  the number of zeros.

So the gap this package fills is not the identity, it is that nothing in
Python exploits it. That is also why the Lee-1997 baseline in
`bench/baselines.py` is written out rather than imported: there is nothing to
import.

## SciPy's pairwise path takes no `base`

`scipy.spatial.distance.pdist(X, "jensenshannon", base=2.0)` raises

```
TypeError: pdist_jensenshannon_double_wrap() takes no keyword arguments
```

(verified on SciPy 1.17.1). The scalar `jensenshannon()` accepts `base`; the
compiled pairwise path does not. SciPy therefore offers no direct route to a
base-2 pairwise JS matrix, and converting a base-e result requires knowing to
post-multiply by `1/sqrt(ln 2)`. This package takes `base` on every entry
point instead.
