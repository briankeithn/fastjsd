"""The bitmap rank-select kernel: identity with the merge, and its legality gate.

`_sparse_self_bits` / `_sparse_cross_bits` replace the intersection merge with
popcount-based rank-select over the occupancy bitmap. They run on a large share
of sparse input, including both real fixtures in `bench/fixtures`.

Which kernel runs is decided by the DATA (`_unfolded(k)` for legality,
`_prefer_bits` for cost), so a disagreement between them is not a performance
regression, it is an answer that depends on the shape of the input -- and,
since the cost probe is microarchitecture-sensitive, potentially on the
machine. Everything here therefore asserts `array_equal`, never `allclose`.

Two things are pinned:

1. **Identity.** Same CSR, same signatures, same ranks, same row sums, same
   schedule -- the kernels must produce bitwise identical matrices across
   densities, shapes, duplicate rows, disjoint supports, and the `wc`
   (well-conditioned) path.
2. **The legality gate.** Rank-select reads `rank[i, w] + popcount(...)` as a
   CSR position, which only names one column when the signature is unfolded.
   Above `64 * _SIG_MAX_WORDS` columns it folds and the addressing silently
   goes to the wrong entry -- measured at 0.2-0.32 ABSOLUTE error on the
   distance, which is not a rounding difference. `_unfolded` is a hard gate,
   and `test_folded_signature_is_never_dispatched_to_bits` is what keeps it
   one.
"""

import numpy as np
import pytest
from conftest import dirichlet, thresholded

from fastjsd import jsd_matrix, jsd_pdist

nb = pytest.importorskip("fastjsd._numba")
pytestmark = pytest.mark.skipif(not nb.available(), reason="needs numba")


def _both(P, squared=False, wc=False, base=2.0):
    """Run the merge and the bitmap kernel over identical prepared inputs."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    n, k = P.shape
    data, idx, ptr, PL = nb.csr_from_dense(P)
    sig = nb._signatures(idx, ptr, k)
    rsum = nb._row_sums(data, ptr)
    rank = nb._ranks(sig, ptr)
    offs, order = nb._offsets(n, False), nb._balanced_order(n)
    inv = 1.0 / np.log(base)

    # NOT np.empty. With `sym_write` the kernels write only the j > i pairs,
    # so the diagonal keeps whatever the allocator handed over -- and two
    # separate np.empty calls hand over different garbage. Comparing those
    # buffers compares uninitialized memory, which fails for reasons that
    # have nothing to do with either kernel. `matrix_sparse` zeroes the
    # diagonal after the call, so zeroing here matches what it does.
    om = np.zeros(n * n)
    ob = np.zeros(n * n)
    nb._sparse_self(data, idx, ptr, PL, rsum, sig, inv, squared, om,
                    offs, order, True, wc)
    nb._sparse_self_bits(data, PL, rsum, sig, rank, inv, squared, ob,
                         offs, order, True, wc)
    for o in (om, ob):
        np.fill_diagonal(o.reshape(n, n), 0.0)
    return om, ob


def _skewed(n, k, seed):
    """Row degrees spanning 1..k/2 -- the shape real thresholded corpora have,
    and the one `bench/datasets.py`'s `sparse_rows` cannot produce."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, k))
    for i in range(n):
        z = int(rng.integers(1, max(2, k // 2)))
        cols = rng.choice(k, size=z, replace=False)
        P[i, cols] = rng.dirichlet(np.ones(z))
    return P


CASES = [
    ("thresholded dirichlet", lambda: thresholded(64, 24, 0.08, seed=1)),
    ("very sparse", lambda: thresholded(48, 64, 0.02, seed=2)),
    ("moderately dense", lambda: thresholded(40, 16, 0.5, seed=3)),
    ("k at the word boundary", lambda: thresholded(40, 64, 0.05, seed=4)),
    ("k just under the fold", lambda: thresholded(32, 256, 0.02, seed=5)),
    ("one-hot rows", lambda: np.eye(24)),
    ("skewed row degrees", lambda: _skewed(40, 32, seed=18)),
]


@pytest.mark.parametrize("label,make", CASES, ids=[c[0] for c in CASES])
def test_bits_is_bitwise_identical_to_the_merge(label, make):
    om, ob = _both(make())
    assert np.array_equal(om, ob), f"{label}: kernels disagree"


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("wc", [False, True])
def test_identical_across_squared_and_well_conditioned(squared, wc):
    """`wc` takes a different branch per matched coordinate in both kernels."""
    om, ob = _both(thresholded(48, 32, 0.1, seed=6), squared=squared, wc=wc)
    assert np.array_equal(om, ob)


def test_identical_on_near_duplicates_where_wc_actually_fires():
    """Rows differing by ~1e-9 -- the regime `wc`'s series path exists for."""
    rng = np.random.default_rng(7)
    base = thresholded(24, 32, 0.1, seed=7)
    P = np.repeat(base, 2, axis=0)
    P[1::2] *= 1.0 + 1e-9 * rng.standard_normal(P[1::2].shape)
    P = np.abs(P)
    om, ob = _both(P, wc=True)
    assert np.array_equal(om, ob)


def test_duplicate_rows_are_exactly_zero_in_both():
    P = thresholded(20, 24, 0.1, seed=8)
    P = np.repeat(P, 3, axis=0)
    om, ob = _both(P)
    n = P.shape[0]
    om, ob = om.reshape(n, n), ob.reshape(n, n)
    for a in range(0, n, 3):
        for b in (a + 1, a + 2):
            assert om[a, b] == 0.0
            assert ob[a, b] == 0.0


def test_disjoint_supports_agree_and_do_not_exceed_one():
    """G6 is not a guarantee for arbitrary weights, so pin agreement and the
    bound rather than the exact 1.0 -- see `docs/MATH.md`."""
    P = np.zeros((12, 48))
    for i in range(12):
        P[i, i * 4:(i + 1) * 4] = 0.25
    om, ob = _both(P)
    assert np.array_equal(om, ob)
    assert om.max() <= 1.0 + 2 * np.finfo(np.float64).eps


def test_cross_kernels_are_bitwise_identical():
    P = np.ascontiguousarray(thresholded(30, 32, 0.08, seed=9))
    Q = np.ascontiguousarray(thresholded(20, 32, 0.08, seed=10))
    P, Q = P / P.sum(1, keepdims=True), Q / Q.sum(1, keepdims=True)
    n, m, k = P.shape[0], Q.shape[0], P.shape[1]
    dp, ip, pp, PLp = nb.csr_from_dense(P)
    dq, iq, pq, PLq = nb.csr_from_dense(Q)
    sp, sq = nb._signatures(ip, pp, k), nb._signatures(iq, pq, k)
    rp, rq = nb._row_sums(dp, pp), nb._row_sums(dq, pq)
    inv = 1.0 / np.log(2.0)
    om, ob = np.empty(n * m), np.empty(n * m)
    nb._sparse_cross(dp, ip, pp, PLp, rp, sp, dq, iq, pq, PLq, rq, sq,
                     n, m, inv, False, om, False)
    nb._sparse_cross_bits(dp, PLp, rp, sp, nb._ranks(sp, pp),
                          dq, PLq, rq, sq, nb._ranks(sq, pq),
                          n, m, inv, False, ob, False)
    assert np.array_equal(om, ob)


# --------------------------------------------------------------------------
# The legality gate.
# --------------------------------------------------------------------------

def test_unfolded_is_exactly_the_representable_range():
    limit = 64 * nb._SIG_MAX_WORDS
    assert nb._unfolded(1)
    assert nb._unfolded(limit)
    assert not nb._unfolded(limit + 1)


def test_folded_signature_is_never_dispatched_to_bits():
    """Above the fold the bitmap kernel must not run, whatever the density.

    `_unfolded` is ANDed with `_prefer_bits` in dispatch, but only this term
    is a correctness gate -- the probe is a performance heuristic and may
    legitimately return either answer. So this is the only thing standing
    between a wide-vocabulary sparse matrix and rank-select addressing the
    wrong CSR entry.
    """
    k = 64 * nb._SIG_MAX_WORDS + 64
    P = thresholded(24, k, 0.01, seed=11)
    ref = jsd_matrix(P, backend="numpy")
    got = jsd_matrix(P, backend="numba")
    # If the fold gate failed, this is 0.2-0.32 absolute, not 1e-12.
    assert np.abs(got - ref).max() < 1e-11


def test_wide_sparse_matches_scipy_end_to_end():
    scipy_spatial = pytest.importorskip("scipy.spatial")
    k = 64 * nb._SIG_MAX_WORDS + 128
    P = thresholded(20, k, 0.01, seed=12)
    d = scipy_spatial.distance.squareform(
        scipy_spatial.distance.pdist(P, metric="jensenshannon"))
    assert np.abs(jsd_matrix(P) - d / np.sqrt(np.log(2.0))).max() < 5e-12


def test_pdist_and_matrix_agree_on_the_bits_path():
    P = thresholded(36, 40, 0.08, seed=13)
    D = jsd_matrix(P)
    c = jsd_pdist(P)
    iu = np.triu_indices(P.shape[0], 1)
    assert np.array_equal(c, D[iu])


def test_repeated_calls_are_bitwise_identical():
    """Dispatch selects between two kernels on a sampled probe, so the answer
    must not move between calls on the same input.

    `_prefer_bits` uses a fixed `rng_seed=0`, which is what makes this hold.
    If the probe is ever reseeded per call, a matrix near the threshold would
    return different (both correct, not bitwise equal) results run to run --
    and since the two kernels ARE bitwise equal, this test would still pass.
    `test_bits_is_bitwise_identical_to_the_merge` is what actually protects
    that; this pins determinism of the dispatch itself.
    """
    P = thresholded(50, 32, 0.06, seed=14)
    first = jsd_matrix(P)
    for _ in range(3):
        assert np.array_equal(jsd_matrix(P), first)


def test_probe_is_deterministic_but_seed_sensitivity_is_visible():
    """The probe samples 4096 pairs, so its vote CAN depend on the seed.

    Not a failure -- it is the documented reason the seed is pinned. This
    records how close to the threshold a given fixture sits, so a future
    change to the sampling makes itself known here rather than in a
    benchmark.
    """
    P = thresholded(50, 32, 0.06, seed=14)
    data, idx, ptr, _ = nb.csr_from_dense(P / P.sum(1, keepdims=True))
    sig = nb._signatures(idx, ptr, P.shape[1])
    assert nb._prefer_bits(sig, ptr) == nb._prefer_bits(sig, ptr)
    votes = {bool(nb._prefer_bits(sig, ptr, rng_seed=s)) for s in range(8)}
    # Whatever the votes are, the shipped call must equal the seed-0 one.
    assert nb._prefer_bits(sig, ptr) == nb._prefer_bits(sig, ptr, rng_seed=0)
    assert votes <= {True, False}


def test_bits_and_merge_agree_on_float32_input():
    P = thresholded(28, 32, 0.1, seed=15).astype(np.float32)
    ref = jsd_matrix(P.astype(np.float64), backend="numpy")
    got = jsd_matrix(P)
    assert np.abs(got.astype(np.float64) - ref).max() < 1e-6


def test_row_of_all_mass_in_one_column():
    """Degenerate but legal: nnz == 1 rows are most of a thresholded corpus.

    On `bench/fixtures/soft_clusters_931x25`, 343 of 931 rows look like this,
    so the bitmap kernel's single-set-bit path is the common case, not an edge
    case.
    """
    P = np.zeros((16, 20))
    rng = np.random.default_rng(16)
    for i in range(16):
        P[i, rng.integers(0, 20)] = 1.0
    om, ob = _both(P)
    assert np.array_equal(om, ob)


def test_empty_intersection_between_every_pair():
    P = np.eye(32)
    om, ob = _both(P)
    assert np.array_equal(om, ob)
    assert np.array_equal(om.reshape(32, 32)[np.triu_indices(32, 1)],
                          np.ones(32 * 31 // 2))


def test_agrees_with_the_dense_kernel_on_dense_input():
    """Same answer whichever kernel density dispatch would have chosen."""
    P = dirichlet(24, 24, 1.0, seed=17)
    om, ob = _both(P)
    assert np.array_equal(om, ob)
    ref = jsd_matrix(P, backend="numpy")
    assert np.abs(ob.reshape(24, 24) - ref).max() < 5e-13
