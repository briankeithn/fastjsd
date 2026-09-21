"""Mathematical properties that must hold regardless of kernel or input."""

import numpy as np
import pytest
from conftest import dirichlet, thresholded

from fastjsd import available_backends, jsd_matrix, jsd_pdist, to_similarity


def test_zero_diagonal_and_symmetry(backend):
    P = thresholded(60, 18, 0.1, seed=20)
    D = jsd_matrix(P, backend=backend)
    assert np.all(np.diag(D) == 0.0)
    assert np.array_equal(D, D.T)


def test_bounded_in_unit_interval_base_two(backend):
    P = thresholded(80, 24, 0.05, seed=21)
    D = jsd_matrix(P, backend=backend)
    assert D.min() >= 0.0
    assert D.max() <= 1.0 + 1e-12


def test_disjoint_supports_are_exactly_one(backend):
    """Maximum divergence, on the one-hot case, where it is exact.

    Only ever assert this for weights that are powers of two -- see the test
    below, which is the honest statement of the property.
    """
    P = np.eye(6)
    D = jsd_matrix(P, backend=backend)
    off = D[~np.eye(6, dtype=bool)]
    assert np.all(off == 1.0)


def test_disjoint_supports_are_one_to_within_two_ulp(backend):
    """The general statement, because exactness here is a coincidence.

    `np.eye(n)` puts weight 1.0 on one coordinate, and `bench/accuracy.py`'s
    disjoint regime uses `w = max(1, k//n)` uniform weights -- both land on
    powers of two, for which `SUM_c a_c ln a_c` happens to round to exactly
    `-S_i ln 2`. With ordinary non-uniform weights it does not: 4.1 % of pairs
    on the sparse kernels and 27.1 % on `numba-svml` come back 1-2 ULP low.
    Row sums are exactly 1.0 throughout, so the residue comes from the
    coordinate summation, not from normalization.

    G4 (duplicate rows give exactly 0.0) is a real bitwise cancellation and is
    asserted exactly. This one is not, and pinning it as exact would only
    encode the fixtures that happen to be chosen here.
    """
    rng = np.random.default_rng(31)
    n, w = 12, 5                       # 5 is not a power of two, which is the point
    P = np.zeros((n, n * w))
    for r in range(n):
        P[r, r * w:(r + 1) * w] = rng.dirichlet(np.ones(w))
    D = jsd_matrix(P, backend=backend)
    off = D[~np.eye(n, dtype=bool)]
    assert np.all(np.abs(off - 1.0) <= 2 * np.finfo(np.float64).eps)
    assert np.all(off <= 1.0), "the metric bound must hold whatever the rounding"


@pytest.mark.parametrize("k,alpha,seed", [(12, 0.3, 22), (25, 0.1, 220),
                                          (64, 0.05, 221), (3, 2.0, 222)])
def test_identical_rows_are_exactly_zero(backend, k, alpha, seed):
    """Not merely small -- exactly 0.0.

    SciPy and the textbook grouping also return exactly 0.0 here, so this is
    parity with them, not an improvement. What it guards is the combination:
    the kernel keeps the exact zero WHILE walking only the support
    intersection. Folding the ``ln 2`` constant out front -- which is how the
    published sparse forms are written (Lee 1997, Eq. 2.8) -- makes it cancel
    numerically instead of symbolically and yields ~3.3e-8 here, breaking
    equality checks, thresholding and near-duplicate detection.
    """
    row = dirichlet(1, k, alpha, seed=seed)
    P = np.repeat(row, 6, axis=0)
    assert np.all(jsd_matrix(P, backend=backend) == 0.0)
    assert np.all(jsd_pdist(P, backend=backend) == 0.0)


def _sparse_rows(n, k, nnz, seed):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, k))
    for r in range(n):
        P[r, rng.choice(k, size=nnz, replace=False)] = rng.random(nnz) + 0.05
    return P / P.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("k", [32, 64, 65, 128, 256])
def test_signature_never_skips_an_overlapping_pair(k):
    """The one correctness property the merge skip depends on.

    The sparse kernels skip the merge when the rows' occupancy bitmaps AND to
    zero in every word. The bitmap is ``nw`` 64-bit words with one bit per
    ``column % (64 nw)``, so columns still fold once ``k`` exceeds the width
    and the test yields false POSITIVES -- pairs that fall through to a merge
    which finds nothing, costing time but not correctness. A false NEGATIVE
    would silently drop real intersection terms, so assert directly that the
    bitmap never claims disjoint when the supports actually overlap.
    """
    from fastjsd._numba import available
    if not available():
        pytest.skip("numba not installed")
    from fastjsd._numba import _signatures, csr_from_dense
    P = _sparse_rows(40, k, 4, seed=700 + k)
    _, idx, ptr, _ = csr_from_dense(P)
    sig = _signatures(idx, ptr, k)
    support = [set(np.flatnonzero(P[r])) for r in range(40)]
    for i in range(40):
        for j in range(i + 1, 40):
            if not (sig[i] & sig[j]).any():
                assert not (support[i] & support[j]), \
                    f"mask claimed rows {i},{j} disjoint but they overlap"


@pytest.mark.parametrize("k", [16, 64, 65, 128, 256, 1024])
def test_signature_width_follows_k(k):
    """Width tracks ``k`` up to the cap, and is exact where it can be.

    A single 64-bit word decays as ``exp(-z^2/64)`` in non-zeros per row: at
    z=16 it fires on 1.8 % of pairs while 36 % of them genuinely share nothing
    at k=256. That is not a correctness bug -- it costs only merges that find
    nothing -- but on wide sparse input those wasted merges are most of the
    kernel's work, so the width is pinned here rather than left to drift.
    """
    from fastjsd._numba import available
    if not available():
        pytest.skip("numba not installed")
    from fastjsd._numba import _SIG_MAX_WORDS, _signatures, csr_from_dense
    P = _sparse_rows(8, k, min(4, k), seed=k)
    _, idx, ptr, _ = csr_from_dense(P)
    sig = _signatures(idx, ptr, k)
    assert sig.shape == (8, min(_SIG_MAX_WORDS, max(1, (k + 63) // 64)))
    if k <= 64 * _SIG_MAX_WORDS:
        # No fold at this width, so the test is exact: bitmaps AND to zero
        # exactly when the supports are disjoint, not merely when they might be.
        support = [set(np.flatnonzero(P[r])) for r in range(8)]
        for i in range(8):
            for j in range(i + 1, 8):
                overlap = bool(support[i] & support[j])
                assert bool((sig[i] & sig[j]).any()) == overlap


@pytest.mark.parametrize("k", [32, 65, 256])
def test_backends_agree_on_mostly_disjoint_rows(k):
    """All four backends on data where most pairs share no coordinate.

    Agreement is to ~1e-13, not bitwise: the sparse kernels recover one-sided
    mass by complement while the dense ones accumulate it, and the two
    associate the same additions differently. Measured at 1 ULP.
    """
    P = _sparse_rows(40, k, 4, seed=900 + k)
    ref = jsd_matrix(P, backend="numpy")
    for b in ("numpy-sparse", "numba", "numba-sparse"):
        if b not in available_backends():
            continue
        got = jsd_matrix(P, backend=b)
        assert np.abs(got - ref).max() < 1e-15, b


def test_fully_disjoint_rows_skip_every_merge(backend):
    """Every pair is disjoint, so every pair takes the skip path. With base 2
    and disjoint support the distance is exactly 1."""
    k = 200
    P = np.zeros((25, k))
    for r in range(25):
        P[r, 8 * r:8 * r + 8] = 0.125        # blocks that cannot overlap
    D = jsd_matrix(P, backend=backend)
    off = D[~np.eye(25, dtype=bool)]
    assert np.all(off == 1.0)


@pytest.mark.parametrize("k,seed", [(16, 9), (128, 2), (128, 20),
                                    (192, 6), (512, 18)])
def test_identical_rows_exact_when_the_two_logs_disagree(backend, k, seed):
    """NumPy's ``log`` and Numba's ``log`` are not the same function.

    On each of these draws they differ by 1 ULP on at least one coordinate,
    which is enough to break duplicate-exactness in the Numba backends alone.
    Precomputing ``PL`` with ``np.log`` on the host while the kernels call libm
    in the loop stops ``PL[i,c] + PL[j,c]`` from cancelling ``2 m log m``
    bitwise, leaving ~1e-17 in the divergence and ~6e-9 in the distance. The
    table and the kernels must therefore use the same logarithm at the same
    precision.

    The parametrized cases in ``test_identical_rows_are_exactly_zero`` use
    draws where the two logs agree, so they do not cover this; the seeds here
    are draws where they disagree.
    """
    P = np.repeat(dirichlet(1, k, 0.1, seed=seed), 6, axis=0)
    D = jsd_matrix(P, backend=backend)
    assert np.all(D == 0.0), f"max {np.abs(D).max():.3e}"
    assert np.all(jsd_pdist(P, backend=backend) == 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k,seed", [(16, 9), (128, 2), (192, 6), (512, 18)])
def test_identical_rows_are_exactly_zero_in_both_dtypes(backend, dtype, k, seed):
    """``PL`` is float64 whatever the input and output dtype.

    The inner loop is float64 whatever the input dtype -- the ``0.5`` and
    ``2.0`` literals promote ``m`` and everything after it. A float32 ``PL``
    would carry a rounding the float64 mixture term does not, so the two sides
    would no longer cancel and duplicate rows would come back at ~1e-4 instead
    of exactly zero.
    """
    P = np.repeat(dirichlet(1, k, 0.1, seed=seed), 6, axis=0)
    D = jsd_matrix(P, backend=backend, dtype=dtype)
    assert D.dtype == dtype
    assert np.all(D == 0.0), f"max {np.abs(D).max():.3e}"
    assert np.all(jsd_pdist(P, backend=backend, dtype=dtype) == 0.0)


def test_precomputed_xlogx_matches_the_kernels_log():
    """The invariant the two tests above depend on, checked directly."""
    from fastjsd import _numba
    if not _numba.available():
        pytest.skip("numba not installed")
    a = dirichlet(1, 512, 0.1, seed=18)[0]
    a = a / a.sum()
    # Same input, same expression the kernel evaluates: must be bitwise equal.
    assert np.array_equal(_numba.xlogx_like_kernels(a[None, :])[0], a * _kernel_log(a))


def _kernel_log(a):
    """``log`` as the compiled kernels see it."""
    from numba import njit

    @njit
    def _f(x, out):
        for i in range(x.shape[0]):
            out[i] = np.log(x[i])
        return out
    return _f(np.ascontiguousarray(a), np.empty_like(a))


def test_duplicate_rows_inside_a_larger_matrix_are_exactly_zero(backend):
    """The same, with the duplicates buried in ordinary data rather than
    filling the matrix."""
    P = thresholded(80, 25, 0.1, seed=223)
    P[17] = P[3]
    P[64] = P[3]
    D = jsd_matrix(P, backend=backend)
    for a, b in [(3, 17), (3, 64), (17, 64)]:
        assert D[a, b] == 0.0, f"D[{a},{b}] = {D[a, b]!r}"


def test_triangle_inequality(backend):
    """The square root of the divergence is a metric; the divergence is not."""
    P = dirichlet(45, 8, 0.4, seed=23)
    D = jsd_matrix(P, backend=backend)
    assert (D[:, None, :] + D[None, :, :].transpose(0, 2, 1) >= D[:, :, None]
            - 1e-12).all()


def test_divergence_is_not_a_metric():
    """Guards the docstring claim: squared=True genuinely breaks the
    triangle inequality, so it must not be recommended as a distance."""
    P = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    V = jsd_matrix(P, squared=True)
    assert V[0, 2] > V[0, 1] + V[1, 2] + 1e-12


def test_column_permutation_invariance(backend):
    P = thresholded(50, 20, 0.1, seed=24)
    perm = np.random.default_rng(0).permutation(20)
    a = jsd_matrix(P, backend=backend)
    b = jsd_matrix(P[:, perm], backend=backend)
    assert np.abs(a - b).max() < 1e-13


def test_row_permutation_equivariance(backend):
    P = thresholded(40, 16, 0.1, seed=25)
    perm = np.random.default_rng(1).permutation(40)
    a = jsd_matrix(P, backend=backend)[np.ix_(perm, perm)]
    assert np.abs(a - jsd_matrix(P[perm], backend=backend)).max() < 1e-13


def test_all_zero_columns_do_not_change_the_answer(backend):
    P = thresholded(40, 12, 0.1, seed=26)
    padded = np.concatenate([P, np.zeros((40, 25))], axis=1)
    assert np.abs(jsd_matrix(P, backend=backend)
                  - jsd_matrix(padded, backend=backend)).max() < 1e-13


def test_similarity_helper():
    P = thresholded(20, 8, 0.2, seed=27)
    assert np.allclose(to_similarity(jsd_matrix(P)), 1.0 - jsd_matrix(P))
    with pytest.raises(ValueError):
        to_similarity(jsd_matrix(P, base=np.e), base=np.e)


def test_single_column_is_all_zero(backend):
    P = np.ones((7, 1))
    assert np.all(jsd_matrix(P, backend=backend) == 0.0)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_degenerate_row_counts(backend, n):
    P = dirichlet(max(n, 1), 4, 0.5, seed=28)[:n]
    D = jsd_matrix(P, backend=backend)
    assert D.shape == (n, n)
    assert jsd_pdist(P, backend=backend).shape == (n * (n - 1) // 2,)


# --------------------------------------------------------------------------
# conditioning near zero
# --------------------------------------------------------------------------

def _near_duplicate(n, k, delta, seed):
    """Rows differing by ~`delta`, with IDENTICAL support -- the regime where
    the per-coordinate summand is ill-conditioned."""
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.full(k, 0.4), size=1)
    P = np.repeat(base, n, axis=0) * (1.0 + rng.normal(0, delta, (n, k)))
    return np.ascontiguousarray(P / P.sum(axis=1, keepdims=True))


@pytest.mark.parametrize("kernel", ["numba", "numba-sparse"])
def test_well_conditioned_fixes_near_duplicate_rows(kernel):
    """The documented limitation, and that the opt-in flag removes it.

    `t = a ln a + b ln b - 2 m ln m` is computed from three pieces each
    O(a ln a) while `t` itself is O(m u^2), u = (a-b)/(a+b). The measured
    condition number is 9.2e18 at `u = 1e-9`, so the default kernel
    returns essentially rounding there -- exactness AT a == b does not extend
    to accuracy NEAR it.

    Tested by SCALING rather than against a reference, which needs no
    high-precision arithmetic and states the property directly: JS is locally
    quadratic, so the distance between rows differing by `delta` is linear in
    `delta`. Shrink `delta` tenfold and a well-conditioned kernel returns
    tenfold smaller distances; an ill-conditioned one plateaus on its own
    rounding floor. `bench/accuracy.py` does the absolute comparison against
    50-digit mpmath (1.5e-08 -> 1.8e-17).
    """
    if kernel not in available_backends():
        pytest.skip(f"{kernel} unavailable")
    got = {}
    for wc in (False, True):
        got[wc] = [jsd_matrix(_near_duplicate(10, 48, d, seed=4242),
                              backend=kernel, well_conditioned=wc).max()
                   for d in (1e-9, 1e-10, 1e-11)]

    # well-conditioned: each tenfold smaller perturbation gives a ~tenfold
    # smaller distance (allow a wide band -- this is a decade, not a fit).
    tight = got[True]
    for lo, hi in zip(tight[1:], tight[:-1], strict=True):
        assert 4.0 < hi / lo < 25.0, f"expected ~10x, got {hi / lo:.1f}: {tight}"

    # default: pinned to its rounding floor, so it stops tracking delta at all
    loose = got[False]
    assert loose[-1] / tight[-1] > 100.0, (
        f"expected the default kernel to plateau: {loose} vs {tight}")


@pytest.mark.parametrize("kernel", ["numba", "numba-sparse"])
def test_well_conditioned_agrees_elsewhere(kernel):
    """It must not perturb the regimes that are already at a few ULP."""
    if kernel not in available_backends():
        pytest.skip(f"{kernel} unavailable")
    P = thresholded(50, 40, 0.2, seed=808)
    a = jsd_matrix(P, backend=kernel, well_conditioned=False)
    b = jsd_matrix(P, backend=kernel, well_conditioned=True)
    assert np.abs(a - b).max() < 1e-14


@pytest.mark.parametrize("kernel", ["numba", "numba-sparse"])
def test_well_conditioned_keeps_the_exact_cases(kernel):
    if kernel not in available_backends():
        pytest.skip(f"{kernel} unavailable")
    dup = np.repeat(dirichlet(1, 96, 0.2, seed=11), 6, axis=0)
    assert np.all(jsd_matrix(dup, backend=kernel, well_conditioned=True) == 0.0)
    assert np.all(jsd_pdist(dup, backend=kernel, well_conditioned=True) == 0.0)
    eye = np.eye(7)
    D = jsd_matrix(eye, backend=kernel, well_conditioned=True)
    assert np.all(D[~np.eye(7, dtype=bool)] == 1.0)


def test_well_conditioned_is_refused_where_unimplemented():
    """Silently ignoring an EXPLICIT request would be worse than refusing: the
    caller asked for accuracy and would not learn they did not get it."""
    P = dirichlet(10, 8, 0.5, seed=12)
    for bad in ("numpy", "numpy-sparse"):
        with pytest.raises(ValueError, match="well_conditioned"):
            jsd_matrix(P, backend=bad, well_conditioned=True)


def test_well_conditioned_defaults_on_where_it_is_free():
    """The default is per kernel, not global.

    On the sparse merge the branch almost never fires -- matched coordinates
    are rare -- so the cancellation-free summand is close to free, and on the
    membership fixtures it improves the worst ill-conditioned pair by 4-5
    orders of magnitude. On the dense loop the same branch fires on every
    coordinate, so it carries a cost the caller should choose rather than
    inherit.
    """
    from fastjsd._api import _resolve_well_conditioned as resolve
    assert resolve(None, "numba-sparse") is True
    assert resolve(None, "numba") is False
    # An unset default must never raise on a backend that cannot do it.
    for k in ("numpy", "numpy-sparse", "numba-svml"):
        assert resolve(None, k) is False
    assert resolve(False, "numba-sparse") is False
    assert resolve(True, "numba") is True


def test_default_sparse_path_is_accurate_near_zero():
    """The consequence of that default, on the shape real data has.

    Rows differing by ~1e-9 with identical support: the sparse kernel is
    accurate there by default, and `well_conditioned=False` selects the
    cancelling summand instead, which is how this test tells the two apart
    without a high-precision reference.
    """
    if "numba-sparse" not in available_backends():
        pytest.skip("numba-sparse unavailable")
    P = _near_duplicate(10, 48, 1e-9, seed=4242)
    P[P < 1e-3] = 0.0                      # give it a support to be sparse on
    P = P / P.sum(axis=1, keepdims=True)
    default = jsd_matrix(P, backend="numba-sparse")
    opted_out = jsd_matrix(P, backend="numba-sparse", well_conditioned=False)
    # The achievable ratio is bounded by the input, not by the well-conditioned
    # summand: the true distance here is ~7e-10, itself only ~20x above the
    # opted-out kernel's error floor, so 20x is the whole available headroom.
    # Shrink `delta` and the ratio grows without limit -- `bench/accuracy.py`
    # measures the absolute improvement (1.4e-13 -> 1.7e-18 on real fixtures).
    assert opted_out.max() > 5.0 * default.max(), (
        f"default {default.max():.2e} vs opted out {opted_out.max():.2e}")


def test_identical_support_gives_exactly_zero_one_sided_mass():
    """The sparse kernel recovers one-sided mass by complement, which is a
    subtraction. Two accumulators (one per row, each summed sequentially in
    stored column order) make it exactly 0.0 whenever the supports match -- for
    ANY values, not only bitwise-identical rows. A single `hit += a + b`
    associates the additions differently and leaves ~k*eps, which caps
    near-duplicate accuracy at ~2e-8."""
    if "numba-sparse" not in available_backends():
        pytest.skip("numba-sparse unavailable")
    rng = np.random.default_rng(99)
    P = np.zeros((30, 64))
    cols = rng.choice(64, size=20, replace=False)      # same support, all rows
    P[:, cols] = rng.random((30, 20)) + 0.05
    P /= P.sum(axis=1, keepdims=True)
    sp = jsd_matrix(P, backend="numba-sparse")
    dn = jsd_matrix(P, backend="numba")
    assert np.abs(sp - dn).max() < 1e-15
