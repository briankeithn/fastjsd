"""The opt-in float32 backend, ``numba-f32``.

It computes in float32, so agreement with the reference is asserted at
float32 tolerance -- against float64 arithmetic on the same float32 rows, so
the rounding of the input itself is not counted. What the package promises
exactly is asserted exactly: identical rows give 0.0 (G4), disjoint supports
1.0 in base 2 (G6), the diagonal is 0.0, the matrix is symmetric, and the
base-2 distance never exceeds 1.0.

The shared ``backend`` fixture in ``conftest.py`` leaves this backend out,
because the float64 tolerances the other test modules assert do not apply.
"""

import numpy as np
import pytest
from conftest import dirichlet, thresholded

import fastjsd
from fastjsd import _f32, jsd_matrix, jsd_pdist
from fastjsd._common import prepare
from fastjsd.reference import scipy_matrix

numba = pytest.importorskip("numba")
B = "numba-f32"
# Each pair is one float32 sum over its k coordinates. Where LLVM vectorizes
# that sum (x86-64) these tests measure up to 1.3e-7; where it leaves it
# scalar (AArch64, docs/MATH.md) up to 3.7e-7.
TOL = 1e-6


def f32(P):
    """The rows the kernel sees: normalized and snapped in float32."""
    return prepare(np.asarray(P, dtype=np.float64), dtype=np.float32)


def run(P, Q=None, **kw):
    return jsd_matrix(P, Q, backend=B, normalize=False, check=False, **kw)


def ref(P, Q=None, base=2.0):
    return scipy_matrix(P.astype(np.float64),
                        None if Q is None else Q.astype(np.float64), base=base)


# --- the backend as the API presents it -------------------------------------

def test_listed_last_and_never_chosen_by_auto():
    assert fastjsd.available_backends()[-1] == B
    for P in (dirichlet(30, 64, 0.5, seed=1), thresholded(30, 128, 0.03, seed=2)):
        assert fastjsd.resolve_backend("auto", P) != B


def test_it_computes_in_float32():
    P = dirichlet(20, 16, 0.5, seed=3)
    assert jsd_matrix(P, backend=B).dtype == np.float32
    assert jsd_pdist(P, backend=B).dtype == np.float32
    with pytest.raises(ValueError, match="float32"):
        jsd_matrix(P, backend=B, dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        jsd_matrix(P, backend=B, out=np.empty((20, 20)))
    buf = np.empty((20, 20), dtype=np.float32)
    assert jsd_matrix(P, backend=B, out=buf) is buf


def test_well_conditioned_has_no_off_switch_here():
    P = dirichlet(10, 16, 0.5, seed=4)
    a = jsd_matrix(P, backend=B, well_conditioned=True)
    assert np.array_equal(a, jsd_matrix(P, backend=B))
    with pytest.raises(ValueError, match="numba-f32"):
        jsd_matrix(P, backend=B, well_conditioned=False)


# --- accuracy -------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 7, 15, 16, 17, 65, 257, 1024])
def test_matches_the_reference(k):
    """Includes k below, at and just past the vector widths LLVM may pick."""
    rng = np.random.default_rng(10 + k)
    P, Q = f32(rng.random((19, k))), f32(rng.random((13, k)))
    assert np.abs(run(P, Q) - ref(P, Q)).max() < TOL
    assert np.abs(run(P) - ref(P)).max() < TOL


@pytest.mark.parametrize("regime", ["dirichlet", "sparse", "decades", "tiny"])
def test_hard_inputs_stay_at_float32_accuracy(regime):
    rng = np.random.default_rng(20)
    k = 200
    if regime == "dirichlet":
        X = rng.dirichlet(np.full(k, 0.1), 50)
    elif regime == "sparse":
        X = rng.random((50, k)) * (rng.random((50, k)) < 0.05)
        X[np.arange(50), rng.integers(0, k, 50)] += 1.0
    elif regime == "decades":                     # 30 decades of dynamic range
        X = 10.0 ** rng.uniform(-30, 0, (50, k))
    else:                                         # float32 subnormals
        X = rng.random((50, k)) * 1e-40 + (rng.random((50, k)) < 0.01)
    X = f32(X)
    assert np.isfinite(run(X)).all()
    assert np.abs(run(X[:30], X[30:]) - ref(X[:30], X[30:])).max() < TOL


@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2, 1e-1])
def test_close_pairs_have_float32_relative_accuracy(eps):
    """The far formula cancels as ``1/u^2`` on close coordinates
    (``docs/MATH.md``, "The limitation"); the series replaces it there, so the
    error on a close pair is relative, not absolute.

    The reference is the float64 dense kernel with ``well_conditioned=True``,
    not SciPy: SciPy's grouping cancels as ``2/u``, and at ``eps = 1e-6`` its
    float64 result is itself ~1e-4 off in relative terms (50-digit check in
    ``bench/f32.py``)."""
    rng = np.random.default_rng(int(1 / eps))
    p = rng.random((15, 257)) + 0.1
    P = f32(p)
    Q = f32(p * (1 + eps * rng.choice([-1.0, 1.0], p.shape)))
    got = np.diag(run(P, Q)).astype(np.float64)
    want = np.diag(jsd_matrix(P.astype(np.float64), Q.astype(np.float64),
                              backend="numba", well_conditioned=True,
                              normalize=False, check=False))
    assert np.all(np.abs(got - want) <= 1e-6 * want)


def test_series_switch_and_far_side():
    """Every coordinate at or just past ``|u| = 1/2``, where the kernel switches
    from the series to the far formula, with entries small enough that the far
    formula's cancellation factor ``~15 |ln m|`` is at its largest."""
    rng = np.random.default_rng(30)
    worst = 0.0
    for uu in (0.49, 0.5, 0.51, 0.6):
        p = rng.random((16, 64)) ** 3 + 1e-6
        g = rng.choice([-1.0, 1.0], p.shape)
        P, Q = f32(p), f32(p * (1 + g * uu) / (1 - g * uu))
        worst = max(worst, float(np.abs(run(P, Q) - ref(P, Q)).max()))
    assert worst < TOL


def test_inlined_log_is_within_two_ulp():
    """Every float32 in [0.5, 1), plus a log-uniform sweep down into the
    subnormals. The kernel's far branch and its ``x ln x`` table share it."""
    @numba.njit
    def logs(x, out):
        for i in range(x.shape[0]):
            out[i] = _f32._log(x[i])
        return out

    x = np.concatenate([
        np.arange(0x3F000000, 0x3F800000, dtype=np.uint32).view(np.float32),
        (10.0 ** np.random.default_rng(40).uniform(-45, 0, 200_000)).astype(np.float32),
    ])
    x = x[x > 0]
    got = logs(x, np.empty_like(x)).astype(np.float64)
    true = np.log(x.astype(np.float64))
    ulp = np.spacing(np.abs(true).astype(np.float32)).astype(np.float64)
    assert np.all(np.abs(got - true) <= 2.0 * ulp)


# --- what is promised exactly ------------------------------------------------

@pytest.mark.parametrize("k,seed", [(5, 1), (13, 2), (16, 3), (65, 4), (1001, 5)])
def test_identical_rows_are_exactly_zero(k, seed):
    """G4. The series gives exactly 0.0 at ``u = 0``; no fallback involved."""
    P = np.repeat(dirichlet(1, k, 0.5, seed=seed), 6, axis=0)
    assert np.all(jsd_matrix(P, backend=B) == 0.0)
    assert np.all(jsd_pdist(P, backend=B) == 0.0)
    assert np.all(jsd_matrix(P[:2], P, backend=B) == 0.0)


def test_disjoint_supports_are_exactly_one_for_any_weights():
    """G6, through the float64 pass that every near-maximal pair takes."""
    rng = np.random.default_rng(31)
    n, w = 12, 5
    P = np.zeros((n, n * w))
    for r in range(n):
        P[r, r * w:(r + 1) * w] = rng.dirichlet(np.ones(w))
    off = ~np.eye(n, dtype=bool)
    assert np.all(jsd_matrix(P, backend=B)[off] == 1.0)
    assert np.all(jsd_matrix(P, backend=B, squared=True)[off] == 1.0)
    assert np.all(jsd_pdist(P, backend=B) == 1.0)
    assert np.all(jsd_matrix(P[:5], P[5:], backend=B) == 1.0)
    assert np.all(jsd_matrix(np.eye(9), backend=B)[~np.eye(9, dtype=bool)] == 1.0)
    # Other bases: the value the float64 kernels give in float32, bit for bit.
    # It is not always float32(sqrt(ln 2)): the snap cannot make float32 rows
    # total exactly 1.0, and in base 2 the residual (~3e-8) is absorbed by
    # rounding to 1.0f, while near sqrt(ln 2) it can tip the last bit.
    e = jsd_matrix(P, backend=B, base=np.e)
    assert np.array_equal(e, jsd_matrix(P, backend="numba", dtype=np.float32,
                                        base=np.e))


def test_nearly_disjoint_rows_never_exceed_one():
    rng = np.random.default_rng(32)
    for k in (8, 64, 3000):
        P = rng.random((25, k)) * (rng.random((25, k)) < 0.2)
        P[:, 0] = 10.0 ** rng.uniform(-9, -3, 25)          # tiny shared mass
        assert jsd_matrix(P, backend=B).max() <= 1.0
        assert jsd_matrix(P, backend=B, squared=True).max() <= 1.0


def test_diagonal_symmetry_and_condensed_form():
    P = f32(dirichlet(40, 33, 0.3, seed=33))
    D = run(P)
    assert np.all(np.diag(D) == 0.0)
    assert np.array_equal(D, D.T)
    i, j = np.triu_indices(40, 1)
    assert np.array_equal(jsd_pdist(P, backend=B, normalize=False, check=False),
                          D[i, j])
    Q = f32(dirichlet(25, 33, 0.3, seed=34))
    assert np.array_equal(run(P, Q), run(Q, P).T)


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2,
                    reason="needs at least two Numba threads")
def test_thread_count_does_not_change_the_result():
    """Each pair is computed by one thread in one order, and skinny crosses
    take the pair-scheduled kernel at 2 threads but not at 1."""
    P, Q = f32(dirichlet(1, 100, 0.4, seed=35)), f32(dirichlet(300, 100, 0.4, seed=36))
    assert np.array_equal(run(P, Q, threads=1), run(P, Q, threads=2))
    assert np.array_equal(run(Q, threads=1), run(Q, threads=2))


def test_squared_and_base():
    P = f32(dirichlet(30, 32, 0.4, seed=37))
    d = run(P).astype(np.float64)
    sq = run(P, squared=True).astype(np.float64)
    assert np.abs(sq - d ** 2).max() < 1e-6
    e = run(P, base=np.e).astype(np.float64)
    assert np.abs(e - ref(P, base=np.e)).max() < TOL
