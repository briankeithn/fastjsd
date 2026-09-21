"""The optional SVML backend, and its absence.

Every test here is skipped when `libsvml` or AVX-512 is missing, which is the
common case -- so the *absence* path is tested unconditionally instead, since
that is what most installations exercise.
"""

import numpy as np
import pytest
from conftest import dirichlet, thresholded

import fastjsd
from fastjsd import _svml, jsd_matrix, jsd_pdist
from fastjsd.reference import scipy_matrix, scipy_pdist

svml = pytest.mark.skipif(not _svml.available(),
                          reason=f"SVML unavailable: {_svml.why_unavailable()}")


# --- absence: always exercised ---------------------------------------------

def test_absence_is_graceful_and_explained():
    """A missing optional runtime must never be a hard error, and the reason
    must be legible -- "unavailable" with no cause sends people guessing at
    llvmlite builds, which is not where the problem is."""
    assert isinstance(_svml.available(), bool)
    assert _svml.why_unavailable()                       # never empty
    if not _svml.available():
        assert "numba-svml" not in fastjsd.available_backends()
        with pytest.raises(RuntimeError, match="numba-svml"):
            jsd_matrix(dirichlet(4, 8, 0.5, seed=1), backend="numba-svml")


def test_auto_never_selects_an_unavailable_backend():
    P = thresholded(30, 16, 0.2, seed=2)
    for Q in (None, thresholded(12, 16, 0.2, seed=3)):
        kernel = fastjsd.resolve_backend("auto", P, Q)
        assert kernel in fastjsd.available_backends()


def test_svml_threshold_is_lower_than_the_libm_one():
    """The SVML dense kernel is nearly density-independent, so the merge stops
    paying much earlier. If these ever converge, one of them is untuned."""
    assert fastjsd.SVML_DENSITY_MAX < fastjsd.SPARSE_DENSITY_MAX


# --- presence -------------------------------------------------------------

@svml
@pytest.mark.parametrize("n,k,alpha", [(40, 8, 0.5), (60, 64, 0.3),
                                       (33, 65, 0.2), (25, 256, 0.1)])
def test_matches_scipy(n, k, alpha):
    """Includes k not divisible by 8 (65), which exercises the scalar tail."""
    P = dirichlet(n, k, alpha, seed=n + k)
    assert np.abs(jsd_matrix(P, backend="numba-svml") - scipy_matrix(P)).max() < 5e-12


@svml
@pytest.mark.parametrize("k", [8, 65, 128])
def test_pdist_and_cross_match_scipy(k):
    P = dirichlet(35, k, 0.3, seed=k)
    Q = dirichlet(11, k, 0.3, seed=k + 1)
    assert np.abs(jsd_pdist(P, backend="numba-svml") - scipy_pdist(P)).max() < 5e-12
    ref = scipy_matrix(np.vstack([P, Q]))[:35, 35:]
    assert np.abs(jsd_matrix(P, Q, backend="numba-svml") - ref).max() < 5e-12


@svml
@pytest.mark.parametrize("density", [0.05, 0.3, 0.7, 1.0])
def test_correct_at_every_density(density):
    """The libm kernel branches on `a > 0`; this one is branch-free and
    substitutes `m <- 1` where both sides are zero. Without that substitution
    `2*m*log(m)` is `0 * -inf` = NaN, so sparse input would silently poison
    the matrix."""
    rng = np.random.default_rng(int(density * 1000))
    P = rng.dirichlet(np.full(48, 0.4), size=40)
    if density < 1.0:
        keep = rng.random(P.shape) < density
        keep[np.arange(40), rng.integers(0, 48, 40)] = True
        P = P * keep
    P = P / P.sum(axis=1, keepdims=True)
    D = jsd_matrix(P, backend="numba-svml")
    assert np.isfinite(D).all(), "NaN/inf leaked from a zero-vs-zero coordinate"
    assert np.abs(D - scipy_matrix(P)).max() < 5e-12


@svml
@pytest.mark.parametrize("k,seed", [(16, 9), (65, 3), (128, 2), (192, 6),
                                    (512, 18)])
def test_identical_rows_are_exactly_zero(k, seed):
    """SVML is up to 3 ULP from libm, so `PL` must be built with the SVML log.
    Mixing the two logarithms breaks the bitwise cancellation that makes
    duplicate rows exactly zero (`docs/MATH.md`), and does so three ULP wide
    rather than one."""
    P = np.repeat(dirichlet(1, k, 0.1, seed=seed), 6, axis=0)
    D = jsd_matrix(P, backend="numba-svml")
    assert np.all(D == 0.0), f"max {np.abs(D).max():.3e}"
    assert np.all(jsd_pdist(P, backend="numba-svml") == 0.0)


@svml
def test_agrees_with_the_libm_backend():
    P = thresholded(50, 40, 0.2, seed=77)
    a = jsd_matrix(P, backend="numba-svml")
    b = jsd_matrix(P, backend="numba")
    assert np.abs(a - b).max() < 1e-14


@svml
def test_squared_and_base_and_out_are_honoured():
    P = dirichlet(30, 32, 0.4, seed=5)
    d = jsd_matrix(P, backend="numba-svml")
    sq = jsd_matrix(P, backend="numba-svml", squared=True)
    assert np.abs(sq - d ** 2).max() < 1e-13
    e = jsd_matrix(P, backend="numba-svml", base=np.e)
    assert np.abs(e - d * np.sqrt(np.log(2.0))).max() < 1e-13
    buf = np.empty((30, 30))
    assert jsd_matrix(P, backend="numba-svml", out=buf) is buf
    assert np.array_equal(buf, d)


@svml
def test_auto_prefers_svml_on_dense_and_the_merge_on_sparse():
    dense = dirichlet(40, 64, 0.5, seed=11)
    assert fastjsd.resolve_backend("auto", dense) == "numba-svml"
    sparse = thresholded(40, 128, 0.03, seed=12)
    assert fastjsd.resolve_backend("auto", sparse) == "numba-sparse"
