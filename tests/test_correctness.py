"""Agreement with SciPy -- the only claim that matters before speed."""

import numpy as np
import pytest
from conftest import dirichlet, thresholded
from scipy.spatial import distance

from fastjsd import jsd, jsd_matrix, jsd_pairwise, jsd_pdist
from fastjsd.reference import broadcast_matrix, python_loop_matrix, scipy_matrix

TOL = 5e-13


@pytest.mark.parametrize("n,k,alpha", [(2, 2, 1.0), (7, 3, 0.5), (60, 16, 0.1),
                                       (129, 64, 0.05), (200, 5, 2.0)])
def test_matrix_matches_scipy_dense(backend, n, k, alpha):
    P = dirichlet(n, k, alpha, seed=n + k)
    got = jsd_matrix(P, backend=backend)
    assert np.abs(got - scipy_matrix(P)).max() < TOL


@pytest.mark.parametrize("n,k,alpha", [(40, 12, 0.05), (150, 48, 0.02),
                                       (90, 25, 0.1)])
def test_matrix_matches_scipy_sparse(backend, n, k, alpha):
    P = thresholded(n, k, alpha, seed=n)
    assert (P > 0).mean() < 0.5, "fixture should exercise the sparse path"
    got = jsd_matrix(P, backend=backend)
    assert np.abs(got - scipy_matrix(P)).max() < TOL


def test_cross_matrix_matches_scipy(backend):
    P = thresholded(50, 20, 0.1, seed=1)
    Q = dirichlet(31, 20, 0.5, seed=2)
    got = jsd_matrix(P, Q, backend=backend)
    assert got.shape == (50, 31)
    assert np.abs(got - scipy_matrix(P, Q)).max() < TOL


def test_pdist_matches_scipy_and_squareform(backend):
    P = thresholded(64, 24, 0.08, seed=3)
    got = jsd_pdist(P, backend=backend)
    ref = distance.pdist(P, metric="jensenshannon") / np.sqrt(np.log(2.0))
    assert np.abs(got - ref).max() < TOL
    assert np.abs(distance.squareform(got)
                  - jsd_matrix(P, backend=backend)).max() < TOL


def test_pairwise_rows_and_broadcast():
    P = dirichlet(37, 9, 0.3, seed=4)
    Q = dirichlet(37, 9, 0.7, seed=5)
    ref = np.array([distance.jensenshannon(P[i], Q[i], base=2.0)
                    for i in range(37)])
    assert np.abs(jsd_pairwise(P, Q) - ref).max() < TOL
    one = Q[:1]
    ref1 = np.array([distance.jensenshannon(P[i], one[0], base=2.0)
                     for i in range(37)])
    assert np.abs(jsd_pairwise(P, one) - ref1).max() < TOL
    assert abs(jsd(P[0], Q[0]) - ref[0]) < TOL


def test_base_e_matches_scipy_default():
    P = dirichlet(24, 11, 0.4, seed=6)
    got = jsd_matrix(P, base=np.e)
    ref = distance.squareform(distance.pdist(P, metric="jensenshannon"))
    assert np.abs(got - ref).max() < TOL


def test_squared_is_the_divergence(backend):
    P = thresholded(40, 16, 0.1, seed=7)
    d = jsd_matrix(P, backend=backend)
    v = jsd_matrix(P, squared=True, backend=backend)
    assert np.abs(v - d ** 2).max() < 1e-12


def test_backends_agree_with_each_other():
    P = thresholded(80, 32, 0.06, seed=8)
    ref = None
    for b in ("numpy", "numba", "numba-sparse"):
        try:
            got = jsd_matrix(P, backend=b)
        except RuntimeError:
            continue                       # numba absent
        if ref is None:
            ref = got
        else:
            assert np.abs(got - ref).max() < TOL


def test_agrees_with_the_reference_implementations():
    """Same answers as both reference implementations in `fastjsd.reference`."""
    P = thresholded(70, 20, 0.1, seed=9)
    got = jsd_matrix(P)
    assert np.abs(got - python_loop_matrix(P)).max() < TOL
    assert np.abs(got - broadcast_matrix(P)).max() < TOL


def test_unnormalized_rows_are_normalized_like_scipy(backend):
    """SciPy normalizes internally; raw HDBSCAN memberships are not
    normalized, so this is the realistic input shape."""
    P = dirichlet(45, 14, 0.3, seed=10) * np.linspace(0.05, 3.0, 45)[:, None]
    assert not np.allclose(P.sum(1), 1.0)
    assert np.abs(jsd_matrix(P, backend=backend) - scipy_matrix(P)).max() < TOL


def test_float32_stays_within_its_documented_error(backend):
    P = dirichlet(96, 40, 0.1, seed=11)
    got = jsd_matrix(P, dtype=np.float32, backend=backend)
    assert got.dtype == np.float32
    assert np.abs(got - scipy_matrix(P)).max() < 5e-4
