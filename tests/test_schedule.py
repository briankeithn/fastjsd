"""Pair scheduling for skinny cross matrices: same bits as the row loop.

`_use_flat_cross` hands cross matrices with fewer rows in `P` than threads to
kernels that iterate over pairs instead of rows. That is a scheduling change
only, so everything here asserts `array_equal`, never `allclose`: a result
that depended on the thread count would be a bug, not a rounding difference.
"""

import numpy as np
import pytest
from conftest import dirichlet, thresholded

from fastjsd import jsd_matrix

numba = pytest.importorskip("numba")
nb = pytest.importorskip("fastjsd._numba")
INV2 = 1.0 / np.log(2.0)


def _norm(P):
    return np.ascontiguousarray(P / P.sum(1, keepdims=True))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("wc", [False, True])
@pytest.mark.parametrize("squared", [False, True])
def test_dense_flat_is_bitwise_the_row_loop(dtype, wc, squared):
    P = _norm(dirichlet(3, 65, 0.3, seed=1)).astype(dtype)
    Q = _norm(dirichlet(70, 65, 0.3, seed=2)).astype(dtype)
    PL, QL = nb.xlogx_like_kernels(P), nb.xlogx_like_kernels(Q)
    a = np.empty(3 * 70, dtype=dtype)
    b = np.empty_like(a)
    nb._dense_cross(P, PL, Q, QL, INV2, squared, a, wc)
    nb._dense_cross_flat(P, PL, Q, QL, INV2, squared, b, wc)
    assert np.array_equal(a, b)


def _sparse_parts(P, Q):
    dp, ip, pp, PLp = nb.csr_from_dense(P)
    dq, iq, pq, PLq = nb.csr_from_dense(Q)
    k = P.shape[1]
    sp, sq = nb._signatures(ip, pp, k), nb._signatures(iq, pq, k)
    rp, rq = nb._row_sums(dp, pp), nb._row_sums(dq, pq)
    return dp, ip, pp, PLp, rp, sp, dq, iq, pq, PLq, rq, sq


@pytest.mark.parametrize("k", [64, 256, 600])     # 600 folds the signature
@pytest.mark.parametrize("wc", [False, True])
def test_sparse_merge_flat_is_bitwise_the_row_loop(k, wc):
    P = _norm(thresholded(3, k, 0.05, seed=k))
    Q = _norm(thresholded(90, k, 0.05, seed=k + 1))
    parts = _sparse_parts(P, Q)
    a, b = np.empty(3 * 90), np.empty(3 * 90)
    nb._sparse_cross(*parts, 3, 90, INV2, False, a, wc)
    nb._sparse_cross_flat(*parts, 3, 90, INV2, False, b, wc)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("k", [64, 130, 256])
@pytest.mark.parametrize("wc", [False, True])
def test_sparse_bits_flat_is_bitwise_the_row_loop(k, wc):
    P = _norm(thresholded(2, k, 0.08, seed=k))
    Q = _norm(thresholded(80, k, 0.08, seed=k + 2))
    dp, ip, pp, PLp, rp, sp, dq, iq, pq, PLq, rq, sq = _sparse_parts(P, Q)
    ka, kb = nb._ranks(sp, pp), nb._ranks(sq, pq)
    a, b = np.empty(2 * 80), np.empty(2 * 80)
    nb._sparse_cross_bits(dp, PLp, rp, sp, ka, dq, PLq, rq, sq, kb,
                          2, 80, INV2, False, a, wc)
    nb._sparse_cross_bits_flat(dp, PLp, rp, sp, ka, dq, PLq, rq, sq, kb,
                               2, 80, INV2, False, b, wc)
    assert np.array_equal(a, b)


def test_gate_only_takes_skinny_crosses():
    t = numba.get_num_threads()
    assert not nb._use_flat_cross(t, 10_000)     # enough rows for every thread
    assert not nb._use_flat_cross(1, 63)         # too little work to spread
    if t > 1:
        assert nb._use_flat_cross(1, 64)
        assert nb._use_flat_cross(t - 1, 5_000)


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2,
                    reason="needs at least two Numba threads")
@pytest.mark.parametrize("backend,data", [("numba", "dense"),
                                          ("numba-sparse", "sparse")])
def test_public_result_does_not_depend_on_thread_count(backend, data):
    """threads=1 takes the row loop, threads=2 the pair loop; same bits."""
    if data == "dense":
        P, Q = dirichlet(1, 48, 0.4, seed=7), dirichlet(500, 48, 0.4, seed=8)
    else:
        P, Q = thresholded(1, 200, 0.05, seed=7), thresholded(500, 200, 0.05, seed=8)
    one = jsd_matrix(P, Q, backend=backend, threads=1)
    two = jsd_matrix(P, Q, backend=backend, threads=2)
    assert np.array_equal(one, two)
