"""Reference implementations, kept for tests and benchmarks -- not for use.

These are the implementations ``fastjsd`` is measured against and validated
on. They are kept verbatim in shape, so that a comparison is against what
people actually run and an agreement check is against what people actually
get.

``scipy_matrix`` / ``scipy_pdist``
    ``scipy.spatial.distance`` with ``metric="jensenshannon"``. Compiled
    per-pair loop; the correctness oracle for this package.

``python_loop_matrix``
    ``cdist`` with a Python lambda calling ``distance.jensenshannon`` per
    pair: one interpreter call and a pair of ``k``-length temporaries per
    pair, over the full square matrix of a symmetric metric.

``broadcast_matrix``
    A chunked NumPy broadcast over the pairwise mixture. Much faster than
    the Python loop, and slower than SciPy once ``k`` is large enough for
    the ``(chunk, n, k)`` temporary to leave cache: chunking bounds peak
    memory but not memory traffic, so the kernel is bandwidth-bound where
    SciPy is compute-bound. See ``docs/PRIOR_ART.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["scipy_matrix", "scipy_pdist", "python_loop_matrix",
           "broadcast_matrix"]


def scipy_pdist(P, base=2.0):
    from scipy.spatial import distance
    d = distance.pdist(np.asarray(P, dtype=np.float64), metric="jensenshannon")
    return d / np.sqrt(np.log(base))


def scipy_matrix(P, Q=None, base=2.0):
    from scipy.spatial import distance
    P = np.asarray(P, dtype=np.float64)
    if Q is None:
        d = distance.squareform(distance.pdist(P, metric="jensenshannon"))
    else:
        d = distance.cdist(P, np.asarray(Q, dtype=np.float64),
                           metric="jensenshannon")
    return d / np.sqrt(np.log(base))


def python_loop_matrix(P, base=2.0):
    """``cdist`` over a per-pair Python lambda."""
    from scipy.spatial import distance
    P = np.asarray(P, dtype=np.float64)
    return distance.cdist(P, P,
                          lambda u, v: distance.jensenshannon(u, v, base=base))


def broadcast_matrix(P, chunk=128):
    """Chunked NumPy broadcast over the pairwise mixture, base 2.

    Requires rows that already sum to 1 -- it does not normalize.
    """
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]

    def entropy(M):
        return -np.sum(np.where(M > 0, M * np.log2(np.where(M > 0, M, 1)), 0.0),
                       axis=-1)

    H_P = entropy(P)
    out = np.empty((n, n))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        M = 0.5 * (P[lo:hi, None, :] + P[None, :, :])
        divergence = entropy(M) - 0.5 * (H_P[lo:hi, None] + H_P[None, :])
        out[lo:hi] = np.sqrt(np.clip(divergence, 0, None))
    return out
