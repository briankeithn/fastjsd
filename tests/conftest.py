import numpy as np
import pytest

from fastjsd import available_backends

BACKENDS = list(available_backends())


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def dirichlet(n, k, alpha, seed=0):
    return np.random.default_rng(seed).dirichlet(np.full(k, alpha), size=n)


def thresholded(n, k, alpha, seed=0):
    """The regime that motivated the sparse kernel: simplex rows thresholded
    at 1/k and renormalized (HDBSCAN soft memberships, topic mixtures)."""
    P = dirichlet(n, k, alpha, seed)
    P[P < 1.0 / k] = 0.0
    empty = ~P.any(axis=1)
    P[empty] = 1.0 / k
    return P / P.sum(axis=1, keepdims=True)
