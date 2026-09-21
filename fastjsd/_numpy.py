"""Pure-NumPy backends: a tiled dense kernel and a column-scatter sparse one.

The obvious NumPy vectorization broadcasts a ``(rows, n, k)`` mixture tensor
and reduces it. That tensor *is* the problem: at ``n=5408, k=192`` a
128-row chunk is a 1.06 GB temporary, so the kernel runs at memory
bandwidth rather than at compute, unlike a cache-resident scalar C loop such
as SciPy's (see ``docs/PRIOR_ART.md``). Two ways out without a compiler:

``matrix`` / ``pdist``
    Keep the ``(ti, tj, k)`` tile inside a fixed byte budget, compute only
    the upper triangle, precompute ``x ln x``.

``matrix_sparse``
    Turn the loop inside out: iterate over *columns*, and for each one
    touch only the rows with support there. Total work becomes
    ``sum_c m_c^2`` instead of ``n^2 k`` -- for density ``d`` that is
    ``d^2`` of the dense work, which is the same asymptotic win the sparse
    Numba kernel gets. Needs the square accumulator, so it has no condensed
    form.
"""

from __future__ import annotations

import numpy as np

from ._common import LN2, xlogx

__all__ = ["matrix", "pdist", "matrix_sparse", "TILE_BYTES"]

#: Byte budget for the ``(ti, tj, k)`` mixture tile. ~8 MB keeps the working
#: set near L2/L3 on typical x86 server cores; a tuning knob, never a
#: correctness parameter.
TILE_BYTES = 8 << 20


def _tile_rows(k: int, itemsize: int, tile_bytes: int) -> int:
    # Three live temporaries of tile shape (mixture, log, product) -> /3.
    per = max(1, k * itemsize * 3)
    return int(max(8, min(4096, np.sqrt(tile_bytes / per))))


class _Scratch:
    """Tile-sized buffers, allocated once per call rather than per tile.

    A ``(ti, tj, k)`` tile is megabytes; reallocating four of them for every
    block is pure allocator churn.
    """

    __slots__ = ("M", "L", "S", "shape")

    def __init__(self, ti, tj, k, dtype):
        self.shape = (ti, tj, k)
        self.M = np.empty(self.shape, dtype=dtype)
        self.L = np.empty(self.shape, dtype=dtype)
        self.S = np.empty(self.shape, dtype=dtype)

    def views(self, ni, nj, k):
        return (self.M[:ni, :nj, :k], self.L[:ni, :nj, :k],
                self.S[:ni, :nj, :k])


def _block(Pi, Pj, Li, Lj, inv, squared, scratch):
    """Distances between every row of ``Pi`` and every row of ``Pj``.

    ``Li + Lj`` is summed *before* being combined with the mixture term.
    Adding them one at a time would make ``D[a, b]`` and ``D[b, a]`` differ
    in the last bit, because float addition is commutative but not
    associative -- and a distance matrix that is not exactly symmetric
    breaks every downstream consumer that assumes it is.

    One-sided coordinates are handled separately rather than left to the
    general term. ``PL_i + PL_j - 2 m ln m`` equals ``p ln 2`` in exact
    arithmetic, but in float64 it evaluates to ``a ln a - a ln(a/2)`` with the
    two logarithms independently rounded, so it reproduces ``a ln 2`` only to
    within an ULP -- which is enough to stop disjoint supports returning
    exactly 1.0 here.

    Each row's one-sided mass is accumulated with ``cumsum`` rather than
    ``sum``. ``sum`` reduces in a tree; ``cumsum`` accumulates left to right,
    which is the order ``_common.snap_row_sums`` made add up to exactly 1.0.
    Reducing the same values in a tree misses the row totals by an ULP again:
    only a third of snapped rows also come to exactly one under ``sum``.
    Masking with ``where(b > 0, 0, a)`` rather than subtracting a matched
    total keeps both limits exact at once: for disjoint rows the masked array
    *is* the row, so the total is exactly 1.0, and for identical rows it is
    all zeros, so the total is exactly 0.0 and duplicates stay at exactly
    0.0.
    """
    ni, nj, k = Pi.shape[0], Pj.shape[0], Pi.shape[1]
    M, L, S = scratch.views(ni, nj, k)
    bi = Pi[:, None, :] > 0.0
    bj = Pj[None, :, :] > 0.0
    both = bi & bj
    np.add(Pi[:, None, :], Pj[None, :, :], out=M)
    M *= 0.5
    np.add(Li[:, None, :], Lj[None, :, :], out=S)
    L.fill(0.0)
    np.log(M, out=L, where=M > 0)
    L *= M
    L *= -2.0
    L += S
    L *= both                       # matched coordinates only
    D = L.sum(axis=-1)

    np.copyto(S, Pi[:, None, :])
    np.copyto(S, 0.0, where=bj)     # row i's mass where row j has none
    np.cumsum(S, axis=-1, out=S)
    one = S[..., -1].copy()
    np.copyto(S, Pj[None, :, :])
    np.copyto(S, 0.0, where=bi)
    np.cumsum(S, axis=-1, out=S)
    one += S[..., -1]

    D += LN2 * one
    D *= 0.5 * inv
    np.clip(D, 0.0, None, out=D)
    if not squared:
        np.sqrt(D, out=D)
    return D


def matrix(P, Q, inv, squared, out, tile_bytes=TILE_BYTES):
    n, k = P.shape
    PL = xlogx(P)
    tile = _tile_rows(k, P.dtype.itemsize, tile_bytes)

    cols = n if Q is None else Q.shape[0]
    scratch = _Scratch(min(tile, n), min(tile, cols), k, P.dtype)

    if Q is None:
        if out is None:
            out = np.empty((n, n), dtype=P.dtype)
        for i0 in range(0, n, tile):
            i1 = min(i0 + tile, n)
            for j0 in range(i0, n, tile):
                j1 = min(j0 + tile, n)
                D = _block(P[i0:i1], P[j0:j1], PL[i0:i1], PL[j0:j1], inv,
                           squared, scratch)
                out[i0:i1, j0:j1] = D
                if j0 != i0:
                    out[j0:j1, i0:i1] = D.T
        np.fill_diagonal(out, 0.0)
        return out

    m = Q.shape[0]
    QL = xlogx(Q)
    if out is None:
        out = np.empty((n, m), dtype=P.dtype)
    for i0 in range(0, n, tile):
        i1 = min(i0 + tile, n)
        for j0 in range(0, m, tile):
            j1 = min(j0 + tile, m)
            out[i0:i1, j0:j1] = _block(P[i0:i1], Q[j0:j1], PL[i0:i1],
                                       QL[j0:j1], inv, squared, scratch)
    return out


def pdist(P, inv, squared, out, tile_bytes=TILE_BYTES):
    """Condensed upper triangle, SciPy ``pdist`` ordering."""
    n, k = P.shape
    PL = xlogx(P)
    tile = _tile_rows(k, P.dtype.itemsize, tile_bytes)
    if out is None:
        out = np.empty(n * (n - 1) // 2, dtype=P.dtype)
    starts = np.zeros(n, dtype=np.int64)
    if n:
        starts[1:] = np.cumsum(np.arange(n - 1, 0, -1, dtype=np.int64))
    scratch = _Scratch(min(tile, n), min(tile, n), k, P.dtype)
    for i0 in range(0, n, tile):
        i1 = min(i0 + tile, n)
        for j0 in range(i0, n, tile):
            j1 = min(j0 + tile, n)
            D = _block(P[i0:i1], P[j0:j1], PL[i0:i1], PL[j0:j1], inv, squared,
                       scratch)
            for i in range(i0, i1):
                lo = max(j0, i + 1)
                if lo >= j1:
                    continue
                base = starts[i] - (i + 1)
                out[base + lo:base + j1] = D[i - i0, lo - j0:j1 - j0]
    return out


def matrix_sparse(P, Q, inv, squared, out):
    """Column-scatter kernel: work scales with ``sum_c m_c^2``, not ``n^2 k``.

    Accumulates the intersection terms column by column, then adds
    ``ln 2`` times the one-sided mass, recovered as
    ``rowsum - intersection mass``.

    The row sums are accumulated in this same column loop rather than with
    ``P.sum(1)``: identical rows must give an intersection mass that
    cancels the row sums *exactly*, and ``P.sum`` reduces in a tree rather
    than left to right, so it would not reproduce the order used here.
    """
    n, k = P.shape
    Qm = P if Q is None else Q
    m = Qm.shape[0]
    PL = xlogx(P)
    QL = PL if Q is None else xlogx(Qm)

    acc = np.zeros((n, m), dtype=P.dtype)      # intersection terms
    hit = np.zeros((n, m), dtype=P.dtype)      # mass inside the intersection
    sp = np.zeros(n, dtype=P.dtype)
    sq = sp if Q is None else np.zeros(m, dtype=P.dtype)
    for c in range(k):
        a = P[:, c]
        b = Qm[:, c]
        sp += a
        if Q is not None:
            sq += b
        ri = np.nonzero(a)[0]
        if ri.size == 0:
            continue
        rj = ri if Q is None else np.nonzero(b)[0]
        if rj.size == 0:
            continue
        av, bv = a[ri], b[rj]
        M = 0.5 * (av[:, None] + bv[None, :])
        term = PL[ri, c][:, None] + QL[rj, c][None, :] - 2.0 * M * np.log(M)
        idx = np.ix_(ri, rj)
        acc[idx] += term
        hit[idx] += av[:, None] + bv[None, :]

    D = acc
    D += LN2 * (sp[:, None] + sq[None, :] - hit)
    D *= 0.5 * inv
    np.clip(D, 0.0, None, out=D)
    if not squared:
        np.sqrt(D, out=D)
    if Q is None:
        np.fill_diagonal(D, 0.0)
    if out is not None:
        out[...] = D
        return out
    return D
