"""Public API: pairwise Jensen-Shannon distance / divergence over row-stacks."""

from __future__ import annotations

import numpy as np

from . import _f32, _numba, _numpy, _svml
from ._common import LN2, check_base, density, prepare, xlogx

__all__ = [
    "jsd", "jsd_matrix", "jsd_pdist", "jsd_pairwise", "to_similarity",
    "available_backends", "resolve_backend", "SPARSE_DENSITY_MAX",
    "SVML_DENSITY_MAX",
]

#: Density at or below which ``auto`` prefers the intersection merge over the
#: **libm** dense kernel: above it, walking a support intersection costs more
#: per surviving coordinate than a straight-line pass over ``k``. With the
#: sparse kernel's two-accumulator complement and empty-intersection mask in
#: place, the measured turning point is near 0.59 at k=40 and near 0.63 at
#: k=128. Its location depends on the host as well as on the data, so assign
#: to this to retune for your own input. ``bench/portable.py --crossover``
#: reports what this constant costs on a given machine, measured against that
#: machine's own per-density optimum: near the turning point the two kernels
#: are within noise of each other, so a bound on the regret travels between
#: hosts where a located crossover does not. ``bench/distributions.py`` shows
#: why support structure matters as much as density here.
SPARSE_DENSITY_MAX = 0.60

#: The same turning point when the dense kernel is the SVML one, which is
#: nearly density-independent: it does the full ``k`` work whatever the input,
#: so it overtakes the merge much earlier than the libm kernel does. Reusing
#: ``SPARSE_DENSITY_MAX`` here would keep dispatching to the merge well past
#: the density at which SVML is faster on thresholded-membership data.
#:
#: Measured turning point, n=800, min of 5 interleaved repetitions:
#:
#:     support structure        k=25   k=64   k=128   k=256
#:     uniform random           0.34   0.27    0.27    0.22
#:     correlated (rho=0.9)     0.42      -       -       -
#:
#: 0.22 is the value for the least realistic corner of that table, and the
#: threshold errs high because the regret is asymmetric: the merge's advantage
#: grows as density falls, while SVML's is nearly flat, so being under the true
#: crossover costs more than being over it. At 0.30 the worst regret over the
#: grid above is 1.26x (k=256, d=0.30, uniform); at 0.22 it is 2.6x on the
#: 931x25 membership fixture, whose density is 0.237.
#:
#: The crossover moves with the *skip rate*, not with density alone: the
#: empty-intersection mask fires on 58.9 % of pairs on that fixture and on
#: 15.3 % of uniform-random rows at the same density, so no single scalar is
#: right everywhere. This one is a bounded-regret compromise, not a constant of
#: nature, and it needs re-tuning in any change that enables the SVML kernel
#: for a new configuration.
SVML_DENSITY_MAX = 0.30

_BACKENDS = ("auto", "numpy", "numpy-sparse", "numba", "numba-sparse",
             "numba-svml", "numba-f32")


def available_backends() -> tuple[str, ...]:
    """Backends usable in this interpreter, in the order ``auto`` prefers.

    ``numba-svml`` appears only when Intel's ``libsvml`` is loadable *and* the
    host has AVX-512; ``_svml.why_unavailable()`` says which is missing.
    ``numba-f32``, which computes in float32 and which ``auto`` never picks,
    comes last.
    """
    if _numba.available():
        base = ("numba-sparse", "numba", "numpy-sparse", "numpy")
        if _svml.available():
            base = ("numba-svml",) + base
        return base + ("numba-f32",)
    return ("numpy-sparse", "numpy")


def resolve_backend(backend: str, P: np.ndarray, Q: np.ndarray | None = None) -> str:
    """Map ``backend`` (possibly ``"auto"``) onto a concrete kernel name."""
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {_BACKENDS}")
    if backend.startswith("numba") and not _numba.available():
        raise RuntimeError(
            f"backend={backend!r} requires numba (pip install 'fastjsd[numba]')"
        )
    if backend == "numba-svml" and not _svml.available():
        raise RuntimeError(
            f"backend='numba-svml' unavailable: {_svml.why_unavailable()}"
        )
    if backend != "auto":
        return backend
    d = density(P) if Q is None else max(density(P), density(Q))
    if not _numba.available():
        return "numpy-sparse" if d <= SPARSE_DENSITY_MAX else "numpy"
    # The dense kernel's speed decides where the merge stops paying, so the
    # threshold has to follow whichever dense kernel is actually available.
    if _svml.available():
        return "numba-sparse" if d <= SVML_DENSITY_MAX else "numba-svml"
    return "numba-sparse" if d <= SPARSE_DENSITY_MAX else "numba"


_WC_KERNELS = ("numba", "numba-sparse")


def _resolve_well_conditioned(flag, kernel):
    """Decide the cancellation-free summand, defaulting per kernel.

    ``None`` (the default) enables it wherever it is free. Measured on the
    membership fixtures, over the pairs whose distance is small but nonzero --
    the ill-conditioned band, and the one that occurs in document data:

        kernel   worst err vs 50-digit truth   cost
        sparse   1.4e-13  ->  1.7e-18          none
        dense    2.4e-14  ->  3.5e-18          1.6-1.8x

    Four to five orders of magnitude of accuracy, for nothing, on the sparse
    path -- matched coordinates are rare there, so the extra branch almost never
    fires. On the dense path the same branch fires on every coordinate, which is
    a trade the caller should make rather than inherit. Hence: on by default for
    the merge, off by default for the dense loop, and an explicit
    ``True``/``False`` overrides either way.

    ``None`` never raises -- a default that failed on a backend the caller
    picked deliberately would cost more than the accuracy it buys.

    ``numba-f32`` has no switch: in float32 the cancellation-free series is
    what keeps close coordinates accurate at all, so it is always on.
    """
    if kernel == "numba-f32":
        if flag is False:
            raise ValueError(
                "backend='numba-f32' always uses the cancellation-free series "
                "on close coordinates; well_conditioned=False is not "
                "available there. Leave well_conditioned unset."
            )
        return True
    if flag is None:
        return kernel == "numba-sparse"
    if flag and kernel not in _WC_KERNELS:
        raise ValueError(
            f"well_conditioned=True is implemented for the 'numba' and "
            f"'numba-sparse' kernels, not {kernel!r}. Pass backend='numba' "
            f"(dense) or 'numba-sparse', or leave well_conditioned unset."
        )
    return bool(flag)


def _drop_empty_columns(P, Q):
    """Columns that are zero in every row cannot affect any pair. Dropping
    them is exact and shrinks the inner loop -- often substantially on
    one-hot-ish or vocabulary-shaped data."""
    keep = P.any(axis=0)
    if Q is not None:
        keep |= Q.any(axis=0)
    if keep.all() or not keep.any():
        return P, Q
    P = np.ascontiguousarray(P[:, keep])
    if Q is not None:
        Q = np.ascontiguousarray(Q[:, keep])
    return P, Q


def _working_dtype(backend, dtype):
    """``numba-f32`` computes in float32, so it fixes the working dtype."""
    if backend != "numba-f32":
        return dtype
    if dtype is not None and np.dtype(dtype) != np.float32:
        raise ValueError(
            f"backend='numba-f32' computes in float32, got dtype="
            f"{np.dtype(dtype)}. Leave dtype unset or pass np.float32."
        )
    return np.float32


def _prep_pair(P, Q, dtype, normalize, check):
    P = prepare(P, name="P", dtype=dtype, normalize=normalize, check=check)
    if Q is None:
        return P, None
    Q = prepare(Q, name="Q", dtype=P.dtype, normalize=normalize, check=check)
    if Q.shape[1] != P.shape[1]:
        raise ValueError(
            f"P and Q must have the same number of columns, got "
            f"{P.shape[1]} and {Q.shape[1]}"
        )
    return P, Q


def jsd_matrix(
    P,
    Q=None,
    *,
    base: float = 2.0,
    squared: bool = False,
    normalize: bool = True,
    check: bool = True,
    dtype=None,
    backend: str = "auto",
    out: np.ndarray | None = None,
    threads: int | None = None,
    well_conditioned: bool | None = None,
):
    """Pairwise Jensen-Shannon distance between the rows of ``P`` (and ``Q``).

    Parameters
    ----------
    P : array_like, shape (n, k) or (k,)
        Rows are distributions over ``k`` outcomes. Rows are normalized to
        sum to 1 unless ``normalize=False``.
    Q : array_like, shape (m, k), optional
        When given, the result is the ``(n, m)`` cross matrix. When omitted,
        the symmetric ``(n, n)`` self matrix (zero diagonal) is returned and
        only the upper triangle is computed.
    base : float, default 2.0
        Logarithm base. ``2`` bounds the distance in ``[0, 1]`` (so
        ``1 - d`` is a similarity); SciPy's ``jensenshannon`` defaults to
        ``e`` instead, which scales every value by ``sqrt(ln 2)``.
    squared : bool, default False
        Return the divergence ``JS`` rather than the distance ``sqrt(JS)``.
        The divergence is not a metric; the distance is.
    normalize : bool, default True
        Divide each row by its sum. With ``False`` the rows are checked
        (unless ``check=False``) and used as given -- the fast identity is
        only valid for normalized rows.
    check : bool, default True
        Validate finiteness, non-negativity and (when ``normalize=False``)
        row sums. Turn off only in a hot loop over already-vetted input.
    dtype : numpy dtype, optional
        Working and output dtype. ``float32`` roughly halves memory and time
        for a worst-case error near ``1e-4`` on the distance; ``float64``
        (the default for integer or float64 input) agrees with SciPy to
        ~1e-13. ``backend="numba-f32"`` implies ``float32``.
    backend : str, default "auto"
        One of ``"auto"``, ``"numpy"``, ``"numpy-sparse"``, ``"numba"``,
        ``"numba-sparse"``, ``"numba-svml"``, ``"numba-f32"``. ``auto`` picks
        the sparse Numba kernel below the applicable nonzero fraction
        (``SVML_DENSITY_MAX`` where the SVML kernel is available,
        ``SPARSE_DENSITY_MAX`` otherwise), the matching dense kernel above
        it, and the NumPy kernels when Numba is absent. ``numba-f32`` is
        opt-in: a dense kernel that computes in float32, with exact 0.0 and
        1.0 where the others are exact. On x86-64 it is several times faster
        than the other dense kernels for an error near ``1e-7`` on the
        distance; on ARM, where LLVM leaves its loop scalar, it is no faster
        and less accurate.
    out : ndarray, optional
        Preallocated output of the right shape and dtype, C-contiguous.
    threads : int, optional
        Numba thread count for this call (restored afterwards).
    well_conditioned : bool, optional
        Use the cancellation-free summand in the Numba kernels, which is
        more accurate on pairs whose distance is small but nonzero. ``None``
        (the default) enables it where it is free: on for
        ``"numba-sparse"``, off for ``"numba"``.

    Returns
    -------
    ndarray, shape (n, n) or (n, m)

    Examples
    --------
    >>> import numpy as np
    >>> from fastjsd import jsd_matrix
    >>> P = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    >>> np.round(jsd_matrix(P), 6)
    array([[0.      , 1.      , 0.557923],
           [1.      , 0.      , 0.557923],
           [0.557923, 0.557923, 0.      ]])
    """
    inv = 1.0 / np.log(check_base(base))
    P, Q = _prep_pair(P, Q, _working_dtype(backend, dtype), normalize, check)
    n, m = P.shape[0], (P.shape[0] if Q is None else Q.shape[0])
    if out is not None:
        _check_out(out, (n, m), P.dtype)
    if n == 0 or m == 0:
        return np.empty((n, m), dtype=P.dtype) if out is None else out
    P, Q = _drop_empty_columns(P, Q)

    kernel = resolve_backend(backend, P, Q)
    wc = _resolve_well_conditioned(well_conditioned, kernel)
    with _threads(threads):
        if kernel == "numpy":
            return _numpy.matrix(P, Q, inv, squared, out)
        if kernel == "numpy-sparse":
            return _numpy.matrix_sparse(P, Q, inv, squared, out)
        if kernel == "numba":
            return _numba.matrix(P, Q, inv, squared, out, wc)
        if kernel == "numba-svml":
            return _svml.matrix(P, Q, inv, squared, out)
        if kernel == "numba-f32":
            return _f32.matrix(P, Q, inv, squared, out)
        return _numba.matrix_sparse(P, Q, inv, squared, out, wc)


def jsd_pdist(
    P,
    *,
    base: float = 2.0,
    squared: bool = False,
    normalize: bool = True,
    check: bool = True,
    dtype=None,
    backend: str = "auto",
    out: np.ndarray | None = None,
    threads: int | None = None,
    well_conditioned: bool | None = None,
):
    """Condensed pairwise distances, in SciPy ``pdist`` order.

    Same arguments as :func:`jsd_matrix` minus ``Q``. Returns an
    ``n(n-1)/2`` vector -- half the memory of the square form, which is the
    difference between fitting and not fitting at large ``n``. Feed it to
    ``scipy.spatial.distance.squareform`` if you need the square form.

    ``backend="numpy-sparse"`` has no condensed form -- its column-scatter
    accumulator is inherently square -- so it is served by the tiled NumPy
    kernel here.
    """
    inv = 1.0 / np.log(check_base(base))
    P = prepare(P, name="P", dtype=_working_dtype(backend, dtype),
                normalize=normalize, check=check)
    n = P.shape[0]
    size = n * (n - 1) // 2
    if out is not None:
        _check_out(out, (size,), P.dtype)
    if n < 2:
        return np.empty(0, dtype=P.dtype) if out is None else out
    P, _ = _drop_empty_columns(P, None)

    kernel = resolve_backend(backend, P)
    wc = _resolve_well_conditioned(well_conditioned, kernel)
    with _threads(threads):
        if kernel in ("numpy", "numpy-sparse"):
            return _numpy.pdist(P, inv, squared, out)
        if kernel == "numba":
            return _numba.pdist(P, inv, squared, out, wc)
        if kernel == "numba-svml":
            return _svml.pdist(P, inv, squared, out)
        if kernel == "numba-f32":
            return _f32.pdist(P, inv, squared, out)
        return _numba.pdist_sparse(P, inv, squared, out, wc)


def jsd_pairwise(
    P,
    Q,
    *,
    base: float = 2.0,
    squared: bool = False,
    normalize: bool = True,
    check: bool = True,
    dtype=None,
):
    """Row-aligned distances: ``out[i] = JS(P[i], Q[i])``.

    ``P`` and ``Q`` broadcast against each other on the row axis, so a
    single distribution can be compared against a whole stack.

    The arithmetic is the NumPy kernel's (``_numpy._block``): the general
    summand on matched coordinates only, ``PL + QL`` grouped before the
    mixture term, and ``ln 2`` times each row's one-sided mass, totalled left
    to right. That is what makes disjoint supports exactly 1.0 and identical
    rows exactly 0.0 here as in ``jsd_matrix`` (``docs/MATH.md`` G4, G6), and
    ``jsd(p, q)`` bitwise equal to ``jsd(q, p)``.
    """
    inv = 1.0 / np.log(check_base(base))
    P, Q = _prep_pair(P, Q, dtype, normalize, check)
    if P.shape[0] != Q.shape[0]:
        if P.shape[0] == 1:
            P = np.broadcast_to(P, Q.shape)
        elif Q.shape[0] == 1:
            Q = np.broadcast_to(Q, P.shape)
        else:
            raise ValueError(
                f"P and Q must have the same number of rows or one must have "
                f"a single row, got {P.shape[0]} and {Q.shape[0]}"
            )
    bp = P > 0.0
    bq = Q > 0.0
    both = bp & bq
    M = 0.5 * (P + Q)
    L = np.zeros_like(M)
    np.log(M, out=L, where=both)
    L *= M
    L *= -2.0
    L += xlogx(P) + xlogx(Q)
    L *= both
    D = L.sum(axis=1)
    zero = P.dtype.type(0.0)
    one = np.cumsum(np.where(bq, zero, P), axis=1)[:, -1]
    one += np.cumsum(np.where(bp, zero, Q), axis=1)[:, -1]
    D += LN2 * one
    D *= 0.5 * inv
    np.clip(D, 0.0, None, out=D)
    return D if squared else np.sqrt(D, out=D)


def jsd(p, q, *, base: float = 2.0, squared: bool = False, **kwargs) -> float:
    """Jensen-Shannon distance between two single distributions.

    Note the ``base`` default differs from
    ``scipy.spatial.distance.jensenshannon`` (which uses ``e``); pass
    ``base=np.e`` for numerically identical results.
    """
    return float(jsd_pairwise(np.atleast_2d(p), np.atleast_2d(q), base=base,
                              squared=squared, **kwargs)[0])


def to_similarity(D, *, base: float = 2.0):
    """``1 - D``, the similarity induced by the base-2 distance.

    Only meaningful for ``base=2``, where the distance is bounded by 1; any
    other base is rejected rather than silently producing negatives.
    """
    if abs(check_base(base) - 2.0) > 1e-12:
        raise ValueError("to_similarity is only defined for base=2, where the "
                         "distance is bounded by 1")
    return 1.0 - np.asarray(D)


# --------------------------------------------------------------------------

def _check_out(out, shape, dtype):
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a numpy array")
    if out.shape != shape:
        raise ValueError(f"out has shape {out.shape}, expected {shape}")
    if out.dtype != dtype:
        raise ValueError(f"out has dtype {out.dtype}, expected {dtype}")
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")


class _threads:
    """Scope a Numba thread-count override."""

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        if self.n is not None:
            _numba.set_num_threads(self.n)

    def __exit__(self, *exc):
        if self.n is not None:
            _numba.set_num_threads(None)
        return False
