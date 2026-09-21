"""Shared validation, normalization and the algebraic identity the kernels use.

Every kernel here computes the *same* quantity through the same
rearrangement of the Jensen-Shannon divergence. Two properties of that
rearrangement are what make the fast kernels both fast and accurate
(``docs/MATH.md`` derives them).

For row-stochastic ``p, q`` and ``m = (p + q) / 2``, in nats:

    JS(p, q) = (1/2) * sum_c [ p_c ln p_c + q_c ln q_c - 2 m_c ln m_c ]

with ``0 ln 0 = 0``. Then:

1. **Every summand is non-negative** -- it is the convexity gap of
   ``x ln x`` at ``p_c, q_c``. So the total never cancels: the result is
   accurate all the way down to ``JS = 0``, where the terms are *exactly*
   zero for identical rows rather than ~1e-16 of float dust that ``sqrt``
   would amplify to ~1e-8.

2. **A summand where only one side is nonzero equals** ``p_c ln 2``
   exactly (substitute ``q_c = 0``, ``m_c = p_c / 2``). So a pair needs
   logarithms only on its *support intersection*; everything else is
   ``ln 2`` times the mass outside it, which the merge accumulates for
   free. This is what the sparse kernel exploits, and it is exact, not an
   approximation.

Precomputing ``p ln p`` once per entry (``xlogx``) leaves exactly one
logarithm per pair-coordinate in the inner loop.

The distance is ``sqrt(JS / ln(base))``; ``base=2`` puts it in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

LN2 = 0.6931471805599453

__all__ = ["LN2", "prepare", "xlogx", "density", "check_base"]


def check_base(base: float) -> float:
    base = float(base)
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError(f"base must be a finite float > 1, got {base!r}")
    return base


def prepare(
    P,
    *,
    name: str = "P",
    dtype=None,
    normalize: bool = True,
    check: bool = True,
    copy_if_normalizing: bool = True,
):
    """Validate and row-normalize a stack of distributions.

    Returns a C-contiguous ``(n, k)`` array whose rows sum to 1.

    ``normalize=False`` skips the division and trusts the caller; the rows
    are still checked (unless ``check=False``) because the identity above is
    only valid for normalized rows -- an unnormalized row does not give a
    slightly-wrong answer, it gives a meaningless one.
    """
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[None, :]
    if P.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {P.ndim}-D")
    if dtype is None:
        dtype = np.float64 if P.dtype.kind not in "fc" else P.dtype
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError(f"dtype must be a floating type, got {dtype}")
    P = np.ascontiguousarray(P, dtype=dtype)

    if P.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")

    # The fused path below validates inside the same pass it normalizes in, so
    # these whole-array scans only run when it is not taken.
    if check and not normalize:
        _validate(P, name)

    if normalize and P.shape[0]:
        # The copy the caller needs (so its input is not mutated) is done by
        # the kernel, in the pass that is already reading the data.
        dst = P if (not copy_if_normalizing and P.flags.owndata) else None
        out = _fused_normalize(P, dst, name, check)
        if out is not None:
            return out

    sums = P.sum(axis=1)
    if normalize:
        if check and P.shape[0] and float(sums.min()) <= 0.0:
            bad = int(np.argmin(sums))
            raise ValueError(f"{name} row {bad} sums to {sums[bad]!r}; cannot "
                             "normalize an all-zero (or negative) row")
        P /= sums[:, None]
        snap_row_sums(P)
    elif check and P.shape[0]:
        tol = 1e-6 if dtype == np.float32 else 1e-9
        off = float(np.abs(sums - 1.0).max())
        if off > tol:
            raise ValueError(
                f"normalize=False but {name} rows are not normalized "
                f"(max |rowsum - 1| = {off:.3g} > {tol:g}). Pass "
                "normalize=True, or fix the input."
            )
    return np.ascontiguousarray(P, dtype=dtype)


def _validate(P, name):
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if P.size and P.min() < 0.0:
        raise ValueError(f"{name} contains negative values; rows must be "
                         "(sub-)probability vectors")


def _fused_normalize(P, dst, name, check):
    """Validate, normalize and snap in one compiled pass, or ``None``.

    Five or six whole-array passes collapse into one traversal per row. Those
    passes are pure memory traffic over the whole input, so they are a real
    part of the total runtime rather than a negligible preamble to the
    distance kernel. ``dst=None`` means allocate, which is the copy the caller
    would otherwise have made.

    Returns the prepared array, or ``None`` if there is no compiled path.
    """
    from . import _numba
    if not _numba.available() or P.dtype != np.float64:
        return None
    if dst is None:
        dst = np.empty_like(P)
    status = _numba.prepare_rows_like_kernels(P, dst)
    if status.size:
        bad = int(np.argmax(status))
        code = int(status[bad])
        if code == 1:
            raise ValueError(f"{name} contains NaN or infinite values")
        if code == 2:
            raise ValueError(f"{name} contains negative values; rows must be "
                             "(sub-)probability vectors")
        if code == 3:
            raise ValueError(f"{name} row {bad} sums to {P[bad].sum()!r}; "
                             "cannot normalize an all-zero (or negative) row")
    return dst


def snap_row_sums(P: np.ndarray) -> np.ndarray:
    """Make each row sum to exactly 1.0 the way the kernels add it up.

    Dispatches to the compiled row-at-a-time kernel when Numba is present.
    The NumPy version below issues up to six ``cumsum`` calls over the whole
    ``(n, k)`` array; that is pure memory traffic over the entire input, and
    on sparse data, where the distance kernel itself is cheap, it can be a
    large fraction of the total runtime.
    """
    if P.shape[0] and P.shape[1]:
        from . import _numba
        if _numba.available():
            return _numba.snap_rows_like_kernels(P)
    return _snap_row_sums_numpy(P)


def _snap_row_sums_numpy(P: np.ndarray) -> np.ndarray:
    """Make each row sum to exactly 1.0 *the way the kernels add it up*.

    Dividing by the row sum does not give rows that sum to one. It gives rows
    that sum to one to within a few ULP, and the shortfall is what stops
    disjoint supports returning exactly 1.0: with an empty intersection the
    whole distance is ``sqrt(0.5 (S_i + S_j))``, so it is exact only when
    ``S_i + S_j`` is exactly 2. After a plain ``P / P.sum(1)`` the fraction of
    rows whose sequential sum is exactly 1.0 falls as ``k`` grows -- 46 % at
    k=25, 6 % at k=1024 -- because more terms means more rounding.

    Two summation orders are in play and they disagree. ``ndarray.sum`` does
    NOT add left to right -- it reduces in a divide-and-conquer tree, which is
    an accuracy optimization -- while the kernels accumulate strictly left to
    right. (NumPy calls that tree reduction "pairwise summation", which is
    unrelated to this package's sense of *pairwise*: "over all pairs of rows",
    as in ``jsd_pairwise`` and ``pdist``.) ``np.sum`` and a left-to-right loop
    disagree on 1860 of 2000 random 1000-element rows. This function targets
    the kernels' order, which is the one that decides the answer, and uses
    ``cumsum`` because that *is* a left-to-right accumulation -- exactly, on
    2000 of 2000.

    Three stages, cheapest and least invasive first. Each stage re-checks the
    row totals and only touches the rows the previous one left short:

    1. **Divide by the sequential sum**, up to three times. This is not a
       correction so much as the normalization the kernels needed in the first
       place, and it moves every entry by under an ULP. It settles about 75 %
       of rows.
    2. **Nudge the row's largest entry** by the residual, up to three times.
       Largest, because the induced error scales with the *relative* change
       and this minimizes it. Settles most of the rest.
    3. **Set the last non-zero entry to ``1 - s``**, where ``s`` is the
       sequential sum of everything before it. Every column after ``L`` is
       zero and adds nothing, so the row's total becomes exactly
       ``s + (1 - s)``; by Sterbenz's lemma ``1 - s`` is exact for ``s`` in
       ``[0.5, 2]`` and the final addition then rounds the exact real 1.0.
       This one always works, which is why it is the terminator -- but the
       last non-zero can be tiny, and forcing a correction onto a tiny entry
       is a large *relative* change. Reaching it for only the 7-9 % of rows
       the first two stages miss is the difference between a worst-case
       relative perturbation of 3.0e-10 and one of 6.3e-05; stage 3 on its
       own would perturb every corrected row by the larger amount.

    Identical rows get identical corrections at every stage, so duplicates
    stay bitwise equal and the exact-zero guarantee is untouched.

    In place. Returns ``P`` for convenience.
    """
    n, k = P.shape
    if n == 0 or k == 0:
        return P
    rows = np.arange(n)

    def totals():
        return np.cumsum(P, axis=1)[:, -1]

    for _ in range(3):
        s = totals()
        bad = s != 1.0
        if not bad.any():
            return P
        P[bad] /= s[bad, None]

    big = np.argmax(P, axis=1)
    for _ in range(3):
        s = totals()
        bad = s != 1.0
        if not bad.any():
            return P
        P[rows[bad], big[bad]] += (1.0 - s[bad]).astype(P.dtype, copy=False)

    left = np.flatnonzero(totals() != 1.0)
    if left.size:
        cs = np.cumsum(P, axis=1)
        # `argmax` on the reversed mask finds the last non-zero. Rows that are
        # all zero cannot reach here -- `prepare` rejects them before
        # normalizing.
        last = k - 1 - np.argmax((P > 0.0)[:, ::-1], axis=1)
        for r in left:
            col = last[r]
            before = cs[r, col - 1] if col > 0 else P.dtype.type(0.0)
            fixed = P.dtype.type(1.0) - before
            if fixed > 0.0 and before + fixed == 1.0:
                P[r, col] = fixed
    return P


def xlogx(P: np.ndarray) -> np.ndarray:
    """Elementwise ``x ln x`` with ``0 ln 0 = 0``.

    Precomputed once per entry so the pair loop needs one logarithm per
    coordinate instead of three. Costs one array of ``P``'s size; the
    ``where=`` form avoids both a ``log(0)`` warning and a second temporary.
    """
    nz = P > 0
    L = np.zeros_like(P)
    np.log(P, out=L, where=nz)
    L *= P
    return L


def density(P: np.ndarray) -> float:
    """Fraction of nonzero entries -- the sparse kernel's selection signal."""
    if P.size == 0:
        return 1.0
    return float(np.count_nonzero(P)) / P.size
