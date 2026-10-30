"""Opt-in backend: a dense kernel that computes in float32.

``backend="numba-f32"`` trades the last digits for throughput. The other
kernels compute in float64 whatever the input dtype; this one keeps float32
data in float32 through the hot loop, which doubles the SIMD width, and gives
the logarithm an inlined polynomial instead of a libm call, which lets LLVM
vectorize the whole row. It is plain Numba -- no compiler, no runtime
library, nothing beyond what ``backend="numba"`` already needs -- so it runs
wherever Numba does, and LLVM picks the vector width for the host. On AArch64
LLVM's cost model declines to vectorize the loop at all: there the kernel is
no faster than ``backend="numba"`` and its float32 sum is sequential
(``docs/MATH.md``, "The float32 kernel").

Per coordinate, in float32 and branch-free once LLVM if-converts it:

- **Both sides positive and close** (``|u| <= 1/2``, ``u = (a-b)/(a+b)``):
  the cancellation-free series ``m u^2 phi(u^2)`` of ``docs/MATH.md``
  ("Removing it: the symbolic cancellation"), through ``u^18``. No
  logarithm and no cancellation however close ``a`` and ``b`` are, and
  ``u = 0`` gives exactly 0.0, so identical rows are exactly 0.0 (G4).
- **Both positive, further apart**: the far formula
  ``PL_i + PL_j - (a + b) log m``, whose cancellation factor is at most
  ``~15 |ln m|`` past the switch (it is ``1/u^2``-sized below it, which is
  what the series is for).
- **One side positive**: ``a ln 2``. Neither: 0.

The log is ``log(1 + t) = t - t^2/2 + t^3 P(t)`` after reducing ``x`` to
``m 2^e``, with a degree-7 ``P`` fitted for this interval: 0.14 ULP of
truncation, and at most 1.06 ULP in all, measured over every float32 in
``[0.5, 2)`` and 16M samples from the subnormals to 1e38. It needs no
division, which is what pays for the series' one. ``PL`` is built with the
same log.

The float32 sum of a pair with no shared coordinate misses ``2 ln 2`` by
rounding, so a pair whose float32 sum lands within ``_high(k)`` of that
maximum -- every disjoint pair, and the nearly disjoint ones -- is
recomputed from its float32 inputs in float64 by ``_exact``: the libm
kernel's well-conditioned summand with one one-sided accumulator per row,
clamped to the maximum divergence. That keeps G6 (disjoint supports give
exactly 1.0 in base 2) and the ``[0, 1]`` bound.

Accuracy is float32-grade, not float64-grade: against float64 arithmetic on
the same float32 rows, on x86-64, the error on the distance is ~1e-7
absolute (at most 1.6e-7 over the regimes ``bench/f32.py`` sweeps, ``k`` up
to 4096), and ~1e-7 *relative* on close pairs -- 1.1e-7 at worst against
50-digit arithmetic, down to distances of 4e-7. (There SciPy's float64 result
is itself ~1e-4 off, its grouping cancelling as ``2/u``; and the float64
kernels fed float32 input reach ~1e-4, because they round ``a + b`` to
float32 first.) Each pair is one float32 sum over ``k`` terms, so the error
grows with ``k`` -- 3.4e-6 at ``k = 262144`` -- and faster where the sum is
sequential: on AArch64, up to 2e-6 over the same regimes. Results agree
across machines to float32 rounding, not bitwise, because the order of the
float32 accumulation follows the host's vector width; on one machine they do
not depend on the thread count.

Two requirements a reimplementation has to meet.

1. ``_exact`` must not be inlined into the fast kernels. Numba applies
   ``fastmath`` flags to every instruction of the function it compiles,
   inlined code included; ``reassoc`` would reorder the per-row one-sided
   sums G6 depends on. Called as a separate function, it keeps strict IEEE
   arithmetic.
2. The bit manipulation in ``_log`` must stay in 32-bit integers. Numba
   widens int32 arithmetic to int64, and int64 to float32 conversion does not
   vectorize on AVX2, so ``_mantissa`` and ``_exponent`` are intrinsics.
"""

from __future__ import annotations

import numpy as np

from ._common import LN2

__all__ = ["available", "why_unavailable", "matrix", "pdist", "xlogx"]

try:  # pragma: no cover - whichever branch this environment has
    from llvmlite import ir
    from numba import njit, prange, types
    from numba.extending import intrinsic

    from . import _numba
    _HAVE = _numba.available()
except Exception:  # pragma: no cover
    _HAVE = False


def available() -> bool:
    return _HAVE


def why_unavailable() -> str:
    return "available" if _HAVE else "numba is not installed"


def _high(k: int) -> float:
    """Float32 symmetric sums ``2 JS`` at or above this go to ``_exact``.

    A disjoint pair's float32 sum is ``k`` rounded terms ``a ln 2``, and a sum
    of ``k`` non-negative terms lands within a relative ``(k + 1) u`` of the
    exact ``2 ln 2`` in any order (``u = 2^-24``; the float32 ``ln 2`` is
    0.05 u off). Twice that margin below the maximum catches every one. Pairs
    above it are nearly disjoint, which on dense data is rare: at ``k = 1024``
    the band starts at a base-2 distance of 0.99994.
    """
    margin = min(0.5, 2.0 * (k + 2) * 2.0 ** -24)
    return 2.0 * LN2 * (1.0 - margin)


if _HAVE:
    _FM = {"reassoc", "contract"}
    _JIT = dict(parallel=True, cache=True, nogil=True, fastmath=_FM)

    _F = np.float32
    _ZERO, _ONE, _HALF = _F(0.0), _F(1.0), _F(0.5)
    _SQRT2 = _F(1.4142135623730951)
    _LN2F = _F(LN2)
    _TWO23 = _F(8388608.0)
    _TINY = _F(1.1754943508222875e-38)            # smallest normal float32
    #: ``P`` in ``log(1 + t) = t - t^2/2 + t^3 P(t)``, ``t`` in
    #: ``[1/sqrt 2 - 1, sqrt 2 - 1]``: least squares on Chebyshev nodes,
    #: weighted to the relative error of the log, reweighted toward minimax.
    #: Max relative error of the polynomial 8.6e-9 (0.14 float32 ULP).
    _LOGP = tuple(_F(v) for v in (
        0.33333303671709857, -0.2500074775631892, 0.20003191397414105,
        -0.16627176545723224, 0.14164435373296178, -0.13088773564155048,
        0.1297439587159972, -0.080724353891928322))
    #: ``phi(z) = sum_n z^n / ((n + 1)(2n + 1))``, through ``z^8``: at the
    #: switch (``z = 1/4``) the first omitted term is 0.35 float32 ULP.
    _PHI = tuple(_F(1.0 / ((n + 1) * (2 * n + 1))) for n in range(9))
    _SWITCH = _F(0.5)

    @intrinsic
    def _mantissa(typingctx, x):
        """Significand of a normal float32 as a float32 in ``[1, 2)``."""
        if x != types.float32:
            return None

        def codegen(context, builder, sig, args):
            i32 = ir.IntType(32)
            bits = builder.bitcast(args[0], i32)
            m = builder.or_(builder.and_(bits, ir.Constant(i32, 0x007FFFFF)),
                            ir.Constant(i32, 0x3F800000))
            return builder.bitcast(m, ir.FloatType())
        return types.float32(types.float32), codegen

    @intrinsic
    def _exponent(typingctx, x):
        """Unbiased exponent of a normal float32, as a float32."""
        if x != types.float32:
            return None

        def codegen(context, builder, sig, args):
            i32 = ir.IntType(32)
            bits = builder.bitcast(args[0], i32)
            e = builder.and_(builder.lshr(bits, ir.Constant(i32, 23)),
                             ir.Constant(i32, 0xFF))
            return builder.sitofp(builder.sub(e, ir.Constant(i32, 127)),
                                  ir.FloatType())
        return types.float32(types.float32), codegen

    @njit(inline="always", fastmath=_FM)
    def _log(x):
        """``log(x)`` for float32 ``x > 0``, normal or subnormal.

        ``x = m 2^e`` with ``m`` in ``(1/sqrt 2, sqrt 2]``, then the
        polynomial in ``t = m - 1``. Called with ``x = 0`` it returns a finite
        value rather than ``-inf``; the callers discard it.
        """
        sub = x < _TINY
        xs = x * _TWO23 if sub else x
        m = _mantissa(xs)
        e = _exponent(xs)
        if sub:
            e = e - _F(23.0)
        if m > _SQRT2:
            m = m * _HALF
            e = e + _ONE
        t = m - _ONE
        p = _LOGP[7]
        p = p * t + _LOGP[6]
        p = p * t + _LOGP[5]
        p = p * t + _LOGP[4]
        p = p * t + _LOGP[3]
        p = p * t + _LOGP[2]
        p = p * t + _LOGP[1]
        p = p * t + _LOGP[0]
        t2 = t * t
        return (t + t2 * (t * p - _HALF)) + e * _LN2F

    @njit(inline="always", fastmath=_FM)
    def _term(a, al, b, bl):
        """The summand in float32; see the module docstring."""
        s = a + b
        d = a - b
        if a > _ZERO:
            if b > _ZERO:
                if abs(d) <= _SWITCH * s:
                    r = d / s
                    z = r * r
                    p = _PHI[8]
                    p = p * z + _PHI[7]
                    p = p * z + _PHI[6]
                    p = p * z + _PHI[5]
                    p = p * z + _PHI[4]
                    p = p * z + _PHI[3]
                    p = p * z + _PHI[2]
                    p = p * z + _PHI[1]
                    p = p * z + _PHI[0]
                    return _HALF * s * z * p
                return (al + bl) - s * _log(_HALF * s)
            return a * _LN2F
        if b > _ZERO:
            return b * _LN2F
        return _ZERO

    @njit(cache=True, nogil=True, fastmath=False)
    def _exact(P, i, Q, j, inv, squared, vmax):
        """One pair in float64 from the float32 rows: the libm kernel's
        well-conditioned summand (``_numba._dense_cross`` with ``wc``), one
        one-sided accumulator per row, clamped to the maximum divergence.

        Never inline this into the fast kernels -- see the module docstring.
        """
        acc = 0.0
        onea = 0.0
        oneb = 0.0
        for c in range(P.shape[1]):
            a = np.float64(P[i, c])
            b = np.float64(Q[j, c])
            if a > 0.0:
                if b > 0.0:
                    sm = a + b
                    m = 0.5 * sm
                    if abs(a - b) <= _numba._TAU * sm:
                        u2 = ((a - b) / sm) ** 2
                        acc += m * u2 * _numba._phi_over_u2(u2)
                    else:
                        acc += (a * np.log(a) + b * np.log(b)
                                - 2.0 * m * np.log(m))
                else:
                    onea += a
            elif b > 0.0:
                oneb += b
        v = 0.5 * (acc + LN2 * (onea + oneb)) * inv
        if v < 0.0:
            v = 0.0
        if v > vmax:
            v = vmax
        return v if squared else np.sqrt(v)

    @njit(inline="always", fastmath=_FM)
    def _pair(P, PL, i, Q, QL, j, inv, squared, vmax, high):
        ss = _ZERO
        for c in range(P.shape[1]):
            ss += _term(P[i, c], PL[i, c], Q[j, c], QL[j, c])
        if ss < high:                       # NaN fails it too: exact pass
            v = 0.5 * np.float64(ss) * inv
            if v < 0.0:
                v = 0.0
            return v if squared else np.sqrt(v)
        return _exact(P, i, Q, j, inv, squared, vmax)

    @njit(**_JIT)
    def _xlogx(P, out):
        n, k = P.shape
        for i in prange(n):
            for c in range(k):
                x = P[i, c]
                out[i, c] = x * _log(x) if x > _ZERO else _ZERO
        return out

    @njit(**_JIT)
    def _self(P, PL, inv, squared, out, offs, order, sym_write, vmax, high):
        n = P.shape[0]
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            for j in range(i + 1, n):
                v = _pair(P, PL, i, P, PL, j, inv, squared, vmax, high)
                out[oi + j] = v
                if sym_write:
                    out[offs[j] + i] = v

    @njit(**_JIT)
    def _cross(P, PL, Q, QL, inv, squared, out, vmax, high):
        n = P.shape[0]
        m_rows = Q.shape[0]
        for i in prange(n):
            oi = i * m_rows
            for j in range(m_rows):
                out[oi + j] = _pair(P, PL, i, Q, QL, j, inv, squared, vmax,
                                    high)

    @njit(**_JIT)
    def _cross_flat(P, PL, Q, QL, inv, squared, out, vmax, high):
        """``_cross`` over pairs, for skinny ``P``; see ``_numba._use_flat_cross``."""
        m_rows = Q.shape[0]
        for pair in prange(P.shape[0] * m_rows):
            i = pair // m_rows
            out[pair] = _pair(P, PL, i, Q, QL, pair - i * m_rows, inv,
                              squared, vmax, high)


def _require(P):
    if not _HAVE:
        raise RuntimeError("backend='numba-f32' requires numba "
                           "(pip install 'fastjsd[numba]')")
    if P.dtype != np.float32:                     # pragma: no cover - _api fixes it
        raise TypeError(f"numba-f32 computes in float32, got {P.dtype}")


def xlogx(P):
    """``x ln x`` in float32 with the kernel's own inlined log."""
    return _xlogx(P, np.empty(P.shape, dtype=np.float32))


def matrix(P, Q, inv, squared, out):
    """Signature-compatible with ``_numba.matrix``."""
    _require(P)
    n, k = P.shape
    vmax, high = LN2 * inv, _high(k)
    if Q is None:
        if out is None:
            out = np.empty((n, n), dtype=np.float32)
        _self(P, xlogx(P), inv, squared, out.reshape(-1),
              _numba._offsets(n, False), _numba._balanced_order(n), True,
              vmax, high)
        np.fill_diagonal(out, 0.0)
        return out
    m = Q.shape[0]
    if out is None:
        out = np.empty((n, m), dtype=np.float32)
    kernel = _cross_flat if _numba._use_flat_cross(n, m) else _cross
    kernel(P, xlogx(P), Q, xlogx(Q), inv, squared, out.reshape(-1), vmax, high)
    return out


def pdist(P, inv, squared, out):
    """Signature-compatible with ``_numba.pdist``."""
    _require(P)
    n, k = P.shape
    if out is None:
        out = np.empty(n * (n - 1) // 2, dtype=np.float32)
    _self(P, xlogx(P), inv, squared, out, _numba._offsets(n, True),
          _numba._balanced_order(n), False, LN2 * inv, _high(k))
    return out
