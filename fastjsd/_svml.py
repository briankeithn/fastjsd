"""Optional backend: the dense kernel with Intel SVML's 8-wide logarithm.

Numba normally reaches SVML through an LLVM pass that rewrites a vectorized
loop into ``__svml_*`` calls. That pass is compiled into llvmlite, and it is
absent from every build on PyPI, conda-forge and Anaconda's ``numba`` channel
alike -- ``llvmlite.binding.targets.has_svml()`` reports
``LLVMPY_HasSVMLSupport()``, a property of the binary, so no amount of
installing changes it.

The *library* is a separate matter. ``libsvml.so`` ships in
``intel-cmplr-lib-rt`` (conda-forge) and in the older ``icc_rt``, and the
missing pass can be bypassed by declaring the vector entry point and calling
it directly. That is what this module does: load the library with
``RTLD_GLOBAL`` so the JIT resolves the symbol, declare
``<8 x double> @__svml_log8(<8 x double>)``, and emit the call from an
intrinsic.

The gain comes from the width: a pair costs ``k`` scalar libm ``log`` calls
on the default kernel and ``k / 8`` vector calls here. What that is worth
depends on the host, so no figure is quoted: measure it. Where this backend is
available ``auto`` selects it on dense input, so the harnesses under ``bench/``
that time ``jsd_matrix`` are timing this kernel. ``bench/scale.py`` is the
exception -- it pins the libm kernel deliberately, so that a host with SVML and
a host without still produce comparable exponent fits. Accuracy is checked
rather than left to the timings: this kernel agrees with the libm one and with
the reference, which ``tests/test_svml.py`` asserts and ``bench/accuracy.py``
reports beside the other backends.

Why it is opt-in: it needs a non-Python runtime library and an AVX-512 host,
neither of which a pure NumPy+Numba package can assume. ``available()`` is
False unless both are present, ``auto`` then ignores it, and nothing else in
the package changes. Set ``FASTJSD_SVML=0`` to force it off, or
``FASTJSD_SVML_LIB=/path/to/libsvml.so`` to point at a specific copy.

Two requirements a reimplementation has to meet.

1. The vector load must be emitted with ``align 8``. LLVM infers 64-byte
   alignment from ``<8 x double>*`` and emits an aligned move; NumPy only
   guarantees element alignment, so that faults.
2. ``PL`` must be built with the SAME SVML log the kernel calls. SVML is up
   to 3 ULP from libm, and the duplicate-row guarantee depends on
   ``PL[i,c] + PL[j,c]`` cancelling ``2 m log m`` bitwise. Mixing the two
   logs leaves a residual of up to 3 ULP where the cancellation has to be
   exact -- the failure mode ``docs/MATH.md`` describes for a mismatched
   log, at three times the magnitude.

Unlike the libm kernel this one is branch-free over ``k``, which is what lets
the row vectorize at any density. That needs no special case for a one-sided
coordinate -- ``t(a,0) = a ln 2`` falls out of the general formula, so the
libm kernel's ``if a > 0 / else one += a`` branch is an optimization to skip a
logarithm, not a correctness requirement. Coordinates where *both* sides are
zero would give ``2 * 0 * log(0)`` = ``0 * -inf`` = NaN, so ``m`` is replaced
by 1 there: ``log(1)`` is 0 and ``PL`` is 0 on both sides, making the term
exactly zero without a branch.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys

import numpy as np

from ._common import LN2

__all__ = ["available", "matrix", "pdist", "xlogx_svml", "why_unavailable"]

_LIB_NAMES = ("libsvml.so", "libsvml.so.2", "svml_dispmd.dll", "libsvml.dylib")
_state: dict = {"checked": False, "ok": False, "why": "", "lib": None}


def _candidate_paths():
    override = os.environ.get("FASTJSD_SVML_LIB")
    if override:
        yield pathlib.Path(override)
        return
    roots = [pathlib.Path(sys.prefix), pathlib.Path(sys.base_prefix)]
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        roots.insert(0, pathlib.Path(conda))
    for root in roots:
        for name in _LIB_NAMES:
            yield root / "lib" / name
            yield root / "Library" / "bin" / name          # conda on Windows


def _load_library():
    """Return the loaded handle, or None. RTLD_GLOBAL so the JIT can resolve."""
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for path in _candidate_paths():
        if path.exists():
            try:
                return ctypes.CDLL(str(path), mode=mode), str(path)
            except OSError:
                continue
    if os.environ.get("FASTJSD_SVML_LIB"):
        return None, ""
    for name in _LIB_NAMES:                       # fall back to the loader path
        try:
            return ctypes.CDLL(name, mode=mode), name
        except OSError:
            continue
    return None, ""


def _has_avx512():
    """``__svml_log8`` takes a zmm register; without AVX-512 the ABI differs
    and the call would not merely be slow, it would be wrong."""
    try:
        from llvmlite import binding as lb
        return bool(lb.get_host_cpu_features().get("avx512f", False))
    except Exception:                             # pragma: no cover
        return False


def _probe():
    """Decide availability once. Verifies numerically rather than assuming."""
    if _state["checked"]:
        return _state["ok"]
    _state["checked"] = True

    if os.environ.get("FASTJSD_SVML", "1") in ("0", "false", "False"):
        _state["why"] = "disabled by FASTJSD_SVML=0"
        return False
    try:
        from . import _numba
        if not _numba.available():
            _state["why"] = "numba is not installed"
            return False
    except Exception as exc:                      # pragma: no cover
        _state["why"] = f"numba import failed: {exc}"
        return False
    if not _has_avx512():
        _state["why"] = "host has no AVX-512 (__svml_log8 needs a zmm register)"
        return False
    lib, where = _load_library()
    if lib is None:
        _state["why"] = ("libsvml.so not found -- conda install -c conda-forge "
                         "intel-cmplr-lib-rt, or set FASTJSD_SVML_LIB")
        return False
    _state["lib"] = lib

    try:
        _build()
        x = np.linspace(1e-6, 0.999, 64)
        got = _log_block(x, np.empty(64))
        if not np.allclose(got, np.log(x), rtol=0, atol=1e-12):
            _state["why"] = "__svml_log8 disagreed with libm on a smoke test"
            return False
    except Exception as exc:                      # pragma: no cover
        _state["why"] = f"could not compile the SVML intrinsic: {exc}"
        return False

    _state["ok"] = True
    _state["why"] = f"available ({where})"
    return True


def available() -> bool:
    """True when the SVML dense kernel can be used in this interpreter."""
    return _probe()


def why_unavailable() -> str:
    """Human-readable reason ``available()`` is False (or why it is True)."""
    _probe()
    return _state["why"]


# --------------------------------------------------------------------------
# built lazily: importing this module must not require numba or llvmlite
# --------------------------------------------------------------------------
_built: dict = {}


def _build():
    if _built:
        return _built
    from llvmlite import ir
    from numba import njit, prange, types
    from numba.extending import intrinsic

    V8 = ir.VectorType(ir.DoubleType(), 8)

    @intrinsic
    def _svml_log8(typingctx, src, dst, off):
        """``dst[off:off+8] = log(src[off:off+8])`` in one vector call."""
        if not (isinstance(src, types.Array) and isinstance(dst, types.Array)):
            return None

        def codegen(context, builder, sig, args):
            s = context.make_array(sig.args[0])(context, builder, args[0])
            d = context.make_array(sig.args[1])(context, builder, args[1])
            sp = builder.gep(s.data, [args[2]])
            dp = builder.gep(d.data, [args[2]])
            ld = builder.load(builder.bitcast(sp, V8.as_pointer()))
            ld.align = 8            # NOT the 64 LLVM infers -- see module docs
            mod = builder.module
            try:
                fn = mod.get_global("__svml_log8")
            except KeyError:
                fn = ir.Function(mod, ir.FunctionType(V8, [V8]), "__svml_log8")
            st = builder.store(builder.call(fn, [ld]),
                               builder.bitcast(dp, V8.as_pointer()))
            st.align = 8
            return context.get_dummy_value()

        return types.void(src, dst, types.intp), codegen

    _JIT = dict(parallel=True, nogil=True, fastmath=False)

    @njit(nogil=True)
    def log_block(x, out):
        """Vector log over a contiguous buffer; scalar tail. Used by the
        smoke test and by the ``x ln x`` precompute."""
        n = x.shape[0]
        nv = (n // 8) * 8
        for i in range(0, nv, 8):
            _svml_log8(x, out, i)
        for i in range(nv, n):
            out[i] = np.log(x[i])
        return out

    @njit(**_JIT)
    def xlogx(P, out):
        n, k = P.shape
        nv = (k // 8) * 8
        for i in prange(n):
            buf = np.empty(k)
            lg = np.empty(k)
            for c in range(k):
                a = np.float64(P[i, c])
                buf[c] = a if a > 0.0 else 1.0
            for c0 in range(0, nv, 8):
                _svml_log8(buf, lg, c0)
            for c in range(nv, k):
                lg[c] = np.log(buf[c])
            for c in range(k):
                a = np.float64(P[i, c])
                out[i, c] = a * lg[c] if a > 0.0 else 0.0
        return out

    @njit(**_JIT)
    def dense_self(P, PL, inv, squared, out, offs, order, sym_write):
        n, k = P.shape
        nv = (k // 8) * 8
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            mb = np.empty(k)
            lg = np.empty(k)
            for j in range(i + 1, n):
                for c in range(k):
                    m = 0.5 * (np.float64(P[i, c]) + np.float64(P[j, c]))
                    mb[c] = m if m > 0.0 else 1.0
                for c0 in range(0, nv, 8):
                    _svml_log8(mb, lg, c0)
                for c in range(nv, k):
                    lg[c] = np.log(mb[c])
                acc = 0.0
                # The vectorized log runs over every column, one-sided
                # ones included -- that is what keeps the row branch-free and
                # lets it vectorize at any density. The COMBINE loop is scalar
                # anyway, so recognising the one-sided case here is nearly
                # free, and it has to be recognised somewhere: evaluating
                # `a ln a - 2 (a/2) ln(a/2)` gives `a ln 2` in exact arithmetic
                # but not in float64, so folding those coordinates into the
                # general term would stop disjoint supports returning exactly
                # 1.0 on this backend.
                # One accumulator per row, for the same reason as the libm
                # kernel: a shared one associates the two rows' masses
                # together and misses `S_i + S_j`.
                onea = 0.0
                oneb = 0.0
                for c in range(k):
                    a = np.float64(P[i, c])
                    b = np.float64(P[j, c])
                    if a > 0.0:
                        if b > 0.0:
                            acc += PL[i, c] + PL[j, c] - 2.0 * mb[c] * lg[c]
                        else:
                            onea += a
                    elif b > 0.0:
                        oneb += b
                v = 0.5 * (acc + LN2 * (onea + oneb)) * inv
                if v < 0.0:
                    v = 0.0
                if not squared:
                    v = np.sqrt(v)
                out[oi + j] = v
                if sym_write:
                    out[offs[j] + i] = v

    @njit(**_JIT)
    def dense_cross(P, PL, Q, QL, inv, squared, out):
        n, k = P.shape
        m_rows = Q.shape[0]
        nv = (k // 8) * 8
        for i in prange(n):
            oi = i * m_rows
            mb = np.empty(k)
            lg = np.empty(k)
            for j in range(m_rows):
                for c in range(k):
                    m = 0.5 * (np.float64(P[i, c]) + np.float64(Q[j, c]))
                    mb[c] = m if m > 0.0 else 1.0
                for c0 in range(0, nv, 8):
                    _svml_log8(mb, lg, c0)
                for c in range(nv, k):
                    lg[c] = np.log(mb[c])
                acc = 0.0
                # The vectorized log runs over every column, one-sided
                # ones included -- that is what keeps the row branch-free and
                # lets it vectorize at any density. The COMBINE loop is scalar
                # anyway, so recognising the one-sided case here is nearly
                # free, and it has to be recognised somewhere: evaluating
                # `a ln a - 2 (a/2) ln(a/2)` gives `a ln 2` in exact arithmetic
                # but not in float64, so folding those coordinates into the
                # general term would stop disjoint supports returning exactly
                # 1.0 on this backend.
                # One accumulator per row, for the same reason as the libm
                # kernel: a shared one associates the two rows' masses
                # together and misses `S_i + S_j`.
                onea = 0.0
                oneb = 0.0
                for c in range(k):
                    a = np.float64(P[i, c])
                    b = np.float64(Q[j, c])
                    if a > 0.0:
                        if b > 0.0:
                            acc += PL[i, c] + QL[j, c] - 2.0 * mb[c] * lg[c]
                        else:
                            onea += a
                    elif b > 0.0:
                        oneb += b
                v = 0.5 * (acc + LN2 * (onea + oneb)) * inv
                if v < 0.0:
                    v = 0.0
                if not squared:
                    v = np.sqrt(v)
                out[oi + j] = v

    _built.update(log_block=log_block, xlogx=xlogx, dense_self=dense_self,
                  dense_cross=dense_cross)
    return _built


def _log_block(x, out):
    return _build()["log_block"](np.ascontiguousarray(x, dtype=np.float64), out)


def xlogx_svml(P):
    """``x ln x`` with the kernel's own log, always in float64.

    Both of those matter, for the reasons the module docstring gives: the
    duplicate-row cancellation needs ``PL`` and the kernel's
    ``2 m log m`` term to come from the same log, and the inner loop runs in
    float64 whatever the input dtype, so a float32 ``PL`` would carry a
    rounding the float64 term does not.
    """
    P = np.ascontiguousarray(P)
    return _build()["xlogx"](P, np.empty(P.shape, dtype=np.float64))


def _require():
    if not available():
        raise RuntimeError(f"backend='numba-svml' unavailable: {why_unavailable()}")


def matrix(P, Q, inv, squared, out):
    """Signature-compatible with ``_numba.matrix``."""
    _require()
    from ._numba import _balanced_order, _offsets
    b = _build()
    n = P.shape[0]
    PL = xlogx_svml(P)
    if Q is None:
        if out is None:
            out = np.empty((n, n), dtype=P.dtype)
        b["dense_self"](P, PL, inv, squared, out.reshape(-1),
                        _offsets(n, False), _balanced_order(n), True)
        np.fill_diagonal(out, 0.0)
        return out
    m = Q.shape[0]
    if out is None:
        out = np.empty((n, m), dtype=P.dtype)
    b["dense_cross"](P, PL, Q, xlogx_svml(Q), inv, squared, out.reshape(-1))
    return out


def pdist(P, inv, squared, out):
    """Signature-compatible with ``_numba.pdist``."""
    _require()
    from ._numba import _balanced_order, _offsets
    n = P.shape[0]
    if out is None:
        out = np.empty(n * (n - 1) // 2, dtype=P.dtype)
    _build()["dense_self"](P, xlogx_svml(P), inv, squared, out,
                           _offsets(n, True), _balanced_order(n), False)
    return out
