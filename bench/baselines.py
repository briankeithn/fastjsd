"""Baselines: what people run, what the literature published, what is installed.

Deliberately NOT included as a baseline: this package's own reference
implementations, the per-pair Python lambda and the chunked NumPy broadcast in
`fastjsd.reference`. Timing against them measures those implementations rather
than the kernel; they are kept as prior art and as correctness references.

Five that do count, in the order they answer questions:

`scipy_pdist`
    `scipy.spatial.distance.pdist(P, "jensenshannon")`. What a practitioner
    reaches for. Compiled, single-threaded, dense-only, base e only.

`lee_sparse_numba` and `lee_modern_numba`
    The PUBLISHED sparse form, on the same parallel harness fastjsd gets --
    same Numba, same thread count, same balanced row decomposition. Lee
    (1997, TR-11-97 Eq. 2.8) / Dagan-Lee-Pereira (ACL 1997) / Connor et al.
    (SISAP 2013) all write it with the constant folded OUT of the sum:

        JS(p,q) = ln2 + (1/2) * SUM_{c in supp(p) & supp(q)}
                      [ p ln(p/(p+q)) + q ln(q/(p+q)) ]

    fastjsd instead keeps every term inside the sum. Folding makes `ln 2`
    cancel numerically rather than symbolically, so identical rows come back
    as float dust instead of exactly 0.

    There are TWO of these, and they answer different questions.

    `lee_sparse_numba` transcribes the equation as printed, so it pays two
    logarithms and two divisions per matched coordinate. That is the right
    target for the EXACTNESS claim -- it is the arrangement as published --
    and `bench/lee_arrangement.py` uses it for exactly that.

    It is the wrong target for a SPEED claim. The printed form factors as
    `a ln a + b ln b - s ln s`, so the folded arrangement accepts the same
    `x ln x` table fastjsd uses; two logarithms against one is a tabulation
    difference with no connection to the fold. `lee_modern_numba` is the same
    identity and the same fold with the tabulation and the occupancy
    prefilter (Faloutsos & Christodoulakis 1984, which predates Lee)
    restored. Quote that one for speed. Its expected behaviour is that it is
    slightly FASTER than fastjsd's merge per matched coordinate, because
    folding absorbs the one-sided mass and so needs no `hita`/`hitb`
    accumulation at all.

`pynndescent_numba`, `torch_batched`
    Third-party implementations, because "what else is installed in a Python
    environment?" is a fair question. pynndescent's Numba kernel computes a
    *smoothed* divergence rather than this one -- see the function.

    sklearn is deliberately NOT here. It has no compiled Jensen-Shannon metric:
    `metric='jensenshannon'` is rejected by its string whitelist, so the only
    route is SciPy's *scalar* function passed as a callable and invoked once per
    pair through the interpreter. Timing that measures interpreter round-trips,
    not a JSD implementation, so sklearn is reported as an absence rather than
    timed.

Every implementation is checked against SciPy before it is timed, against the
agreement it can actually hold (`TOL`) -- a baseline that silently computes
something else is worse than no baseline.
"""

from __future__ import annotations

import numpy as np

LN2 = 0.6931471805599453

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                        # pragma: no cover
    _HAVE_NUMBA = False


def scipy_pdist(P, base=2.0):
    """`scipy.spatial.distance.pdist(..., 'jensenshannon')`, square form."""
    from scipy.spatial import distance
    P = np.asarray(P, dtype=np.float64)
    d = distance.squareform(distance.pdist(P, metric="jensenshannon"))
    return d / np.sqrt(np.log(base))


if _HAVE_NUMBA:
    _JIT = dict(parallel=True, cache=True, nogil=True, fastmath=False)

    @njit(**_JIT)
    def _lee_self(data, indices, indptr, inv, out, offs, order):
        """Eq. (2.8) transcribed LITERALLY, as printed.

        Same merge and same schedule as fastjsd's sparse kernel, but this is
        deliberately a transcription rather than an optimized implementation:
        it evaluates ``a ln(a/s) + b ln(b/s)`` as written, paying **two**
        logarithms and two divisions per matched coordinate.

        Two logarithms against fastjsd's one is a difference in the
        *tabulation*, entirely orthogonal to the fold, and on a sparse kernel
        whose per-match work is nearly all transcendental it plausibly
        dominates whatever the fold costs.

        This function exists for the exactness argument -- it is what the
        literature prints, so it is the right target for
        ``bench/lee_arrangement.py``. For SPEED, compare against
        ``_lee_modern_self`` below, which is the same identity and the same
        fold, implemented the way an optimizing implementer would write it.
        """
        n = offs.shape[0]
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            ai0, ai1 = indptr[i], indptr[i + 1]
            for j in range(i + 1, n):
                aj0, aj1 = indptr[j], indptr[j + 1]
                acc = 0.0
                pa, pb = ai0, aj0
                while pa < ai1 and pb < aj1:
                    ca, cb = indices[pa], indices[pb]
                    if ca < cb:
                        pa += 1
                    elif cb < ca:
                        pb += 1
                    else:
                        a = data[pa]
                        b = data[pb]
                        s = a + b
                        # p ln(p/s) + q ln(q/s), as published
                        acc += a * np.log(a / s) + b * np.log(b / s)
                        pa += 1
                        pb += 1
                v = (LN2 + 0.5 * acc) * inv
                if v < 0.0:
                    v = 0.0
                v = np.sqrt(v)
                out[oi + j] = v
                out[offs[j] + i] = v


if _HAVE_NUMBA:
    @njit(parallel=True, nogil=True, cache=False)
    def _lee_modern_self(data, indices, indptr, PL, sig, inv, out, offs,
                         order):
        """Eq. (2.8) as an optimizing implementer would write it.

        Same identity, same fold -- ``ln 2`` still lives OUTSIDE the sum, which
        is the thing that makes this Lee's arrangement and not fastjsd's. What
        it no longer carries is two handicaps that have nothing to do with the
        fold and that the literal transcription pays only because of how the
        equation is typeset:

        1. **One logarithm, not two.** The printed form factors, exactly:

               a ln(a/s) + b ln(b/s) = a ln a + b ln b - s ln s

           so the per-row ``x ln x`` table that fastjsd builds serves the
           folded form just as well. ``PL`` is built with
           ``xlogx_like_kernels``, the same log at the same precision the
           kernels use -- not because this baseline needs the exactness
           argument, but because anything else would be comparing two
           different ``log`` implementations.
        2. **The occupancy prefilter.** Superimposed coding (Faloutsos &
           Christodoulakis 1984) is not this package's invention and predates
           Lee; withholding it from the baseline would measure whether a 1984
           technique was implemented, not the kernel. It is exact for the
           folded form: a pair with no shared coordinate leaves ``acc = 0``,
           so ``v = ln2 * inv``, which is what the merge would have produced.

        What REMAINS different from ``fastjsd._numba._sparse_self`` after
        this is exactly two things, and they are the two the arrangement
        changes: where ``ln 2`` sits, and the two-accumulator one-sided
        complement. Note which way that cuts -- the folded form needs no
        ``hita``/``hitb`` at all, since the one-sided mass is what folding
        absorbs. So this baseline does strictly LESS work per matched
        coordinate than fastjsd does, and the expectation is that it is
        slightly FASTER on the merge, not slower. The arrangement buys
        exactness; it cannot buy speed.
        """
        n = offs.shape[0]
        nw = sig.shape[1]
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            ai0 = indptr[i]
            ai1 = indptr[i + 1]
            gi = sig[i]
            for j in range(i + 1, n):
                acc = 0.0
                shared = False
                for w in range(nw):
                    if gi[w] & sig[j, w]:
                        shared = True
                        break
                if shared:
                    aj0 = indptr[j]
                    aj1 = indptr[j + 1]
                    pa = ai0
                    pb = aj0
                    while pa < ai1 and pb < aj1:
                        ca = indices[pa]
                        cb = indices[pb]
                        if ca < cb:
                            pa += 1
                        elif cb < ca:
                            pb += 1
                        else:
                            s = data[pa] + data[pb]
                            # a ln a + b ln b - s ln s, one logarithm.
                            acc += PL[pa] + PL[pb] - s * np.log(s)
                            pa += 1
                            pb += 1
                v = (LN2 + 0.5 * acc) * inv
                if v < 0.0:
                    v = 0.0
                v = np.sqrt(v)
                out[oi + j] = v
                out[offs[j] + i] = v


def _lee_prepare(P):
    """Shared setup for both Lee baselines: normalize and build CSR."""
    from fastjsd._numba import csr_from_dense
    P = np.ascontiguousarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    return P, csr_from_dense(P)


def lee_sparse_numba(P, base=2.0):
    """Published sparse JS (Lee 1997 Eq. 2.8) transcribed literally.

    Two logarithms per matched coordinate. This is the EXACTNESS target, not
    the speed target -- see `lee_modern_numba`.
    """
    if not _HAVE_NUMBA:
        raise RuntimeError("lee_sparse_numba needs numba")
    from fastjsd._numba import _balanced_order, _offsets
    P, (data, idx, ptr, _) = _lee_prepare(P)
    n = P.shape[0]
    out = np.empty(n * n)
    _lee_self(data, idx, ptr, 1.0 / np.log(base), out,
              _offsets(n, False), _balanced_order(n))
    D = out.reshape(n, n)
    np.fill_diagonal(D, 0.0)
    return D


def lee_modern_numba(P, base=2.0):
    """Lee 1997 Eq. 2.8, same fold, tabulated logs and occupancy prefilter.

    **This is the speed baseline to quote.** It isolates the arrangement,
    which is the claim under test, instead of crediting this package for a log
    tabulation and a 1984 prefilter that the folded form accepts just as
    readily.
    """
    if not _HAVE_NUMBA:
        raise RuntimeError("lee_modern_numba needs numba")
    from fastjsd._numba import (
        _balanced_order,
        _offsets,
        _signatures,
        xlogx_like_kernels,
    )
    P, (data, idx, ptr, _) = _lee_prepare(P)
    n, k = P.shape
    PL = xlogx_like_kernels(data)
    sig = _signatures(idx, ptr, k)
    out = np.empty(n * n)
    _lee_modern_self(data, idx, ptr, PL, sig, 1.0 / np.log(base), out,
                     _offsets(n, False), _balanced_order(n))
    D = out.reshape(n, n)
    np.fill_diagonal(D, 0.0)
    return D


BASELINES = {
    "scipy pdist": scipy_pdist,
    "Lee-1997 sparse (numba)": lee_sparse_numba,
    "Lee-1997 modern (numba)": lee_modern_numba,
}


# ---------------------------------------------------------------------------
# Third-party implementations.
#
# The two baselines above answer "is the kernel good?" (SciPy = what people
# reach for; Lee-1997 = the published identity on the same harness). These
# answer a different question: "what else is actually installed in a Python
# environment, and how does it compare?"
#
# Each declares the agreement it can hold, because they do not all compute the
# same function. `TOL` is asserted, not assumed -- a baseline that silently
# computes something else is worse than no baseline.
# ---------------------------------------------------------------------------

if _HAVE_NUMBA:
    @njit(parallel=True, nogil=True, cache=False)
    def _pynn_all_pairs(P, out, kern):
        """pynndescent's kernel over every pair, in this harness's schedule.

        Driving a third-party per-pair function from the same parallel harness
        is the only way to compare kernels rather than compare harnesses --
        pynndescent itself never forms a full matrix, it feeds this into an
        approximate-nearest-neighbour graph build.
        """
        n = P.shape[0]
        for i in prange(n):
            for j in range(i + 1, n):
                v = kern(P[i], P[j])
                if v < 0.0:
                    v = 0.0
                v = np.sqrt(v)
                out[i * n + j] = v
                out[j * n + i] = v
        return out


def pynndescent_numba(P, base=2.0):
    """pynndescent 0.6's `jensen_shannon_divergence`, njit, over all pairs.

    NOT exact: it adds `FLOAT32_EPS` (1.19e-07) to every coordinate before
    normalizing, which guards `log(0)` at the cost of ~1e-6 relative deviation
    from the true divergence. Identical rows still come back at exactly 0.0,
    because the smoothing is symmetric. Reported with that deviation rather
    than excluded: it is among the most widely installed Numba JSD kernels in
    the Python ecosystem, and the trade it makes is worth showing.
    """
    from pynndescent.distances import jensen_shannon_divergence
    P = np.ascontiguousarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    n = P.shape[0]
    out = np.zeros(n * n)
    _pynn_all_pairs(P, out, jensen_shannon_divergence)
    D = out.reshape(n, n) / np.sqrt(np.log(base))
    np.fill_diagonal(D, 0.0)
    return D


def torch_batched(P, base=2.0, chunk=64):
    """The tensorized form an ML practitioner writes, in PyTorch.

    Same `(chunk, n, k)` mixture tensor as the NumPy broadcast in
    `fastjsd.reference`, so it inherits the same bandwidth problem -- included
    because torch is widely installed, and because "would torch have rescued
    the broadcast approach?" is a question to settle with a measurement rather
    than an argument. CPU only; torch uses all cores.
    """
    import torch
    torch.set_num_threads(len(__import__("os").sched_getaffinity(0)))
    X = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float64))
    X = X / X.sum(1, keepdim=True)
    n = X.shape[0]
    out = torch.empty((n, n), dtype=torch.float64)
    xlx = torch.special.xlogy(X, X).sum(1)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        M = 0.5 * (X[lo:hi, None, :] + X[None, :, :])
        hm = torch.special.xlogy(M, M).sum(-1)
        out[lo:hi] = 0.5 * (xlx[lo:hi, None] + xlx[None, :]) - hm
    out.clamp_(min=0.0)
    D = (out / np.log(base)).sqrt().numpy()
    np.fill_diagonal(D, 0.0)
    return D


#: Agreement each baseline can hold against SciPy, asserted by the harness.
TOL = {
    "scipy pdist": 0.0,
    "Lee-1997 sparse (numba)": 5e-12,
    # The modern Lee folds ln2 outside the sum exactly as the published one
    # does, so it inherits the same behaviour at zero and the same tolerance.
    "Lee-1997 modern (numba)": 5e-12,
    # pynndescent's deviation is data-dependent, because FLOAT32_EPS matters
    # more the more zeros a row has: measured 6.0e-06 on dense synthetic rows
    # and 3.6e-05 on real thresholded memberships. The bound here is a sanity
    # check that it computes SOMETHING like JSD, not a claim of agreement --
    # the per-case deviation is printed instead.
    "pynndescent (numba)": 1e-02,
    "torch batched": 5e-12,
}

THIRD_PARTY = {
    "pynndescent (numba)": pynndescent_numba,
    "torch batched": torch_batched,
}
