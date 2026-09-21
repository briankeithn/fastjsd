"""Numba backends: a dense parallel kernel and an exact sparse kernel.

Both accumulate a scalar per pair, so no ``(rows, n, k)`` tensor is ever
materialized -- the pair loop is compute-bound rather than
bandwidth-bound, and it parallelizes cleanly over rows.

The sparse kernel walks only the *support intersection* of each pair. That
is exact, not an approximation: by fact (2) in ``_common``, a coordinate
where only one side is nonzero contributes exactly ``p ln 2``, so the merge
accumulates that mass and multiplies once at the end -- no logarithm. For
thresholded simplex data (soft cluster memberships, topic mixtures,
short-document term distributions), where density is often 3-25 %, the
intersection is quadratically smaller than ``k``.

Import is guarded: ``available()`` is False when Numba is not installed, and
the dispatcher falls back to the NumPy backend.
"""

from __future__ import annotations

import numpy as np

from ._common import LN2

__all__ = ["available", "matrix", "pdist", "matrix_sparse", "pdist_sparse",
           "xlogx_like_kernels", "snap_rows_like_kernels",
           "prepare_rows_like_kernels",
           "csr_from_dense", "set_num_threads"]

try:  # pragma: no cover - whichever branch this environment has
    import numba
    from numba import njit, prange, types
    from numba.extending import intrinsic

    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False
    numba = None


def available() -> bool:
    return _HAVE


if _HAVE:
    _JIT = dict(parallel=True, cache=True, nogil=True, fastmath=False)

    @njit(cache=True, nogil=True, fastmath=False, inline="always")
    def _snap_one_row(P, i):
        """Three staged attempts to make row ``i`` total exactly 1.0.

        Left-to-right, because that is the order the distance kernels
        accumulate in. Each stage is verified, not assumed, and each only runs
        if the previous one left the row short. `_common.snap_row_sums`
        documents why the order of the stages matters -- briefly, stage 3
        always works but puts the whole correction on what may be a tiny
        entry, so it is the terminator and not the method.
        """
        k = P.shape[1]

        for _ in range(3):                     # stage 1: divide by the total
            s = 0.0
            for c in range(k):
                s += P[i, c]
            if s == 1.0:
                return
            for c in range(k):
                P[i, c] = P[i, c] / s

        big = 0                                # stage 2: nudge the largest
        bv = P[i, 0]
        for c in range(1, k):
            if P[i, c] > bv:
                bv = P[i, c]
                big = c
        for _ in range(3):
            s = 0.0
            for c in range(k):
                s += P[i, c]
            if s == 1.0:
                return
            P[i, big] += 1.0 - s

        last = -1                              # stage 3: last non-zero
        for c in range(k - 1, -1, -1):
            if P[i, c] > 0.0:
                last = c
                break
        if last >= 0:
            before = 0.0
            for c in range(last):
                before += P[i, c]
            fixed = 1.0 - before
            if fixed > 0.0 and before + fixed == 1.0:
                P[i, last] = fixed

    @njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _snap_rows_nb(P):
        """Row-at-a-time version of ``_common.snap_row_sums``.

        One scalar pass per row instead of a ``cumsum`` over the whole
        ``(n, k)`` array, and nothing allocated. The NumPy version issues up
        to six full-array ``cumsum`` calls, each streaming the entire array
        from memory, so its cost scales with ``n k`` and is charged to every
        call. Where the distance kernel itself is cheap -- sparse rows, small
        ``k`` -- that setup is a large fraction of the total.
        """
        for i in prange(P.shape[0]):
            _snap_one_row(P, i)
        return P

    @njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _prepare_rows_nb(src, dst, status):
        """Validate, normalize and snap, in one pass per row.

        The NumPy path does these as five or six separate whole-array passes:
        ``isfinite``, ``min``, ``sum``, the division, then the snap. Each
        streams the full ``(n, k)`` array from memory. At n=2048, k=1024 that
        is ~16 MB a pass, a setup cost charged to every call and independent
        of how much work the distance kernel itself does.

        ``dst`` may be ``src``. When it is not, the copy the caller would have
        made to avoid mutating its input happens here instead, inside the pass
        that was reading the data anyway -- worth ~16 MB of traffic at
        n=2048, k=1024.

        Row at a time, so each row stays in cache across all of it. Validation
        reports through ``status`` rather than raising, because the caller owns
        the error messages: 0 ok, 1 non-finite, 2 negative, 3 non-positive sum.

        The division is a division, not a multiply by the reciprocal. They
        round differently, and control over rounding is the entire point of
        what follows.
        """
        n, k = src.shape
        for i in prange(n):
            bad = 0
            s = 0.0
            for c in range(k):
                v = np.float64(src[i, c])
                if np.isnan(v) or np.isinf(v):
                    bad = 1
                elif v < 0.0 and bad == 0:
                    bad = 2
                s += v
            if bad != 0:
                status[i] = bad
                continue
            if not (s > 0.0):
                status[i] = 3
                continue
            status[i] = 0
            for c in range(k):
                dst[i, c] = src[i, c] / s
            _snap_one_row(dst, i)
        return status

    @intrinsic
    def _popcount(typingctx, x):
        """Population count of a uint64. Lowers to POPCNT on x86-64.

        Numba has no NumPy coverage for ``np.bitwise_count``, so this goes
        through LLVM directly. It survives both ``parallel=True`` and
        ``cache=True``, and emits the instruction rather than a libcall.
        """
        def codegen(context, builder, sig, args):
            (v,) = args
            fn = builder.module.declare_intrinsic("llvm.ctpop", [v.type])
            return builder.zext(builder.call(fn, [v]),
                                context.get_value_type(types.int64))
        return types.int64(types.uint64), codegen

    @njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _csr_counts_nb(P, counts):
        for i in prange(P.shape[0]):
            c = 0
            for j in range(P.shape[1]):
                if P[i, j] > 0.0:
                    c += 1
            counts[i] = c
        return counts

    @njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _csr_fill_nb(P, indptr, indices, data):
        """Fill the triplet. Columns come out ascending, which the merge needs.

        The NumPy version builds this with boolean fancy indexing --
        ``cols[nz]`` over a broadcast ``arange``, then ``P[nz]`` -- which
        costs three whole-array temporaries and a gather. The compiled fill
        allocates nothing beyond the triplet it writes.
        """
        for i in prange(P.shape[0]):
            p = indptr[i]
            for j in range(P.shape[1]):
                v = P[i, j]
                if v > 0.0:
                    indices[p] = j
                    data[p] = v
                    p += 1
        return data

    def snap_rows_like_kernels(P):
        """Public entry: snap in place with the compiled kernel."""
        return _snap_rows_nb(P)

    def prepare_rows_like_kernels(src, dst):
        """Validate + normalize + snap in one pass. Returns the status codes."""
        return _prepare_rows_nb(src, dst,
                                np.empty(src.shape[0], dtype=np.int8))

    @njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _xlogx_nb(x, out):
        """``x ln x`` exactly as the kernels' inner loop would compute it.

        Two things have to match, or the duplicate-row guarantee is lost. The
        loop cancels ``PL[i,c] + PL[j,c]`` against ``2 m log m``, which is
        exact only when both sides are produced by identical arithmetic: for
        identical rows ``m == a`` bitwise, so ``2 a L - 2 a L`` is zero
        *provided both L's are the same L*.

        1. **Same log.** Not ``_common.xlogx``: NumPy dispatches ``np.log`` to
           its own vectorized implementation, Numba lowers it to libm, and the
           two differ by 1 ULP on a small minority of inputs -- a few rows in
           a hundred Dirichlet draws, and roughly one coordinate in each of
           those. A mismatch there leaves ~1e-17 in the divergence of a pair
           of identical rows and ~6e-9 in the distance.

        2. **Same precision.** ``out`` is float64 even for float32 input,
           because the inner loop is float64 regardless -- the ``0.5`` and
           ``2.0`` literals promote ``m`` and everything downstream of it. A
           float32 ``PL`` would carry a rounding the float64 term does not,
           which for float32 input reaches ~1e-4 on duplicate rows.

        Costs O(nk) against the loop's O(n^2 k), so it is free either way.
        """
        n = x.shape[0]
        for i in prange(n):
            v = np.float64(x[i])
            out[i] = v * np.log(v) if v > 0.0 else 0.0
        return out

    def _xlogx_shaped(P):
        flat = np.ascontiguousarray(P).reshape(-1)
        out = np.empty(flat.shape[0], dtype=np.float64)
        return _xlogx_nb(flat, out).reshape(P.shape)

    @njit(parallel=True, cache=True, nogil=True, fastmath=False)
    def _row_sums_nb(data, indptr, out):
        """Per-row mass, summed **sequentially in stored column order**.

        The sparse kernels recover one-sided mass as
        ``(rsum[i] - hita) + (rsum[j] - hitb)``, where ``hita``/``hitb``
        accumulate each row's MATCHED mass separately, rather than by adding up
        the skipped entries during the merge. That is what lets them step past
        a non-matching entry with a bare pointer bump -- the one structural
        advantage the folded arrangement has, which ``docs/MATH.md``
        Consequence 2b sets out -- while keeping the terms unfolded.

        The subtraction is only safe because of the summation order. ``hita``
        sums row i's matched entries in stored column order, which is exactly
        what this function sums, so when a pair's supports are IDENTICAL every
        entry matches and ``hita`` is bitwise ``rsum[i]``: ``one`` is exactly
        0.0, for any values.

        Two accumulators, not one. A single ``hit += a + b`` associates the
        additions differently from ``rsum``, so it leaves ~k*eps instead of 0 --
        harmless for well-separated rows, but for NEAR-duplicates the true
        divergence is O(u^2) and that residue dominates it. With one
        accumulator the sparse kernel bottoms out at ~2e-08 error on rows
        differing by 1e-9, where the dense kernel reaches ~1e-17.

        ``np.sum`` would break it for the same reason -- it reduces in a
        divide-and-conquer tree, not left to right. (NumPy's name for that is
        "pairwise summation", which in this package collides with *pairwise*
        meaning "over all pairs of rows"; they are unrelated.) Accumulate left
        to right, and only left to right.
        """
        n = indptr.shape[0] - 1
        for i in prange(n):
            acc = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += np.float64(data[p])
            out[i] = acc
        return out

    def _row_sums(data, indptr):
        return _row_sums_nb(data, indptr,
                            np.empty(indptr.shape[0] - 1, dtype=np.float64))

    @njit(parallel=True, cache=True, nogil=True)
    def _signatures_nb(indices, indptr, out):
        """Occupancy bitmap per row, ``nw`` 64-bit words wide.

        The expected intersection size of two rows of density ``d`` is
        ``k d^2``. At k=256, d=3 % that is 0.23 -- the overwhelming majority
        of pairs share no coordinate at all, yet the merge has to walk both
        rows to find that out. ANDing the bitmaps settles it without walking.

        Columns fold modulo ``64 * nw``, and a fold produces false POSITIVES
        only: those pairs go to the merge and get the same answer. There are no
        false negatives, so the skip stays exact -- an empty intersection
        leaves ``acc = 0`` and no matched mass, which is bit-for-bit what the
        merge would have left.

        **The width has to follow k.** With ``B`` buckets and ``z`` non-zeros
        per row the skip fires with probability about ``exp(-z^2 / B)``, so a
        single word decays fast in ``z``: at z=16 that is ``exp(-4)`` = 1.8 %,
        against the 36 % of pairs that genuinely share no coordinate at k=256.
        One word therefore discards most of the skip's own opportunity. Four
        words take the same case to ``exp(-1)`` = 37 %, and for ``k <= 256``
        there is no fold at all, so the test becomes exact.
        """
        n = indptr.shape[0] - 1
        nw = out.shape[1]
        nbits = nw * 64
        for i in prange(n):
            for w in range(nw):
                out[i, w] = np.uint64(0)
            for p in range(indptr[i], indptr[i + 1]):
                b = indices[p] % nbits
                out[i, b >> 6] |= np.uint64(1) << np.uint64(b & 63)
        return out

    #: Cap on signature words. Four covers k <= 256 exactly, which is the range
    #: topic and cluster mixtures live in; past that the fold returns, and the
    #: test's own cost has to stay negligible against walking two rows.
    _SIG_MAX_WORDS = 4

    def _signatures(indices, indptr, k):
        nw = min(_SIG_MAX_WORDS, max(1, (int(k) + 63) // 64))
        out = np.empty((indptr.shape[0] - 1, nw), dtype=np.uint64)
        return _signatures_nb(indices, indptr, out)

    def _unfolded(k):
        """Does the signature name every column uniquely?

        Only then does a set bit identify one column, which is what
        rank-select and the ordering argument both need. Dispatch is by
        DENSITY, not by k -- a 20k-column vocabulary matrix at 1 % density
        reaches the sparse kernel -- so this is a live branch, not an edge
        case. Under a fold, rank-select silently addresses the wrong CSR
        entry, which shows up as 0.2-0.32 ABSOLUTE error on the distance,
        not as a rounding difference. Hard gate, never a tolerance.
        """
        return 64 * min(_SIG_MAX_WORDS, max(1, (int(k) + 63) // 64)) >= int(k)

    def _prefer_bits(sig, indptr, rng_seed=0, sample=4096):
        """Is there enough wasted merge work to pay for bit iteration?

        Bit iteration deletes the merge's non-matching steps, which are the
        expensive part. The merge's three-way branch is data-dependent, so a
        non-matching step costs a branch mispredict rather than arithmetic.
        Bit iteration pays for deleting them with more work per MATCH --
        isolate, mask, two popcounts, two rank adds -- so where the signature
        already skips almost everything there is nothing to win and a little
        to lose.

        The quantity that decides it is the mean number of non-matching merge
        steps per pair, which is a COUNT and therefore reproducible on any
        machine. Density is not a proxy for it: k=256 z=16 and k=64 z=4 are
        both 6.2 % dense and want opposite kernels, because their skip rates
        are 34 % and 77 %. So this samples the count rather than modelling it
        -- with an unfolded signature ``popcount(sig_i & sig_j)`` IS the
        intersection size, so a few thousand random pairs give the mean
        exactly, for the data in hand, in O(sample).

        ---- What the statistic does not capture ----

        The mean step count does not separate the two kernels cleanly. Within
        one machine's measurements the winners interleave along that axis
        rather than sorting onto either side of a cut, so no threshold on this
        quantity alone is right everywhere and raising or lowering 1.5 does
        not fix that.

        The leading candidate for what the mean misses is the VARIANCE of the
        row degrees. The proposed mechanism: the merge's cost is branch
        mispredicts, and a mispredict comes from an UNPREDICTABLE inner trip
        count rather than from a large average one, so with every row exactly
        ``z`` long the merge's loop predicts perfectly, while on skewed
        degrees it does not -- and the bitmap kernel, whose intersection walk
        has no data-dependent branch at all, is indifferent either way. The
        pairing that motivates it is a synthetic k=64 z=4 grid cell against a
        real argmax-thresholded fixture: same mean intersection (0.25),
        similar skip rates (76 % and 83 %), degree CV 0.00 against 0.77.

        Treat that as a hypothesis to measure, not as a rule. It is stated
        here because it is the most plausible account of why this probe can be
        wrong, but the separation it predicts has not held up across machines;
        ``bench/bits_vs_merge.py`` reports degree CV beside the step count and
        splits its summary on it, so the check is cheap to repeat on any host.

        Two consequences:

        * On real thresholded fixtures with skewed degrees the probe can pick
          the merge where the bitmap kernel would have been faster. Both
          kernels return bit-identical results, so a wrong pick costs time
          only, never accuracy.
        * ``sparse_rows`` in ``bench/datasets.py`` gives every row exactly
          ``nnz`` non-zeros, so the whole synthetic sparse grid sits at degree
          CV = 0 -- degenerate in precisely the dimension under suspicion, and
          a shape real data does not have. That grid alone cannot settle the
          question either way; ``bench/bits_vs_merge.py`` carries skewed
          fixtures for this reason.

        ---- Portability ----

        The crossover rides on the branch predictor, so it is
        microarchitecture-dependent: the same case can pick opposite winners
        on two different machines, well outside the tie-band an A/A control
        reports. Three consequences for anyone changing this:

        * Do not re-tune the constant from a single machine. Run
          ``bench/bits_vs_merge.py`` on several, and read its ratios against
          the tie-bands from ``bench/aa_control.py``.
        * A richer rule such as ``steps > 2.0 or degreeCV > 0.5`` can be made
          to fit a handful of measured cells exactly, which is the kind of
          two-parameter fit to few points that should not be trusted. It is
          not adopted.
        * The durable fix is not a better constant. It is either a portable
          cost model with a measured per-mispredict term, or a one-off
          calibration at import time on the host that will run the kernel.

        The probe is kept in preference to a fixed policy -- always taking the
        bitmap kernel when ``_unfolded(k)`` holds -- because across the
        measured cases it picks the faster kernel more often, and because the
        failure mode of a mis-set threshold is bounded while that of a fixed
        policy is not.
        """
        n = indptr.shape[0] - 1
        if n < 3:
            return False
        z = np.diff(indptr)
        rng = np.random.default_rng(rng_seed)
        a = rng.integers(0, n, sample)
        b = rng.integers(0, n, sample)
        keep = a != b
        a, b = a[keep], b[keep]
        if a.size == 0:
            return False
        inter = np.bitwise_count(sig[a] & sig[b]).sum(axis=1)
        steps = np.where(inter > 0, z[a] + z[b] - 2 * inter, 0)
        # 1.5 sits between the synthetic cases measured on either side of
        # the crossover, with room on both sides. See the docstring for why
        # no single threshold on this quantity is right everywhere.
        return float(steps.mean()) > 1.5

    def _ranks(sig, indptr):
        """``rank[i, w]``: CSR position of the first stored column in word w.

        A prefix popcount along the words, offset by the row's CSR start.
        Host-side, so ``np.bitwise_count`` is available here even though Numba
        has no coverage for it.
        """
        pc = np.bitwise_count(sig).astype(np.int64)
        r = np.empty_like(pc)
        r[:, 0] = 0
        if pc.shape[1] > 1:
            np.cumsum(pc[:, :-1], axis=1, out=r[:, 1:])
        return r + indptr[:-1, None]


    # ----------------------------------------------------------------------
    # well-conditioned summand, for a ~ b
    # ----------------------------------------------------------------------
    # t(a,b) = a ln a + b ln b - 2 m ln m is computed from three pieces each
    # O(a ln a) while t itself is O(m u^2), u = (a-b)/(a+b). Measured condition
    # number sum|pieces| / t is 9.2e6 at u = 1e-3 and 9.2e18 at u = 1e-9, so
    # near-duplicate rows come back as pure rounding: 1.5e-8 absolute against a
    # 50-digit reference, tens of times the true value. This is a property of
    # THIS grouping, chosen because a ln a and b ln b tabulate per row; SciPy's
    # a ln(a/m) + b ln(b/m) cancels by 2.0e9 on the same input, better but also
    # unbounded. bench/conditioning.py measures all three.
    #
    # Substituting a = m(1+u), b = m(1-u) makes ln m cancel SYMBOLICALLY:
    #
    #     t = m [ (1+u)ln(1+u) + (1-u)ln(1-u) ]
    #       = m * SUM_{n>=1} u^(2n) / (n(2n-1))
    #       = m ( u^2 + u^4/6 + u^6/15 + u^8/28 + ... )
    #
    # Every series term is positive, so there is no cancellation at any u --
    # 1.8e-17 on the same input. It is only needed where u is small, and the
    # series diverges as u -> 1, so it is a branch rather than a replacement.
    _TAU = 0.25                        # series for |u| <= 1/4
    # Truncation after n=13 leaves ~u^26/(14*27); at u=1/4 that is under eps
    # relative to the leading 1.
    _PHI = tuple(1.0 / (n * (2 * n - 1)) for n in range(13, 0, -1))

    @njit(inline="always")
    def _phi_over_u2(u2):
        p = _PHI[0]
        for i in range(1, 13):
            p = p * u2 + _PHI[i]
        return p

    @njit(**_JIT)
    def _dense_self(P, PL, inv, squared, out, offs, order, sym_write, wc):
        n, k = P.shape
        # Row i does (n-1-i) pairs, so a plain triangular prange leaves the
        # last threads idle. `order` interleaves 0, n-1, 1, n-2, ... so each
        # contiguous chunk the scheduler hands a thread carries the same
        # total pair count.
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            for j in range(i + 1, n):
                acc = 0.0
                # One accumulator per ROW, not one shared. A shared `one`
                # adds a's and b's interleaved in column order, which
                # associates differently from either row's own sum, so a pair
                # with disjoint supports lands near `S_i + S_j` instead of on
                # it -- measured, 23 % of disjoint pairs. Per-row accumulators
                # replay each row's own left-to-right order, and `prepare` has
                # snapped that order's total to exactly 1.0, so the two sum to
                # exactly 2.0 and the distance is exactly 1. Same argument as
                # the sparse merge's two-accumulator complement.
                onea = 0.0
                oneb = 0.0
                for c in range(k):
                    a = P[i, c]
                    b = P[j, c]
                    if a > 0.0:
                        if b > 0.0:
                            sm = a + b
                            m = 0.5 * sm
                            # The branch test avoids a division: |u| <= 1/4 iff
                            # |a-b| <= s/4.
                            if wc and abs(a - b) <= _TAU * sm:
                                u2 = ((a - b) / sm) ** 2
                                acc += m * u2 * _phi_over_u2(u2)
                            else:
                                acc += (PL[i, c] + PL[j, c]
                                        - 2.0 * m * np.log(m))
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
    def _dense_cross(P, PL, Q, QL, inv, squared, out, wc):
        n, k = P.shape
        m_rows = Q.shape[0]
        for i in prange(n):
            oi = i * m_rows
            for j in range(m_rows):
                acc = 0.0
                # One accumulator per ROW, not one shared. A shared `one`
                # adds a's and b's interleaved in column order, which
                # associates differently from either row's own sum, so a pair
                # with disjoint supports lands near `S_i + S_j` instead of on
                # it -- measured, 23 % of disjoint pairs. Per-row accumulators
                # replay each row's own left-to-right order, and `prepare` has
                # snapped that order's total to exactly 1.0, so the two sum to
                # exactly 2.0 and the distance is exactly 1. Same argument as
                # the sparse merge's two-accumulator complement.
                onea = 0.0
                oneb = 0.0
                for c in range(k):
                    a = P[i, c]
                    b = Q[j, c]
                    if a > 0.0:
                        if b > 0.0:
                            sm = a + b
                            m = 0.5 * sm
                            if wc and abs(a - b) <= _TAU * sm:
                                u2 = ((a - b) / sm) ** 2
                                acc += m * u2 * _phi_over_u2(u2)
                            else:
                                acc += (PL[i, c] + QL[j, c]
                                        - 2.0 * m * np.log(m))
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

    @njit(**_JIT)
    def _sparse_self(data, indices, indptr, PL, rsum, sig, inv, squared,
                     out, offs, order, sym_write, wc):
        n = offs.shape[0]
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            ai0 = indptr[i]
            ai1 = indptr[i + 1]
            si = rsum[i]
            gi = sig[i]
            nw = sig.shape[1]
            for j in range(i + 1, n):
                acc = 0.0
                hita = 0.0
                hitb = 0.0
                shared = False
                for w in range(nw):
                    if gi[w] & sig[j, w]:
                        shared = True
                        break
                if shared:                 # else: no shared coordinate at all
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
                            va = data[pa]
                            vb = data[pb]
                            s = va + vb
                            m = 0.5 * s
                            if wc and abs(va - vb) <= _TAU * s:
                                u2 = ((va - vb) / s) ** 2
                                acc += m * u2 * _phi_over_u2(u2)
                            else:
                                acc += PL[pa] + PL[pb] - 2.0 * m * np.log(m)
                            hita += va
                            hitb += vb
                            pa += 1
                            pb += 1
                # One-sided mass by complement, NOT by accumulating the
                # skipped entries. TWO accumulators, each summing in the same
                # order ``rsum`` did, so identical supports give exactly 0.0 --
                # see ``_row_sums_nb``.
                one = (si - hita) + (rsum[j] - hitb)
                v = 0.5 * (acc + LN2 * one) * inv
                if v < 0.0:
                    v = 0.0
                if not squared:
                    v = np.sqrt(v)
                out[oi + j] = v
                if sym_write:
                    out[offs[j] + i] = v

    @njit(**_JIT)
    def _sparse_self_bits(data, PL, rsum, sig, rank, inv, squared, out, offs,
                          order, sym_write, wc):
        """The merge, with the merge deleted.

        Requires an UNFOLDED signature -- ``64 * nw >= k`` -- and the callers
        enforce it. Under that condition the map from stored column to set bit
        is a bijection, so ``sig[i] & sig[j]`` names exactly the intersecting
        columns and the two-pointer walk has nothing left to discover.

        Why this is worth a second kernel: the walk's cost is not the compares,
        it is the *mispredicts*. On uniformly-random supports the three-way
        branch is close to a coin flip -- direction-switch rate 0.34-0.47 per
        step -- and a fit to measured timings puts a non-matching step at ~17.7
        cycles, the price of a mispredict rather than of arithmetic. At k=64,
        z=16 the merge takes 26.5 trips per pair to find 4 matches; 22.5 of
        those are pure loss. Bit iteration visits the 4 and nothing else.

        Reading the values back is the part that has to be right. Rather than
        indexing a dense ``P`` -- which needs a dense ``PL`` too, 8 MB at
        k=256, off the end of L2 -- the position in the CSR row is recovered by
        rank-select: ``rank[i, w]`` is the CSR index of the first stored column
        in word ``w``, and ``bit - 1`` is precisely the mask of the bits below
        the isolated bit, so ``rank[i, w] + popcount(sig[i, w] & (bit - 1))``
        is the position. The column index is never materialized, and
        ``indices`` leaves the hot loop entirely.

        EXACTNESS. Word index is monotone in the column, and within a word
        ``bit = m & -m`` peels the lowest set bit first, so columns are visited
        strictly ascending -- the same order the merge used and the same order
        ``_row_sums_nb`` summed in. So ``hita`` is bitwise equal to ``rsum[i]``
        when supports coincide, ``one`` is exactly 0.0, and duplicate rows stay
        at exactly 0.0. ``tests/test_bits_kernel.py`` checks this directly:
        the kernel is bit-for-bit identical to ``_sparse_self`` on every
        fixture there, including k not a multiple of 64.
        """
        n = offs.shape[0]
        nw = sig.shape[1]
        for t in prange(n):
            i = order[t]
            oi = offs[i]
            si = rsum[i]
            for j in range(i + 1, n):
                acc = 0.0
                hita = 0.0
                hitb = 0.0
                for w in range(nw):
                    gi = sig[i, w]
                    gj = sig[j, w]
                    m = gi & gj
                    if m == np.uint64(0):
                        continue
                    ra = rank[i, w]
                    rb = rank[j, w]
                    while m:
                        bit = m & (~m + np.uint64(1))
                        lo = bit - np.uint64(1)
                        pa = ra + _popcount(gi & lo)
                        pb = rb + _popcount(gj & lo)
                        va = data[pa]
                        vb = data[pb]
                        sm = va + vb
                        mm = 0.5 * sm
                        if wc and abs(va - vb) <= _TAU * sm:
                            u2 = ((va - vb) / sm) ** 2
                            acc += mm * u2 * _phi_over_u2(u2)
                        else:
                            acc += PL[pa] + PL[pb] - 2.0 * mm * np.log(mm)
                        hita += va
                        hitb += vb
                        m ^= bit
                one = (si - hita) + (rsum[j] - hitb)
                v = 0.5 * (acc + LN2 * one) * inv
                if v < 0.0:
                    v = 0.0
                if not squared:
                    v = np.sqrt(v)
                out[oi + j] = v
                if sym_write:
                    out[offs[j] + i] = v

    @njit(**_JIT)
    def _sparse_cross_bits(dp, PLp, rp, sp, ka, dq, PLq, rq, sq, kb,
                           n, m_rows, inv, squared, out, wc):
        """Cross-matrix form of ``_sparse_self_bits``. Same gate, same proof."""
        nw = sp.shape[1]
        for i in prange(n):
            oi = i * m_rows
            si = rp[i]
            for j in range(m_rows):
                acc = 0.0
                hita = 0.0
                hitb = 0.0
                for w in range(nw):
                    gi = sp[i, w]
                    gj = sq[j, w]
                    mask = gi & gj
                    if mask == np.uint64(0):
                        continue
                    ra = ka[i, w]
                    rb = kb[j, w]
                    while mask:
                        bit = mask & (~mask + np.uint64(1))
                        lo = bit - np.uint64(1)
                        pa = ra + _popcount(gi & lo)
                        pb = rb + _popcount(gj & lo)
                        va = dp[pa]
                        vb = dq[pb]
                        sm = va + vb
                        mm = 0.5 * sm
                        if wc and abs(va - vb) <= _TAU * sm:
                            u2 = ((va - vb) / sm) ** 2
                            acc += mm * u2 * _phi_over_u2(u2)
                        else:
                            acc += PLp[pa] + PLq[pb] - 2.0 * mm * np.log(mm)
                        hita += va
                        hitb += vb
                        mask ^= bit
                one = (si - hita) + (rq[j] - hitb)
                v = 0.5 * (acc + LN2 * one) * inv
                if v < 0.0:
                    v = 0.0
                if not squared:
                    v = np.sqrt(v)
                out[oi + j] = v

    @njit(**_JIT)
    def _sparse_cross(dp, ip, pp, PLp, rp, sp, dq, iq, pq, PLq, rq, sq,
                      n, m_rows, inv, squared, out, wc):
        for i in prange(n):
            oi = i * m_rows
            ai0 = pp[i]
            ai1 = pp[i + 1]
            si = rp[i]
            gi = sp[i]
            nw = sp.shape[1]
            for j in range(m_rows):
                acc = 0.0
                hita = 0.0
                hitb = 0.0
                shared = False
                for w in range(nw):
                    if gi[w] & sq[j, w]:
                        shared = True
                        break
                if shared:
                    aj0 = pq[j]
                    aj1 = pq[j + 1]
                    pa = ai0
                    pb = aj0
                    while pa < ai1 and pb < aj1:
                        ca = ip[pa]
                        cb = iq[pb]
                        if ca < cb:
                            pa += 1
                        elif cb < ca:
                            pb += 1
                        else:
                            va = dp[pa]
                            vb = dq[pb]
                            s = va + vb
                            m = 0.5 * s
                            if wc and abs(va - vb) <= _TAU * s:
                                u2 = ((va - vb) / s) ** 2
                                acc += m * u2 * _phi_over_u2(u2)
                            else:
                                acc += PLp[pa] + PLq[pb] - 2.0 * m * np.log(m)
                            hita += va
                            hitb += vb
                            pa += 1
                            pb += 1
                one = (si - hita) + (rq[j] - hitb)
                v = 0.5 * (acc + LN2 * one) * inv
                if v < 0.0:
                    v = 0.0
                if not squared:
                    v = np.sqrt(v)
                out[oi + j] = v


# --------------------------------------------------------------------------
# host-side wrappers
# --------------------------------------------------------------------------

def xlogx_like_kernels(P):
    """``x ln x`` using whichever ``log`` this module's kernels use.

    Falls back to the NumPy version when Numba is absent, which is consistent:
    with no Numba kernels there is nothing for it to disagree with.
    """
    if _HAVE:
        return _xlogx_shaped(P)
    from ._common import xlogx
    return xlogx(P)


def _offsets(n, condensed, dtype=np.int64):
    """Flat write offset per row, for the full matrix or the condensed form."""
    if condensed:
        starts = np.zeros(n, dtype=dtype)
        if n:
            starts[1:] = np.cumsum(np.arange(n - 1, 0, -1, dtype=dtype))
        return starts - (np.arange(n, dtype=dtype) + 1)
    return np.arange(n, dtype=dtype) * n


def _balanced_order(n, dtype=np.int64):
    """Rows as 0, n-1, 1, n-2, ... -- see ``_dense_self``."""
    order = np.empty(n, dtype=dtype)
    order[0::2] = np.arange((n + 1) // 2, dtype=dtype)
    order[1::2] = np.arange(n - 1, (n - 1) // 2, -1, dtype=dtype)
    return order


def csr_from_dense(P):
    """CSR triplet plus ``x ln x`` of the stored entries.

    Columns come out in increasing order per row, which the merge loops
    rely on.
    """
    indptr = np.zeros(P.shape[0] + 1, dtype=np.int64)
    if _HAVE:
        counts = _csr_counts_nb(P, np.empty(P.shape[0], dtype=np.int64))
        np.cumsum(counts, out=indptr[1:])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=P.dtype)
        _csr_fill_nb(P, indptr, indices, data)
    else:
        nz = P > 0
        np.cumsum(nz.sum(axis=1), out=indptr[1:])
        cols = np.broadcast_to(np.arange(P.shape[1], dtype=np.int64), P.shape)
        indices = np.ascontiguousarray(cols[nz])
        data = np.ascontiguousarray(P[nz])
    # PL must use the kernels' log, not NumPy's -- see ``_xlogx_nb``.
    return data, indices, indptr, xlogx_like_kernels(data)


def matrix(P, Q, inv, squared, out, wc=False):
    n = P.shape[0]
    PL = xlogx_like_kernels(P)
    if Q is None:
        if out is None:
            out = np.empty((n, n), dtype=P.dtype)
        _dense_self(P, PL, inv, squared, out.reshape(-1), _offsets(n, False),
                    _balanced_order(n), True, wc)
        np.fill_diagonal(out, 0.0)
        return out
    m = Q.shape[0]
    if out is None:
        out = np.empty((n, m), dtype=P.dtype)
    _dense_cross(P, PL, Q, xlogx_like_kernels(Q), inv, squared,
                 out.reshape(-1), wc)
    return out


def pdist(P, inv, squared, out, wc=False):
    n = P.shape[0]
    if out is None:
        out = np.empty(n * (n - 1) // 2, dtype=P.dtype)
    _dense_self(P, xlogx_like_kernels(P), inv, squared, out,
                _offsets(n, True),
                _balanced_order(n), False, wc)
    return out


def matrix_sparse(P, Q, inv, squared, out, wc=False):
    n = P.shape[0]
    data, idx, ptr, PL = csr_from_dense(P)
    if Q is None:
        if out is None:
            out = np.empty((n, n), dtype=P.dtype)
        sig = _signatures(idx, ptr, P.shape[1])
        if _unfolded(P.shape[1]) and _prefer_bits(sig, ptr):
            _sparse_self_bits(data, PL, _row_sums(data, ptr), sig,
                              _ranks(sig, ptr), inv, squared, out.reshape(-1),
                              _offsets(n, False), _balanced_order(n), True, wc)
        else:
            _sparse_self(data, idx, ptr, PL, _row_sums(data, ptr), sig, inv,
                         squared, out.reshape(-1),
                         _offsets(n, False), _balanced_order(n), True, wc)
        np.fill_diagonal(out, 0.0)
        return out
    m = Q.shape[0]
    dq, iq, pq, PLq = csr_from_dense(Q)
    if out is None:
        out = np.empty((n, m), dtype=P.dtype)
    sp = _signatures(idx, ptr, P.shape[1])
    sq = _signatures(iq, pq, Q.shape[1])
    if (_unfolded(P.shape[1]) and _unfolded(Q.shape[1])
            and _prefer_bits(sp, ptr)):
        _sparse_cross_bits(data, PL, _row_sums(data, ptr), sp,
                           _ranks(sp, ptr),
                           dq, PLq, _row_sums(dq, pq), sq, _ranks(sq, pq),
                           n, m, inv, squared, out.reshape(-1), wc)
    else:
        _sparse_cross(data, idx, ptr, PL, _row_sums(data, ptr), sp,
                      dq, iq, pq, PLq, _row_sums(dq, pq), sq,
                      n, m, inv, squared, out.reshape(-1), wc)
    return out


def pdist_sparse(P, inv, squared, out, wc=False):
    n = P.shape[0]
    data, idx, ptr, PL = csr_from_dense(P)
    if out is None:
        out = np.empty(n * (n - 1) // 2, dtype=P.dtype)
    sig = _signatures(idx, ptr, P.shape[1])
    if _unfolded(P.shape[1]) and _prefer_bits(sig, ptr):
        _sparse_self_bits(data, PL, _row_sums(data, ptr), sig,
                          _ranks(sig, ptr), inv, squared, out,
                          _offsets(n, True), _balanced_order(n), False, wc)
    else:
        _sparse_self(data, idx, ptr, PL, _row_sums(data, ptr), sig, inv,
                     squared, out, _offsets(n, True), _balanced_order(n),
                     False, wc)
    return out


def set_num_threads(n: int | None) -> None:
    """Bound the kernel thread count (``None`` restores Numba's default)."""
    if not _HAVE:
        return
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS if n is None else n)
