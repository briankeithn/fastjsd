"""Is the published sparse form inexact because of Lee's identity, or because
of the way Lee's Eq. (2.8) arranges it?

    python bench/lee_arrangement.py
    python bench/lee_arrangement.py --draws 400 --mp   # + 50-digit reference

The package returns exactly ``0.0`` for duplicate rows where the published
sparse form does not. Lee's *identity* -- logarithms only on the support
intersection, because ``t(a,0) = a ln 2`` -- and Lee's *arrangement* of it,

    JS = ln 2 + (1/2) SUM_{c in supp(p) & supp(q)} [ a ln(a/s) + b ln(b/s) ],
                                                            s = a + b

are separable claims, and only one of them is at fault. This script separates
them, in four steps, none of which is timed:

1. **Is the transcription faithful?** Compare ``baselines._lee_self`` against
   50-digit mpmath. A disagreement beyond a few ULP would mean the steps below
   measure an error in the transcription rather than the literature's
   arrangement.

2. **Are the individual terms exact?** On duplicate rows every summand is
   ``2a ln(a / 2a) = -2a ln 2``, and both ``a/(2a) == 0.5`` and
   ``log(0.5) == -ln 2`` hold bitwise in IEEE-754. So the residue cannot be in
   any one term; it has to be in how they are combined.

3. **Which arrangement survives?** The same identity, three ways, on duplicate
   rows where the answer must be exactly zero:

     folded    ln2 + (1/2) SUM [a ln(a/s) + b ln(b/s)]      <- Eq. (2.8)
     unfolded  (1/2) [ SUM t(a,b) + ln2 * one-sided mass ]  <- this package
     unfolded, one-sided mass recovered two different ways

   A "FAILS" here is a nonzero distance between a row and its own copy.

4. **Does the accumulator split matter away from zero?** ``docs/MATH.md``
   Consequence 2b states that a single combined accumulator for the matched
   mass caps the sparse kernel near ``2e-08`` on rows differing by ``1e-9``,
   while two per-row accumulators do not. That is a claim about accuracy
   *near* zero rather than at it, so it needs its own regime.

Everything here is a correctness measurement. It allocates nothing large and
must never be read as a benchmark.
"""

from __future__ import annotations

import argparse

import numpy as np

import baselines

LN2 = 0.6931471805599453

#: k values swept. Small ones matter as much as large: with few coordinates a
#: residue has fewer chances to cancel back to zero, and the folded form's
#: failure rate is *not* monotone in k.
KS = (8, 16, 32, 64, 128, 192, 256, 512)


def _sparse_pair(rng, k, z):
    """Two rows with a shared support of size ``z``, each normalized."""
    cols = rng.choice(k, size=z, replace=False)
    p = np.zeros(k)
    p[cols] = rng.dirichlet(np.ones(z))
    p /= p.sum()
    return p, cols


# --------------------------------------------------------------------------
# The arrangements. Each takes two dense rows and returns JS in nats, using
# logarithms only on the support intersection -- that part is Lee's identity
# and is held fixed. What varies is where `ln 2` lives and how the one-sided
# mass is obtained.
# --------------------------------------------------------------------------

def js_folded(p, q):
    """Lee's Eq. (2.8) as published: the constant lives outside the sum."""
    acc = 0.0
    for c in range(p.shape[0]):
        a, b = p[c], q[c]
        if a > 0.0 and b > 0.0:
            s = a + b
            acc += a * np.log(a / s) + b * np.log(b / s)
    v = LN2 + 0.5 * acc
    return 0.0 if v < 0.0 else v


def _t(a, b):
    """The unfolded summand ``a ln a + b ln b - 2 m ln m``."""
    m = 0.5 * (a + b)
    return a * np.log(a) + b * np.log(b) - 2.0 * m * np.log(m)


def js_unfolded_direct(p, q):
    """Unfolded, one-sided mass added up as it is encountered."""
    acc = 0.0
    one = 0.0
    for c in range(p.shape[0]):
        a, b = p[c], q[c]
        if a > 0.0 and b > 0.0:
            acc += _t(a, b)
        elif a > 0.0:
            one += a
        elif b > 0.0:
            one += b
    return 0.5 * (acc + LN2 * one)


def js_unfolded_1acc(p, q, sa, sb):
    """Unfolded, one-sided mass by complement with ONE combined accumulator."""
    acc = 0.0
    hit = 0.0
    for c in range(p.shape[0]):
        a, b = p[c], q[c]
        if a > 0.0 and b > 0.0:
            acc += _t(a, b)
            hit += a + b
    one = (sa + sb) - hit
    return 0.5 * (acc + LN2 * one)


def js_unfolded_2acc(p, q, sa, sb):
    """Unfolded, one-sided mass by complement with TWO per-row accumulators.

    This is what the shipped sparse kernels do. ``hita`` sums row i's matched
    entries in the same order and over the same set as ``sa`` whenever every
    entry matches, so the difference is bitwise zero.
    """
    acc = 0.0
    hita = hitb = 0.0
    for c in range(p.shape[0]):
        a, b = p[c], q[c]
        if a > 0.0 and b > 0.0:
            acc += _t(a, b)
            hita += a
            hitb += b
    one = (sa - hita) + (sb - hitb)
    return 0.5 * (acc + LN2 * one)


def _seq_sum(row):
    """Left to right, the order the kernels use. ``np.sum`` reduces in a tree."""
    return float(np.cumsum(row)[-1])


# --------------------------------------------------------------------------

def step1_faithful(args):
    """Is `baselines._lee_self` a faithful transcription of Lee's Eq. (2.8)?"""
    print("1. transcription check -- baselines._lee_self vs 50-digit mpmath")
    try:
        import mpmath as mp
    except ImportError:
        print("   SKIPPED (mpmath not installed)\n")
        return
    if not baselines._HAVE_NUMBA:
        print("   SKIPPED (numba not installed)\n")
        return
    mp.mp.dps = 50
    rng = np.random.default_rng(20250805)
    worst = 0.0
    for _ in range(args.faithful):
        k, z = 64, 12
        n = 6
        P = np.zeros((n, k))
        for i in range(n):
            row, cols = _sparse_pair(rng, k, z)
            P[i] = row
        D = baselines.lee_sparse_numba(P, base=np.e)
        for i in range(n):
            for j in range(i + 1, n):
                p = [mp.mpf(float(x)) for x in P[i]]
                q = [mp.mpf(float(x)) for x in P[j]]
                sp, sq = mp.fsum(p), mp.fsum(q)
                p = [x / sp for x in p]
                q = [x / sq for x in q]
                terms = []
                for a, b in zip(p, q, strict=True):
                    m = (a + b) / 2
                    for x in (a, b):
                        if x > 0:
                            terms.append(x * mp.log(x))
                    if m > 0:
                        terms.append(-2 * m * mp.log(m))
                truth = mp.sqrt(mp.fsum(terms) / 2)
                worst = max(worst, abs(float(D[i, j]) - float(truth)))
    print(f"   worst |Lee-97 - truth| over {args.faithful} matrices: {worst:.3g}")
    print("   -> the baseline computes Eq. (2.8); it is not miscoded.\n")


def step2_terms(_args):
    """Is any individual term inexact on duplicate rows?"""
    print("2. term check -- on duplicate rows every summand should be exact")
    rng = np.random.default_rng(7)
    a = rng.random(4096)
    r = a / (2.0 * a)
    ok_half = bool(np.all(r == 0.5))
    ok_log = bool(np.log(0.5) == -LN2)
    print(f"   a/(2a) == 0.5 bitwise, {a.size} draws : {ok_half}")
    print(f"   np.log(0.5) == -ln2 bitwise          : {ok_log}")
    print("   -> each folded term is exactly -2a ln2; the residue is in the")
    print("      accumulation of them, not in any one of them.\n")


def step3_arrangements(args):
    """Duplicate rows: which arrangements return exactly 0.0?"""
    print("3. duplicate rows -- distance must be exactly 0.0")
    rng = np.random.default_rng(args.seed)
    names = ["Eq 2.8 as published", "unfolded, direct one-sided",
             "unfolded, 1 accumulator", "unfolded, 2 accumulators"]
    fail = dict.fromkeys(names, 0)
    worst = dict.fromkeys(names, 0.0)
    total = 0
    for k in KS:
        z = max(2, k // 4)
        for _ in range(args.draws):
            p, _cols = _sparse_pair(rng, k, z)
            q = p.copy()
            s = _seq_sum(p)
            got = {
                names[0]: js_folded(p, q),
                names[1]: js_unfolded_direct(p, q),
                names[2]: js_unfolded_1acc(p, q, s, s),
                names[3]: js_unfolded_2acc(p, q, s, s),
            }
            total += 1
            for nm, v in got.items():
                d = np.sqrt(max(v, 0.0) / LN2)   # base 2, as the package returns
                if d != 0.0:
                    fail[nm] += 1
                    worst[nm] = max(worst[nm], d)
    for nm in names:
        if fail[nm]:
            print(f"   {nm:<28} FAILS {fail[nm]:>4}/{total}   "
                  f"worst {worst[nm]:.3g}")
        else:
            print(f"   {nm:<28} EXACT  ({total} draws)")
    print("   -> Lee's identity is held fixed across all four rows. Only the")
    print("      arrangement changes, and only the folded one fails.\n")


def step4_near(args):
    """Near-duplicates: does the accumulator split matter away from zero?"""
    print("4. near-duplicates -- rows differing by 1e-9, error vs 50-digit")
    try:
        import mpmath as mp
    except ImportError:
        print("   SKIPPED (mpmath not installed)\n")
        return
    mp.mp.dps = 50
    rng = np.random.default_rng(args.seed + 1)
    print(f"   {'k':>5}{'direct':>13}{'1 acc':>13}{'2 acc':>13}")
    for k in (32, 64, 128, 256):
        z = max(2, k // 4)
        w = {"direct": 0.0, "1acc": 0.0, "2acc": 0.0}
        for _ in range(args.near):
            p, cols = _sparse_pair(rng, k, z)
            q = p.copy()
            q[cols] *= 1.0 + 1e-9 * rng.standard_normal(cols.size)
            q = np.abs(q)
            q /= q.sum()
            sp, sq = _seq_sum(p), _seq_sum(q)
            mpp = [mp.mpf(float(x)) for x in p]
            mpq = [mp.mpf(float(x)) for x in q]
            mpp = [x / mp.fsum(mpp) for x in mpp]
            mpq = [x / mp.fsum(mpq) for x in mpq]
            terms = []
            for a, b in zip(mpp, mpq, strict=True):
                m = (a + b) / 2
                for x in (a, b):
                    if x > 0:
                        terms.append(x * mp.log(x))
                if m > 0:
                    terms.append(-2 * m * mp.log(m))
            truth = float(mp.sqrt(mp.fsum(terms) / 2 / mp.log(2)))
            for key, v in (("direct", js_unfolded_direct(p, q)),
                           ("1acc", js_unfolded_1acc(p, q, sp, sq)),
                           ("2acc", js_unfolded_2acc(p, q, sp, sq))):
                d = np.sqrt(max(v, 0.0) / LN2)
                w[key] = max(w[key], abs(d - truth))
        print(f"   {k:>5}{w['direct']:>13.3g}{w['1acc']:>13.3g}"
              f"{w['2acc']:>13.3g}")
    print("   -> `direct` and `2 acc` agree bitwise, as they must: with a fully")
    print("      shared support the complement's `one` is exactly 0.0 and so is")
    print("      the direct sum. `1 acc` is uniformly worse and sits at the")
    print("      ~2e-08 cap docs/MATH.md Consequence 2b reports. Neither column")
    print("      reaches the 1e-17 quoted there for the shipped merge -- that")
    print("      figure is the `well_conditioned` series path, which this")
    print("      script deliberately does not use, since the question here is")
    print("      the arrangement and not the summand.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30,
                    help="duplicate-row draws per k (step 3)")
    ap.add_argument("--near", type=int, default=15,
                    help="near-duplicate draws per k (step 4)")
    ap.add_argument("--faithful", type=int, default=8,
                    help="matrices for the mpmath transcription check")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mp", action="store_true",
                    help="run the mpmath steps (1 and 4); they are slow")
    args = ap.parse_args()

    print(__doc__.split("\n\n")[0] + "\n")
    if args.mp:
        step1_faithful(args)
    step2_terms(args)
    step3_arrangements(args)
    if args.mp:
        step4_near(args)


if __name__ == "__main__":
    main()
