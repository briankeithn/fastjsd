"""Do the synthetic inputs look like the real ones where it matters?

    python bench/distributions.py                # the comparison table
    python bench/distributions.py --rho-sweep    # E|I| invariance under rho
    python bench/distributions.py --zipf-sweep   # Phi vs s, the k d^2 failure

`bench/datasets.py` builds its sparse cases with `sparse_rows`, which places
nonzeros at uniformly random positions. Real supports are not uniform, and the
question this harness settles is which of the differences the sparse kernel can
actually feel.

Summing matched coordinates by column rather than by pair gives, with no model
at all,

    E|I| = sum_c m_c (m_c - 1) / (n (n - 1)),   m_c = rows nonzero in column c

so the mean intersection is a function of the COLUMN DEGREES alone. Row-to-row
support correlation -- documents sharing vocabulary, points sharing a cluster --
cannot move it, because it does not move a column degree. ``k d^2`` is exact iff
the column degrees are all equal, and the only inflation is the concentration
factor ``Phi = sum_c (m_c/n)^2 / (k d^2) >= 1``. So the family that breaks the
``k d^2`` estimate is skewed vocabulary, not clustered documents, and the `Phi`
column below measures which of the two a given input has.

What correlation moves instead is the SPREAD, and the kernel is assembled out
of things that see the spread and not the mean: the occupancy mask fires on
``|I| == 0``, the merge walks ``s_i + s_j - |I|``, and `_balanced_order`
balances pair count rather than work. `sparse_rows` reproduces none of that
spread, because it gives every row the same ``nnz``.

Everything here is an array statistic -- counts of nonzeros, of matched
coordinates, of masked-off pairs. There is no clock in this file and nothing it
prints is a speed. That is deliberate: these are the numbers a timing run
should be tuned AGAINST, and they can be produced on a busy machine, which a
timing run cannot.

Columns
-------
dens          nonzero fraction over the NOMINAL k. Dispatch runs after
              `_drop_empty_columns`, so it sees ``nnz / (n keff)``, which is
              larger whenever `keff` below is short of `k`.
E|I|, k d^2   measured mean intersection and the claimed prediction
Phi           their ratio; > 1 means skewed column degrees
med, p99      percentiles of |I|; the mean hides the bimodality entirely
skip          pairs a 64-bit occupancy signature masks off, with column c
              folded onto bit ``c % 64``
null          the same, after a per-column shuffle -- which preserves every
              column degree exactly, hence E|I| exactly, so any gap between
              `skip` and `null` is row-level structure the mean cannot see. It
              is a null for ALL row structure, including constant nnz, which is
              why `sparse_rows` scores BELOW its own null: the shuffle gives
              rows a spread of nnz and short rows skip more.
disj          pairs that truly share no coordinate. `disj - skip` is what the
              mod-64 fold throws away; it is exactly zero for k <= 64, where
              two columns cannot collide.
walk          merge steps actually executed per pair, E[(s_i+s_j-|I|) if !skip]
cv            std(nnz)/mean(nnz) over rows; `sparse_rows` is 0.00 by
              construction, real membership data is 1.4-1.6
top1%         share of all logarithms carried by the busiest 1 % of pairs
dispatch      what `resolve_backend("auto", .)` picks on THIS host, after
              `_drop_empty_columns`, which is why `keff` is reported too

MEASUREMENTS THAT STILL NEED A CLOCK
------------------------------------
Serially, on an idle box. Every item below is a number this file can predict
the direction of but not the size of.

1. `SPARSE_DENSITY_MAX` under CORRELATED supports. The shipped crossover sweep
   (`bench/portable.py --crossover`) varies `sparse_rows`, whose signature skip
   rate goes as ``exp(-z^2 / 64)`` in the nonzeros per row and is therefore
   already inert at the densities where the crossover itself sits. The constant
   is tuned on input that switches one of the sparse kernel's two optimizations
   off, then applied to data where that optimization fires on the majority of
   pairs. Re-run the crossover on `correlated_rows` at the fixtures' own k (25
   and 40, where the fold is exact) with rho tuned to reproduce their measured
   skip rate, and report the crossover per (k, Phi, skip) rather than as one
   scalar. Prediction, stated in advance: `SPARSE_DENSITY_MAX` moves by <=
   0.03, because the log term is identical in both kernels and cancels from the
   comparison, leaving ``k`` scan steps against ``(2d - d^2) k`` merge steps --
   a kernel property.

2. `SVML_DENSITY_MAX` against the thresholded 931x25 fixture, whose density
   lands close to it. This is the constant genuinely at risk: the SVML kernel
   does ``k`` logarithms per pair whatever the density, so unlike the libm
   crossover neither ``|I|`` nor the skip rate cancels from the comparison, and
   that fixture skips 59 % of its pairs. Its crossover is therefore a function
   of the skip rate rather than of density alone, and locating it needs an
   input whose skip rate matches the target data.

3. `zipf_rows(s=1.0)` versus `sparse_rows` at identical density. Phi = 6.5
   means the merge does 6.5x the logarithms while the skip rate falls from
   about 36 % to about 6 %. If the merge still wins there, the sparse advantage
   is robust to vocabulary skew; if it does not, `SPARSE_DENSITY_MAX` is not a
   function of density alone.

4. `mixed_density(shuffle=True)` versus `shuffle=False`. Identical work,
   identical dispatch, different `_balanced_order` chunking. Any gap is pure
   scheduling loss and it is invisible to every current benchmark input.

5. `near_duplicates` and `with_duplicates` with `well_conditioned` on and off.
   It is on by default for the merge because matched coordinates are rare
   there and the branch almost never fires, an argument measured on inputs
   where the ill-conditioned pairs are a small fraction of the total. These
   generators make that fraction a knob, and the branch is a predicted one, so
   the cost of a high rate is not a small extrapolation from a low one.

6. `threshold(P, factor)` on a `row_mass` input. The footer below shows the
   density curve has an interior MINIMUM, and that both real fixtures put it at
   ``factor=1.0`` -- the value `suite()` uses. So the suite's two sparse cases
   sit where a small change to the cleanup moves the density most, and on the
   931x25 fixture the neighbouring factors sit on the other side of
   `SVML_DENSITY_MAX`. One timed sweep over factor, on one fixture, prices the
   flip.

7. `mixed_density(shuffle=True)` against itself with `shuffle=False` is the
   only comparison in this list that this harness cannot pre-empt at all:
   every statistic here is permutation-invariant, and the two inputs differ
   only by a row permutation.
"""

from __future__ import annotations

import argparse

import numpy as np

from fastjsd import resolve_backend

import datasets

try:
    from scipy.spatial.distance import pdist as _chebyshev_pdist
except ImportError:                                     # pragma: no cover
    _chebyshev_pdist = None

#: Rows sampled for the O(n^2 k) value statistics (duplicate detection, the
#: well_conditioned branch rate). Support statistics are always exact and use
#: every row; only these two are subsampled, and the caps are printed.
NEAR_ROWS = 1200
WC_ROWS = 400

#: `_numba._TAU`, the band the series branch covers. Kept as a literal rather
#: than imported so this file reads the same with numba absent.
TAU = 0.25

#: Chebyshev radius counted as a near-duplicate pair. `accuracy.py`'s
#: near-duplicate regime perturbs by 1e-9, well inside it.
NEAR_TOL = 1e-6


def _sig_words(k):
    """Signature width in 64-bit words, mirroring ``_numba._signatures``.

    The kernel uses ``min(4, ceil(k / 64))`` words and folds columns modulo
    ``64 * nw``. Modelling a single word here would understate the skip rate
    for every ``k`` above 64 and would not match what the kernel computes.
    """
    return min(4, max(1, (int(k) + 63) // 64))


def _signature_fold(S):
    """Occupancy per 64-bit lane: column ``c`` sets bit ``c & 63``.

    Built in NumPy so this harness runs with numba absent.
    `check_signature_fold` compares it against the shipped
    `_numba._signatures` when numba is present.
    """
    nbits = 64 * _sig_words(S.shape[1])
    F = np.zeros((S.shape[0], nbits), dtype=bool)
    for c in range(S.shape[1]):
        F[:, c % nbits] |= S[:, c]
    return F


def _pair_counts(S, iu):
    """(|I| per pair, signature-collides per pair) for the boolean support S."""
    A = S.astype(np.float32)
    inter = (A @ A.T)[iu]
    F = _signature_fold(S).astype(np.float32)
    collide = (F @ F.T)[iu] > 0
    return inter, collide


def _shuffled_skip(S, iu, seed):
    """Skip rate after permuting each column independently.

    The null that matters: a per-column shuffle preserves every column degree,
    hence E|I| exactly, so anything it changes is row structure and nothing
    else. Comparing the MEAN against this null therefore returns a ratio of
    1.00 on every input, because the shuffle cannot move the mean at all. What
    it does move is the skip rate, by tens of percentage points on structured
    input, and that is what the comparison is for.
    """
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random(S.shape), axis=0)
    _, collide = _pair_counts(np.take_along_axis(S, idx, axis=0), iu)
    return float((~collide).mean())


def _duplicate_pairs(P, rows, seed):
    """Pairs at Chebyshev distance 0 and below ``NEAR_TOL``, on a row sample."""
    if _chebyshev_pdist is None:                        # pragma: no cover
        return None, None, 0
    n = P.shape[0]
    if n > rows:
        rng = np.random.default_rng(seed)
        P = P[np.sort(rng.choice(n, size=rows, replace=False))]
    d = _chebyshev_pdist(np.ascontiguousarray(P), "chebyshev")
    return int((d == 0.0).sum()), int((d < NEAR_TOL).sum()), P.shape[0]


def _wc_branch_rate(P, rows, seed):
    """Fraction of MATCHED coordinates that take the series branch.

    ``|a - b| <= TAU (a + b)`` is the test in the kernels. This is the number
    that decides whether `well_conditioned=True` is free: the branch replaces
    one logarithm with a short polynomial, so a low rate is cheap and a high
    rate is not. Chunked, on a row sample, because it is inherently O(n^2 k).
    """
    n = P.shape[0]
    if n > rows:
        rng = np.random.default_rng(seed)
        P = P[np.sort(rng.choice(n, size=rows, replace=False))]
    m = P.shape[0]
    matched = inband = 0
    for i0 in range(0, m, 16):
        i1 = min(i0 + 16, m)
        a, b = P[i0:i1, None, :], P[None, :, :]
        both = (a > 0) & (b > 0)
        both &= (np.arange(m)[None, :] > np.arange(i0, i1)[:, None])[:, :, None]
        matched += int(both.sum())
        inband += int((both & (np.abs(a - b) <= TAU * (a + b))).sum())
    return inband / matched if matched else 0.0


def profile(P, near_rows=NEAR_ROWS, wc_rows=WC_ROWS, seed=0, values=True):
    """Every structural statistic for one matrix.

    Nothing here is timed and no kernel runs; the only call into the package is
    `resolve_backend`, which reads the density and returns a name.
    """
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(axis=1, keepdims=True)
    n, k = P.shape
    S = P > 0
    nnz = S.sum(axis=1)
    deg = S.sum(axis=0).astype(np.float64)
    dens = float(nnz.sum()) / P.size

    iu = np.triu_indices(n, 1)
    inter, collide = _pair_counts(S, iu)
    npairs = inter.size
    order = np.argsort(inter)[::-1]
    total = float(inter.sum())
    cuts = [max(1, int(round(f * npairs))) for f in (0.01, 0.10)]
    walk = (nnz[iu[0]] + nnz[iu[1]] - inter) * collide

    # Dispatch sees the matrix AFTER empty columns are dropped, and a skewed
    # generator can leave columns nonzero in no row at all.
    keff = int(deg.astype(bool).sum())
    Peff = P[:, deg > 0]

    row = {
        "n": n, "k": k, "keff": keff, "dens": dens,
        "EI": float((deg * (deg - 1)).sum() / (n * (n - 1))),
        "kd2": k * dens * dens,
        "med": float(np.median(inter)), "p99": float(np.percentile(inter, 99)),
        "skip": float((~collide).mean()), "disj": float((inter == 0).mean()),
        "walk": float(walk.mean()),
        "cv": float(nnz.std() / nnz.mean()) if nnz.mean() else 0.0,
        "top1": float(inter[order[:cuts[0]]].sum() / total) if total else 0.0,
        "top10": float(inter[order[:cuts[1]]].sum() / total) if total else 0.0,
        "dispatch": resolve_backend("auto", Peff),
        "null": _shuffled_skip(S, iu, seed),
    }
    row["phi"] = row["EI"] / row["kd2"] if row["kd2"] else float("nan")
    if values:
        row["dup"], row["near"], row["sampled"] = _duplicate_pairs(
            P, near_rows, seed)
        row["wc"] = _wc_branch_rate(P, wc_rows, seed)
    return row


# --- the case list ---------------------------------------------------------
#
# Real fixtures first, then generators matched to a real case on (n, k,
# density) so every column is a like-for-like comparison rather than a
# comparison of two different sizes.

def cases():
    out = []
    for name, nnz_match, dt_head in (("soft_clusters_931x25", 6, 0.37),
                                     ("soft_clusters_1014x40", 5, 0.46)):
        P = datasets.real(name)
        n, k = P.shape
        out.append((f"real {n}x{k} dense", P))
        T = datasets.threshold(P)
        d = float(np.count_nonzero(T)) / T.size
        out.append((f"real {n}x{k} thr f=1.0", T))
        out.append((f"sparse_rows {n}x{k} nnz={nnz_match}",
                    datasets.sparse_rows(n, k, nnz_match, seed=11)))
        out.append((f"correlated {n}x{k} rho=0.0",
                    datasets.correlated_rows(n, k, d, rho=0.0, seed=12)))
        out.append((f"correlated {n}x{k} rho=0.9",
                    datasets.correlated_rows(n, k, d, rho=0.9, seed=13)))
        # Dense dominant-tail rows put through the same cleanup the real
        # fixture gets, which is the only way to compare the two shapes on
        # the path the data actually travels.
        out.append((f"dominant thr {n}x{k}",
                    datasets.threshold(
                        datasets.dominant_tail(n, k, k, head=dt_head,
                                               seed=14))))
        out.append((f"near_dup {n}x{k} g=50",
                    datasets.near_duplicates(n, k, 50, seed=15)))
        out.append((f"real {n}x{k} thr + 10% dup",
                    datasets.with_duplicates(T, frac=0.1, seed=16)))
    out.append(("dirichlet 2000x64 a=0.3", datasets.dirichlet(2000, 64, 0.3,
                                                              seed=1)))
    out.append(("dirichlet 2000x256 a=0.1", datasets.dirichlet(2000, 256, 0.1,
                                                               seed=2)))
    out.append(("sparse_rows 2000x256 nnz=8", datasets.sparse_rows(2000, 256, 8,
                                                                   seed=3)))
    out.append(("zipf 2000x256 nnz=8 s=1.0", datasets.zipf_rows(2000, 256, 8,
                                                                s=1.0, seed=4)))
    out.append(("zipf 2000x256 nnz=8 s=1.5", datasets.zipf_rows(2000, 256, 8,
                                                                s=1.5, seed=5)))
    out.append(("mixed 2000x64 50% dense shuf",
                datasets.mixed_density(2000, 64, 0.5, shuffle=True, seed=6)))
    out.append(("mixed 2000x64 50% dense sort",
                datasets.mixed_density(2000, 64, 0.5, shuffle=False, seed=6)))
    return out


HEAD = (f"{'case':<30}{'n':>6}{'k':>5}{'keff':>6}{'dens':>7}{'E|I|':>9}"
        f"{'k d^2':>9}{'Phi':>7}{'med':>6}{'p99':>6}{'skip':>8}{'null':>8}"
        f"{'disj':>8}{'walk':>8}{'cv':>6}{'top1%':>7}  dispatch")


def line(label, r):
    return (f"{label:<30}{r['n']:>6}{r['k']:>5}{r['keff']:>6}{r['dens']:>7.3f}"
            f"{r['EI']:>9.3f}{r['kd2']:>9.3f}{r['phi']:>7.2f}"
            f"{r['med']:>6.0f}{r['p99']:>6.0f}"
            f"{r['skip']:>7.1%} {r['null']:>7.1%} {r['disj']:>7.1%} "
            f"{r['walk']:>7.2f}{r['cv']:>6.2f}{r['top1']:>7.1%}  "
            f"{r['dispatch']}")


def check_signature_fold():
    """Confirm the NumPy fold here matches the kernel's, or say why it cannot."""
    try:
        from fastjsd import _numba
    except ImportError:                                 # pragma: no cover
        return "numba absent; signature fold not cross-checked"
    if not _numba.available():
        return "numba absent; signature fold not cross-checked"
    k = 64 * 4 + 100                     # above 64 * nw, so the fold is live
    P = datasets.sparse_rows(64, k, 5, seed=99)
    _, idx, ptr, _ = _numba.csr_from_dense(P)
    F = _signature_fold(P > 0)
    nw = F.shape[1] // 64
    bits = np.uint64(1) << np.arange(64, dtype=np.uint64)
    mine = np.stack(
        [np.bitwise_or.reduce(
            np.where(F[:, w * 64:(w + 1) * 64], bits, np.uint64(0)), axis=1)
         for w in range(nw)], axis=1)
    theirs = _numba._signatures(idx, ptr, k)
    ok = bool(mine.shape == theirs.shape and np.array_equal(mine, theirs))
    return (f"signature fold matches _numba._signatures at k={k}, "
            f"{nw} words: {ok}")


def main_table(args):
    rows = []
    print("\nstructural statistics -- counts, not milliseconds\n")
    print(HEAD)
    print("-" * len(HEAD.rstrip()))
    for label, P in cases():
        r = profile(P, near_rows=args.near_rows, wc_rows=args.wc_rows,
                    values=not args.no_values)
        rows.append((label, r))
        print(line(label, r))

    print("\nwhat the mod-64 fold costs (k > 64 only)")
    print(f"{'case':<30}{'disj':>9}{'skip':>9}{'lost':>9}")
    for label, r in rows:
        if r["keff"] > 64:
            lost = 100.0 * (r["disj"] - r["skip"])
            print(f"{label:<30}{r['disj']:>8.1%} {r['skip']:>8.1%} "
                  f"{lost:>7.1f} pp")
    print("A skippable pair the signature misses is not wrong, only slow: the")
    print("fold produces false positives only. It costs a merge that finds")
    print("nothing, and the loss is a function of nnz/row alone -- exp(-z^2/64)")
    print("-- so it is identical to what a k=64 matrix would lose, whatever k is.")

    if not args.no_values:
        print("\nthe well_conditioned regime, on a row sample")
        print(f"{'case':<30}{'rows':>7}{'dup pairs':>11}{'near pairs':>12}"
              f"{'wc branch':>11}")
        for label, r in rows:
            if r["dup"] is None:
                continue
            print(f"{label:<30}{r['sampled']:>7}{r['dup']:>11}{r['near']:>12}"
                  f"{r['wc']:>10.1%}")
        print(f"dup = Chebyshev distance exactly 0; near = below {NEAR_TOL:g}.")
        print("wc branch = matched coordinates with |a-b| <= 0.25 (a+b), the")
        print("fraction of the merge's logarithms the series replaces. It is")
        print("an average over ALL pairs, so a `near_dup` row is low because")
        print("only 1/groups of its pairs are the near ones.")
        print("The thresholded fixtures' duplicates are not planted: see")
        print("`threshold` in datasets.py and the footer below.")

    threshold_footer()

    print("\nThe two `mixed` rows are the same matrix in two row orders. Every")
    print("statistic in this file is permutation-invariant, which is exactly")
    print("why the scheduling loss `_balanced_order` can suffer on that input")
    print("is invisible here and needs the operator's clock.")
    return rows


def threshold_footer():
    """What the standard cleanup does to density, and so to dispatch.

    `threshold` drops entries below ``factor/k`` and renormalizes, and resets a
    row the cut empties to uniform ``1/k`` -- which is ``k`` nonzeros. So a
    HARSHER cut can raise the density, and the density is the only thing
    dispatch looks at.

    The rows that get emptied are not the flat ones. The comparison is against
    RAW values, before normalization, and the fixtures ship un-normalized with
    row sums spanning 0.077-1.0: at ``factor=1.0``, 138 of the 931 rows have no
    raw entry above ``1/k`` at all. Normalized generators cannot reproduce
    that, which is what `row_mass` is for.
    """
    dirich = datasets.dirichlet(931, 25, 0.3, seed=40)
    domin = datasets.dominant_tail(931, 25, 25, head=0.37, seed=41)
    shapes = [
        ("real 931x25", datasets.real("soft_clusters_931x25")),
        ("real 1014x40", datasets.real("soft_clusters_1014x40")),
        ("dirichlet 931x25 a=0.3", dirich),
        # spread=4 is picked to reproduce the SHAPE of the fixtures' curve, not
        # their density: their raw sums are 0.077-1.0 but median 0.742, so most
        # rows sit near the top of that range.
        ("  + row_mass spread=4", datasets.row_mass(dirich, 4.0, seed=42)),
        ("dominant 931x25 head=0.37", domin),
        ("  + row_mass spread=4", datasets.row_mass(domin, 4.0, seed=43)),
    ]
    factors = (0.5, 1.0, 2.0, 4.0)
    print("\ndensity after threshold(P, factor), and the dispatch it implies")
    print(f"{'shape':<28}" + "".join(f"{f'factor={f}':>21}" for f in factors))
    for label, P in shapes:
        cells = []
        for f in factors:
            T = datasets.threshold(P, f)
            d = float(np.count_nonzero(T)) / T.size
            cells.append(f"{d:.3f} {resolve_backend('auto', T)}".rjust(21))
        print(f"{label:<28}" + "".join(cells))
    print("Monotone decreasing on every normalized input. Once rows carry")
    print("unequal mass the empty-row reset takes over and density RISES with")
    print("factor, so where the two effects cross the curve has an interior")
    print("minimum -- and on both real fixtures that minimum is at factor=1.0,")
    print("the value `suite()` uses. The suite's two sparse cases sit at the")
    print("least stable point of the cleanup that produced them. The cause is")
    print("not the value shape: it is `threshold` comparing UN-NORMALIZED")
    print("values against a fixed 1/k, which no generator here can trigger")
    print("without `row_mass`.")


def rho_sweep(args):
    """Correlation holds E|I| fixed and moves everything the kernel sees."""
    n, k, d = args.n, 40, 0.12
    print(f"\ncorrelated_rows n={n} k={k} d={d}, w=d -- E|I| must not move")
    print(f"{'rho':>6}{'dens':>8}{'E|I|':>9}{'k d^2':>9}{'Phi':>7}{'skip':>8}"
          f"{'walk':>8}{'E|I| given not skipped':>24}")
    for rho in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        P = datasets.correlated_rows(n, k, d, rho=rho, seed=20)
        r = profile(P, values=False)
        live = r["EI"] / (1.0 - r["skip"]) if r["skip"] < 1.0 else float("nan")
        print(f"{rho:>6.2f}{r['dens']:>8.3f}{r['EI']:>9.3f}{r['kd2']:>9.3f}"
              f"{r['phi']:>7.2f}{r['skip']:>7.1%} {r['walk']:>8.2f}"
              f"{live:>24.2f}")
    print("The mean intersection is a function of the column degrees, which a")
    print("row-level correlation cannot change. The merge steps and the skip")
    print("rate are not, and they are what the kernel is built out of.")


def zipf_sweep(args):
    """Skewed column popularity is the one axis that breaks k d^2."""
    n, k, nnz = args.n, 256, 8
    print(f"\nzipf_rows n={n} k={k} nnz={nnz} -- density fixed, Phi is not")
    print(f"{'s':>6}{'keff':>7}{'dens':>8}{'E|I|':>9}{'k d^2':>9}{'Phi':>8}"
          f"{'skip':>8}{'disj':>8}{'walk':>8}")
    for s in (0.0, 0.3, 0.6, 1.0, 1.5, 2.0):
        P = datasets.zipf_rows(n, k, nnz, s=s, seed=30)
        r = profile(P, values=False)
        print(f"{s:>6.1f}{r['keff']:>7}{r['dens']:>8.4f}{r['EI']:>9.3f}"
              f"{r['kd2']:>9.3f}{r['phi']:>8.2f}{r['skip']:>7.1%} "
              f"{r['disj']:>7.1%} {r['walk']:>8.2f}")
    print("Phi = sum_c (m_c/n)^2 / (k d^2) >= 1, equality iff the column")
    print("degrees are equal. s=0 is sparse_rows, and it is the only row of")
    print("this table in which the package's k d^2 claim is exact.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-sweep", action="store_true",
                    help="E|I| invariance under support correlation")
    ap.add_argument("--zipf-sweep", action="store_true",
                    help="Phi against the column-popularity exponent")
    ap.add_argument("--n", type=int, default=1200,
                    help="rows in the sweeps (the main table is fixed)")
    ap.add_argument("--near-rows", type=int, default=NEAR_ROWS,
                    help="row cap for the O(n^2 k) duplicate scan")
    ap.add_argument("--wc-rows", type=int, default=WC_ROWS,
                    help="row cap for the well_conditioned branch rate")
    ap.add_argument("--no-values", action="store_true",
                    help="support statistics only; skip the value statistics")
    args = ap.parse_args()

    print(check_signature_fold())
    if args.rho_sweep:
        rho_sweep(args)
    elif args.zipf_sweep:
        zipf_sweep(args)
    else:
        main_table(args)
