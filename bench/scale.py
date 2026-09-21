"""Fit the cost model instead of asserting it.

    python bench/scale.py --dry-run       # grids, memory, mask rates; times nothing
    python bench/scale.py                 # the sweep (IDLE MACHINE ONLY)
    python bench/scale.py --sweep n --json scale.json

`bench/datasets.py::suite()` traces a path through (n, k, density) rather
than a grid: its synthetic block fixes n and varies only k and density, and
its two real fixtures have shapes dictated by the corpora they came from, so
each point away from the synthetic n also carries a k that occurs nowhere
else. n is never varied at fixed k, and no exponent in n is identifiable from
a path -- `A n^2 k` and `A' n^a k^b` describe those points equally well -- so
"O(n^2 k)" is a claim about the source code rather than a measurement.

This harness makes n and k independently varying axes and FITS the exponents,
printing the worst residual next to each one so a fit that does not describe
the data is visible instead of being rounded to 2.00. Two models per row:

    log T = a log n + c          the exponent usually quoted. Biased LOW at
                                 small n by the (n-1) factor and by any O(n)
                                 term, and biased HIGH across a cache
                                 boundary, so it is never reported alone.
    T = A n(n-1)/2 + B n + C     the model the kernel actually has. B/A gives
                                 the size above which the O(nk) prologue is
                                 negligible, which is the quantity the
                                 O(n^2 k) framing assumes without checking.

The second model is here because of the sparse path. Its prologue is O(nk)
whatever the density -- `csr_from_dense` has to scan every entry to find the
nonzeros -- while its pair loop shrinks by d^2, so a large fraction of a
sparse measurement can be CSR construction rather than kernel. That is
invisible at one n and obvious across six, and it means a sparse margin
measured at the sizes the suite uses is a lower bound.

Three further axes are swept because they change the answer and are otherwise
untimed: threads (1 against all, interleaved, so that excess exponent at full
threads is attributable to the memory system rather than to the algorithm),
dtype, and the API surface -- no other harness in `bench/` times `jsd_pdist`
or the cross-matrix form, and `jsd_pdist` is both the fair comparison against
`scipy.pdist` (which is condensed) and the only form that reaches the top of
this grid.

Every point over budget is refused and listed rather than truncated away: a
grid that stops silently where the machine ran out of memory reports an
exponent fitted on whatever happened to fit.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import numpy as np

from fastjsd import (
    available_backends,
    jsd_matrix,
    jsd_pdist,
    resolve_backend,
)
from fastjsd.reference import scipy_matrix

import datasets
from _machine import machine

HYGIENE = """\
================================================================================
scale.py is a TIMING harness. Every figure below is a single-machine
measurement and is only meaningful on an idle box: no test run, no fuzz sweep,
no second benchmark, nothing else on the machine. A sweep that fits an exponent
is more sensitive to this than a single table is, because interference that
lands on the large-n end of a grid is indistinguishable from a cache wall and
comes back as a super-quadratic exponent.

`--dry-run` prints the grids, the memory forecast and the signature statistics
without timing anything, and is the mode that is safe to run beside other work.
================================================================================
"""

#: Geometric in n, ratio 2, so six points span 32x -- enough to identify the
#: exponent once B and C are in the model. The top of the grid is decided by
#: `--max-mem`, never by hand.
N_GRID = (512, 1024, 2048, 4096, 8192, 16384)

#: 64 and 65 straddle a 64-bit signature word, the point at which the sparse
#: k-exponent moves if the occupancy mask contributes to it.
K_GRID = (16, 32, 48, 64, 65, 96, 128, 256, 512, 1024)

K_SWEEP_N = 2000

#: (label, k, regime, role). `regime` is ("dense", alpha), ("z", nnz/row) or
#: ("d", density); `role` selects the pinned kernel.
N_ROWS = (
    ("dense  k=64", 64, ("dense", 0.3), "dense"),
    ("dense  k=256", 256, ("dense", 0.1), "dense"),
    ("sparse k=64  z=8", 64, ("z", 8), "sparse"),
    ("sparse k=256 z=8", 256, ("z", 8), "sparse"),
)

#: (label, regime, role, predicted k-exponent). The prediction is printed in
#: the header so that this harness can be seen to FAIL rather than merely to
#: produce a number: the merge costs 2z pointer steps and z*d logs per pair, so
#: at fixed z only the O(nk) CSR build and the mask depend on k at all.
K_ROWS = (
    ("dense", ("dense", 0.3), "dense", 1.00),
    ("sparse z=8", ("z", 8), "sparse", 0.00),
    ("sparse d=0.10", ("d", 0.10), "sparse", 1.00),
)

#: Rows compared against SciPy at every point. Full verification is
#: `abs(got - ref).max()`, four n^2 arrays -- 8 GiB to check a 1 GiB answer at
#: n=16384 -- so the paired comparison runs on a subsample of the same input
#: (the kernel is row-local and takes the same path) and the full output is
#: checked with allocation-free reductions only.
VERIFY_ROWS = 256

TOL = {8: 5e-12, 4: 2e-4}


# ----------------------------------------------------------------------------
# inputs and their statistics -- no timing in this section
# ----------------------------------------------------------------------------

def make_input(n, k, regime, seed):
    kind, param = regime
    if kind == "dense":
        return datasets.dirichlet(n, k, param, seed=seed)
    if kind == "z":
        return datasets.sparse_rows(n, k, min(int(param), k), seed=seed)
    return datasets.sparse_rows(n, k, max(1, min(k, round(param * k))), seed=seed)


def regime_label(regime):
    kind, param = regime
    if kind == "dense":
        return f"dirichlet(alpha={param})"
    return f"z={param:g}" if kind == "z" else f"d={param:g}"


def work_units(pairs, k, regime):
    """Relative cost of one point, used only to forecast the next one.

    The proxy has to follow the regime, not the shape. Forecasting a fixed-z
    sparse point with the dense proxy over-predicts by k/z and would refuse
    points that are in fact affordable -- and it would do so on exactly the
    row whose k-independence is the claim under test.
    """
    kind, param = regime
    if kind == "dense":
        return pairs * k
    if kind == "z":
        return pairs * min(param, k)
    return pairs * k * param


def mask_rates(P, sample=512):
    """(signature skip, true disjoint) over the pairs of a row subsample.

    Counts, not clocks. Both are here because the 64-bit fold computed here
    maps column c onto bit ``c % 64``, so its skip rate is a function of the
    nonzeros per row alone and stops tracking the true disjoint fraction as
    soon as k > 64. Reported side by side: the two agree at k=64 and diverge
    above it, and the gap is the part of the sparse advantage that does not
    scale with k.
    """
    B = np.ascontiguousarray(P[:sample] > 0)
    rows, k = B.shape
    if rows < 2:
        return None, None
    fold = np.zeros((rows, 64), dtype=bool)
    for c in range(k):
        fold[:, c & 63] |= B[:, c]
    iu = np.triu_indices(rows, 1)
    Bf = B.astype(np.float32)
    ff = fold.astype(np.float32)
    return (float(((ff @ ff.T)[iu] == 0).mean()),
            float(((Bf @ Bf.T)[iu] == 0).mean()))


def cache_n(k):
    """Row counts where the streaming window crosses L2 and L3.

    The dense self kernel streams P and PL, 16 bytes per (row, coordinate), so
    the boundaries are at n*k = 65,536 (1 MiB per core) and n*k ~ 1,048,576
    (a 16 MiB shared L3). Printed next to every fitted exponent so that a
    super-quadratic fit is attributable instead of merely reported.
    """
    return max(1, 65_536 // k), max(1, 1_048_576 // k)


# ----------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------

def fmt_bytes(b):
    return f"{b / 2**20:,.1f} MiB" if b < 2**30 else f"{b / 2**30:,.2f} GiB"


def mem_available():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return None


def mem_budget(args):
    """Byte ceiling in force, and where it came from.

    `--max-mem` is the operator's stated budget; MemAvailable is the machine's.
    The smaller wins, because a sweep that OOMs takes the rest of the sweep
    with it, and a point that died cannot be reported.
    """
    ceiling, why = args.max_mem * 1e9, f"--max-mem {args.max_mem:g} GB"
    avail = mem_available()
    if avail is not None and 0.75 * avail < ceiling:
        return 0.75 * avail, f"75 % of MemAvailable ({avail / 1e9:.1f} GB)"
    return ceiling, why


def predict_bytes(rows, out_elems, k, item, dens, backend):
    """Peak resident bytes: the output plus the working set.

    The output dominates everywhere this grid runs, but the working set is why
    float32 is not half of float64: `_xlogx_nb` forces the PL table to float64
    whatever the input dtype, so the output halves and the table does not.
    `numpy-sparse` is the exception that matters -- it holds two (n, m)
    accumulators, 16n^2 on top of the result, which dominates at scale and is
    why it is predicted separately rather than lumped in.
    """
    work = rows * k * (8 + item + 8)            # caller's P, prepared copy, PL
    if backend.endswith("sparse"):
        work += int(24 * rows * k * dens)       # CSR data/indices/indptr
    if backend == "numpy-sparse":
        work += 16 * out_elems
    return out_elems * item + work


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------

def measure(variants, reps):
    """Time every variant at one grid point, interleaved round-robin.

    The same discipline as `bench_honest.py`, for the reason recorded there:
    timing each variant to completion in turn lets one transient land entirely
    on one variant, which can move its reported ratio by more than the
    difference being measured. Round-robin instead, so that drift hits
    everyone, and keep each variant's own minimum: interference is one-sided,
    so the minimum estimates the noise floor.

    What is interleaved here is the set whose RATIO gets quoted -- threads=1
    against threads=all, f32 against f64, pdist against matrix. Points of
    different sizes are NOT interleaved with each other, because their outputs
    do not fit simultaneously, so thermal drift over a long sweep lands on the
    large-n end of a fitted exponent. That is not fixable within one process;
    it is why the sweep runs smallest-first and why no exponent is printed
    without its residual.
    """
    best = [np.inf] * len(variants)
    for _ in range(reps):
        for idx, (_, fn) in enumerate(variants):
            t0 = time.perf_counter()
            fn()
            best[idx] = min(best[idx], time.perf_counter() - t0)
    return best


def output_defect(D, api):
    """Allocation-free sanity check on a full-size output.

    These are reductions, so they cost no memory, and they catch the failures
    that actually happen at scale: a NaN from an unwritten tile, a triangle
    never filled, a value outside [0, 1]. The paired comparison against SciPy
    is done on a subsample instead -- see VERIFY_ROWS.
    """
    lo, hi = float(D.min()), float(D.max())
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"non-finite output (min {lo}, max {hi})"
    if lo < -1e-9 or hi > 1.0 + 1e-9:
        return f"output outside [0, 1]: [{lo:.3e}, {hi:.3e}]"
    if api == "matrix" and D.ndim == 2 and D.shape[0] == D.shape[1]:
        worst = float(np.abs(np.diagonal(D)).max())
        if worst != 0.0:
            return f"diagonal not exactly zero (max {worst:.3e})"
    return None


def verify(P, backend, api, dtype, rows=VERIFY_ROWS):
    """Max |error| against SciPy on a subsample of the same input."""
    S = np.ascontiguousarray(P[:rows])
    ref = scipy_matrix(S)
    if api == "pdist":
        got = jsd_pdist(S, backend=backend, dtype=dtype)
        iu = np.triu_indices(S.shape[0], 1)
        ref = ref[iu]
    else:
        got = jsd_matrix(S, backend=backend, dtype=dtype)
    err = float(np.abs(np.asarray(got, dtype=np.float64) - ref).max())
    tol = TOL[np.dtype(dtype).itemsize]
    if err > tol:
        raise AssertionError(
            f"{backend} {api} {np.dtype(dtype).name}: subsample disagrees "
            f"with scipy by {err:.2e} > {tol:.0e}"
        )
    return err


# ----------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------

def loglog_fit(x, y):
    """Slope of log y on log x, its standard error, and the worst residual.

    The residual is reported as a relative deviation because that is the form
    in which a bad fit is obvious: 2 % means the power law describes the data,
    40 % means the exponent was fitted through a cache wall and does not.
    """
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.column_stack([lx, np.ones_like(lx)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    dof = max(1, lx.size - 2)
    var = float(resid @ resid) / dof
    spread = float(((lx - lx.mean()) ** 2).sum())
    se = float(np.sqrt(var / spread)) if spread > 0 else float("nan")
    return float(coef[0]), se, float(np.abs(np.expm1(resid)).max())


def quadratic_fit(n, t):
    """T = A n(n-1)/2 + B n + C, and the worst relative residual.

    Fitted rather than assumed. A is the per-pair cost, B the O(n) prologue
    (`prepare`'s streaming passes, the PL table, and on the sparse path the
    whole CSR build, which is O(nk) at any density), C the fixed dispatch
    cost. 200*B/A is the n above which the prologue is under 1 % of the total,
    which is the quantity the O(n^2 k) framing leaves unstated.
    """
    N = np.asarray(n, float)
    t = np.asarray(t, float)
    A = np.column_stack([N * (N - 1) / 2.0, N, np.ones_like(N)])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    worst = float(np.abs((A @ coef) / t - 1.0).max())
    return float(coef[0]), float(coef[1]), float(coef[2]), worst


# ----------------------------------------------------------------------------
# the sweeps
# ----------------------------------------------------------------------------

HEAD = (f"{'n':>7}{'k':>6}{'k_eff':>6}{'z':>7}{'density':>9}"
        f"{'thr':>5}{'ms':>11}{'ns/pair':>10}{'ns/pair-k':>11}"
        f"{'skip':>7}{'disj':>7}{'out MiB':>9}")


def backend_for(role):
    """The pinned kernel for a role, degrading when numba is absent.

    Every point pins its backend and records what `auto` would have chosen, so
    that a host with SVML and a host without produce comparable fits. On this
    grid `auto` would route most dense points to `numba-svml`, which would
    silently change kernel in the middle of a density sweep.
    """
    have = available_backends()
    if role == "sparse":
        return "numba-sparse" if "numba-sparse" in have else "numpy-sparse"
    return "numba" if "numba" in have else "numpy"


def max_threads():
    try:
        import numba
    except ImportError:
        return 1
    return int(numba.config.NUMBA_NUM_THREADS)


def sweep_row(args, report, sweep, sizes, regime, role, api, predicted=None):
    """One (regime, backend, api) row of the grid, smallest size first."""
    backend = backend_for(role)
    item = 8
    thread_set = [max_threads()]
    if args.threads1 and max_threads() > 1:
        thread_set = [1, max_threads()]

    kinds = sorted({k for _, k in sizes})
    kdesc = f"k={kinds[0]}" if len(kinds) == 1 else f"k={kinds[0]}..{kinds[-1]}"
    print(f"\n{sweep}-sweep: {regime_label(regime)}  {kdesc}  "
          f"backend={backend}  api={api}  dtype=float64")
    if predicted is not None:
        print(f"  predicted {sweep}-exponent for this row: {predicted:.2f}")
    print(HEAD)
    print("-" * len(HEAD))

    prev, prev_work, fitted = {}, None, []
    for n, k in sizes:
        P = make_input(n, k, regime, seed=(n * 31 + k) % 2**31)
        dens = float(np.count_nonzero(P)) / P.size
        k_eff = int(P.any(axis=0).sum())
        auto = resolve_backend("auto", P)
        pairs = n * (n - 1) // 2
        out_elems = pairs if api == "pdist" else n * n
        need = predict_bytes(n, out_elems, k, item, dens, backend)
        skip, disj = mask_rates(P, args.mask_sample)
        work = work_units(pairs, k, regime)

        common = {"sweep": sweep, "n": n, "k": k, "k_eff": k_eff,
                  "density": dens, "nnz_per_row": dens * k,
                  "regime": regime_label(regime),
                  "backend": backend, "auto_would_pick": auto, "api": api,
                  "dtype": "float64", "mask_skip": skip, "true_disjoint": disj,
                  "peak_bytes_predicted": int(need)}

        if need > args.budget_bytes:
            report["refused"].append(dict(
                common, reason=f"{fmt_bytes(need)} > "
                               f"{fmt_bytes(args.budget_bytes)} budget"))
            continue

        keep = []
        for t in thread_set:
            name = f"threads={t}"
            fn = jsd_pdist if api == "pdist" else jsd_matrix
            call = partial(fn, P, backend=backend, threads=t)
            fore = None
            if prev.get(name) is not None and prev_work:
                fore = prev[name] * work / prev_work
                if fore > args.budget_s:
                    report["refused"].append(dict(
                        common, threads=t,
                        reason=f"forecast {fore:.3g} s > budget "
                               f"{args.budget_s:g} s"))
                    continue
            keep.append((name, call, t, fore))
        if not keep:
            continue

        if args.dry_run:
            ss = "-" if skip is None else f"{skip:.1%}"
            ds = "-" if disj is None else f"{disj:.1%}"
            for _, _, t, fore in keep:
                fs = "-" if fore is None else f"{fore:.2f} s"
                print(f"{n:>7}{k:>6}{k_eff:>6}{dens * k:>7.1f}{dens:>9.4f}"
                      f"{t:>5}{fs:>11}{'forecast':>10}{'':>11}"
                      f"{ss:>7}{ds:>7}{out_elems * item / 2**20:>9.1f}")
            continue

        # Warm-up before any timing: numba specializes on dtype and
        # contiguity, not on size, so one call per variant covers the point --
        # and it is also where the full output is checked, then dropped, so
        # that two n^2 arrays are never resident at once.
        for _, call, _, _ in keep:
            D = call()
            defect = output_defect(D, api)
            if defect:
                raise AssertionError(f"n={n} k={k} {backend} {api}: {defect}")
            del D
        err = verify(P, backend, api, np.float64, min(VERIFY_ROWS, n))

        best = measure([(nm, fn) for nm, fn, _, _ in keep], args.reps)
        for (name, _, t, _), sec in zip(keep, best, strict=True):
            ms = sec * 1e3
            nsp = sec * 1e9 / pairs
            mark = ("~" if ms < args.floor_ms else " ")
            mark += ("!" if auto != backend else " ")
            ss = "-" if skip is None else f"{skip:.1%}"
            ds = "-" if disj is None else f"{disj:.1%}"
            print(f"{n:>7}{k:>6}{k_eff:>6}{dens * k:>7.1f}{dens:>9.4f}"
                  f"{t:>5}{ms:>11.2f}{nsp:>10.2f}{nsp / k:>11.4f}"
                  f"{ss:>7}{ds:>7}{out_elems * item / 2**20:>9.1f}  {mark}")
            report["points"].append(dict(
                common, threads=t, reps=args.reps, ms=ms, ns_per_pair=nsp,
                ns_per_pair_coord=nsp / k, out_bytes=out_elems * item,
                max_abs_err=err, verified=f"subsample-{min(VERIFY_ROWS, n)}",
                below_floor=bool(ms < args.floor_ms), interleaved=True))
            prev[name] = sec
        prev_work = work
        fitted.append(((n if sweep == "n" else k),
                       {nm: s for (nm, _, _, _), s
                        in zip(keep, best, strict=True)}))

    if not args.dry_run:
        report["fits"].extend(
            row_fits(args, sweep, fitted, regime, role, api, backend,
                     kinds, predicted))


def row_fits(args, sweep, fitted, regime, role, api, backend, kinds, predicted):
    """Fit and print the exponents for one completed row."""
    out = []
    labels = sorted({lab for _, d in fitted for lab in d})
    for lab in labels:
        pts = [(x, d[lab]) for x, d in fitted if lab in d]
        xs = np.array([p[0] for p in pts], float)
        ts = np.array([p[1] for p in pts], float)
        # Points under the floor stay in the three-term fit, where they are
        # what identifies B and C, and are dropped from the log-log fit, where
        # a point dominated by fixed dispatch cost pulls the exponent down.
        used = ts * 1e3 >= args.floor_ms
        rec = {"sweep": sweep, "regime": regime_label(regime), "role": role,
               "api": api, "backend": backend, "threads": lab,
               "points": int(xs.size), "predicted_exponent": predicted}
        if used.sum() >= 3:
            a, se, resid = loglog_fit(xs[used], ts[used])
            rec.update(loglog_exponent=a, loglog_se=se,
                       loglog_worst_residual=resid,
                       loglog_points=int(used.sum()))
            note = "" if used.all() else f", {int((~used).sum())} below floor"
            print(f"  {lab:<12} log-log exponent in {sweep} = {a:6.3f} "
                  f"+/- {se:.3f}   worst residual {resid:6.1%}"
                  f"   ({int(used.sum())} pts{note})")
            if predicted is not None:
                # 3 SE plus 0.05: the standard error alone is the fit's own
                # scatter and says nothing about the run-to-run noise the
                # points were measured under, so a bare SE test would call a
                # 2 % drift a refuted prediction.
                agrees = abs(a - predicted) < 3 * se + 0.05
                print(f"{'':<15}predicted {predicted:.2f}; measured is "
                      f"{'consistent' if agrees else 'NOT consistent'}")
        else:
            print(f"  {lab:<12} log-log exponent: too few points above the "
                  f"{args.floor_ms:g} ms floor ({int(used.sum())})")
        if sweep == "n" and xs.size >= 3:
            A, B, C, resid = quadratic_fit(xs, ts)
            n1 = 200.0 * B / A if A > 0 else float("nan")
            l2, l3 = cache_n(kinds[0])
            rec.update(A=A, B=B, C=C, quad_worst_residual=resid,
                       n_prologue_1pct=n1, predicted_L2_n=l2,
                       predicted_L3_n=l3)
            print(f"{'':<15}T = A n(n-1)/2 + B n + C:  A={A:.3e}  B={B:.3e}  "
                  f"C={C:.3e}   worst residual {resid:.1%}")
            print(f"{'':<15}O(nk) prologue under 1 % of total above "
                  f"n = {n1:,.0f}    L2 wall n~{l2:,}, L3 wall n~{l3:,}")
        out.append(rec)
    return out


def sweep_n(args, report):
    grid = (16, 24, 32, 40) if args.smoke else N_GRID
    for _, k, regime, role in N_ROWS:
        if args.smoke and k > 64:
            continue
        sweep_row(args, report, "n", [(n, k) for n in grid], regime, role,
                  "pdist", predicted=2.00)


def sweep_k(args, report):
    grid = (8, 16, 32, 64) if args.smoke else K_GRID
    n = 40 if args.smoke else K_SWEEP_N
    for _, regime, role, predicted in K_ROWS:
        sweep_row(args, report, "k", [(n, k) for k in grid], regime, role,
                  "pdist", predicted=predicted)


def api_inputs(args):
    if args.smoke:
        return (("dense  n=40 k=32", 40, 32, ("dense", 0.3), "dense"),
                ("sparse n=40 k=64 z=8", 40, 64, ("z", 8), "sparse"))
    return (("dense  n=4000 k=64", 4000, 64, ("dense", 0.3), "dense"),
            ("sparse n=4000 k=256 z=8", 4000, 256, ("z", 8), "sparse"))


def api_panel(args, report):
    """The API surface no other harness times, interleaved at one point.

    `jsd_pdist` is timed nowhere else in `bench/` and is the only fair
    comparison against `scipy.pdist`, which is condensed; the cross form is
    likewise untimed elsewhere and takes a different schedule (n*m pairs, no
    `_balanced_order`, two CSRs); `out=` is the only place the first-touch
    page faults that min-of-N hides can be made visible; and
    `check=False, normalize=False` is a direct measurement of the O(nk)
    prologue that the O(n^2 k) framing leaves unstated.

    Every variant here is timed in ONE round-robin, because these are ratios
    against each other and nothing else -- see `measure`.
    """
    thr = max_threads()
    for label, n, k, regime, role in api_inputs(args):
        backend = backend_for(role)
        P = make_input(n, k, regime, seed=(n * 17 + k) % 2**31)
        m = max(2, n // 2)
        Q = make_input(m, k, regime, seed=(n * 17 + k + 1) % 2**31)
        dens = float(np.count_nonzero(P)) / P.size
        auto = resolve_backend("auto", P)
        pairs = n * (n - 1) // 2
        need = predict_bytes(n + m, n * n + pairs + n * m, k, 8, dens, backend)
        base = {"panel": label, "n": n, "k": k, "m": m, "density": dens,
                "backend": backend, "auto_would_pick": auto}
        if need > args.budget_bytes:
            report["refused"].append(dict(
                base, reason=f"api panel {fmt_bytes(need)} > "
                             f"{fmt_bytes(args.budget_bytes)} budget"))
            continue

        buf64 = np.empty(pairs, dtype=np.float64)
        variants = [
            ("jsd_matrix f64", partial(jsd_matrix, P, backend=backend,
                                       threads=thr), pairs),
            ("jsd_pdist  f64", partial(jsd_pdist, P, backend=backend,
                                       threads=thr), pairs),
            ("jsd_matrix f32", partial(jsd_matrix, P, backend=backend,
                                       threads=thr, dtype=np.float32), pairs),
            ("jsd_pdist  f32", partial(jsd_pdist, P, backend=backend,
                                       threads=thr, dtype=np.float32), pairs),
            ("jsd_pdist  f64 out=", partial(jsd_pdist, P, backend=backend,
                                            threads=thr, out=buf64), pairs),
            ("jsd_matrix f64 no check/norm",
             partial(jsd_matrix, P, backend=backend, threads=thr,
                     check=False, normalize=False), pairs),
            ("jsd_matrix f64 squared", partial(jsd_matrix, P, backend=backend,
                                               threads=thr, squared=True),
             pairs),
            (f"jsd_matrix f64 cross m={m}",
             partial(jsd_matrix, P, Q, backend=backend, threads=thr), n * m),
            ("jsd_matrix f64 threads=1", partial(jsd_matrix, P,
                                                 backend=backend, threads=1),
             pairs),
        ]

        print(f"\napi panel: {label}   backend={backend}"
              + ("" if auto == backend else f"   (auto would pick {auto})"))
        err64 = verify(P, backend, "matrix", np.float64, min(VERIFY_ROWS, n))
        err32 = verify(P, backend, "matrix", np.float32, min(VERIFY_ROWS, n))
        S, T = P[:VERIFY_ROWS], Q[:VERIFY_ROWS]
        ref = scipy_matrix(np.vstack([S, T]))[:S.shape[0], S.shape[0]:]
        errx = float(np.abs(jsd_matrix(S, T, backend=backend) - ref).max())
        if errx > TOL[8]:
            raise AssertionError(f"{label}: cross form disagrees with scipy "
                                 f"by {errx:.2e}")
        print(f"  vs scipy on {min(VERIFY_ROWS, n)} rows: f64 {err64:.1e}, "
              f"f32 {err32:.1e}, cross {errx:.1e}")

        if args.dry_run:
            print(f"  (dry run) {len(variants)} variants, "
                  f"{need / 2**20:,.0f} MiB predicted")
            continue

        for _, call, _ in variants:
            D = call()
            del D
        best = measure([(nm, fn) for nm, fn, _ in variants], args.reps)
        head = f"  {'variant':<30}{'ms':>11}{'ns/pair':>10}{'vs matrix f64':>15}"
        print(head)
        print("  " + "-" * (len(head) - 2))
        for (name, _, npairs), sec in zip(variants, best, strict=True):
            ratio = best[0] / sec
            print(f"  {name:<30}{sec * 1e3:>11.2f}"
                  f"{sec * 1e9 / npairs:>10.2f}{ratio:>14.2f}x")
            report["api"].append(dict(
                base, variant=name, pairs=int(npairs), reps=args.reps,
                ms=sec * 1e3, ns_per_pair=sec * 1e9 / npairs,
                ratio_vs_matrix_f64=ratio, interleaved=True,
                max_abs_err_f64=err64, max_abs_err_f32=err32,
                max_abs_err_cross=errx))
        print("  ns/pair, not ms, is the comparable column: the cross form "
              "computes n*m pairs")
        print("  where the self forms compute n(n-1)/2, and pdist writes half "
              "the bytes.")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="scale sweep: fit the n and k exponents of fastjsd")
    ap.add_argument("--sweep", choices=("n", "k", "api", "all"), default="all")
    ap.add_argument("--reps", type=int, default=3,
                    help="round-robin repetitions; the minimum is kept")
    ap.add_argument("--max-mem", type=float, default=4.0,
                    help="GB ceiling on predicted peak bytes per point "
                         "(default 4.0); points over it are refused, not run")
    ap.add_argument("--budget-s", type=float, default=30.0,
                    help="per-rep wall ceiling; a point forecast above it "
                         "from the previous point is refused")
    ap.add_argument("--floor-ms", type=float, default=None,
                    help="points below this are kept but excluded from the "
                         "log-log fit (default 1.0, 0.0 under --smoke)")
    ap.add_argument("--mask-sample", type=int, default=512,
                    help="rows used for the signature statistics")
    ap.add_argument("--no-threads1", dest="threads1", action="store_false",
                    help="skip the single-threaded variant of every point")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grids, memory and mask rates; time nothing")
    ap.add_argument("--smoke", action="store_true",
                    help="cap n at 40 and k at 64: a crash test, NOT a "
                         "measurement -- every number it prints is noise")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.floor_ms is None:
        args.floor_ms = 0.0 if args.smoke else 1.0

    print(HYGIENE)
    if args.smoke:
        print("SMOKE RUN: n <= 40. This exercises every code path and "
              "measures nothing.\n"
              "Every millisecond below is dispatch overhead. Do not record "
              "or quote any of it.\n")
    info = machine()
    print(json.dumps(info, indent=2))

    args.budget_bytes, why = mem_budget(args)
    avail = mem_available()
    have = "unknown" if avail is None else fmt_bytes(avail)
    print(f"\nmemory ceiling {fmt_bytes(args.budget_bytes)} ({why}); "
          f"MemAvailable {have}")
    print(f"time budget {args.budget_s:g} s/rep, reps={args.reps}, "
          f"floor {args.floor_ms:g} ms, threads {max_threads()}")
    print("marks: '~' below the timing floor, '!' auto would have picked a "
          "different kernel")

    report = {"machine": info,
              "config": {"reps": args.reps, "max_mem_gb": args.max_mem,
                         "budget_bytes": args.budget_bytes,
                         "budget_s": args.budget_s, "floor_ms": args.floor_ms,
                         "threads1": args.threads1, "smoke": args.smoke,
                         "dry_run": args.dry_run,
                         "threads": max_threads(),
                         "n_grid": list(N_GRID), "k_grid": list(K_GRID),
                         "k_sweep_n": K_SWEEP_N},
              "points": [], "fits": [], "api": [], "refused": []}

    if args.sweep in ("n", "all"):
        sweep_n(args, report)
    if args.sweep in ("k", "all"):
        sweep_k(args, report)
    if args.sweep in ("api", "all"):
        api_panel(args, report)

    if report["refused"]:
        print(f"\nREFUSED ({len(report['refused'])}) -- listed, not dropped. "
              f"A sweep that stops silently where the\nmachine ran out reports "
              f"an exponent fitted on whatever happened to fit.")
        for r in report["refused"]:
            where = (f"api panel {r['panel']}" if "panel" in r else
                     f"n={r['n']} k={r['k']} {r['backend']} {r['api']}"
                     + (f" threads={r['threads']}" if "threads" in r else ""))
            print(f"  {where:<46}{r['reason']}")

    if args.smoke:
        print("\nSMOKE RUN COMPLETE: nothing above is a measurement.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
