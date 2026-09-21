"""Benchmark against what practitioners run and what the literature published.

    python bench/bench_honest.py             # the table
    python bench/bench_honest.py --threads 1 # single-threaded, to separate
                                             # kernel quality from parallelism

The baselines are limited on purpose to implementations a caller can install.
Timing this package against variants of its own kernels would measure an
internal choice rather than the alternatives a user would otherwise reach
for; `bench/bits_vs_merge.py` is where its own kernels are compared with
each other.

Every implementation is checked against SciPy before it is timed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fastjsd import jsd_matrix

import baselines
import datasets


def timed(fn, *a, reps=5, **kw):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        fn(*a, **kw)
        best = min(best, time.perf_counter() - t)
    return best


def _pin_threads(n):
    """Pin every runtime that has a thread pool, not just Numba.

    Numba covers fastjsd, the Lee baseline and the pynndescent harness, since
    all three run in the same `prange` loop. Torch has its own pool and takes
    every core while the others are pinned to one, which would compare a
    pinned implementation against a threaded torch rather than comparing
    single-threaded kernels.
    """
    import numba
    numba.set_num_threads(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def run(args, threads):
    _pin_threads(threads)
    import numba
    print(f"\n{'=' * 96}")
    if threads == 1:
        print("SINGLE THREAD -- the like-for-like comparison.")
        print("SciPy's pdist is a scalar single-threaded C loop, so this is the")
        print("row where nothing is being credited to parallelism. Read it first.")
    else:
        print(f"{threads} THREADS -- what a caller actually gets.")
        print("The margin over SciPy here is kernel AND threading AND SIMD "
              "together;")
        print("the single-thread block above is what separates them.")
    print(f"numba threads: {numba.get_num_threads()}   "
          f"(NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS})")
    impls = [("scipy", baselines.scipy_pdist),
             ("Lee-1997", baselines.lee_sparse_numba)]
    if threads != 1:
        # The single-thread block exists to compare KERNELS, and the two
        # baselines that bear on that are SciPy (what practitioners run, and
        # already single-threaded) and Lee-1997 (the same identity on the
        # same harness). pynndescent and torch are ecosystem context: no
        # claim here rests on their single-threaded behaviour, and at one
        # thread they dominate the run's wall clock. They stay in the
        # all-threads block, which is where "what else is installed in a
        # Python environment" is the point.
        impls += list(baselines.THIRD_PARTY.items())

    SHORT = {"scipy": "scipy", "Lee-1997": "Lee-97",
             "pynndescent (numba)": "pynnd", "torch batched": "torch"}
    hdr = [SHORT[n] for n, _ in impls]
    print(f"reps={args.reps}, interleaved round-robin\n")
    print(f"{'case':31}" + "".join(f"{h:>10}" for h in hdr)
          + f"{'fastjsd':>10}   " + "".join(f"{h + ' x':>10}" for h in hdr))
    print("-" * (31 + 10 * len(hdr) + 13 + 10 * len(hdr)))

    groups = [("SYNTHETIC -- the primary grid", datasets.synthetic()),
              ("CASE STUDY -- real HDBSCAN memberships", datasets.case_study())]

    # COSTING PASS -- one call of everything, before any timing that counts.
    #
    # A progress estimate should not require inspecting the process. One warm
    # call per (case, implementation) is a small fraction of the run and buys
    # an ETA good to a few per cent, because the duel count and the budget are
    # both known in advance. It also pays every JIT, so the first timed duel
    # is not measuring compilation.
    plan = []
    t_plan = time.perf_counter()
    print("costing (one warm call each) ...", flush=True)
    for gname, cases in groups:
        for label, P in cases:
            per = []
            for _n, fn in impls:
                t0 = time.perf_counter()
                fn(P)
                per.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            jsd_matrix(P)
            tf1 = time.perf_counter() - t0
            # duels stop at --budget-s, never under 3
            est = 0.0
            for t in per:
                one = t + tf1
                est += one * max(3, min(args.reps, int(args.budget_s / one) + 1))
            plan.append((gname, label, est))
    total = sum(e for _, _, e in plan)
    spent_plan = time.perf_counter() - t_plan
    print(f"costing took {spent_plan:.0f}s; {len(plan)} cases, "
          f"estimated {total / 60:.1f} min of timed work")
    for _gname, label, est in plan:
        print(f"    {label:38} ~{est:6.1f}s")
    print(flush=True)

    est_of = {label: est for _, label, est in plan}
    done_s = 0.0
    t_start = time.perf_counter()

    rows = []
    # Worst across ALL cases, for every implementation INCLUDING fastjsd.
    # A worst-case column for the baselines printed beside a single case's
    # value for fastjsd is not a comparison: the same reduction has to apply
    # to every column, this kernel's included.
    worst = {n: None for n, _ in impls}
    worst["fastjsd"] = None
    group_ratios = {}
    for gname, cases in groups:
        print(f"\n{gname}")
        group_ratios[gname] = []
        for label, P in cases:
            ref = baselines.scipy_pdist(P)
            jsd_matrix(P)
            err = np.abs(jsd_matrix(P) - ref).max()
            assert err < 5e-12, f"{label}: fastjsd disagrees with scipy ({err:.2e})"
            if worst["fastjsd"] is None or err > worst["fastjsd"]:
                worst["fastjsd"] = err

            # Verify first, time second, and time everything INTERLEAVED.
            #
            # Timing each implementation to completion in turn lets a single
            # transient land entirely on one implementation, which can move
            # its reported ratio by more than the difference being measured.
            # Round-robin instead, so drift is spread across all of them.
            errs = []
            for name, fn in impls:
                jit = "numba" in name or name == "Lee-1997"
                if jit:
                    fn(P)                                       # pay the JIT once
                e = np.abs(fn(P) - ref).max()
                tol = baselines.TOL.get(name, 5e-12)
                assert e <= tol, f"{label}: {name} err {e:.2e} > tol {tol:.0e}"
                errs.append(e)
                if worst[name] is None or e > worst[name]:
                    worst[name] = e

            # PAIRED DUELS, one baseline at a time, order alternating.
            #
            # Two simpler designs are inadequate here, for different
            # reasons.
            #
            # 1. Round-robin in a fixed order, ratio of minima. Every
            #    implementation always runs after the same predecessor, so
            #    one that follows a memory-heavy neighbour (torch allocates
            #    tens of GB of tensors) inherits its cache and allocator
            #    state while one that follows SciPy does not.
            # 2. Rotate the order, ratio of minima. Rotation removes the
            #    predecessor bias, but min-of-N is biased by unequal
            #    VARIANCE: the noisier implementation needs more draws to
            #    approach its floor, so at small N it is reported slower than
            #    it is. Rotation also makes a within-rep paired ratio
            #    meaningless -- the two implementations sit at different
            #    positions by construction, so there is nothing to pair.
            #
            # A ratio between two implementations needs them to run
            # ADJACENTLY under the same conditions, repeatedly, with
            # which-goes-first alternating so the second position's penalty
            # falls on each equally. That is a paired duel. fastjsd is
            # re-timed inside every duel, so each ratio is formed from two
            # calls that ran adjacently rather than from timings taken
            # elsewhere in the run.
            tf = np.inf
            times, ratio, used = [], [], []
            for _name, fn in impls:
                a_t, b_t = [], []
                spent = 0.0
                for r in range(args.reps):
                    pair = ((fn, a_t), (jsd_matrix, b_t))
                    if r % 2:
                        pair = pair[::-1]
                    for g, acc in pair:
                        t0 = time.perf_counter()
                        g(P)
                        dt = time.perf_counter() - t0
                        acc.append(dt)
                        spent += dt
                    # A budget, because one cell can otherwise dominate the
                    # run: a single-threaded baseline on the largest cells can
                    # take tens of seconds per call, so five duels against it
                    # is many minutes for one number. Never below 3 duels --
                    # the median of two is meaningless.
                    if r + 1 >= 3 and spent > args.budget_s:
                        break
                used.append(len(a_t))
                times.append(min(a_t))
                tf = min(tf, min(b_t))
                pr = sorted(x / y for x, y in zip(a_t, b_t, strict=True))
                ratio.append(pr[len(pr) // 2])

            print(f"{label:31}"
                  + "".join("      skip" if t is None else f"{t*1e3:>10.1f}"
                            for t in times)
                  + f"{tf*1e3:>10.1f}   "
                  + "".join("         -" if r is None else f"{r:>9.2f}x"
                            for r in ratio))
            done_s += est_of.get(label, 0.0)
            frac = done_s / total if total else 1.0
            elapsed = time.perf_counter() - t_start
            eta = elapsed / frac - elapsed if frac > 0 else 0.0
            note = ""
            if min(used) < args.reps:
                note = (f"  duels {'/'.join(map(str, used))} of {args.reps} "
                        f"(budget {args.budget_s:.0f}s)")
            print(f"{'':31}  [{frac:5.0%}]  elapsed {elapsed / 60:4.1f}m  "
                  f"eta {eta / 60:4.1f}m{note}", flush=True)
            rows.append((label, ratio, tf))
            group_ratios[gname].append((label, ratio, tf))

    print(f"\n{'AGREEMENT with scipy, worst case':31}"
          + "".join("      skip" if worst[n] is None else f"{worst[n]:>10.1e}"
                    for n, _ in impls)
          + f"{worst['fastjsd']:>10.1e}")
    print("  This row is AGREEMENT WITH SCIPY, which is not accuracy. fastjsd is")
    print("  more accurate than scipy near coincidence (well_conditioned is on")
    print("  by default for the merge), so improving accuracy INCREASES this")
    print("  number: the closer this kernel gets to the true value, the further")
    print("  it sits from scipy's. bench/accuracy.py measures against 50-digit")
    print("  truth, which is the metric that means something.")
    print("  pynndescent adds FLOAT32_EPS to EVERY coordinate: it computes a")
    print("  smoothed divergence, not this one, and the deviation grows with the")
    print("  number of zeros. Reported, not excluded -- see baselines.py.")
    print("  sklearn is not timed: it has no compiled JS metric, so the only")
    print("  route is a per-pair Python callable. That measures interpreter")
    print("  round-trips, not an implementation -- see baselines.py.")

    # Reported per group, never blended. A range that mixes a chosen grid with
    # two fixed real matrices is a range whose endpoints nobody can reproduce
    # without those matrices, and the synthetic grid is the part a reader can
    # regenerate anywhere.
    for gname, grows in group_ratios.items():
        print(f"\n{gname}")
        for idx, (name, _) in enumerate(impls):
            r = [t[idx] for _, t, f in grows if t[idx] is not None]
            if r:
                note = "" if idx else "   (what practitioners run)"
                if name == "Lee-1997":
                    note = "   (the published identity, same harness)"
                print(f"  fastjsd vs {name:22}: {min(r):8.2f}x - "
                      f"{max(r):8.2f}x{note}")
    print()

    # The claim the Lee comparison cannot show: behaviour at zero.
    #
    # This MUST be a sweep, not one draw. The constant-folded form fails
    # intermittently -- whether ln2 cancels numerically depends on the
    # particular row -- so a single seed comes back clean often enough to
    # hide the failure entirely.
    print("identical rows (the difference the timing cannot show):")
    impl_zero = {"fastjsd": jsd_matrix,
                 "Lee-1997 sparse": baselines.lee_sparse_numba,
                 "scipy": baselines.scipy_pdist,
                 "pynndescent": baselines.pynndescent_numba}
    # pynndescent and torch are optional: the `bench` extra does not install
    # them, because they are ecosystem context rather than something a claim
    # rests on. Drop any implementation that cannot be imported here instead
    # of aborting the whole report over it.
    probe = np.repeat(datasets.dirichlet(1, 8, 0.1, seed=0), 2, axis=0)
    missing = []
    for nm in list(impl_zero):
        try:
            impl_zero[nm](probe)
        except ImportError:
            impl_zero.pop(nm)
            missing.append(nm)
    if missing:
        print(f"  (not installed, skipped: {', '.join(sorted(missing))})")

    bad = {k: [] for k in impl_zero}
    trials = 0
    for k in (8, 16, 32, 64, 128, 192, 256, 512):
        for seed in range(12):
            dup = np.repeat(datasets.dirichlet(1, k, 0.1, seed=seed), 6, axis=0)
            trials += 1
            for nm, fn in impl_zero.items():
                w = np.abs(fn(dup)).max()
                if w != 0.0:
                    bad[nm].append(w)
    for nm in impl_zero:
        h = bad[nm]
        print(f"  {nm:16} " + (f"exactly 0.0 in all {trials}" if not h
                               else f"NONZERO in {len(h)}/{trials}, "
                                    f"max {max(h):.2e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget-s", type=float, default=90.0,
                    help="wall-clock ceiling per duel; reps stop early once "
                         "exceeded, never below 3. A cell that stops early "
                         "prints its duel count, so it is visible, not silent.")
    ap.add_argument("--threads", default="1,max",
                    help="comma-separated thread counts; 'max' means all cores. "
                         "Single thread is measured and printed FIRST because it "
                         "is the only block that compares kernels rather than "
                         "core counts.")
    args = ap.parse_args()

    import numba
    counts = []
    for tok in args.threads.split(","):
        tok = tok.strip()
        counts.append(numba.config.NUMBA_NUM_THREADS if tok == "max" else int(tok))
    for t in counts:
        run(args, t)


if __name__ == "__main__":
    main()
