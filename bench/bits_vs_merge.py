"""Where is the crossover between the merge and the bitmap kernel?

    python bench/bits_vs_merge.py
    python bench/bits_vs_merge.py --threads 4 --reps 15

`fastjsd._numba._prefer_bits` decides between the intersection merge and the
bitmap rank-select kernel by sampling the mean number of non-matching merge
steps per pair and comparing it to 1.5. That constant is a threshold on a
sampled statistic; this script is the experiment that locates the crossover it
stands in for.

It calls both kernels directly -- same CSR, same signatures, same ranks, same
row sums, same schedule -- so nothing but the kernel body differs, and it
reports:

  * `steps`, the statistic `_prefer_bits` actually thresholds on;
  * `degCV`, the coefficient of variation of the row degrees;
  * the measured merge and bitmap times;
  * which kernel `_prefer_bits` picks, and whether that was the right call;
  * the cost of every wrong call, which is what the threshold is for.

It also asserts the two kernels agree bit for bit (`array_equal`, not
`allclose`). Which kernel runs depends on the data, so if the two ever
diverge the result depends on the data too: dispatch stops being an
optimization and becomes a correctness bug.

Benchmark hygiene: measure on an idle machine with nothing else running. Read
the ratios against the tie-bands in `bench/results/aa_control_1t.txt`, which
`bench/run_all.py` writes from the A/A control: a ratio inside the band for
its cell is noise rather than a crossover.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fastjsd._numba import (
    _balanced_order,
    _offsets,
    _prefer_bits,
    _ranks,
    _row_sums,
    _signatures,
    _sparse_self,
    _sparse_self_bits,
    _unfolded,
    csr_from_dense,
    xlogx_like_kernels,
)

import datasets


def prep(P):
    """Everything both kernels need, built once so neither pays for setup."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    n, k = P.shape
    data, idx, ptr, _ = csr_from_dense(P)
    return {
        "n": n, "k": k, "data": data, "idx": idx, "ptr": ptr,
        "PL": xlogx_like_kernels(data),
        "sig": _signatures(idx, ptr, k),
        "rsum": _row_sums(data, ptr),
        "offs": _offsets(n, False), "order": _balanced_order(n),
    }


def steps_stat(c, rng_seed=0, sample=4096):
    """`_prefer_bits`'s statistic, recomputed so it can be printed."""
    z = np.diff(c["ptr"])
    rng = np.random.default_rng(rng_seed)
    a = rng.integers(0, c["n"], sample)
    b = rng.integers(0, c["n"], sample)
    keep = a != b
    a, b = a[keep], b[keep]
    inter = np.bitwise_count(c["sig"][a] & c["sig"][b]).sum(axis=1)
    return float(np.where(inter > 0, z[a] + z[b] - 2 * inter, 0).mean())


def best(fn, reps):
    fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    import numba
    numba.set_num_threads(args.threads)

    cells = [(lbl, P) for lbl, P in datasets.synthetic() if "dense" not in lbl]
    for name in ("soft_clusters_931x25", "soft_clusters_1014x40"):
        raw = datasets.real(name)
        n, k = raw.shape
        for mode in ("uniform", "argmax"):
            cells.append((f"real {n}x{k} {mode}",
                          datasets.threshold(raw, empty=mode)))

    print(__doc__.split("\n\n")[0])
    print(f"\n{numba.get_num_threads()} thread(s), min of {args.reps}. "
          "Both kernels called directly on identical inputs.\n")
    print(f"{'case':<30}{'steps':>7}{'degCV':>7}{'merge':>9}{'bits':>9}"
          f"{'bits win':>10}{'picks':>8}{'right?':>8}{'cost':>8}")
    print("-" * 97)

    rows = []
    for label, P in cells:
        c = prep(P)
        if not _unfolded(c["k"]):
            print(f"{label:<30}   folded signature -- bitmap kernel illegal")
            continue
        rank = _ranks(c["sig"], c["ptr"])
        inv = 1.0 / np.log(2.0)
        # np.zeros, not np.empty: `sym_write` leaves the diagonal untouched,
        # so two np.empty buffers would differ there by allocator luck and
        # the array_equal below would be comparing uninitialized memory.
        om = np.zeros(c["n"] * c["n"])
        ob = np.zeros(c["n"] * c["n"])

        def run_merge(c=c, om=om, inv=inv):
            _sparse_self(c["data"], c["idx"], c["ptr"], c["PL"], c["rsum"],
                         c["sig"], inv, False, om, c["offs"], c["order"],
                         True, False)

        def run_bits(c=c, ob=ob, rank=rank, inv=inv):
            _sparse_self_bits(c["data"], c["PL"], c["rsum"], c["sig"], rank,
                              inv, False, ob, c["offs"], c["order"],
                              True, False)

        run_merge()
        run_bits()
        for o in (om, ob):
            np.fill_diagonal(o.reshape(c["n"], c["n"]), 0.0)
        # Bit for bit, not merely close. Dispatch depends on the data, so if
        # these ever diverge the answer depends on the data too.
        assert np.array_equal(om, ob), f"{label}: kernels DISAGREE"

        st = steps_stat(c)
        z = np.diff(c["ptr"])
        cv = float(z.std() / z.mean()) if z.mean() else 0.0
        tm = best(run_merge, args.reps)
        tb = best(run_bits, args.reps)
        win = tm / tb                      # >1 means bits is faster
        picks = "bits" if _prefer_bits(c["sig"], c["ptr"]) else "merge"
        right = (picks == "bits") == (win > 1.0)
        # What the wrong call costs, as a multiple of the better kernel.
        cost = 1.0 if right else max(win, 1 / win)
        rows.append((label, st, win, picks, right, cost, cv))
        print(f"{label:<30}{st:>7.2f}{cv:>7.2f}{tm:>9.2f}{tb:>9.2f}"
              f"{win:>9.2f}x{picks:>8}{'yes' if right else 'NO':>8}"
              f"{'' if right else f'{cost:.2f}x':>8}")

    print("\n" + "-" * 97)
    wrong = [r for r in rows if not r[4]]
    bits_wins = [r for r in rows if r[2] > 1.0]
    print(f"bitmap faster on              : {len(bits_wins)} of {len(rows)} cells")
    print(f"_prefer_bits picks wrong on   : {len(wrong)} of {len(rows)} cells")
    if wrong:
        worst = max(r[5] for r in wrong)
        print(f"worst cost of a wrong pick    : {worst:.2f}x")
        print("  " + ", ".join(f"{r[0]} (steps {r[1]:.2f})" for r in wrong))
    if bits_wins:
        lo = min(r[1] for r in bits_wins)
        merge_wins = [r[1] for r in rows if r[2] <= 1.0]
        hi = max(merge_wins) if merge_wins else None
        print(f"\nbitmap wins down to steps={lo:.2f}"
              + (f"; merge wins up to steps={hi:.2f}" if hi is not None
                 else "; NO cell prefers the merge"))
        skew = [r for r in rows if r[6] > 0.5]
        flat = [r for r in rows if r[6] <= 0.5]
        if skew and flat:
            print(f"\nby row-degree CV: bitmap wins "
                  f"{sum(r[2] > 1 for r in skew)}/{len(skew)} of the SKEWED "
                  f"cells (CV>0.5) and "
                  f"{sum(r[2] > 1 for r in flat)}/{len(flat)} of the FLAT "
                  f"ones.\nThe merge's cost is branch mispredicts, and "
                  "mispredicts come from unpredictable\ntrip counts -- so "
                  "the deciding quantity is the VARIANCE of the row degrees, "
                  "not\nthe mean of the step count. `_prefer_bits` samples "
                  "the mean.\nNote that `datasets.sparse_rows` gives every "
                  "row exactly `nnz` non-zeros, so\nthe whole synthetic "
                  "sparse grid sits at CV=0 -- degenerate in precisely the\n"
                  "dimension that decides dispatch, and unreachable by real "
                  "data.")
        if hi is None:
            print("If that holds across the batch run, the threshold is not")
            print("mis-tuned -- it is unnecessary, and the gate collapses to")
            print("`_unfolded(k)`. That deletes the threshold constant, a")
            print("4096-pair sampled probe and its rng_seed from dispatch.")


if __name__ == "__main__":
    main()
