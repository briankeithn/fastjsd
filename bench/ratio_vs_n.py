"""Does the ratio to the baselines hold across n, or only at one n?

    python bench/ratio_vs_n.py            # k=64 dense, n = 512 .. 8192
    python bench/ratio_vs_n.py --k 256 --max-n 4096

`bench/bench_honest.py` fixes n=2048 and varies k and density. Its ratios
characterise the implementations only if they do not move with n, and that
has to be measured rather than assumed: the scale sweep in `bench/scale.py`
fits exponents for *this package's* kernels and never runs a baseline, so
"all of them are O(n^2 k), so the ratio is flat" is an argument and not a
measurement.

There are concrete reasons it might not be flat.

  small n   fastjsd pays a fixed cost the baselines do not: `prepare`, the
            O(nk) `xlogx` tabulation and a parallel region per call. SciPy's
            pdist is a scalar C loop with none of that. So the ratio should
            RISE with n at the low end and then level off, and the level-off
            point is what justifies a choice of n.

  large n   the kernels stream P and PL, `n*k*16` bytes, so they leave L3 at
            a size that depends on k and on the machine: 16 MiB is reached
            around n=16k at k=64 and n=4k at k=256, and the boundary moves
            with whatever cache the run actually has. SciPy re-reads two rows
            per pair and stays cache-resident, so a cliff there costs these
            kernels and not SciPy. The torch baseline is the other way round:
            its mixture tensor grows with the chunk, so it should degrade
            first.

Both effects are properties of the ratio rather than of any single
implementation's exponent, so both belong here rather than in an exponent
fit that runs one implementation.

Timed interleaved round-robin, minimum of `--reps`, on an idle machine -- the
same discipline as `bench/bench_honest.py`, for the same reason.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fastjsd import jsd_matrix

import baselines
import datasets

#: float64 output bytes per point; the sweep refuses anything over --max-mem.
_OUT_BYTES = 8


def cases(k, z, ns, max_mem_gb):
    for n in ns:
        need = _OUT_BYTES * n * n
        if need > max_mem_gb * (1 << 30):
            print(f"  refused n={n}: output would be {need / (1 << 30):.1f} GB "
                  f"> --max-mem {max_mem_gb} GB")
            continue
        if z is None:
            yield n, datasets.dirichlet(n, k, 1.0, seed=n)
        else:
            yield n, datasets.sparse_rows(n, k, z, seed=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--z", type=int, default=None,
                    help="non-zeros per row; omit for dense")
    ap.add_argument("--max-n", type=int, default=8192)
    ap.add_argument("--min-n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-mem", type=float, default=3.0)
    ap.add_argument("--with-thirdparty", action="store_true",
                    help="also time pynndescent and torch (much slower)")
    args = ap.parse_args()

    impls = [("scipy", baselines.scipy_pdist),
             ("Lee-97", baselines.lee_sparse_numba)]
    if args.with_thirdparty:
        impls += list(baselines.THIRD_PARTY.items())

    ns = []
    n = args.min_n
    while n <= args.max_n:
        ns.append(n)
        n *= 2

    print(f"k={args.k}  {'dense' if args.z is None else f'z={args.z}'}  "
          f"reps={args.reps}, interleaved round-robin, idle machine required\n")
    hdr = [nm for nm, _ in impls]
    print(f"{'n':>7}{'fastjsd ms':>12}{'ns/pair-k':>11}"
          + "".join(f"{h + ' ms':>12}" for h in hdr)
          + "".join(f"{h + ' x':>11}" for h in hdr))
    print("-" * (30 + 23 * len(hdr)))

    ratios = {nm: [] for nm, _ in impls}
    for n, P in cases(args.k, args.z, ns, args.max_mem):
        cand = [fn for _, fn in impls] + [jsd_matrix]
        for fn in cand:                       # pay the JIT, verify once
            fn(P)
        ref = baselines.scipy_pdist(P)
        err = np.abs(jsd_matrix(P) - ref).max()
        assert err < 5e-12, f"n={n}: fastjsd disagrees with scipy ({err:.2e})"

        best = [np.inf] * len(cand)
        for _ in range(args.reps):
            for idx, fn in enumerate(cand):
                t0 = time.perf_counter()
                fn(P)
                best[idx] = min(best[idx], time.perf_counter() - t0)
        tf = best[-1]
        pairs = n * (n - 1) / 2
        print(f"{n:>7}{tf * 1e3:>12.1f}{tf * 1e9 / (pairs * args.k):>11.3f}"
              + "".join(f"{t * 1e3:>12.1f}" for t in best[:-1])
              + "".join(f"{t / tf:>10.2f}x" for t in best[:-1]))
        for (nm, _), t in zip(impls, best[:-1], strict=True):
            ratios[nm].append(t / tf)

    print()
    for nm, r in ratios.items():
        if len(r) < 2:
            continue
        spread = max(r) / min(r)
        verdict = ("flat" if spread < 1.25 else
                   "MOVES -- one n does not stand for the others")
        print(f"  vs {nm:8}: {min(r):6.2f}x - {max(r):6.2f}x   "
              f"spread {spread:.2f}x   {verdict}")
    print("\n`ns/pair-k` is the quantity that must be flat if the cost model")
    print("holds; a step in it locates a cache boundary, and a ratio that")
    print("moves where it steps is a memory-system result, not an algorithmic")
    print("one. Quote the ratio range, never a single n's value.")


if __name__ == "__main__":
    main()
