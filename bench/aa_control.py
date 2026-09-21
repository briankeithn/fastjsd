"""What is the smallest ratio this harness can actually resolve?

    python bench/aa_control.py                 # single thread, full grid
    python bench/aa_control.py --threads 4
    python bench/aa_control.py --quick         # three cells, for a smoke test

The ratios `bench/bench_honest.py` reports come out of a paired duel. This
script runs that duel with the same implementation on both sides, and what it
reports is the instrument's resolution: a measured ratio closer to 1.000 than
that is an unresolved tie rather than a small win.

So: duel `jsd_matrix` against `jsd_matrix`. Same function, same array, same
alternating order, same statistics. The true ratio is exactly 1.0 by
construction, so every deviation printed below is instrument noise and nothing
else.

Four numbers come out of it, and they answer different questions.

  `median`   what `bench_honest.py`'s ratio columns report (median of the
             within-duel paired ratios). Its distance from 1.000 is the bias.
  `IQR`      the resolution band, and the one to quote. A reported ratio
             inside `1.0 +/- IQR` is indistinguishable from no difference.
  `p10-p90`  the tail. Reported for context only: from a small sample it is
             essentially the range, so it grows with `--reps` and must not be
             used as a threshold.
  `min/min`  the ratio of minima, which is what the `ms` columns are built
             from. It can sit several percent from 1.000 on cells whose median
             is within 1 %. Min-of-N does not converge here -- prefer the
             median paired ratio for anything reported.

It also reports the POSITION EFFECT: the median of second-slot over first-slot
time. The duel alternates order precisely to cancel this, and the cancellation
is only as good as the effect is stable. A large position effect with a
median ratio near 1.0 means the design is working but the cell is noisy; a
large position effect that does NOT cancel means the alternation is not
enough and the cell needs more reps.

Read the verdict column, not the ratios: the `tie-band` decides which of the
ratios reported by the other harnesses count as measurements at all.

Benchmark hygiene: measure on an idle machine with nothing else running. This
script is short precisely so it is cheap to re-run whenever the machine
changes.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from fastjsd import jsd_matrix
from fastjsd._common import density

import datasets


def _pin_threads(n):
    """Same thread pinning as `bench_honest.py`, so both harnesses time
    under identical thread settings.
    """
    import numba
    numba.set_num_threads(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def duel(P, reps):
    """`bench_honest.py`'s paired duel, with the candidate replaced by the
    reference. Returns (candidate times, reference times, first-slot times,
    second-slot times) -- all in seconds.

    The two sides are deliberately the same callable, and both calls are paid
    on every rep. Calling once and halving would measure something else: the
    two slots of a duel are not interchangeable, which is the effect the
    alternating order exists to cancel.
    """
    a_t, b_t, first, second = [], [], [], []
    for r in range(reps):
        pair = ((jsd_matrix, a_t), (jsd_matrix, b_t))
        if r % 2:
            pair = pair[::-1]
        slot = []
        for g, acc in pair:
            t0 = time.perf_counter()
            g(P)
            dt = time.perf_counter() - t0
            acc.append(dt)
            slot.append(dt)
        first.append(slot[0])
        second.append(slot[1])
    return a_t, b_t, first, second


def _pct(xs, q):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=9,
                    help="duels per cell; bench_honest.py defaults to 7")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    _pin_threads(args.threads)
    import numba

    # This must cover every cell any other script will ask a band for, under
    # every label they use: `bench/lee_control.py` looks its bands up here by
    # label, and an unmatched label silently falls back to the worst band in
    # the file. That is conservative but it is not a measurement, so the
    # `argmax` cleanup variants are included explicitly rather than left to
    # `datasets.case_study()`, which only builds the `uniform` one.
    cells = list(datasets.synthetic())
    for name in ("soft_clusters_931x25", "soft_clusters_1014x40"):
        raw = datasets.real(name)
        n, k = raw.shape
        cells.append((f"real n={n} k={k} dense", raw))
        for mode in ("uniform", "argmax"):
            cells.append((f"real n={n} k={k} thresholded/{mode}",
                          datasets.threshold(raw, empty=mode)))
    if args.quick:
        cells = [cells[0], cells[len(cells) // 2], cells[-1]]

    print(__doc__.split("\n\n")[0])
    print(f"\nA/A control: jsd_matrix vs jsd_matrix, {args.reps} duels per "
          f"cell, {numba.get_num_threads()} thread(s).")
    print("True ratio is exactly 1.000 by construction. Everything printed "
          "is noise.\n")
    print(f"{'case':<44}{'ms':>8}{'median':>9}{'IQR':>8}{'p10-p90':>10}"
          f"{'min/min':>9}{'pos':>7}  verdict")
    print("-" * 110)

    bands = []
    for label, P in cells:
        P = np.ascontiguousarray(P, dtype=np.float64)
        jsd_matrix(P)                                   # pay the JIT
        a_t, b_t, first, second = duel(P, args.reps)

        ratios = [a / b for a, b in zip(a_t, b_t, strict=True)]
        med = statistics.median(ratios)
        p25, p75 = _pct(ratios, 0.25), _pct(ratios, 0.75)
        p10, p90 = _pct(ratios, 0.10), _pct(ratios, 0.90)
        mm = min(a_t) / min(b_t)
        pos = statistics.median(s / f for f, s in zip(first, second,
                                                      strict=True))
        # The IQR is the band to quote. p10-p90 from a small sample is
        # essentially the range: it has high variance and grows with the
        # number of draws, so it describes --reps rather than the cell. The
        # IQR is stable under resampling, which is what a threshold someone
        # else will reuse has to be.
        band = p75 - p25
        bands.append((label, band, med))

        # The tie band is what a reader needs: a reported ratio closer to
        # 1.0 than this measures the instrument, not the implementations.
        if band < 0.02:
            verdict = f"tie-band {band * 100:.1f}%"
        elif band < 0.05:
            verdict = f"tie-band {band * 100:.1f}%  (coarse)"
        else:
            verdict = f"tie-band {band * 100:.1f}%  UNUSABLE"

        short = f"{label} [{density(P):.0%} nz]"
        print(f"{short:<44}{min(a_t) * 1e3:>8.1f}{med:>9.3f}"
              f"{band * 100:>7.1f}%{(p90 - p10) * 100:>9.1f}%"
              f"{mm:>9.3f}{pos:>7.3f}  {verdict}")

    worst = max(b for _, b, _ in bands)
    bias = max(abs(m - 1.0) for _, _, m in bands)
    print("\n" + "-" * 108)
    print(f"worst tie-band across cells : {worst * 100:.1f}%")
    print(f"worst median bias from 1.000: {bias * 100:.1f}%")
    print()
    print("How to use this: a ratio reported by bench_honest.py is only a")
    print("result if it sits outside its own cell's tie-band. Compare every")
    print("margin over a baseline against the band for the same cell on this")
    print("machine; a margin narrower than the band is a tie, whatever the")
    print("ratio itself looks like.")
    if bias > 0.02:
        print()
        print("WARNING: the median is biased away from 1.000 by more than 2 %.")
        print("The duel is supposed to be symmetric. Investigate before")
        print("trusting any ratio from this harness.")


if __name__ == "__main__":
    main()
