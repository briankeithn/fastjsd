"""How much of any margin over Lee is arrangement, and how much implementation?

    python bench/lee_control.py                    # sparse cells + real, 1 thread
    python bench/lee_control.py --with-dense
    python bench/lee_control.py --threads 4

A margin measured against a literal transcription of "the published sparse
form" -- Lee (1997) Eq. (2.8), paying two logarithms and two divisions per
matched coordinate against this package's one -- is a difference in
tabulation, not in arrangement. The printed form factors as

    a ln(a/s) + b ln(b/s) = a ln a + b ln b - s ln s

so the folded arrangement accepts the same `x ln x` table fastjsd builds.
Following it through, `2 m ln m = s ln s - s ln 2`, so folded-with-tabulation
and unfolded cost the same arithmetic per matched coordinate -- and unfolded
costs marginally MORE, because it still has to accumulate the two one-sided
mass accumulators that folding absorbs into the constant. The arrangement
cannot buy speed; it can only cost.

So this script measures three ratios per cell, and the decomposition is the
result:

    ours / published   the margin against the transcription as printed
    ours / modern      the honest margin: same identity, same fold, same
                       tabulation, same 1984 prefilter. What is left is where
                       ln 2 sits and the two-accumulator complement.
    published / modern how much of that first number is the handicap the
                       literal transcription carries

If `ours / modern` lands inside the A/A tie-band, there is no resolved speed
difference against the published form on that cell, and what the package adds
there is exactness and availability rather than throughput.

Same paired-duel protocol as `bench_honest.py` (alternating order, median of
within-duel ratios). The median is reported, not the ratio of minima:
`bench/aa_control.py` can show min/min sitting several percent from 1.000
on cells whose median paired ratio is within 1 %.

Benchmark hygiene: measure on an idle machine with nothing else running.
Tie-bands come from `bench/results/aa_control_1t.txt`, which `bench/run_all.py`
writes from the A/A control, and are only valid for the machine that produced
them -- re-run `bench/run_all.py` first on any new host.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import time

import numpy as np

from fastjsd import jsd_matrix
from fastjsd._common import density

import baselines
import datasets

#: Tie-bands are read from the A/A control's output rather than hardcoded.
#:
#: A band is a property of a machine -- its noise floor and its branch
#: predictor -- so a table of bands pinned in source eventually describes a
#: different machine than the one the ratios came from. Reading them from
#: `bench/aa_control.py`'s own output keeps the band and the ratio it
#: calibrates on the same host. `bench/run_all.py` writes this file.
AA_RESULTS = pathlib.Path(__file__).parent / "results" / "aa_control_1t.txt"


def load_bands(path=AA_RESULTS):
    """Parse `{cell label: IQR}` out of `bench/aa_control.py`'s table.

    Returns `({}, reason)` if the file is missing or unparseable, which the
    caller must treat as "no band is known", not as "the band is zero".
    """
    if not path.exists():
        return {}, f"{path.name} not found -- run bench/run_all.py, which writes it"
    bands = {}
    for line in path.read_text().splitlines():
        if "tie-band" not in line or "%" not in line:
            continue
        head, _, tail = line.partition("tie-band")
        try:
            pct = float(tail.strip().split("%")[0])
        except ValueError:
            continue
        label = head.split("[")[0].strip()
        if label:
            bands[label] = pct / 100.0
    if not bands:
        return {}, f"{path.name} present but no tie-band rows parsed"
    return bands, ""


def band_for(label, bands):
    """Longest matching cell label wins, so `k=64 z=16` never matches `k=64`."""
    def norm(s):
        # Both files must be reduced the same way. aa_control labels the real
        # fixtures "thresholded/<mode>" and this script labels them "<mode>",
        # so stripping the prefix from only one side makes every real-fixture
        # lookup miss and silently fall back to the worst band in the file.
        return s.replace("synth n=2048 ", "").replace("thresholded/", "").strip()

    best, blen = None, -1
    probe = norm(label)
    for key, v in bands.items():
        core = norm(key)
        if (core in probe or probe in core) and len(core) > blen:
            best, blen = v, len(core)
    if best is None:
        return (max(bands.values()) if bands else float("inf")), "*"
    return best, ""


def _pin_threads(n):
    import numba
    numba.set_num_threads(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def duel(fa, fb, P, reps):
    """Paired duel: alternate which side goes first, return per-rep ratios."""
    ra = []
    for r in range(reps):
        pair = ((fa, 0), (fb, 1))
        if r % 2:
            pair = pair[::-1]
        t = [0.0, 0.0]
        for g, slot in pair:
            t0 = time.perf_counter()
            g(P)
            t[slot] = time.perf_counter() - t0
        ra.append(t[0] / t[1])
    return ra


def cells(with_dense):
    """Sparse synthetic + both real fixtures under both cleanup modes.

    The real fixtures appear twice on purpose. `threshold(empty="uniform")`
    resets every emptied row to 1/k, which manufactures a fully dense block
    (138 and 42 rows) and 9453 and 861 bit-identical pairs; `empty="argmax"`
    does not. The dispatch, and therefore the margin, differs between them,
    so reporting one without the other would hide that choice.
    """
    out = []
    for label, P in datasets.synthetic():
        if with_dense or "dense" not in label:
            out.append((label, P))
    for name in ("soft_clusters_931x25", "soft_clusters_1014x40"):
        raw = datasets.real(name)
        n, k = raw.shape
        for mode in ("uniform", "argmax"):
            T = datasets.threshold(raw, empty=mode)
            out.append((f"real n={n} k={k} thresholded/{mode}", T))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--with-dense", action="store_true",
                    help="include dense cells, where fastjsd dispatches away "
                         "from the merge entirely")
    args = ap.parse_args()

    _pin_threads(args.threads)
    import numba

    pub = baselines.lee_sparse_numba
    mod = baselines.lee_modern_numba
    bands, why = load_bands()
    if why:
        print(f"\nWARNING: no tie-bands available -- {why}.")
        print("Every ratio below is therefore UNCALIBRATED: without the A/A")
        print("control there is no way to say which of them are ties.")

    print(__doc__.split("\n\n")[0])
    print(f"\n{args.reps} paired duels per pair of implementations, "
          f"{numba.get_num_threads()} thread(s), median paired ratio.")
    print(">1.00 means fastjsd is FASTER. `band` is the A/A tie-band for the "
          "cell,\nread from bench/results/aa_control_1t.txt; any ratio inside "
          "1.000 +/- band\nis not distinguishable from a tie. Re-run the A/A "
          "control on THIS machine\nfirst -- a band from another machine is "
          "worse than no band at all.\n")
    print(f"{'case':<38}{'nz':>6}{'ours ms':>9}{'/pub':>8}{'/mod':>8}"
          f"{'pub/mod':>9}{'band':>7}  verdict")
    print("-" * 104)

    rows = []
    for label, P in cells(args.with_dense):
        P = np.ascontiguousarray(P, dtype=np.float64)
        for fn in (jsd_matrix, pub, mod):
            fn(P)                                        # pay the JIT
        # Agreement before timing. Both Lee variants fold, so both carry the
        # duplicate-row failure; that is what the exactness guarantee is
        # about, and not a reason to reject them as timing baselines.
        ref = baselines.scipy_pdist(P)
        assert np.abs(mod(P) - ref).max() < 1e-6, f"{label}: modern Lee wrong"

        r_pub = statistics.median(duel(jsd_matrix, pub, P, args.reps))
        r_mod = statistics.median(duel(jsd_matrix, mod, P, args.reps))
        r_hand = statistics.median(duel(pub, mod, P, args.reps))
        t0 = time.perf_counter()
        jsd_matrix(P)
        ms = (time.perf_counter() - t0) * 1e3

        # `duel(a, b)` returns t_a / t_b, so a ratio BELOW 1 means the first
        # argument was faster. Invert to report "x faster than the baseline".
        inv_pub, inv_mod = 1.0 / r_pub, 1.0 / r_mod
        band, star = band_for(label, bands)
        if abs(inv_mod - 1.0) <= band:
            verdict = "TIE vs modern Lee"
        elif inv_mod > 1.0:
            verdict = f"{inv_mod:.2f}x over modern Lee"
        else:
            verdict = f"LOSES {1 / inv_mod:.2f}x to modern Lee"
        rows.append((label, inv_pub, inv_mod, band))

        short = label.replace("synth n=2048 ", "").replace("thresholded/", "")
        print(f"{short:<38}{density(P):>5.0%}{ms:>9.1f}{inv_pub:>8.2f}"
              f"{inv_mod:>8.2f}{1 / r_hand:>9.2f}{band:>6.1%}{star}  {verdict}")

    print("\n" + "-" * 104)
    ties = [lbl for lbl, _, m, b in rows if abs(m - 1.0) <= b]
    wins = [(lbl, m) for lbl, _, m, b in rows if m - 1.0 > b]
    print(f"resolved wins over modern Lee : {len(wins)} of {len(rows)} cells")
    if wins:
        lo, hi = min(m for _, m in wins), max(m for _, m in wins)
        print(f"  range {lo:.2f}x - {hi:.2f}x")
    print(f"ties (inside the A/A band)    : {len(ties)} of {len(rows)}")
    print("\n`pub/mod` is how much of the margin against the published form")
    print("is the literal transcription's two logarithms and missing")
    print("prefilter, rather than anything about the identity's arrangement.")
    print("A `*` on the band means no A/A measurement for that cell; the")
    print("worst observed band was substituted, which is conservative.")


if __name__ == "__main__":
    main()
