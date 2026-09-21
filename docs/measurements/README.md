# One machine's measurements

Everything here came from a single host, in one sitting, produced by
`bench/run_all.py` and `bench/bench_honest.py`. It is a worked example of what
the harnesses report and one data point on the claims -- not a specification.
Run them on your own hardware before relying on any number.

## The machine

```
Intel(R) Xeon(R) Processor @ 2.10GHz, 4 physical cores, 15.7 GB
Linux 6.18.44, Python 3.11.15, numpy 2.4.6, numba 0.67.0, scipy 1.17.1
Intel SVML absent, so the numba-svml backend does not appear.
CPU 0.6 % busy before the run; drift probe 1.01x end to end.
```

`machine.json` carries the full fingerprint.

## Read `aa_control.txt` first

It duels the implementation against *itself*, so the true ratio is 1.000 by
construction and everything it reports is this machine's noise floor. Worst
tie-band 3.5 %, and 1.0-1.8 % on the real fixtures. A ratio elsewhere in this
directory that is closer to 1.000 than its cell's band is not a result.

## What the timings say

`bench_honest_1thread.txt`, single thread, so this compares kernels rather
than thread counts (SciPy's `pdist` is single-threaded regardless):

| against | synthetic grid | real fixtures |
|---|---|---|
| SciPy | 2.19x - 33.17x | 2.14x - 8.93x |
| Lee 1997, as printed | 1.86x - 5.06x | 1.52x - 2.10x |

The spread against SciPy tracks density, which is the support-intersection
identity doing its work: about 2.2x on dense input, above 30x at 1.6 %
non-zero.

`lee_control.txt` is the harder comparison and the one to weigh. It replaces
the literal transcription of Eq. (2.8) with the same identity implemented
competently -- tabulated logarithms and an occupancy prefilter, neither of
which has anything to do with where `ln 2` sits. Against that baseline: nine
of fourteen cells are resolved wins, 1.19x to 3.61x, and **five are losses**,
up to 1.25x, including both real fixtures under the `argmax` cleanup. The
`pub/mod` column, 0.39 to 0.88, is how much of the apparent margin against the
published form was the transcription rather than the arrangement.

Losing those cells is expected rather than surprising: the folded arrangement
absorbs the one-sided mass into a constant and so needs no per-row matched-mass
accumulators at all. What the unfolded arrangement buys is the exact zero, not
speed.

## What the timings cannot show

From the end of `bench_honest_1thread.txt`, 96 duplicate-row trials across
`k = 8 .. 512`:

```
fastjsd          exactly 0.0 in all 96
Lee-1997 sparse  NONZERO in 30/96, max 3.58e-08
scipy            exactly 0.0 in all 96
```

SciPy is at parity here. The difference is against the *constant-folded*
arrangement used by the published sparse form, which is the claim
`docs/MATH.md` makes and the only one these numbers support.

## Dispatch is machine-dependent

`bits_vs_merge.txt` reports the crossover between the two sparse kernels on
this host: the bitmap kernel is faster on 13 of 14 cells. That does **not**
generalise. The same experiment on two other machines put three of those cells
the other way, including both real fixtures. The merge's cost is dominated by
branch mispredicts, so the crossover follows the branch predictor and is not
portable. `fastjsd._numba._prefer_bits` documents this; re-run this harness
rather than assuming either answer.
