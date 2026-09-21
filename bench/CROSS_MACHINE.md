# Cross-machine runs: an operator's guide

A latency figure is a fact about one machine. Measuring harder on that machine
does not turn it into a fact about any other, and the crossover between the two
sparse kernels is sensitive enough to the branch predictor that two hosts can
disagree about which kernel is faster on identical input. This directory's
cross-machine tooling exists to answer a narrower question that *is* answerable
elsewhere: **do the claims point the same way on different silicon, a different
libm and a different toolchain?**

Two programs:

| file | what it does |
|---|---|
| `bench/portable.py` | one leg -- runs on one box, writes one self-describing JSON |
| `bench/aggregate.py` | combines legs into a claim, and refuses to combine them wrongly |

`.github/workflows/cross-machine.yml` runs the legs on GitHub's hosted runners
(manual dispatch only) and runs the aggregation over the uploaded artifacts.
Its header comment states what CI can and cannot measure; this file is the same
story from the operator's side, plus the part about which numbers may be
quoted.

`bench/portable.py` is **not** a replacement for `bench/bench_honest.py`. A
single-machine comparison comes from running `bench_honest.py` on an idle
machine, and nothing in this directory changes that.

---

## Running one leg

From the repository root, with the package importable:

```bash
python bench/portable.py --leg-id my-box --json portable-my-box.json
```

Useful flags:

```bash
python bench/portable.py --fingerprint-only     # machine + software blocks, no timing at all
python bench/portable.py --reps 7 --repeats 5   # more evidence per ratio
python bench/portable.py --full-suite           # datasets.suite() instead of the CI case set
python bench/portable.py --threads --crossover  # the opt-in extras (recorded, not concluded from)
```

`--leg-id` is how the box is named everywhere downstream. Use something that
identifies the hardware, not the person (`lin-x64-epyc`, `m2-macbook`,
`hpc-login-node`), because that string is what a reader of the aggregate report
sees next to a ratio.

Things worth knowing before the first run:

- **Benchmark hygiene applies to a leg as it does to every other harness here.**
  On a machine you control, run it on an idle box: no test suite, no build, no
  second measurement in parallel. On a hosted runner that cannot be enforced,
  which is precisely why the leg grades its own reliability (below) instead of
  assuming the box was quiet.
- **Nothing is required beyond numpy.** `scipy`, `numba` and `mpmath` are each
  probed and recorded as absent with a reason if missing. A leg without scipy
  falls back to fastjsd's own NumPy backend as the reference and sets
  `config.reference_is_independent = false`, so the aggregator cannot mistake
  a self-comparison for a real one. A leg without numba runs the NumPy backends
  only and is reported in its own section -- that is a different claim (see
  below).
- The default case set is smaller than `datasets.suite()`: the two n=2000 k=256
  cases are dropped because SciPy needs seconds on each, and at
  `reps x repeats` that is most of a runner's budget spent on the baseline.
  `--full-suite` puts them back and is the right choice on a box you own.
- `--fingerprint-only` is safe to run while other work is in progress. It
  starts no timer. It is also the fastest way to find out whether a new box can
  run the full harness at all.

---

## Collecting legs from several boxes

The only rule is that every leg keeps its own file and its own `leg_id`. Do not
merge, edit or re-key the JSON by hand; the aggregator cross-checks the ratios
in each document against the milliseconds in the same document and will tell
you if they disagree.

**By hand:**

```bash
# on each box
python bench/portable.py --leg-id BOXNAME --json portable-BOXNAME.json
# then, anywhere
scp box1:portable-box1.json box2:portable-box2.json ./legs/
```

**From CI:** dispatch `cross-machine benchmark (manual)` from the Actions tab.
Each leg uploads an artifact named `bench-<leg_id>` containing
`portable-<leg_id>.json` (the document), `portable-<leg_id>.txt` (what it
printed) and `fingerprint-<leg_id>.json` (written before any timing, so it
survives a leg that crashes later). Download them all; the workflow's own
aggregation job already does this with `pattern: bench-*` and
`merge-multiple: true`.

Legs from different dates and different fastjsd commits can be mixed, but the
aggregator has no way to notice that you did. Each document records
`machine.github.sha`; check it before combining.

---

## Aggregating

```bash
python bench/aggregate.py 'legs/portable-*.json'
python bench/aggregate.py 'legs/portable-*.json' --json legs/summary.json
python bench/aggregate.py 'legs/portable-*.json' --markdown \
    >> "$GITHUB_STEP_SUMMARY"
```

Match `portable-*.json`, never `*.json`. A leg directory also holds each leg's
`fingerprint-*.json` and the `summary.json` written above; neither is a leg.
`aggregate.py` rejects both by shape and names them under `PROBLEMS`, but a
glob that does not pick them up in the first place is the better habit.

It is stdlib-only on purpose: the matrix it has to summarise is one where numpy,
scipy or numba may be exactly what went missing, so the aggregation must not
depend on them resolving.

Read the output in the order it prints, which is not the order you want:

1. **`LOSSES`** -- any box on which fastjsd was slower than a baseline, printed
   before every range and geomean. A summary that reports a wide speedup range
   while one of its cells sits below 1.0 has told the reader the opposite of
   what happened.
2. **`EXCLUDED` / `SUSPECT`** -- boxes that graded themselves unusable are
   dropped and the exclusion is printed. A silently dropped box is worse than a
   bad one.
3. **`PROBLEMS`** -- schema mismatches, ratios that disagree with their own
   milliseconds, missing blocks.
4. The per-configuration sections. `numba` and `numpy-only` are partitioned
   **before** anything is combined, and only the `numba` section belongs to the
   headline.
5. The consistency verdict, then the caveats, reprinted with the numbers rather
   than linked from them.

Three properties of the arithmetic, because they are the reason to trust it:

- Ratios only ever come from two implementations timed in the same interleaved
  round-robin on the same box. **No time from one runner is ever divided by a
  time from another.**
- The per-box summary is the **geometric** mean over cases. Ratios compose
  multiplicatively; the arithmetic mean of a ratio set is simply the wrong
  number.
- The unit of independence is the **box**, not the case. Five boxes x seven
  cases is not thirty-five observations, because cases within a box share the
  hardware. Counts over boxes are the claim; counts over cells are detail.

A cell whose across-repeat spread straddles 1.0 is reported as
`indistinguishable`, not as a number. That is what stops a ratio being asserted
from a box that cannot resolve it.

---

## How a leg grades itself

Published thresholds, applied by the leg so that the aggregator's rule stays
mechanical ("drop unusable, warn on suspect") rather than a judgement made
later while looking at a number:

| signal | suspect | unusable | why |
|---|---|---|---|
| `steal_frac` | >= 0.02 | >= 0.10 | the only direct evidence of a noisy neighbour available inside a guest; Linux only |
| fastjsd ms spread `(max-min)/median` | >= 0.25 | >= 0.50 | the box cannot resolve its own repeats |
| baseline `min/median` < 0.85 while fastjsd's is higher | warn | -- | the min-of-N duration bias: the slowest implementation is exposed to transients longest, so its ratio comes out inflated |

`validity.verdict` is `ok`, `suspect` or `unusable`, with `validity.reasons`
naming each trigger.

---

## What may be quoted, and what may not

This is the part that matters. A CI figure is a **robustness check on a claim**,
never a replacement for a single-machine measurement made with
`bench/bench_honest.py` on a quiet box.

### Quotable

| quantity | how to phrase it |
|---|---|
| **Direction, counted over boxes** | "fastjsd was faster than SciPy on `<w>`/`<B>` boxes and `<c>`/`<N>` case-box pairs." The strong claim, and the one CI is actually for. |
| **The ratio against Lee-1997** | Same Numba, same parallel harness, same schedule, same thread count, same working set -- clock rate and core count divide out, and it is nearly free of the duration bias. Quote it **with its across-repeat spread**, and as `indistinguishable` where the spread crosses 1.0. |
| **A range, never a point** | "the ratio range across boxes and cases is `<min>`x-`<max>`x (median per-box geomean `<gm>`x)" -- the aggregator prints all three; quote them together or not at all. |
| **Accuracy against 50-digit mpmath truth** | Deterministic. Noise cannot touch it. None of the caveats apply. |
| **Duplicate rows returning exactly 0.0** | Deterministic, and the highest-value output here: the guarantee depends on `PL[i,c] + PL[j,c]` cancelling `2 m log m` bitwise, which is a property of one libm and one code generator until it is checked against another. A failure on any box is a real portability defect in the guarantee, not a measurement artifact. |
| **Degradation behaviour** | "on a runner with no numba wheel the package falls back to the NumPy backends and reports it" is a fact about the software, not about the hardware. |

### Not quotable

| quantity | why not |
|---|---|
| **Any absolute millisecond** | Frequency is unobservable inside these guests and the pool mixes microarchitectures behind one label. Times are recorded so ratios can be audited, not to be read. |
| **"The" speedup over SciPy** | SciPy's pdist is single-threaded and fastjsd is parallel, so this ratio scales with usable core count -- 2 vCPU on a standard private-repo runner, more elsewhere -- and is biased upward on a noisy box by min-of-N. A CI SciPy ratio quoted as the speedup is the single most likely way this goes wrong. |
| **Thread scaling** | 2-4 vCPUs that may be SMT siblings on a shared host is not a scaling study; a noisy neighbour is indistinguishable from sublinear scaling. The defensible statement is only "monotone and > 1". Thread scaling should stay a single-machine measurement, stated as such. |
| **A crossover location** | Near the crossover the kernels are by definition within noise. Report the **regret** from using the shipped `SPARSE_DENSITY_MAX` against each box's own optimum, which is portable; the location is not. |
| **Anything about `numba-svml`** | libsvml is not on PyPI and no ARM or Apple host can run it. Every CI figure is the libm path. |
| **A pooled number across configurations** | Legs without numba measure the support-intersection identity rather than the kernel, and on dense inputs can be *slower* than SciPy. That is a real finding, and pooling it into a headline range destroys both claims. Keep the section separate and give it its own sentence. |
| **Any claim about hardware GitHub does not rent** | N runners is not a random sample of computing hardware. Nothing here supports a claim about GPUs, other clouds, bare metal, or CPUs outside the set. |
| **A CPU identified by its runner label** | `ubuntu-latest` is not one machine. The recorded `machine.cpu_model` is the fact; the label is not. That is why the workflow draws `ubuntu-24.04` three times. |

Two model sentences the aggregate report should be able to support. Every
figure in them is a placeholder in angle brackets, filled from an actual
aggregate run:

> fastjsd was faster than SciPy on `<w>`/`<B>` boxes and `<c>`/`<N>` case-box
> pairs; the ratio range is `<min>`x-`<max>`x (median per-box geomean
> `<gm>`x).

> Against Lee-1997 on the same harness, fastjsd won on `<w>`/`<B>` boxes;
> `<r>`/`<N>` cells were resolvable above their own repeat noise, range
> `<min>`x-`<max>`x, and the remaining `<N-r>` (all on `<case>`) were
> indistinguishable.

The second sentence is the one that makes a report credible, because it names
the case whose cells could not be resolved, and "which cells" is the part a
reader cannot reconstruct from a range.

---

## Gotchas

- **Input hashes may differ across boxes, and that is expected.** The two `real`
  fixtures are committed `.npy` and are bit-identical everywhere. The synthetic
  cases go through `np.random.Generator.dirichlet`, whose gamma sampler calls
  libm `log`/`exp`/`pow` -- the very thing being varied -- so they can differ in
  the last ulp on ARM or macOS. The aggregator prints this as a **warning, not
  a rejection**: one ulp of input is irrelevant to a timing and fatal only to a
  paired accuracy claim. Draw accuracy conclusions from the `real` cases.
- **`numba.threading_layer()` raises until a parallel function has actually
  run.** `portable.py` therefore probes every implementation before it builds
  the software block. If you write a new harness, do the same, or the
  fingerprint crashes the run it was supposed to describe.
- **`bench/bench_honest.py` is not cross-machine safe.** It builds its
  implementation list from `baselines.THIRD_PARTY` and calls every entry, so it
  needs pynndescent and torch installed. Use `portable.py` on any box where
  those may be absent.
- **Pin the BLAS thread environment, not `OMP_NUM_THREADS`.** An unset
  `OPENBLAS_NUM_THREADS` on a 2-vCPU box oversubscribes during dataset
  construction. But numba's OpenMP threading layer shares `OMP_NUM_THREADS`,
  and pinning that to 1 silently serializes the kernel under test. The harness
  records the whole block as seen, so check `software.thread_env` in the JSON
  before trusting a thread number.
- **Runner cost is estimated, not measured.** The case set was sized from
  single-machine timings rather than from a dispatch, so the
  `timeout-minutes: 45` in the workflow is the real guard; time the first
  dispatch and size the rest from that. The billing multiplier is documented
  rather than guessed: macOS runners bill at 10x and ARM/Windows differ again,
  so reduce `--repeats` on an expensive leg rather than dropping cases --
  dropping cases changes the case set and breaks per-case aggregation across
  boxes.
