# The identity

Everything in this package follows from one rearrangement of the
Jensen-Shannon divergence and its two consequences. Both consequences are
exact -- nothing here is an approximation or a tolerance.

## Definitions

For discrete distributions `p, q` over `k` outcomes with `sum p = sum q = 1`,
and `m = (p + q) / 2`:

```
JS(p, q) = H(m) - (H(p) + H(q)) / 2,        H(x) = - sum_c x_c ln x_c
```

with the usual convention `0 ln 0 = 0`. The **distance** is `sqrt(JS)`,
which is a metric (Endres & Schindelin 2003; Osterreicher & Vajda 2003).
Dividing by `ln(base)` converts units; `base = 2` bounds the distance in
`[0, 1]`, which is why it is the default here -- `1 - d` is then a
similarity. SciPy's `jensenshannon` defaults to base `e` instead, so its
values are this package's times `sqrt(ln 2) ~ 0.8326`.

## The rearrangement

Expand the three entropies and collect per coordinate:

```
JS(p, q) = (1/2) * sum_c [ p_c ln p_c + q_c ln q_c - 2 m_c ln m_c ]
```

Write `t(a, b) = a ln a + b ln b - 2 m ln m` for the summand.

### Consequence 1 -- no cancellation

`t(a, b) >= 0` for every coordinate, with equality iff `a == b`.

*Proof.* `x ln x` is convex, so `((a ln a) + (b ln b)) / 2 >= m ln m` with
`m = (a + b) / 2`. Multiply by 2. QED

So `JS` is a **sum of non-negative terms**: the SUM has no cancellation and
condition number exactly 1. That is a statement about the sum only; it does
not extend to "accurate all the way down to `JS = 0`". Each individual term
is *evaluated* as a difference of three quantities much larger than itself,
and for `a ~ b` that evaluation is ill-conditioned. See
[What is actually guaranteed](#what-is-actually-guaranteed) below, which
states the bounds and the fix.

The **textbook** grouping `H(m) - (H(p)+H(q))/2` and SciPy's `rel_entr`
grouping both already return exactly `0.0` for bitwise-identical rows: with
`m = (p+q)/2` computed as bitwise `p`, the expression is an exact
self-subtraction. Measured over 300 trials at k=192: max distance `0.0e+00`
for both. So against SciPy this form is **parity, not an improvement**.

Where it does help is against the *constant-folded* forms --
`ln 2 + (A_p + A_q - sum_c s_c ln s_c)/2` and its sparse relatives -- which
fold the constant *outside* the sum so that it must cancel numerically
rather than symbolically. That variant returns `3.35e-08` on the same 300
trials. Folding the constant out front is how the published sparse forms are
written (Lee 1997 Eq. 2.8; Dagan-Lee-Pereira 1999), so a sparse
implementation transcribed from the literature inherits it. Keeping the terms
unfolded is what buys the guarantee back. The fault is in the *arrangement*,
not in Lee's support-intersection identity, which is exact and which this
package uses unchanged;
[Consequence 2c](#consequence-2c----the-identity-is-exact-the-arrangement-is-not)
separates the two and measures each.

**And unfolding alone is not sufficient.** The cancellation is
`PL[i,c] + PL[j,c] - 2 m log m`, exact only when both sides come from the
*same* `log`. NumPy's vectorized `log` and the libm `log` a compiled kernel
calls in the loop disagree by 1 ULP on a minority of inputs -- 3 of 96
Dirichlet draws, roughly one coordinate in each -- and that is enough to lose
the exact zero: precomputing `PL` with the one while the mixture term uses the
other returns `~6e-9` on duplicate rows, where a backend that uses one log
throughout stays exact. `_numba.xlogx_like_kernels` therefore builds `PL` with
the kernels' own `log`; it costs `O(nk)` against the loop's `O(n^2 k)`, so it
is free, and `test_identical_rows_exact_when_the_two_logs_disagree` pins it to
the seeds where the two implementations differ.

The same requirement applies a second way, on **precision**. The inner loop is
float64 whatever the input dtype, because the `0.5` and `2.0` literals promote
`m` and everything downstream of it. A float32 `PL` therefore carries a
rounding the float64 mixture term does not, and float32 duplicate rows then
land at ~1e-4 rather than zero. `PL` is float64 regardless of input dtype;
the final distance is cast on the way out, after the cancellation has already
happened.

The requirement generalizes past this package: an exactness guarantee that
relies on a term cancelling itself also relies on every occurrence of that
term being computed by identical code. Two `log`s that are both correctly
rounded to within 1 ULP are not the same `log`.

### Consequence 2 -- sparsity, for free and exactly

If exactly one side is nonzero, the coordinate's term is `ln 2` times that
side's mass:

```
t(a, 0) = a ln a - 2 (a/2) ln (a/2)
        = a ln a - a (ln a - ln 2)
        = a ln 2
```

and `t(0, 0) = 0`. So:

```
JS(p, q) = (1/2) * [  sum over supp(p) & supp(q) of t(p_c, q_c)
                    + ln 2 * (mass of p outside supp(q)
                              + mass of q outside supp(p)) ]
```

A pair needs **logarithms only on its support intersection**. Everything
else is one multiplication by `ln 2` at the end. For rows of density `d`
with independent supports, the expected intersection is `k d^2` rather than
`k` -- a quadratic reduction in work, exactly, with no threshold or error
bound involved.

The one-sided mass must NOT be recovered as `1 - matched`: the row sums are
not exactly 1 after normalization, so that subtraction reintroduces exactly
the cancellation Consequence 1 removed and duplicate rows stop being exactly
zero. Recovering it against the *measured* row sums is safe, but only under a
condition on summation order -- see Consequence 2b.

### Consequence 2b -- recovering the one-sided mass without losing the zero

Consequence 2 lets a pair skip non-intersecting coordinates, but the one-sided
mass still has to be counted. Accumulating it during the merge costs a load and
an add per skipped entry, which on sparse rows is nearly the whole kernel.

Recovering it by complement instead reintroduces a subtraction. Two conditions
make that subtraction exact rather than merely small. Accumulate each row's
matched mass **separately**, and sum the row totals **sequentially in stored
column order**:

    one = (S_i - hit_a) + (S_j - hit_b)

`hit_a` then sums row i's matched entries in exactly the order and over exactly
the set that `S_i` sums whenever every entry matches, so the two are bitwise
equal and `one` is exactly `0.0` -- for **any values**, not only for identical
rows. That is strictly stronger than what a single accumulator gives (G5).

What the conditions do *not* buy is the duplicate-row zero: a single combined
accumulator is exact on duplicates too, 1000/1000 in `bench/lee_arrangement.py`
(Consequence 2c), because on duplicates the residue enters through a factor
that is itself zero. The claim in this section is about accuracy **near** zero,
not at it. A single `hit += a + b` associates the additions differently from
`S_i` and leaves `~k eps`; harmless for well-separated rows, but for
near-duplicates the true divergence is `O(u^2)` and that residue swamps it.
Measured on rows differing by `1e-9`, worst absolute error in the distance
against 50-digit arithmetic:

| `k` | 1 accumulator | 2 accumulators |
|---:|---:|---:|
| 32 | 1.5e-08 | 1.2e-08 |
| 64 | 2.3e-08 | 1.6e-08 |
| 128 | 1.7e-08 | 1.4e-08 |
| 256 | 1.9e-08 | 8.1e-09 |

Both columns are far above the `1.8e-17` G7 reports for the shipped merge:
that figure is the `well_conditioned` series path, which addresses the
*summand*'s conditioning and is a separate mechanism from the accumulator
split. The comparison that isolates the split is the one in the table.

`np.sum` breaks the second condition for the same reason the first one
matters: it reduces in a divide-and-conquer tree rather than left to right
(what NumPy calls pairwise summation -- no relation to this package's
*pairwise*, which means over all pairs of rows). `_row_sums_nb` sums
sequentially.

Emptiness can also be decided before the merge starts. Two rows share no
coordinate iff their occupancy masks AND to zero. The mask is one to four
64-bit words, widened to cover `k`, and column `c` sets bit
`c % (64 * words)`. A fold produces false positives only -- never false
negatives -- so skipping on a zero AND is exact: it leaves `acc = 0` and no
matched mass, which is what the merge would have left. For `k <= 256` there is
no fold at all and the test is exact rather than merely conservative. The
expected intersection size is `k d^2`, so at `k=256, d=3 %` this skips most
pairs.

### Consequence 2c -- the identity is exact; the *arrangement* is not

This package's exactness claim is stated against "the published sparse form".
That form has two separable parts, and only one of them fails.

Lee's contribution (1997, TR-11-97) is the **identity**: `t(a,0) = a ln 2`, so
logarithms are needed only on the support intersection. That is Consequence 2
above, it is exact, and this package uses it unchanged. What Lee's Eq. (2.8)
does *in addition* is arrange the identity with the constant folded outside
the sum,

    JS(p,q) = ln 2 + (1/2) SUM_{c in supp(p) & supp(q)} [ a ln(a/s) + b ln(b/s) ],
                                                                       s = a + b

which is the natural way to write it once every one-sided coordinate is known
to contribute `a ln 2` and the row masses sum to one. It is also how
Dagan-Lee-Pereira (1999) and Connor et al. (2013) write it, so a sparse
implementation transcribed from the literature inherits the arrangement along
with the identity.

The arrangement is what breaks the exact zero. On duplicate rows the folded
form asks a sum of `k` rounded terms to cancel a constant it never met
symbolically: every summand is exactly `-2a ln 2` (`a/(2a) == 0.5` and
`log(0.5) == -ln 2` are both bitwise exact in IEEE-754, so no individual term
is at fault), but their *accumulation* lands a few ULP away from `-ln 2`, and
`sqrt` turns `~1e-16` into `~1e-8`. The clamp `if v < 0: v = 0` that every
implementation needs absorbs the residues that fall short, which is why the
failure rate is a fraction rather than all of them.

Measured by `bench/lee_arrangement.py --draws 125`, holding Lee's identity
fixed and varying only the arrangement -- 1000 duplicate-row draws,
`k = 8..512`, shared support of `k/4` non-zeros:

| arrangement | duplicate rows return exactly 0.0 |
|---|---|
| Eq. (2.8), constant folded out | **fails 262/1000**, worst `2.8e-08` |
| unfolded, one-sided mass summed directly | exact, 1000/1000 |
| unfolded, one-sided mass by complement, 1 accumulator | exact, 1000/1000 |
| unfolded, one-sided mass by complement, 2 accumulators | exact, 1000/1000 |

Run with `--mp`, the same script checks first that `bench/baselines.py`'s
Eq. (2.8) baseline is a faithful transcription (worst `1.1e-16` against 50-digit
mpmath), so the top row is the literature's arrangement and not a local bug.

Two consequences for how the claim should be read:

- **Unfolding alone is sufficient** for the exact zero. All three unfolded
  arrangements pass. The two-accumulator complement of Consequence 2b is
  therefore justified by *speed* -- summing the one-sided mass directly costs a
  load and an add per skipped entry, and on sparse rows skipped entries are
  nearly all of the work -- and by near-duplicate accuracy, not by duplicate
  exactness.
- **The claim is against the published arrangement, not against Lee's
  identity.** The identity is exact, and a sparse implementation that keeps
  the terms inside the sum gets both the intersection-only logarithms and the
  exact zero.

## What this buys, mechanically

Precompute `x ln x` once per entry (`O(n k)`) -- with `_common.xlogx` for
the NumPy backends and `_numba.xlogx_like_kernels` for the compiled ones,
which must not be swapped for each other (Consequence 1). Then the
inner loop over a pair's coordinates is:

```
m    = 0.5 * (a + b)
term = PL[i, c] + PL[j, c] - 2 * m * ln(m)
```

-- **one** logarithm, two loads, no allocation. That is what makes the loop
compilable and threadable, and it is the whole difference from the NumPy
broadcast approach, which spends its time moving an `(n, n, k)` tensor
through memory (see `PRIOR_ART.md`).

## What is actually guaranteed

Each claim below is stated with its proof, or with the measurement that
supports it. `eps` is the float64 unit roundoff, `2^-53 ~ 1.11e-16`.

**G1 -- every summand is non-negative.** `t(a,b) >= 0`, with equality iff
`a == b`. *Proof:* `x ln x` is convex, so `(a ln a + b ln b)/2 >= m ln m` for
`m = (a+b)/2`; multiply by 2. QED

**G2 -- the summation over coordinates is perfectly conditioned.** By G1 the
terms are all of one sign, so `sum |t_c| = sum t_c = 2 JS`, and the condition
number of the sum is exactly 1. Sequential summation of `k` non-negative terms
therefore satisfies the standard bound (Higham, *ASNA* 2nd ed., section 4.2)

        |computed - exact| <= (k-1) eps / (1 - (k-1) eps) * exact

so the *relative* error of the sum is at most about `(k-1) eps` -- 5.7e-14 at
k=512 -- and it cannot be worse than that however small `JS` is. This is the
real content of "no cancellation": it is a statement about the SUM.

**G3 -- the square root improves things.** `sqrt` is correctly rounded in
IEEE-754 (section 5.4.1) and halves relative error:
`rel(sqrt x) = rel(x)/2 + O(eps)`. So the distance is better conditioned than
the divergence, not worse.

**G4 -- identical rows give exactly 0.0.** *Proof:* `fl(a+a) = 2a` and
`fl(2a * 0.5) = a` are both exact (scaling by a power of two), so `m == a`
bitwise. `PL[i,c]` and `PL[j,c]` are the same stored value `p`, so
`fl(p + p) = 2p` exactly, and the mixture term is `fl(fl(2 * a) * L)` for the
same `L = log(a)`, which is `fl(2 * fl(a * L)) = 2p`. The difference is
`fl(2p - 2p) = 0`, a sum of zeros is zero, and `sqrt(0) = 0`. QED
The proof has two side conditions. `PL` and the kernel must use the *same*
`log` at the *same* precision (hence `xlogx_like_kernels`), and the compiler
must not contract `PL_i + PL_j - 2*m*L` into an FMA -- `fma(-2a, L, 2p)`
rounds the product once instead of twice and returns `2*(p - a*L)`, the
product's rounding error, rather than zero. This is why the kernels are
compiled `fastmath=False`.

**G5 -- the sparse kernels' one-sided mass is exactly 0.0 when the supports
match.** One-sided mass is recovered as `(rsum[i] - hita) + (rsum[j] - hitb)`.
`hita` accumulates row i's matched entries in stored column order, which is
exactly the order and the set `rsum[i]` sums when every entry matches, so the
two are bitwise equal and the difference is exactly zero -- for *any* values,
not only for identical rows. QED This needs two accumulators: a single
`hit += a + b` associates differently and leaves `~k eps`, and it needs
left-to-right summation in `_row_sums_nb`, since `np.sum` reduces in a tree.

**G6 -- disjoint supports give exactly 1.0 in base 2 for SOME inputs, and this
is not a guarantee.** With an empty intersection the merge contributes nothing
and the result is `sqrt(0.5 * ln2 * (S_i + S_j) / ln2)`. Whether that lands on
exactly 1.0 is tested, not proved, and it has **two** independent causes:
normalized row sums that are not exactly 1.0, and the rounding of the
one-sided accumulation itself.

Over 40 random disjoint-support matrices (n=4..20, 1..8 non-zeros per row,
weights drawn from a Dirichlet rather than uniform), 5460 off-diagonal pairs.
The second column restricts to the matrices whose normalized row sums are all
exactly 1.0 -- 1946 of the 5460 pairs -- which rules normalization out:

| backend | all pairs | row sums exactly 1.0 |
|---|---:|---:|
| `numba-svml` | 27.1 % | 19.0 % |
| `numba` | 13.6 % | 11.0 % |
| `numpy` | 12.1 % | 8.3 % |
| `numba-sparse` | 4.1 % | 0.8 % |
| `numpy-sparse` | 4.1 % | 0.8 % |

Always by 1-2 ULP, and always low. So inexact row sums do contribute -- they
account for roughly a third of the dense kernels' failures and most of the
sparse ones -- but the residual is the accumulation of the one-sided terms:
`SUM_c a_c ln a_c` has to land on exactly `-S_i ln 2` after rounding, and
whether it does depends on the weights and on the summation order. The sparse
kernels fail least, and are nearly immune once row sums are exact, because
their complement recovers `S_i + S_j` directly rather than re-summing the
coordinates. Uniform weights `1/w` with `w` a power of two happen to work,
which is why every fixture this package ships hits the exact value --
`np.eye(n)` is `w=1`, and `bench/accuracy.py`'s disjoint regime uses
`w = max(1, k//n)`.

So `jsd_matrix(np.eye(4))` does give exactly 1.0, and the property holds on
every fixture this package ships, but **do not rely on it for arbitrary
input**. What is safe is `abs(d - 1.0) <= 2 * eps`. G4 (duplicate rows give
exactly 0.0) is unaffected -- that one is a genuine bitwise cancellation, not
a rounding coincidence.

**G7 -- measured error against 50-digit arithmetic** (`bench/accuracy.py`,
n=24, k=48, worst over all backends):

| regime | max abs | max rel |
|---|---:|---:|
| random | 5.6e-16 | 8.3e-16 |
| partial overlap | 3.3e-16 | 4.2e-16 |
| disjoint | 0 (exact) | 0 |
| identical | 0 (exact) | 0 |
| **near-duplicate (delta ~ 1e-9)** | **2.4e-08** | **6.0e+01** |

Worst-over-backends is the pessimistic reading of that last row: it is the
NumPy kernels, which do not implement `well_conditioned`. The `numba-sparse`
merge, which is what sparse input dispatches to, returns **1.8e-17** on the
same fixture. The dense Numba and SVML kernels sit at 1.5e-08 with the option
off, which is their default -- see below.

### The limitation

G1 and G2 are about the sum. **They say nothing about the conditioning of an
individual term**, and they do not imply that the result stays accurate all
the way down to `JS = 0`. `t(a,b)` is *evaluated* as a difference of three
quantities each of size `O(a ln a)`, while `t` itself is `O(m u^2)` for
`u = (a-b)/(a+b)`.

This is a property of **this grouping**, not of the divergence, and there is
no standard evaluation to contrast it against. The reason this package uses
the grouping anyway is speed: `a ln a` and `b ln b` each depend on one row, so
they tabulate once per (row, coordinate) and leave exactly **one** logarithm
per (pair, coordinate). SciPy's compiled kernel groups it as
`a ln(a/m) + b ln(b/m)` instead (verified bitwise against
`pdist(X, "jensenshannon")`), which cancels less -- but `m` depends on the
pair, so neither logarithm can be hoisted, and it pays **two** per
(pair, coordinate). Cancellation factor `sum|pieces| / t` at `a = 0.1`,
50-digit reference, from `bench/conditioning.py`:

| `u = (a-b)/(a+b)` | `t` | this grouping | SciPy's | series |
|---:|---:|---:|---:|---:|
| 1e-1 | 9.1e-04 | 9.6e+02 | 2.0e+01 | 1.00 |
| 1e-3 | 1.0e-07 | 9.2e+06 | 2.0e+03 | 1.00 |
| 1e-6 | 1.0e-13 | 9.2e+12 | 2.0e+06 | 1.00 |
| 1e-9 | 1.0e-19 | 9.2e+18 | 2.0e+09 | 1.00 |

Asymptotically `4 a ln(1/a) / (m u^2)` here against `2/u` for SciPy's: the
cheap grouping is the one that cancels worse, but **neither is safe** as
`u -> 0`. Only the series (next section) is. The index above is
`u = (a-b)/(a+b)`, not `b/a - 1`: the two differ by a factor of about 2, and
since the factor grows like `1/u^2`, a table indexed by the latter reports
figures about 4x larger against the same label.

So G4's exact zero is a single point, not the limit of a smooth
approximation: at `a == b` the answer is exactly right, and at `u = 1e-9` it
is essentially rounding.

### Removing it: the symbolic cancellation

Substituting `a = m(1+u)`, `b = m(1-u)` makes `ln m` cancel *algebraically*
rather than numerically, because its coefficient `(1+u) + (1-u) - 2` is
identically zero:

        t(a,b) = m [ (1+u) ln(1+u) + (1-u) ln(1-u) ]
               = a log1p(u) + b log1p(-u)
               = m * SUM_{n>=1} u^(2n) / ( n (2n-1) )
               = m ( u^2 + u^4/6 + u^6/15 + u^8/28 + ... )

All three forms agree with the definition to 1e-39 in 50-digit arithmetic.
The **series** form is the useful one: every term is positive, so it inherits
G2's condition number of 1 at the level of the individual summand, and it is
accurate to a few `eps` however small `u` is. It is also cheaper than a
logarithm -- 13 fused multiply-adds, no libm call.

It is not a replacement, because the series diverges as `u -> 1` (a one-sided
coordinate is `u = 1`) and needs ~22 terms to cover `|u| <= 1/2`. So
`well_conditioned=True` switches on `|u| <= 1/4`, tested as
`|a - b| <= s/4` so that the common path never pays for a division it
discards:

| `|u|` | form | error |
|---|---|---|
| `<= 1/4` | series, 13 terms | `~eps` |
| `> 1/4` | `PL_i + PL_j - 2 m log m` | condition number `<= ~145`, so `~3e-14` |

Measured effect on the near-duplicate regime: **1.5e-08 -> 1.8e-17**
absolute, with every other regime unchanged. It is off by default **on the
dense loop only**, where the extra branch in the inner loop costs a large
fraction of the loop's runtime and is not removable by cheapening the test
(making the branch division-free changes nothing measurable). On the sparse
merge the branch is essentially free, because matched coordinates are rare
there, so it is **on by default** there. See `_resolve_well_conditioned`.

One further consequence: `a == b` gives `u == 0` exactly, which takes the
series branch and returns `m * 0 * P(0) = 0` without reading `PL` at all. With
`well_conditioned=True`, G4 no longer depends on its two side conditions.


### Can the symbolic form be made cheaper? Measured: no, not meaningfully

The series hybrid's cost on the dense loop has three candidate remedies, and
none of them removes it.

**1. The cost is the BRANCH, not the polynomial.** Replacing the 13-term Horner
chain with a single multiply -- same test, same branch, wrong answer --
recovers almost none of the cost. So the arithmetic is nearly free and the
penalty is the data-dependent branch itself: it is roughly 50/50 on ordinary
rows, so it does not predict, and it stops the compiler keeping the loop
tight. Nothing about the series body can recover that.

**2. Fewer terms at a tighter threshold buys a few per cent and costs 1800x
accuracy.** Taylor needs 12 terms for `|u| <= 1/4`, 8 for 1/8, 6 for 1/16, 4
for 1/64. On ordinary inputs all four are indistinguishable (2.6e-17
near-duplicate, ~2e-16 elsewhere), and the shorter chains recover a few per
cent of the branch cost. But the seam moves with `tau`, and on input
deliberately concentrated just above it:

| input | `tau=1/4`, 12 terms | `tau=1/64`, 4 terms |
|---|---:|---:|
| `\|u\|` just over 1/4 | 5.3e-16 | 1.0e-15 |
| `\|u\|` just over 1/64 | **5.2e-18** | **9.4e-15** |

which is what the log path's condition number at the seam predicts (145 at
`tau=1/4`, 37723 at `tau=1/64`). Trading 1800x in the band the flag exists to
protect, for a few per cent of one kernel, is the wrong direction.
`tau = 1/4` stays.

**3. A branch-free closed form: the same substitution gives a second identity,**

        phi(u) = ln(1 - u^2) + 2 u atanh(u)

collecting the even part as `ln((1+u)(1-u))` and the odd part as
`u ln((1+u)/(1-u))`. It is exact (agrees to 1e-44 at 45 digits) and needs no
threshold, no series and no branch -- and for small `u` the two pieces are
`-u^2` and `+2u^2`, giving a cancellation factor of **exactly 3** where the
shipped form has `1/u^2`. `libsvml` even exports `__svml_log1p8` and
`__svml_atanh8`, so it vectorizes.

It still loses, for two reasons that only appear at the *other* end:

- It is the **mirror** of the problem it solves. The factor is 3.0 up to
  `u ~ 0.5` but grows like `log(1/(1-u))` after that -- 6.9 at `u=0.99`, 40 at
  `u = 1 - 1e-12` -- and well-separated coordinates live there, so sampling
  that stops at `u = 0.5` misses the regime that decides the question.
- Worse, it fails by **representation**, not conditioning. For `u = 1 - 2e-19`,
  `u*u` rounds to exactly `1.0`, so `ln(1-u^2)` is `-inf` and `atanh(u)` is
  `+inf`. Lopsided coordinates -- routine in Dirichlet or thresholded data --
  hit this. Measured on ordinary random rows: **7.0e-11** absolute error,
  five orders of magnitude worse than the shipped kernel.

Computing `1 - u^2` accurately means computing `(1-u)(1+u) = 4ab/s^2`, at which
point the form collapses to `a log1p(u) + b log1p(-u)` -- two logarithms, and a
cancellation factor of `2/|u|` rather than 3. Better than `1/u^2`, worse than
the series, and twice the transcendental cost. Not shipped.


## Implementation notes that are not obvious

- **Symmetry must be arranged, not assumed.** `D[a,b]` and `D[b,a]` are the
  same sum in a different association order, and floating-point addition is
  not associative. The NumPy kernel groups `PL_i + PL_j` before combining
  with the mixture term so both orders produce identical bits; the Numba
  kernels compute each pair once and write both cells. A distance matrix
  that is only symmetric to 1 ulp breaks callers that compare it to its own
  transpose.
- **Balanced parallel decomposition.** Row `i` of the upper triangle owns
  `n-1-i` pairs, so a plain triangular `prange` gives the first thread
  ~2x the work of the last. Iterating rows in the order `0, n-1, 1, n-2, ...`
  makes every contiguous chunk carry the same total, which is what lets
  parallel scaling approach the thread count.
- **The diagonal is set, not computed.** Even in the per-coordinate form
  the diagonal is exactly zero by Consequence 1, but writing it explicitly
  costs nothing and documents the intent.
- **Rows are normalized first.** Both consequences assume `sum p = 1`; an
  unnormalized row does not give a slightly wrong answer, it gives a
  meaningless one. `normalize=False` therefore *checks* the row sums rather
  than trusting them, unless checking is also switched off.
- **Empty columns are dropped** before dispatch. A coordinate that is zero
  in every row cannot affect any pair, and dropping it shrinks the inner
  loop -- which matters on vocabulary-shaped data where most columns are
  unused.

## References

- Lin, J. (1991). Divergence measures based on the Shannon entropy.
  *IEEE Transactions on Information Theory* 37(1), 145-151.
- Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability
  distributions. *IEEE Transactions on Information Theory* 49(7),
  1858-1860.
- Osterreicher, F., & Vajda, I. (2003). A new class of metric divergences
  on probability spaces and its applicability in statistics. *Annals of the
  Institute of Statistical Mathematics* 55(3), 639-653.
