"""Benchmark inputs: real membership fixtures plus synthetic generators."""

from __future__ import annotations

import pathlib

import numpy as np

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def threshold(P, factor=1.0, empty="uniform"):
    """Drop entries below ``factor/k`` and renormalize.

    The standard cleanup for soft memberships and topic mixtures, and the
    thing that turns a dense simplex row into a sparse one.

    **The ``empty=`` choice is not cosmetic, and the default manufactures
    structure that is not in the data.** A row whose every entry falls below
    the cut has to become *something*, and the three defensible answers give
    materially different matrices:

    ``"uniform"``
        Reset the row to ``1/k``. This is the default, and it replaces a row
        that had no mass above the cut with a **fully dense** one, which is
        the single densest row possible.
    ``"argmax"``
        Keep the row's largest entry and drop the rest. The row stays sparse
        and keeps its own mode instead of acquiring a uniform one it never
        had.
    ``"drop"``
        Remove the row. Honest, but it changes ``n``, so every cross-variant
        comparison stops being like-for-like.

    On the two real fixtures shipped in ``bench/fixtures`` the default changes
    the matrix substantially (``bench/fixture_shape.py`` regenerates all of
    this):

    ========================  ===============  ===============
    quantity                  931x25           1014x40
    ========================  ===============  ===============
    rows reset by the default  138              42
    their nnz after reset      25 (= k, dense)  40 (= k, dense)
    every other row's max nnz  9                13
    identical pairs created    9453             861
    density, ``uniform``       0.2370           0.1225
    density, ``argmax``        0.0947           0.0821
    nnz/row CV, ``uniform``    1.375            1.573
    nnz/row CV, ``argmax``     0.775            0.772
    ========================  ===============  ===============

    Those reset rows are bitwise identical to each other, so they contribute
    9453 and 861 pairs whose true distance is exactly ``0.0`` -- pairs that
    exercise the duplicate-exactness path and nothing else. The nnz-per-row
    histogram is empty between 10 and 24 on the first fixture and between 14
    and 39 on the second, and the rows in the spike are the rows the cut
    emptied, index for index. The "sparse block plus dense block" shape is
    this function's output, not the corpus's. Under ``"argmax"`` **no row in
    either fixture exceeds** ``SPARSE_DENSITY_MAX``, so every pair dispatches
    to the sparse kernel.

    Row-degree *spread* survives either choice (CV 0.77 under ``"argmax"``, a
    genuine 9-13x range), so the fixtures are still heterogeneous. It is only
    the dense block that is an artifact. ``bench/fixture_shape.py`` prints
    this table and checks the row-index coincidence.
    """
    P = np.array(P, dtype=np.float64, copy=True)
    k = P.shape[1]
    keep = np.argmax(P, axis=1)
    P[P < factor / k] = 0.0
    blank = ~P.any(axis=1)
    if empty == "uniform":
        P[blank] = 1.0 / k
    elif empty == "argmax":
        # `keep` was taken before the cut, so this is the row's own mode.
        # The value is irrelevant -- the renormalization below makes the row
        # one-hot either way.
        for r in np.flatnonzero(blank):
            P[r, keep[r]] = 1.0
    elif empty == "drop":
        P = P[~blank]
    else:
        raise ValueError(f"empty must be 'uniform', 'argmax' or 'drop', "
                         f"got {empty!r}")
    return P / P.sum(axis=1, keepdims=True)


def dirichlet(n, k, alpha, seed=0):
    """Dense simplex rows. Small ``alpha`` concentrates mass on few outcomes."""
    return np.random.default_rng(seed).dirichlet(np.full(k, alpha), size=n)


def sparse_rows(n, k, nnz, seed=0):
    """Rows with exactly ``nnz`` nonzero outcomes at uniformly random positions.

    **Every row gets the same degree, so the row-degree coefficient of
    variation is exactly 0.** That is not a neutral choice, and it is not a
    shape real data has: on the real fixtures the CV is 0.77 under
    ``threshold(empty="argmax")``, spanning 1 to 9-13 non-zeros per row.

    It matters because uniform degrees make the merge's inner trip count
    perfectly predictable, so its branches predict and mispredicts -- which
    otherwise dominate its cost -- largely vanish. Row-degree variance is
    therefore a candidate explanation for which sparse kernel wins, and a grid
    built only from this generator holds that variable fixed at a value real
    data does not take.

    Whether degree variance actually decides the merge/bitmap choice is not
    settled: the separation it predicts has not held up across machines, which
    is consistent with the crossover depending on the branch predictor. Run
    `bench/bits_vs_merge.py` on the host in question rather than assuming
    either answer.

    Use `skewed_rows` alongside this generator so both ends of the range are
    represented.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((n, k))
    for i in range(n):
        cols = rng.choice(k, size=nnz, replace=False)
        P[i, cols] = rng.dirichlet(np.ones(nnz))
    return P


def skewed_rows(n, k, mean_nnz, cv=0.8, seed=0):
    """Sparse rows whose DEGREES vary, at a requested coefficient of variation.

    The companion to `sparse_rows`, and the generator that gives the synthetic
    grid a non-zero row-degree CV. Degrees are drawn from a gamma with the
    requested mean and CV, then clipped to ``[1, k]`` -- gamma because it is
    the maximum-entropy positive distribution at a fixed mean and log-mean, so
    nothing beyond the two requested moments is smuggled in.

    ``cv=0`` reproduces `sparse_rows`' constant degree, which makes the two
    generators a controlled pair: hold mean degree and `k` fixed, sweep `cv`,
    and the only thing that changes is the property that decides dispatch.
    """
    if cv < 0.0:
        raise ValueError(f"cv must be >= 0, got {cv}")
    rng = np.random.default_rng(seed)
    if cv == 0.0:
        degrees = np.full(n, max(1, int(round(mean_nnz))))
    else:
        shape = 1.0 / (cv * cv)
        degrees = rng.gamma(shape, mean_nnz / shape, size=n)
        degrees = np.clip(np.round(degrees), 1, k).astype(int)
    P = np.zeros((n, k))
    for i in range(n):
        z = int(degrees[i])
        cols = rng.choice(k, size=z, replace=False)
        P[i, cols] = rng.dirichlet(np.ones(z))
    return P


# --- shape generators ------------------------------------------------------
#
# The generators above vary n, k and density. These vary the *shape* of the
# support and of the values at fixed density, which is what the sparse
# kernel's two optimizations actually see: the 64-bit occupancy mask fires on
# an empty intersection, and the merge walks `s_i + s_j - |I|`. Neither is a
# function of density alone. `bench/distributions.py` is the harness that
# measures the difference and the reason these exist.


def correlated_rows(n, k, d, rho=0.0, w=None, seed=0):
    """Bernoulli supports sharing a latent block; ``rho`` mixes toward identical.

    Real supports are clustered -- documents share a vocabulary, points share a
    topic -- and `sparse_rows` has none of that. Coordinate ``c`` is nonzero in a
    row of block ``T`` with probability ``(1-rho) d + rho (d/w) [c in T]``, which
    sums to ``k d`` for every ``rho``, so density is held fixed while supports go
    from independent (``rho=0``, which is `sparse_rows` in Bernoulli form) to
    identical within a block (``rho=1``, with the default ``w = d``).

    Counting matched coordinates by column rather than by pair gives, with no
    model at all, ``E|I| = sum_c m_c (m_c - 1) / (n (n-1))`` for column degrees
    ``m_c``. The mean intersection is therefore a function of the column degrees
    ALONE: it is ``k d^2`` here at every ``rho``, and row-to-row correlation
    cannot move it, because correlation does not change a column degree.

    What correlation moves is the SPREAD, and the kernel is built out of things
    that see the spread rather than the mean. Uniform-random supports flatter
    the sparse kernel on one axis and hurt it on another: every row gets the
    same ``nnz``, so the static schedule is perfectly balanced and no pair is
    expensive -- but empty intersections become rare, and an empty intersection
    is the only thing the 64-bit signature can skip. At ``k=25`` and ``d=0.24``
    the real fixture skips 59 % of its pairs where `sparse_rows` skips 15 %.
    """
    rng = np.random.default_rng(seed)
    if w is None:
        w = d
    nblocks = max(1, int(round(1.0 / w)))
    blocks = np.array_split(np.arange(k), nblocks)
    which = rng.integers(0, nblocks, size=n)
    P = np.zeros((n, k))
    for b, cols in enumerate(blocks):
        rows = np.flatnonzero(which == b)
        if rows.size == 0:
            continue
        p = np.full(k, (1.0 - rho) * d)
        p[cols] += rho * d * k / cols.size
        np.clip(p, 0.0, 1.0, out=p)
        S = rng.random((rows.size, k)) < p
        # A Bernoulli row can come out empty. Refill it from the row's own p
        # rather than uniformly, so patching does not dilute the block.
        empty = np.flatnonzero(~S.any(axis=1))
        if empty.size:
            S[empty, rng.choice(k, size=empty.size, p=p / p.sum())] = True
        # The normalized subvector of a flat Dirichlet is a flat Dirichlet on
        # the subset, so masking and renormalizing matches `sparse_rows`.
        P[rows] = S * rng.dirichlet(np.ones(k), size=rows.size)
    return P / P.sum(axis=1, keepdims=True)


def zipf_rows(n, k, nnz, s=1.0, seed=0):
    """``nnz`` columns drawn without replacement with weights ``c**-s``.

    The bag-of-words case, and the only shape here that moves the mean
    intersection off ``k d^2``. Skewed column popularity raises the
    concentration factor ``Phi = sum_c (m_c/n)^2 / (k d^2)``, which is ``>= 1``
    by Cauchy-Schwarz with equality iff all column degrees are equal. At
    ``k=256, nnz=8`` it measures 1.0, 6.5 and 13.6 for ``s`` = 0, 1.0, 1.5: the
    merge does six to fourteen times the logarithms the density predicts, at
    unchanged density, and the signature skip collapses with it. ``s=0`` is
    `sparse_rows`.
    """
    rng = np.random.default_rng(seed)
    # Gumbel top-k: adding Gumbel noise to log-weights and taking the largest
    # ``nnz`` is exactly weighted sampling without replacement, vectorized.
    keys = -s * np.log(np.arange(1, k + 1)) + rng.gumbel(size=(n, k))
    cols = np.argsort(keys, axis=1)[:, -nnz:]
    P = np.zeros((n, k))
    np.put_along_axis(P, cols, rng.dirichlet(np.ones(nnz), size=n), axis=1)
    return P


def dominant_tail(n, k, nnz, head=0.45, seed=0):
    """One coordinate holding ``head`` of a row's mass, the rest a flat tail.

    The real fixtures' row shape, which no Dirichlet reproduces: the mean
    largest-coordinate share is 0.365 at k=25 and 0.461 at k=40, against 0.145
    for ``dirichlet(alpha=0.3)``. It is also the shape that makes `threshold`
    non-monotone in ``factor`` -- on the 931x25 fixture the density runs
    0.321, 0.237, 0.364 for factor 0.5, 1.0, 2.0, because rows the cut empties
    are reset to uniform -- so a density-keyed dispatch inherits a
    discontinuity that Dirichlet inputs never show.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((n, k))
    for i in range(n):
        cols = rng.choice(k, size=nnz, replace=False)
        vals = np.empty(nnz)
        vals[0] = 1.0 if nnz == 1 else head
        if nnz > 1:
            vals[1:] = (1.0 - head) * rng.dirichlet(np.ones(nnz - 1))
        P[i, cols] = vals
    return P


def near_duplicates(n, k, groups, delta=1e-9, alpha=0.4, seed=0):
    """``groups`` clusters of rows differing by ~``delta``, identical support.

    The `well_conditioned` regime. Each coordinate's contribution is computed
    from three pieces of size ``O(a ln a)`` for a result of size ``O(m u^2)``,
    so a pair this close is evaluated at a condition number around 1e18 unless
    the series branch takes over. The two real fixtures hold 556 and 756 pairs
    within 1e-6 of each other and every synthetic case in `suite()` holds
    exactly none, so the branch `_resolve_well_conditioned` turns on by default
    is exercised only by the real half of the suite -- and there only on 0.1 %
    of pairs. This makes the density a knob.
    """
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.full(k, alpha), size=groups)
    idx = np.arange(n) % groups          # every group populated, then shuffled
    rng.shuffle(idx)
    P = base[idx] * (1.0 + rng.normal(0.0, delta, size=(n, k)))
    return P / P.sum(axis=1, keepdims=True)


def with_duplicates(P, frac=0.1, seed=0):
    """Replace ``frac`` of ``P``'s rows with exact copies of rows it keeps.

    The exact-zero guarantee at benchmark scale, as a knob rather than an
    accident. Identical rows must give exactly 0.0, not 1e-16 of dust that
    ``sqrt`` would amplify to 1e-8, and the sparse kernel gets there by summing
    one-sided mass in the same order ``_row_sums`` did.

    `threshold` already manufactures duplicates of its own: the suite's two
    thresholded cases carry 12762 and 2691 exactly-equal pairs. They arrive
    two ways -- rows the cut empties are reset to uniform ``1/k`` (138 and 42
    rows), and rows the cut leaves with one entry renormalize to the same
    one-hot vector (3309 and 1830 further pairs). So the guarantee is already
    exercised by the benchmark suite, but only at a fraction that is not
    chosen; this function makes the fraction explicit.
    """
    P = np.array(P, dtype=np.float64, copy=True)
    n = P.shape[0]
    ndup = int(round(frac * n))
    if ndup < 1 or ndup >= n:
        return P
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    victims, keepers = order[:ndup], order[ndup:]
    P[victims] = P[rng.choice(keepers, size=ndup)]
    return P


def row_mass(P, spread=20.0, seed=0):
    """Rescale each row by a random factor spanning ``spread``; rows stop summing
    to 1.

    No kernel can see this -- every entry point normalizes -- but `threshold`
    runs BEFORE normalization and compares raw values against ``factor/k``, so
    it is not scale-invariant, and a row whose raw mass is small is wiped
    entirely and reset to uniform. Both real fixtures ship un-normalized, with
    raw row sums spanning 0.077-1.0, and at the suite's own ``factor=1.0`` this
    empties 138 of 931 rows and 42 of 1014. It is what makes `threshold`
    non-monotone in ``factor`` -- density on the 931x25 fixture runs 0.321,
    0.237, 0.364, 0.575 for factor 0.5, 1.0, 2.0, 4.0 -- and every generator in
    this file returns normalized rows, so none of them reproduces it. The
    missing axis is row SCALE, not row shape.
    """
    P = np.asarray(P, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s = np.exp(rng.uniform(0.0, np.log(spread), size=P.shape[0]))
    return P * (s / s.max())[:, None]


def mixed_density(n, k, frac_dense=0.5, nnz_sparse=2, alpha=0.3, shuffle=True,
                  seed=0):
    """Dense rows and near-empty rows in one matrix; dispatch sees only the mean.

    `resolve_backend` keys on a single global density, so a matrix that is half
    dense and half ``nnz=2`` is dispatched as though every row were half full.
    The per-step cost of getting that wrong is bounded -- the merge walks
    ``s_i + s_j - |I| <= (2d - d^2) k`` steps, never more than the dense
    kernel's ``k`` below the threshold -- but the VARIANCE is not bounded: the
    dense-vs-dense sub-block does ``k`` logarithms per pair while the
    sparse-vs-sparse sub-block does none, and `_balanced_order` balances pair
    count, not work. ``shuffle=False`` leaves the dense rows contiguous, which
    is exactly the same work under a worse chunking of the thread schedule.
    """
    rng = np.random.default_rng(seed)
    ndense = int(round(frac_dense * n))
    P = np.zeros((n, k))
    if ndense:
        P[:ndense] = rng.dirichlet(np.full(k, alpha), size=ndense)
    for i in range(ndense, n):
        cols = rng.choice(k, size=nnz_sparse, replace=False)
        P[i, cols] = rng.dirichlet(np.ones(nnz_sparse))
    if shuffle:
        P = P[rng.permutation(n)]
    return P


def real(name):
    return np.load(FIXTURES / f"{name}.npy")


#: The synthetic grid. Powers of two in both axes, and densities that are
#: powers of two as a fraction of ``k``, so every point is a stated choice.
#: ``n`` is fixed at 2048 across the grid so that ``k`` and density are the
#: only things varying; the size sweep lives in ``bench/scale.py``, which is
#: where n belongs.
#:
#: Density is set by non-zeros per row, ``z``, not by a Dirichlet
#: concentration: `z` is the quantity the merge's cost actually depends on, and
#: fixing it makes the sparse columns comparable across `k`.
SYNTHETIC_N = 2048
#: k brackets where Jensen-Shannon is actually used. Topic mixtures and soft
#: cluster memberships have tens of components, not thousands: the real
#: fixtures shipped here are k=25 and k=40. Stretching the grid far above that
#: range spends cells outside the regime the metric is used in, and costs wall
#: clock in proportion, because SciPy's kernel is O(k) and single-threaded on
#: the dense cells.
SYNTHETIC_K = (16, 64, 256)
SYNTHETIC_Z = (None, 16, 4)          # None = dense
#: Row-degree coefficient of variation for the skewed cells. 0.8 is not a
#: tuning parameter -- it is what the two real fixtures measure (0.775 and
#: 0.772) once `threshold(empty="argmax")` stops manufacturing a dense block.
#: `sparse_rows` fixes CV at 0, which no real corpus does, and dispatch turns
#: on this quantity, so the grid needs both ends.
SYNTHETIC_CV = 0.8


def synthetic():
    """The primary grid: clean sizes, one factor varied at a time.

    Dense rows are Dirichlet with alpha = 1, which is the uniform distribution
    over the simplex -- the choice with no free parameter.
    """
    out = []
    for k in SYNTHETIC_K:
        for z in SYNTHETIC_Z:
            if z is None:
                P = dirichlet(SYNTHETIC_N, k, 1.0, seed=k)
                out.append((f"synth n={SYNTHETIC_N} k={k:<4} dense", P))
            elif z < k:
                P = sparse_rows(SYNTHETIC_N, k, z, seed=k * 1000 + z)
                out.append((f"synth n={SYNTHETIC_N} k={k:<4} z={z} "
                            f"({z / k:.1%} nz)", P))
    # Skewed-degree cells, alongside the uniform-degree cells above rather than
    # in place of them: keeping both is what makes each pair controlled. Same
    # mean degree and same k, so a skewed cell and its uniform counterpart
    # differ only in row-degree CV -- the property `sparse_rows` fixes at 0,
    # real data puts near 0.8, and sparse dispatch turns on.
    for k in SYNTHETIC_K:
        for z in SYNTHETIC_Z:
            if z is not None and z < k:
                P = skewed_rows(SYNTHETIC_N, k, z, cv=SYNTHETIC_CV,
                                seed=k * 1000 + z + 7)
                out.append((f"synth n={SYNTHETIC_N} k={k:<4} z~{z} "
                            f"cv={SYNTHETIC_CV}", P))
    return out


def case_study():
    """Two real HDBSCAN soft-membership matrices, dense and thresholded.

    Kept separate from `synthetic()` on purpose. Their shapes -- 931x25 and
    1014x40 -- are not chosen, they are however many documents those corpora
    had and however many clusters HDBSCAN found, so they cannot be varied and
    nothing can be swept over them. They are validation that the synthetic
    result transfers to data with real structure (unequal row masses,
    correlated supports, exact duplicates after thresholding), not the primary
    evidence. `bench/distributions.py` quantifies how they differ.
    """
    out = []
    for name in ("soft_clusters_931x25", "soft_clusters_1014x40"):
        P = real(name)
        n, k = P.shape
        out.append((f"real n={n} k={k} dense", P))
        T = threshold(P)
        out.append((f"real n={n} k={k} thresholded ({(T > 0).mean():.0%} nz)",
                    T))
    return out


def suite():
    """Everything, synthetic first, for callers that want a single list."""
    return synthetic() + case_study()
