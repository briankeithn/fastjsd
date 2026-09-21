"""What shape the real fixtures have, and how much of that shape the fixture
pipeline puts there.

    python bench/fixture_shape.py

Two questions this answers.

**Is there block substructure to dispatch on?** The nnz-per-row histogram of
`bench/datasets.py`'s thresholded fixtures is bimodal: a sparse mass, a gap,
and a spike at exactly ``nnz = k``. That looks like "partly sparse, partly
dense blocks" and would justify per-block kernel dispatch. The structure comes
from the thresholding step rather than from the corpus:
`threshold(..., empty="uniform")` resets every row it empties to ``1/k``,
which is the densest row expressible, and the spike is exactly those rows --
this script checks the coincidence row-index by row-index rather than by
counting.

**Is there column structure to split on?** A "core" column (non-zero in every
row) would let the coordinate set be partitioned into a branchless dense core
and a sparse fringe, which is exact by construction because JS is a plain sum
over coordinates. There are none, and there structurally cannot be: a core
column is an AND over ``n`` rows, so its probability is about ``q**n`` for
column occupancy ``q``, and the largest occupancy in either fixture is under
0.3 against ``n`` near 1000.

No timing here -- this is data characterisation, cheap enough to run any time.
"""

from __future__ import annotations

import numpy as np

import datasets

MODES = ("uniform", "argmax")


def describe(P):
    n, k = P.shape
    nnz = (P > 0).sum(1)
    occ = (P > 0).mean(0)
    return {
        "n": n, "k": k,
        "density": (P > 0).mean(),
        "nnz_min": int(nnz.min()), "nnz_med": int(np.median(nnz)),
        "nnz_max": int(nnz.max()),
        "nnz_cv": float(nnz.std() / nnz.mean()),
        "full_rows": int((nnz == k).sum()),
        "core_cols": int((occ == 1.0).sum()),
        "dead_cols": int((occ == 0.0).sum()),
        "occ_max": float(occ.max()),
        "occ_cv": float(occ.std() / occ.mean()),
    }


def main():
    for name in ("soft_clusters_931x25", "soft_clusters_1014x40"):
        P = datasets.real(name)
        n, k = P.shape
        print(f"\n{'=' * 78}\n{name}   n={n} k={k}   raw density "
              f"{(P > 0).mean():.4f}\n{'=' * 78}")

        # Which rows does the cut empty? Recompute it here rather than
        # trusting the label, so the coincidence below is a real check.
        Q = np.array(P, dtype=np.float64, copy=True)
        Q[Q < 1.0 / k] = 0.0
        blank = np.flatnonzero(~Q.any(axis=1))

        stats = {m: describe(datasets.threshold(P, empty=m)) for m in MODES}

        T = datasets.threshold(P, empty="uniform")
        spike = np.flatnonzero((T > 0).sum(1) == k)
        same = np.array_equal(blank, spike)
        print(f"rows emptied by the cut      : {blank.size}")
        print(f"rows at nnz == k after reset : {spike.size}")
        print(f"same rows, index for index   : {same}")
        print(f"identical uniform pairs made : "
              f"{spike.size * (spike.size - 1) // 2}")

        h = np.bincount((T > 0).sum(1), minlength=k + 1)
        occupied = [i for i, c in enumerate(h) if c]
        gap = [i for i in range(min(occupied), max(occupied)) if not h[i]]
        if gap:
            print(f"empty histogram bins between : {gap[0]}..{gap[-1]}  "
                  f"(the 'block' is this gap)")

        print(f"\n{'quantity':<28}" + "".join(f"{m:>14}" for m in MODES))
        rows = [("density", "{:.4f}"), ("nnz_min", "{:d}"),
                ("nnz_med", "{:d}"), ("nnz_max", "{:d}"),
                ("nnz_cv", "{:.3f}"), ("full_rows", "{:d}"),
                ("core_cols", "{:d}"), ("dead_cols", "{:d}"),
                ("occ_max", "{:.3f}"), ("occ_cv", "{:.3f}")]
        for key, fmt in rows:
            print(f"{key:<28}"
                  + "".join(fmt.format(stats[m][key]).rjust(14) for m in MODES))

        from fastjsd._api import SPARSE_DENSITY_MAX
        for m in MODES:
            T2 = datasets.threshold(P, empty=m)
            over = int(((T2 > 0).mean(1) > SPARSE_DENSITY_MAX).sum())
            print(f"rows over SPARSE_DENSITY_MAX ({SPARSE_DENSITY_MAX:.2f}), "
                  f"{m:<8}: {over}")

    print(f"\n{'=' * 78}")
    print("Read `core_cols` before designing any column-partition scheme, and")
    print("`full_rows` before designing any row-block scheme. If the first is")
    print("zero there is no dense core to split off, and if the second equals")
    print("the number of rows the cut emptied, the block is an artifact of")
    print("the thresholding step rather than a feature of the corpus.")


if __name__ == "__main__":
    main()
