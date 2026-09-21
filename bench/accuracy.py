"""True error, against 50-digit arithmetic, split by regime.

Agreement with SciPy compares two float64 implementations and so cannot
distinguish "both correct" from "both wrong the same way". This measures
against `mpmath` at 50 digits instead, and splits by input regime, because
the regimes have genuinely different conditioning:

  random          well-separated rows; nothing near a cancellation
  near-duplicate  rows differing by ~1e-9, IDENTICAL support. The regime the
                  exact-zero guarantee is the limit of, and where the sparse
                  kernel's one-sided-mass complement is a subtraction of two
                  nearly equal quantities.
  partial-overlap supports overlapping in about half their entries -- the
                  complement's genuine working regime
  disjoint        no shared coordinate; distance is exactly 1 in base 2
  identical       bitwise-equal rows; distance must be exactly 0

Reported per backend so the dense, sparse and SVML paths can be compared on
the same inputs rather than on whatever the benchmark suite happens to contain.
"""

from __future__ import annotations

import argparse

import numpy as np
from mpmath import mp, mpf

from fastjsd import _svml, available_backends, jsd_matrix

mp.dps = 50


def exact_matrix(P):
    """JS distance, base 2, at 50 digits. O(n^2 k) in mpmath -- keep n small."""
    n, k = P.shape
    rows = [[mpf(float(x)) for x in row] for row in P]
    rows = [[x / sum(r) for x in r] for r in rows]
    ln2 = mp.log(2)

    def xlx(x):
        return x * mp.log(x) if x > 0 else mpf(0)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            acc = mpf(0)
            for a, b in zip(rows[i], rows[j], strict=True):
                m = (a + b) / 2
                acc += xlx(a) + xlx(b) - 2 * xlx(m)
            v = acc / 2 / ln2
            if v < 0:
                v = mpf(0)
            d = mp.sqrt(v)
            D[i, j] = D[j, i] = float(d)
    return D


# --- regimes ---------------------------------------------------------------

def regime(name, n, k, seed):
    rng = np.random.default_rng(seed)
    if name == "random":
        P = rng.dirichlet(np.full(k, 0.4), size=n)
    elif name == "near-duplicate":
        base = rng.dirichlet(np.full(k, 0.4), size=1)
        P = np.repeat(base, n, axis=0)
        P = P * (1.0 + rng.normal(0, 1e-9, P.shape))     # same support
    elif name == "partial-overlap":
        P = rng.dirichlet(np.full(k, 0.4), size=n)
        keep = rng.random(P.shape) < 0.5
        keep[np.arange(n), rng.integers(0, k, n)] = True
        P = P * keep
    elif name == "disjoint":
        P = np.zeros((n, k))
        w = max(1, k // n)
        for r in range(n):
            P[r, r * w:(r + 1) * w] = 1.0 / w
    elif name == "identical":
        P = np.repeat(rng.dirichlet(np.full(k, 0.4), size=1), n, axis=0)
    else:
        raise ValueError(name)
    return np.ascontiguousarray(P / P.sum(axis=1, keepdims=True))


REGIMES = ("random", "near-duplicate", "partial-overlap", "disjoint",
           "identical")

# Fixed per regime. A seed derived from `hash(name)` would not be
# reproducible: string hashing is randomized by PYTHONHASHSEED, so every run
# would draw different fixtures and the errors reported below would move
# between invocations.
SEEDS = {"random": 1301, "near-duplicate": 4507, "partial-overlap": 8821,
         "disjoint": 2003, "identical": 6199}


def errors(P, ref, D):
    ae = np.abs(D - ref)
    nz = ref > 0
    return ae.max(), (ae[nz] / ref[nz]).max() if nz.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=48)
    args = ap.parse_args()

    backends = list(available_backends())
    print(f"n={args.n} k={args.k}, reference = mpmath at {mp.dps} digits")
    if not _svml.available():
        print(f"(numba-svml absent: {_svml.why_unavailable()})")
    print()

    fixtures = {}
    print(f"{'regime':17}{'backend':14}{'max abs err':>13}{'max rel err':>13}"
          f"{'  note'}")
    print("-" * 78)
    for name in REGIMES:
        P = regime(name, args.n, args.k, seed=SEEDS[name])
        ref = exact_matrix(P)
        fixtures[name] = (P, ref)
        off = ~np.eye(args.n, dtype=bool)
        for b in backends:
            D = jsd_matrix(P, backend=b)
            ae, re = errors(P, ref, D)
            note = ""
            if name == "identical":
                note = "exact 0" if np.all(D == 0.0) else "NOT EXACT"
            elif name == "disjoint":
                note = "exact 1" if np.all(D[off] == 1.0) else "not exact"
            print(f"{name:17}{b:14}{ae:>13.2e}{re:>13.2e}  {note}")
        print()

    # The `well_conditioned` A/B, at fixed backend, which is the comparison
    # that isolates the summand from the kernel. The per-backend table above
    # cannot show it: each backend there runs its OWN default (on for the
    # merge, off for the dense loop), so its columns differ in two things.
    print("well_conditioned A/B, backend=numba-sparse (same kernel, both ways)")
    print(f"{'regime':17}{'False':>13}{'True':>13}")
    print("-" * 43)
    for name in REGIMES:
        P, ref = fixtures[name]
        row = []
        for wc in (False, True):
            ae, _ = errors(P, ref,
                           jsd_matrix(P, backend="numba-sparse",
                                      well_conditioned=wc))
            row.append(ae)
        print(f"{name:17}{row[0]:>13.2e}{row[1]:>13.2e}")


if __name__ == "__main__":
    main()
