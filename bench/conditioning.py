"""Cancellation in each grouping of the per-coordinate summand.

    python bench/conditioning.py

The quantity to compute is one coordinate's contribution to the divergence,

    t(a, b) = a ln a + b ln b - 2 m ln m,      m = (a+b)/2,

which is non-negative by convexity of x ln x. There are several algebraically
identical ways to arrange it, and they are NOT numerically equivalent. This
script measures, at 50 digits, the cancellation factor

    sum of |each term actually added|  /  t

for the three arrangements that matter. A factor of C means float64 evaluation
can only be trusted to about C * eps relative -- it is the standard
"sum of magnitudes over magnitude of sum" measure of a summation's
conditioning (Higham, ASNA 2nd ed., 1.7).

  ours     a ln a + b ln b - 2 m ln m
           Three terms, each O(a ln a), for a result of O(m u^2). This is the
           arrangement fastjsd's kernels use, and the reason is speed, not
           accuracy: a ln a and b ln b depend on one row each, so they are
           computed once per (row, coordinate) into a table and reused across
           the whole matrix. Exactly one logarithm is left per
           (pair, coordinate).

  scipy    a ln(a/m) + b ln(b/m)
           The grouping in SciPy's compiled pdist kernel (verified bitwise
           against `pdist(X, "jensenshannon")`). Two terms of O(m u), so it
           cancels less -- but neither term can be hoisted out of the pair
           loop, because m depends on the pair. Two logarithms per
           (pair, coordinate), and still unbounded as u -> 0.

  series   m * sum_{n>=1} u^(2n) / (n (2n-1)),   u = (a-b)/(a+b)
           Every term positive, so the factor is exactly 1 at any u. This is
           what `well_conditioned=True` switches to for |u| <= 1/4.

Leading behaviour, confirmed by the table below: ours ~ 4 a ln(1/a) / (m u^2),
scipy ~ 2/u, series ~ 1.
"""

from __future__ import annotations

import argparse

from mpmath import mp, mpf

mp.dps = 50


def terms_ours(a, b, m):
    return [a * mp.log(a), b * mp.log(b), -2 * m * mp.log(m)]


def terms_scipy(a, b, m):
    return [a * mp.log(a / m), b * mp.log(b / m)]


def terms_series(a, b, m, nmax=40):
    u = (a - b) / (a + b)
    return [m * u ** (2 * n) / (n * (2 * n - 1)) for n in range(1, nmax + 1)]


def factor(terms, t):
    return float(sum(abs(x) for x in terms) / t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.1,
                    help="the larger of the two coordinate values")
    args = ap.parse_args()

    a = mpf(args.a)
    print(f"a = {args.a}, reference at {mp.dps} digits")
    print("b = a(1-u)/(1+u), so u = (a-b)/(a+b) exactly\n")
    print(f"{'u':>8}{'t(a,b)':>13}{'ours':>12}{'scipy':>12}{'series':>10}")
    print("-" * 55)
    for e in (1, 3, 6, 9):
        u = mpf(10) ** -e
        b = a * (1 - u) / (1 + u)
        m = (a + b) / 2
        t = a * mp.log(a) + b * mp.log(b) - 2 * m * mp.log(m)
        print(f"{f'1e-{e}':>8}{float(t):>13.2e}"
              f"{factor(terms_ours(a, b, m), t):>12.2e}"
              f"{factor(terms_scipy(a, b, m), t):>12.2e}"
              f"{factor(terms_series(a, b, m), t):>10.2f}")
    print("\nAsymptotes (u -> 0):  ours = 4 a ln(1/a) / (m u^2),"
          "  scipy = 2/u,  series = 1")


if __name__ == "__main__":
    main()
