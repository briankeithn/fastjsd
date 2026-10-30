"""The float32 backend, numba-f32: accuracy, exactness, and (optionally) speed.

    python bench/f32.py            # timing-free: accuracy and exactness
    python bench/f32.py --time     # also time it against the other dense kernels

Accuracy is measured against float64 arithmetic (SciPy) on the SAME float32
rows the kernel sees, so the rounding of the input itself is not counted --
the question is what the kernel's float32 arithmetic adds. Regimes:

  uniform          rng.random rows, the easy case
  dirichlet(0.1)   many small entries
  sparse 5%        mostly one-sided coordinates
  30 decades       entries from 1e-30 to 1, for the inlined log
  near duplicate   relative perturbations 1e-6 .. 1e-2, scored RELATIVE to
                   the distance, which is what float32 can promise there,
                   against 50-digit mpmath rather than SciPy
  band + switch    perturbations that put many coordinates near the
                   series/far switch (|u| = 1/2), and the close-but-not-equal
                   pairs where a float32 far formula cancels worst
  large k          Dirichlet(0.1) rows of 16384 to 262144 coordinates: each
                   pair is one float32 sum over k terms, so its error grows
                   with k

Exactness: identical rows (G4), disjoint supports (G6) and the base-2 bound,
as in tests/test_f32.py but over more draws.

The timing is a single-host figure, not a portable one: run it on an idle
machine, and read bench/CROSS_MACHINE.md before comparing hosts. Each kernel
is timed interleaved with the others, 5 rounds of 5 calls, keeping the best
round median; inputs are prepared once, so the time is the x log x table plus
the pair loop.
"""

from __future__ import annotations

import argparse
import statistics
import time
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist

from fastjsd import available_backends, jsd_matrix, jsd_pdist
from fastjsd._common import prepare

B = "numba-f32"


def f32(P):
    return prepare(np.asarray(P, dtype=np.float64), dtype=np.float32)


def ref(P, Q):
    return (cdist(P.astype(np.float64), Q.astype(np.float64), "jensenshannon")
            / np.sqrt(np.log(2.0)))


def got(P, Q):
    return jsd_matrix(P, Q, backend=B, normalize=False,
                      check=False).astype(np.float64)


def exact_pair(a, b):
    """Base-2 distance at 50 digits. SciPy cannot referee near duplicates:
    its grouping cancels as ``2/u``, ~1e-4 relative at ``eps = 1e-6``."""
    from mpmath import log, mp, mpf, sqrt

    mp.dps = 50
    t = mpf(0)
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        x, y = mpf(x), mpf(y)
        m = (x + y) / 2
        if x > 0:
            t += x * log(x / m)
        if y > 0:
            t += y * log(y / m)
    return float(sqrt(t / 2 / log(2)))


def accuracy(rng):
    worst = {}

    def note(name, e):
        worst[name] = max(worst.get(name, 0.0), e)

    for k in (1, 2, 3, 7, 15, 16, 17, 65, 257, 1024, 4096):
        P, Q = f32(rng.random((40, k))), f32(rng.random((30, k)))
        note("uniform", np.abs(got(P, Q) - ref(P, Q)).max())
        P = f32(rng.dirichlet(np.full(k, 0.1), 40))
        Q = f32(rng.dirichlet(np.full(k, 0.1), 30))
        note("dirichlet(0.1)", np.abs(got(P, Q) - ref(P, Q)).max())
        S = rng.random((70, k)) * (rng.random((70, k)) < 0.05)
        S[np.arange(70), rng.integers(0, k, 70)] += 1.0
        S = f32(S)
        note("sparse 5%", np.abs(got(S[:40], S[40:]) - ref(S[:40], S[40:])).max())
        W = f32(10.0 ** rng.uniform(-30, 0, (70, k)))
        note("30 decades", np.abs(got(W[:40], W[40:]) - ref(W[:40], W[40:])).max())
        if k in (2, 17, 257, 4096):
            p = rng.random((6, k)) + 0.1
            for eps in (1e-6, 1e-4, 1e-2):
                P = f32(p)
                Q = f32(p * (1 + eps * rng.choice([-1.0, 1.0], p.shape)))
                r = np.array([exact_pair(a, b) for a, b in zip(P, Q, strict=True)])
                g = np.diag(got(P, Q))
                nz = r > 0        # rows float32 rounded to identical: G4's job
                if nz.any():
                    note("near duplicate (rel)", (np.abs(g - r)[nz] / r[nz]).max())
        for trial in range(10):
            p = (rng.random((24, k)) ** 3 + 1e-6 if trial % 2
                 else rng.random((24, k)) + rng.random() * 2)
            delta = rng.uniform(0.003, 0.6)
            q = np.abs(p * (1 + delta * rng.choice([-1.0, 1.0], p.shape))) + 1e-7
            P, Q = f32(p), f32(q)
            note("band + switch", np.abs(got(P, Q) - ref(P, Q)).max())
    for k in (16384, 65536, 262144):      # each pair is one float32 sum over k
        P = f32(rng.dirichlet(np.full(k, 0.1), 8))
        Q = f32(rng.dirichlet(np.full(k, 0.1), 8))
        note(f"large k={k}", np.abs(got(P, Q) - ref(P, Q)).max())
    print("accuracy: max error on the base-2 distance, vs float64 on the same "
          "float32 rows")
    for name, e in worst.items():
        print(f"  {name:22} {e:.2e}")


def exactness(rng):
    bad = 0
    for k in (5, 13, 25, 65, 129, 1001):
        for seed in range(100):
            row = np.random.default_rng(seed).dirichlet(np.full(k, 0.5))
            D = jsd_matrix(np.repeat(row[None], 5, axis=0), backend=B)
            bad += int(not np.all(D == 0.0))
    print("\nexactness")
    print(f"  identical rows (G4)     matrices with a nonzero: {bad} / 600")
    miss = above = total = 0
    for seed in range(200):
        r = np.random.default_rng(seed)
        n = int(r.integers(4, 21))
        nnz = r.integers(1, 9, n)
        P = np.zeros((n, int(nnz.sum())))
        cols = r.permutation(P.shape[1])
        s = 0
        for i in range(n):
            P[i, cols[s:s + nnz[i]]] = r.dirichlet(np.ones(nnz[i]))
            s += nnz[i]
        for squared in (False, True):
            D = jsd_matrix(P, backend=B, squared=squared).astype(np.float64)
            off = D[~np.eye(n, dtype=bool)]
            miss += int((off != 1.0).sum())
            above += int((off > 1.0).sum())
            total += off.size
    print(f"  disjoint supports (G6)  not exactly 1.0: {miss} / {total}, "
          f"above 1.0: {above}")
    worst = 0.0
    for seed in range(100):
        r = np.random.default_rng(seed)
        k = int(r.choice([8, 64, 3000]))
        P = r.random((20, k)) * (r.random((20, k)) < 0.2)
        P[:, 0] = 10.0 ** r.uniform(-9, -3, 20)
        worst = max(worst, float(jsd_matrix(P, backend=B).max()))
    print(f"  nearly disjoint         max distance {worst!r}")


def timing(rng):
    from _machine import machine

    print("\n" + "  ".join(f"{k}={v}" for k, v in machine().items()))
    try:
        import numba
        nthreads = numba.config.NUMBA_NUM_THREADS
    except ImportError:                       # pragma: no cover - B needs numba
        return
    others = [b for b in ("numba-svml", "numba") if b in available_backends()]
    cases = [("cross 192 x 192, k=192", "matrix", (192, 192), 192),
             ("self 384, k=256", "matrix", (384, None), 256),
             ("pdist 384, k=256", "pdist", (384, None), 256),
             ("self 1024, k=256", "matrix", (1024, None), 256)]
    print("\nmilliseconds; float32 input for every kernel; ratio = kernel / "
          f"{B}")
    for label, api, (n, m), k in cases:
        P = f32(rng.random((n, k)))
        Q = None if m is None else f32(rng.random((m, k)))
        for threads in sorted({1, nthreads}):
            kw = dict(normalize=False, check=False, threads=threads)
            fns = {b: (partial(jsd_pdist, P, backend=b, **kw) if api == "pdist"
                       else partial(jsd_matrix, P, Q, backend=b, **kw))
                   for b in [B] + others}
            for f in fns.values():
                f()
                f()
            best = dict.fromkeys(fns, float("inf"))
            for _ in range(5):
                for b, f in fns.items():
                    ts = []
                    for _ in range(5):
                        t0 = time.perf_counter()
                        f()
                        ts.append(time.perf_counter() - t0)
                    best[b] = min(best[b], statistics.median(ts))
            base = best[B]
            cells = "  ".join(f"{b} {t * 1e3:8.2f} ({t / base:4.1f}x)"
                              for b, t in best.items())
            print(f"  {label:24} threads={threads}  {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--time", action="store_true",
                    help="also time numba-f32 against the other dense kernels")
    args = ap.parse_args()
    if B not in available_backends():
        raise SystemExit(f"{B} needs numba")
    rng = np.random.default_rng(20261025)
    accuracy(rng)
    exactness(rng)
    if args.time:
        timing(rng)


if __name__ == "__main__":
    main()
