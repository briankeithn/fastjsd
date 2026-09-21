"""Measure whether a GPU makes this package irrelevant, instead of guessing.

    python bench/gpu_baseline.py                  # auto-detect device
    python bench/gpu_baseline.py --device cpu     # torch on CPU, for contrast
    python bench/gpu_baseline.py --sizes 1024,2048,4096

Run this ALONE. It drives the CPU hard too (kernel launches, host-side
preparation, the transfer back), so interleaving it with `bench/run_all.py`
would contaminate the CPU tie-bands that make everything else interpretable.

The answer to "why not just use a GPU?" has to be a measurement. It rests on
three facts, each of which this script checks rather than assumes.

**1. Consumer NVIDIA cards are crippled at float64.** Ada and Ampere consumer
parts run fp64 at 1/64 of their fp32 rate; Turing at 1/32. An RTX 4080 is
~34 TFLOPS fp32 and ~0.5 TFLOPS fp64 -- about level with a good 8-core CPU.
Datacenter parts (A100, V100, H100) are 1/2. The script prints the ratio for
whatever it finds, because the entire GPU argument turns on it.

**2. This kernel is the worst case for that.** Jensen-Shannon needs one
logarithm per (pair, coordinate). Consumer cards have no fp64 special-function
unit, so `log` in double precision is a software routine of many instructions,
while in fp32 it is essentially one. So the fp64 penalty here is worse than the
headline FLOP ratio suggests, and the fp32 number is not an alternative --
it is a different computation, which fact 3 quantifies.

**3. float32 is not "nearly as good" for a distance built on cancellation.**
`sqrt` doubles relative error going in, and the summand `t(a,b)` is evaluated
as a difference of quantities much larger than itself. This script reports
worst absolute error against the CPU float64 result. It also tracks duplicate
rows, but read that column carefully: the grouping used here is
`H(m) - (H(p)+H(q))/2`, which returns exactly 0.0 for duplicates on any device
because `m` is bitwise `p` -- so a zero there is NOT evidence about GPU versus
CPU. That property separates unfolded from constant-folded arrangements
(`docs/MATH.md`, Consequence 1); it is tracked so a regression would show.

Timing reports transfer-inclusive and transfer-exclusive numbers separately.
Excluding transfer flatters the GPU and is the right number only if your data
is already resident; including it is what a caller computing one matrix pays.
The output matrix alone is `n^2 * itemsize` coming back -- 33 MB at n=2048 in
fp64, 80 GB at n=1e5, which is its own answer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from fastjsd import jsd_matrix

import datasets

#: fp64:fp32 throughput by CUDA compute capability. The single most important
#: number for this comparison, and not something to infer from the card's name.
FP64_RATIO = {
    (9, 0): "1/2  (H100, datacenter)",
    (8, 9): "1/64 (Ada consumer: RTX 4090/4080/4070)",
    (8, 6): "1/64 (Ampere consumer: RTX 3090/3080)",
    (8, 0): "1/2  (A100, datacenter)",
    (7, 5): "1/32 (Turing: T4, RTX 2080)",
    (7, 0): "1/2  (V100, datacenter)",
    (6, 1): "1/32 (Pascal consumer: GTX 1080)",
}


def device_report(torch, device):
    info = {"device": str(device)}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        cc = (p.major, p.minor)
        info.update({
            "name": p.name,
            "compute_capability": f"{p.major}.{p.minor}",
            "fp64_rate": FP64_RATIO.get(cc, "unknown for this capability"),
            "vram_gb": round(p.total_memory / (1 << 30), 1),
            "sms": p.multi_processor_count,
        })
    info["torch"] = torch.__version__
    info["cuda_build"] = getattr(torch.version, "cuda", None)
    return info


def jsd_torch(torch, X, hp, chunk, out):
    """JS distance matrix, base 2, in whatever dtype ``X`` is.

    Grouped as ``H(m) - (H(p)+H(q))/2`` rather than the package's
    ``sum t(a,b)``: the per-row entropies hoist out of the pair loop, which is
    the only way to keep this from being three times the work. It is also the
    grouping SciPy uses, so the comparison is against a reasonable GPU
    implementation rather than a naive one.
    """
    n = X.shape[0]
    ln2 = float(np.log(2.0))
    for i0 in range(0, n, chunk):
        A = X[i0:i0 + chunk]
        # (c, n, k) mixture -- the reason this is bandwidth-bound and the
        # reason chunking is mandatory rather than an optimization.
        M = 0.5 * (A[:, None, :] + X[None, :, :])
        Hm = -torch.special.xlogy(M, M).sum(-1)
        js = Hm - 0.5 * (hp[i0:i0 + chunk, None] + hp[None, :])
        js = torch.clamp(js, min=0.0)
        out[i0:i0 + chunk] = torch.sqrt(js / ln2)
    return out


def autochunk(n, k, itemsize, budget_gb):
    per = n * k * itemsize * 3          # mixture + xlogy temp + slack
    c = max(1, int(budget_gb * (1 << 30) // max(per, 1)))
    return min(c, n)


def run_case(torch, device, P, dtype, chunk_budget, reps):
    ref = jsd_matrix(P)                                  # CPU float64 truth
    tdt = torch.float64 if dtype == "fp64" else torch.float32
    itemsize = 8 if dtype == "fp64" else 4
    n, k = P.shape
    chunk = autochunk(n, k, itemsize, chunk_budget)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    host = np.ascontiguousarray(P, dtype=np.float64)
    X = torch.as_tensor(host, dtype=tdt, device=device)
    hp = -torch.special.xlogy(X, X).sum(1)
    out = torch.empty((n, n), dtype=tdt, device=device)

    for _ in range(2):        # twice: the first pays allocator and, on CPU
        jsd_torch(torch, X, hp, chunk, out)   # torch, thread-pool spin-up
    sync()

    # Transfer-EXCLUSIVE: data already resident, result left on device.
    best_dev = np.inf
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        jsd_torch(torch, X, hp, chunk, out)
        sync()
        best_dev = min(best_dev, time.perf_counter() - t0)

    # Transfer-INCLUSIVE: what a caller computing one matrix actually pays.
    best_e2e = np.inf
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        Xi = torch.as_tensor(host, dtype=tdt, device=device)
        hpi = -torch.special.xlogy(Xi, Xi).sum(1)
        oi = torch.empty((n, n), dtype=tdt, device=device)
        jsd_torch(torch, Xi, hpi, chunk, oi)
        got = oi.to("cpu").numpy()
        sync()
        best_e2e = min(best_e2e, time.perf_counter() - t0)

    got = out.to("cpu").numpy().astype(np.float64)
    err = float(np.abs(got - ref).max())

    # The exactness property, on the GPU. Duplicate rows must give exactly 0.
    D = np.repeat(P[:16], 2, axis=0)
    Xd = torch.as_tensor(np.ascontiguousarray(D, dtype=np.float64),
                         dtype=tdt, device=device)
    hd = -torch.special.xlogy(Xd, Xd).sum(1)
    od = torch.empty((D.shape[0], D.shape[0]), dtype=tdt, device=device)
    jsd_torch(torch, Xd, hd, D.shape[0], od)
    od = od.to("cpu").numpy()
    dup = max(abs(float(od[a, a + 1])) for a in range(0, D.shape[0], 2))

    return {"chunk": chunk, "dev_ms": best_dev * 1e3, "e2e_ms": best_e2e * 1e3,
            "max_abs_err": err, "duplicate_row_distance": dup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda, cpu, or omit to autodetect")
    ap.add_argument("--sizes", default="1024,2048",
                    help="comma-separated n values, k fixed by --k")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chunk-budget", type=float, default=1.5,
                    help="GB of device memory for the mixture tensor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed.\n"
              "  CUDA build:  pip install torch --index-url "
              "https://download.pytorch.org/whl/cu124\n"
              "  The CPU-only wheel will import fine and silently give you no "
              "GPU, which is\n  why this script prints cuda_build below -- "
              "check it is not None.")
        return 1

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but torch.cuda.is_available() is False.")
        return 1

    info = device_report(torch, device)
    print(__doc__.split("\n\n")[0])
    print("\n" + "=" * 88)
    for key, val in info.items():
        print(f"  {key:<22} {val}")
    print("=" * 88)
    if device.type == "cuda" and "1/64" in str(info.get("fp64_rate", "")):
        print("  This card is a 1/64 fp64 part. The fp64 row below is the "
              "honest number for\n  exact work on it, and it is expected to be "
              "poor -- that is the finding, not a\n  failure of the "
              "measurement.")
    if device.type == "cpu":
        print("  Running torch on the CPU. Useful as a contrast, but it does "
              "not answer the\n  GPU question.")

    ns = [int(x) for x in args.sizes.split(",") if x.strip()]
    print(f"\nk={args.k}, min of {args.reps}. `dev ms` excludes transfer "
          "(data already resident);\n`e2e ms` includes it. `fastjsd` is this "
          "package on the CPU in float64.\n")
    print(f"{'n':>6}{'dtype':>7}{'chunk':>7}{'fastjsd ms':>12}{'dev ms':>10}"
          f"{'e2e ms':>10}{'dev x':>8}{'e2e x':>8}{'max err':>11}{'dup':>10}")
    print("-" * 99)

    rows = []
    for n in ns:
        P = datasets.dirichlet(n, args.k, 1.0, seed=n)
        jsd_matrix(P)
        best_cpu = min(
            (lambda t0: (jsd_matrix(P), time.perf_counter() - t0)[1])(
                time.perf_counter()) for _ in range(args.reps)) * 1e3
        for dtype in ("fp32", "fp64"):
            try:
                r = run_case(torch, device, P, dtype, args.chunk_budget,
                             args.reps)
            except RuntimeError as exc:
                print(f"{n:>6}{dtype:>7}   FAILED: {str(exc)[:60]}")
                continue
            r.update(n=n, k=args.k, dtype=dtype, fastjsd_ms=best_cpu)
            rows.append(r)
            print(f"{n:>6}{dtype:>7}{r['chunk']:>7}{best_cpu:>12.1f}"
                  f"{r['dev_ms']:>10.1f}{r['e2e_ms']:>10.1f}"
                  f"{best_cpu / r['dev_ms']:>7.2f}x"
                  f"{best_cpu / r['e2e_ms']:>7.2f}x"
                  f"{r['max_abs_err']:>11.2e}"
                  f"{r['duplicate_row_distance']:>10.1e}")

    print("\n" + "-" * 99)
    print("How to read this:")
    print("  `dev x` / `e2e x` above 1.00 mean the GPU is faster than fastjsd.")
    print("  `max err` is against the CPU float64 result. For fp32 this is")
    print("  the cost of the speedup, not a bug -- and it is the reason the")
    print("  fp32 column is not a substitute for the fp64 one.")
    print("  `dup` is the distance between two IDENTICAL rows, and it is")
    print("  expected to be 0.0 here -- do NOT read a zero as this package")
    print("  having no advantage, or a non-zero as the GPU being broken.")
    print("  The grouping above is H(m)-(H(p)+H(q))/2, where m is bitwise")
    print("  equal to p for duplicate rows, so the subtraction is exact for")
    print("  the same reason SciPy's is (docs/MATH.md, Consequence 1). The")
    print("  duplicate-zero property separates UNFOLDED from CONSTANT-FOLDED")
    print("  arrangements, not CPU from GPU. It is tracked here only so a")
    print("  regression in the GPU path would be visible.")
    print("\n  Transfer is charged in `e2e` only. The output alone is n^2 *")
    print("  itemsize coming back: 33 MB at n=2048 fp64, 80 GB at n=1e5 --")
    print("  which is the real answer at the sizes where a GPU would matter.")

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps({"device": info, "rows": rows}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
