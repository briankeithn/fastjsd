"""One command, one results bundle, fingerprinted to the machine that made it.

    python bench/run_all.py              # the full set (~20-40 min)
    python bench/run_all.py --quick      # smoke test first (~2 min)

A latency figure means nothing without the machine that produced it, and on
virtualized or shared hosts that machine is hard to pin down: the CPU model
string can be generic or absent, and the results themselves move. The
merge-vs-bitmap comparison is the clearest case -- it can pick a different
winner on different hardware, so a dispatch threshold validated on one machine
has to be re-checked on the next.

Load checks are advisory for a related reason. `uptime` and CPU-busy
percentages report only this operating system's own runqueue, so a co-tenant
saturating a shared last-level cache or the memory controller stays invisible
to them. That is precisely the contention a memory-streaming kernel is most
sensitive to.

So this script does three things a hand-run sequence does not:

1. **Fingerprints the machine** before measuring anything -- CPU model, core
   counts, RAM, OS, every library version, and whether Numba found SVML. A
   result without this is not reproducible.
2. **Runs the experiments in dependency order.** `aa_control` first, because
   the other two are uninterpretable without it -- `lee_control` reads its
   tie-bands from `aa_control`'s output, and a ratio without a resolution
   floor is not a measurement.
3. **Bundles everything into one directory** with a manifest, so the whole run
   travels as a unit and cannot be quoted a piece at a time.

It deliberately does NOT try to make the machine quiet for you. It checks,
warns, and records what it found; making the machine idle is the operator's job.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TESTS = ROOT / "tests"

#: (script, args, why it runs here). Order is a dependency, not a preference.
STEPS = [
    ("aa_control.py", ["--threads", "1", "--reps", "25"],
     "resolution floor -- everything below is uninterpretable without it"),
    ("bits_vs_merge.py", ["--threads", "1", "--reps", "9"],
     "does the merge/bitmap crossover land where it did on other hardware?"),
    ("lee_control.py", ["--threads", "1", "--reps", "15"],
     "the honest margin over a competently implemented Lee 1997"),
]
QUICK = {
    "aa_control.py": ["--threads", "1", "--reps", "5", "--quick"],
    "bits_vs_merge.py": ["--threads", "1", "--reps", "3"],
    "lee_control.py": ["--threads", "1", "--reps", "3"],
}


def child_env():
    """PYTHONPATH for the subprocesses, with this OS's own separator.

    `os.pathsep` is ';' on Windows and ':' elsewhere. Hardcoding ':' yields a
    harness that silently works only on Linux.
    """
    env = dict(os.environ)
    parts = [str(ROOT), str(HERE)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def cpu_name():
    """The most specific CPU string this OS provides."""
    if sys.platform == "win32":
        for key in ("PROCESSOR_IDENTIFIER",):
            if os.environ.get(key):
                return os.environ[key]
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=30)
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def fingerprint():
    """Everything needed to know what produced the numbers."""
    fp = {
        "captured_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": platform.node(),
        "cpu": cpu_name(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil
        fp["physical_cores"] = psutil.cpu_count(logical=False)
        fp["ram_gb"] = round(psutil.virtual_memory().total / (1 << 30), 1)
        bat = psutil.sensors_battery()
        if bat is not None:
            fp["on_battery"] = not bat.power_plugged
    except Exception:
        fp["psutil"] = "not installed (pip install psutil for core/RAM detail)"

    for mod in ("numpy", "numba", "scipy", "mpmath"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = "absent"
    try:
        import numba
        fp["numba_svml"] = bool(getattr(numba.config, "USING_SVML", False))
        fp["numba_default_threads"] = numba.config.NUMBA_NUM_THREADS
        fp["numba_threading_layer"] = str(
            getattr(numba.config, "THREADING_LAYER", "?"))
    except Exception:
        pass
    try:
        fp["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=30).stdout.strip() or "?"
        fp["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:
        fp["git_commit"] = "not a git checkout"
    return fp


def preflight():
    """Fail loudly now rather than 20 minutes in. Returns a list of problems."""
    bad = []
    if not (ROOT / "fastjsd" / "__init__.py").exists():
        bad.append(f"package not found at {ROOT / 'fastjsd'}")
    for fx in ("soft_clusters_931x25.npy", "soft_clusters_1014x40.npy"):
        if not (HERE / "fixtures" / fx).exists():
            bad.append(f"missing fixture {fx} -- is this a shallow/partial clone?")
    env = child_env()
    probe = ("import numpy, fastjsd, fastjsd._numba as nb; "
             "assert nb.available(); "
             "import numpy as np; "
             "d = fastjsd.jsd_matrix(np.eye(8)); "
             "assert d.shape == (8, 8); print('ok')")
    r = subprocess.run([sys.executable, "-c", probe], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0 or "ok" not in r.stdout:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        bad.append("import/JIT check failed: " + " | ".join(tail))
    return bad


def drift_probe(env):
    """Time a fixed small workload. Called before and after the whole bundle.

    A 20-40 minute run is far longer than any turbo budget, so the machine
    that finishes the run is slower than the one that started it, and on a
    chassis where CPU and GPU share a power envelope it can also be a machine
    that throttled partway through.

    The paired-duel design already cancels *slow monotonic* drift, because the
    two sides alternate position within every duel and so decay together. What
    it cannot cancel is a step change: a thermal cliff, a power-plan switch, or
    another process starting. This probe cannot tell those apart either, but it
    does say whether the endpoints differ -- and if they differ a lot, the run
    should be repeated on a cooler machine before anything is quoted.

    Returns milliseconds, or None if the probe could not run.
    """
    code = (
        "import time, numpy as np, fastjsd\n"
        "rng = np.random.default_rng(0)\n"
        "P = rng.dirichlet(np.ones(64), size=384)\n"
        "fastjsd.jsd_matrix(P)\n"
        "ts = []\n"
        "for _ in range(7):\n"
        "    t0 = time.perf_counter(); fastjsd.jsd_matrix(P)\n"
        "    ts.append(time.perf_counter() - t0)\n"
        "print(min(ts) * 1e3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def quiet_check(seconds=3):
    """Report machine load. Advisory: this cannot see a noisy co-tenant."""
    notes = []
    try:
        import psutil
        busy = psutil.cpu_percent(interval=seconds)
        notes.append(f"CPU busy over {seconds}s before starting: {busy:.1f} %")
        if busy > 10.0:
            notes.append("WARNING: >10 % busy before the run. Close other "
                         "applications; a background indexer or update "
                         "service will widen every tie-band in the results.")
        bat = psutil.sensors_battery()
        if bat is not None and not bat.power_plugged:
            notes.append("WARNING: running on battery. Most laptops cap "
                         "sustained clocks on battery, and the run is long "
                         "enough to hit that cap partway through -- which "
                         "biases whichever implementation is timed later.")
    except Exception:
        notes.append("psutil not installed -- cannot check machine load. "
                     "`pip install psutil` for this check.")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny reps, three cells -- proves it runs, proves "
                         "nothing about performance")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    fp = fingerprint()
    stamp = fp["captured_utc"].replace(":", "").replace("-", "")
    host = "".join(c for c in fp["hostname"] if c.isalnum()) or "host"
    out = pathlib.Path(args.outdir) if args.outdir else (
        HERE / "results" / f"{host}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("fastjsd measurement bundle")
    print("=" * 78)
    for key in ("hostname", "cpu", "machine", "os", "logical_cores",
                "physical_cores", "ram_gb", "on_battery", "python", "numpy",
                "numba", "scipy", "numba_svml", "numba_default_threads",
                "git_commit", "git_dirty"):
        if key in fp:
            print(f"  {key:<24} {fp[key]}")
    print(f"  {'output':<24} {out}")

    problems = preflight()
    if problems:
        print("\nPREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("\nNothing was measured. Fix the above and re-run.")
        return 1
    print("\npreflight ok")

    if fp.get("git_dirty"):
        print("NOTE: working tree has uncommitted changes, so `git_commit` "
              "does not fully identify what ran.")

    notes = quiet_check(1 if args.quick else 3)
    for n in notes:
        print(f"  {n}")

    if not args.skip_tests:
        print("\n--- test suite (validates the build on this machine) ---")
        r = subprocess.run([sys.executable, "-m", "pytest", str(TESTS), "-q"],
                           env=child_env(), cwd=str(ROOT),
                           capture_output=True, text=True)
        (out / "pytest.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        print("  " + (tail[0] if tail else "no output"))
        if r.returncode != 0:
            print("  WARNING: tests failed. Measurements below are still "
                  "recorded, but do not quote them until this is understood.")

    drift_before = drift_probe(child_env())
    if drift_before:
        print(f"  drift probe (before): {drift_before:.2f} ms")

    results = {}
    for script, default_args, why in STEPS:
        argv = QUICK[script] if args.quick else default_args
        print(f"\n--- {script} ---\n  {why}")
        print(f"  running: {script} {' '.join(argv)}")
        r = subprocess.run([sys.executable, str(HERE / script), *argv],
                           env=child_env(), cwd=str(ROOT),
                           capture_output=True, text=True)
        text = r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")
        # aa_control's output must land where lee_control looks for it, or
        # lee_control runs uncalibrated and says so.
        (out / script.replace(".py", ".txt")).write_text(text, encoding="utf-8")
        # NEVER in quick mode: a 3-cell, 5-rep A/A is not a band table, and
        # copying it over the canonical file would hand lee_control bands
        # measured at rep counts that cannot resolve anything.
        if script == "aa_control.py" and r.returncode == 0 and not args.quick:
            canon = HERE / "results" / "aa_control_1t.txt"
            canon.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / "aa_control.txt", canon)
            print(f"  -> also written to {canon.name} for lee_control to read")
        elif script == "aa_control.py" and args.quick:
            print("  -> NOT copied to aa_control_1t.txt (quick mode); "
                  "lee_control will use whatever bands are already there")
        results[script] = r.returncode
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}) -- see "
                  f"{script.replace('.py', '.txt')}")
            for ln in text.strip().splitlines()[-4:]:
                print(f"    {ln}")
        else:
            for ln in text.strip().splitlines():
                if "tie-band" in ln and "worst" in ln:
                    print(f"  {ln.strip()}")
                if ln.startswith("resolved wins") or ln.startswith("bitmap faster"):
                    print(f"  {ln.strip()}")

    drift_after = drift_probe(child_env())
    drift = None
    if drift_before and drift_after:
        drift = drift_after / drift_before
        print(f"\ndrift probe: {drift_before:.2f} ms before, "
              f"{drift_after:.2f} ms after  ->  {drift:.2f}x")
        if drift > 1.15:
            print("  WARNING: the machine finished the run more than 15 % "
                  "slower than it\n  started it. Paired duels cancel smooth "
                  "drift, but not a thermal\n  cliff. Let it cool, close "
                  "everything, and re-run before quoting\n  any of this.")
        elif drift < 0.87:
            print("  WARNING: the machine got FASTER during the run, which "
                  "usually means\n  it was still busy at the start. Re-run.")
        else:
            print("  within 15 % end to end -- no evidence of a thermal cliff.")

    (out / "MANIFEST.json").write_text(
        json.dumps({"fingerprint": fp, "quiet_check": notes,
                    "quick_mode": args.quick, "exit_codes": results,
                    "drift_ms_before": drift_before,
                    "drift_ms_after": drift_after, "drift_ratio": drift},
                   indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    if args.quick:
        print("QUICK MODE -- this proves the harness runs. It proves nothing")
        print("about performance: the rep counts are far too low to resolve")
        print("anything. Re-run without --quick for numbers.")
    else:
        print("Done. Send back the whole directory, not individual files --")
        print("a ratio without its tie-band and its fingerprint is not a")
        print("result, and that is the specific mistake this bundle prevents.")
    print(f"  {out}")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")
    return 0 if all(v == 0 for v in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
