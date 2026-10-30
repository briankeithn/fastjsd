"""One leg of a cross-machine run: do the claims survive different hardware?

    python bench/portable.py --json portable-lin-x64.json
    python bench/portable.py --fingerprint-only      # machine block, no timing
    python bench/portable.py --threads --crossover   # the opt-in extras

This does NOT replace `bench/bench_honest.py`, and no number it prints is
"the" speedup. It runs on shared, oversubscribed cloud VMs whose clock rate,
core count and neighbours are all invisible, so absolute milliseconds cannot
travel between boxes. What *can* travel is a ratio between two implementations
timed in the same interleaved round-robin on the same runner: the box's clock
rate divides out of it. So ratios are the output, milliseconds are recorded
only so the ratios can be audited, and `bench/aggregate.py` combines legs by
ratio and never by time.

Three properties this harness is built around:

Degrading instead of crashing.
    `bench_honest.py` calls every entry of `baselines.THIRD_PARTY`, so it needs
    pynndescent and torch installed on the machine that runs it. Here every
    implementation is *probed* on a 4x3 matrix first and recorded as
    ``status: "absent"`` with the reason if it cannot run. Same for scipy
    (no reference: the harness falls back to fastjsd's own NumPy backend and
    says so) and for numba (NumPy backends only, reported in its own section
    because it measures a different claim).

Never letting a missing field look like a measured one.
    Every fingerprint key is always present, ``null`` when unobtainable, with
    the reason appended to ``machine.unavailable``. An aggregator cannot tell a
    dropped key from an unmeasurable quantity, and on a matrix that deliberately
    mixes Linux, macOS and Windows most keys are unobtainable somewhere.
    `psutil` and `threadpoolctl` would each answer several of these in one
    call; neither is a dependency this harness takes, hence the per-OS probing.

Recording its own reliability.
    ``ms`` is an ARRAY of per-repeat minima, not a scalar. A scalar minimum
    throws away the only evidence about how much to trust it, and without the
    array there is no way to say whether a ratio is above this box's noise.
    Every ratio therefore carries ``resolvable``: false when the across-repeat
    spread straddles 1.0, in which case the honest report is
    "indistinguishable" rather than a number.

Statistics-only checks (accuracy against 50-digit truth, the exactly-zero
guarantee on duplicate rows, exactly 1.0 on disjoint supports) are
deterministic and DO travel across boxes. They are the highest-value output
here, because the exact-zero property depends on ``PL[i,c] + PL[j,c]``
cancelling ``2 m log m`` bitwise, which is a property of one libm and one code
generator until it is checked against another.

``numba-f32`` is timed in its own section, against the float64 dense kernel on
the same float32 rows. ``auto`` never selects it, so it has no place in the
``fastjsd auto`` comparison, and its claim is a different one: that a float32
pair loop LLVM vectorizes beats the float64 loop on this CPU. Vectorization is
exactly the part that depends on the hardware, which makes that ratio a
direction claim of the kind this harness exists for.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

import fastjsd
from fastjsd import available_backends, jsd_matrix, resolve_backend
from fastjsd._common import prepare

import baselines
import datasets

#: /2 added the ``f32`` section, ``fastjsd numba-f32`` in the exact-zero check,
#: and the disjoint-supports check under ``accuracy``.
SCHEMA = "fastjsd-crossmachine/2"

#: Validity thresholds, published here so the aggregator's rule stays
#: mechanical: drop unusable, warn on suspect.
STEAL_SUSPECT = 0.02          # fraction of CPU time stolen by the hypervisor
STEAL_UNUSABLE = 0.10
SPREAD_SUSPECT = 0.25         # (max-min)/median of fastjsd's own per-repeat ms
SPREAD_UNUSABLE = 0.50
#: A baseline whose min sits this far below its own median, on a box where
#: fastjsd's does not, is evidence of the min-of-N duration bias: per repetition
#: the chance of catching a transient scales with how long the run takes, so the
#: slowest implementation is punished hardest and its ratio comes out inflated.
SKEW_WARN = 0.85

#: A tiny matrix, used to probe every implementation before anything is timed.
#: Calling each one once is also what pays the Numba compile, so no measured
#: repetition contains a JIT.
_PROBE = np.array([[0.5, 0.3, 0.2],
                   [0.1, 0.1, 0.8],
                   [0.4, 0.4, 0.2],
                   [0.2, 0.5, 0.3]])

#: The ``f32`` section: ``F32_SUBJECT`` timed against ``F32_BASELINE``, and the
#: ratio is baseline / subject, so above 1.0 the float32 kernel is faster.
F32_SUBJECT = "numba-f32"
F32_BASELINE = "numba"
#: Agreement bound for numba-f32 against the float64 kernel on the same float32
#: rows, the tolerance `tests/test_f32.py` asserts. It tells a working kernel
#: from a broken one; the error itself is recorded beside it, up to 1.9e-7 on
#: x86-64 and 3.2e-7 on ARM, where LLVM leaves the float32 sum scalar.
F32_TOL = 1e-6
#: How `bench/f32.py --time` calls both kernels: rows prepared once, as float32.
F32_KW = {"normalize": False, "check": False}
#: Decimals kept on a timed `ms`: the nanosecond. `aggregate.crosscheck` holds
#: the ratio of two minima to its ratio block within 1e-3, so the rounding has
#: to stay far below that even for a case timed in microseconds.
MS_DIGITS = 6

CAVEATS = (
    "These figures were measured on shared, oversubscribed cloud VMs. They "
    "establish the DIRECTION and rough MAGNITUDE of this package's claims "
    "across hardware. They do not replace a single-machine measurement made "
    "with `bench/bench_honest.py` on a quiet box, and no number here should be "
    "quoted as 'the' speedup.",

    "Absolute milliseconds are not comparable across boxes. They are recorded "
    "only so the ratios can be recomputed and audited, never to be read "
    "directly.",

    "Every ratio is between two implementations timed in the SAME interleaved "
    "round-robin on the SAME runner. No time from one runner was ever divided "
    "by a time from another.",

    "The speedup over SciPy is not a constant. SciPy's pdist is "
    "single-threaded and fastjsd is parallel, so the ratio scales with the "
    "runner's usable core count -- 2 vCPU on standard private-repo runners, "
    "3-4 elsewhere. It is also biased upward on a noisy box, because min-of-N "
    "favours short-running implementations: SciPy is exposed to transients for "
    "several times longer per repetition. Use the Lee-1997 ratio if you want a "
    "number that does not depend on core count.",

    "'ubuntu-latest' is not one machine. GitHub's x86-64 pool mixes at least "
    "Intel Cascade Lake, Intel Ice Lake and AMD EPYC. The CPU model recorded "
    "in each result is the fact; the label is not.",

    "Thread scaling on a 2-4 vCPU VM whose vCPUs may be SMT siblings on a "
    "shared host is not a scaling study. Steal time is recorded beside it; "
    "runs above threshold are marked suspect and excluded from the verdict "
    "rather than averaged in.",

    "The dispatch crossover density is a property of the hardware, not of the "
    "algorithm. SPARSE_DENSITY_MAX = 0.60 was measured on a single machine. "
    "What this checks is that using that constant costs less than the stated "
    "regret against each box's own optimum -- not that the crossover sits at "
    "0.60 everywhere.",

    "The numba-svml backend does not appear here: the legs do not install "
    "libsvml (PyPI's intel-cmplr-lib-rt), and no ARM or Apple host can run it. "
    "Every figure here is the libm path.",

    "numba-f32 is timed against the float64 dense kernel on the same float32 "
    "rows, as `bench/f32.py --time` does, on the cases `auto` sends to a dense "
    "kernel. That ratio says whether this CPU runs the float32 loop faster. It "
    "says nothing about numba-svml, which no leg installs, so the README's "
    "numba-svml comparison stays a one-machine figure.",

    "Legs without numba report the NumPy backends only. Those measure a "
    "different claim -- the support-intersection identity, not the kernel -- "
    "and on dense inputs they can be SLOWER than SciPy. They belong in a "
    "separate section and must not be folded into the headline range.",

    "N boxes is not a random sample of computing hardware. It is the hardware "
    "GitHub rents. Nothing here supports any claim about GPUs, other clouds, "
    "bare metal, or CPUs outside this set.",

    "The accuracy and exactness figures are deterministic and DO travel across "
    "boxes. None of the caveats above apply to them.",
)


# ---------------------------------------------------------------------------
# fingerprint plumbing
# ---------------------------------------------------------------------------

class Fields:
    """A dict that records WHY a field is missing instead of dropping it.

    On a matrix that spans Linux, macOS and Windows most of these keys are
    unobtainable somewhere. An aggregator cannot distinguish a key that was
    never written from a quantity that could not be measured, so every field is
    always emitted -- ``None`` when unavailable, with the reason collected into
    ``unavailable``.
    """

    def __init__(self):
        self.data: dict = {}
        self.unavailable: list[str] = []

    def add(self, name, fn, note="not available on this platform"):
        try:
            value = fn()
        except Exception as exc:
            value = None
            self.unavailable.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            if value is None:
                self.unavailable.append(f"{name}: {note}")
        self.data[name] = value
        return value


def _read(path):
    return pathlib.Path(path).read_text().strip()


def _run(cmd, timeout=8.0):
    out = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=timeout, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {out.returncode}")
    return out.stdout.strip()


def _sysctl(name):
    """`sysctl -n` on Darwin, ``None`` anywhere it is absent or unset."""
    if platform.system() != "Darwin":
        return None
    try:
        return _run(["sysctl", "-n", name]) or None
    except Exception:
        return None


def _env(name):
    v = os.environ.get(name)
    return v if v else None


def _cpu_times():
    """`/proc/stat`'s aggregate cpu line, as a list of jiffy counters.

    Field 8 is steal -- time this vCPU was runnable but the hypervisor gave the
    physical core to somebody else. It is the only direct evidence of a noisy
    neighbour available inside the guest, and it does not exist off Linux.
    """
    with open("/proc/stat") as fh:
        parts = fh.readline().split()
    if parts[0] != "cpu":
        raise RuntimeError("/proc/stat did not start with the cpu line")
    return [int(x) for x in parts[1:]]


def _steal_fraction(before, after):
    if before is None or after is None:
        return None
    n = min(len(before), len(after))
    delta = [after[i] - before[i] for i in range(n)]
    total = sum(delta)
    if total <= 0 or n < 8:
        return None
    return delta[7] / total


def _loadavg():
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None


def _nivcsw():
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw


# ---------------------------------------------------------------------------
# fingerprint sections
# ---------------------------------------------------------------------------

def _libc():
    """glibc version, which is the variable the exactness guarantee is a
    function of. `os.confstr` is the authoritative answer where it exists;
    `platform.libc_ver()` scans the binary and is less reliable. musl returns
    nothing from either, so the workflow is expected to put `ldd --version`
    into FASTJSD_CI_LIBC."""
    if platform.system() == "Linux":
        try:
            v = os.confstr("CS_GNU_LIBC_VERSION")
        except (ValueError, OSError):
            v = None
        return v or _env("FASTJSD_CI_LIBC")
    return None


def _cpu_model():
    system = platform.system()
    if system == "Linux":
        try:
            for line in _read("/proc/cpuinfo").splitlines():
                if line.split(":")[0].strip() in ("model name", "Model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        # aarch64 has no `model name` in /proc/cpuinfo at all.
        try:
            for line in _run(["lscpu"]).splitlines():
                if line.startswith("Model name:"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        try:
            midr = _read(
                "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
            return f"aarch64 MIDR_EL1={midr}"
        except OSError:
            return None
    if system == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    return platform.processor() or _env("PROCESSOR_IDENTIFIER")


def _cpu_vendor():
    """Vendor/family/model/stepping. `ubuntu-latest` is not one machine --
    GitHub's x86-64 pool has served Cascade Lake, Ice Lake and EPYC -- so this
    is the field that identifies the silicon, not the runner label."""
    system = platform.system()
    if system == "Linux":
        want = {"vendor_id": "vendor", "cpu family": "family",
                "model": "model", "stepping": "stepping"}
        out = {}
        for line in _read("/proc/cpuinfo").splitlines():
            key = line.split(":")[0].strip()
            if key in want and want[key] not in out:
                out[want[key]] = line.split(":", 1)[1].strip()
            if line.startswith("CPU implementer"):       # aarch64 spelling
                out["implementer"] = line.split(":", 1)[1].strip()
            if line.startswith("CPU part"):
                out["part"] = line.split(":", 1)[1].strip()
        return out or None
    if system == "Darwin":
        out = {}
        for field in ("vendor", "family", "model", "stepping"):
            v = _sysctl(f"machdep.cpu.{field}")
            if v:
                out[field] = v
        return out or None
    ident = _env("PROCESSOR_IDENTIFIER")
    return {"identifier": ident} if ident else None


def _cpu_usable():
    """Cores this process may actually run on -- the number the thread claim
    has to be normalized by, which is not `os.cpu_count()` inside a container
    or under an affinity mask."""
    fn = getattr(os, "sched_getaffinity", None)
    if fn is not None:
        return len(fn(0))
    fn = getattr(os, "process_cpu_count", None)      # 3.13+, absent on 3.11
    if fn is not None:
        return fn()
    return os.cpu_count()


def _cpu_physical():
    system = platform.system()
    if system == "Linux":
        pairs = set()
        root = pathlib.Path("/sys/devices/system/cpu")
        for topo in sorted(root.glob("cpu[0-9]*/topology")):
            try:
                pairs.add((_read(topo / "physical_package_id"),
                           _read(topo / "core_id")))
            except OSError:
                continue
        return len(pairs) or None
    if system == "Darwin":
        v = _sysctl("hw.physicalcpu")
        return int(v) if v else None
    v = _env("FASTJSD_CI_PHYSICAL_CORES")
    return int(v) if v else None


def _perflevels():
    """Apple Silicon P/E split. A '3-core' macOS runner is a slice of a bigger
    part, and which cores it got changes everything about the timings."""
    if platform.system() != "Darwin":
        return None
    out = {}
    for lvl in (0, 1):
        for field in ("physicalcpu", "logicalcpu"):
            v = _sysctl(f"hw.perflevel{lvl}.{field}")
            if v:
                out[f"perflevel{lvl}_{field}"] = int(v)
    return out or None


def _smt_active():
    if platform.system() == "Linux":
        try:
            return _read("/sys/devices/system/cpu/smt/active") == "1"
        except OSError:
            return None
    return None


def _cgroup_cpu_max():
    """cgroup v2 `cpu.max`, or the v1 quota/period pair. Only meaningful if a
    leg runs under `container:`, but a silently throttled runner is exactly the
    kind of thing that looks like bad scaling."""
    try:
        return _read("/sys/fs/cgroup/cpu.max")
    except OSError:
        pass
    try:
        quota = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        return f"{quota} {period}"
    except OSError:
        return None


def _cpu_features():
    """numpy's own dispatch table -- one code path on every platform, no
    subprocess, and it reports NEON/ASIMD/SVE on ARM as readily as AVX-512 on
    x86."""
    from numpy._core import _multiarray_umath
    feats = _multiarray_umath.__cpu_features__
    return sorted(name for name, on in feats.items() if on)


def _numpy_simd():
    cfg = np.__config__.show(mode="dicts")
    return cfg.get("SIMD Extensions")


def _blas():
    cfg = np.__config__.show(mode="dicts")
    blas = cfg["Build Dependencies"]["blas"]
    return {"name": blas.get("name"), "version": blas.get("version")}


def _llvm_host():
    """What LLVM will actually target, which predicts kernel speed better than
    the marketing string does."""
    from llvmlite import binding
    keep = ("avx", "avx2", "avx512f", "avx512dq", "avx512vl", "fma",
            "neon", "sve", "sve2", "asimd")
    feats = binding.get_host_cpu_features()
    return {
        "cpu": binding.get_host_cpu_name(),
        "features": {k: bool(feats.get(k, False)) for k in keep},
        # False on hosts where fastjsd's SVML backend nonetheless works: the
        # package reaches libsvml by ctypes rather than through this pass.
        # Recorded so both facts are visible together.
        "has_svml": bool(binding.targets.has_svml()),
        "llvm_version": list(binding.llvm_version_info),
    }


def _cache():
    system = platform.system()
    if system == "Linux":
        out = []
        base = pathlib.Path("/sys/devices/system/cpu/cpu0/cache")
        for idx in sorted(base.glob("index*")):
            try:
                out.append({
                    "level": int(_read(idx / "level")),
                    "type": _read(idx / "type"),
                    "size": _read(idx / "size"),
                    "line_bytes": int(_read(idx / "coherency_line_size")),
                    "shared_cpu_list": _read(idx / "shared_cpu_list"),
                })
            except OSError:
                continue
        return out or None
    if system == "Darwin":
        line = _sysctl("hw.cachelinesize")
        out = []
        for level, key in ((1, "hw.l1dcachesize"), (2, "hw.l2cachesize"),
                           (3, "hw.l3cachesize")):
            v = _sysctl(key)
            # Apple Silicon reports hw.l3cachesize = 0 rather than omitting it.
            if v and int(v) > 0:
                out.append({"level": level,
                            "type": "Data" if level == 1 else "Unified",
                            "size": f"{int(v) // 1024}K",
                            "line_bytes": int(line) if line else None,
                            "shared_cpu_list": None})
        return out or None
    raw = _env("FASTJSD_CI_CACHE")            # written by the workflow
    return json.loads(raw) if raw else None


def _ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        pass
    if platform.system() != "Windows":
        return None

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(st.ullTotalPhys)


def _virt():
    system = platform.system()
    if system == "Linux":
        try:
            return _run(["systemd-detect-virt"])
        except Exception:
            pass
        try:
            return _read("/sys/class/dmi/id/sys_vendor")
        except OSError:
            return None
    if system == "Darwin":
        return _sysctl("kern.hv_vmm_present")
    return _env("FASTJSD_CI_VIRT")


def _cpu_mhz():
    """Recorded, though clock frequency is effectively unobservable on a hosted
    runner, which is one reason absolute milliseconds cannot travel between
    boxes."""
    if platform.system() == "Linux":
        try:
            for line in _read("/proc/cpuinfo").splitlines():
                if line.startswith("cpu MHz"):
                    return float(line.split(":", 1)[1])
        except OSError:
            pass
        try:
            khz = _read(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            return float(khz) / 1000.0
        except OSError:
            return None
    if platform.system() == "Darwin":
        v = _sysctl("hw.cpufrequency")           # 0 on Apple Silicon
        return float(v) / 1e6 if v and int(v) > 0 else None
    return None


def machine(leg_id):
    """Everything about the box, portably, with a reason for every gap."""
    f = Fields()
    f.data["schema_version"] = SCHEMA
    f.data["produced_by"] = "bench/portable.py"
    f.data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    f.data["leg_id"] = leg_id

    f.data["github"] = {
        "repo": _env("GITHUB_REPOSITORY"), "sha": _env("GITHUB_SHA"),
        "ref": _env("GITHUB_REF"), "run_id": _env("GITHUB_RUN_ID"),
        "run_attempt": _env("GITHUB_RUN_ATTEMPT"), "job": _env("GITHUB_JOB"),
    }
    f.data["runner"] = {
        "label": _env("FASTJSD_CI_LABEL"), "os": _env("RUNNER_OS"),
        "arch": _env("RUNNER_ARCH"), "name": _env("RUNNER_NAME"),
        "environment": _env("RUNNER_ENVIRONMENT"),
    }
    f.data["image"] = {"os": _env("ImageOS"), "version": _env("ImageVersion")}

    f.data["platform_str"] = platform.platform()
    f.data["system"] = platform.system()
    f.data["release"] = platform.release()
    f.data["machine"] = platform.machine()
    f.add("arch_bits", lambda: platform.architecture()[0])
    f.add("libc", _libc, note="not glibc, or not Linux")
    f.add("mac_ver", lambda: platform.mac_ver()[0] or None,
          note="not macOS")
    f.add("win_ver", lambda: (platform.win32_ver()[0] or None),
          note="not Windows")
    f.add("ucrt", lambda: _env("FASTJSD_CI_UCRT"),
          note="not Windows, or the workflow did not export it")

    f.add("cpu_model", _cpu_model)
    f.add("cpu_vendor", _cpu_vendor)
    f.data["cpu_logical"] = os.cpu_count()
    f.add("cpu_usable", _cpu_usable)
    f.add("cpu_physical", _cpu_physical)
    f.add("perflevels", _perflevels, note="not Apple Silicon")
    f.add("smt_active", _smt_active, note="only readable on Linux")
    f.add("cgroup_cpu_max", _cgroup_cpu_max, note="no cgroup cpu controller")

    f.add("cpu_features", _cpu_features)
    f.add("numpy_simd", _numpy_simd)
    f.add("llvm_host", _llvm_host, note="llvmlite not installed")

    f.add("cache", _cache)
    f.add("ram_bytes", _ram_bytes)
    f.add("virt", _virt)
    f.add("cpu_mhz", _cpu_mhz,
          note="frequency is not exposed inside this guest")

    f.data["timer"] = {
        "implementation": time.get_clock_info("perf_counter").implementation,
        "resolution_s": time.get_clock_info("perf_counter").resolution,
        "monotonic": time.get_clock_info("perf_counter").monotonic,
    }
    f.data["unavailable"] = f.unavailable
    return f.data


def software():
    """Library versions, thread environment, and what fastjsd can reach here.

    ``numba.threading_layer()`` raises until a parallel function has actually
    run, so this must be called AFTER the probe pass; calling it first raises
    inside the fingerprint.
    """
    f = Fields()
    f.data["python"] = sys.version.split()[0]
    f.data["python_full"] = sys.version
    f.data["implementation"] = platform.python_implementation()
    f.add("gil_disabled",
          lambda: __import__("sysconfig").get_config_var("Py_GIL_DISABLED"),
          note="not a free-threaded build")
    f.data["numpy"] = np.__version__
    f.data["numpy_path"] = np.__file__
    f.add("blas", _blas)

    def _ver(mod):
        def go():
            return __import__(mod).__version__
        return go

    for mod in ("scipy", "numba", "llvmlite", "mpmath"):
        f.add(mod, _ver(mod), note="not installed")

    f.add("numba_threading_layer",
          lambda: __import__("numba").threading_layer(),
          note="numba absent, or no parallel region has run yet")
    f.add("numba_num_threads",
          lambda: __import__("numba").config.NUMBA_NUM_THREADS,
          note="numba not installed")
    f.add("numba_get_num_threads",
          lambda: __import__("numba").get_num_threads(),
          note="numba not installed")
    f.add("numba_threading_layer_requested",
          lambda: __import__("numba").config.THREADING_LAYER,
          note="numba not installed")
    # `generic` here silently disables host-specific code generation, which is
    # a complete explanation for an otherwise inexplicable result.
    f.add("numba_cpu_name",
          lambda: __import__("numba").config.CPU_NAME, note="unset")
    f.add("numba_cpu_features",
          lambda: __import__("numba").config.CPU_FEATURES or None,
          note="unset")

    f.data["thread_env"] = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
                     "NUMBA_THREADING_LAYER")
    }

    from fastjsd import _svml
    f.data["fastjsd"] = {
        "version": fastjsd.__version__,
        "backends": list(available_backends()),
        "svml_why": _svml.why_unavailable(),
        "sparse_density_max": fastjsd.SPARSE_DENSITY_MAX,
        "svml_density_max": fastjsd.SVML_DENSITY_MAX,
    }
    f.data["unavailable"] = f.unavailable
    return f.data


# ---------------------------------------------------------------------------
# the case set
# ---------------------------------------------------------------------------

def portable_cases():
    """Small, fast, and chosen to finish on a 2-vCPU runner in a few minutes.

    The two `real` fixtures are committed `.npy` and therefore bit-identical on
    every box, which is what makes their accuracy comparison strictly paired.
    The synthetic rows are NOT guaranteed to be: `np.random.Generator.dirichlet`
    goes through libm `log`/`exp`/`pow`, the very thing this exercise varies, so
    they can differ in the last ulp on another platform. Their sha256 is
    recorded and a mismatch is a warning, not a rejection -- one ulp of input is
    irrelevant to a timing and fatal only to a paired accuracy claim.

    `datasets.suite()`'s two n=2000 k=256 cases are dropped: SciPy needs several
    seconds each, which at reps x repeats is most of a CI budget spent on the
    baseline. `--full-suite` puts them back.
    """
    out = []
    for name in ("soft_clusters_931x25", "soft_clusters_1014x40"):
        P = datasets.real(name)
        n, k = P.shape
        out.append((f"real dense   n={n} k={k}", P))
        T = datasets.threshold(P)
        out.append((f"real sparse  n={n} k={k} ({(T > 0).mean():.0%} nz)", T))
    out.append(("synth dense  n=800 k=64", datasets.dirichlet(800, 64, 0.3, seed=1)))
    out.append(("synth sparse n=800 k=256 (3% nz)",
                datasets.sparse_rows(800, 256, 8, seed=3)))
    return out


# ---------------------------------------------------------------------------
# probing: what can this box actually run?
# ---------------------------------------------------------------------------

def _numpy_backend_matrix(P):
    return jsd_matrix(P, backend="numpy")


def probe():
    """Import-and-call every candidate once, recording why each cannot run.

    `bench_honest.py` builds its implementation list from
    `baselines.THIRD_PARTY` and calls every entry, so it needs pynndescent and
    torch installed before it measures anything. Here absence is a fact about
    the box and belongs in the output, not in a traceback.

    Returns ``(runnable, absent)`` where ``runnable`` is a list of
    ``(name, fn)`` and ``absent`` maps name -> reason.
    """
    candidates = [("scipy pdist", baselines.scipy_pdist),
                  ("Lee-1997", baselines.lee_sparse_numba)]
    candidates += list(baselines.THIRD_PARTY.items())
    candidates += [("fastjsd auto", jsd_matrix),
                   ("fastjsd numpy", _numpy_backend_matrix)]

    runnable, absent = [], {}
    for name, fn in candidates:
        try:
            fn(_PROBE)                   # also pays the JIT, once, for good
        except Exception as exc:
            absent[name] = f"{type(exc).__name__}: {exc}"
        else:
            runnable.append((name, fn))
    return runnable, absent


def _f32_matrix(P):
    return jsd_matrix(P, backend=F32_SUBJECT)


def probe_f32():
    """``(fn, None)`` if numba-f32 runs on this box, else ``(None, reason)``.

    Separate from `probe` because numba-f32 is not one of the implementations
    timed against the baselines. Probed all the same, so that a kernel which
    does not compile here is recorded as absent instead of raising inside the
    timing or the exactness checks and taking the whole leg down with it.
    """
    if F32_SUBJECT not in available_backends():
        return None, "numba-f32 needs numba, which is not installed here"
    try:
        _f32_matrix(_PROBE)              # also pays its JIT, once
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return _f32_matrix, None


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def interleaved(fns, P, reps):
    """Min of ``reps``, round-robin across implementations.

    Timing each implementation to completion in turn lets one transient land
    entirely on one implementation; on a shared runner that is not a risk, it
    is the expected case. Round-robin so drift hits everyone equally, and take
    each one's own minimum so the common one-sided noise cancels out of the
    ratio.
    """
    best = [float("inf")] * len(fns)
    for _ in range(reps):
        for idx, fn in enumerate(fns):
            t0 = time.perf_counter()
            fn(P)
            best[idx] = min(best[idx], time.perf_counter() - t0)
    return best


def bench_case(label, P, runnable, absent, args, ref_name):
    n, k = P.shape
    Pc = np.ascontiguousarray(P, dtype=np.float64)
    row = {
        "case": label, "n": int(n), "k": int(k),
        "density": float(np.count_nonzero(P)) / P.size,
        "input_sha256": hashlib.sha256(Pc.tobytes()).hexdigest(),
        "auto_backend": resolve_backend("auto", Pc),
        "working_set_bytes": {"P": int(n * k * 8), "PL": int(n * k * 8),
                              "D": int(n * n * 8)},
        "impls": {}, "ratios": {},
    }

    # Verify before timing: a fast wrong answer is not a result. The reference
    # is SciPy where it exists; where it does not, fastjsd's own NumPy backend
    # stands in and the field says so, because a self-check is not a check.
    ref, ref_error = None, None
    for nm, fn in runnable:
        if nm == ref_name:
            try:
                ref = np.asarray(fn(Pc))
            except Exception as exc:
                ref_error = f"{type(exc).__name__}: {exc}"
    errs, agrees, failed = {}, {}, {}
    for nm, fn in runnable:
        if ref is None:
            failed[nm] = (f"reference {ref_name} failed on this case "
                          f"({ref_error or 'not runnable'}); nothing to "
                          f"verify against")
            errs[nm] = None
            continue
        try:
            errs[nm] = float(np.abs(np.asarray(fn(Pc)) - ref).max())
        except Exception as exc:
            errs[nm] = None
            failed[nm] = f"verification failed: {type(exc).__name__}: {exc}"
            continue
        # Recorded, not asserted. An assert is right on a machine somebody is
        # watching and wrong on a CI leg, where it turns one disagreeing
        # baseline into zero results for the whole box. The disagreement is a
        # fact about this hardware and belongs in the output next to the
        # timing it invalidates.
        agrees[nm] = bool(errs[nm] <= baselines.TOL.get(nm, 5e-12))

    # Time only what verification actually ran. `probe()` clears an
    # implementation on a 4x3 matrix, which does not clear it on this case: a
    # size-, dtype- or density-dependent failure gets through the probe and
    # then raises inside `interleaved`, where nothing catches it and the whole
    # leg dies. Dropping the failures here is what makes the
    # `status: "absent"` record below reachable rather than decorative.
    timed = [(nm, fn) for nm, fn in runnable if nm not in failed]
    names = [nm for nm, _ in timed]
    fns = [fn for _, fn in timed]

    per_repeat = []
    for _ in range(args.repeats):
        per_repeat.append(interleaved(fns, Pc, args.reps))

    for idx, nm in enumerate(names):
        ms = [round(rep[idx] * 1e3, MS_DIGITS) for rep in per_repeat]
        row["impls"][nm] = {
            "status": "ok", "ms": ms,
            "ms_min": min(ms), "ms_median": float(np.median(ms)),
            "max_abs_err": errs.get(nm), "agrees": agrees.get(nm),
            "tol": baselines.TOL.get(nm, 5e-12),
        }
    for nm, why in list(absent.items()) + list(failed.items()):
        row["impls"].setdefault(nm, {"status": "absent", "reason": why})

    # Ratios are computed HERE, not by the aggregator, because only this leg
    # knows which measurements shared a round-robin. An aggregator recomputing
    # them from the `ms` arrays can pair across repeats and manufacture
    # precision that was never measured.
    if "fastjsd auto" in names:
        fi = names.index("fastjsd auto")
        for idx, nm in enumerate(names):
            if nm.startswith("fastjsd"):
                continue
            rs = [rep[idx] / rep[fi] for rep in per_repeat]
            row["ratios"][nm] = {
                "per_repeat": [round(r, 4) for r in rs],
                "min": min(rs), "median": float(np.median(rs)), "max": max(rs),
                # A spread that straddles 1.0 means this box cannot tell the
                # two apart. Reporting the median anyway asserts a ratio from
                # hardware that cannot resolve it.
                "resolvable": bool(min(rs) > 1.0 or max(rs) < 1.0),
                "fastjsd_wins": bool(float(np.median(rs)) > 1.0),
            }
    return row


def f32_case(label, P, args):
    """numba-f32 against the float64 dense kernel, both on the same float32 rows.

    The rows are prepared once and both kernels are called the way
    `bench/f32.py --time` calls them, so this is the published ratio measured
    on this box. Verified first, against the float64 kernel's well-conditioned
    path on those rows -- what the float32 kernel's documented accuracy is
    stated against. A failure is recorded on the case, not raised: one kernel
    that will not run here must not cost the leg its other results.
    """
    n, k = P.shape
    row = {"case": label, "n": int(n), "k": int(k), "status": "ok",
           "impls": {}, "ratio": None, "max_abs_err": None, "agrees": None,
           "tol": F32_TOL}

    def baseline(X):
        return jsd_matrix(X, backend=F32_BASELINE, **F32_KW)

    def subject(X):
        return jsd_matrix(X, backend=F32_SUBJECT, **F32_KW)

    try:
        P32 = prepare(P, dtype=np.float32)
        ref = jsd_matrix(P32.astype(np.float64), backend=F32_BASELINE,
                         well_conditioned=True, **F32_KW)
        err = float(np.abs(np.asarray(subject(P32), dtype=np.float64)
                           - ref).max())
        baseline(P32)                    # the float32-input specialization's JIT
        per_repeat = [interleaved([baseline, subject], P32, args.reps)
                      for _ in range(args.repeats)]
    except Exception as exc:
        row["status"] = "failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row

    row["max_abs_err"] = err
    row["agrees"] = bool(err <= F32_TOL)
    for idx, nm in enumerate((F32_BASELINE, F32_SUBJECT)):
        ms = [round(rep[idx] * 1e3, MS_DIGITS) for rep in per_repeat]
        row["impls"][nm] = {"ms": ms, "ms_min": min(ms),
                            "ms_median": float(np.median(ms))}
    rs = [rep[0] / rep[1] for rep in per_repeat]
    row["ratio"] = {
        "per_repeat": [round(r, 4) for r in rs],
        "min": min(rs), "median": float(np.median(rs)), "max": max(rs),
        "resolvable": bool(min(rs) > 1.0 or max(rs) < 1.0),
        "f32_wins": bool(float(np.median(rs)) > 1.0),
    }
    return row


def f32_section(cases, args, f32_fn, f32_absent):
    """numba-f32 on every case that `auto` sends to a dense kernel.

    The sparse cases are left out because the comparison there would be with
    the sparse kernel, which is a different claim, and not the one 0.2.0 makes.
    """
    out = {"subject": F32_SUBJECT, "baseline": F32_BASELINE,
           "ratio": f"{F32_BASELINE} / {F32_SUBJECT}",
           "input": "float32 rows prepared once; normalize=False, check=False",
           "tol": F32_TOL}
    if f32_fn is None:
        return {**out, "status": "absent", "reason": f32_absent, "cases": []}
    dense = [(label, P) for label, P in cases
             if resolve_backend("auto", np.ascontiguousarray(P, dtype=np.float64))
             in ("numba", "numba-svml")]
    return {**out, "status": "ok",
            "cases": [f32_case(label, P, args) for label, P in dense]}


def threads_scan(P, args):
    """Recorded, never concluded from.

    Two points on a private-repo runner, four on a public one, and the vCPUs
    are quite possibly SMT siblings on a host shared with somebody else. That
    is not a scaling study; the only defensible reading is "monotone and > 1".
    """
    if "numba" not in available_backends():
        return []
    import numba
    jsd_matrix(P, backend="numba")
    out, base = [], None
    for t in range(1, numba.config.NUMBA_NUM_THREADS + 1):
        best = min(_time_once(jsd_matrix, P, backend="numba", threads=t)
                   for _ in range(args.reps))
        base = base or best
        out.append({"threads": t, "ms": round(best * 1e3, 4),
                    "speedup": base / best, "efficiency": base / best / t})
    return out


def _time_once(fn, *a, **kw):
    t0 = time.perf_counter()
    fn(*a, **kw)
    return time.perf_counter() - t0


def crossover(args):
    """Regret against this box's own optimum, not the crossover location.

    Near the turning point the two kernels are BY DEFINITION within noise of
    each other, so a shared runner localizes it badly and a regression gate on
    the location flaps forever. What is robust, and what matters to a caller,
    is how much the shipped constant costs: pick with SPARSE_DENSITY_MAX,
    compare against picking the per-density winner, report the loss. A bound
    on the regret from the shipped constant travels between boxes; a crossover
    location does not.
    """
    if "numba" in available_backends():
        dense_b, sparse_b = "numba", "numba-sparse"
    else:
        dense_b, sparse_b = "numpy", "numpy-sparse"
    n, k = 600, 64
    out, worst = [], 0.0
    empirical = None
    for nnz in (2, 4, 8, 16, 24, 32, 48, 64):
        P = datasets.sparse_rows(n, k, nnz, seed=nnz)
        d = nnz / k
        jsd_matrix(P, backend=dense_b)
        jsd_matrix(P, backend=sparse_b)
        td = min(_time_once(jsd_matrix, P, backend=dense_b)
                 for _ in range(args.reps))
        ts = min(_time_once(jsd_matrix, P, backend=sparse_b)
                 for _ in range(args.reps))
        chosen = ts if d <= fastjsd.SPARSE_DENSITY_MAX else td
        best = min(td, ts)
        regret = 100.0 * (chosen - best) / best
        worst = max(worst, regret)
        winner = "sparse" if ts < td else "dense"
        if winner == "dense" and empirical is None:
            empirical = d
        out.append({"pair": f"{dense_b}/{sparse_b}", "nnz": nnz, "density": d,
                    "dense_ms": round(td * 1e3, 4),
                    "sparse_ms": round(ts * 1e3, 4), "winner": winner,
                    "regret_using_shipped_constant_pct": round(regret, 3)})
    summary = {"shipped_constant": fastjsd.SPARSE_DENSITY_MAX,
               "empirical_crossover": empirical, "max_regret_pct": round(worst, 3),
               "pair": f"{dense_b}/{sparse_b}", "n": n, "k": k}
    return out, summary


# ---------------------------------------------------------------------------
# the deterministic checks -- these DO travel
# ---------------------------------------------------------------------------

def exactness(runnable):
    """Identical rows must come back exactly 0.0. Not approximately.

    This is the one guarantee in the package that can plausibly BREAK on
    different hardware, because it rests on ``PL[i,c] + PL[j,c]`` cancelling
    ``2 m log(m)`` bitwise -- a property of one libm plus one code generator
    until it is checked against another, whether that is Apple's libm, MSVC's
    UCRT, or aarch64 FMA contraction. It costs nothing, no timer is involved,
    and no amount of runner noise can touch it.

    It must be a sweep and not one draw: whether ln2 cancels numerically in the
    constant-folded form depends on the particular row, so one seed can come
    back clean on a build where other rows do not.
    """
    out, trials = {}, 0
    fns = dict(runnable)
    for name in fns:
        out[name] = {"nonzero": 0, "max": 0.0}
    for k in (8, 16, 32, 64, 128):
        for seed in range(8):
            dup = np.repeat(datasets.dirichlet(1, k, 0.1, seed=seed), 6, axis=0)
            trials += 1
            for name, fn in fns.items():
                w = float(np.abs(fn(dup)).max())
                if w != 0.0:
                    out[name]["nonzero"] += 1
                    out[name]["max"] = max(out[name]["max"], w)
    return {"trials": trials,
            "exact_zero": {nm: v["nonzero"] == 0 for nm, v in out.items()},
            "worst_nonzero": {nm: v["max"] for nm, v in out.items()}}


def disjoint_exactness(runnable):
    """Disjoint supports must come back exactly 1.0 in base 2. Not approximately.

    The other end of the scale from the zero check, and as deterministic
    (`docs/MATH.md` G6). Only fastjsd's entries are held to it: SciPy works in
    base e, where the distance has no exact value to hit, and the published
    form makes no such promise. The weights are Dirichlet draws, not powers of
    two, because powers of two make every one-sided total exact by luck and
    would pass on an engine that never snaps a row total.
    """
    fns = {nm: fn for nm, fn in runnable if nm.startswith("fastjsd")}
    misses = dict.fromkeys(fns, 0)
    worst = dict.fromkeys(fns, 0.0)
    trials, n = 0, 4
    off = ~np.eye(n, dtype=bool)
    for w in (2, 4, 8, 16, 32):
        for seed in range(8):
            P = np.zeros((n, n * w))
            for r in range(n):
                P[r, r * w:(r + 1) * w] = datasets.dirichlet(
                    1, w, 0.3, seed=100 * seed + r)[0]
            trials += 1
            for name, fn in fns.items():
                miss = float(np.abs(np.asarray(fn(P), dtype=np.float64)[off]
                                    - 1.0).max())
                if miss != 0.0:
                    misses[name] += 1
                    worst[name] = max(worst[name], miss)
    return {"trials": trials,
            "exact_one": {nm: misses[nm] == 0 for nm in fns},
            "worst_miss": worst}


def accuracy_vs_mpmath(n=16, k=24, f32_ok=True):
    """Error against 50-digit truth, per regime, per available backend.

    Deterministic, so it is directly comparable across boxes -- unlike every
    millisecond in this file. Skipped, with a reason, where mpmath is absent.

    Both the absolute and the relative error are recorded, because on the
    near-duplicate regime the relative one is enormous by construction: the
    true distances there are ~1e-9, so a 1e-8 absolute error is a relative
    error of tens. That is the ill-conditioned band the package documents, not
    a defect, and a single field named "error" invites reading it as one.

    numba-f32 is measured against the truth of the float32 rows it actually
    receives. Rows closer than float32 can resolve round to the SAME row, whose
    exact distance is 0.0, so against the float64 input the near-duplicate
    regime would report that rounding as a 100 % error of the kernel.
    """
    try:
        import accuracy as acc
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    out = {"status": "ok", "n": n, "k": k, "per_regime": {}}
    for name in ("random", "near-duplicate", "partial-overlap"):
        P = acc.regime(name, n, k, seed=acc.SEEDS[name])
        ref = acc.exact_matrix(P)
        per_backend = {}
        for b in available_backends():
            if b == F32_SUBJECT:
                if not f32_ok:
                    continue
                P32 = prepare(P, dtype=np.float32).astype(np.float64)
                abs_err, rel_err = acc.errors(P32, acc.exact_matrix(P32),
                                              jsd_matrix(P, backend=b))
                per_backend[b] = {"abs": float(abs_err), "rel": float(rel_err),
                                  "reference": "its float32-rounded input"}
                continue
            abs_err, rel_err = acc.errors(P, ref, jsd_matrix(P, backend=b))
            per_backend[b] = {"abs": float(abs_err), "rel": float(rel_err)}
        out["per_regime"][name] = per_backend
    return out


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------

def validity(cases, steal, load_before, load_after, nivcsw_delta, wall):
    """The leg grades its own reliability against published thresholds.

    Grading happens in the leg rather than in the aggregator so that the
    aggregator's rule stays mechanical: drop unusable, warn on suspect.
    """
    reasons, verdict = [], "ok"

    def worse(level):
        nonlocal verdict
        order = {"ok": 0, "suspect": 1, "unusable": 2}
        if order[level] > order[verdict]:
            verdict = level

    if steal is not None:
        if steal >= STEAL_UNUSABLE:
            reasons.append(f"steal_frac {steal:.3f} >= {STEAL_UNUSABLE}")
            worse("unusable")
        elif steal >= STEAL_SUSPECT:
            reasons.append(f"steal_frac {steal:.3f} >= {STEAL_SUSPECT}")
            worse("suspect")

    for row in cases:
        me = row["impls"].get("fastjsd auto")
        if not me or me.get("status") != "ok" or len(me["ms"]) < 2:
            continue
        med = me["ms_median"] or 1.0
        spread = (max(me["ms"]) - min(me["ms"])) / med
        if spread >= SPREAD_UNUSABLE:
            reasons.append(f"{row['case']}: fastjsd ms spread {spread:.2f}")
            worse("unusable")
        elif spread >= SPREAD_SUSPECT:
            reasons.append(f"{row['case']}: fastjsd ms spread {spread:.2f}")
            worse("suspect")
        # The min-of-N duration bias: if a slow baseline's minimum sits well
        # below its own median while fastjsd's does not, the ratio for that
        # case is inflated by noise rather than earned by the kernel.
        mine = me["ms_min"] / med
        for nm, rec in row["impls"].items():
            if nm.startswith("fastjsd") or rec.get("status") != "ok":
                continue
            if not rec["ms_median"]:
                continue
            skew = rec["ms_min"] / rec["ms_median"]
            if skew < SKEW_WARN and mine > skew + 0.05:
                reasons.append(
                    f"{row['case']}: {nm} min/median {skew:.2f} vs fastjsd "
                    f"{mine:.2f} -- ratio inflated by min-of-N")
                worse("suspect")

    return {"steal_frac": steal, "loadavg_before": load_before,
            "loadavg_after": load_after, "nivcsw_delta": nivcsw_delta,
            "timer_resolution_s": time.get_clock_info("perf_counter").resolution,
            "wall_seconds": round(wall, 2),
            "verdict": verdict, "reasons": reasons,
            "thresholds": {"steal_suspect": STEAL_SUSPECT,
                           "steal_unusable": STEAL_UNUSABLE,
                           "spread_suspect": SPREAD_SUSPECT,
                           "spread_unusable": SPREAD_UNUSABLE,
                           "skew_warn": SKEW_WARN}}


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def print_f32(f32):
    title = (f"{F32_SUBJECT} against {F32_BASELINE}, dense cases, float32 input "
             f"for both (ratio = {F32_BASELINE} / {F32_SUBJECT}):")
    print("\n" + title)
    if f32["status"] != "ok":
        print(f"  absent: {f32['reason']}")
        return
    if not f32["cases"]:
        print("  no case in this set goes to a dense kernel")
        return
    print(f"  {'case':34}{F32_BASELINE:>11}{F32_SUBJECT:>11}{'ratio':>9}"
          f"{'max err':>10}")
    for row in f32["cases"]:
        if row["status"] != "ok":
            print(f"  {row['case']:34}failed: {row['reason']}")
            continue
        r = row["ratio"]
        cell = f"{r['median']:.2f}x" if r["resolvable"] else "~"
        print(f"  {row['case']:34}"
              f"{row['impls'][F32_BASELINE]['ms_min']:>11,.1f}"
              f"{row['impls'][F32_SUBJECT]['ms_min']:>11,.1f}"
              f"{cell:>9}{row['max_abs_err']:>10.1e}")
    bad = [row for row in f32["cases"]
           if row["status"] == "ok" and not row["agrees"]]
    for row in bad:
        print(f"*** DISAGREEMENT: {F32_SUBJECT} on {row['case']} is "
              f"{row['max_abs_err']:.2e} from the float64 kernel, above "
              f"{row['tol']:.0e}. Its timing measures something else.")
    slow = [row for row in f32["cases"]
            if row["status"] == "ok" and row["ratio"]["median"] < 1.0]
    for row in slow:
        print(f"*** {F32_SUBJECT} was SLOWER than {F32_BASELINE} on "
              f"{row['case']}: {row['ratio']['median']:.2f}x")


def print_table(doc):
    names = []
    for row in doc["cases"]:
        for nm, rec in row["impls"].items():
            if rec.get("status") == "ok" and nm not in names:
                names.append(nm)
    base_names = [n for n in names if not n.startswith("fastjsd")]

    w = max([len(n) for n in names] + [9]) + 2
    head = ("case".ljust(34) + "".join(n.rjust(w) for n in names)
            + "".join((n + " x").rjust(w) for n in base_names))
    print("\n" + head)
    print("-" * len(head))
    for row in doc["cases"]:
        line = row["case"].ljust(34)
        for nm in names:
            rec = row["impls"].get(nm, {})
            line += ("absent" if rec.get("status") != "ok"
                     else f"{rec['ms_min']:,.1f}").rjust(w)
        for nm in base_names:
            r = row["ratios"].get(nm)
            if r is None:
                cell = "-"
            elif not r["resolvable"]:
                cell = "~"
            else:
                cell = f"{r['median']:,.2f}x"
            line += cell.rjust(w)
        print(line)
    print(f"\nmilliseconds, min of {doc['config']['reps']} x "
          f"{doc['config']['repeats']} interleaved repeats.")
    print("THE MILLISECONDS ARE NOT COMPARABLE ACROSS BOXES -- they are here so")
    print("the ratios can be audited. The ratio columns are the result. '~' means")
    print("the across-repeat spread straddles 1.0, i.e. this box cannot tell the")
    print("two implementations apart; report 'indistinguishable', not a number.")

    bad = [(row["case"], nm, rec["max_abs_err"], rec["tol"])
           for row in doc["cases"] for nm, rec in row["impls"].items()
           if rec.get("status") == "ok" and rec.get("agrees") is False]
    if bad:
        print("\n*** DISAGREEMENT: these implementations did not match the")
        print(f"*** reference ({doc['config']['reference']}) to their own")
        print("*** tolerance. Their timings above measure something else.")
        for case, nm, err, tol in bad:
            print(f"      {case:34} {nm:22} {err:.2e} > {tol:.0e}")

    losses = [(row["case"], nm, r["median"])
              for row in doc["cases"] for nm, r in row["ratios"].items()
              if r["median"] < 1.0]
    if losses:
        print("\n*** fastjsd LOST on this box:")
        for case, nm, med in losses:
            print(f"      {case:34} vs {nm:22} {med:.2f}x")

    if not doc["software"]["numba"]:
        print("\nNUMBA IS NOT INSTALLED on this box. Only the NumPy backends ran,")
        print("which measure the support-intersection identity rather than the")
        print("kernel -- a different claim. On dense inputs they can be SLOWER")
        print("than SciPy. Keep these numbers out of the headline range.")
    if not doc["software"]["scipy"]:
        print("\nSCIPY IS NOT INSTALLED on this box. There is no independent")
        print("reference and no SciPy baseline; agreement was checked against")
        print("fastjsd's own NumPy backend, which is not a check.")

    print_f32(doc["f32"])

    ex = doc["accuracy"]["duplicate_rows_exact_zero"]
    print("\nidentical rows, exactly 0.0 "
          f"({doc['accuracy']['trials']} draws, deterministic -- this DOES "
          "travel across boxes):")
    for nm, ok in ex.items():
        worst = doc["accuracy"]["worst_nonzero"][nm]
        print(f"  {nm:24} " + ("exact 0.0 in all" if ok
                               else f"NONZERO, max {worst:.2e}"))

    one = doc["accuracy"]["disjoint_supports_exact_one"]
    print("\ndisjoint supports, exactly 1.0 in base 2 "
          f"({doc['accuracy']['disjoint_trials']} draws, deterministic):")
    for nm, ok in one.items():
        worst = doc["accuracy"]["disjoint_worst_miss"][nm]
        print(f"  {nm:24} " + ("exact 1.0 in all" if ok
                               else f"MISSED, worst |d - 1| {worst:.2e}"))

    vs = doc["accuracy"]["vs_mpmath"]
    per = {reg: be.get(F32_SUBJECT)
           for reg, be in vs.get("per_regime", {}).items()}
    if any(per.values()):
        print(f"\n{F32_SUBJECT} against 50-digit truth of its float32-rounded "
              "input (deterministic):")
        for reg, e in per.items():
            if e:
                print(f"  {reg:24} abs {e['abs']:.1e}   rel {e['rel']:.1e}")

    v = doc["validity"]
    print(f"\nvalidity: {v['verdict']}   steal={v['steal_frac']}   "
          f"wall={v['wall_seconds']}s")
    for r in v["reasons"]:
        print(f"  ! {r}")

    print("\ncaveats printed with the numbers, not linked from them:")
    for c in doc["caveats"]:
        print(f"  - {c}")


def main():
    ap = argparse.ArgumentParser(
        description="one leg of a cross-machine fastjsd comparison")
    ap.add_argument("--json", type=str, default=None,
                    help="write the leg document here")
    ap.add_argument("--leg-id", type=str,
                    default=os.environ.get("FASTJSD_CI_LEG", "local"),
                    help="identifies this box in the aggregated report")
    ap.add_argument("--reps", type=int, default=5,
                    help="runs per implementation per repeat; the minimum is kept")
    ap.add_argument("--repeats", type=int, default=3,
                    help="whole-suite repetitions; each yields one ratio, and "
                         "their spread is what decides `resolvable`")
    ap.add_argument("--full-suite", action="store_true",
                    help="use datasets.suite() instead of the small CI case set "
                         "(several times slower -- not for a hosted runner)")
    ap.add_argument("--threads", action="store_true",
                    help="also scan thread counts (recorded, never concluded from)")
    ap.add_argument("--crossover", action="store_true",
                    help="also measure regret against the shipped "
                         "SPARSE_DENSITY_MAX")
    ap.add_argument("--fingerprint-only", action="store_true",
                    help="print the machine and software blocks and exit; "
                         "no timing whatsoever")
    args = ap.parse_args()

    steal_before = None
    try:
        steal_before = _cpu_times()
    except Exception:
        pass
    load_before, nivcsw_before = _loadavg(), _nivcsw()
    t_start = time.perf_counter()

    # Probe first: it pays every JIT, and it is what makes
    # `numba.threading_layer()` answerable instead of raising.
    runnable, absent = probe()
    f32_fn, f32_absent = probe_f32()
    doc = {"schema": SCHEMA, "leg_id": args.leg_id,
           "produced_at_utc": datetime.now(timezone.utc).isoformat(),
           "machine": machine(args.leg_id), "software": software()}

    if args.fingerprint_only:
        print(json.dumps({"schema": doc["schema"], "leg_id": doc["leg_id"],
                          "machine": doc["machine"],
                          "software": doc["software"],
                          "runnable": [n for n, _ in runnable],
                          "absent": absent,
                          F32_SUBJECT: f32_absent or "runs"},
                         indent=2, default=str))
        return

    ref_name = ("scipy pdist" if any(n == "scipy pdist" for n, _ in runnable)
                else "fastjsd numpy")
    cases = portable_cases() if not args.full_suite else datasets.suite()
    doc["config"] = {"reps": args.reps, "repeats": args.repeats,
                     "case_set": "suite" if args.full_suite else "portable",
                     "reference": ref_name,
                     "reference_is_independent": ref_name == "scipy pdist",
                     "dtype": "float64"}
    doc["absent"] = absent
    doc["cases"] = [bench_case(label, P, runnable, absent, args, ref_name)
                    for label, P in cases]
    doc["f32"] = f32_section(cases, args, f32_fn, f32_absent)

    doc["threads"] = threads_scan(cases[0][1], args) if args.threads else []
    if args.crossover:
        doc["crossover"], doc["crossover_summary"] = crossover(args)
    else:
        doc["crossover"], doc["crossover_summary"] = [], None

    acc = accuracy_vs_mpmath(f32_ok=f32_fn is not None)
    exact_fns = list(runnable)
    if f32_fn is not None:
        exact_fns.append((f"fastjsd {F32_SUBJECT}", f32_fn))
    zero = exactness(exact_fns)
    one = disjoint_exactness(exact_fns)
    doc["accuracy"] = {"trials": zero["trials"],
                       "duplicate_rows_exact_zero": zero["exact_zero"],
                       "worst_nonzero": zero["worst_nonzero"],
                       "disjoint_trials": one["trials"],
                       "disjoint_supports_exact_one": one["exact_one"],
                       "disjoint_worst_miss": one["worst_miss"],
                       "vs_mpmath": acc}

    steal_after = None
    try:
        steal_after = _cpu_times()
    except Exception:
        pass
    nivcsw_after = _nivcsw()
    delta = (None if nivcsw_before is None or nivcsw_after is None
             else nivcsw_after - nivcsw_before)
    doc["validity"] = validity(doc["cases"],
                               _steal_fraction(steal_before, steal_after),
                               load_before, _loadavg(), delta,
                               time.perf_counter() - t_start)
    doc["caveats"] = list(CAVEATS)

    print_table(doc)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
