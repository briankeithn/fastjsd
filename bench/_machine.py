"""Machine and environment fingerprint, recorded alongside every timing.

A latency figure without the machine that produced it is not reproducible.
Harnesses import ``machine()`` from here rather than each assembling its own
description, so the fields they record stay comparable across runs.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def cpu_name():
    """The most specific CPU string this operating system provides."""
    if sys.platform == "win32":
        if os.environ.get("PROCESSOR_IDENTIFIER"):
            return os.environ["PROCESSOR_IDENTIFIER"]
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
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def machine():
    """Compact environment description: interpreter, OS, CPU, library versions."""
    import numpy

    from fastjsd import available_backends

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": cpu_name(),
        "logical_cores": os.cpu_count(),
        "numpy": numpy.__version__,
        "backends": list(available_backends()),
    }
    for mod in ("scipy", "numba", "mpmath"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        import numba
        info["numba_threads"] = numba.config.NUMBA_NUM_THREADS
        info["numba_svml"] = bool(getattr(numba.config, "USING_SVML", False))
    except ImportError:
        pass
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1 << 30), 1)
        bat = psutil.sensors_battery()
        if bat is not None:
            info["on_battery"] = not bat.power_plugged
    except Exception:
        pass
    return info
