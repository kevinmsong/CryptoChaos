"""Environment, source, and backend identity capture for run provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from .kem import available_kem_backends, get_kem_backend

# Modules that cannot change measured values; editing them must not invalidate a resumable run.
NON_MEASUREMENT_MODULES = {"reporting.py", "artifact.py", "gpu_metrics.py", "cli.py", "__main__.py"}

PACKAGES = ("cryptography", "pqcrypto", "kyber-py", "blake3", "numpy", "scipy", "pandas", "pillow", "matplotlib",
            "psutil", "pytest", "torch", "triton-windows", "triton")


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() and (p / "src" / "cryptochaos").exists():
            return p
    return Path.cwd()


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        for base in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
                     Path(r"C:\Program Files\NVIDIA Corporation\NVSMI")):
            cand = base / "nvidia-smi.exe"
            if cand.exists():
                return str(cand)
        store = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "DriverStore" / "FileRepository"
        if store.exists():
            for cand in store.glob("nv*/nvidia-smi.exe"):
                return str(cand)
    return None


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"nvidia_smi": None, "gpus": [], "torch": None}
    smi = find_nvidia_smi()
    if smi:
        info["nvidia_smi"] = smi
        q = _run([smi, "--query-gpu=index,name,driver_version,memory.total,pci.bus_id,compute_cap",
                  "--format=csv,noheader"])
        if q:
            for line in q.splitlines():
                parts = [p.strip() for p in line.split(",")]
                info["gpus"].append(dict(zip(("index", "name", "driver", "memory", "pci_bus", "compute_cap"), parts)))
        info["cuda_driver"] = _run([smi, "--query-gpu=driver_version", "--format=csv,noheader"])
    try:
        import torch  # optional

        t: dict[str, Any] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                             "cuda_runtime": torch.version.cuda}
        if torch.cuda.is_available():
            t["devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                             "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                             "total_memory": torch.cuda.get_device_properties(i).total_memory}
                            for i in range(torch.cuda.device_count())]
        info["torch"] = t
    except Exception as exc:  # noqa: BLE001 - optional dependency probing
        info["torch"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton"] = f"unavailable: {type(exc).__name__}"
    return info


def power_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        bat = psutil.sensors_battery()
        out["on_ac_power"] = None if bat is None else bool(bat.power_plugged)
        out["battery_percent"] = None if bat is None else bat.percent
    except (AttributeError, NotImplementedError):
        out["on_ac_power"] = None
    if sys.platform == "win32":
        out["power_scheme"] = _run(["powercfg", "/getactivescheme"])
    return out


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_identity(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    diff = None
    if status:
        raw = subprocess.run(["git", "diff", "HEAD", "--", "src", "configs", "pyproject.toml"], cwd=root,
                             capture_output=True).stdout
        diff = hashlib.sha256(raw).hexdigest() if raw else None
    src_hash, meas_hash = hashlib.sha256(), hashlib.sha256()
    for p in sorted((root / "src" / "cryptochaos").rglob("*.py")):
        chunk = p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n")
        src_hash.update(chunk)
        if p.name not in NON_MEASUREMENT_MODULES:
            meas_hash.update(chunk)
    return {"git_commit": commit, "git_dirty": bool(status), "dirty_patch_sha256": diff,
            "package_source_sha256": src_hash.hexdigest(), "measurement_source_sha256": meas_hash.hexdigest()}


def backend_identity() -> dict[str, Any]:
    import ssl

    out: dict[str, Any] = {"python": sys.version, "python_openssl": ssl.OPENSSL_VERSION,
                           "packages": package_versions(), "kem_backends": {}}
    for name in available_kem_backends():
        out["kem_backends"][name] = get_kem_backend(name).provenance()
    return out


def capture() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    clock = time.get_clock_info("perf_counter")
    return {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": None if freq is None else {"current": freq.current, "max": freq.max},
        "ram_bytes": vm.total,
        "perf_counter": {"implementation": clock.implementation, "resolution_s": clock.resolution},
        "power": power_state(),
        "gpu": gpu_info(),
        "source": source_identity(),
        "backends": backend_identity(),
    }


def prevent_sleep(enable: bool) -> bool:
    """Ask Windows not to sleep while a run is active (no effect elsewhere)."""
    if sys.platform != "win32":
        return False
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


def set_priority(level: str) -> str:
    proc = psutil.Process()
    if level == "high":
        try:
            proc.nice(psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -5)
        except (psutil.AccessDenied, PermissionError, AttributeError):
            return "requested-high-denied"
    return level
