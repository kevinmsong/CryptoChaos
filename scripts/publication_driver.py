"""Drive the principal run in three stages, pausing named background workloads only during timing.

Stage 1: non-timing passes (environment, map analysis, authentication suite, key sensitivity).
Stage 2: timing-sensitive passes (stages, timing, scaling, KDF sensitivity, memory). Processes whose
         command line matches --pause-pattern are suspended (never killed) for this stage only and
         resumed in a finally block; their PIDs are written to --pid-file for manual recovery.
Stage 3: CPU-parallel, non-timing passes (calibration, diagnostics).

    python scripts/publication_driver.py --config configs/principal.toml \
        --pause-pattern <command-line-substring> --pid-file runs/paused_pids.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

STAGES = [
    ("non-timing", ["environment", "map_analysis", "auth", "key_sensitivity"], False),
    ("timing", ["stages", "timing", "scaling", "kdf_sensitivity", "memory"], True),
    ("parallel", ["calibration", "diagnostics"], False),
]


def log(msg: str) -> None:
    print(f"[driver {time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


PYTHON_NAMES = ("python.exe", "pythonw.exe", "python", "python3")


def named_processes(names: list[str]) -> list[psutil.Process]:
    """Processes whose executable name matches one of ``names`` (case-insensitive), never this driver."""
    me = psutil.Process()
    protected = {me.pid} | {p.pid for p in me.parents()} | {c.pid for c in me.children(recursive=True)}
    wanted = {n.lower() for n in names}
    return [p for p in psutil.process_iter(["pid", "name"])
            if (p.info["name"] or "").lower() in wanted and p.pid not in protected]


def matching_processes(pattern: str) -> list[psutil.Process]:
    """Python processes whose command line contains ``pattern``, plus their children.

    Never includes this driver, its ancestors (the shells that launched it, whose command lines
    contain the pattern as an argument), or anything running the benchmark itself.
    """
    me = psutil.Process()
    protected = {me.pid} | {p.pid for p in me.parents()} | {c.pid for c in me.children(recursive=True)}
    found: dict[int, psutil.Process] = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (p.info["name"] or "").lower()
        cmd = " ".join(p.info["cmdline"] or [])
        if p.pid in protected or "publication_driver" in cmd or "cryptochaos" in cmd:
            continue
        if (name in PYTHON_NAMES or name.startswith("python3")) and pattern in cmd:
            found[p.pid] = p
            for c in p.children(recursive=True):
                if c.pid not in protected:
                    found[c.pid] = c
    assert not set(found) & protected, "refusing to suspend the driver or its ancestors"
    return list(found.values())


def run_passes(config: str, passes: list[str], resume: bool) -> int:
    cmd = [sys.executable, "-m", "cryptochaos", "benchmark", "--config", config, "--passes", ",".join(passes)]
    if resume:
        cmd.append("--resume")
    log("running " + " ".join(cmd[2:]))
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pause-pattern", default=None)
    ap.add_argument("--pause-name", action="append", default=[], help="process name to pause (repeatable)")
    ap.add_argument("--pid-file", default="runs/paused_pids.json")
    ap.add_argument("--first-resume", action="store_true", help="the run directory already exists")
    ap.add_argument("--stages", default=None,
                    help="override stages as 'pre-passes;timing-passes;post-passes' (comma-separated pass names)")
    a = ap.parse_args()
    resume = a.first_resume
    stages = STAGES
    if a.stages:
        pre, timing, post = (part.split(",") if part else [] for part in a.stages.split(";"))
        stages = [("non-timing", pre, False), ("timing", timing, True), ("parallel", post, False)]
    for name, passes, pause in stages:
        if not passes:
            continue
        paused: list[psutil.Process] = []
        try:
            if pause and (a.pause_pattern or a.pause_name):
                paused = (matching_processes(a.pause_pattern) if a.pause_pattern else []) + named_processes(a.pause_name)
                Path(a.pid_file).write_text(json.dumps([p.pid for p in paused]), encoding="utf-8")
                for p in paused:
                    p.suspend()
                log(f"suspended {len(paused)} processes ({a.pause_pattern!r}, {a.pause_name}): "
                    f"{[(p.pid, p.name()) for p in paused]}")
            rc = run_passes(a.config, passes, resume)
        finally:
            for p in paused:
                try:
                    p.resume()
                except psutil.NoSuchProcess:
                    pass
            if paused:
                states = {p.pid: (p.status() if p.is_running() else "exited") for p in paused}
                log(f"resumed paused processes: {states}")
        resume = True
        if rc != 0:
            log(f"stage {name} failed with exit code {rc}; stopping")
            return rc
        log(f"stage {name} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
