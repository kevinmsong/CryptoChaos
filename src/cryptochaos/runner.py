"""Run directories: identity, checkpoint/resume, append-only raw CSV sinks, load sampling, hashing."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import psutil

from . import environment
from .config import config_hash
from .dataset import ImageRecord, manifest_sha256, read_manifest, resolve_dataset_path, select, build_manifest, \
    write_manifest
from .errors import ConfigurationError


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repair_tail(path: Path) -> int:
    """Drop an incomplete final row left by a crash (NUL padding or a line without newline).

    Rows are flushed one at a time, so only the last line can be torn. Returns the number of bytes
    removed (recorded in run.json by the caller when non-zero).
    """
    data = path.read_bytes()
    clean = data.rstrip(b"\x00")
    if clean and not clean.endswith(b"\n"):
        clean = clean[: clean.rfind(b"\n") + 1]
    removed = len(data) - len(clean)
    if removed:
        path.write_bytes(clean)
    return removed


class CsvSink:
    """Append-only CSV with a fixed header and a unique ``key`` column (enables resume)."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        if "key" not in columns:
            raise ValueError("sinks need a key column")
        self.path = path
        self.columns = columns
        self.keys: set[str] = set()
        self.repaired_bytes = repair_tail(path) if path.exists() else 0
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames != columns:
                    raise ConfigurationError(f"{path.name}: existing columns differ from this code version")
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ConfigurationError(f"{path.name}: malformed row {reader.line_num}; refusing to resume")
                    self.keys.add(row["key"])
        self.fh = path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        if not exists:
            self.writer.writeheader()
            self.fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        if row["key"] in self.keys:
            raise ConfigurationError(f"duplicate key {row['key']} in {self.path.name}")
        self.writer.writerow({c: row.get(c) for c in self.columns})
        self.keys.add(row["key"])
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LoadSampler:
    """System CPU-utilization sampling during timing passes (QC only).

    Runs in a separate process (``python -m cryptochaos.runner --sample-load``) so that it never
    competes with the timed process for the interpreter lock. On Windows ``psutil.cpu_freq``
    reports the nominal frequency, so the frequency column is informative only on Linux.
    """

    def __init__(self, path: Path, interval: float) -> None:
        self.path, self.interval = path, interval
        self._proc: subprocess.Popen | None = None

    def __enter__(self):
        self._proc = subprocess.Popen([sys.executable, "-m", "cryptochaos.runner", "--sample-load", str(self.path),
                                       str(self.interval)], stdin=subprocess.PIPE)
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            try:
                self._proc.stdin.close()  # EOF tells the sampler to finish its current line and exit
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()


def _sample_load(path: Path, interval: float) -> None:
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(["wall_time_utc", "cpu_percent_total", "cpu_percent_max_core", "cpu_freq_mhz"])
        psutil.cpu_percent(percpu=True)
        while not stop.wait(interval):
            per = psutil.cpu_percent(percpu=True)
            f = psutil.cpu_freq()
            w.writerow([utc_now(), round(sum(per) / len(per), 1), max(per), None if f is None else f.current])
            fh.flush()


class RunContext:
    def __init__(self, cfg: dict, resume: bool, root: Path | None = None) -> None:
        self.cfg = cfg
        self.root = root or environment.repo_root()
        exp = cfg["experiment"]
        self.dir = (self.root / exp["output_root"] / exp["name"]).resolve()
        self.raw = self.dir / "raw"
        self.status = self.dir / "status"
        self._images: list[ImageRecord] | None = None
        self.manifest_path = self._manifest_path()
        identity = self.identity()
        run_json = self.dir / "run.json"
        if run_json.exists():
            if not resume:
                raise ConfigurationError(f"{self.dir} already exists; pass --resume or change experiment.name")
            stored = json.loads(run_json.read_text(encoding="utf-8"))["identity"]
            mismatch = [k for k in identity if stored.get(k) != identity[k]]
            if mismatch:
                raise ConfigurationError(f"refusing to resume: identity mismatch in {mismatch}")
            self.meta = json.loads(run_json.read_text(encoding="utf-8"))
            self.meta.setdefault("resumed_at_utc", []).append(utc_now())
        else:
            for d in (self.raw, self.status):
                d.mkdir(parents=True, exist_ok=True)
            self.meta = {"identity": identity, "started_at_utc": utc_now(), "config_source": cfg.get("_source"),
                         "source": environment.source_identity(self.root)}
            (self.dir / "config.effective.json").write_text(
                json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=1), encoding="utf-8")
            src = cfg.get("_source", {}).get("path")
            if src and Path(src).exists():
                shutil.copyfile(src, self.dir / "config.toml")
        self.save_meta()

    # --- identity ---------------------------------------------------------------------------
    def _manifest_path(self) -> Path | None:
        if not {"timing", "diagnostics", "key_sensitivity", "memory", "legacy_reconstruction"} & set(
                self.cfg["experiment"]["passes"]):
            return None
        m = self.cfg["dataset"]["manifest"]
        if m:
            return (self.root / m).resolve()
        return self.dir / "manifest.csv"  # built from dataset.path at run start (smoke/tests)

    def identity(self) -> dict[str, Any]:
        b = environment.backend_identity()
        kem = self.cfg["protocol"]["kem_backend"]
        data = None
        if self.manifest_path is not None:
            if not self.manifest_path.exists() and not self.cfg["dataset"]["manifest"]:
                self.dir.mkdir(parents=True, exist_ok=True)
                write_manifest(build_manifest(resolve_dataset_path(self.cfg["dataset"]), self.root), self.manifest_path)
            data = {"manifest_sha256": manifest_sha256(self.manifest_path),
                    "selection": hashlib.sha256("\n".join(r.image_id for r in self.images).encode()).hexdigest()}
        return {
            "config_sha256": config_hash(self.cfg),
            "measurement_source_sha256": environment.source_identity(self.root)["measurement_source_sha256"],
            "data": data,
            "kem_backend": b["kem_backends"].get(kem),
            "libraries": {k: b["packages"].get(k) for k in ("cryptography", "pqcrypto", "kyber-py", "blake3", "numpy",
                                                              "scipy")},
            "python": b["python"],
        }

    @property
    def images(self) -> list[ImageRecord]:
        if self._images is None:
            if self.manifest_path is None:
                self._images = []
            else:
                self._images = select(read_manifest(self.manifest_path), self.cfg["dataset"]["selection"], self.root)
        return self._images

    def image_path(self, rec: ImageRecord) -> Path:
        return resolve_dataset_path(self.cfg["dataset"]) / rec.filename

    # --- bookkeeping ------------------------------------------------------------------------
    def save_meta(self) -> None:
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    def sink(self, name: str, columns: list[str]) -> CsvSink:
        sink = CsvSink(self.raw / f"{name}.csv", columns)
        if sink.repaired_bytes:
            self.meta.setdefault("tail_repairs", []).append(
                {"file": f"raw/{name}.csv", "bytes_removed": sink.repaired_bytes, "at_utc": utc_now()})
            self.save_meta()
        return sink

    def is_done(self, name: str) -> bool:
        return (self.status / f"{name}.json").exists()

    def mark_done(self, name: str, summary: dict[str, Any]) -> None:
        summary = dict(summary, finished_at_utc=utc_now())
        (self.status / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def finalize(self) -> None:
        self.meta["finished_at_utc"] = utc_now()
        self.save_meta()
        write_hashes(self.dir)


def write_hashes(run_dir: Path) -> Path:
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run_dir).as_posix()}")
    out = run_dir / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--sample-load":
    _sample_load(Path(sys.argv[2]), float(sys.argv[3]))
