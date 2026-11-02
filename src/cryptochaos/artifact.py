"""verify-artifact: completeness, integrity, identity, and secret-hygiene checks for a run directory."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import environment
from .dataset import read_manifest, select

FORBIDDEN_COLUMNS = {"password", "passphrase", "secret", "secret_key", "shared_secret", "private_key", "psk",
                     "pw_key", "aead_key", "decapsulation_key", "seed"}


def _secret_bearing(column: str) -> bool:
    c = column.lower()
    return c in FORBIDDEN_COLUMNS or c.endswith(("_secret", "_password", "_passphrase", "_private_key"))


class _RunView:
    """Read-only view exposing what ``benchmark.expected_keys`` needs."""

    def __init__(self, run: Path) -> None:
        self.dir = run
        self.cfg = json.loads((run / "config.effective.json").read_text(encoding="utf-8"))
        self.root = environment.repo_root()
        m = self.cfg["dataset"]["manifest"]
        path = (self.root / m) if m else (run / "manifest.csv")
        self.images = select(read_manifest(path), self.cfg["dataset"]["selection"], self.root) if path.exists() else []


def _check_sums(base: Path, sums: Path) -> tuple[list[str], set[str]]:
    """Files listed in a SHA256SUMS file that are missing or differ, and the set of listed paths."""
    bad, listed = [], set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        f = base / rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    return bad, listed


def _json_keys(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [k for v in obj.values() for k in _json_keys(v)]
    if isinstance(obj, list):
        return [k for v in obj for k in _json_keys(v)]
    return []


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False).stdout


def measurement_hash_at(root: Path, commit: str) -> str:
    """``measurement_source_sha256`` of the package as stored in a Git commit (same rule as the runner)."""
    names = _git(root, "ls-tree", "-r", "--name-only", commit, "--", "src/cryptochaos").decode().split()
    h = hashlib.sha256()
    for rel in sorted(n for n in names if n.endswith(".py")):
        if Path(rel).name in environment.NON_MEASUREMENT_MODULES:
            continue
        data = _git(root, "show", f"{commit}:{rel}").replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\0" + data)
    return h.hexdigest()


def find_measurement_commit(root: Path, target: str, limit: int = 500) -> str | None:
    """Newest commit whose measurement code has the given hash (only commits that changed the package)."""
    for commit in _git(root, "log", f"--max-count={limit}", "--format=%H", "--", "src/cryptochaos").decode().split():
        if measurement_hash_at(root, commit) == target:
            return commit
    return None


def verify_hashes_only(run: Path) -> dict[str, Any]:
    """For output directories without a run configuration (the GPU diagnostics): SHA-256 check only."""
    report: dict[str, Any] = {"run": str(run), "mode": "hashes only", "problems": [], "warnings": []}
    sums = run / "SHA256SUMS"
    if not sums.exists():
        report["problems"].append("SHA256SUMS missing")
    else:
        bad, listed = _check_sums(run, sums)
        report["hash_mismatches"] = bad
        if bad:
            report["problems"].append(f"{len(bad)} files differ from SHA256SUMS")
        unlisted = [f.relative_to(run).as_posix() for f in run.rglob("*") if f.is_file()
                    and f.relative_to(run).as_posix() not in listed and f.name != "SHA256SUMS"]
        if unlisted:
            report["warnings"].append(f"files not covered by SHA256SUMS: {unlisted[:10]}")
    report["ok"] = not report["problems"]
    return report


def verify_run(run: Path) -> dict[str, Any]:
    from .benchmark import expected_keys

    run = run.resolve()
    if not (run / "config.effective.json").exists():
        return verify_hashes_only(run)
    view = _RunView(run)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {"run": str(run), "passes": {}, "problems": [], "warnings": []}
    for name in view.cfg["experiment"]["passes"]:
        info: dict[str, Any] = {"complete_marker": (run / "status" / f"{name}.json").exists()}
        path = run / "raw" / f"{name}.csv"
        exp = expected_keys(view, name)
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                cols = reader.fieldnames or []
                rows = list(reader)
            keys = [r["key"] for r in rows]
            info["rows"] = len(rows)
            info["duplicate_keys"] = len(keys) - len(set(keys))
            info["error_rows"] = sum(1 for r in rows if r.get("error"))
            info["not_ok_rows"] = sum(1 for r in rows if r.get("ok") == "False")
            bad_cols = [c for c in cols if _secret_bearing(c)]
            if bad_cols:
                report["problems"].append(f"{name}: columns that look secret-bearing: {bad_cols}")
            if exp is None:
                report["warnings"].append(f"{name}: completeness not enumerated; {len(rows)} rows present")
            else:
                info["expected_rows"] = len(exp)
                info["missing_rows"] = len(exp - set(keys))
                info["unexpected_rows"] = len(set(keys) - exp)
                if info["missing_rows"]:
                    report["problems"].append(f"{name}: {info['missing_rows']} expected rows missing")
                if info["unexpected_rows"]:
                    report["warnings"].append(f"{name}: {info['unexpected_rows']} rows not in the enumeration")
            if info["duplicate_keys"]:
                report["problems"].append(f"{name}: duplicate keys")
            if info["error_rows"] or info["not_ok_rows"]:
                report["warnings"].append(f"{name}: {info['error_rows']} error rows, {info['not_ok_rows']} not-ok rows")
        elif name not in ("environment", "map_analysis"):
            report["problems"].append(f"{name}: raw file missing")
        if not info["complete_marker"]:
            report["problems"].append(f"{name}: no completion marker")
        report["passes"][name] = info
    sums = run / "SHA256SUMS"
    if sums.exists():
        bad = []
        listed = set()
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, rel = line.split("  ", 1)
            listed.add(rel)
            p = run / rel
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                bad.append(rel)
        report["hash_mismatches"] = bad
        if bad:
            report["problems"].append(f"{len(bad)} files differ from SHA256SUMS")
        unlisted = [p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()
                    and p.relative_to(run).as_posix() not in listed and p.name != "SHA256SUMS"
                    and not p.relative_to(run).as_posix().startswith("report/")]
        if unlisted:
            report["warnings"].append(f"files not covered by SHA256SUMS: {unlisted[:10]}")
    else:
        report["problems"].append("SHA256SUMS missing (run not finalized)")
    # derived outputs: report/ carries its own SHA256SUMS, written by `report`
    rdir = run / "report"
    if (rdir / "SHA256SUMS").exists():
        rbad, rlisted = _check_sums(rdir, rdir / "SHA256SUMS")
        report["report_hash_mismatches"] = rbad
        if rbad:
            report["problems"].append(f"{len(rbad)} report files differ from report/SHA256SUMS")
        runl = [f.relative_to(rdir).as_posix() for f in rdir.rglob("*") if f.is_file()
                and f.relative_to(rdir).as_posix() not in rlisted and f.name != "SHA256SUMS"]
        if runl:
            report["warnings"].append(f"report files not covered by report/SHA256SUMS: {runl[:10]}")
    elif rdir.exists():
        report["warnings"].append("report/ has no SHA256SUMS; regenerate it with `report`")
    # secret hygiene beyond CSV headers: key names in the run's JSON records
    for jf in sorted(run.rglob("*.json")):
        try:
            keys = _json_keys(json.loads(jf.read_text(encoding="utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue
        # "seed" marks key material only as a CSV column; in JSON records it is the experiment seed or the
        # configuration D map seed of the public demo passphrases (map analysis)
        bad_keys = sorted({k for k in keys if _secret_bearing(k) and k.lower() != "seed"})
        if bad_keys:
            report["problems"].append(f"{jf.relative_to(run).as_posix()}: secret-looking keys {bad_keys}")
    current = environment.source_identity(view.root)["measurement_source_sha256"]
    stored = meta["identity"].get("measurement_source_sha256")
    report["measurement_source_matches_current_code"] = current == stored
    if current != stored:
        commit = find_measurement_commit(view.root, stored) if stored else None
        report["measurement_source_commit"] = commit
        report["warnings"].append(
            f"measurement code differs from the current code; it matches commit {commit[:7]} of this repository"
            if commit else "measurement code differs from the code that produced this run")
    report["source_at_start"] = meta.get("source")
    report["ok"] = not report["problems"]
    return report
