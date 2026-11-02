"""Dataset manifests with file and decoded-payload hashes; RGB uint8 decoding; selection rules.

Images are opened read-only and decoded with Pillow to 8-bit RGB (``Image.convert("RGB")``),
then serialized in C order (row-major, interleaved R, G, B). The payload of an image is exactly
those ``width * height * 3`` bytes. Spatial pixels are ``width * height``; channel samples are
``width * height * 3``.
"""
from __future__ import annotations

import csv
import hashlib
import os
import stat
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import ConfigurationError

LEGACY_MANIFEST = "legacy/results/20260314_141838/csv/dataset_manifest.csv"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filename: str
    file_bytes: int
    file_sha256: str
    width: int
    height: int
    source_mode: str
    pixels: int
    payload_bytes: int
    decoded_sha256: str
    legacy20: bool
    legacy_category: str


def decode_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        rgb = im if im.mode == "RGB" else im.convert("RGB")
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: decoded array has shape {arr.shape}")
    return arr


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def is_hydrated(path: str | Path) -> bool:
    """False for cloud-only placeholders (OneDrive) that would be downloaded on access."""
    if sys.platform != "win32":
        return True
    attrs = os.stat(path).st_file_attributes
    recall = 0x00400000 | 0x00040000  # RECALL_ON_DATA_ACCESS | RECALL_ON_OPEN
    return not (attrs & stat.FILE_ATTRIBUTE_OFFLINE or attrs & recall)


def resolve_dataset_path(cfg_dataset: dict) -> Path:
    candidates = []
    if cfg_dataset.get("path"):
        candidates.append(Path(cfg_dataset["path"]))
    env = os.environ.get(cfg_dataset.get("path_env", "CRYPTOCHAOS_DIV2K_PATH"))
    if env:
        candidates.append(Path(env))
    for c in candidates:
        if c.is_dir():
            return c
    raise ConfigurationError("dataset directory not found; set CRYPTOCHAOS_DIV2K_PATH (or dataset.path)")


def legacy_selection(root: Path) -> dict[str, str]:
    path = root / LEGACY_MANIFEST
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["filename"]: row["category"] for row in csv.DictReader(fh)}


def build_manifest(image_dir: Path, root: Path, pattern: str = "*.png") -> list[ImageRecord]:
    legacy = legacy_selection(root)
    records = []
    for p in sorted(image_dir.glob(pattern)):
        if not p.is_file():
            continue
        arr = decode_rgb(p)
        with Image.open(p) as im:
            mode = im.mode
        payload = arr.tobytes()
        records.append(ImageRecord(
            image_id=p.stem, filename=p.name, file_bytes=p.stat().st_size, file_sha256=sha256_file(p),
            width=arr.shape[1], height=arr.shape[0], source_mode=mode, pixels=arr.shape[0] * arr.shape[1],
            payload_bytes=len(payload), decoded_sha256=hashlib.sha256(payload).hexdigest(),
            legacy20=p.name in legacy, legacy_category=legacy.get(p.name, "")))
    return records


def write_manifest(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(ImageRecord)], lineterminator="\n")
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def read_manifest(path: Path) -> list[ImageRecord]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(ImageRecord(
                image_id=row["image_id"], filename=row["filename"], file_bytes=int(row["file_bytes"]),
                file_sha256=row["file_sha256"], width=int(row["width"]), height=int(row["height"]),
                source_mode=row["source_mode"], pixels=int(row["pixels"]), payload_bytes=int(row["payload_bytes"]),
                decoded_sha256=row["decoded_sha256"], legacy20=row["legacy20"] == "True",
                legacy_category=row["legacy_category"]))
    return out


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def select(records: list[ImageRecord], rule: str, root: Path | None = None) -> list[ImageRecord]:
    """Prespecified selection rules; none depends on algorithm results."""
    if rule == "all":
        return list(records)
    if rule == "legacy20":
        order = list(legacy_selection(root or Path.cwd()))
        by_name = {r.filename: r for r in records}
        return [by_name[name] for name in order]
    if rule.startswith("first:"):
        return list(records[: int(rule.split(":", 1)[1])])
    if rule.startswith("stratified:"):
        n = int(rule.split(":", 1)[1])
        ordered = sorted(records, key=lambda r: (r.payload_bytes, r.filename))
        strata = np.array_split(np.arange(len(ordered)), n)
        return [ordered[int(s[len(s) // 2])] for s in strata if len(s)]
    raise ConfigurationError(f"unsupported selection {rule!r}")


def make_synthetic_dataset(directory: Path, count: int = 4, shape=(96, 128), seed: int = 7) -> list[Path]:
    """Deterministic small RGB PNGs for tests and smoke runs (gradients plus seeded noise)."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    paths = []
    for i in range(count):
        h, w = shape[0] + 8 * i, shape[1] + 4 * i
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), ((xx + yy) % 256)], axis=-1)
        img = np.clip(base + rng.integers(-20, 21, base.shape), 0, 255).astype(np.uint8)
        p = directory / f"{i + 1:04d}.png"
        Image.fromarray(img, "RGB").save(p)
        paths.append(p)
    return paths
