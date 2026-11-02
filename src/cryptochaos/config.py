"""Versioned experiment configuration (TOML) with validation and a stable identity hash."""
from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .protocol import SUITES

ALL_PASSES = ("environment", "map_analysis", "auth", "key_sensitivity", "stages", "timing", "diagnostics",
              "scaling", "kdf_sensitivity", "memory", "calibration")
DEFAULT_PASSES = list(ALL_PASSES)

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "unnamed", "description": "", "seed": 20261101, "output_root": "runs",
                   "passes": list(DEFAULT_PASSES), "process_priority": "normal", "prevent_sleep": True},
    "dataset": {"source": "div2k", "manifest": "configs/manifests/div2k_train_hr.csv",
                "path_env": "CRYPTOCHAOS_DIV2K_PATH", "path": "", "selection": "all", "verify": "decoded_hash"},
    "protocol": {"suites": list("ABCDEF"), "kem_backend": "openssl", "scrypt_log2_n": 14,
                 "aad": "CryptoChaos-v2 benchmark"},
    "timing": {"repeats": 5, "warmup_messages": 8, "warmup_payload_bytes": 1 << 20,
               "password_policy": "per_message", "cold_provisioning": True, "shuffle": True,
               "load_sample_interval_s": 1.0},
    "diagnostics": {"workers": 8, "sample_bits": 1_000_000, "lags": [1, 2, 3], "include_plaintext": True},
    "auth": {"trials_per_class": 20},
    "key_sensitivity": {"images": 20, "synthetic_bytes": 1 << 20},
    "stages": {"repeats": 200, "kem_backends": ["openssl", "pqcrypto", "kyber-py"], "reference_repeats": 20,
               "payload_bytes": 1 << 20, "scrypt_log2_n": [14, 16, 18], "scrypt_repeats": 10},
    "scaling": {"sizes": [1024, 16384, 65536, 262144, 1 << 20, 4 << 20, 16 << 20],
                "repeats": [400, 300, 200, 100, 60, 30, 15], "policies": ["per_message", "session"],
                "cold_repeats_max": 50},
    "kdf_sensitivity": {"log2_n": [14, 16, 18], "repeats": 20, "payload_bytes": 1 << 20},
    "memory": {"payload_bytes": [1 << 20, 16 << 20], "include_median_image": True, "repeats": 3,
               "operations": 3},
    "calibration": {"uniform_streams": 1000, "positive_control_streams": 100, "negative_control_streams": 50,
                    "workers": 8, "n_bits": 1_000_000},
    "map_analysis": {"v2_reduced_bits_1d": [8, 12, 16, 20], "v2_reduced_bits_henon": [8, 10],
                     "v2_samples": 100, "v2_cycle_search_steps": 65536},
}


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in out:
            raise ConfigurationError(f"unknown configuration key {path}{k}")
        if isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise ConfigurationError(f"{path}{k} must be a table")
            out[k] = _merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def validate(cfg: dict) -> None:
    for code in cfg["protocol"]["suites"]:
        if code not in SUITES:
            raise ConfigurationError(f"unknown suite {code}")
    for p in cfg["experiment"]["passes"]:
        if p not in ALL_PASSES:
            raise ConfigurationError(f"unknown pass {p}")
    if cfg["timing"]["password_policy"] not in ("per_message", "session"):
        raise ConfigurationError("timing.password_policy must be per_message or session")
    s = cfg["scaling"]
    if len(s["sizes"]) != len(s["repeats"]):
        raise ConfigurationError("scaling.sizes and scaling.repeats must have equal length")
    if cfg["timing"]["repeats"] < 1:
        raise ConfigurationError("timing.repeats must be >= 1")
    sel = cfg["dataset"]["selection"]
    if not (sel in ("all", "synthetic") or sel.startswith(("first:", "stratified:"))):
        raise ConfigurationError(f"unsupported dataset.selection {sel!r}")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg = _merge(DEFAULTS, raw)
    validate(cfg)
    cfg["_source"] = {"path": str(path).replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of the effective configuration (defaults merged), excluding provenance metadata."""
    body = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
