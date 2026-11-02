"""Experiment passes. Timing passes are sequential and single-process; diagnostics run separately.

Measurement conventions:

* ``time.perf_counter_ns()`` brackets exactly one seal or one open call. Image decoding,
  hashing, round-trip comparison, diagnostics, logging, and file writes are outside timers, and
  the garbage collector is disabled inside each timed call.
* Cryptographic randomness always comes from the OS CSPRNG via the protocol API. The
  experiment seed only orders work (suite order within each image/repeat block) and
  generates synthetic payload contents.
* Every expected observation is enumerated up front with a unique key; errors are recorded as
  rows, never dropped.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import kdf, metrics, statistical_tests as st
from ._fixtures import seal_fixed
from .calibration import NEGATIVE_CONTROLS, POSITIVE_CONTROLS, StreamSpec, generate
from .dataset import ImageRecord, decode_rgb
from .errors import CryptoChaosError
from .kem import KemBackend, get_kem_backend
from .maps import legacy_chaotic_key, v2_chaos_value
from .protocol import (
    MAGIC,
    SUITES,
    TAG_BYTES,
    PasswordSession,
    Receiver,
    Sender,
    envelope_overhead,
    header_length,
    parse_envelope,
)
from .runner import LoadSampler, RunContext, utc_now

# ---------------------------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------------------------


def benchmark_password() -> str:
    """A fresh random benchmark password per process; never written to any output."""
    return base64.b64encode(os.urandom(15)).decode()  # 20 printable characters


@dataclass
class Endpoint:
    code: str
    sender: Sender
    receiver: Receiver
    backend: KemBackend | None


class Endpoints:
    """One recipient key pair and one pre-established key shared by all suites (identical lifecycle)."""

    def __init__(self, cfg: dict, suites: list[str] | None = None, scrypt_log2_n: int | None = None) -> None:
        p = cfg["protocol"]
        self.backend = get_kem_backend(p["kem_backend"])
        self.keypair = self.backend.generate_keypair()
        self.psk = os.urandom(32)
        self.password = benchmark_password()
        self.aad = p["aad"].encode()
        self.scrypt = kdf.ScryptParams(scrypt_log2_n or p["scrypt_log2_n"])
        self.suites = suites or list(p["suites"])
        self.ep = {c: self._make(c) for c in self.suites}
        self.sessions: dict[str, tuple[PasswordSession, PasswordSession]] = {}

    def _make(self, code: str) -> Endpoint:
        s = Sender(code, recipient_public_key=self.keypair.public_key, kem_backend=self.backend, psk=self.psk,
                   scrypt=self.scrypt)
        r = Receiver(recipient_secret_key=self.keypair.secret_key, kem_backend=self.backend, psk=self.psk,
                     accepted_suites={code})
        return Endpoint(code, s, r, self.backend if SUITES[code].kem else None)

    def password_for(self, code: str) -> str | None:
        return self.password if SUITES[code].password else None

    def session(self, code: str) -> tuple[PasswordSession, PasswordSession]:
        if code not in self.sessions:
            s = PasswordSession(code, self.password, self.scrypt)
            r = PasswordSession(code, self.password, self.scrypt, pw_salt=s.pw_salt)
            self.sessions[code] = (s, r)
        return self.sessions[code]


def _timed(fn: Callable[[], Any]) -> tuple[Any, int]:
    gc_on = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        out = fn()
        t1 = time.perf_counter_ns()
    finally:
        if gc_on:
            gc.enable()
    return out, t1 - t0


def measure_warm(eps: Endpoints, code: str, payload: bytes, policy: str) -> dict[str, Any]:
    ep = eps.ep[code]
    pw = eps.password_for(code)
    if policy == "session" and SUITES[code].password:
        ss, rs = eps.session(code)
        env, t_seal = _timed(lambda: ep.sender.seal(payload, aad=eps.aad, session=ss))
        out, t_open = _timed(lambda: ep.receiver.open(env, aad=eps.aad, session=rs))
    else:
        env, t_seal = _timed(lambda: ep.sender.seal(payload, password=pw, aad=eps.aad))
        out, t_open = _timed(lambda: ep.receiver.open(env, password=pw, aad=eps.aad))
    return {"seal_ns": t_seal, "open_ns": t_open, "envelope_bytes": len(env), "ok": out == payload}


def measure_cold(eps: Endpoints, code: str, payload: bytes) -> dict[str, Any]:
    """Cold provisioning: new recipient key generation + sender setup + seal; receiver key load + open."""
    pw = eps.password_for(code)
    b = eps.backend

    def seal():
        kp = b.generate_keypair()
        s = Sender(code, recipient_public_key=kp.public_key, kem_backend=b, scrypt=eps.scrypt)
        return kp, s.seal(payload, password=pw, aad=eps.aad)

    (kp, env), t_seal = _timed(seal)
    out, t_open = _timed(lambda: Receiver(recipient_secret_key=kp.secret_key, kem_backend=b,
                                          accepted_suites={code}).open(env, password=pw, aad=eps.aad))
    return {"cold_seal_ns": t_seal, "cold_open_ns": t_open, "cold_ok": out == payload}


def order_rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([seed, *parts]))


def warmup(eps: Endpoints, ctx: RunContext, n: int, nbytes: int) -> None:
    """Discarded warm-up messages; the first call per suite in this process is logged as cold-start."""
    payload = os.urandom(nbytes)
    rows = []
    for code in eps.suites:
        first = measure_warm(eps, code, payload, "per_message")
        rows.append({"suite": code, "first_seal_ns": first["seal_ns"], "first_open_ns": first["open_ns"],
                     "payload_bytes": nbytes, "wall_time_utc": utc_now()})
        for _ in range(n):
            measure_warm(eps, code, payload, "per_message")
            if SUITES[code].kem:
                measure_cold(eps, code, payload)
    path = ctx.raw / "coldstart.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("suite,first_seal_ns,first_open_ns,payload_bytes,wall_time_utc\n")
        for r in rows:
            fh.write(f"{r['suite']},{r['first_seal_ns']},{r['first_open_ns']},{r['payload_bytes']},{r['wall_time_utc']}\n")


def load_payload(ctx: RunContext, rec: ImageRecord) -> bytes:
    arr = decode_rgb(ctx.image_path(rec))
    payload = arr.tobytes()
    if ctx.cfg["dataset"]["verify"] != "none" and hashlib.sha256(payload).hexdigest() != rec.decoded_sha256:
        raise CryptoChaosError(f"{rec.filename}: decoded payload hash differs from the manifest")
    return payload


# ---------------------------------------------------------------------------------------------
# timing (main corpus)
# ---------------------------------------------------------------------------------------------

TIMING_COLUMNS = ["key", "image_id", "image_index", "repeat", "order_position", "suite", "password_policy",
                  "payload_bytes", "pixels", "envelope_bytes", "overhead_bytes", "seal_ns", "open_ns", "ok",
                  "cold_seal_ns", "cold_open_ns", "cold_ok", "error", "wall_time_utc"]


def timing_items(ctx: RunContext) -> Iterator[tuple[str, int, ImageRecord, int, int, str]]:
    cfg = ctx.cfg
    suites = list(cfg["protocol"]["suites"])
    for idx, rec in enumerate(ctx.images):
        for rep in range(1, cfg["timing"]["repeats"] + 1):
            order = list(suites)
            if cfg["timing"]["shuffle"]:
                order = [suites[i] for i in order_rng(cfg["experiment"]["seed"], 1, idx, rep).permutation(len(suites))]
            for pos, code in enumerate(order):
                yield f"{rec.image_id}|{rep}|{code}", idx, rec, rep, pos, code


def pass_timing(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    t = cfg["timing"]
    eps = Endpoints(cfg)
    warmup(eps, ctx, t["warmup_messages"], t["warmup_payload_bytes"])
    n_written = 0
    with ctx.sink("timing", TIMING_COLUMNS) as sink, \
            LoadSampler(ctx.raw / "load_timing.csv", t["load_sample_interval_s"]):
        current_idx, payload = -1, b""
        for key, idx, rec, rep, pos, code in timing_items(ctx):
            if key in sink.keys:
                continue
            if idx != current_idx:
                payload, current_idx = b"", idx
                gc.collect()
                payload = load_payload(ctx, rec)
            row = {"key": key, "image_id": rec.image_id, "image_index": idx, "repeat": rep, "order_position": pos,
                   "suite": code, "password_policy": t["password_policy"] if SUITES[code].password else "none",
                   "payload_bytes": len(payload), "pixels": rec.pixels, "wall_time_utc": utc_now()}
            try:
                row.update(measure_warm(eps, code, payload, t["password_policy"]))
                row["overhead_bytes"] = row["envelope_bytes"] - len(payload)
                if t["cold_provisioning"] and SUITES[code].kem:
                    row.update(measure_cold(eps, code, payload))
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                row["ok"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            sink.write(row)
            n_written += 1
    return {"rows_written_this_session": n_written}


# ---------------------------------------------------------------------------------------------
# diagnostics (parallel; never concurrent with timing)
# ---------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _diag_init(cfg: dict) -> None:
    _WORKER["eps"] = Endpoints(cfg)
    _WORKER["cfg"] = cfg


def payload_ciphertext(env: bytes, code: str) -> np.ndarray:
    """The AEAD ciphertext body only: framing and the 16-byte tag are excluded."""
    hl = header_length(code)
    return np.frombuffer(env, dtype=np.uint8, count=len(env) - hl - TAG_BYTES, offset=hl)


def _battery_row(arr: np.ndarray, n_bits: int) -> dict[str, Any]:
    bits = st.sample_bits(arr, n_bits)
    if bits is None:
        return st.not_applicable_record(f"payload shorter than {n_bits} bits")
    row = st.battery_record(st.run_battery(bits))
    row["sts_reason"] = ""
    return row


def _diag_task(item: tuple[str, ImageRecord, str]) -> list[dict[str, Any]]:
    image_path, rec, _ = item
    eps: Endpoints = _WORKER["eps"]
    cfg = _WORKER["cfg"]
    d = cfg["diagnostics"]
    rows = []
    try:
        arr = decode_rgb(image_path)
        payload = arr.tobytes()
        if hashlib.sha256(payload).hexdigest() != rec.decoded_sha256:
            raise CryptoChaosError("decoded payload hash differs from the manifest")
    except Exception as exc:  # noqa: BLE001
        return [{"key": f"{rec.image_id}|{c}", "image_id": rec.image_id, "suite": c,
                 "error": f"{type(exc).__name__}: {exc}"} for c in eps.suites]
    pt = np.frombuffer(payload, dtype=np.uint8)
    if d["include_plaintext"]:
        row = {"key": f"{rec.image_id}|P", "image_id": rec.image_id, "suite": "P", "ok": True,
               "payload_bytes": len(payload)}
        row.update(metrics.ciphertext_diagnostics(pt, d["lags"]))
        row.update(_battery_row(pt, d["sample_bits"]))
        rows.append(row)
    for code in eps.suites:
        row = {"key": f"{rec.image_id}|{code}", "image_id": rec.image_id, "suite": code, "payload_bytes": len(payload)}
        try:
            ep, pw = eps.ep[code], eps.password_for(code)
            env1 = ep.sender.seal(payload, password=pw, aad=eps.aad)
            env2 = ep.sender.seal(payload, password=pw, aad=eps.aad)
            row["ok"] = ep.receiver.open(env1, password=pw, aad=eps.aad) == payload
            c1, c2 = payload_ciphertext(env1, code), payload_ciphertext(env2, code)
            row.update(metrics.ciphertext_diagnostics(c1, d["lags"]))
            row.update(metrics.difference_metrics(pt, c1, "pt_ct"))
            row.update(metrics.difference_metrics(c1, c2, "reenc"))
            row.update(_battery_row(c1, d["sample_bits"]))
        except Exception as exc:  # noqa: BLE001
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def diagnostics_columns(lags: list[int]) -> list[str]:
    cols = ["key", "image_id", "suite", "payload_bytes", "ok", "n_bytes", "shannon_entropy_bits_per_byte",
            "min_entropy_bits_per_byte", "chi2_uniform", "chi2_uniform_p", "bit_ones_fraction"]
    cols += [f"lag{lag}_correlation" for lag in lags]
    for prefix in ("pt_ct", "reenc"):
        cols += [f"{prefix}_unequal_byte_pct", f"{prefix}_mean_abs_diff_pct", f"{prefix}_bit_diff_pct"]
    for name in st.TEST_NAMES:
        cols.append(f"sts_{name}_status")
        for label in (("p1", "p2") if name == "serial" else ("p",)):
            cols += [f"sts_{name}_{label}", f"sts_{name}_{label}_pass"]
    return cols + ["sts_reason", "error"]


def pass_diagnostics(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    d = cfg["diagnostics"]
    codes = list(cfg["protocol"]["suites"]) + (["P"] if d["include_plaintext"] else [])
    with ctx.sink("diagnostics", diagnostics_columns(d["lags"])) as sink:
        todo = [(str(ctx.image_path(r)), r, "") for r in ctx.images
                if any(f"{r.image_id}|{c}" not in sink.keys for c in codes)]
        with cf.ProcessPoolExecutor(max_workers=d["workers"], initializer=_diag_init, initargs=(cfg,)) as ex:
            for rows in ex.map(_diag_task, todo, chunksize=1):
                for row in rows:
                    if row["key"] not in sink.keys:
                        sink.write(row)
    return {"images": len(ctx.images)}


# ---------------------------------------------------------------------------------------------
# authentication / rejection suite
# ---------------------------------------------------------------------------------------------

AUTH_COLUMNS = ["key", "suite", "case", "trial", "expected", "outcome", "detail", "plaintext_released"]
LEGACY_COLLISION = ("cryptochaos-demo-12996", "cryptochaos-demo-49085")


def auth_cases(code: str) -> list[str]:
    s = SUITES[code]
    cases = ["valid", "wrong_aad", "header_bitflip", "payload_bitflip", "tag_bitflip", "truncation",
             "trailing_data", "bad_version", "unknown_suite"]
    cases.append("wrong_recipient_key" if s.kem else "wrong_psk")
    if s.kem:
        cases.append("kem_ct_bitflip")
    if s.password:
        cases += ["wrong_password", "legacy_collision_password", "suite_downgrade"]
    if s.uses_scrypt:
        cases.append("kdf_params_out_of_policy")
    return cases


def pass_auth(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    trials = cfg["auth"]["trials_per_class"]
    eps = Endpoints(cfg)
    other = Endpoints(cfg)
    rng = order_rng(cfg["experiment"]["seed"], 2)
    counts: dict[str, int] = {}
    with ctx.sink("auth", AUTH_COLUMNS) as sink:
        for code in eps.suites:
            ep, pw, aad = eps.ep[code], eps.password_for(code), eps.aad
            hl = header_length(code)
            for case in auth_cases(code):
                for trial in range(trials):
                    key = f"{code}|{case}|{trial}"
                    if key in sink.keys:
                        continue
                    payload = os.urandom(4096)
                    password = pw
                    receiver = ep.receiver
                    if case == "legacy_collision_password":
                        env = ep.sender.seal(payload, password=LEGACY_COLLISION[0], aad=aad)
                        password = LEGACY_COLLISION[1]
                    else:
                        env = ep.sender.seal(payload, password=pw, aad=aad)
                    bad, use_aad = bytearray(env), aad
                    if case == "wrong_aad":
                        use_aad = aad + b"!"
                    elif case == "header_bitflip":
                        i = int(rng.integers(0, hl))
                        bad[i] ^= 1 << int(rng.integers(0, 8))
                    elif case == "kem_ct_bitflip":
                        i = 10 + int(rng.integers(0, 1568))
                        bad[i] ^= 1 << int(rng.integers(0, 8))
                    elif case == "payload_bitflip":
                        bad[hl + int(rng.integers(0, len(env) - hl - TAG_BYTES))] ^= 1 << int(rng.integers(0, 8))
                    elif case == "tag_bitflip":
                        bad[len(env) - TAG_BYTES + int(rng.integers(0, TAG_BYTES))] ^= 1 << int(rng.integers(0, 8))
                    elif case == "truncation":
                        bad = bad[: int(rng.integers(0, len(env)))]
                    elif case == "trailing_data":
                        bad += os.urandom(int(rng.integers(1, 17)))
                    elif case == "bad_version":
                        bad[4] = 1 + int(rng.integers(3, 250))
                    elif case == "unknown_suite":
                        bad[5] = int(rng.choice([0x00, 0x03, 0x14, 0x7F, 0xFF]))
                    elif case == "wrong_recipient_key":
                        receiver = other.ep[code].receiver
                    elif case == "wrong_psk":
                        receiver = other.ep[code].receiver
                    elif case == "wrong_password":
                        password = pw + "x"
                    elif case == "suite_downgrade":
                        bad = bytearray(Sender("C", recipient_public_key=eps.keypair.public_key,
                                               kem_backend=eps.backend).seal(payload, aad=aad))
                    elif case == "kdf_params_out_of_policy":
                        off = header_length(code) - (4 + 1 + 16) - 1 - 12 - 8  # start of the scrypt fields
                        bad[off + 1] = int(rng.choice([0, 1, 10, 13, 19, 20, 31, 63, 255]))
                    expected = "accept" if case == "valid" else "reject"
                    if case == "legacy_collision_password" and code == "D":
                        expected = "reject (v1 accepts: documented weakness)"
                    try:
                        out = receiver.open(bytes(bad), password=password, aad=use_aad)
                        outcome, detail, released = "accepted", "", out == payload
                    except (CryptoChaosError, ValueError) as exc:
                        outcome, detail, released = "rejected", type(exc).__name__, False
                    counts[outcome] = counts.get(outcome, 0) + 1
                    sink.write({"key": key, "suite": code, "case": case, "trial": trial, "expected": expected,
                                "outcome": outcome, "detail": detail, "plaintext_released": released})
    return counts


# ---------------------------------------------------------------------------------------------
# key-sensitivity fixtures (diagnostic only; fixed inputs except the one varied)
# ---------------------------------------------------------------------------------------------

KEYSENS_COLUMNS = ["key", "source", "image_id", "suite", "variation", "n_bytes", "unequal_byte_pct",
                   "mean_abs_diff_pct", "bit_diff_pct", "differing_bits"]


def keysens_variations(code: str) -> list[str]:
    s = SUITES[code]
    out = ["hkdf_salt_one_bit", "fresh_encapsulation"] if s.kem else ["psk_one_bit"]
    if s.password:
        out += ["password_last_char", "legacy_collision_password"]
    if s.uses_scrypt:
        out.append("pw_salt_one_bit")
    return out


def _flip(b: bytes, bit: int = 0) -> bytes:
    x = bytearray(b)
    x[bit // 8] ^= 1 << (bit % 8)
    return bytes(x)


def pass_key_sensitivity(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    ks = cfg["key_sensitivity"]
    eps = Endpoints(cfg)
    sources: list[tuple[str, str, Callable[[], bytes]]] = []
    for rec in ctx.images[: ks["images"]]:
        sources.append(("image", rec.image_id, lambda rec=rec: load_payload(ctx, rec)[: 1 << 20]))
    synth = np.random.Generator(np.random.PCG64([cfg["experiment"]["seed"], 3])).bytes(ks["synthetic_bytes"])
    sources.append(("synthetic", "synthetic", lambda: synth))
    n = 0
    with ctx.sink("key_sensitivity", KEYSENS_COLUMNS) as sink:
        for source, sid, get in sources:
            payload = None
            for code in eps.suites:
                variations = keysens_variations(code) + (["plaintext_one_bit_fixed_key_nonce"]
                                                         if source == "synthetic" else [])
                for var in variations:
                    key = f"{sid}|{code}|{var}"
                    if key in sink.keys:
                        continue
                    payload = payload if payload is not None else get()
                    c1, c2 = _keysens_pair(eps, code, var, payload)
                    a, b = payload_ciphertext(c1, code), payload_ciphertext(c2, code)
                    sink.write({"key": key, "source": source, "image_id": sid, "suite": code, "variation": var,
                                "n_bytes": a.size, "unequal_byte_pct": metrics.unequal_byte_pct(a, b),
                                "mean_abs_diff_pct": metrics.mean_abs_diff_pct(a, b),
                                "bit_diff_pct": metrics.bit_diff_pct(a, b),
                                "differing_bits": int(np.bitwise_count(a ^ b).sum(dtype=np.int64))})
                    n += 1
    return {"rows": n}


def _keysens_pair(eps: Endpoints, code: str, var: str, payload: bytes) -> tuple[bytes, bytes]:
    s = SUITES[code]
    nonce, hkdf_salt, pw_salt = os.urandom(12), os.urandom(32), os.urandom(16)
    pw = eps.password_for(code)
    enc = eps.backend.encapsulate(eps.backend.load_public_key(eps.keypair.public_key)) if s.kem else None
    sender = eps.ep[code].sender

    def seal(p=payload, password=pw, hs=hkdf_salt, ps=pw_salt, e=enc, snd=sender):
        return seal_fixed(snd, p, nonce=nonce, hkdf_salt=hs if s.kem else None, pw_salt=ps if s.uses_scrypt else None,
                          encapsulation=e, password=password, aad=eps.aad)

    base = seal()
    if var == "hkdf_salt_one_bit":
        return base, seal(hs=_flip(hkdf_salt))
    if var == "fresh_encapsulation":
        return base, seal(e=eps.backend.encapsulate(eps.backend.load_public_key(eps.keypair.public_key)))
    if var == "psk_one_bit":
        other = Sender(code, psk=_flip(eps.psk))
        return base, seal(snd=other)
    if var == "password_last_char":
        return base, seal(password=pw[:-1] + ("A" if pw[-1] != "A" else "B"))
    if var == "legacy_collision_password":
        return seal(password=LEGACY_COLLISION[0]), seal(password=LEGACY_COLLISION[1])
    if var == "pw_salt_one_bit":
        return base, seal(ps=_flip(pw_salt))
    if var == "plaintext_one_bit_fixed_key_nonce":
        # Deliberate (key, nonce) reuse on synthetic data only: shows that AEAD payload encryption is a
        # keystream XOR, so a one-bit plaintext change flips exactly one ciphertext bit.
        return base, seal(p=_flip(payload, 8 * (len(payload) // 2)))
    raise ValueError(var)


# ---------------------------------------------------------------------------------------------
# size scaling and KDF sensitivity (synthetic payloads)
# ---------------------------------------------------------------------------------------------

SCALING_COLUMNS = ["key", "payload_bytes", "repeat", "order_position", "suite", "workload", "password_policy",
                   "envelope_bytes", "seal_ns", "open_ns", "ok", "error", "wall_time_utc"]


def synthetic_payload(seed: int, size: int) -> bytes:
    return np.random.Generator(np.random.PCG64([seed, 4, size])).bytes(size)


def scaling_items(cfg: dict) -> Iterator[tuple[str, int, int, int, str, str, str]]:
    sc = cfg["scaling"]
    suites = list(cfg["protocol"]["suites"])
    for size, reps in zip(sc["sizes"], sc["repeats"]):
        for rep in range(reps):
            order = [suites[i] for i in order_rng(cfg["experiment"]["seed"], 5, size, rep).permutation(len(suites))]
            for pos, code in enumerate(order):
                s = SUITES[code]
                yield f"{size}|{rep}|{code}|warm|per_message", size, rep, pos, code, "warm", "per_message"
                if s.password and "session" in sc["policies"]:
                    yield f"{size}|{rep}|{code}|warm|session", size, rep, pos, code, "warm", "session"
                if s.kem and rep < sc["cold_repeats_max"]:
                    yield f"{size}|{rep}|{code}|cold|per_message", size, rep, pos, code, "cold", "per_message"


def pass_scaling(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    eps = Endpoints(cfg)
    warmup(eps, ctx, cfg["timing"]["warmup_messages"], 1 << 16)
    payloads: dict[int, bytes] = {}
    n = 0
    with ctx.sink("scaling", SCALING_COLUMNS) as sink, \
            LoadSampler(ctx.raw / "load_scaling.csv", cfg["timing"]["load_sample_interval_s"]):
        for key, size, rep, pos, code, workload, policy in scaling_items(cfg):
            if key in sink.keys:
                continue
            if size not in payloads:
                payloads = {size: synthetic_payload(cfg["experiment"]["seed"], size)}
            p = payloads[size]
            row = {"key": key, "payload_bytes": size, "repeat": rep, "order_position": pos, "suite": code,
                   "workload": workload, "password_policy": policy if SUITES[code].password else "none",
                   "wall_time_utc": utc_now()}
            try:
                if workload == "warm":
                    r = measure_warm(eps, code, p, policy)
                else:
                    r = measure_cold(eps, code, p)
                    r = {"seal_ns": r["cold_seal_ns"], "open_ns": r["cold_open_ns"], "ok": r["cold_ok"],
                         "envelope_bytes": size + envelope_overhead(code)}
                row.update(r)
            except Exception as exc:  # noqa: BLE001
                row["ok"], row["error"] = False, f"{type(exc).__name__}: {exc}"
            sink.write(row)
            n += 1
    return {"rows_written_this_session": n}


KDF_COLUMNS = ["key", "log2_n", "repeat", "order_position", "suite", "payload_bytes", "seal_ns", "open_ns", "ok",
               "error", "wall_time_utc"]


def pass_kdf_sensitivity(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    k = cfg["kdf_sensitivity"]
    codes = [c for c in cfg["protocol"]["suites"] if SUITES[c].uses_scrypt]
    payload = synthetic_payload(cfg["experiment"]["seed"], k["payload_bytes"])
    n = 0
    with ctx.sink("kdf_sensitivity", KDF_COLUMNS) as sink:
        for log2_n in k["log2_n"]:
            eps = Endpoints(cfg, suites=codes, scrypt_log2_n=log2_n)
            for code in codes:
                measure_warm(eps, code, payload, "per_message")  # warm-up
            for rep in range(k["repeats"]):
                order = [codes[i] for i in order_rng(cfg["experiment"]["seed"], 6, log2_n, rep).permutation(len(codes))]
                for pos, code in enumerate(order):
                    key = f"{log2_n}|{rep}|{code}"
                    if key in sink.keys:
                        continue
                    row = {"key": key, "log2_n": log2_n, "repeat": rep, "order_position": pos, "suite": code,
                           "payload_bytes": len(payload), "wall_time_utc": utc_now()}
                    try:
                        row.update({k2: v for k2, v in measure_warm(eps, code, payload, "per_message").items()
                                    if k2 != "envelope_bytes"})
                    except Exception as exc:  # noqa: BLE001
                        row["ok"], row["error"] = False, f"{type(exc).__name__}: {exc}"
                    sink.write(row)
                    n += 1
    return {"rows_written_this_session": n}


# ---------------------------------------------------------------------------------------------
# stage microbenchmarks (separate from end-to-end timing; never summed to replace it)
# ---------------------------------------------------------------------------------------------

STAGE_COLUMNS = ["key", "stage", "variant", "repeat", "ns", "payload_bytes"]


def pass_stages(ctx: RunContext) -> dict[str, Any]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

    cfg = ctx.cfg
    sg = cfg["stages"]
    R = sg["repeats"]
    payload = synthetic_payload(cfg["experiment"]["seed"], sg["payload_bytes"])
    password = benchmark_password()
    pw_bytes = kdf.encode_password(password)
    benches: list[tuple[str, str, int, Callable[[], Any], int]] = []
    for name in sg["kem_backends"]:
        b = get_kem_backend(name)
        reps = R if name != "kyber-py" else sg["reference_repeats"]
        kp = b.generate_keypair()
        pub, sec = b.load_public_key(kp.public_key), b.load_secret_key(kp.secret_key)
        ct = b.encapsulate(pub).ciphertext
        benches += [("kem_keygen", name, reps, b.generate_keypair, 0),
                    ("kem_load_public_key", name, reps, lambda b=b, kp=kp: b.load_public_key(kp.public_key), 0),
                    ("kem_encapsulate", name, reps, lambda b=b, pub=pub: b.encapsulate(pub), 0),
                    ("kem_load_secret_key", name, reps, lambda b=b, kp=kp: b.load_secret_key(kp.secret_key), 0),
                    ("kem_decapsulate", name, reps, lambda b=b, sec=sec, ct=ct: b.decapsulate(sec, ct), 0)]
    ss, pw_key, salt32, salt16 = os.urandom(32), os.urandom(32), os.urandom(32), os.urandom(16)
    benches += [
        ("blake3_shared_secret", "blake3", R, lambda: kdf.kem_input(ss), 0),
        ("hkdf_aead_key", "2-fields", R, lambda: kdf.derive_aead_key(2, 0x12, [(1, ss), (2, pw_key)], salt32), 0),
        ("hkdf_aead_key", "3-fields", R,
         lambda: kdf.derive_aead_key(2, 0x13, [(1, ss), (2, pw_key), (3, pw_key)], salt32), 0),
        ("legacy_chaotic_key", "v1-maps", R, lambda: legacy_chaotic_key(password), 0),
        ("v2_chaos_value", "v2-maps", R, lambda: v2_chaos_value(pw_key), 0),
        ("password_encode", "nfc-utf8", R, lambda: kdf.encode_password(password), 0),
    ]
    for log2_n in sg["scrypt_log2_n"]:
        params = kdf.ScryptParams(log2_n)
        benches.append(("scrypt", f"log2N={log2_n}", sg["scrypt_repeats"],
                        lambda params=params: kdf.derive_password_key(pw_bytes, salt16, params), 0))
    key = os.urandom(32)
    aes, cha = AESGCM(key), ChaCha20Poly1305(key)
    nonce = os.urandom(12)
    ct_aes, ct_cha = aes.encrypt(nonce, payload, b"a"), cha.encrypt(nonce, payload, b"a")
    benches += [
        ("aead_seal", "AES-256-GCM", R, lambda: aes.encrypt(nonce, payload, b"a"), len(payload)),
        ("aead_open", "AES-256-GCM", R, lambda: aes.decrypt(nonce, ct_aes, b"a"), len(payload)),
        ("aead_seal", "ChaCha20-Poly1305", R, lambda: cha.encrypt(nonce, payload, b"a"), len(payload)),
        ("aead_open", "ChaCha20-Poly1305", R, lambda: cha.decrypt(nonce, ct_cha, b"a"), len(payload)),
    ]
    eps = Endpoints(cfg)
    for code in eps.suites:
        env = eps.ep[code].sender.seal(payload, password=eps.password_for(code), aad=eps.aad)
        benches.append(("envelope_parse", code, R,
                        lambda env=env, code=code: parse_envelope(env, frozenset({code})), len(payload)))
    n = 0
    with ctx.sink("stages", STAGE_COLUMNS) as sink:
        for stage, variant, reps, fn, nbytes in benches:
            fn()  # warm-up
            for rep in range(reps):
                key = f"{stage}|{variant}|{rep}"
                if key in sink.keys:
                    continue
                _, ns = _timed(fn)
                sink.write({"key": key, "stage": stage, "variant": variant, "repeat": rep, "ns": ns,
                            "payload_bytes": nbytes})
                n += 1
    return {"rows_written_this_session": n}


# ---------------------------------------------------------------------------------------------
# memory (isolated subprocesses)
# ---------------------------------------------------------------------------------------------

MEMORY_COLUMNS = ["key", "suite", "payload_label", "payload_bytes", "repeat", "mode", "baseline_rss_bytes",
                  "baseline_peak_bytes", "peak_bytes", "peak_minus_baseline_rss_bytes", "ops_raised_peak",
                  "tracemalloc_peak_bytes", "ok", "platform_metric", "error"]


def pass_memory(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    m = cfg["memory"]
    targets: list[tuple[str, int, str | None]] = [(f"synthetic-{b}", b, None) for b in m["payload_bytes"]]
    if m["include_median_image"] and ctx.images:
        med = sorted(ctx.images, key=lambda r: (r.payload_bytes, r.image_id))[len(ctx.images) // 2]
        raw_path = ctx.dir / "tmp" / f"{med.image_id}.rgb"  # pre-decoded so decoding never sets the baseline peak
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_bytes(load_payload(ctx, med))
        targets.append((f"image-{med.image_id}", med.payload_bytes, str(raw_path)))
    n = 0
    with ctx.sink("memory", MEMORY_COLUMNS) as sink:
        for label, nbytes, image in targets:
            for code in cfg["protocol"]["suites"]:
                for rep in range(m["repeats"]):
                    for mode in ("rss", "tracemalloc"):
                        key = f"{label}|{code}|{rep}|{mode}"
                        if key in sink.keys:
                            continue
                        cmd = [sys.executable, "-m", "cryptochaos.memory_probe", "--suite", code, "--payload-bytes",
                               str(nbytes), "--operations", str(m["operations"]), "--scrypt-log2n",
                               str(cfg["protocol"]["scrypt_log2_n"]), "--kem-backend", cfg["protocol"]["kem_backend"],
                               "--seed", str(cfg["experiment"]["seed"])]
                        if image:
                            cmd += ["--image", image]
                        if mode == "tracemalloc":
                            cmd.append("--tracemalloc")
                        row = {"key": key, "suite": code, "payload_label": label, "payload_bytes": nbytes,
                               "repeat": rep, "mode": mode}
                        try:
                            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
                            row.update(json.loads(out.stdout.strip().splitlines()[-1]))
                        except Exception as exc:  # noqa: BLE001
                            row["ok"], row["error"] = False, f"{type(exc).__name__}: {exc}"
                        sink.write(row)
                        n += 1
    for _, _, path in targets:
        if path and path.endswith(".rgb"):
            Path(path).unlink(missing_ok=True)
    return {"rows_written_this_session": n}


# ---------------------------------------------------------------------------------------------
# randomness-test calibration (parallel)
# ---------------------------------------------------------------------------------------------


def calibration_specs(cfg: dict) -> list[StreamSpec]:
    c = cfg["calibration"]
    specs = [StreamSpec("chacha20", i, "uniform", POSITIVE_CONTROLS["chacha20"]) for i in range(c["uniform_streams"])]
    specs += [StreamSpec("mt19937", i, "uniform", POSITIVE_CONTROLS["mt19937"])
              for i in range(c["positive_control_streams"])]
    for kind, desc in NEGATIVE_CONTROLS.items():
        specs += [StreamSpec(kind, i, "reject", desc) for i in range(c["negative_control_streams"])]
    return specs


def _calib_task(args: tuple[StreamSpec, int]) -> dict[str, Any]:
    spec, n_bits = args
    bits = generate(spec, n_bits)
    row = {"key": f"{spec.kind}|{spec.index}", "kind": spec.kind, "index": spec.index, "expected": spec.expected,
           "bits_sha256": hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()}
    row.update(st.battery_record(st.run_battery(bits)))
    return row


def calibration_columns() -> list[str]:
    cols = ["key", "kind", "index", "expected", "bits_sha256"]
    for name in st.TEST_NAMES:
        cols.append(f"sts_{name}_status")
        for label in (("p1", "p2") if name == "serial" else ("p",)):
            cols += [f"sts_{name}_{label}", f"sts_{name}_{label}_pass"]
    return cols


def pass_calibration(ctx: RunContext) -> dict[str, Any]:
    c = ctx.cfg["calibration"]
    specs = calibration_specs(ctx.cfg)
    with ctx.sink("calibration", calibration_columns()) as sink:
        todo = [(s, c["n_bits"]) for s in specs if f"{s.kind}|{s.index}" not in sink.keys]
        with cf.ProcessPoolExecutor(max_workers=c["workers"]) as ex:
            for row in ex.map(_calib_task, todo, chunksize=4):
                sink.write(row)
    return {"streams": len(specs)}


# ---------------------------------------------------------------------------------------------
# historical reconstruction: v1 code paths re-timed under v2 conventions (bridge run only)
# ---------------------------------------------------------------------------------------------

LEGACY_COLUMNS = ["key", "image_id", "repeat", "order_position", "wrapper", "payload_bytes", "encrypt_ns",
                  "decrypt_ns", "ok", "error", "wall_time_utc"]


def _legacy_wrappers(root: Path) -> dict[str, tuple[Callable[[bytes], Any], Callable[[Any], bytes]]]:
    """Exact v1 behaviour: the unmodified v1 CryptoChaos class plus faithful copies of two v1 wrappers."""
    import importlib.util

    import blake3
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from kyber_py.ml_kem import ML_KEM_1024

    spec = importlib.util.spec_from_file_location("legacy_v1", root / "legacy" / "src" / "cryptochaos_benchmark.py")
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    cc = v1.CryptoChaosOptimized("benchmark_passphrase_2025")
    ek, dk = ML_KEM_1024.keygen()  # v1 W_MLKEM: key pair generated once, reused
    aes_key = os.urandom(32)

    def pure_enc(pt: bytes):
        k, c = ML_KEM_1024.encaps(ek)
        n = os.urandom(12)
        return AESGCM(blake3.blake3(k).digest(length=32)).encrypt(n, pt, None), (n, c)

    def pure_dec(state):
        ct, (n, c) = state
        return AESGCM(blake3.blake3(ML_KEM_1024.decaps(dk, c)).digest(length=32)).decrypt(n, ct, None)

    def aes_enc(pt: bytes):
        n = os.urandom(12)
        return AESGCM(aes_key).encrypt(n, pt, None), n

    def aes_dec(state):
        ct, n = state
        return AESGCM(aes_key).decrypt(n, ct, None)

    return {
        "v1 CryptoChaos-Kyber1024 (keygen per message, kyber-py)": (cc.encrypt, lambda s: cc.decrypt(*s)),
        "v1 Kyber1024-Pure (reused keys, kyber-py)": (pure_enc, pure_dec),
        "v1 AES-256-GCM": (aes_enc, aes_dec),
    }


def pass_legacy_reconstruction(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.cfg
    wrappers = _legacy_wrappers(ctx.root)
    names = list(wrappers)
    n = 0
    with ctx.sink("legacy_reconstruction", LEGACY_COLUMNS) as sink:
        for idx, rec in enumerate(ctx.images):
            payload = None
            for rep in range(1, cfg["timing"]["repeats"] + 1):
                order = [names[i] for i in order_rng(cfg["experiment"]["seed"], 8, idx, rep).permutation(len(names))]
                for pos, name in enumerate(order):
                    key = f"{rec.image_id}|{rep}|{name}"
                    if key in sink.keys:
                        continue
                    payload = payload if payload is not None else load_payload(ctx, rec)
                    enc, dec = wrappers[name]
                    row = {"key": key, "image_id": rec.image_id, "repeat": rep, "order_position": pos, "wrapper": name,
                           "payload_bytes": len(payload), "wall_time_utc": utc_now()}
                    try:
                        state, t_enc = _timed(lambda: enc(payload))
                        out, t_dec = _timed(lambda: dec(state))
                        row.update({"encrypt_ns": t_enc, "decrypt_ns": t_dec, "ok": out == payload})
                    except Exception as exc:  # noqa: BLE001
                        row["ok"], row["error"] = False, f"{type(exc).__name__}: {exc}"
                    sink.write(row)
                    n += 1
    return {"rows_written_this_session": n}


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------


def pass_environment(ctx: RunContext) -> dict[str, Any]:
    from . import environment

    env = environment.capture()
    (ctx.dir / "environment.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")
    return {"cpu": env["cpu"]}


def pass_map_analysis(ctx: RunContext) -> dict[str, Any]:
    from . import map_analysis

    return map_analysis.run(ctx)


PASSES: dict[str, Callable[[RunContext], dict[str, Any]]] = {
    "environment": pass_environment, "map_analysis": pass_map_analysis, "auth": pass_auth,
    "key_sensitivity": pass_key_sensitivity, "stages": pass_stages, "timing": pass_timing,
    "diagnostics": pass_diagnostics, "scaling": pass_scaling, "kdf_sensitivity": pass_kdf_sensitivity,
    "memory": pass_memory, "calibration": pass_calibration, "legacy_reconstruction": pass_legacy_reconstruction,
}


def expected_keys(ctx: RunContext, name: str) -> set[str] | None:
    """Every observation a complete pass must contain (used by verify-artifact)."""
    cfg = ctx.cfg
    suites = list(cfg["protocol"]["suites"])
    if name == "timing":
        return {k for k, *_ in timing_items(ctx)}
    if name == "diagnostics":
        codes = suites + (["P"] if cfg["diagnostics"]["include_plaintext"] else [])
        return {f"{r.image_id}|{c}" for r in ctx.images for c in codes}
    if name == "auth":
        return {f"{c}|{case}|{t}" for c in suites for case in auth_cases(c)
                for t in range(cfg["auth"]["trials_per_class"])}
    if name == "scaling":
        return {k for k, *_ in scaling_items(cfg)}
    if name == "kdf_sensitivity":
        codes = [c for c in suites if SUITES[c].uses_scrypt]
        return {f"{n}|{r}|{c}" for n in cfg["kdf_sensitivity"]["log2_n"] for r in range(cfg["kdf_sensitivity"]["repeats"])
                for c in codes}
    if name == "calibration":
        return {f"{s.kind}|{s.index}" for s in calibration_specs(cfg)}
    if name == "legacy_reconstruction":
        names = ["v1 CryptoChaos-Kyber1024 (keygen per message, kyber-py)", "v1 Kyber1024-Pure (reused keys, kyber-py)",
                 "v1 AES-256-GCM"]
        return {f"{r.image_id}|{rep}|{w}" for r in ctx.images for rep in range(1, cfg["timing"]["repeats"] + 1)
                for w in names}
    if name == "key_sensitivity":
        ids = [r.image_id for r in ctx.images[: cfg["key_sensitivity"]["images"]]]
        keys = {f"{i}|{c}|{v}" for i in ids for c in suites for v in keysens_variations(c)}
        keys |= {f"synthetic|{c}|{v}" for c in suites
                 for v in keysens_variations(c) + ["plaintext_one_bit_fixed_key_nonce"]}
        return keys
    return None


def run(config_path: str | Path, resume: bool = False, passes: list[str] | None = None) -> Path:
    from . import environment
    from .config import load_config

    cfg = load_config(config_path)
    ctx = RunContext(cfg, resume)
    exp = cfg["experiment"]
    selected = passes or exp["passes"]
    if exp["prevent_sleep"]:
        environment.prevent_sleep(True)
    ctx.meta["process_priority"] = environment.set_priority(exp["process_priority"])
    ctx.save_meta()
    try:
        for name in selected:
            if ctx.is_done(name):
                print(f"[skip] {name} (complete)", flush=True)
                continue
            print(f"[pass] {name} started {utc_now()}", flush=True)
            t0 = time.time()
            summary = PASSES[name](ctx)
            exp_keys = expected_keys(ctx, name)
            if exp_keys is not None:
                from .runner import iter_csv

                got = {r["key"] for r in iter_csv(ctx.raw / f"{name}.csv")}
                summary["expected_rows"], summary["rows"] = len(exp_keys), len(got)
                summary["missing_rows"] = len(exp_keys - got)
            summary["elapsed_s"] = round(time.time() - t0, 1)
            ctx.mark_done(name, summary)
            print(f"[pass] {name} done in {summary['elapsed_s']} s: {summary}", flush=True)
        ctx.finalize()
    finally:
        if exp["prevent_sleep"]:
            environment.prevent_sleep(False)
    return ctx.dir


__all__ = ["run", "PASSES", "Endpoints", "expected_keys", "MAGIC"]
