"""Supplementary run: process cold start, header serialization, and field-stratified header tampering.

    python scripts/supplementary_run.py --out results/supplement-new

Three measurements that complement the principal run:

* cold start: one fresh Python process per configuration and repeat times the package import, key
  setup, and the first and second seal and open of a 1 MiB payload, with garbage collection disabled
  after the import;
* header serialization: batched timing of ``Header.encode`` and ``parse_envelope`` (the principal run
  timed parsing only, as one stage), every sample covering all configurations and both operations in
  a seeded order, with garbage collection disabled;
* tampering: single-bit flips within every header field (not uniformly over header bytes): every bit
  of a field of at most ``--tamper-trials`` bits, otherwise that many distinct random bits, with the
  flipped byte and bit recorded and every outcome classified.

Outputs raw CSVs, the CPU load sampled once per second during the timed phases, a report (summary,
LaTeX table and macros), the environment, the code identity, and SHA256SUMS. Everything is written
after the timed phases so that file synchronization or scanning of the output directory cannot
overlap a measurement. Keys, salts, and nonces come from the operating system; the seed only orders
work and chooses flip positions.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cryptochaos import Receiver, Sender, environment, generate_recipient_keypair, kdf
from cryptochaos.protocol import CT_BYTES, NONCE_BYTES, get_suite, header_length, parse_envelope
from cryptochaos.reporting import Macros, fmt, tex_table
from cryptochaos.runner import LoadSampler, write_hashes

SUITES = list("ABCDEF")
PASSWORD = "supplementary-run passphrase"

CHILD = r"""
import gc, json, os, sys, time
t0 = time.perf_counter_ns()
from cryptochaos import Receiver, Sender, generate_recipient_keypair
t_import = time.perf_counter_ns() - t0
gc.disable()
code, nbytes, password = sys.argv[1], int(sys.argv[2]), sys.argv[3]
password = password if code in "DEF" else None
payload = os.urandom(nbytes)
t1 = time.perf_counter_ns()
if code in "AB":
    psk = os.urandom(32)
    s, r = Sender(code, psk=psk), Receiver(psk=psk, accepted_suites={code})
else:
    kp = generate_recipient_keypair()
    s = Sender(code, recipient_public_key=kp.public_key)
    r = Receiver(recipient_secret_key=kp.secret_key, accepted_suites={code})
t_setup = time.perf_counter_ns() - t1
out = {"import_ns": t_import, "setup_ns": t_setup}
for k in (1, 2):
    t = time.perf_counter_ns(); env = s.seal(payload, password=password); out[f"seal{k}_ns"] = time.perf_counter_ns() - t
    t = time.perf_counter_ns(); pt = r.open(env, password=password); out[f"open{k}_ns"] = time.perf_counter_ns() - t
    out[f"ok{k}"] = pt == payload
print(json.dumps(out))
"""


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def endpoints(code: str):
    """Sender, receiver, and password for one configuration (fresh keys from the OS)."""
    if code in "AB":
        psk = os.urandom(32)
        return Sender(code, psk=psk), Receiver(psk=psk, accepted_suites={code}), None
    kp = generate_recipient_keypair()
    s = Sender(code, recipient_public_key=kp.public_key)
    r = Receiver(recipient_secret_key=kp.secret_key, accepted_suites={code})
    return s, r, (PASSWORD if code in "DEF" else None)


def header_fields(code: str) -> list[tuple[str, int, int]]:
    """Byte spans of the header fields, in the order of ``Header.encode``."""
    s, spans, i = get_suite(code), [], 0

    def add(name: str, n: int) -> None:
        nonlocal i
        spans.append((name, i, i + n))
        i += n

    add("format identifier", 4)
    add("version", 1)
    add("configuration", 1)
    add("flags and reserved", 2)
    if s.kem:
        add("KEM ciphertext length", 2)
        add("KEM ciphertext", CT_BYTES)
        add("HKDF salt length", 1)
        add("HKDF salt", kdf.HKDF_SALT_BYTES)
        add("commitment length", 1)
        add("key commitment", kdf.COMMITMENT_BYTES)
    if s.uses_scrypt:
        add("scrypt identifier and parameters", 4)
        add("password salt length", 1)
        add("password salt", kdf.PW_SALT_BYTES)
    add("nonce length", 1)
    add("nonce", NONCE_BYTES)
    add("ciphertext length", 8)
    assert i == header_length(code), (code, i, header_length(code))
    return spans


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def cold_start(repeats: int, nbytes: int, seed: int) -> list[dict]:
    rows = []
    for rep in range(repeats):
        order = np.random.default_rng([seed, 1, rep]).permutation(SUITES)
        for pos, code in enumerate(order):
            t = time.perf_counter_ns()
            res = subprocess.run([sys.executable, "-c", CHILD, code, str(nbytes), PASSWORD],
                                 capture_output=True, text=True, check=False)
            wall = time.perf_counter_ns() - t
            row = {"suite": code, "repeat": rep, "order_position": pos, "payload_bytes": nbytes, "process_ns": wall}
            try:
                row.update(json.loads(res.stdout.strip().splitlines()[-1]))
                row["error"] = ""
            except (IndexError, ValueError):
                row["error"] = (res.stderr.strip().splitlines() or ["no output"])[-1][:200]
            rows.append(row)
    return rows


def serialization(nbytes: int, samples: int, batch: int, seed: int) -> list[dict]:
    payload = os.urandom(nbytes)
    calls = {}
    for code in SUITES:
        s, _, pw = endpoints(code)
        env = bytes(s.seal(payload, password=pw))
        header, _, _ = parse_envelope(env, frozenset({code}))
        calls[(code, "encode")] = header.encode
        calls[(code, "parse")] = lambda env=env, code=code: parse_envelope(env, frozenset({code}))
    keys = list(calls)
    for fn in calls.values():
        for _ in range(10):
            fn()
    # each sample times every configuration and operation in a fresh order, so a burst of background
    # activity is spread over all of them instead of falling on one configuration's block
    orders = [np.random.default_rng([seed, 3, k]).permutation(len(keys)) for k in range(samples)]
    rows = []
    gc.collect()
    gc.disable()
    try:
        for k, order in enumerate(orders):
            for pos, i in enumerate(order):
                code, op = keys[i]
                fn = calls[keys[i]]
                t = time.perf_counter_ns()
                for _ in range(batch):
                    fn()
                rows.append({"suite": code, "operation": op, "sample": k, "order_position": pos, "batch": batch,
                             "ns_per_call": (time.perf_counter_ns() - t) / batch})
    finally:
        gc.enable()
    return rows


def tampering(trials: int, seed: int) -> list[dict]:
    rows = []
    payload = os.urandom(4096)
    for si, code in enumerate(SUITES):
        s, r, pw = endpoints(code)
        env = bytes(s.seal(payload, password=pw))
        for fi, (field, lo, hi) in enumerate(header_fields(code)):
            nbits = 8 * (hi - lo)
            if nbits <= trials:  # every bit of a short field
                picks = range(nbits)
            else:  # distinct bits, drawn without replacement
                picks = np.random.default_rng([seed, 2, si, fi]).choice(nbits, size=trials, replace=False)
            for t, b in enumerate(picks):
                pos, bit = lo + int(b) // 8, int(b) % 8
                bad = bytearray(env)
                bad[pos] ^= 1 << bit
                try:
                    out = r.open(bytes(bad), password=pw)
                    outcome, released, matches = "accepted", True, out == payload
                except Exception as exc:  # every rejection is recorded by its class
                    outcome, released, matches = type(exc).__name__, False, False
                rows.append({"suite": code, "field": field, "field_bits": nbits, "trial": t, "byte": pos, "bit": bit,
                             "outcome": outcome, "plaintext_released": released, "output_matches": matches})
    return rows


def report(out: Path, cold: list[dict], ser: list[dict], tam: list[dict], trials: int, load: list[dict]) -> dict:
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    macros = Macros("ccSupplement")
    summary: dict = {"cold_start": {}, "serialization": {}, "tampering": {}}
    table = []
    for code in SUITES:
        c = [r for r in cold if r["suite"] == code and not r["error"] and r["ok1"] and r["ok2"]]
        med = {k: float(np.median([r[k] for r in c])) / 1e6 for k in
               ("import_ns", "setup_ns", "seal1_ns", "seal2_ns", "open1_ns", "open2_ns", "process_ns")}
        summary["cold_start"][code] = {"n": len(c), **{k.replace("_ns", "_ms"): v for k, v in med.items()},
                                       "first_seal_extra_ms": float(np.median([r["seal1_ns"] - r["seal2_ns"]
                                                                               for r in c])) / 1e6}
        sv = {op: float(np.median([r["ns_per_call"] for r in ser if r["suite"] == code and r["operation"] == op])) / 1e3
              for op in ("encode", "parse")}
        summary["serialization"][code] = {f"{op}_us": v for op, v in sv.items()}
        t = [r for r in tam if r["suite"] == code]
        rejected = sum(1 for r in t if not r["plaintext_released"])
        summary["tampering"][code] = {"trials": len(t), "rejected": rejected,
                                      "fields": len({r["field"] for r in t}),
                                      "outcomes": sorted({r["outcome"] for r in t})}
        table.append([code, fmt(med["seal1_ns"], 2), fmt(med["seal2_ns"], 2), fmt(med["open1_ns"], 2),
                      fmt(med["open2_ns"], 2), f"{fmt(sv['encode'], 1)}/{fmt(sv['parse'], 1)}",
                      f"{rejected}/{len(t)}"])
        for k in ("seal1", "seal2", "open1", "open2", "setup", "import", "process"):
            macros.add(f"Cold{k.capitalize()}{code}Ms", fmt(med[f"{k}_ns"], 2))
        macros.add(f"Encode{code}Us", fmt(sv["encode"], 1))
        macros.add(f"Parse{code}Us", fmt(sv["parse"], 1))
    tex_table(rep / "tab_supplement.tex",
              ["Config.", "Seal 1", "Seal 2", "Open 1", "Open 2", "Header ($\\mu$s)", "Rejected"], table, "lrrrrrr",
              "cold start: medians over fresh processes (ms); header encode/parse: medians per call; "
              "field-stratified tampering")
    n_proc = min(v["n"] for v in summary["cold_start"].values())
    macros.add("ColdProcesses", n_proc)
    summary["tampering_distinct_flips"] = len({(r["suite"], r["byte"], r["bit"]) for r in tam})
    macros.add("TamperTrials", f"{len(tam):,}".replace(",", "{,}"))
    macros.add("TamperRejected", f"{sum(1 for r in tam if not r['plaintext_released']):,}".replace(",", "{,}"))
    macros.add("TamperFieldsMax", max(v["fields"] for v in summary["tampering"].values()))
    macros.add("TamperPerField", trials)
    macros.add("TamperExhaustiveMaxBytes", max([r["field_bits"] for r in tam if r["field_bits"] <= trials],
                                               default=0) // 8)
    # E and F seal in tens of milliseconds (scrypt), so a first-call cost is measurable only for A-D
    first_extra = [summary["cold_start"][c]["first_seal_extra_ms"] for c in "ABCD"]
    macros.add("ColdFirstSealExtraMinAtoDMs", fmt(min(first_extra), 2))
    macros.add("ColdFirstSealExtraMaxAtoDMs", fmt(max(first_extra), 2))
    macros.add("ColdImportMs", fmt(float(np.median([summary["cold_start"][c]["import_ms"] for c in SUITES])), 0))
    for op, name in (("encode", "Encode"), ("parse", "Parse")):
        vals = [summary["serialization"][c][f"{op}_us"] for c in SUITES]
        macros.add(f"{name}MinUs", fmt(min(vals), 1))
        macros.add(f"{name}MaxUs", fmt(max(vals), 1))
    cpu = [float(r["cpu_percent_total"]) for r in load]
    summary["load"] = {"samples": len(cpu), "cpu_percent_total_median": float(np.median(cpu)) if cpu else None,
                       "cpu_percent_total_p90": float(np.percentile(cpu, 90)) if cpu else None}
    if cpu:
        macros.add("LoadCpuMedianPct", fmt(float(np.median(cpu)), 1))
    macros.write(rep / "macros.tex")
    (rep / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/supplement"))
    ap.add_argument("--cold-repeats", type=int, default=10)
    ap.add_argument("--payload-bytes", type=int, default=1 << 20)
    ap.add_argument("--serialization-samples", type=int, default=200)
    ap.add_argument("--serialization-batch", type=int, default=50)
    ap.add_argument("--tamper-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261102)
    a = ap.parse_args()
    out = a.out
    if (out / "run.json").exists():
        print(f"{out} already exists; choose another --out", file=sys.stderr)
        return 1
    root = environment.repo_root()
    meta = {"started_at_utc": utc(), "argv": sys.argv[1:], "source": environment.source_identity(root),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "seed_purpose": "orders cold-start processes and serialization batches and chooses tamper positions; "
                            "never cryptographic"}
    env = environment.capture()
    load_path = Path(tempfile.mkdtemp(prefix="cryptochaos-load-")) / "load.csv"
    with LoadSampler(load_path, 1.0):
        print("cold start ...", flush=True)
        cold = cold_start(a.cold_repeats, a.payload_bytes, a.seed)
        print("serialization ...", flush=True)
        ser = serialization(a.payload_bytes, a.serialization_samples, a.serialization_batch, a.seed)
    print("tampering ...", flush=True)
    tam = tampering(a.tamper_trials, a.seed)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")
    write_csv(out / "raw" / "coldstart.csv", cold)
    write_csv(out / "raw" / "serialization.csv", ser)
    write_csv(out / "raw" / "tamper.csv", tam)
    load = []
    if load_path.exists():
        shutil.copyfile(load_path, out / "raw" / "load.csv")
        with load_path.open(newline="", encoding="utf-8") as fh:
            load = list(csv.DictReader(fh))
    shutil.rmtree(load_path.parent, ignore_errors=True)
    summary = report(out, cold, ser, tam, a.tamper_trials, load)
    meta["finished_at_utc"] = utc()
    meta["errors"] = sum(1 for r in cold if r["error"])
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    write_hashes(out)
    print(json.dumps({k: summary["tampering"][k]["rejected"] for k in SUITES}), "rejected per configuration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
