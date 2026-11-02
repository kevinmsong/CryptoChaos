"""Generate STS 2.1.2 reference p-values for tests/vectors/sts/sts_reference.json.

Requires a locally built NIST STS ``assess`` executable (see docs/reproduction.md: download
sts-2_1_2.zip from NIST and compile, e.g. with ``python -m ziglang cc``). The stored vectors let
routine tests compare ``cryptochaos.statistical_tests`` with STS without the binary.

    python scripts/build_sts_vectors.py --sts-dir third_party/build/sts_src/sts-2.1.2/sts-2.1.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cryptochaos import statistical_tests as T
from cryptochaos.calibration import StreamSpec, generate

N_BITS = 1_000_000
TEST_FLAGS = "110110110000010"  # Frequency, BlockFrequency, Runs, LongestRun, FFT, NonOverlapping, Serial
STS_DIRS = {"frequency": "Frequency", "block_frequency": "BlockFrequency", "runs": "Runs",
            "longest_run": "LongestRun", "spectral": "FFT", "nonoverlapping_template": "NonOverlappingTemplate",
            "serial": "Serial"}

GENERATED = [StreamSpec("chacha20", i, "uniform", "ChaCha20 keystream") for i in range(3)] + [
    StreamSpec("mt19937", 7, "uniform", "MT19937"),
    StreamSpec("biased_0.502", 1, "reject", "Bernoulli(0.502)"),
    StreamSpec("biased_0.51", 2, "reject", "Bernoulli(0.51)"),
    StreamSpec("lcg_low_bit", 3, "reject", "LCG low bit"),
    StreamSpec("legacy_logistic", 77, "reject", "8-bit logistic map (configuration D)"),
    StreamSpec("repeated_block", 4, "reject", "repeated 1 KiB block"),
]
CONSTANTS = ["data.e", "data.pi", "data.sqrt2", "data.sqrt3"]


def run_sts(sts_dir: Path, exe: Path, input_path: Path, binary: bool) -> tuple[dict, list[float]]:
    """P-values of the declared subset, plus every aperiodic m = 9 template in STS's file order."""
    rel = input_path.resolve()
    script = "\n".join(["0", str(rel), "0", TEST_FLAGS, "1", str(T.BLOCK_FREQUENCY_M), "0", "1",
                        "1" if binary else "0", ""])
    subprocess.run([str(exe), str(N_BITS)], input=script.encode(), cwd=sts_dir, capture_output=True, timeout=600)
    out, all_templates = {}, []
    for name, d in STS_DIRS.items():
        values = [float(x) for x in (sts_dir / "experiments" / "AlgorithmTesting" / d / "results.txt").read_text().split()]
        if name == "nonoverlapping_template":
            all_templates = values
            values = values[:1]  # first template in template9 == 000000001
        out[name] = values
    return out, all_templates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sts-dir", required=True, type=Path)
    ap.add_argument("--exe", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("tests/vectors/sts/sts_reference.json"))
    a = ap.parse_args()
    exe = a.exe or (a.sts_dir / "assess.exe")
    work = a.sts_dir / "cc_inputs"
    work.mkdir(exist_ok=True)
    cases = []
    for name in CONSTANTS:
        path = a.sts_dir / "data" / name
        digits = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        bits = (digits[(digits == 48) | (digits == 49)] - 48).astype(np.uint8)[:N_BITS]
        cases.append(({"source": f"STS {name}", "kind": "sts_constant", "file": name}, bits, path, False))
    for spec in GENERATED:
        bits = generate(spec, N_BITS)
        path = work / f"{spec.kind}_{spec.index}.bin"
        path.write_bytes(np.packbits(bits).tobytes())
        cases.append(({"source": spec.description, "kind": spec.kind, "index": spec.index,
                       "expected": spec.expected}, bits, path, True))
    records = []
    for meta, bits, path, binary in cases:
        ref, all_templates = run_sts(a.sts_dir, exe, path, binary)
        if len(all_templates) != len(T.aperiodic_templates(9)):
            raise SystemExit(f"{meta['source']}: STS reported {len(all_templates)} template p-values")
        meta.update({"n_bits": N_BITS, "bits_sha256": hashlib.sha256(np.packbits(bits).tobytes()).hexdigest(),
                     "sts_p_values": ref, "sts_template9_all": all_templates})
        records.append(meta)
        print(meta["source"], {k: [round(x, 6) for x in v] for k, v in ref.items()}, file=sys.stderr)
    doc = {
        "generator": "NIST STS 2.1.2 (sts-2_1_2.zip, SHA-256 0238d2f1d26e120e3cc748ed2d4c674cdc636de37fc4027c76cc2a394fff9157), "
                     "compiled with zig cc",
        "parameters": {"n": N_BITS, "block_frequency_M": T.BLOCK_FREQUENCY_M, "template": T.TEMPLATE,
                       "template_blocks": T.TEMPLATE_BLOCKS, "serial_m": T.SERIAL_M},
        "precision_note": "STS prints six decimal places",
        "cases": records,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
