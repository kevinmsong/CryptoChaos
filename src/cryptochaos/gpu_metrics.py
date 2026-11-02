"""Optional GPU byte-diagnostic reductions (Triton) with a CPU reference and parity checks.

Only bulk descriptive reductions run on the GPU: a 256-bin byte histogram and one fused pass
computing unequal-byte count, sum of absolute byte differences, and XOR popcount for two
equal-length arrays. No cryptographic operation (AES, ML-KEM, KDFs, RNG) is implemented here.

Integer results must equal the CPU reference exactly. Per-program partial sums are int32 (at
most BLOCK * 255 < 2**31); global accumulators are int64 updated with atomic adds.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BackendUnavailableError

BLOCK = 1 << 16


# --- CPU reference -----------------------------------------------------------------------------

def cpu_histogram(a: np.ndarray) -> np.ndarray:
    return np.bincount(a, minlength=256).astype(np.int64)


def cpu_diffs(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    unequal = int(np.count_nonzero(a != b))
    absdiff = int(np.abs(np.subtract(a, b, dtype=np.int16)).sum(dtype=np.int64))
    popcnt = int(np.bitwise_count(np.bitwise_xor(a, b)).sum(dtype=np.int64))
    return unequal, absdiff, popcnt


# --- Triton --------------------------------------------------------------------------------------

_KERNELS: dict[str, Any] = {}


def _triton_kernels():
    if _KERNELS:
        return _KERNELS
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailableError(f"triton/torch unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise BackendUnavailableError("no CUDA device")

    @triton.jit
    def hist_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int32)
        h = tl.histogram(x, 256)
        # masked-off lanes were loaded as 0: remove them from bin 0
        pad = BLOCK_SIZE - tl.sum(mask.to(tl.int32), axis=0)
        bins = tl.arange(0, 256)
        h = h - tl.where(bins == 0, pad, 0)
        tl.atomic_add(out_ptr + bins, h.to(tl.int64))

    @triton.jit
    def diff_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int32)
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int32)
        neq = tl.sum((a != b).to(tl.int32), axis=0)
        d = a - b
        absd = tl.sum(tl.where(d < 0, -d, d), axis=0)
        x = a ^ b
        x = x - ((x >> 1) & 0x55)
        x = (x & 0x33) + ((x >> 2) & 0x33)
        x = (x + (x >> 4)) & 0x0F
        pc = tl.sum(x, axis=0)
        tl.atomic_add(out_ptr + 0, neq.to(tl.int64))
        tl.atomic_add(out_ptr + 1, absd.to(tl.int64))
        tl.atomic_add(out_ptr + 2, pc.to(tl.int64))

    _KERNELS.update(torch=torch, triton=triton, hist=hist_kernel, diff=diff_kernel)
    return _KERNELS


def gpu_histogram(x) -> "Any":
    k = _triton_kernels()
    torch = k["torch"]
    out = torch.zeros(256, dtype=torch.int64, device=x.device)
    n = x.numel()
    if n:
        grid = (k["triton"].cdiv(n, BLOCK),)
        k["hist"][grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


def gpu_diffs(a, b) -> "Any":
    k = _triton_kernels()
    torch = k["torch"]
    if a.numel() != b.numel():
        raise ValueError("arrays must have equal length")
    out = torch.zeros(3, dtype=torch.int64, device=a.device)
    n = a.numel()
    if n:
        grid = (k["triton"].cdiv(n, BLOCK),)
        k["diff"][grid](a, b, out, n, BLOCK_SIZE=BLOCK)
    return out


def torch_reference(a, b) -> tuple["Any", "Any"]:
    """Plain-PyTorch GPU reductions (no Triton) as a second GPU reference."""
    k = _triton_kernels()
    torch = k["torch"]
    hist = torch.bincount(a.to(torch.int32), minlength=256)
    d = a.to(torch.int16) - b.to(torch.int16)
    x = (a ^ b).to(torch.int32)
    pc = sum(((x >> i) & 1).sum() for i in range(8))
    return hist, torch.stack([(a != b).sum(), d.abs().sum(), pc])


# --- parity and benchmark ------------------------------------------------------------------------

def parity_cases(seed: int = 1) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    cases = []
    for n in (1, 2, 3, 255, 256, 257, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 17, 1_000_003):
        cases.append((f"random-{n}", rng.integers(0, 256, n, dtype=np.uint8), rng.integers(0, 256, n, dtype=np.uint8)))
    n = 4 * BLOCK + 5
    z, f = np.zeros(n, np.uint8), np.full(n, 255, np.uint8)
    ramp = (np.arange(n) % 256).astype(np.uint8)
    cases += [("zeros-vs-ff", z, f), ("ff-vs-zeros", f, z), ("ramp-vs-ramp", ramp, ramp.copy()),
              ("ramp-vs-reversed", ramp, ramp[::-1].copy()), ("alternating", np.tile(np.array([0, 255], np.uint8), n // 2),
                                                             np.tile(np.array([255, 0], np.uint8), n // 2))]
    return cases


def check_parity() -> list[dict[str, Any]]:
    k = _triton_kernels()
    torch = k["torch"]
    results = []
    for name, a, b in parity_cases():
        ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
        h = gpu_histogram(ta).cpu().numpy()
        d = tuple(int(v) for v in gpu_diffs(ta, tb).cpu().numpy())
        results.append({"case": name, "n": a.size, "hist_equal": bool(np.array_equal(h, cpu_histogram(a))),
                        "diffs_equal": d == cpu_diffs(a, b)})
    return results


def benchmark_backends(backend: str, sizes: list[int], repeats: int, out: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.Generator(np.random.PCG64(2026))
    use_gpu = backend in ("triton", "torch", "all")
    k = None
    if use_gpu:
        try:
            k = _triton_kernels()
        except BackendUnavailableError as exc:
            rows.append({"backend": backend, "status": "skipped", "reason": str(exc)})
            use_gpu = False
    for n in sizes:
        a = rng.integers(0, 256, n, dtype=np.uint8)
        b = rng.integers(0, 256, n, dtype=np.uint8)
        if backend in ("cpu", "all"):
            t = []
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                cpu_histogram(a)
                cpu_diffs(a, b)
                t.append(time.perf_counter_ns() - t0)
            rows.append({"backend": "cpu-numpy", "bytes": n, "kernel_ms": float(np.median(t)) / 1e6,
                         "end_to_end_ms": float(np.median(t)) / 1e6, "status": "ok"})
        if use_gpu and k is not None:
            torch = k["torch"]
            ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
            variants = []
            if backend in ("triton", "all"):
                variants.append(("gpu-triton", lambda x, y: (gpu_histogram(x), gpu_diffs(x, y))))
            if backend in ("torch", "all"):
                variants.append(("gpu-torch", torch_reference))
            for vname, fn in variants:
                torch.cuda.synchronize()
                c0 = time.perf_counter_ns()
                fn(ta, tb)  # first call includes JIT compilation (Triton) / kernel loading
                torch.cuda.synchronize()
                first_ms = (time.perf_counter_ns() - c0) / 1e6
                kt, et = [], []
                for _ in range(repeats):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn(ta, tb)
                    e.record()
                    torch.cuda.synchronize()
                    kt.append(s.elapsed_time(e))
                    t0 = time.perf_counter_ns()
                    xa, xb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
                    h, d = fn(xa, xb)
                    h.cpu(), d.cpu()
                    torch.cuda.synchronize()
                    et.append((time.perf_counter_ns() - t0) / 1e6)
                rows.append({"backend": vname, "bytes": n, "kernel_ms": float(np.median(kt)),
                             "end_to_end_ms": float(np.median(et)), "first_call_ms": first_ms, "status": "ok",
                             "device": torch.cuda.get_device_name(0)})
    if out is not None:
        import csv

        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k2 for r in rows for k2 in r})
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return rows
