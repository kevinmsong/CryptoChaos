"""Declared bit-stream generators for randomness-test calibration and reference vectors.

Uniform streams are ChaCha20 keystreams under a key derived from a public label, so every
stream is reproducible from (label, index) while having the statistical quality of a modern
stream cipher. Structured negative controls are expected to be rejected by some tests.
These streams exist only to calibrate the statistical tests; they are not a key-generation
method, and fixed seeds are never used for protocol randomness.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from .maps import legacy_logistic_step

CALIBRATION_LABEL = "CryptoChaos-v2/calibration"


def chacha20_stream(label: str, index: int, n_bytes: int) -> bytes:
    key = hashlib.sha256(f"{CALIBRATION_LABEL}/{label}".encode()).digest()
    nonce = (0).to_bytes(4, "little") + index.to_bytes(12, "little")  # 32-bit counter || 96-bit nonce
    return Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(bytes(n_bytes))


def biased_bits(seed: int, n_bits: int, p_one: float) -> np.ndarray:
    return (np.random.Generator(np.random.PCG64(seed)).random(n_bits) < p_one).astype(np.uint8)


def mt19937_bits(seed: int, n_bits: int) -> np.ndarray:
    """A non-cryptographic but statistically strong generator (expected to pass)."""
    return np.random.Generator(np.random.MT19937(seed)).integers(0, 2, n_bits, dtype=np.uint8)


def lcg_low_bit(seed: int, n_bits: int) -> np.ndarray:
    """Lowest bit of x <- (1103515245 x + 12345) mod 2^31; it alternates with period 2."""
    out = np.empty(n_bits, dtype=np.uint8)
    x = seed % (1 << 31)
    for i in range(n_bits):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out[i] = x & 1
    return out


def legacy_logistic_bits(seed: int, n_bits: int) -> np.ndarray:
    """Bytes of the 8-bit logistic map of configuration D (finite-state, quickly periodic)."""
    n_bytes = (n_bits + 7) // 8
    out = np.empty(n_bytes, dtype=np.uint8)
    x = seed & 0xFF
    for i in range(n_bytes):
        x = legacy_logistic_step(x)
        out[i] = x
    return np.unpackbits(out)[:n_bits]


def repeated_block_bits(seed: int, n_bits: int, block_bytes: int = 1024) -> np.ndarray:
    block = np.frombuffer(chacha20_stream("repeated-block", seed, block_bytes), dtype=np.uint8)
    reps = -(-n_bits // (8 * block_bytes))
    return np.unpackbits(np.tile(block, reps))[:n_bits]


def uniform_bits(label: str, index: int, n_bits: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(chacha20_stream(label, index, (n_bits + 7) // 8), dtype=np.uint8))[:n_bits]


@dataclass(frozen=True)
class StreamSpec:
    kind: str
    index: int
    expected: str  # "uniform" | "reject"
    description: str


def generate(spec: StreamSpec, n_bits: int) -> np.ndarray:
    gens: dict[str, Callable[[], np.ndarray]] = {
        "chacha20": lambda: uniform_bits("uniform", spec.index, n_bits),
        "mt19937": lambda: mt19937_bits(spec.index, n_bits),
        "biased_0.502": lambda: biased_bits(spec.index, n_bits, 0.502),
        "biased_0.51": lambda: biased_bits(spec.index, n_bits, 0.51),
        "lcg_low_bit": lambda: lcg_low_bit(spec.index + 1, n_bits),
        "legacy_logistic": lambda: legacy_logistic_bits(spec.index, n_bits),
        "repeated_block": lambda: repeated_block_bits(spec.index, n_bits),
    }
    return gens[spec.kind]()


NEGATIVE_CONTROLS = {
    "biased_0.502": "Bernoulli(0.502) bits (PCG64): mild global bias",
    "biased_0.51": "Bernoulli(0.51) bits (PCG64): strong global bias",
    "lcg_low_bit": "low bit of a 31-bit LCG (period 2)",
    "legacy_logistic": "8-bit logistic-map bytes of configuration D (finite-state cycle)",
    "repeated_block": "a 1 KiB ChaCha20 block repeated",
}
POSITIVE_CONTROLS = {
    "chacha20": "ChaCha20 keystream (uniform reference)",
    "mt19937": "Mersenne Twister MT19937 (non-cryptographic, statistically strong)",
}
