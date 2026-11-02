"""CPU reference byte diagnostics (descriptive only).

These quantities describe byte arrays. They cannot establish indistinguishability, IND-CCA
security, password strength, or superiority of one AEAD over another.

Conventions: inputs are 1-D ``uint8`` arrays; bit order within a byte is irrelevant for the
bit counts here. Differences use signed 16-bit subtraction and 64-bit integer accumulation, so
nothing overflows. For two independent uniform byte arrays the nominal references are
``UNEQUAL_BYTES_REF`` (99.609375 %), ``MEAN_ABS_DIFF_REF`` (33.463542 % of 255) and 50 % unequal bits.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

UNEQUAL_BYTES_REF = 100.0 * 255 / 256  # 99.609375
MEAN_ABS_DIFF_REF = 100.0 * (256 * 256 - 1) / (3 * 256) / 255  # 33.4635416...
BIT_DIFF_REF = 50.0


def as_u8(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            raise TypeError("expected a uint8 array")
        return data.reshape(-1)
    return np.frombuffer(data, dtype=np.uint8)


def byte_histogram(a: np.ndarray) -> np.ndarray:
    return np.bincount(as_u8(a), minlength=256).astype(np.int64)


def shannon_entropy(hist: np.ndarray) -> float:
    """Plug-in Shannon entropy of the byte histogram, bits per byte (biased low for finite n)."""
    n = hist.sum()
    p = hist[hist > 0] / n
    return float(-(p * np.log2(p)).sum())


def min_entropy(hist: np.ndarray) -> float:
    """Plug-in min-entropy of the byte histogram, bits per byte."""
    return float(-math.log2(hist.max() / hist.sum()))


def chi_square_uniform(hist: np.ndarray) -> tuple[float, float]:
    """Pearson chi-square of the byte histogram against uniform (255 degrees of freedom)."""
    n = hist.sum()
    expected = n / 256.0
    chi2 = float(((hist - expected) ** 2).sum() / expected)
    return chi2, float(stats.chi2.sf(chi2, 255))


def ones_count(a: np.ndarray) -> int:
    return int(np.bitwise_count(as_u8(a)).sum(dtype=np.int64))


def bit_balance(a: np.ndarray) -> float:
    a = as_u8(a)
    return ones_count(a) / (8.0 * a.size)


def lag_correlation(a: np.ndarray, lag: int) -> float:
    """Pearson correlation between bytes a[i] and a[i+lag], from exact integer moments."""
    a = as_u8(a)
    if lag < 1 or a.size <= lag + 1:
        return float("nan")
    x = a[:-lag].astype(np.int64)
    y = a[lag:].astype(np.int64)
    n = x.size
    sx, sy = int(x.sum()), int(y.sum())
    sxx, syy, sxy = int(np.dot(x, x)), int(np.dot(y, y)), int(np.dot(x, y))
    cov = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if vx == 0 or vy == 0:
        return float("nan")
    return float(cov / math.sqrt(vx * vy))


def unequal_byte_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Percentage of positions where two equal-length arrays differ (NPCR when used on images)."""
    a, b = as_u8(a), as_u8(b)
    _same_length(a, b)
    return 100.0 * int(np.count_nonzero(a != b)) / a.size


def mean_abs_diff_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute byte difference as a percentage of 255 (UACI when used on images)."""
    a, b = as_u8(a), as_u8(b)
    _same_length(a, b)
    d = np.abs(np.subtract(a, b, dtype=np.int16))
    return 100.0 * int(d.sum(dtype=np.int64)) / (255.0 * a.size)


def bit_diff_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Percentage of differing bits (Hamming distance / 8n)."""
    a, b = as_u8(a), as_u8(b)
    _same_length(a, b)
    return 100.0 * int(np.bitwise_count(np.bitwise_xor(a, b)).sum(dtype=np.int64)) / (8.0 * a.size)


def _same_length(a: np.ndarray, b: np.ndarray) -> None:
    if a.size != b.size:
        raise ValueError("arrays must have equal length")
    if a.size == 0:
        raise ValueError("arrays must be non-empty")


def ciphertext_diagnostics(ct: np.ndarray, lags=(1, 2, 3)) -> dict[str, float]:
    """Descriptive diagnostics of one byte array (full length, no sampling)."""
    ct = as_u8(ct)
    hist = byte_histogram(ct)
    chi2, chi2_p = chi_square_uniform(hist)
    out = {
        "n_bytes": int(ct.size),
        "shannon_entropy_bits_per_byte": shannon_entropy(hist),
        "min_entropy_bits_per_byte": min_entropy(hist),
        "chi2_uniform": chi2,
        "chi2_uniform_p": chi2_p,
        "bit_ones_fraction": bit_balance(ct),
    }
    for lag in lags:
        out[f"lag{lag}_correlation"] = lag_correlation(ct, lag)
    return out


def difference_metrics(a: np.ndarray, b: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_unequal_byte_pct": unequal_byte_pct(a, b),
        f"{prefix}_mean_abs_diff_pct": mean_abs_diff_pct(a, b),
        f"{prefix}_bit_diff_pct": bit_diff_pct(a, b),
    }
