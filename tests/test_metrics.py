"""Descriptive byte diagnostics: exact formulas, overflow safety, nominal references."""
import numpy as np
import pytest

from cryptochaos import metrics as M


def test_reference_constants():
    assert M.UNEQUAL_BYTES_REF == 99.609375
    assert abs(M.MEAN_ABS_DIFF_REF - 33.463541666) < 1e-8
    assert M.BIT_DIFF_REF == 50.0


def test_difference_metrics_exact_and_overflow_safe():
    a = np.array([0, 255, 10, 200], dtype=np.uint8)
    b = np.array([255, 0, 10, 100], dtype=np.uint8)
    assert M.unequal_byte_pct(a, b) == 75.0
    assert M.mean_abs_diff_pct(a, b) == pytest.approx(100.0 * (255 + 255 + 0 + 100) / (255 * 4))
    assert M.bit_diff_pct(a, b) == pytest.approx(100.0 * (8 + 8 + 0 + bin(200 ^ 100).count("1")) / 32)
    zeros, ones = np.zeros(1000, np.uint8), np.full(1000, 255, np.uint8)
    assert M.mean_abs_diff_pct(zeros, ones) == 100.0 and M.bit_diff_pct(zeros, ones) == 100.0
    with pytest.raises(ValueError):
        M.unequal_byte_pct(a, b[:3])
    with pytest.raises(ValueError):
        M.bit_diff_pct(a[:0], b[:0])


def test_independent_uniform_arrays_approach_nominal_references():
    rng = np.random.Generator(np.random.PCG64(11))
    a = rng.integers(0, 256, 4_000_000, dtype=np.uint8)
    b = rng.integers(0, 256, 4_000_000, dtype=np.uint8)
    assert abs(M.unequal_byte_pct(a, b) - M.UNEQUAL_BYTES_REF) < 0.02
    assert abs(M.mean_abs_diff_pct(a, b) - M.MEAN_ABS_DIFF_REF) < 0.05
    assert abs(M.bit_diff_pct(a, b) - 50.0) < 0.02


def test_histogram_entropy_chi2():
    uniform = np.tile(np.arange(256, dtype=np.uint8), 100)
    h = M.byte_histogram(uniform)
    assert h.tolist() == [100] * 256
    assert M.shannon_entropy(h) == pytest.approx(8.0)
    assert M.min_entropy(h) == pytest.approx(8.0)
    chi2, p = M.chi_square_uniform(h)
    assert chi2 == 0.0 and p == 1.0
    const = M.byte_histogram(np.zeros(1000, np.uint8))
    assert M.shannon_entropy(const) == 0.0 and M.min_entropy(const) == 0.0


def test_bit_balance_and_lag_correlation():
    a = np.array([0xFF, 0x00, 0x0F, 0xF0], dtype=np.uint8)
    assert M.bit_balance(a) == 0.5
    rng = np.random.Generator(np.random.PCG64(3))
    x = rng.integers(0, 256, 100_000, dtype=np.uint8)
    for lag in (1, 2, 3):
        ref = np.corrcoef(x[:-lag].astype(float), x[lag:].astype(float))[0, 1]
        assert M.lag_correlation(x, lag) == pytest.approx(ref, abs=1e-12)
    ramp = np.arange(256, dtype=np.uint8)
    assert M.lag_correlation(ramp, 1) == pytest.approx(1.0)
    assert np.isnan(M.lag_correlation(np.zeros(10, np.uint8), 1))


def test_diagnostics_record():
    rng = np.random.Generator(np.random.PCG64(9))
    ct = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    d = M.ciphertext_diagnostics(ct)
    assert d["n_bytes"] == 1 << 20
    assert 7.99 < d["shannon_entropy_bits_per_byte"] <= 8.0
    assert set(M.difference_metrics(ct, ct[::-1].copy(), "x")) == {
        "x_unequal_byte_pct", "x_mean_abs_diff_pct", "x_bit_diff_pct"}
