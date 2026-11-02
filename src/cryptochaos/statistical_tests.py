"""A validated subset of the NIST SP 800-22 Rev. 1a statistical tests.

Implemented exactly as specified in SP 800-22 Rev. 1a and NIST STS 2.1.2 (formulas, category
boundaries, probability tables, and cephes ``igamc`` semantics), vectorized with NumPy:
frequency (monobit), block frequency, runs, longest run of ones, discrete Fourier transform
(spectral), non-overlapping template matching (one declared aperiodic template), and serial
(both p-values reported separately). This is a declared subset, not the full suite.

Every result carries an explicit status: ``ok``, ``not_applicable`` (input too short or
parameters outside the documented recommendations), or ``prerequisite_failed`` (runs test when
the monobit prerequisite fails). Undefined statistics are never converted into favourable
values, and errors are never reported as ordinary failures. Statistical testing is a
diagnostic, not certification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sp_fft
from scipy import special

ALPHA = 0.01
SAMPLE_BITS = 1_000_000

# Principal parameters for 1,000,000-bit samples (see docs/methodology.md):
BLOCK_FREQUENCY_M = 20_000  # SP 800-22 2.2.7: M >= 20, M > 0.01 n, N < 100  ->  N = 50
TEMPLATE_M = 9
TEMPLATE = "000000001"  # first aperiodic m=9 template in the STS template file
TEMPLATE_BLOCKS = 8
SERIAL_M = 16  # requires m < floor(log2 n) - 2 = 17


@dataclass(frozen=True)
class TestResult:
    test: str
    p_values: tuple[float, ...]
    status: str = "ok"  # ok | not_applicable | prerequisite_failed
    reason: str = ""
    statistic: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def passed(self, alpha: float = ALPHA) -> tuple[bool, ...] | None:
        """Per-p-value pass flags at level alpha; None when the test did not produce p-values."""
        if self.status != "ok":
            return None
        return tuple(p >= alpha for p in self.p_values)


def igamc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) with cephes semantics (Q = 1 for x <= 0)."""
    if x <= 0 or a <= 0:
        return 1.0
    return float(special.gammaincc(a, x))


def _na(test: str, reason: str, **params) -> TestResult:
    return TestResult(test, (), "not_applicable", reason, {}, params)


def as_bits(data: bytes) -> np.ndarray:
    """Bytes -> 0/1 uint8 array, most significant bit of each byte first (STS convention)."""
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_from_string(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")


def sample_bits(payload: bytes | np.ndarray, n_bits: int = SAMPLE_BITS) -> np.ndarray | None:
    """The prespecified sample: the first n_bits of the payload ciphertext, MSB-first.

    Returns None when the payload is shorter than n_bits (recorded as not applicable).
    """
    raw = np.frombuffer(payload, dtype=np.uint8) if not isinstance(payload, np.ndarray) else payload
    if raw.size * 8 < n_bits:
        return None
    nbytes = (n_bits + 7) // 8
    return np.unpackbits(raw[:nbytes])[:n_bits]


# --- 2.1 frequency (monobit) -----------------------------------------------------------------

def frequency(bits: np.ndarray, strict: bool = True) -> TestResult:
    n = bits.size
    if strict and n < 100:
        return _na("frequency", "n < 100", n=n)
    s_n = 2 * int(np.count_nonzero(bits)) - n
    s_obs = abs(s_n) / math.sqrt(n)
    return TestResult("frequency", (math.erfc(s_obs / math.sqrt(2)),), statistic={"S_n": s_n, "s_obs": s_obs},
                      params={"n": n})


# --- 2.2 block frequency ---------------------------------------------------------------------

def block_frequency(bits: np.ndarray, M: int = BLOCK_FREQUENCY_M, strict: bool = True) -> TestResult:
    n = bits.size
    N = n // M
    if N < 1:
        return _na("block_frequency", "fewer than one block", n=n, M=M)
    if strict and not (n >= 100 and M >= 20 and M > 0.01 * n and N < 100):
        return _na("block_frequency", "parameters outside SP 800-22 2.2.7 recommendations", n=n, M=M, N=N)
    sums = bits[: N * M].reshape(N, M).sum(axis=1, dtype=np.int64)
    pi = sums / M
    chi2 = 4.0 * M * float(((pi - 0.5) ** 2).sum())
    return TestResult("block_frequency", (igamc(N / 2.0, chi2 / 2.0),), statistic={"chi2": chi2},
                      params={"n": n, "M": M, "N": N})


# --- 2.3 runs --------------------------------------------------------------------------------

def runs(bits: np.ndarray, strict: bool = True) -> TestResult:
    n = bits.size
    if strict and n < 100:
        return _na("runs", "n < 100", n=n)
    pi = int(np.count_nonzero(bits)) / n
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:  # SP 800-22 2.3.4 step 2
        return TestResult("runs", (), "prerequisite_failed", "frequency prerequisite |pi - 1/2| >= 2/sqrt(n)",
                          {"pi": pi}, {"n": n})
    v_obs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    arg = abs(v_obs - 2.0 * n * pi * (1 - pi)) / (2.0 * math.sqrt(2 * n) * pi * (1 - pi))
    return TestResult("runs", (math.erfc(arg),), statistic={"V_obs": v_obs, "pi": pi}, params={"n": n})


# --- 2.4 longest run of ones in a block --------------------------------------------------------

_LONGEST_RUN_REGIMES = (
    # (min n, M, K, category values V0..VK, probabilities) -- STS 2.1.2 constants
    (750_000, 10_000, 6, (10, 11, 12, 13, 14, 15, 16),
     (0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727)),
    (6_272, 128, 5, (4, 5, 6, 7, 8, 9),
     (0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847)),
    (128, 8, 3, (1, 2, 3, 4), (0.21484375, 0.3671875, 0.23046875, 0.1875)),
)


def longest_run_per_block(blocks: np.ndarray) -> np.ndarray:
    """Longest run of ones in each row of a 0/1 matrix (vectorized)."""
    n_blocks, m = blocks.shape
    padded = np.zeros((n_blocks, m + 2), dtype=np.int8)
    padded[:, 1:-1] = blocks
    flat = padded.reshape(-1)
    zeros = np.flatnonzero(flat == 0)
    gaps = np.diff(zeros) - 1
    rows = zeros[:-1] // (m + 2)
    longest = np.zeros(n_blocks, dtype=np.int64)
    np.maximum.at(longest, rows, gaps)
    return longest


def longest_run(bits: np.ndarray, probabilities: tuple[float, ...] | None = None) -> TestResult:
    n = bits.size
    for min_n, M, K, V, pis in _LONGEST_RUN_REGIMES:
        if n >= min_n:
            break
    else:
        return _na("longest_run", "n < 128", n=n)
    if probabilities is not None:
        pis = probabilities
    N = n // M
    v = longest_run_per_block(bits[: N * M].reshape(N, M))
    nu = np.zeros(K + 1, dtype=np.int64)
    nu[0] = np.count_nonzero(v <= V[0])
    for j in range(1, K):
        nu[j] = np.count_nonzero(v == V[j])
    nu[K] = np.count_nonzero(v >= V[K])
    expected = N * np.asarray(pis)
    chi2 = float(((nu - expected) ** 2 / expected).sum())
    return TestResult("longest_run", (igamc(K / 2.0, chi2 / 2.0),), statistic={"chi2": chi2, "nu": nu.tolist()},
                      params={"n": n, "M": M, "K": K, "N": N})


# --- 2.6 discrete Fourier transform (spectral) ------------------------------------------------

def spectral(bits: np.ndarray, strict: bool = True) -> TestResult:
    n = bits.size
    if strict and n < 1000:
        return _na("spectral", "n < 1000", n=n)
    x = bits.astype(np.float64) * 2.0 - 1.0
    mags = np.abs(sp_fft.rfft(x))[: n // 2]
    t = math.sqrt(math.log(1 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = int(np.count_nonzero(mags < t))
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return TestResult("spectral", (math.erfc(abs(d) / math.sqrt(2)),), statistic={"N1": n1, "N0": n0, "d": d},
                      params={"n": n, "threshold": t})


# --- 2.7 non-overlapping template matching ----------------------------------------------------

def is_aperiodic(template: str) -> bool:
    m = len(template)
    return all(template[k:] != template[: m - k] for k in range(1, m))


def aperiodic_templates(m: int) -> list[str]:
    return [t for t in (format(i, f"0{m}b") for i in range(1 << m)) if is_aperiodic(t)]


def _window_values(bits: np.ndarray, m: int) -> np.ndarray:
    n = bits.size - m + 1
    w = np.zeros(n, dtype=np.uint32)
    for k in range(m):
        w = (w << np.uint32(1)) | bits[k : k + n].astype(np.uint32)
    return w


def non_overlapping_template(bits: np.ndarray, template: str = TEMPLATE, N: int = TEMPLATE_BLOCKS,
                             strict: bool = True) -> TestResult:
    m = len(template)
    n = bits.size
    if not is_aperiodic(template):
        return _na("nonoverlapping_template", "template is periodic", template=template)
    M = n // N
    if M < m:
        return _na("nonoverlapping_template", "block shorter than template", n=n, N=N, m=m)
    if strict and not (N <= 100 and M > 0.01 * n):
        return _na("nonoverlapping_template", "parameters outside SP 800-22 2.7.7 recommendations", n=n, N=N, M=M)
    target = int(template, 2)
    counts = []
    for i in range(N):
        block = bits[i * M : (i + 1) * M]
        hits = np.flatnonzero(_window_values(block, m) == target)
        w_obs, next_free = 0, -1
        for pos in hits.tolist():
            if pos >= next_free:
                w_obs += 1
                next_free = pos + m
        counts.append(w_obs)
    mu = (M - m + 1) / 2**m
    var = M * (1.0 / 2**m - (2 * m - 1) / 2 ** (2 * m))
    chi2 = float(sum((w - mu) ** 2 for w in counts) / var)
    return TestResult("nonoverlapping_template", (igamc(N / 2.0, chi2 / 2.0),),
                      statistic={"chi2": chi2, "W": counts}, params={"n": n, "m": m, "N": N, "M": M,
                                                                       "template": template})


# --- 2.11 serial ------------------------------------------------------------------------------

def _psi2(bits: np.ndarray, m: int) -> float:
    if m <= 0:
        return 0.0
    n = bits.size
    ext = np.concatenate([bits, bits[: m - 1]]) if m > 1 else bits
    w = _window_values(ext, m)[:n]
    nu = np.bincount(w, minlength=1 << m).astype(np.int64)
    return float(int(np.dot(nu, nu)) * (2.0**m) / n - n)


def serial(bits: np.ndarray, m: int = SERIAL_M, strict: bool = True) -> TestResult:
    n = bits.size
    if m < 2:
        return _na("serial", "m < 2", m=m)
    if strict and not (m < int(math.floor(math.log2(n))) - 2):
        return _na("serial", "m >= floor(log2 n) - 2", n=n, m=m)
    p0, p1, p2 = _psi2(bits, m), _psi2(bits, m - 1), _psi2(bits, m - 2)
    del1 = p0 - p1
    del2 = p0 - 2 * p1 + p2
    pv1 = igamc(2.0 ** (m - 2), del1 / 2.0)
    pv2 = igamc(2.0 ** (m - 3), del2 / 2.0)
    return TestResult("serial", (pv1, pv2), statistic={"psi2_m": p0, "del1": del1, "del2": del2},
                      params={"n": n, "m": m})


# --- the principal battery -------------------------------------------------------------------

TEST_NAMES = ("frequency", "block_frequency", "runs", "longest_run", "spectral", "nonoverlapping_template",
              "serial")


def run_battery(bits: np.ndarray) -> list[TestResult]:
    """The prespecified battery for 1,000,000-bit samples (parameters in the module constants)."""
    return [
        frequency(bits),
        block_frequency(bits, BLOCK_FREQUENCY_M),
        runs(bits),
        longest_run(bits),
        spectral(bits),
        non_overlapping_template(bits, TEMPLATE, TEMPLATE_BLOCKS),
        serial(bits, SERIAL_M),
    ]


def battery_record(results: list[TestResult], alpha: float = ALPHA, prefix: str = "sts") -> dict:
    """Flatten battery results into one row: status, each p-value, and each pass flag."""
    row: dict = {}
    for r in results:
        key = f"{prefix}_{r.test}"
        row[f"{key}_status"] = r.status
        labels = ("p1", "p2") if r.test == "serial" else ("p",)
        for i, label in enumerate(labels):
            p = r.p_values[i] if r.status == "ok" else float("nan")
            row[f"{key}_{label}"] = p
            row[f"{key}_{label}_pass"] = (p >= alpha) if r.status == "ok" else None
    return row


def not_applicable_record(reason: str, prefix: str = "sts") -> dict:
    row: dict = {}
    for name in TEST_NAMES:
        key = f"{prefix}_{name}"
        row[f"{key}_status"] = "not_applicable"
        for label in (("p1", "p2") if name == "serial" else ("p",)):
            row[f"{key}_{label}"] = float("nan")
            row[f"{key}_{label}_pass"] = None
    row[f"{prefix}_reason"] = reason
    return row


# --- second-level checks (SP 800-22 section 4.2) -----------------------------------------------

def uniformity_p_value(p_values: np.ndarray, bins: int = 10) -> float:
    """SP 800-22 4.2.2: chi-square of p-value counts in ten equal bins (9 degrees of freedom)."""
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    counts, _ = np.histogram(p, bins=bins, range=(0.0, 1.0))
    expected = p.size / bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return igamc((bins - 1) / 2.0, chi2 / 2.0)


def proportion_interval(n_sequences: int, alpha: float = ALPHA) -> tuple[float, float]:
    """SP 800-22 4.2.1 acceptable range for the proportion passing: p_hat +/- 3 sqrt(p_hat(1-p_hat)/m)."""
    p_hat = 1 - alpha
    half = 3 * math.sqrt(p_hat * alpha / n_sequences)
    return p_hat - half, p_hat + half
