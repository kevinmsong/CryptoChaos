"""Deterministic chaotic maps: the 8-bit negative-control maps (configuration D) and the exact-integer
CryptoChaos maps (configuration F).

Neither pathway adds entropy. Each output is a deterministic function of its input:

* Configuration D (negative control): five bytes of ``BLAKE3(passphrase)`` seed four 8-bit maps. The output can
  take at most 2**40 values, and far fewer in practice (see ``map_analysis``).
* CryptoChaos (configuration F): the full 32-byte scrypt password key seeds four 64-bit exact-integer maps. The
  resulting value is appended to (never substituted for) the password key in the key schedule.

All arithmetic is on Python integers with explicitly specified floor division, so outputs are
platform-independent. Any change to these definitions changes the protocol and requires a new
suite or version identifier.
"""
from __future__ import annotations

from fractions import Fraction

import blake3

# ---------------------------------------------------------------------------------------------
# Negative-control maps (configuration D): 8-bit fixed point, seeded from five passphrase-digest bytes
# ---------------------------------------------------------------------------------------------

LEGACY_OUTPUT_BYTES = 32
LEGACY_LOGISTIC_R_FIXED = 1021  # int(3.99 * 256)
LEGACY_TENT_MU_FIXED = 256  # int(2.0 * 128)
LEGACY_HENON_A_FIXED = 179  # int(1.4 * 128)
LEGACY_HENON_B_FIXED = 76  # int(0.3 * 256)
LEGACY_CHEBYSHEV_DEGREE = 5


def legacy_chebyshev_lut_exact() -> tuple[int, ...]:
    """The 256-entry table floor((T5((2k-255)/255) + 1) * 255/2) mod 256 in exact rationals.

    The 8-bit design builds this table at run time as ``int((cos(5*arccos(2k/255 - 1)) + 1) * 127.5) & 0xFF``
    in floating point. cos(5 arccos x) is the Chebyshev polynomial T5(x) = 16x^5 - 20x^3 + 5x,
    so the exact rational evaluation is the platform-independent definition. It equals the
    floating-point table on the reference platform for all 256 entries (see tests); entries
    k=0 and k=255 lie exactly on integer boundaries, where floating-point evaluation could differ
    across platforms.
    """
    table = []
    for k in range(256):
        x = Fraction(2 * k - 255, 255)
        t5 = 16 * x**5 - 20 * x**3 + 5 * x
        table.append(int((t5 + 1) * Fraction(255, 2)) & 0xFF)  # value >= 0, so int() == floor
    return tuple(table)


LEGACY_CHEBYSHEV_LUT: tuple[int, ...] = (
    0, 24, 46, 68, 87, 106, 123, 139, 154, 167, 179, 191, 201, 210, 218, 225,
    232, 237, 242, 246, 249, 251, 253, 254, 254, 254, 254, 253, 251, 249, 246, 243,
    240, 236, 232, 228, 224, 219, 214, 209, 203, 198, 192, 186, 180, 174, 168, 162,
    155, 149, 143, 137, 130, 124, 118, 112, 106, 100, 94, 89, 83, 78, 72, 67,
    62, 57, 52, 48, 44, 39, 35, 32, 28, 25, 21, 18, 16, 13, 11, 9,
    7, 5, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5,
    6, 8, 10, 12, 14, 17, 19, 22, 25, 28, 31, 35, 38, 42, 46, 50,
    54, 58, 62, 67, 71, 76, 81, 85, 90, 95, 100, 105, 110, 115, 120, 125,
    129, 134, 139, 144, 149, 154, 159, 164, 169, 173, 178, 183, 187, 192, 196, 200,
    204, 208, 212, 216, 219, 223, 226, 229, 232, 235, 237, 240, 242, 244, 246, 248,
    249, 251, 252, 253, 254, 254, 254, 254, 254, 254, 253, 253, 252, 250, 249, 247,
    245, 243, 241, 238, 236, 233, 229, 226, 222, 219, 215, 210, 206, 202, 197, 192,
    187, 182, 176, 171, 165, 160, 154, 148, 142, 136, 130, 124, 117, 111, 105, 99,
    92, 86, 80, 74, 68, 62, 56, 51, 45, 40, 35, 30, 26, 22, 18, 14,
    11, 8, 5, 3, 1, 0, 0, 0, 0, 1, 3, 5, 8, 12, 17, 22,
    29, 36, 44, 53, 63, 75, 87, 100, 115, 131, 148, 167, 186, 208, 230, 255,
)


def legacy_logistic_step(x: int) -> int:
    return ((LEGACY_LOGISTIC_R_FIXED * x * (256 - x)) >> 16) & 0xFF


def legacy_chebyshev_step(x: int) -> int:
    return LEGACY_CHEBYSHEV_LUT[x]


def legacy_tent_step(x: int) -> int:
    if x < 128:
        x = (LEGACY_TENT_MU_FIXED * x) >> 7
    else:
        x = (LEGACY_TENT_MU_FIXED * (256 - x)) >> 7
    return x & 0xFF


def legacy_henon_step(x: int, y: int) -> tuple[int, int]:
    # Python's >> floors negative operands; the 8-bit design relies on that behavior.
    x_new = ((LEGACY_HENON_A_FIXED * (128 - ((x * x) >> 8))) >> 7) + y
    y_new = (LEGACY_HENON_B_FIXED * x) >> 8
    return x_new & 0xFF, y_new & 0xFF


def _iterate_1d(step, x0: int, size: int) -> bytes:
    out = bytearray(size)
    x = x0
    for i in range(size):
        x = step(x)
        out[i] = x
    return bytes(out)


def legacy_logistic(size: int, x0: int) -> bytes:
    return _iterate_1d(legacy_logistic_step, x0, size)


def legacy_chebyshev(size: int, x0: int) -> bytes:
    return _iterate_1d(legacy_chebyshev_step, x0, size)


def legacy_tent(size: int, x0: int) -> bytes:
    return _iterate_1d(legacy_tent_step, x0, size)


def legacy_henon(size: int, x0: int, y0: int) -> bytes:
    out = bytearray(size)
    x, y = x0, y0
    for i in range(size):
        x, y = legacy_henon_step(x, y)
        out[i] = x
    return bytes(out)


def legacy_seed(passphrase: str) -> bytes:
    """The only passphrase information configuration D uses: the first five bytes of BLAKE3(UTF-8 passphrase)."""
    return blake3.blake3(passphrase.encode("utf-8")).digest()[:5]


def legacy_map_outputs(seed: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    if len(seed) != 5:
        raise ValueError("the configuration D seed is exactly 5 bytes")
    h0, h1, h2, h3, h4 = seed
    return (
        legacy_logistic(LEGACY_OUTPUT_BYTES, h0),
        legacy_chebyshev(LEGACY_OUTPUT_BYTES, h1),
        legacy_tent(LEGACY_OUTPUT_BYTES, h2),
        legacy_henon(LEGACY_OUTPUT_BYTES, h3, h4),
    )


def legacy_chaotic_key_from_seed(seed: bytes) -> bytes:
    return blake3.blake3(b"".join(legacy_map_outputs(seed))).digest()


def legacy_chaotic_key(passphrase: str) -> bytes:
    """Configuration D's chaotic key: a function of five passphrase-digest bytes only."""
    return legacy_chaotic_key_from_seed(legacy_seed(passphrase))


# ---------------------------------------------------------------------------------------------
# Exact-integer maps of CryptoChaos (configuration F)
# ---------------------------------------------------------------------------------------------

V2_WORD_BITS = 64
V2_BURN_IN = 16
V2_OUTPUT_BYTES = 32
V2_SEED_LABEL = b"CryptoChaos-v2/chaos-seed"
V2_VALUE_LABEL = b"CryptoChaos-v2/chaos-value"


class V2Maps:
    """The configuration F maps on W-bit unsigned states (W=64 in the protocol; smaller W for analysis only).

    With M = 2**W:

    * logistic, r = 399/100:  x <- floor(399 x (M - x) / (100 M))
    * tent, mu = 199/100:     t = min(x, M-1-x);  x <- floor(199 t / 100)
    * Chebyshev degree 5:     u = 2x - (M-1);  w = floor((16u^5 - 20u^3 M^2 + 5u M^4) / M^4);
                              x <- min(M-1, floor((w + M) / 2))
    * Henon, a = 7/5, b = 3/10, on a torus with scale Q = M/8 and signed views X = x - M/2,
      Y = y - M/2:  X' = Q - floor(7 X^2 / (5 Q)) + Y,  Y' = floor(3 X / 10),
      x <- (X' + M/2) mod M,  y <- (Y' + M/2) mod M

    The logistic and tent fixed point 0 (and the tent preimage M-1) are excluded at seeding by
    mapping such seeds to 1; for the logistic and tent maps above, 0 is unreachable from any
    other state when W >= 8.
    """

    def __init__(self, word_bits: int = V2_WORD_BITS) -> None:
        if word_bits < 8:
            raise ValueError("word_bits must be >= 8 (each step emits the most significant byte)")
        self.W = word_bits
        self.M = 1 << word_bits
        self.half = self.M >> 1
        self.Q = self.M >> 3
        self.M2 = self.M * self.M
        self.M4 = self.M2 * self.M2

    def logistic(self, x: int) -> int:
        return (399 * x * (self.M - x)) // (100 * self.M)

    def tent(self, x: int) -> int:
        t = x if x < self.half else self.M - 1 - x
        return (199 * t) // 100

    def chebyshev(self, x: int) -> int:
        u = 2 * x - (self.M - 1)
        u2 = u * u
        w = (16 * u2 * u2 * u - 20 * u2 * u * self.M2 + 5 * u * self.M4) // self.M4
        return min(self.M - 1, (w + self.M) // 2)

    def henon(self, x: int, y: int) -> tuple[int, int]:
        X = x - self.half
        Y = y - self.half
        x_new = self.Q - (7 * X * X) // (5 * self.Q) + Y
        y_new = (3 * X) // 10
        return (x_new + self.half) % self.M, (y_new + self.half) % self.M

    def fix_seed_1d(self, x: int) -> int:
        return 1 if x in (0, self.M - 1) else x

    def emit_1d(self, step, x: int, burn_in: int = V2_BURN_IN, size: int = V2_OUTPUT_BYTES) -> bytes:
        shift = self.W - 8
        for _ in range(burn_in):
            x = step(x)
        out = bytearray(size)
        for i in range(size):
            x = step(x)
            out[i] = x >> shift
        return bytes(out)

    def emit_henon(self, x: int, y: int, burn_in: int = V2_BURN_IN, size: int = V2_OUTPUT_BYTES) -> bytes:
        shift = self.W - 8
        for _ in range(burn_in):
            x, y = self.henon(x, y)
        out = bytearray(size)
        for i in range(size):
            x, y = self.henon(x, y)
            out[i] = x >> shift
        return bytes(out)


_V2 = V2Maps(V2_WORD_BITS)


def v2_seed_words(pw_key: bytes) -> tuple[int, int, int, int, int]:
    """Five 64-bit big-endian words from BLAKE3(label || pw_key) (domain-separated seeding)."""
    if len(pw_key) != 32:
        raise ValueError("the configuration F maps are seeded from the full 32-byte password key")
    material = blake3.blake3(V2_SEED_LABEL + pw_key).digest(length=40)
    return tuple(int.from_bytes(material[8 * i : 8 * i + 8], "big") for i in range(5))  # type: ignore[return-value]


def v2_map_outputs(pw_key: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    x_l, x_t, x_c, x_h, y_h = v2_seed_words(pw_key)
    m = _V2
    return (
        m.emit_1d(m.logistic, m.fix_seed_1d(x_l)),
        m.emit_1d(m.tent, m.fix_seed_1d(x_t)),
        m.emit_1d(m.chebyshev, x_c),
        m.emit_henon(x_h, y_h),
    )


def v2_chaos_value(pw_key: bytes) -> bytes:
    """Suite F diversification value: a deterministic 32-byte function of pw_key (no added entropy)."""
    return blake3.blake3(V2_VALUE_LABEL + b"".join(v2_map_outputs(pw_key))).digest()
