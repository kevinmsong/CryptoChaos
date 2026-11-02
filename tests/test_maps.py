"""Configuration D maps (negative control), the five-byte bottleneck, and frozen golden vectors."""
import collections

import numpy as np
import pytest

from cryptochaos import maps as M
from conftest import LEGACY_KEY_COLLISION, LEGACY_SEED_COLLISION


def test_frozen_lut_matches_exact_rational_definition():
    assert M.LEGACY_CHEBYSHEV_LUT == M.legacy_chebyshev_lut_exact()


def test_frozen_lut_matches_floating_point_construction_on_this_platform():
    flt = tuple(int((np.cos(5 * np.arccos(2.0 * k / 255.0 - 1.0)) + 1.0) * 127.5) & 0xFF for k in range(256))
    assert flt == M.LEGACY_CHEBYSHEV_LUT


def test_legacy_golden_vector():
    assert M.legacy_seed("benchmark_passphrase_2025").hex() == "8295de35e8"
    assert M.legacy_chaotic_key("benchmark_passphrase_2025").hex() == (
        "359610cea609d940443cb8a38d27d32ea0bb92d1e233745040ea986721865887")


def test_legacy_key_depends_only_on_five_digest_bytes():
    a, b = LEGACY_SEED_COLLISION
    assert a != b and M.legacy_seed(a) == M.legacy_seed(b)
    assert M.legacy_chaotic_key(a) == M.legacy_chaotic_key(b)


def test_distinct_seeds_can_still_collide():
    a, b = LEGACY_KEY_COLLISION
    assert M.legacy_seed(a) != M.legacy_seed(b)
    assert M.legacy_chaotic_key(a) == M.legacy_chaotic_key(b)


def test_legacy_exhaustive_distinct_counts_and_tent_collapse():
    assert len({M.legacy_logistic(32, s) for s in range(256)}) == 113
    assert len({M.legacy_chebyshev(32, s) for s in range(256)}) == 165
    assert len({M.legacy_tent(32, s) for s in range(256)}) == 128
    assert all(M.legacy_tent(32, s)[7:] == bytes(25) for s in range(256))


@pytest.mark.slow
def test_legacy_henon_exhaustive_distinct_count():
    c = collections.Counter(M.legacy_henon(32, x, y) for x in range(256) for y in range(256))
    assert len(c) == 19456


V2_GOLDEN = {
    bytes(32): "fe719c82750f6b4f01e125fea216b0a42c347be7467847dd15d591e1769ef39d",
    bytes(range(32)): "41f8755f9f42a1cd610e9be28446035a0e8bdb8b1ed3e22916c4c9645a6a51a2",
    b"\xff" * 32: "6e370d3b32b96236f6240989dfc728a3039cfc0398da792dadec3c61c1382dab",
}


@pytest.mark.parametrize("pw_key,expected", list(V2_GOLDEN.items()))
def test_v2_golden_vectors(pw_key, expected):
    assert M.v2_chaos_value(pw_key).hex() == expected


def test_v2_depends_on_every_password_key_byte():
    base = bytes(range(32))
    ref = M.v2_chaos_value(base)
    for i in range(32):
        k = bytearray(base)
        k[i] ^= 0x01
        assert M.v2_chaos_value(bytes(k)) != ref


def test_v2_requires_full_width_key():
    with pytest.raises(ValueError):
        M.v2_chaos_value(bytes(16))


def test_v2_maps_stay_in_range_and_avoid_zero_fixed_points():
    m = M.V2Maps(64)
    for x in (1, 2, m.half - 1, m.half, m.half + 1, m.M - 2, m.M - 1):
        for step in (m.logistic, m.tent, m.chebyshev):
            y = step(x)
            assert 0 <= y < m.M
        assert m.logistic(x) != 0 and m.tent(m.fix_seed_1d(x)) != 0
    for x, y in ((0, 0), (m.M - 1, m.M - 1), (m.half, m.half)):
        nx, ny = m.henon(x, y)
        assert 0 <= nx < m.M and 0 <= ny < m.M
