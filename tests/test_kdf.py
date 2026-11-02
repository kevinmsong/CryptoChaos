"""scrypt (RFC 7914) and HKDF (RFC 5869) vectors, IKM encoding, password semantics, full-width keys."""
import os
import unicodedata

import pytest

from cryptochaos import kdf
from cryptochaos.errors import UnsupportedParameterError
from cryptochaos.maps import legacy_chaotic_key, v2_chaos_value
from conftest import LEGACY_KEY_COLLISION

RFC7914 = [
    (b"", b"", 16, 1, 1,
     "77d6576238657b203b19ca42c18a0497f16b4844e3074ae8dfdffa3fede21442"
     "fcd0069ded0948f8326a753a0fc81f17e8d3e0fb2e0d3628cf35e20c38d18906"),
    (b"password", b"NaCl", 1024, 8, 16,
     "fdbabe1c9d3472007856e7190d01e9fe7c6ad7cbc8237830e77376634b373162"
     "2eaf30d92e22a3886ff109279d9830dac727afb94a83ee6d8360cbdfa2cc0640"),
    (b"pleaseletmein", b"SodiumChloride", 16384, 8, 1,
     "7023bdcb3afd7348461c06cd81fd38ebfda8fbba904f8e3ea9b543f6545da1f2"
     "d5432955613f0fcf62d49705242a9af9e61e85dc0d651e40dfcf017b45575887"),
]


@pytest.mark.parametrize("password,salt,n,r,p,expected", RFC7914)
def test_scrypt_rfc7914_vectors(password, salt, n, r, p, expected):
    assert kdf.scrypt_raw(password, salt, n, r, p, 64).hex() == expected


RFC5869 = [
    ("0b" * 22, "000102030405060708090a0b0c", "f0f1f2f3f4f5f6f7f8f9", 42,
     "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"),
    (bytes(range(0x00, 0x50)).hex(), bytes(range(0x60, 0xB0)).hex(), bytes(range(0xB0, 0x100)).hex(), 82,
     "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c59045a99cac7827271cb41c65e590e09"
     "da3275600c2f09b8367793a9aca3db71cc30c58179ec3e87c14c01d5c1f3434f1d87"),
    ("0b" * 22, "", "", 42,
     "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"),
]


@pytest.mark.parametrize("ikm,salt,info,length,expected", RFC5869)
def test_hkdf_rfc5869_vectors(ikm, salt, info, length, expected):
    out = kdf.hkdf_sha256(bytes.fromhex(ikm), bytes.fromhex(salt), bytes.fromhex(info), length)
    assert out.hex() == expected


def test_ikm_encoding_is_unambiguous():
    a = kdf.encode_ikm(2, 0x12, [(1, b"ab"), (2, b"c")])
    b = kdf.encode_ikm(2, 0x12, [(1, b"a"), (2, b"bc")])
    c = kdf.encode_ikm(2, 0x13, [(1, b"ab"), (2, b"c")])
    assert len({a, b, c}) == 3
    with pytest.raises(ValueError):
        kdf.encode_ikm(2, 0x12, [(1, b"a"), (1, b"b")])


def test_info_domain_separation():
    assert kdf.hkdf_info(2, 0x12) != kdf.hkdf_info(2, 0x13)
    assert kdf.hkdf_info(2, 0x12).startswith(b"CryptoChaos-v2")


def test_password_encoding_nfc_utf8():
    composed, decomposed = "café", "café"
    assert composed != decomposed
    assert kdf.encode_password(composed) == kdf.encode_password(decomposed) == "café".encode("utf-8")
    assert kdf.encode_password(b"\xff\x00raw") == b"\xff\x00raw"
    for bad in ("", b""):
        with pytest.raises(ValueError):
            kdf.encode_password(bad)
    with pytest.raises(ValueError):
        kdf.encode_password("x" * (kdf.MAX_PASSWORD_BYTES + 1))


@pytest.mark.parametrize("params", [kdf.ScryptParams(13), kdf.ScryptParams(19), kdf.ScryptParams(14, 16, 1),
                                    kdf.ScryptParams(14, 8, 2)])
def test_scrypt_policy_rejects_outside_allow_list(params):
    with pytest.raises(UnsupportedParameterError):
        kdf.DEFAULT_SCRYPT_POLICY.check(params)


def test_scrypt_memory_accounting():
    assert kdf.ScryptParams(14).memory_bytes == 16 * 2**20
    assert kdf.ScryptParams(18).memory_bytes == 256 * 2**20


def _key(suite_id, fields, salt=b"\x07" * 32):
    return kdf.derive_aead_key(2, suite_id, fields, salt)


def test_full_width_password_key_regression():
    """Every bit of the 32-byte password key reaches the AEAD key in suites E and F."""
    kem = os.urandom(32)
    pw_key = os.urandom(32)
    for suite_id, with_chaos in ((0x12, False), (0x13, True)):
        def fields(k):
            f = [(kdf.FIELD_KEM, kem), (kdf.FIELD_PW_KEY, k)]
            if with_chaos:
                f.append((kdf.FIELD_CHAOS_V2, v2_chaos_value(k)))
            return f
        ref = _key(suite_id, fields(pw_key))
        keys = {ref}
        for bit in range(256):
            k = bytearray(pw_key)
            k[bit // 8] ^= 1 << (bit % 8)
            keys.add(_key(suite_id, fields(bytes(k))))
        assert len(keys) == 257


def test_password_key_is_never_replaced_by_chaos_value():
    """F keeps pw_key as its own field; two keys with equal chaos values would still differ."""
    kem = os.urandom(32)
    k1, k2 = os.urandom(32), os.urandom(32)
    same_chaos = v2_chaos_value(k1)
    a = _key(0x13, [(kdf.FIELD_KEM, kem), (kdf.FIELD_PW_KEY, k1), (kdf.FIELD_CHAOS_V2, same_chaos)])
    b = _key(0x13, [(kdf.FIELD_KEM, kem), (kdf.FIELD_PW_KEY, k2), (kdf.FIELD_CHAOS_V2, same_chaos)])
    assert a != b


def test_legacy_pathway_bottleneck_regression():
    """Suite D: distinct passphrases with colliding legacy keys derive the identical AEAD key."""
    kem = os.urandom(32)
    p1, p2 = LEGACY_KEY_COLLISION
    k1 = _key(0x11, [(kdf.FIELD_KEM, kem), (kdf.FIELD_LEGACY_CHAOS, legacy_chaotic_key(p1))])
    k2 = _key(0x11, [(kdf.FIELD_KEM, kem), (kdf.FIELD_LEGACY_CHAOS, legacy_chaotic_key(p2))])
    assert k1 == k2


def test_derive_password_key_shape():
    salt = os.urandom(16)
    k = kdf.derive_password_key(b"pw", salt, kdf.ScryptParams(14))
    assert len(k) == 32
    assert k == kdf.scrypt_raw(b"pw", salt, 2**14, 8, 1, 32)
    with pytest.raises(ValueError):
        kdf.derive_password_key(b"pw", salt[:15], kdf.ScryptParams(14))
    assert unicodedata.is_normalized("NFC", "café")
