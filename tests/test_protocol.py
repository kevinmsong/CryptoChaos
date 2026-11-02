"""CryptoChaos envelope: round trips, rejection of every modification class, framing, roles."""
import hashlib
import os
import struct
import time

import pytest

import cryptochaos as cc
from cryptochaos import _fixtures, kdf
from cryptochaos.errors import (
    DecryptionError,
    EnvelopeFormatError,
    LengthLimitError,
    UnsupportedParameterError,
    UnsupportedSuiteError,
    UnsupportedVersionError,
)
from cryptochaos.kem import get_kem_backend
from cryptochaos.protocol import (
    SUITES,
    TAG_BYTES,
    PasswordSession,
    Receiver,
    Sender,
    envelope_overhead,
    header_length,
    parse_envelope,
)
from conftest import FAST_POLICY, FAST_SCRYPT, LEGACY_KEY_COLLISION

PW = "correct horse ⚡ battery staple"
ALL = "ABCDEF"


def pair(code, keypair, psk, *, fast=True, accepted=None):
    s = Sender(code, recipient_public_key=keypair.public_key, psk=psk,
               scrypt=FAST_SCRYPT if fast else kdf.DEFAULT_SCRYPT,
               scrypt_policy=FAST_POLICY if fast else kdf.DEFAULT_SCRYPT_POLICY)
    r = Receiver(recipient_secret_key=keypair.secret_key, psk=psk, accepted_suites=accepted or {code},
                 scrypt_policy=FAST_POLICY if fast else kdf.DEFAULT_SCRYPT_POLICY)
    return s, r


def pw_for(code):
    return PW if SUITES[code].password else None


@pytest.mark.parametrize("code", list(ALL))
@pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 1000, 1 << 20])
def test_round_trip_sizes(code, size, keypair, psk):
    s, r = pair(code, keypair, psk)
    data = os.urandom(size)
    env = s.seal(data, password=pw_for(code), aad=b"context")
    assert len(env) == size + envelope_overhead(code)
    assert r.open(env, password=pw_for(code), aad=b"context") == data


@pytest.mark.parametrize("code", list(ALL))
def test_default_cost_round_trip(code, keypair, psk):
    s, r = pair(code, keypair, psk, fast=False)
    data = "non-ASCII payload: ü 漢字 🔐".encode("utf-8")
    assert r.open(s.seal(data, password=pw_for(code)), password=pw_for(code)) == data


def test_exact_wire_overhead():
    assert {c: envelope_overhead(c) for c in ALL} == {"A": 45, "B": 45, "C": 1681, "D": 1681, "E": 1702, "F": 1702}
    assert {c: header_length(c) + TAG_BYTES for c in ALL} == {c: envelope_overhead(c) for c in ALL}


@pytest.mark.parametrize("code", list(ALL))
def test_wrong_aad_and_missing_aad_rejected(code, keypair, psk):
    s, r = pair(code, keypair, psk)
    env = s.seal(b"m" * 100, password=pw_for(code), aad=b"context")
    for bad in (b"", b"contexT", b"context\x00"):
        with pytest.raises(DecryptionError):
            r.open(env, password=pw_for(code), aad=bad)


@pytest.mark.parametrize("code", "CDEF")
def test_wrong_recipient_key_rejected(code, keypair, other_keypair, psk):
    s, _ = pair(code, keypair, psk)
    _, wrong = pair(code, other_keypair, psk)
    env = s.seal(b"secret", password=pw_for(code))
    with pytest.raises(DecryptionError):
        wrong.open(env, password=pw_for(code))


@pytest.mark.parametrize("code", "AB")
def test_wrong_psk_rejected(code, keypair, psk):
    s, _ = pair(code, keypair, psk)
    _, wrong = pair(code, keypair, os.urandom(32))
    with pytest.raises(DecryptionError):
        wrong.open(s.seal(b"secret"))


@pytest.mark.parametrize("code", "DEF")
def test_wrong_password_rejected(code, keypair, psk):
    s, r = pair(code, keypair, psk)
    env = s.seal(b"secret", password=PW)
    for bad in (PW + " ", PW[:-1], "Correct horse ⚡ battery staple"):
        with pytest.raises(DecryptionError):
            r.open(env, password=bad)
    with pytest.raises(ValueError):
        r.open(env)  # password required


def test_unicode_password_semantics(keypair, psk):
    composed, decomposed = "café", "café"
    for code in "EF":
        s, r = pair(code, keypair, psk)
        assert r.open(s.seal(b"x", password=composed), password=decomposed) == b"x"
    s, r = pair("D", keypair, psk)  # configuration D: raw UTF-8, no normalization
    with pytest.raises(DecryptionError):
        r.open(s.seal(b"x", password=composed), password=decomposed)


def test_legacy_suite_accepts_a_different_colliding_password(keypair, psk):
    """The five-byte bottleneck made concrete: suite D opens with a different passphrase."""
    p1, p2 = LEGACY_KEY_COLLISION
    s, r = pair("D", keypair, psk)
    assert r.open(s.seal(b"payload", password=p1), password=p2) == b"payload"
    for code in "EF":
        s, r = pair(code, keypair, psk)
        with pytest.raises(DecryptionError):
            r.open(s.seal(b"payload", password=p1), password=p2)


def _expect_rejection(r, env, code):
    try:
        out = r.open(env, password=pw_for(code), aad=b"ctx")
    except (DecryptionError, EnvelopeFormatError):
        return
    pytest.fail(f"modified envelope accepted (returned {len(out)} bytes)")


@pytest.mark.parametrize("code", list(ALL))
def test_every_header_byte_flip_rejected(code, keypair, psk):
    s, r = pair(code, keypair, psk)
    env = s.seal(b"p" * 64, password=pw_for(code), aad=b"ctx")
    hl = header_length(code)
    kem_ct = range(10, 10 + 1568) if SUITES[code].kem else range(0)  # sample the KEM ciphertext
    positions = [i for i in range(hl) if i not in kem_ct or i % 97 == 0 or i in (10, 1577)]
    for i in positions:
        for bit in (0x01, 0x80):
            bad = bytearray(env)
            bad[i] ^= bit
            _expect_rejection(r, bytes(bad), code)


@pytest.mark.parametrize("code", list(ALL))
def test_ciphertext_and_tag_modifications_rejected(code, keypair, psk):
    s, r = pair(code, keypair, psk)
    env = s.seal(b"q" * 256, password=pw_for(code), aad=b"ctx")
    hl = header_length(code)
    for i in (hl, hl + 1, hl + 128, len(env) - TAG_BYTES - 1, len(env) - TAG_BYTES, len(env) - 1):
        bad = bytearray(env)
        bad[i] ^= 0x01
        _expect_rejection(r, bytes(bad), code)


@pytest.mark.parametrize("code", list(ALL))
def test_truncation_and_trailing_data_rejected(code, keypair, psk):
    s, r = pair(code, keypair, psk)
    env = s.seal(b"t" * 40, password=pw_for(code), aad=b"ctx")
    for n in sorted({0, 1, 3, 4, 7, 8, 9, 12, header_length(code) - 1, header_length(code),
                     len(env) - TAG_BYTES, len(env) - 1}):
        with pytest.raises(EnvelopeFormatError):
            r.open(env[:n], password=pw_for(code), aad=b"ctx")
    with pytest.raises(EnvelopeFormatError):
        r.open(env + b"\x00", password=pw_for(code), aad=b"ctx")


def test_version_suite_flags_and_downgrade(keypair, psk):
    s, _ = pair("F", keypair, psk)
    env = bytearray(s.seal(b"x", password=PW))
    _, r = pair("F", keypair, psk)
    bad = bytearray(env); bad[0:4] = b"CCv1"
    with pytest.raises(UnsupportedVersionError):
        r.open(bytes(bad), password=PW)
    bad = bytearray(env); bad[4] = 3
    with pytest.raises(UnsupportedVersionError):
        r.open(bytes(bad), password=PW)
    bad = bytearray(env); bad[5] = 0x7F
    with pytest.raises(UnsupportedSuiteError):
        r.open(bytes(bad), password=PW)
    bad = bytearray(env); bad[6] = 1
    with pytest.raises(EnvelopeFormatError):
        r.open(bytes(bad), password=PW)
    # an attacker who knows only the public key can always produce a suite-C envelope:
    c_env = Sender("C", recipient_public_key=keypair.public_key).seal(b"forged")
    with pytest.raises(UnsupportedSuiteError):
        r.open(c_env, password=PW)
    with pytest.raises(ValueError):
        Receiver(recipient_secret_key=keypair.secret_key)  # accepted_suites is mandatory


def test_kdf_parameters_rejected_before_work(keypair, psk):
    s, r = pair("E", keypair, psk, fast=False)
    env = bytearray(s.seal(b"x", password=PW))
    off = 8 + 2 + 1568 + 1 + 32 + 1 + 32  # after the key commitment
    assert env[off] == kdf.KDF_ID_SCRYPT and env[off + 1] == 14
    for field, value, err in ((1, 30, UnsupportedParameterError), (1, 13, UnsupportedParameterError),
                              (2, 255, UnsupportedParameterError), (3, 64, UnsupportedParameterError),
                              (0, 2, UnsupportedParameterError)):
        bad = bytearray(env)
        bad[off + field] = value
        t0 = time.perf_counter()
        with pytest.raises(err):
            r.open(bytes(bad), password=PW)
        assert time.perf_counter() - t0 < 0.02, "rejection must precede any password-KDF work"


def test_oversized_length_field_rejected(keypair, psk):
    s, r = pair("A", keypair, psk)
    env = bytearray(s.seal(b"x"))
    hl = header_length("A")
    env[hl - 8 : hl] = struct.pack(">Q", (1 << 40))
    with pytest.raises(LengthLimitError):
        r.open(bytes(env))
    env[hl - 8 : hl] = struct.pack(">Q", 3)
    with pytest.raises(EnvelopeFormatError):
        r.open(bytes(env))


def test_no_secret_material_on_the_wire(keypair, psk):
    backend = get_kem_backend("openssl")
    enc = backend.encapsulate(backend.load_public_key(keypair.public_key))
    for code in "CDEF":
        s = Sender(code, recipient_public_key=keypair.public_key, scrypt=FAST_SCRYPT, scrypt_policy=FAST_POLICY)
        pw_salt = os.urandom(16)
        env = _fixtures.seal_fixed(s, b"\x00" * 64, nonce=os.urandom(12), hkdf_salt=os.urandom(32),
                                   pw_salt=pw_salt, encapsulation=enc, password=pw_for(code))
        assert keypair.secret_key not in env
        assert enc.shared_secret not in env
        assert kdf.kem_input(enc.shared_secret) not in env
        if SUITES[code].uses_scrypt:
            pw_key = kdf.derive_password_key(kdf.encode_password(PW), pw_salt, FAST_SCRYPT)
            assert pw_key not in env
        assert PW.encode() not in env
    r = Receiver(recipient_secret_key=keypair.secret_key, accepted_suites={"C"})
    assert keypair.secret_key.hex() not in repr(r) and "redacted" in repr(r)


def test_password_session_cached_workload(keypair, psk):
    for code in "DEF":
        s, r = pair(code, keypair, psk)
        sess = PasswordSession(code, PW, scrypt=FAST_SCRYPT)
        e1 = s.seal(b"one", session=sess)
        e2 = s.seal(b"two", session=sess)
        h1, _, _ = parse_envelope(e1, frozenset({code}), FAST_POLICY)
        h2, _, _ = parse_envelope(e2, frozenset({code}), FAST_POLICY)
        assert h1.pw_salt == h2.pw_salt and h1.hkdf_salt != h2.hkdf_salt and h1.nonce != h2.nonce
        rsess = PasswordSession(code, PW, scrypt=FAST_SCRYPT, pw_salt=sess.pw_salt)
        assert r.open(e1, session=rsess) == b"one"
        assert r.open(e2, password=PW) == b"two"  # a receiver without the session derives normally
        wrong = PasswordSession(code, PW + "!", scrypt=FAST_SCRYPT, pw_salt=sess.pw_salt)
        with pytest.raises(DecryptionError):
            r.open(e1, session=wrong)
        assert "redacted" in repr(sess)


def test_password_and_session_are_mutually_exclusive(keypair, psk):
    for code in "DEF":
        s, r = pair(code, keypair, psk)
        sess = PasswordSession(code, PW, scrypt=FAST_SCRYPT)
        env = s.seal(b"m", session=sess)
        with pytest.raises(ValueError):
            s.seal(b"m", password=PW, session=sess)
        rsess = PasswordSession(code, PW, scrypt=FAST_SCRYPT, pw_salt=sess.pw_salt)
        with pytest.raises(ValueError):  # a wrong typed password must never be masked by a cached session
            r.open(env, password="WRONG", session=rsess)


def test_session_parameters_must_match_sender(keypair, psk):
    s, _ = pair("E", keypair, psk)  # sender uses FAST_SCRYPT
    other = PasswordSession("E", PW, scrypt=kdf.ScryptParams(11))
    with pytest.raises(ValueError):
        s.seal(b"m", session=other)
    default_sender = Sender("E", recipient_public_key=keypair.public_key, scrypt=kdf.ScryptParams(14))
    with pytest.raises(ValueError):  # a weak session cannot smuggle its parameters into the header
        default_sender.seal(b"m", session=PasswordSession("E", PW, scrypt=kdf.ScryptParams(1, 1, 1)))
    with pytest.raises(UnsupportedParameterError):
        Sender("E", recipient_public_key=keypair.public_key, scrypt=kdf.ScryptParams(1, 1, 1))


def test_buffer_plaintexts_are_measured_in_bytes(keypair, psk):
    import numpy as np

    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    for code in "AC":
        s, r = pair(code, keypair, psk)
        env = s.seal(memoryview(img))
        assert len(env) == img.nbytes + envelope_overhead(code)
        assert r.open(env) == img.tobytes()
        env = s.seal(img)  # numpy arrays expose the buffer protocol
        assert r.open(env) == img.tobytes()
    with pytest.raises(TypeError):
        pair("A", keypair, psk)[0].seal(img[:, ::2])  # non-contiguous view


def test_encapsulation_repr_redacts_shared_secret(keypair):
    b = get_kem_backend("openssl")
    enc = b.encapsulate(b.load_public_key(keypair.public_key))
    assert enc.shared_secret.hex() not in repr(enc) and "redacted" in repr(enc)


def test_commitment_is_checked_before_decryption(keypair, psk):
    for code in "CDEF":
        s, r = pair(code, keypair, psk)
        env = bytearray(s.seal(b"x" * 32, password=pw_for(code)))
        commit_at = 8 + 2 + 1568 + 1 + 32 + 1
        assert env[commit_at - 1] == kdf.COMMITMENT_BYTES
        env[commit_at] ^= 0x01
        with pytest.raises(DecryptionError):
            r.open(bytes(env), password=pw_for(code))


def test_fresh_randomness_per_message(keypair, psk):
    for code in ALL:
        s, _ = pair(code, keypair, psk)
        e1, e2 = (s.seal(b"same", password=pw_for(code)) for _ in range(2))
        h1, _, _ = parse_envelope(e1, frozenset({code}), FAST_POLICY)
        h2, _, _ = parse_envelope(e2, frozenset({code}), FAST_POLICY)
        assert h1.nonce != h2.nonce and e1 != e2
        if SUITES[code].kem:
            assert h1.kem_ciphertext != h2.kem_ciphertext and h1.hkdf_salt != h2.hkdf_salt
        if SUITES[code].uses_scrypt:
            assert h1.pw_salt != h2.pw_salt


def test_psk_message_limit(keypair, psk, monkeypatch):
    import cryptochaos.protocol as p
    monkeypatch.setattr(p, "MAX_MESSAGES_PER_PSK", 3)
    s, _ = pair("A", keypair, psk)
    for _ in range(3):
        s.seal(b"x")
    with pytest.raises(LengthLimitError):
        s.seal(b"x")


def _golden(code):
    ref = get_kem_backend("kyber-py")
    kp = ref.keypair_from_seed(bytes(range(64)))
    enc = ref.encapsulate_deterministic(kp.public_key, bytes(range(100, 132)))
    s = Sender(code, recipient_public_key=kp.public_key, kem_backend="kyber-py", psk=bytes(range(32)))
    env = _fixtures.seal_fixed(s, b"CryptoChaos-v2 golden vector", nonce=bytes(range(200, 212)),
                               hkdf_salt=bytes(range(32, 64)), pw_salt=bytes(range(64, 80)),
                               encapsulation=enc, password="golden-password" if SUITES[code].password else None,
                               aad=b"golden-aad")
    return kp, env


# SHA-256 of the deterministic golden envelopes (any change here is a wire-format change).
GOLDEN_SHA256 = {
    "A": "ab9d1c056f05d5034a4cc3e3480556de0a82594467f2244e6a08af33db35abb1",  # 73 bytes
    "B": "13d7d35bb2b0f93d7a57c6846d60f2127f52221ce60c068ed133d110569c3dde",  # 73 bytes
    "C": "b3782e5b00a7e22f1a47a09a085e6e30fb401f4b0a7e69bdcd8f8777b657c5ef",  # 1709 bytes
    "D": "ae5d5028d0b9576351554ebfe0e7a80e0279e5ae60493d6c959495b579051084",  # 1709 bytes
    "E": "c41b4df500ed5900e3f917ece159227f56b43ea20077129154d1238fd238cf33",  # 1730 bytes
    "F": "a63704bc77be68e96e57016415d50b67efbb5ce3e6324000a20b347d98b61636",  # 1730 bytes
}


@pytest.mark.parametrize("code", list(ALL))
def test_golden_envelopes(code):
    kp, env = _golden(code)
    assert hashlib.sha256(env).hexdigest() == GOLDEN_SHA256[code]
    r = Receiver(recipient_secret_key=kp.secret_key, kem_backend="kyber-py", psk=bytes(range(32)),
                 accepted_suites={code})
    assert r.open(env, password=pw_for(code) and "golden-password", aad=b"golden-aad") == (
        b"CryptoChaos-v2 golden vector")
