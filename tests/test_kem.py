"""ML-KEM-1024 adapters: sizes, tuple order, NIST ACVP known answers, cross-backend agreement.

Known-answer coverage differs by backend because of what each API exposes:

* kyber-py (reference): keyGen, encapsulation, decapsulation, and key-check vectors.
* openssl (cryptography): keyGen (seed -> encapsulation key); encapsulation-key check.
  No deterministic encapsulation and no expanded decapsulation-key import are exposed.
* pqcrypto: decapsulation vectors (expanded keys); encapsulation-key check through the adapter.
  No deterministic key generation or encapsulation are exposed.
"""
import os

import pytest

from cryptochaos.errors import EnvelopeFormatError
from cryptochaos.kem import (
    CT_BYTES,
    DK_BYTES,
    EK_BYTES,
    SS_BYTES,
    available_kem_backends,
    decapsulation_key_hash_ok,
    encapsulation_key_modulus_ok,
    get_kem_backend,
)
from conftest import acvp_group

BACKENDS = ["openssl", "pqcrypto", "kyber-py"]


def test_all_backends_available():
    assert set(BACKENDS) <= set(available_kem_backends())


@pytest.mark.parametrize("name", BACKENDS)
def test_sizes_and_normalized_order(name):
    b = get_kem_backend(name)
    kp = b.generate_keypair()
    assert len(kp.public_key) == EK_BYTES == 1568
    assert len(kp.secret_key) == (64 if b.secret_key_format == "seed64" else DK_BYTES)
    enc = b.encapsulate(b.load_public_key(kp.public_key))
    assert len(enc.shared_secret) == SS_BYTES == 32
    assert len(enc.ciphertext) == CT_BYTES == 1568
    assert b.decapsulate(b.load_secret_key(kp.secret_key), enc.ciphertext) == enc.shared_secret
    assert "redacted" in repr(kp) and kp.secret_key.hex() not in repr(kp)


def test_acvp_keygen_kyber_py_and_openssl():
    ref, ossl = get_kem_backend("kyber-py"), get_kem_backend("openssl")
    tests = acvp_group("keyGen")
    assert len(tests) == 25
    for t in tests:
        seed = bytes.fromhex(t["d"]) + bytes.fromhex(t["z"])
        kp = ref.keypair_from_seed(seed)
        assert kp.public_key.hex().upper() == t["ek"].upper()
        assert kp.secret_key.hex().upper() == t["dk"].upper()
        assert ossl.keypair_from_seed(seed).public_key.hex().upper() == t["ek"].upper()


def test_acvp_encapsulation_kyber_py_and_decaps_native():
    ref, pq = get_kem_backend("kyber-py"), get_kem_backend("pqcrypto")
    tests = acvp_group("encapDecap", "encapsulation")
    assert len(tests) == 25
    for t in tests:
        ek, dk, m = (bytes.fromhex(t[k]) for k in ("ek", "dk", "m"))
        enc = ref.encapsulate_deterministic(ek, m)
        assert enc.ciphertext.hex().upper() == t["c"].upper()
        assert enc.shared_secret.hex().upper() == t["k"].upper()
        # the same vectors as decapsulation known answers for the expanded-key backends
        for b in (ref, pq):
            assert b.decapsulate(b.load_secret_key(dk), enc.ciphertext).hex().upper() == t["k"].upper()


@pytest.mark.parametrize("name", ["kyber-py", "pqcrypto"])
def test_acvp_decapsulation_including_implicit_rejection(name):
    b = get_kem_backend(name)
    tests = acvp_group("encapDecap", "decapsulation")
    assert len(tests) == 10
    for t in tests:
        dk, c = bytes.fromhex(t["dk"]), bytes.fromhex(t["c"])
        assert b.decapsulate(b.load_secret_key(dk), c).hex().upper() == t["k"].upper(), t.get("reason")


def test_acvp_encapsulation_key_check():
    tests = acvp_group("encapDecap", "encapsulationKeyCheck")
    assert len(tests) == 10
    for name in BACKENDS:
        b = get_kem_backend(name)
        for t in tests:
            ek = bytes.fromhex(t["ek"])
            assert encapsulation_key_modulus_ok(ek) == t["testPassed"], t.get("reason")
            if t["testPassed"]:
                b.load_public_key(ek)
            else:
                with pytest.raises(ValueError):
                    b.load_public_key(ek)


def test_acvp_decapsulation_key_check():
    tests = acvp_group("encapDecap", "decapsulationKeyCheck")
    assert len(tests) == 10
    for t in tests:
        dk = bytes.fromhex(t["dk"])
        assert decapsulation_key_hash_ok(dk) == t["testPassed"], t.get("reason")
        for name in ("pqcrypto", "kyber-py"):
            b = get_kem_backend(name)
            if t["testPassed"]:
                b.load_secret_key(dk)
            else:
                with pytest.raises(ValueError):
                    b.load_secret_key(dk)


def _seeded_keys():
    seed = os.urandom(64)
    ref = get_kem_backend("kyber-py").keypair_from_seed(seed)
    secrets = {"openssl": seed, "pqcrypto": ref.secret_key, "kyber-py": ref.secret_key}
    return ref.public_key, secrets


@pytest.mark.parametrize("sender", BACKENDS)
@pytest.mark.parametrize("receiver", BACKENDS)
def test_cross_backend_round_trips(sender, receiver):
    ek, secrets = _seeded_keys()
    s, r = get_kem_backend(sender), get_kem_backend(receiver)
    enc = s.encapsulate(s.load_public_key(ek))
    assert r.decapsulate(r.load_secret_key(secrets[receiver]), enc.ciphertext) == enc.shared_secret


def test_implicit_rejection_consistent_across_backends():
    ek, secrets = _seeded_keys()
    ossl = get_kem_backend("openssl")
    enc = ossl.encapsulate(ossl.load_public_key(ek))
    for pos in (0, 777, CT_BYTES - 1):
        bad = bytearray(enc.ciphertext)
        bad[pos] ^= 0x01
        outs = {n: get_kem_backend(n).decapsulate(get_kem_backend(n).load_secret_key(secrets[n]), bytes(bad))
                for n in BACKENDS}
        assert len(set(outs.values())) == 1, "backends disagree on implicit rejection"
        assert next(iter(outs.values())) != enc.shared_secret


@pytest.mark.parametrize("name", BACKENDS)
def test_malformed_inputs_rejected(name):
    b = get_kem_backend(name)
    kp = b.generate_keypair()
    sk = b.load_secret_key(kp.secret_key)
    for bad_len in (0, CT_BYTES - 1, CT_BYTES + 1):
        with pytest.raises(EnvelopeFormatError):
            b.decapsulate(sk, bytes(bad_len))
    for bad_len in (0, EK_BYTES - 1, EK_BYTES + 1):
        with pytest.raises(ValueError):
            b.load_public_key(bytes(bad_len))
    bad_ek = bytearray(kp.public_key)
    bad_ek[0], bad_ek[1] = 0xFF, 0xFF  # first 12-bit coefficient = 4095 >= q
    with pytest.raises(ValueError):
        b.load_public_key(bytes(bad_ek))
    if b.secret_key_format == "expanded3168":
        bad_dk = bytearray(kp.secret_key)
        bad_dk[384 * 4 + 5] ^= 1  # corrupt the embedded encapsulation key -> hash check fails
        with pytest.raises(ValueError):
            b.load_secret_key(bytes(bad_dk))
    else:
        with pytest.raises(ValueError):
            b.load_secret_key(kp.secret_key[:-1])
