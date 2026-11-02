"""Hardening: password-binding guarantees, zero-copy envelopes, and a seeded mutation fuzzer."""
import os

import numpy as np
import pytest

from cryptochaos.errors import DecryptionError, EnvelopeFormatError, UnsupportedSuiteError
from cryptochaos.protocol import SUITES, Receiver, Sender, header_length
from conftest import FAST_POLICY, FAST_SCRYPT

PW = "fuzz pass phrase"


def endpoints(code, keypair, psk, accepted=None):
    s = Sender(code, recipient_public_key=keypair.public_key, psk=psk, scrypt=FAST_SCRYPT, scrypt_policy=FAST_POLICY)
    r = Receiver(recipient_secret_key=keypair.secret_key, psk=psk, accepted_suites=accepted or {code},
                 scrypt_policy=FAST_POLICY)
    return s, r


def test_password_is_never_silently_ignored(keypair, psk):
    # a receiver that accepts both C and E: a C envelope forged from the public key must not
    # satisfy a caller who believes it is checking a password
    _, r = endpoints("E", keypair, psk, accepted={"C", "E"})
    forged = Sender("C", recipient_public_key=keypair.public_key).seal(b"forged")
    with pytest.raises(UnsupportedSuiteError):
        r.open(forged, password=PW)
    assert r.open(forged) == b"forged"  # without a password the caller gets exactly what it asked for
    e_env = Sender("E", recipient_public_key=keypair.public_key, scrypt=FAST_SCRYPT,
                   scrypt_policy=FAST_POLICY).seal(b"bound", password=PW)
    assert r.open(e_env, password=PW) == b"bound"
    for code in "ABC":
        s, _ = endpoints(code, keypair, psk)
        with pytest.raises(ValueError):
            s.seal(b"x", password=PW)


def test_seal_returns_single_buffer_with_exact_layout(keypair, psk):
    for code in "ACF":
        s, r = endpoints(code, keypair, psk)
        data = os.urandom(4096)
        env = s.seal(data, password=PW if SUITES[code].password else None)
        assert isinstance(env, bytearray)
        assert len(env) == header_length(code) + len(data) + 16
        assert r.open(env, password=PW if SUITES[code].password else None) == data
        assert r.open(bytes(env), password=PW if SUITES[code].password else None) == data


MUTATIONS = ("flip", "truncate", "extend", "delete_span", "insert_span", "overwrite_span", "swap_bytes")


def mutate(env: bytes, rng: np.random.Generator) -> bytes:
    b = bytearray(env)
    kind = MUTATIONS[int(rng.integers(len(MUTATIONS)))]
    i = int(rng.integers(len(b)))
    span = int(rng.integers(1, 33))
    if kind == "flip":
        b[i] ^= 1 << int(rng.integers(8))
    elif kind == "truncate":
        b = b[:i]
    elif kind == "extend":
        b += rng.bytes(span)
    elif kind == "delete_span":
        del b[i:i + span]
    elif kind == "insert_span":
        b[i:i] = rng.bytes(span)
    elif kind == "overwrite_span":
        b[i:i + span] = rng.bytes(min(span, len(b) - i))
    else:
        j = int(rng.integers(len(b)))
        if b[i] == b[j]:
            b[i] ^= 0xFF
        else:
            b[i], b[j] = b[j], b[i]
    return bytes(b)


@pytest.mark.parametrize("code", list("ABCDEF"))
def test_mutation_fuzzer_never_releases_plaintext(code, keypair, psk):
    rng = np.random.Generator(np.random.PCG64([2026, ord(code)]))
    s, r = endpoints(code, keypair, psk)
    pw = PW if SUITES[code].password else None
    env = bytes(s.seal(os.urandom(200), password=pw, aad=b"ctx"))
    rejected = 0
    for _ in range(250):
        bad = mutate(env, rng)
        if bad == env:
            continue
        try:
            r.open(bad, password=pw, aad=b"ctx")
        except (DecryptionError, EnvelopeFormatError):
            rejected += 1
            continue
        pytest.fail(f"mutated envelope accepted for suite {code}")
    assert rejected > 200
