"""Deterministic sealing for golden vectors and diagnostic fixtures. TEST USE ONLY.

The public ``Sender.seal`` always draws salts and nonces from ``os.urandom`` and performs a
fresh KEM encapsulation. These helpers inject fixed values so tests can produce stable vectors
and so the key-sensitivity diagnostic can hold every input fixed except the one under study.
Reusing a (key, nonce) pair is catastrophic for AES-GCM and ChaCha20-Poly1305: fixtures built
with these helpers must only ever encrypt disposable synthetic or experiment data, never be used
on the normal protocol path.
"""
from __future__ import annotations

from .kem import Encapsulation
from .protocol import PasswordSession, Sender


def seal_fixed(sender: Sender, plaintext: bytes, *, nonce: bytes, hkdf_salt: bytes | None = None,
               pw_salt: bytes | None = None, encapsulation: Encapsulation | None = None,
               password: str | bytes | None = None, aad: bytes = b"",
               session: PasswordSession | None = None) -> bytes:
    values = {12: nonce}
    if hkdf_salt is not None:
        values[32] = hkdf_salt
    if pw_salt is not None:
        values[16] = pw_salt

    def rand(n: int) -> bytes:
        try:
            return values[n]
        except KeyError:
            raise ValueError(f"fixture needs an explicit {n}-byte value") from None

    if sender.suite.kem and encapsulation is None:
        raise ValueError("fixture needs an explicit KEM encapsulation for KEM suites")
    return sender._seal(plaintext, password, aad, session, rand, encapsulation)
