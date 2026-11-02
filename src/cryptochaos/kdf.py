"""Password derivation, IKM encoding, domain separation, and the CryptoChaos-v2 key schedule.

Key schedule for KEM suites (C-F)::

    k_kem       = BLAKE3(ss)                                   # 32 bytes; fixed-length encoding of ss
    pw_key      = scrypt(NFC-UTF8(password), pw_salt, N, r, p, 32)          # suites E, F
    chaos_value = v2_chaos_value(pw_key)                                     # suite F only
    legacy      = legacy_chaotic_key(password)                               # suite D only
    IKM         = "CryptoChaos-v2/IKM" || u8(version) || u8(suite) || u8(n_fields)
                  || for each field: u8(tag) || u16be(len) || value
    AEAD key    = HKDF-SHA256(IKM, salt=hkdf_salt, info=INFO(version, suite), L=32)

The full 32-byte ``pw_key`` is always an IKM field in E and F; the optional chaos value is an
additional field and never replaces it. Deterministic maps add no entropy: a password's
guessing resistance comes entirely from the password itself and the scrypt cost.
"""
from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass

import blake3
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .errors import UnsupportedParameterError

PROTOCOL_LABEL = b"CryptoChaos-v2"
IKM_LABEL = b"CryptoChaos-v2/IKM"
ROLE_PAYLOAD_KEY = b"payload-key"

PW_KEY_BYTES = 32
PW_SALT_BYTES = 16
HKDF_SALT_BYTES = 32
AEAD_KEY_BYTES = 32
COMMITMENT_BYTES = 32
MAX_PASSWORD_BYTES = 1024

# IKM field tags
FIELD_KEM = 0x01
FIELD_PW_KEY = 0x02
FIELD_CHAOS_V2 = 0x03
FIELD_LEGACY_CHAOS = 0x04

KDF_ID_SCRYPT = 0x01


@dataclass(frozen=True)
class ScryptParams:
    """scrypt cost parameters. Core memory is 128 * r * N bytes (16 MiB for N=2^14, r=8)."""

    log2_n: int = 14
    r: int = 8
    p: int = 1

    @property
    def n(self) -> int:
        return 1 << self.log2_n

    @property
    def memory_bytes(self) -> int:
        return 128 * self.r * self.n

    def label(self) -> str:
        return f"scrypt(N=2^{self.log2_n},r={self.r},p={self.p})"


@dataclass(frozen=True)
class ScryptPolicy:
    """Receiver-side allow-list, enforced before any password-KDF work.

    The default admits N = 2^14 .. 2^18 with r = 8 and p = 1 (16 MiB .. 256 MiB). An attacker
    controls the envelope header, so the receiver never accepts arbitrary cost parameters.
    """

    min_log2_n: int = 14
    max_log2_n: int = 18
    allowed_r: tuple[int, ...] = (8,)
    allowed_p: tuple[int, ...] = (1,)

    def check(self, params: ScryptParams) -> None:
        if not (self.min_log2_n <= params.log2_n <= self.max_log2_n):
            raise UnsupportedParameterError(f"scrypt log2(N)={params.log2_n} outside allow-list")
        if params.r not in self.allowed_r or params.p not in self.allowed_p:
            raise UnsupportedParameterError("scrypt r/p outside allow-list")


DEFAULT_SCRYPT = ScryptParams(14, 8, 1)
DEFAULT_SCRYPT_POLICY = ScryptPolicy()


def encode_password(password: str | bytes) -> bytes:
    """Passwords are the UTF-8 encoding of their Unicode NFC normalization.

    ``bytes`` are used verbatim (the caller has already chosen an encoding).
    """
    if isinstance(password, str):
        data = unicodedata.normalize("NFC", password).encode("utf-8")
    elif isinstance(password, (bytes, bytearray)):
        data = bytes(password)
    else:
        raise TypeError("password must be str or bytes")
    if not data:
        raise ValueError("password must be non-empty")
    if len(data) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password longer than {MAX_PASSWORD_BYTES} bytes")
    return data


def scrypt_raw(password: bytes, salt: bytes, n: int, r: int, p: int, length: int) -> bytes:
    """Unrestricted scrypt (RFC 7914) for test vectors; the protocol uses ``derive_password_key``."""
    return Scrypt(salt=salt, length=length, n=n, r=r, p=p).derive(password)


def derive_password_key(password: bytes, salt: bytes, params: ScryptParams) -> bytes:
    if len(salt) != PW_SALT_BYTES:
        raise ValueError(f"password salt must be {PW_SALT_BYTES} bytes")
    return scrypt_raw(password, salt, params.n, params.r, params.p, PW_KEY_BYTES)


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = AEAD_KEY_BYTES) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def kem_input(shared_secret: bytes) -> bytes:
    if len(shared_secret) != 32:
        raise ValueError("ML-KEM shared secret must be 32 bytes")
    return blake3.blake3(shared_secret).digest()


def encode_ikm(version: int, suite_id: int, fields: list[tuple[int, bytes]]) -> bytes:
    """Unambiguous TLV encoding with a label, version, suite, and field count."""
    if len(fields) > 255:
        raise ValueError("too many IKM fields")
    out = bytearray(IKM_LABEL)
    out += struct.pack(">BBB", version, suite_id, len(fields))
    seen = set()
    for tag, value in fields:
        if tag in seen:
            raise ValueError("duplicate IKM field")
        seen.add(tag)
        if len(value) > 0xFFFF:
            raise ValueError("IKM field too long")
        out += struct.pack(">BH", tag, len(value)) + value
    return bytes(out)


def hkdf_info(version: int, suite_id: int, role: bytes = ROLE_PAYLOAD_KEY) -> bytes:
    return PROTOCOL_LABEL + struct.pack(">BB", version, suite_id) + role


def derive_aead_key(version: int, suite_id: int, fields: list[tuple[int, bytes]], hkdf_salt: bytes) -> bytes:
    """The 32-byte AEAD key (first half of the 64-byte HKDF output)."""
    return derive_key_and_commitment(version, suite_id, fields, hkdf_salt)[0]


def derive_key_and_commitment(version: int, suite_id: int, fields: list[tuple[int, bytes]],
                              hkdf_salt: bytes) -> tuple[bytes, bytes]:
    """HKDF-SHA256 with L = 64: bytes 0..31 are the AEAD key, bytes 32..63 a key commitment.

    AES-GCM and ChaCha20-Poly1305 are not key-committing: one ciphertext can be made to verify
    under many keys, which would let an attacker test many passwords per envelope against a
    receiver (a partitioning oracle). The commitment is carried in the header and checked before
    decryption, so an envelope can open under at most one derived key.
    """
    if len(hkdf_salt) != HKDF_SALT_BYTES:
        raise ValueError(f"HKDF salt must be {HKDF_SALT_BYTES} bytes")
    okm = hkdf_sha256(encode_ikm(version, suite_id, fields), hkdf_salt, hkdf_info(version, suite_id), 64)
    return okm[:AEAD_KEY_BYTES], okm[AEAD_KEY_BYTES:]
