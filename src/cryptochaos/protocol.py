"""CryptoChaos-v2 authenticated envelope (protocol version 2).

Wire format (all integers big-endian)::

    magic "CCv2" (4) | version u8 = 2 | suite u8 | flags u8 = 0 | reserved u8 = 0
    KEM suites C-F:   kem_ct_len u16 = 1568 | kem_ct | hkdf_salt_len u8 = 32 | hkdf_salt
                      | commit_len u8 = 32 | key_commitment
    scrypt suites E-F: kdf_id u8 = 1 | log2_n u8 | r u8 | p u8 | pw_salt_len u8 = 16 | pw_salt
    nonce_len u8 = 12 | nonce | ct_len u64 | ciphertext || 16-byte tag   (ct_len covers both)

The complete header and the caller's associated data are authenticated:
``AAD = "CryptoChaos-v2/AAD" || u32(len(header)) || header || u64(len(aad)) || aad``.
In KEM suites HKDF also yields a 32-byte key commitment that the receiver checks (in constant time)
before decryption, so no envelope can verify under more than one derived key.

Roles are separated. A ``Sender`` holds only the recipient's public key (or, for the symmetric
reference suites A/B, the pre-established key). A ``Receiver`` holds the recipient's secret key.
An envelope contains only public transport data and ciphertext. No API returns or serializes a
secret key or shared secret. Nonces and salts come from the operating system CSPRNG
(``os.urandom``); deterministic values are only used by ``cryptochaos._fixtures`` in tests.
"""
from __future__ import annotations

import hmac
import os
import struct
from dataclasses import dataclass
from typing import Callable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from . import kdf
from .errors import (
    DecryptionError,
    EnvelopeFormatError,
    LengthLimitError,
    UnsupportedParameterError,
    UnsupportedSuiteError,
    UnsupportedVersionError,
)
from .kem import CT_BYTES, DEFAULT_KEM_BACKEND, Encapsulation, KemBackend, get_kem_backend
from .maps import legacy_chaotic_key, v2_chaos_value

MAGIC = b"CCv2"
VERSION = 2
NONCE_BYTES = 12
TAG_BYTES = 16
PSK_BYTES = 32
AAD_LABEL = b"CryptoChaos-v2/AAD"
MAX_PLAINTEXT_BYTES = 1 << 30  # implementation limit for the one-shot API (1 GiB)
MAX_AAD_BYTES = 1 << 20
# Random 96-bit nonces under one pre-established key: SP 800-38D section 8.3 limits a key to 2^32
# invocations; applied to both symmetric reference suites. KEM suites derive a fresh key per message.
MAX_MESSAGES_PER_PSK = 1 << 32

AEAD_AES_GCM = "AES-256-GCM"
AEAD_CHACHA = "ChaCha20-Poly1305"


@dataclass(frozen=True)
class Suite:
    suite_id: int
    code: str
    name: str
    kem: bool
    aead: str
    password: str | None  # None | "legacy-chaos" | "scrypt" | "scrypt+chaos"

    @property
    def uses_scrypt(self) -> bool:
        return self.password in ("scrypt", "scrypt+chaos")


SUITES: dict[str, Suite] = {
    "A": Suite(0x01, "A", "AES-256-GCM, pre-established key", False, AEAD_AES_GCM, None),
    "B": Suite(0x02, "B", "ChaCha20-Poly1305, pre-established key", False, AEAD_CHACHA, None),
    "C": Suite(0x10, "C", "ML-KEM-1024 + HKDF-SHA256 + AES-256-GCM", True, AEAD_AES_GCM, None),
    "D": Suite(0x11, "D", "C + legacy chaotic password diversification", True, AEAD_AES_GCM, "legacy-chaos"),
    "E": Suite(0x12, "E", "C + scrypt password key", True, AEAD_AES_GCM, "scrypt"),
    "F": Suite(0x13, "F", "C + scrypt password key + v2 chaotic value", True, AEAD_AES_GCM, "scrypt+chaos"),
}
SUITES_BY_ID: dict[int, Suite] = {s.suite_id: s for s in SUITES.values()}


def get_suite(suite: str | Suite) -> Suite:
    if isinstance(suite, Suite):
        return suite
    try:
        return SUITES[suite]
    except KeyError:
        raise UnsupportedSuiteError(f"unknown suite {suite!r}") from None


@dataclass(frozen=True)
class Header:
    suite: Suite
    nonce: bytes
    ct_len: int
    kem_ciphertext: bytes | None = None
    hkdf_salt: bytes | None = None
    scrypt: kdf.ScryptParams | None = None
    pw_salt: bytes | None = None
    commitment: bytes | None = None

    def encode(self) -> bytes:
        s = self.suite
        out = bytearray(MAGIC)
        out += struct.pack(">BBBB", VERSION, s.suite_id, 0, 0)
        if s.kem:
            out += struct.pack(">H", len(self.kem_ciphertext)) + self.kem_ciphertext
            out += struct.pack(">B", len(self.hkdf_salt)) + self.hkdf_salt
            out += struct.pack(">B", len(self.commitment)) + self.commitment
        if s.uses_scrypt:
            p = self.scrypt
            out += struct.pack(">BBBB", kdf.KDF_ID_SCRYPT, p.log2_n, p.r, p.p)
            out += struct.pack(">B", len(self.pw_salt)) + self.pw_salt
        out += struct.pack(">B", len(self.nonce)) + self.nonce
        out += struct.pack(">Q", self.ct_len)
        return bytes(out)


def header_length(suite: str | Suite) -> int:
    s = get_suite(suite)
    n = 8
    if s.kem:
        n += 2 + CT_BYTES + 1 + kdf.HKDF_SALT_BYTES + 1 + kdf.COMMITMENT_BYTES
    if s.uses_scrypt:
        n += 4 + 1 + kdf.PW_SALT_BYTES
    return n + 1 + NONCE_BYTES + 8


def envelope_overhead(suite: str | Suite) -> int:
    """Exact bytes added to the plaintext on the wire (header plus authentication tag)."""
    return header_length(suite) + TAG_BYTES


def build_aad(header_bytes: bytes, aad: bytes) -> bytes:
    return (AAD_LABEL + struct.pack(">I", len(header_bytes)) + header_bytes
            + struct.pack(">Q", len(aad)) + aad)


class _Reader:
    def __init__(self, data: memoryview) -> None:
        self.d = data
        self.i = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.i + n > len(self.d):
            raise EnvelopeFormatError("envelope truncated")
        v = bytes(self.d[self.i : self.i + n])
        self.i += n
        return v

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.take(2))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]

    def exact(self, expected: int, what: str) -> bytes:
        n = self.u8() if expected < 256 else self.u16()
        if n != expected:
            raise EnvelopeFormatError(f"{what} length {n} != {expected}")
        return self.take(n)


def parse_envelope(
    envelope: bytes,
    accepted_suites: frozenset[str] | None = None,
    scrypt_policy: kdf.ScryptPolicy = kdf.DEFAULT_SCRYPT_POLICY,
) -> tuple[Header, bytes, memoryview]:
    """Strictly parse an envelope. Returns (header, header_bytes, aead_payload_view).

    Rejects unknown magic/version/suite, suites the receiver does not accept, non-zero
    flags, wrong field lengths, KDF parameters outside the allow-list (before any KDF work),
    lengths beyond the implementation limit, truncation, and trailing data.
    """
    if not isinstance(envelope, (bytes, bytearray, memoryview)):
        raise TypeError("envelope must be bytes-like")
    view = memoryview(envelope)
    r = _Reader(view)
    if r.take(4) != MAGIC:
        raise UnsupportedVersionError("not a CryptoChaos-v2 envelope (bad magic)")
    version, suite_id, flags, reserved = struct.unpack(">BBBB", r.take(4))
    if version != VERSION:
        raise UnsupportedVersionError(f"unsupported protocol version {version}")
    suite = SUITES_BY_ID.get(suite_id)
    if suite is None:
        raise UnsupportedSuiteError(f"unknown suite id 0x{suite_id:02x}")
    if accepted_suites is not None and suite.code not in accepted_suites:
        raise UnsupportedSuiteError(f"suite {suite.code} not accepted by this receiver")
    if flags != 0 or reserved != 0:
        raise EnvelopeFormatError("non-zero reserved header fields")
    kem_ct = hkdf_salt = pw_salt = commitment = None
    params = None
    if suite.kem:
        kem_ct = r.exact(CT_BYTES, "KEM ciphertext")
        hkdf_salt = r.exact(kdf.HKDF_SALT_BYTES, "HKDF salt")
        commitment = r.exact(kdf.COMMITMENT_BYTES, "key commitment")
    if suite.uses_scrypt:
        kdf_id, log2_n, rr, pp = struct.unpack(">BBBB", r.take(4))
        if kdf_id != kdf.KDF_ID_SCRYPT:
            raise UnsupportedParameterError(f"unsupported password KDF id {kdf_id}")
        params = kdf.ScryptParams(log2_n, rr, pp)
        scrypt_policy.check(params)
        pw_salt = r.exact(kdf.PW_SALT_BYTES, "password salt")
    nonce = r.exact(NONCE_BYTES, "nonce")
    ct_len = r.u64()
    if ct_len < TAG_BYTES:
        raise EnvelopeFormatError("ciphertext shorter than the authentication tag")
    if ct_len > MAX_PLAINTEXT_BYTES + TAG_BYTES:
        raise LengthLimitError("ciphertext exceeds the implementation limit")
    header_end = r.i
    remaining = len(view) - header_end
    if remaining < ct_len:
        raise EnvelopeFormatError("envelope truncated")
    if remaining > ct_len:
        raise EnvelopeFormatError("trailing data after ciphertext")
    header = Header(suite, nonce, ct_len, kem_ct, hkdf_salt, params, pw_salt, commitment)
    return header, bytes(view[:header_end]), view[header_end:]


def _aead(suite: Suite, key: bytes):
    return AESGCM(key) if suite.aead == AEAD_AES_GCM else ChaCha20Poly1305(key)


def _as_bytes_view(data) -> bytes | memoryview:
    """Byte-accurate view of a bytes-like plaintext (lengths are always in bytes)."""
    if isinstance(data, bytes):
        return data
    if not isinstance(data, (bytearray, memoryview)) and not hasattr(data, "__buffer__"):
        raise TypeError("plaintext must be bytes-like")
    view = memoryview(data)
    if not view.c_contiguous:
        raise TypeError("plaintext buffer must be C-contiguous")
    return view.cast("B") if view.format != "B" or view.ndim != 1 else view


def _check_aad(aad: bytes) -> bytes:
    if not isinstance(aad, (bytes, bytearray)):
        raise TypeError("aad must be bytes")
    if len(aad) > MAX_AAD_BYTES:
        raise LengthLimitError("associated data exceeds the implementation limit")
    return bytes(aad)


class PasswordSession:
    """Password material derived once and reused for many messages (the "cached" workload).

    For suites E/F the session fixes one password salt and derives ``pw_key`` (and the F chaos
    value) once; every envelope in the session carries the same salt and parameters, so a
    receiver holding the same session skips scrypt. For suite D the session caches the legacy
    chaotic key. Caching trades per-message KDF cost for linkability of envelopes within the
    session; it is a declared workload, never a hidden optimization.
    """

    def __init__(self, suite: str | Suite, password: str | bytes,
                 scrypt: kdf.ScryptParams = kdf.DEFAULT_SCRYPT, pw_salt: bytes | None = None) -> None:
        self.suite = get_suite(suite)
        self.scrypt = scrypt
        self.pw_salt = None
        self.pw_key = None
        self.chaos_value = None
        self.legacy_value = None
        if self.suite.password == "legacy-chaos":
            if not isinstance(password, str):
                raise TypeError("suite D (legacy) takes a str passphrase")
            self.legacy_value = legacy_chaotic_key(password)
        elif self.suite.uses_scrypt:
            self.pw_salt = pw_salt if pw_salt is not None else os.urandom(kdf.PW_SALT_BYTES)
            self.pw_key = kdf.derive_password_key(kdf.encode_password(password), self.pw_salt, scrypt)
            if self.suite.password == "scrypt+chaos":
                self.chaos_value = v2_chaos_value(self.pw_key)
        else:
            raise ValueError(f"suite {self.suite.code} does not use a password")

    def matches(self, header: Header) -> bool:
        if self.suite.code != header.suite.code:
            return False
        if self.suite.uses_scrypt:
            return header.pw_salt == self.pw_salt and header.scrypt == self.scrypt
        return True

    def __repr__(self) -> str:
        return f"PasswordSession(suite={self.suite.code!r}, {self.scrypt.label()}, <secret material redacted>)"


def _password_fields(suite: Suite, password: str | bytes | None, header_scrypt: kdf.ScryptParams | None,
                     pw_salt: bytes | None, session: PasswordSession | None) -> list[tuple[int, bytes]]:
    """IKM fields contributed by the password pathway of suites D, E, F."""
    if suite.password is None:
        return []
    if session is not None and session.matches(Header(suite, b"", 0, scrypt=header_scrypt, pw_salt=pw_salt)):
        if suite.password == "legacy-chaos":
            return [(kdf.FIELD_LEGACY_CHAOS, session.legacy_value)]
        fields = [(kdf.FIELD_PW_KEY, session.pw_key)]
        if suite.password == "scrypt+chaos":
            fields.append((kdf.FIELD_CHAOS_V2, session.chaos_value))
        return fields
    if password is None:
        raise ValueError(f"suite {suite.code} requires a password")
    if suite.password == "legacy-chaos":
        if not isinstance(password, str):
            raise TypeError("suite D (legacy) takes a str passphrase")
        return [(kdf.FIELD_LEGACY_CHAOS, legacy_chaotic_key(password))]
    pw_key = kdf.derive_password_key(kdf.encode_password(password), pw_salt, header_scrypt)
    fields = [(kdf.FIELD_PW_KEY, pw_key)]
    if suite.password == "scrypt+chaos":
        fields.append((kdf.FIELD_CHAOS_V2, v2_chaos_value(pw_key)))
    return fields


def _resolve_backend(kem_backend: str | KemBackend) -> KemBackend:
    return kem_backend if isinstance(kem_backend, KemBackend) else get_kem_backend(kem_backend)


class Sender:
    """Seals envelopes. Holds only public material (or the pre-established key for suites A/B)."""

    def __init__(self, suite: str | Suite, *, recipient_public_key: bytes | None = None,
                 kem_backend: str | KemBackend = DEFAULT_KEM_BACKEND, psk: bytes | None = None,
                 scrypt: kdf.ScryptParams = kdf.DEFAULT_SCRYPT,
                 scrypt_policy: kdf.ScryptPolicy = kdf.DEFAULT_SCRYPT_POLICY) -> None:
        self.suite = get_suite(suite)
        self.scrypt = scrypt
        self.scrypt_policy = scrypt_policy
        self._psk = None
        self._public = None
        self._backend = None
        self._psk_messages = 0
        if self.suite.kem:
            if recipient_public_key is None:
                raise ValueError(f"suite {self.suite.code} requires recipient_public_key")
            self._backend = _resolve_backend(kem_backend)
            self._public = self._backend.load_public_key(recipient_public_key)
        else:
            if psk is None or len(psk) != PSK_BYTES:
                raise ValueError(f"suite {self.suite.code} requires a {PSK_BYTES}-byte pre-established key")
            self._psk = bytes(psk)
        if self.suite.uses_scrypt:
            scrypt_policy.check(scrypt)

    def seal(self, plaintext: bytes, *, password: str | bytes | None = None, aad: bytes = b"",
             session: PasswordSession | None = None) -> bytearray:
        """Seal ``plaintext``. Returns the envelope as a ``bytearray``.

        The AEAD output is written directly behind the header in one buffer (no copy of the
        ciphertext), which is why the result is a ``bytearray`` rather than ``bytes``.
        """
        return self._seal(plaintext, password, aad, session, os.urandom, None)

    def _seal(self, plaintext: bytes, password, aad: bytes, session: PasswordSession | None,
              rand: Callable[[int], bytes], encapsulation: Encapsulation | None) -> bytearray:
        if password is not None and session is not None:
            raise ValueError("pass either password or session, not both")
        if self.suite.password is None and (password is not None or session is not None):
            raise ValueError(f"suite {self.suite.code} does not bind a password")
        plaintext = _as_bytes_view(plaintext)
        if len(plaintext) > MAX_PLAINTEXT_BYTES:
            raise LengthLimitError("plaintext exceeds the implementation limit")
        aad = _check_aad(aad)
        s = self.suite
        if s.kem:
            enc = encapsulation if encapsulation is not None else self._backend.encapsulate(self._public)
            hkdf_salt = rand(kdf.HKDF_SALT_BYTES)
            scrypt = pw_salt = None
            if s.uses_scrypt:
                if session is not None and session.suite.code == s.code:
                    if session.scrypt != self.scrypt:
                        raise ValueError("session scrypt parameters differ from this sender's parameters")
                    self.scrypt_policy.check(session.scrypt)
                    scrypt, pw_salt = session.scrypt, session.pw_salt
                else:
                    scrypt, pw_salt = self.scrypt, rand(kdf.PW_SALT_BYTES)
            fields = [(kdf.FIELD_KEM, kdf.kem_input(enc.shared_secret))]
            fields += _password_fields(s, password, scrypt, pw_salt, session)
            key, commitment = kdf.derive_key_and_commitment(VERSION, s.suite_id, fields, hkdf_salt)
            header = Header(s, rand(NONCE_BYTES), len(plaintext) + TAG_BYTES, enc.ciphertext, hkdf_salt,
                            scrypt, pw_salt, commitment)
        else:
            if self._psk_messages >= MAX_MESSAGES_PER_PSK:
                raise LengthLimitError("message limit for this pre-established key reached; rekey")
            self._psk_messages += 1
            key = self._psk
            header = Header(s, rand(NONCE_BYTES), len(plaintext) + TAG_BYTES)
        header_bytes = header.encode()
        hl = len(header_bytes)
        out = bytearray(hl + len(plaintext) + TAG_BYTES)
        out[:hl] = header_bytes
        _aead(s, key).encrypt_into(header.nonce, plaintext, build_aad(header_bytes, aad), memoryview(out)[hl:])
        return out


class Receiver:
    """Opens envelopes with the recipient's secret key (or the pre-established key for A/B)."""

    def __init__(self, *, recipient_secret_key: bytes | None = None,
                 kem_backend: str | KemBackend = DEFAULT_KEM_BACKEND, psk: bytes | None = None,
                 accepted_suites: set[str] | frozenset[str] | None = None,
                 scrypt_policy: kdf.ScryptPolicy = kdf.DEFAULT_SCRYPT_POLICY) -> None:
        if accepted_suites is None:
            raise ValueError("accepted_suites must be given explicitly (prevents suite downgrade)")
        self.accepted = frozenset(accepted_suites)
        for code in self.accepted:
            get_suite(code)
        self.scrypt_policy = scrypt_policy
        self._secret = None
        self._backend = None
        self._psk = bytes(psk) if psk is not None else None
        if self._psk is not None and len(self._psk) != PSK_BYTES:
            raise ValueError(f"pre-established key must be {PSK_BYTES} bytes")
        if any(SUITES[c].kem for c in self.accepted):
            if recipient_secret_key is None:
                raise ValueError("KEM suites require recipient_secret_key")
            self._backend = _resolve_backend(kem_backend)
            self._secret = self._backend.load_secret_key(recipient_secret_key)
        if any(not SUITES[c].kem for c in self.accepted) and self._psk is None:
            raise ValueError("suites A/B require psk")

    def __repr__(self) -> str:
        return f"Receiver(accepted={sorted(self.accepted)}, <secret material redacted>)"

    def open(self, envelope: bytes, *, password: str | bytes | None = None, aad: bytes = b"",
             session: PasswordSession | None = None) -> bytes:
        if password is not None and session is not None:
            raise ValueError("pass either password or session, not both")
        header, header_bytes, payload = parse_envelope(envelope, self.accepted, self.scrypt_policy)
        aad = _check_aad(aad)
        s = header.suite
        if s.password is None and (password is not None or session is not None):
            # A caller that supplies a password must never accept an envelope that does not bind it
            # (for example a suite-C envelope forged from the public key, offered to a receiver that
            # accepts both C and E).
            raise UnsupportedSuiteError(f"suite {s.code} does not bind a password")
        if s.kem:
            ss = self._backend.decapsulate(self._secret, header.kem_ciphertext)
            fields = [(kdf.FIELD_KEM, kdf.kem_input(ss))]
            fields += _password_fields(s, password, header.scrypt, header.pw_salt, session)
            key, commitment = kdf.derive_key_and_commitment(VERSION, s.suite_id, fields, header.hkdf_salt)
            if not hmac.compare_digest(commitment, header.commitment):
                raise DecryptionError("authentication failed")
        else:
            key = self._psk
        try:
            return _aead(s, key).decrypt(header.nonce, payload, build_aad(header_bytes, aad))
        except InvalidTag:
            raise DecryptionError("authentication failed") from None
