"""ML-KEM-1024 backend adapters behind one explicit interface.

Backends disagree on tuple order: ``cryptography`` returns ``(shared_secret,
ciphertext)`` from ``encapsulate()``, ``pqcrypto`` 1.0.0 returns
``(ciphertext, shared_secret)`` from ``encaps()``, and ``kyber-py`` 1.2.0
returns ``(shared_secret, ciphertext)``. Every adapter returns a named
``Encapsulation`` and checks lengths, so callers never depend on tuple order.

Backends:

* ``openssl``  -- pyca/cryptography (OpenSSL ML-KEM). Primary performance backend.
  Secret keys are the 64-byte FIPS 203 seed ``d || z``.
* ``pqcrypto`` -- pqcrypto 1.0.0 (Rust ``backbone-ml-kem``). Secondary native backend.
  Secret keys are the 3168-byte expanded decapsulation key.
* ``kyber-py`` -- kyber-py 1.2.0, a pure-Python FIPS 203 implementation. Reference only
  (known-answer tests and cross-checks); it is never the primary timed backend.

FIPS 203 input checks (encapsulation-key modulus check, decapsulation-key hash check,
ciphertext length check) are applied by these adapters for every backend.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .errors import BackendUnavailableError, EnvelopeFormatError

PARAMETER_SET = "ML-KEM-1024"
K = 4
Q = 3329
EK_BYTES = 384 * K + 32  # 1568
DK_BYTES = 768 * K + 96  # 3168
CT_BYTES = 32 * (11 * K + 5)  # 1568 (du=11, dv=5)
SS_BYTES = 32
SEED_BYTES = 64


class Encapsulation(NamedTuple):
    shared_secret: bytes
    ciphertext: bytes

    def __repr__(self) -> str:
        return f"Encapsulation(shared_secret=<redacted>, ciphertext=<{len(self.ciphertext)} bytes>)"


@dataclass(frozen=True, repr=False)
class RecipientKeyPair:
    """A recipient key pair. Only ``public_key`` may leave the recipient."""

    public_key: bytes
    secret_key: bytes
    backend: str
    secret_key_format: str

    def __repr__(self) -> str:
        return (
            f"RecipientKeyPair(backend={self.backend!r}, public_key=<{len(self.public_key)} bytes>, "
            f"secret_key=<redacted {self.secret_key_format}>)"
        )


def encapsulation_key_modulus_ok(ek: bytes) -> bool:
    """FIPS 203 section 7.2 modulus check: every 12-bit coefficient of ek's t-hat is < q."""
    if len(ek) != EK_BYTES:
        return False
    b = np.frombuffer(ek[: 384 * K], dtype=np.uint8).astype(np.uint16).reshape(-1, 3)
    c0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    c1 = (b[:, 1] >> 4) | (b[:, 2] << 4)
    return bool(np.all(c0 < Q) and np.all(c1 < Q))


def decapsulation_key_hash_ok(dk: bytes) -> bool:
    """FIPS 203 section 7.3 hash check on an expanded decapsulation key."""
    if len(dk) != DK_BYTES:
        return False
    ek = dk[384 * K : 768 * K + 32]
    h = dk[768 * K + 32 : 768 * K + 64]
    return hashlib.sha3_256(ek).digest() == h


def _check_public_key(public_key: bytes) -> None:
    if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != EK_BYTES:
        raise ValueError(f"ML-KEM-1024 encapsulation key must be {EK_BYTES} bytes")
    if not encapsulation_key_modulus_ok(bytes(public_key)):
        raise ValueError("ML-KEM-1024 encapsulation key failed the FIPS 203 modulus check")


def _check_ciphertext(ciphertext: bytes) -> None:
    if len(ciphertext) != CT_BYTES:
        raise EnvelopeFormatError(f"ML-KEM-1024 ciphertext must be {CT_BYTES} bytes")


def _check_shared_secret(ss: bytes) -> bytes:
    if len(ss) != SS_BYTES:
        raise RuntimeError(f"backend returned a {len(ss)}-byte shared secret")
    return bytes(ss)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class KemBackend(ABC):
    """Normalized ML-KEM-1024 interface. Handles are backend-specific opaque objects."""

    name: str
    secret_key_format: str

    @abstractmethod
    def provenance(self) -> dict[str, Any]: ...

    @abstractmethod
    def generate_keypair(self) -> RecipientKeyPair: ...

    @abstractmethod
    def load_public_key(self, public_key: bytes) -> Any:
        """Validate an encapsulation key and return a reusable handle."""

    @abstractmethod
    def load_secret_key(self, secret_key: bytes) -> Any:
        """Validate a decapsulation key and return a reusable handle."""

    @abstractmethod
    def encapsulate(self, public_handle: Any) -> Encapsulation: ...

    @abstractmethod
    def decapsulate(self, secret_handle: Any, ciphertext: bytes) -> bytes: ...

    def __repr__(self) -> str:
        return f"<KemBackend {self.name}>"


class OpenSSLMlKem1024(KemBackend):
    name = "openssl"
    secret_key_format = "seed64"

    def __init__(self) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric import mlkem
        except ImportError as exc:  # pragma: no cover - depends on installed version
            raise BackendUnavailableError("cryptography>=50 with ML-KEM support is required") from exc
        self._mlkem = mlkem

    def provenance(self) -> dict[str, Any]:
        from cryptography.hazmat.backends.openssl.backend import backend

        return {
            "backend": self.name,
            "parameter_set": PARAMETER_SET,
            "library": "cryptography",
            "library_version": _dist_version("cryptography"),
            "implementation": backend.openssl_version_text(),
            "secret_key_format": self.secret_key_format,
        }

    def generate_keypair(self) -> RecipientKeyPair:
        sk = self._mlkem.MLKEM1024PrivateKey.generate()
        return RecipientKeyPair(
            public_key=sk.public_key().public_bytes_raw(),
            secret_key=sk.private_bytes_raw(),
            backend=self.name,
            secret_key_format=self.secret_key_format,
        )

    def keypair_from_seed(self, seed: bytes) -> RecipientKeyPair:
        """Deterministic key generation from the 64-byte seed d || z (FIPS 203 ML-KEM.KeyGen_internal)."""
        if len(seed) != SEED_BYTES:
            raise ValueError("seed must be 64 bytes (d || z)")
        sk = self._mlkem.MLKEM1024PrivateKey.from_seed_bytes(bytes(seed))
        return RecipientKeyPair(sk.public_key().public_bytes_raw(), bytes(seed), self.name, self.secret_key_format)

    def load_public_key(self, public_key: bytes) -> Any:
        _check_public_key(public_key)
        return self._mlkem.MLKEM1024PublicKey.from_public_bytes(bytes(public_key))

    def load_secret_key(self, secret_key: bytes) -> Any:
        if len(secret_key) != SEED_BYTES:
            raise ValueError("openssl backend secret keys are 64-byte seeds (d || z)")
        return self._mlkem.MLKEM1024PrivateKey.from_seed_bytes(bytes(secret_key))

    def encapsulate(self, public_handle: Any) -> Encapsulation:
        ss, ct = public_handle.encapsulate()  # cryptography order: (shared_secret, ciphertext)
        if len(ct) != CT_BYTES:
            raise RuntimeError("backend returned an unexpected ciphertext length")
        return Encapsulation(_check_shared_secret(ss), bytes(ct))

    def decapsulate(self, secret_handle: Any, ciphertext: bytes) -> bytes:
        _check_ciphertext(ciphertext)
        return _check_shared_secret(secret_handle.decapsulate(bytes(ciphertext)))


class PqcryptoMlKem1024(KemBackend):
    name = "pqcrypto"
    secret_key_format = "expanded3168"

    def __init__(self) -> None:
        try:
            from pqcrypto.kem import ml_kem_1024
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("pqcrypto is not installed") from exc
        self._m = ml_kem_1024
        sizes = (ml_kem_1024.PUBLIC_KEY_SIZE, ml_kem_1024.SECRET_KEY_SIZE, ml_kem_1024.CIPHERTEXT_SIZE,
                 ml_kem_1024.SHARED_SECRET_SIZE)
        if sizes != (EK_BYTES, DK_BYTES, CT_BYTES, SS_BYTES):
            raise BackendUnavailableError(f"pqcrypto ml_kem_1024 reports unexpected sizes {sizes}")

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "parameter_set": PARAMETER_SET,
            "library": "pqcrypto",
            "library_version": _dist_version("pqcrypto"),
            "implementation": "Rust crate backbone-ml-kem 0.2.0 via PyO3 (per bundled CycloneDX SBOM)",
            "secret_key_format": self.secret_key_format,
        }

    def generate_keypair(self) -> RecipientKeyPair:
        pk, sk = self._m.keygen()
        return RecipientKeyPair(bytes(pk), bytes(sk), self.name, self.secret_key_format)

    def load_public_key(self, public_key: bytes) -> Any:
        _check_public_key(public_key)
        return bytes(public_key)

    def load_secret_key(self, secret_key: bytes) -> Any:
        if not decapsulation_key_hash_ok(bytes(secret_key)):
            raise ValueError("ML-KEM-1024 decapsulation key failed the FIPS 203 hash check")
        return bytes(secret_key)

    def encapsulate(self, public_handle: Any) -> Encapsulation:
        ct, ss = self._m.encaps(public_handle)  # pqcrypto 1.0.0 order: (ciphertext, shared_secret)
        if len(ct) != CT_BYTES:
            raise RuntimeError("backend returned an unexpected ciphertext length")
        return Encapsulation(_check_shared_secret(ss), bytes(ct))

    def decapsulate(self, secret_handle: Any, ciphertext: bytes) -> bytes:
        _check_ciphertext(ciphertext)
        return _check_shared_secret(self._m.decaps(secret_handle, bytes(ciphertext)))


class KyberPyMlKem1024(KemBackend):
    name = "kyber-py"
    secret_key_format = "expanded3168"

    def __init__(self) -> None:
        try:
            from kyber_py.ml_kem import ML_KEM_1024
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("kyber-py is not installed") from exc
        self._k = ML_KEM_1024

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "parameter_set": PARAMETER_SET,
            "library": "kyber-py",
            "library_version": _dist_version("kyber-py"),
            "implementation": "pure-Python FIPS 203 reference (not constant-time; reference use only)",
            "secret_key_format": self.secret_key_format,
        }

    def generate_keypair(self) -> RecipientKeyPair:
        ek, dk = self._k.keygen()
        return RecipientKeyPair(bytes(ek), bytes(dk), self.name, self.secret_key_format)

    def keypair_from_seed(self, seed: bytes) -> RecipientKeyPair:
        if len(seed) != SEED_BYTES:
            raise ValueError("seed must be 64 bytes (d || z)")
        ek, dk = self._k._keygen_internal(bytes(seed[:32]), bytes(seed[32:]))
        return RecipientKeyPair(bytes(ek), bytes(dk), self.name, self.secret_key_format)

    def encapsulate_deterministic(self, public_key: bytes, m: bytes) -> Encapsulation:
        """FIPS 203 ML-KEM.Encaps_internal with explicit randomness m (known-answer tests only)."""
        _check_public_key(public_key)
        ss, ct = self._k._encaps_internal(bytes(public_key), bytes(m))
        return Encapsulation(_check_shared_secret(ss), bytes(ct))

    def load_public_key(self, public_key: bytes) -> Any:
        _check_public_key(public_key)
        return bytes(public_key)

    def load_secret_key(self, secret_key: bytes) -> Any:
        if not decapsulation_key_hash_ok(bytes(secret_key)):
            raise ValueError("ML-KEM-1024 decapsulation key failed the FIPS 203 hash check")
        return bytes(secret_key)

    def encapsulate(self, public_handle: Any) -> Encapsulation:
        ss, ct = self._k.encaps(public_handle)  # kyber-py 1.2.0 order: (shared_secret, ciphertext)
        if len(ct) != CT_BYTES:
            raise RuntimeError("backend returned an unexpected ciphertext length")
        return Encapsulation(_check_shared_secret(ss), bytes(ct))

    def decapsulate(self, secret_handle: Any, ciphertext: bytes) -> bytes:
        _check_ciphertext(ciphertext)
        return _check_shared_secret(self._k.decaps(secret_handle, bytes(ciphertext)))


_BACKENDS = {
    OpenSSLMlKem1024.name: OpenSSLMlKem1024,
    PqcryptoMlKem1024.name: PqcryptoMlKem1024,
    KyberPyMlKem1024.name: KyberPyMlKem1024,
}
_INSTANCES: dict[str, KemBackend] = {}

DEFAULT_KEM_BACKEND = OpenSSLMlKem1024.name


def available_kem_backends() -> list[str]:
    names = []
    for name in _BACKENDS:
        try:
            get_kem_backend(name)
        except BackendUnavailableError:
            continue
        names.append(name)
    return names


def get_kem_backend(name: str = DEFAULT_KEM_BACKEND) -> KemBackend:
    if name not in _BACKENDS:
        raise BackendUnavailableError(f"unknown KEM backend {name!r}; choose from {sorted(_BACKENDS)}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _BACKENDS[name]()
    return _INSTANCES[name]
