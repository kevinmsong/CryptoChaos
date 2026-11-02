"""CryptoChaos: passphrase-bound ML-KEM-1024/AEAD envelopes with optional chaotic diversification.

Public API::

    from cryptochaos import Sender, Receiver, generate_recipient_keypair

    kp = generate_recipient_keypair()                       # recipient side
    env = Sender("F", recipient_public_key=kp.public_key).seal(data, password=pw, aad=ctx)
    data = Receiver(recipient_secret_key=kp.secret_key, accepted_suites={"F"}).open(
        env, password=pw, aad=ctx)
"""
from .errors import (
    BackendUnavailableError,
    ConfigurationError,
    CryptoChaosError,
    DecryptionError,
    EnvelopeFormatError,
    LengthLimitError,
    UnsupportedParameterError,
    UnsupportedSuiteError,
    UnsupportedVersionError,
)
from .kdf import DEFAULT_SCRYPT, ScryptParams, ScryptPolicy
from .kem import DEFAULT_KEM_BACKEND, RecipientKeyPair, get_kem_backend
from .protocol import SUITES, VERSION, PasswordSession, Receiver, Sender, envelope_overhead

__version__ = "2.0.0"
PROTOCOL_NAME = "CryptoChaos"


def generate_recipient_keypair(kem_backend: str = DEFAULT_KEM_BACKEND) -> RecipientKeyPair:
    """Recipient setup: generate an ML-KEM-1024 key pair with the chosen backend."""
    return get_kem_backend(kem_backend).generate_keypair()


__all__ = [
    "BackendUnavailableError", "ConfigurationError", "CryptoChaosError", "DecryptionError",
    "EnvelopeFormatError", "LengthLimitError", "UnsupportedParameterError", "UnsupportedSuiteError",
    "UnsupportedVersionError", "DEFAULT_SCRYPT", "ScryptParams", "ScryptPolicy", "DEFAULT_KEM_BACKEND",
    "RecipientKeyPair", "get_kem_backend", "SUITES", "VERSION", "PasswordSession", "Receiver", "Sender",
    "envelope_overhead", "generate_recipient_keypair", "PROTOCOL_NAME", "__version__",
]
