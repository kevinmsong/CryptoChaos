"""Exception hierarchy for the CryptoChaos package.

Format errors are raised for problems that are visible in the public envelope
structure (they reveal nothing secret). Every cryptographic failure after
parsing (wrong recipient key, wrong password, wrong associated data, modified
header or ciphertext) is reported as a single ``DecryptionError`` so that
callers cannot distinguish its causes.
"""


class CryptoChaosError(Exception):
    """Base class for all package errors."""


class ConfigurationError(CryptoChaosError):
    """Invalid experiment or API configuration."""


class BackendUnavailableError(CryptoChaosError):
    """A requested optional backend (KEM implementation, GPU) is not installed."""


class EnvelopeFormatError(CryptoChaosError):
    """The envelope is malformed, truncated, has trailing data, or bad field lengths."""


class UnsupportedVersionError(EnvelopeFormatError):
    """The envelope magic or protocol version is not supported."""


class UnsupportedSuiteError(EnvelopeFormatError):
    """The suite identifier is unknown or not accepted by this receiver."""


class UnsupportedParameterError(EnvelopeFormatError):
    """A declared parameter (e.g. password-KDF cost) is outside the allow-list."""


class LengthLimitError(EnvelopeFormatError):
    """A length exceeds the implementation limit."""


class DecryptionError(CryptoChaosError):
    """Authentication failed; no plaintext is released."""
