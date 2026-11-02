# Security components

CryptoChaos makes no end-to-end security claim and reports no "quantum security bits". This table
lists each component, its role in the envelope, and the property it contributes according to the
cited standard. A composition of standardized parts is not itself standardized, proven, or validated
(see `docs/threat_model.md`).

| Component | Role | Parameters | Basis | Contributes | Does not contribute |
|---|---|---|---|---|---|
| ML-KEM-1024 | Key establishment to the recipient | FIPS 203 parameter set ML-KEM-1024; 1,568-byte ciphertext, 32-byte shared secret | FIPS 203 (NIST security category 5, defined relative to key search on AES-256) | Confidentiality of the payload key against anyone without the recipient secret key, under module-lattice assumptions; implicit rejection of modified ciphertexts | Sender authentication; forward secrecy (the recipient key is static) |
| HKDF-SHA256 | Combines all key material into the payload key and the key commitment | 32-byte salt per message; `info` binds version, suite, and role; 64-byte output | RFC 5869; Krawczyk (CRYPTO 2010) | A uniformly distributed key if at least one input is secret and the extractor assumption holds | Entropy beyond its inputs |
| Key commitment | First 32 HKDF output bytes are the AEAD key, the next 32 are carried in the header and checked in constant time | 32 bytes | Partitioning-oracle attacks (Len, Grubbs, Ristenpart, USENIX Security 2021) | An envelope opens under at most one derived key, so one submitted suite E or F envelope tests at most one password (in suite D, one class of passphrases that share a truncated-seed chaotic value) | Anything if the receiver skips the check (it cannot through the public API) |
| scrypt | Password key | N = 2^14, r = 8, p = 1, fresh 16-byte salt per message (or per session); receiver allow-list log2 N in [14, 18] | RFC 7914; Percival (BSDCan 2009) | Raises the cost of each password guess (16 MiB, about 33 ms per evaluation on the study machine) | Entropy: a weak passphrase stays weak |
| AES-256-GCM | Payload AEAD | 256-bit key (a fresh key per message in suites C-F; a pre-established key in suite A, at most 2^32 messages per `Sender`), 96-bit random nonce, 128-bit tag; whole header as associated data | NIST SP 800-38D | Confidentiality and integrity of the payload and authenticity of every header bit | Key commitment (supplied separately above) |
| ChaCha20-Poly1305 | Reference AEAD (suite B only) | 256-bit key, 96-bit nonce | RFC 8439 | Same as AES-256-GCM, for the symmetric baseline | Key commitment |
| BLAKE3 | Fixed-length encoding of the KEM shared secret; labeled hashing inside the chaotic maps | 32-byte output | BLAKE3 specification (O'Connor et al., 2020) | Domain-separated fixed-length inputs | A security requirement of the envelope: HKDF provides the extraction |
| Chaotic maps (suite F) | Optional deterministic IKM field computed from the password key | Exact 64-bit integer logistic, tent, Chebyshev T5, and torus Hénon maps; 32-byte output | This work (`docs/protocol.md`, golden vectors in `tests/`) | Nothing measurable: the value is a function of an input HKDF already receives | Entropy, independence from the password key, or any hardness assumption |
| Truncated-seed chaotic maps (suite D, negative control) | Passphrase value that replaces password hashing | Four 8-bit fixed-point maps seeded from 5 digest bytes | `src/cryptochaos/map_analysis.py` | Nothing; included only as a negative control | Passphrase binding: at most 2^35.43 distinct values and known colliding passphrases |

## Quantum considerations

NIST security category 5 classifies ML-KEM-1024 relative to the cost of exhaustive key search on
AES-256; it is not a measured bit strength. The symmetric layer uses 256-bit keys. Grover's algorithm
offers at most a quadratic speed-up for exhaustive key search, and concrete attack costs depend on
circuit depth and error-correction assumptions (Grassl et al., PQCrypto 2016; Jaques et al.,
EUROCRYPT 2020). Password guessing against suites E and F is a classical cost question governed by
the passphrase distribution and the scrypt parameters.

## Randomness

Every salt, nonce, and key pair comes from the operating system's random number generator
(`os.urandom` or the backend's equivalent). The experiment seed in the configurations orders work,
generates synthetic payloads, and drives bootstrap resampling only; it never reaches key material.
