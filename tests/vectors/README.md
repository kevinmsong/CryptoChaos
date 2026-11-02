# Test vectors

## `acvp/` — NIST ACVP ML-KEM (FIPS 203) known-answer vectors

Extracted unmodified from the `internalProjection.json` files of the NIST ACVP-Server repository,
keeping only the `ML-KEM-1024` test groups. Each file records its source URL, the ACVP-Server commit,
and the SHA-256 of the original upstream file.

| File | Upstream | Commit | Upstream SHA-256 | Tests kept |
|---|---|---|---|---|
| `ML-KEM-1024-keyGen.json` | `gen-val/json-files/ML-KEM-keyGen-FIPS203/internalProjection.json` | `15c0f3deeefbfa8cb6cd32a99e1ca3b738c66bf0` | `d7a62a2c3476957f56dd8d24f9004ea6776ccfe995ffe71a65bb9506dc9c7b1b` | 25 AFT |
| `ML-KEM-1024-encapDecap.json` | `gen-val/json-files/ML-KEM-encapDecap-FIPS203/internalProjection.json` | `ad33b3d9504491767f1aa76382464f3b3fa2359e` | `a556952ce869bb89c3a3196a701dad89647c193a34c86eafb61a9d710d5b810f` | 25 encapsulation AFT, 10 decapsulation VAL, 10 encapsulation-key check, 10 decapsulation-key check |

Coverage by backend (see `tests/test_kem.py`):

| Backend | keyGen | encapsulation | decapsulation | ek check | dk check |
|---|---|---|---|---|---|
| kyber-py 1.2.0 (reference) | 25/25 (ek, dk) | 25/25 | 35/35 | 10/10 (adapter) | 10/10 (adapter) |
| openssl (cryptography 50.0.2) | 25/25 (ek only; dk not exportable) | not exposed | not exposed (no expanded-key import) | 10/10 (adapter) | n/a (seed keys) |
| pqcrypto 1.0.0 | not exposed | not exposed | 35/35 | 10/10 (adapter) | 10/10 (adapter) |

"Adapter" means the FIPS 203 input check implemented in `cryptochaos.kem` and applied before the
backend is called. All three backends additionally agree in all nine sender/receiver combinations
and on implicit-rejection outputs for modified ciphertexts. Passing these vectors is evidence of
functional agreement with FIPS 203; it is not FIPS 140 validation of any module.

## Other vectors (inline in tests)

- scrypt: RFC 7914 section 12 (first three vectors), `tests/test_kdf.py`.
- HKDF-SHA256: RFC 5869 test cases 1-3, `tests/test_kdf.py`.
- Golden envelopes (suites A-F) and chaos values: frozen digests in
  `tests/test_protocol.py` and `tests/test_maps.py`.
