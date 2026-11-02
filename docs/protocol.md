# CryptoChaos protocol specification (wire-format version 2)

Reference implementation: `src/cryptochaos/protocol.py`, `kdf.py`, `kem.py`, `maps.py`.
Golden envelopes for every suite are frozen in `tests/test_protocol.py` (SHA-256 of the full
envelope bytes); any change to the encoding below changes those digests and requires a new
version or suite identifier.

## Primitives

| Role | Primitive | Parameters |
|---|---|---|
| Key encapsulation | ML-KEM-1024 (FIPS 203) | ek 1,568 B, dk 3,168 B (or 64-byte seed d‖z), ciphertext 1,568 B, shared secret 32 B |
| Shared-secret encoding | BLAKE3 | 32-byte output of BLAKE3(ss) |
| Password KDF | scrypt (RFC 7914) | N = 2^14 default, r = 8, p = 1, 16-byte salt, 32-byte output |
| Key derivation | HKDF-SHA256 (RFC 5869) | 32-byte salt, 64-byte output (AEAD key and key commitment) |
| Payload | AES-256-GCM (SP 800-38D), or ChaCha20-Poly1305 for suite B | 96-bit nonce, 128-bit tag |
| Optional diversification | chaotic maps (`maps.v2_chaos_value`) | deterministic, 32 bytes, no added entropy |

## Suites

| Code | ID | Key material entering HKDF |
|---|---|---|
| A | 0x01 | none (pre-established 256-bit key used directly with AES-256-GCM) |
| B | 0x02 | none (pre-established 256-bit key used directly with ChaCha20-Poly1305) |
| C | 0x10 | KEM |
| D | 0x11 | KEM, TRUNCATED-SEED CHAOTIC VALUE of the passphrase (negative control only) |
| E | 0x12 | KEM, PW_KEY |
| F | 0x13 | KEM, PW_KEY, CHAOS_V2 |

Suite D is a negative control for the ablation. It is insecure by design: its
password-dependent input takes at most 2^35.4 values and is computed without salt or cost. Do not
deploy it.

## Envelope encoding

All integers are big-endian. Lengths are explicit and must equal the values below exactly.

```
offset size  field
0      4     magic        "CCv2"
4      1     version      0x02
5      1     suite        suite ID
6      1     flags        0x00 (reserved)
7      1     reserved     0x00
-- suites C-F --
8      2     kem_ct_len   1568
10     1568  kem_ct       ML-KEM-1024 ciphertext
1578   1     salt_len     32
1579   32    hkdf_salt    OS-random
1611   1     commit_len   32
1612   32    commitment   HKDF output bytes 32..63 (key commitment)
-- suites E-F --
1644   1     kdf_id       0x01 (scrypt)
1645   1     log2_n       14..18 under the default receiver policy
1646   1     r            8
1647   1     p            1
1648   1     pw_salt_len  16
1649   16    pw_salt      OS-random (or session salt, see below)
-- all suites --
       1     nonce_len    12
       12    nonce        OS-random
       8     ct_len       len(ciphertext || tag) = len(plaintext) + 16
       ct_len            AEAD ciphertext || 16-byte tag
```

Header length: 29 B (A, B), 1,665 B (C, D), 1,686 B (E, F). Overhead including the tag: 45 B,
1,681 B, and 1,702 B respectively. The implementation limits plaintexts to 2^30 bytes and caller
AAD to 2^20 bytes.

## Associated data

```
AAD = "CryptoChaos-v2/AAD" || u32(len(header)) || header || u64(len(aad)) || aad
```

Every header byte (version, suite, flags, KEM ciphertext, salts, KDF parameters, nonce, length) is
authenticated, as is the caller's associated data, which is not transmitted.

## Key schedule (suites C-F)

```
ss          = ML-KEM-1024.Decaps/Encaps shared secret
k_kem       = BLAKE3(ss)                                            (field 0x01)
pw          = UTF-8(NFC(password))                                   (E, F)
pw_key      = scrypt(pw, pw_salt, N = 2^log2_n, r, p, 32)            (field 0x02; E, F)
chaos_v2    = v2_chaos_value(pw_key)                                 (field 0x03; F)
d_value     = truncated-seed chaotic key of UTF-8(password)            (field 0x04; D)
IKM         = "CryptoChaos-v2/IKM" || u8(2) || u8(suite) || u8(#fields)
              || for each field in the order above: u8(tag) || u16(len) || value
info        = "CryptoChaos-v2" || u8(2) || u8(suite) || "payload-key"
OKM         = HKDF-SHA256(IKM, salt = hkdf_salt, info, 64)
AEAD key    = OKM[0:32]
commitment  = OKM[32:64]          (carried in the header, compared in constant time before decryption)
```

AES-GCM and ChaCha20-Poly1305 are not key-committing: an attacker who knows only the public key can
build one ciphertext and tag that verify under many derived keys (one per guessed password), so a
receiver's accept/reject decision would test many passwords per envelope (a partitioning oracle).
The commitment makes an envelope valid under at most one derived key
(`tests/test_partitioning_oracle.py` builds the multicollision and checks this).

The full 32-byte `pw_key` is always present in E and F. The chaos value is an additional field
and never replaces it. A fresh KEM encapsulation and a fresh HKDF salt make every message key
single-use in suites C-F.

## Chaotic value

Seeds are five 64-bit big-endian words of `BLAKE3("CryptoChaos-v2/chaos-seed" || pw_key)` (40-byte
XOF output). Four maps run on 64-bit unsigned states with exact integer arithmetic: logistic
(r = 399/100), tent (mu = 199/100), Chebyshev T5, and a torus-wrapped Henon map (a = 7/5,
b = 3/10, scale 2^61). The exact formulas are given in `maps.V2Maps`. Each map discards 16
iterations, then emits the most significant byte of the state for 32 iterations.
`chaos_v2 = BLAKE3("CryptoChaos-v2/chaos-value" || s_logistic || s_tent || s_chebyshev || s_henon)`.

## Processing rules

Sender:
1. Validate the recipient encapsulation key (length and FIPS 203 modulus check) once, at setup.
2. Per message: encapsulate; draw `hkdf_salt` (32 B), `pw_salt` (16 B, E/F), and `nonce` (12 B) from
   `os.urandom`; derive the key; encode the header; AEAD-encrypt with the AAD above directly into the
   envelope buffer (`seal` returns a `bytearray`; the ciphertext is never copied).
3. A password passed for a suite that does not bind one (A, B, C) is an error, never silently ignored.

Receiver:
1. Parse strictly. Reject unknown magic or version, unknown suites, suites outside the receiver's
   explicit `accepted_suites`, non-zero reserved fields, wrong field lengths, KDF parameters
   outside the allow-list (checked before any KDF work), lengths above the limits, truncation,
   and trailing data. These format errors reveal only public structure.
2. Decapsulate (ML-KEM implicit rejection yields an unrelated key for invalid ciphertexts), derive
   the key and commitment, compare the commitment in constant time, and AEAD-decrypt.
3. If the caller supplies a password (or session) and the envelope's suite does not bind a password,
   reject it. A caller that checks a password therefore never accepts an envelope whose key does not
   depend on that password, even when the receiver accepts several suites. Any authentication failure raises one generic `DecryptionError`. No
   plaintext is returned on failure.

## Session-cached password material

A `PasswordSession` fixes one `pw_salt` and caches `pw_key` (and `chaos_v2`) for a sequence of
messages. Envelopes in a session carry the same `pw_salt` and parameters, which links them to each
other, and after a recipient-key compromise one password guess (one scrypt evaluation) is tested
against all of them at once. A receiver uses its cached key only when the envelope's salt and
parameters match. Otherwise `open()` raises `ValueError` (a call passes either a password or a
session, not both), and the caller must call `open()` again with the password.

## Nonce and key lifetime

- Suites C-F: one AEAD key per message, so (key, nonce) reuse would require a KEM shared-secret
  and HKDF-salt collision.
- Suites A, B: the pre-established key is used with random 96-bit nonces. A `Sender` refuses more
  than 2^32 messages under its pre-established key (SP 800-38D section 8.3). The count belongs to
  the `Sender` object and is not persisted, so an application must route every use of a
  pre-established key through one `Sender` or track the count itself. Random generation is not a mathematical
  guarantee against reuse, and the bound keeps the collision probability below about 2^-33.
