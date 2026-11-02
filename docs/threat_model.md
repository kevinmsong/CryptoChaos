# Threat model and security scope

## Setting

A sender encrypts data (for example, an image or an archive) to a recipient who holds an ML-KEM-1024
key pair. The recipient may additionally require a passphrase shared out of band (suites E and F).
Envelopes are stored or transmitted over untrusted channels and may be archived for a long time.

## Assumptions

- The recipient's public key reaches the sender through an authenticated external mechanism (a
  certificate, a signed directory, or manual verification). CryptoChaos does not authenticate
  public keys.
- The recipient's secret key is stored and protected by the recipient. Private-key storage and
  public-key provisioning are operational concerns outside the per-message protocol.
- The operating-system CSPRNG (`os.urandom`) is sound. ML-KEM-1024, HKDF-SHA256, scrypt,
  AES-256-GCM, ChaCha20-Poly1305, and BLAKE3 meet their standard security properties as implemented
  by the libraries listed in `docs/reproduction.md`.

## Adversary

The adversary sees every envelope and all public parameters, including the recipient's public key.
It may know or choose plaintexts and may submit modified envelopes to the recipient. Separate
scenarios analyse what happens when the adversary also learns the password or the recipient's
secret key.

| Adversary also knows | Suite C | Suite E / F | Suite D (negative control) |
|---|---|---|---|
| nothing else | confidentiality and integrity rest on ML-KEM-1024 and the AEAD | same, plus the password factor | same as C |
| the password only | not applicable | still needs the recipient secret key (KEM) | still needs the recipient secret key |
| the recipient secret key only | all envelopes readable | an offline guessing attack on the password: each guess costs one scrypt evaluation per distinct password salt (per envelope with per-message salts; once for all envelopes of a password session, which share one salt); strength depends on password entropy | trivial: the password-dependent input takes at most 2^35.4 values, needs no salt or cost, and colliding passphrases are interchangeable |
| both | all readable | all readable | all readable |

## Properties provided

- **Confidentiality and ciphertext integrity** of the payload and the full header, plus binding to
  the caller's associated data, under the assumptions above.
- **Password binding (E, F)**: the AEAD key depends on the full 32-byte scrypt output. A human
  passphrase has limited entropy. scrypt raises the cost of each guess; it does not create entropy.
  The chaotic value is a deterministic function of the scrypt key and adds no entropy or
  independent secret.
- **Fail-closed parsing**: malformed, truncated, oversized, downgraded, or cost-inflated envelopes
  are rejected before any expensive work, and authentication failures release no plaintext.
- **Key commitment**: each KEM-suite envelope commits to its derived key, so an attacker-built
  envelope opens under at most one derived key. For suites E and F this means at most one password
  (barring scrypt or HKDF collisions), so a receiver's responses test one password guess per
  submitted envelope, not many (partitioning-oracle resistance). Suite D's derived key depends on the
  passphrase only through its truncated-seed value, so every passphrase with the same value opens it.

## Properties not provided

- **Sender authentication**: anyone holding the recipient's public key can create valid suite-C
  envelopes. With E or F, anyone who also knows the password can create valid envelopes. A KEM/AEAD
  envelope does not identify its sender; use a signature scheme where that matters.
- **Forward secrecy for stored envelopes**: the recipient key is static. Later compromise of the
  recipient secret key exposes every suite-C envelope. For E and F, the compromise reduces
  protection to the password's guessing resistance. Password compromise alone does not expose
  envelopes without the secret key.
- **Suite negotiation**: receivers must name the suites they accept. A receiver that requires
  password binding should not accept suite C, because a suite-C envelope requires no password. The
  API also refuses to open a non-binding envelope when the caller supplies a password.
- **Side-channel and fault resistance**: not evaluated. The kyber-py reference backend is not
  constant-time and is used only for known-answer tests.
- **Validated implementation or proof of the composition**: the components are standardized or
  widely analysed, but this composition has neither a formal proof nor FIPS 140 validation, and
  passing known-answer tests is not validation.
- **Denial of service**: an attacker can make a receiver run scrypt up to the allow-listed cost
  (default maximum N = 2^18, 256 MiB) per submitted envelope. Deployments should rate-limit.
- **Envelope linkability and shared guessing cost**: envelopes in a password session share a salt,
  so they are linkable, and after a recipient-key compromise one password guess (one scrypt
  evaluation) is tested against all of them at once.

## Quantum considerations

ML-KEM-1024 is NIST security category 5 for key establishment (FIPS 203). That category is a
classification relative to the cost of attacking AES-256, not a measured bit strength. The symmetric
layer uses 256-bit keys. Grover's algorithm gives at most a quadratic speed-up for exhaustive key
search, and practical attack costs depend on circuit depth and error-correction overheads. This
work makes no end-to-end "quantum security bits" claim. `docs/security_components.md` lists each
component with its role and source.
