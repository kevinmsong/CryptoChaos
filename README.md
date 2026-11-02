# CryptoChaos

A reference implementation and reproducible evaluation of passphrase-bound ML-KEM-1024 envelopes with
an optional deterministic chaotic component.

> **Status**: research artifact. CryptoChaos has no formal security proof, no FIPS 140 validation, and
> no side-channel evaluation. Do not use it to protect real data without independent review. See
> [docs/threat_model.md](docs/threat_model.md).

## The construction

To seal a message, the sender encapsulates to the recipient's ML-KEM-1024 public key. In the
password-bound configurations it also derives a 32-byte key from the passphrase with scrypt
(N = 2^14, r = 8, p = 1, fresh 16-byte salt). HKDF-SHA256 combines the KEM secret, the full password
key, and, in configuration F only, a value computed from the password key by exact-integer chaotic
maps. It outputs a one-time AES-256-GCM key and a 32-byte key commitment. The whole header (version,
configuration, KEM ciphertext, salts, commitment, scrypt parameters, nonce, length) is authenticated as
associated data. The chaotic value adds no entropy: a passphrase's strength comes from the passphrase
and the scrypt cost alone. Full specification: [docs/protocol.md](docs/protocol.md).

| ID | Construction | Overhead (bytes) |
|---|---|---:|
| A | AES-256-GCM, pre-established key | 45 |
| B | ChaCha20-Poly1305, pre-established key | 45 |
| C | ML-KEM-1024 + HKDF-SHA256 + AES-256-GCM | 1,681 |
| D | C + truncated-seed chaotic value (negative control; insecure by design) | 1,681 |
| E | C + scrypt password key | 1,702 |
| F | E + chaotic value (CryptoChaos) | 1,702 |

The study's principal comparison is F against E, which differ only in the chaotic value. D shows what
happens when a chaotic value seeded from five bytes of the passphrase hash replaces password hashing.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest
```

```python
from cryptochaos import Receiver, Sender, generate_recipient_keypair

kp = generate_recipient_keypair()                        # recipient: keep kp.secret_key private
env = Sender("F", recipient_public_key=kp.public_key).seal(b"data", password="pass phrase", aad=b"ctx")
pt = Receiver(recipient_secret_key=kp.secret_key, accepted_suites={"F"}).open(env, password="pass phrase", aad=b"ctx")
```

## Results

Principal run: 800 DIV2K training images × 5 repeats × 6 configurations on an Intel Core i7-11800H
laptop ([results/principal](results/principal)).

- **Chaotic value (F vs. E, primary endpoint)**: +0.268 ms per seal-and-open round trip (95% CI
  0.225–0.311), a geometric-mean ratio of 1.0039 (about 0.4%). Rejection outcomes were identical, and
  ciphertext diagnostics and randomness-test results showed no meaningful difference.
- **KEM-based key establishment (C vs. A)**: ML-KEM-1024 with HKDF, key commitment, and header
  handling added 0.292 ms per message (0.281–0.299).
- **Password binding with scrypt (E vs. C)**: +65.4 ms, two scrypt evaluations at N = 2^14.
- **Rejection tests**: 3,750 trials with no unexpected acceptance in A, B, C, E, and F. The negative
  control D accepts a different, colliding passphrase.
- **Randomness tests**: 431 of 38,400 SP 800-22 results rejected at α = 0.01 (1.12%, 95% CI
  1.02–1.23%), not distinguishable from uniform ChaCha20 calibration streams (0.975%, Fisher p = 0.26).

The configuration without the chaotic value (E) is the recommended way to bind a passphrase.
Limitations: [docs/limitations.md](docs/limitations.md). Component-level security notes:
[docs/security_components.md](docs/security_components.md).

## Reproducing the study

See [docs/reproduction.md](docs/reproduction.md). In short: set `CRYPTOCHAOS_DIV2K_PATH` to the DIV2K
training high-resolution images, then run

```powershell
python -m cryptochaos doctor
python -m cryptochaos benchmark --config configs/principal.toml --resume
python -m cryptochaos report --run results/principal
python -m cryptochaos verify-artifact --run results/principal
python scripts/supplementary_run.py --out results/supplement-new
```

`verify-artifact` checks completeness, SHA-256 sums, and secret hygiene, and it identifies the commit
whose measurement code produced each run. The experiment design and analysis plan are in
[docs/methodology.md](docs/methodology.md).

## Repository layout

| Path | Content |
|---|---|
| `src/cryptochaos/` | Protocol, KEM adapters, key schedule, maps, metrics, statistical tests, benchmark, reporting, CLI |
| `tests/` | Unit, vector (NIST ACVP ML-KEM, RFC 7914, RFC 5869, NIST STS), protocol, harness, and GPU parity tests |
| `configs/` | Experiment configuration and the dataset manifest |
| `results/` | Raw observations, run identities, environment records, SHA-256 sums, and generated reports |
| `scripts/` | Run driver, supplementary run, STS reference-vector builder, GPU results table |
| `docs/` | Protocol specification, threat model, security components, methodology, reproduction, limitations |

## License

MIT License (see [LICENSE](LICENSE)).
