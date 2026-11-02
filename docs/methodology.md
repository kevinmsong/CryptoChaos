# Experimental methodology and prespecified analysis plan

This plan was fixed before the principal run (`configs/principal.toml`) started. Changes made
after data collection are listed at the end with their reason.

## 1. Question and expectations

The study evaluates a passphrase-bound ML-KEM-1024/AEAD envelope and the incremental role of
deterministic chaotic diversification. The principal comparison is **F versus E**: the same
envelope, KEM backend, scrypt cost, framing, and payload, differing only in an additional IKM field
derived deterministically from the scrypt key by the chaotic maps.

Stated before measurement:

- The map output is a deterministic function of `pw_key`. It adds no entropy. Because the full
  32-byte `pw_key` is itself an IKM field in both E and F, F cannot be weaker than E under standard
  HKDF assumptions, and we expect it to be no stronger either.
- Expected results: (H1) no difference between F and E in any ciphertext diagnostic beyond
  sampling variation; (H2) identical wire overhead (1,702 bytes each); (H3) identical rejection
  behaviour in the authentication suite; (H4) a small positive latency cost for F from evaluating
  the maps once per password derivation.
- Evidence that chaos adds value would be any diagnostic or rejection metric on which F is
  better than E, with a 95 % interval excluding zero in F's favour. Evidence of cost is a
  positive F−E latency difference whose interval excludes zero. A null or negative result is
  reported as such, and the simpler suite E is then preferred operationally.

Secondary comparisons, all descriptive: E vs C (cost of password binding), D vs C (the negative
control under matched framing), C vs A (cost of the KEM envelope), and B vs A (alternative AEAD).

## 2. Configurations (ablation matrix)

| ID | Construction | Purpose |
|---|---|---|
| A | AES-256-GCM, pre-established 256-bit key, random 96-bit nonce | Symmetric payload reference |
| B | ChaCha20-Poly1305, pre-established key, random 96-bit nonce | Alternative AEAD reference |
| C | ML-KEM-1024 + HKDF-SHA256 + AES-256-GCM | KEM-envelope reference |
| D | C + chaotic value seeded from 5 bytes of BLAKE3(passphrase) (no salt or cost) | Negative control: a chaotic pathway that replaces password hashing |
| E | C + scrypt(N=2^14, r=8, p=1) password key | Primary non-chaotic password-bound comparator |
| F | E + chaotic value derived from the scrypt key | CryptoChaos |

D is deliberately weak: it shows the effect of a truncated chaotic seed under matched framing.
Unauthenticated ciphers are not compared.

All KEM suites share one recipient key pair per process (`openssl` backend, cryptography 50.0.2 /
OpenSSL 4.0.3), the same serialization code, the same OS-random nonces and salts, and the same
payload bytes. A and B share one pre-established key per process. The benchmark password is
generated randomly per process and never written to disk.

## 3. Workloads

1. **Recipient setup**: ML-KEM-1024 key generation (stage microbenchmark).
2. **Warm-recipient message** (principal): seal with an existing recipient public key; open with
   the loaded recipient secret key. Includes encapsulation/decapsulation, password derivation,
   HKDF, framing, and AEAD.
3. **Cold provisioning plus message**: seal = new recipient key generation + sender setup
   (encapsulation-key check) + seal; open = recipient secret-key load + open. KEM suites only.
4. **Password derivation policy**: `per_message` (fresh 16-byte salt and scrypt per message, on
   both seal and open; principal) and `session` (salt and derived key cached per session; size
   scaling only). The cache policy is a declared workload, not an optimization of E or F alone.

## 4. Data

- All 800 DIV2K training HR PNGs (manifest `configs/manifests/div2k_train_hr.csv` with file
  SHA-256, decoded-payload SHA-256, dimensions). Files are opened read-only, decoded with Pillow to
  8-bit RGB, serialized in C order (`width × height × 3` bytes). Every decode is checked against the
  manifest hash outside the timers.
- The image is the unit of analysis. No image is selected or excluded based on algorithm results.
- Size scaling, KDF sensitivity, stage, and memory passes use synthetic payloads generated from the
  experiment seed (content is irrelevant to AEAD/KEM timing).

## 5. Timing

- `time.perf_counter_ns()` brackets exactly one `seal` or one `open` call. Decoding, hashing,
  round-trip comparison, diagnostics, logging, and file writes are outside timers. The Python
  garbage collector is disabled inside each timed call.
- Warm-up differs by pass. Timing: per suite, one recorded first call and 8 discarded messages (and
  cold-provisioning messages for KEM suites) on a 1 MiB payload. Scaling: the same on 64 KiB. KDF
  sensitivity: one message per suite and scrypt setting. Stages: one call per microbenchmark.
- The first warm-up call of the timing and scaling passes is logged in `raw/coldstart.csv`. Earlier
  passes run in the same process, so these are first calls of a pass, not process cold-start
  latency, which was not measured. The file has no column naming its pass: in the principal run the
  1 MiB rows come from the timing pass and the 64 KiB rows from the scaling pass.
- For each image and repeat, the six suites run in an order drawn from a seeded permutation, so
  no suite systematically runs first or last. Methods are never run concurrently. CPU-parallel
  passes (diagnostics, calibration) run after all timing passes.
- System CPU utilization is sampled once per second during the timing and scaling passes as a
  contention check. The frequency column is informative only on Linux; Windows reports the nominal
  value. The machine is kept awake and on AC power, at normal process priority.
- Stage microbenchmarks are run separately and are never summed to replace end-to-end figures.

## 6. Throughput, overhead, and memory

- Throughput uses actual plaintext bytes: MiB/s = bytes / 2^20 / seconds. The tables report the
  aggregate (total bytes / total time); `report/timing_summary.csv` and the report macros also give
  the median of per-image rates.
- Spatial pixels are `width × height`. Channel samples (bytes) are `width × height × 3`.
- Wire overhead is the exact serialized envelope size minus the plaintext length (header + 16-byte
  tag): 45 B (A, B), 1,681 B (C, D), 1,702 B (E, F). Key provisioning is not per-message overhead.
- Memory is measured in isolated subprocesses: baseline RSS after imports, key setup, and payload
  allocation, and the process peak (Windows peak working set, Linux `ru_maxrss`) after three
  seal/open operations. `tracemalloc` peaks are reported separately as Python-allocation
  accounting, not total memory.

## 7. Descriptive diagnostics

| Experiment | What changes | Interpretation |
|---|---|---|
| Plaintext/ciphertext difference | plaintext vs its payload ciphertext | descriptive image transformation; depends on the plaintext |
| Independent re-encryption distance | same plaintext, two fresh encryptions | distance between independently randomized ciphertexts |
| Key-sensitivity fixture | one input changed, all others fixed | descriptive dependence on key material; no security proof |
| Authentication suite | header, AAD, KEM ciphertext, payload, tag, key, password, framing | correct rejection; no plaintext released |

Diagnostics use the payload ciphertext only (envelope header and tag excluded) over the full
length. The key-sensitivity fixture uses the first 1 MiB of the payloads of the first 50 images
in manifest order, plus one 1 MiB synthetic payload. Byte histogram, Shannon and min-entropy (plug-in), chi-square uniformity, bit balance, and
lag-1..3 byte correlation describe a single ciphertext. Difference metrics are reported as unequal
bytes (%), mean absolute byte difference (% of 255; signed 16-bit subtraction), and unequal bits
(%). For two independent uniform arrays the nominal references are 99.609375 %, 33.463542 %, and
50 %. They do not apply to plaintext/ciphertext comparisons. These diagnostics cannot establish
indistinguishability or IND-CCA security.
The fixture shows that under a fixed key and nonce a one-bit plaintext change flips exactly one
payload bit, and this demonstration runs on synthetic data only.

## 8. Randomness tests

- Subset of SP 800-22 Rev. 1a: frequency, block frequency (M = 20,000, N = 50), runs, longest run
  (M = 10^4 regime), DFT spectral, non-overlapping template (B = 000000001, m = 9, N = 8), and
  serial (m = 16; both p-values reported separately). Not the full suite.
- Sample: the first 1,000,000 bits (MSB first) of each payload ciphertext, one ciphertext per image
  and suite. Shorter inputs are recorded as `not_applicable`; the runs prerequisite failure is
  recorded as `prerequisite_failed`. Nothing is padded or resampled.
- Validation: the implementation agrees with NIST STS 2.1.2 (built from the NIST source) to six
  decimals on 13 reference streams, for every battery test (both serial p-values) and for all 148
  aperiodic m = 9 templates on each stream. The M = 128 longest-run regime (n below 750,000) is
  checked against the SP 800-22 tables only; the principal sample uses the M = 10^4 regime.
- The parameters each test applies are recorded in `report/summary.json` (`sts_parameters`).
- Calibration: 1,000 ChaCha20 keystreams (reproducible from a public label), 100 MT19937 streams,
  and 50 streams each of five structured negative controls. The LCG low-bit control depends only
  on the seed's parity, so its 50 streams are 2 distinct sequences. Rejection rates are compared
  with the nominal 1 % using Clopper-Pearson 95 % intervals and SP 800-22 section 4.2 checks
  (proportion interval, p-value uniformity). The uniformity check is applied only with at least 55
  sequences (section 4.2.2), so not to the 50-stream negative controls. Not every random stream is
  expected to pass every test.
- The same section 4.2 checks summarize the ciphertext results per test and suite, and pooled over
  suites (`report/sts_summary.csv`, `report/sts_pooled_summary.csv`). The eight results of one
  sample are not independent (the two serial p-values share a sample).

## 9. Statistical analysis

- Repeated timings of an image are summarized by their median; corpus summaries use per-image
  medians. Tail latencies (p95, p99) use all observations.
- Paired comparisons (Section 1) use per-image differences and ratios of medians: median and mean
  difference (ms) and geometric-mean ratio, with 95 % percentile intervals from a two-level
  bootstrap (10,000 resamples: images with replacement, then repeats within image and suite;
  fixed seed). Repeats are resampled independently for each suite, which ignores their shared
  blocks and makes the intervals slightly conservative.
- Tail latencies over all observations also span image sizes; the report adds the within-image
  ratio of each observation to its image's median.
- **Primary endpoint**: F−E per-image median round-trip (seal + open) latency, warm recipient,
  per-message password policy. Secondary endpoints and all diagnostic comparisons are descriptive
  and exploratory, with no multiplicity correction. They are labelled as such.
- No equivalence margin was prespecified, so no equivalence is claimed. Overlapping intervals or
  non-significance are reported as "no evidence of a difference".
- Pass rates use exact Clopper-Pearson intervals.
- Results generalize only to this hardware, operating system, library versions, and image corpus.

## 10. Integrity

Every expected observation of the timing, scaling, KDF-sensitivity, diagnostics, authentication,
key-sensitivity, and calibration passes is enumerated in advance with a unique
key, and `verify-artifact` fails if any of them is missing. The stage and memory passes are not
enumerated; `verify-artifact` reports their row counts with a warning. Errors are recorded as rows
(never dropped). Missing,
not-applicable, failed, and successful observations are kept distinct. A resumed run must match
the stored identity: effective-configuration hash, measurement-code hash, manifest and selection
hashes, KEM backend provenance, library versions, and interpreter. The identity does not include
the hardware or operating system, so a resume on another machine would not be refused. Each
completed run records its Git commit, a dirty-patch hash, the environment, and SHA-256 sums of its
configuration, environment, raw, run, and status files. `report` writes `report/SHA256SUMS` for
the derived tables, figures, and macros, and `verify-artifact` checks both.

Seeds have separate purposes: the experiment seed orders suites within blocks, generates synthetic
payloads, and drives bootstrap resampling. All cryptographic randomness comes from the OS CSPRNG.

## 11. Changes after data collection

- **2026-11-01, restart of the principal run.** The first principal run completed its timing pass
  (24,000 of 24,000 observations), but an unplanned operating-system reboot at 08:45 UTC interrupted
  its scaling pass. No results from that run were analysed. Before the study was restarted, the
  implementation was optimized and hardened. `Sender.seal` now writes the AEAD output directly behind
  the header instead of concatenating, which removes one full copy of the ciphertext; the wire format
  is unchanged. Passing a password to a suite that does not bind one is now an error. Because these
  changes alter the measured code path, the entire study was run again from the beginning, and the
  interrupted run was archived outside the published results. The configuration, endpoints, and
  analysis plan did not change.
- **Run conditions.** During the timing-sensitive passes the OneDrive client was shut down, and an
  unrelated workload on the same machine (a hyperparameter search) was suspended and later
  resumed. Background services such as the OneDrive sync service, Windows Search, and Defender kept
  running. Their aggregate load is recorded once per second in `raw/load_*.csv`.
- **2026-11-02, corrections to the analysis and reporting.** No measurement changed. The
  report now adds the SP 800-22 section 4.2 checks for the ciphertext results, an error count, the
  within-image tail ratio, the median of per-image throughput, the warm C column in the session
  table, the recorded test parameters, and `report/SHA256SUMS`. The p-value uniformity check is no
  longer reported for groups of fewer than 55 sequences. Figure text was enlarged and labels were
  corrected. This document now describes the warm-up, the first-call log, CPU sampling, the
  key-sensitivity inputs, the bootstrap pairing, and the completeness checks as implemented.

- **2026-11-02, repository layout.** The run output moved to `results/principal` (recorded in
  `run.json`; observations unchanged).
