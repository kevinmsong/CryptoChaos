# Verification and limitations

## Verification

- **Tests**: unit and vector tests cover NIST ACVP ML-KEM-1024 vectors (key generation,
  encapsulation and decapsulation, input checks), RFC 7914 and RFC 5869 vectors, NIST STS 2.1.2
  reference values (the declared SP 800-22 subset and all 148 aperiodic m = 9 templates on 13
  reference streams), golden envelopes for every configuration, rejection cases, a mutation fuzzer,
  a partitioning-oracle regression test (GCM multicollision), and harness resume and tail repair.
  GPU tests skip when no GPU environment is present.
- **GPU parity**: the Triton kernels matched the CPU reference exactly in 16 of 16 recorded cases
  (`results/gpu/parity.json`, inputs of 1 to 1,000,003 bytes). A counter-width test on
  2^31 + 12,345 bytes was run in the GPU environment, but its log is not part of the artifact.
- **Artifact integrity**: `verify-artifact` checks that all enumerated rows are present, that the
  SHA-256 sums of the run and of its report match, and that no secret-looking column names or JSON
  keys appear. It names the commit whose measurement code produced each run. The stage and memory
  passes are not enumerated and are reported with a warning.

## Runs

All runs used one Windows 11 laptop: Intel Core i7-11800H (8 cores, 16 threads), 63.8 GiB, CPython
3.13.14.

| Run | Content |
|---|---|
| Principal (`results/principal`) | 800 DIV2K images × 5 repeats × 6 configurations (24,000 timed round trips), cold provisioning, stage microbenchmarks, size scaling (1 KiB–16 MiB), scrypt cost sensitivity (log2 N = 14, 16, 18), memory, 3,750 rejection trials, key-sensitivity test, descriptive diagnostics and SP 800-22 subset on every image and configuration, 1,350 calibration streams, exhaustive map analysis |
| Supplementary (`results/supplement`) | Process cold start in fresh processes, header serialization and parsing, and single-bit header tampering stratified by header field |
| GPU (`results/gpu`) | Diagnostic kernels, 1–256 MiB, GeForce RTX 5060 Ti, torch 2.15.0+cu132, triton-windows 3.9.0 |

During the timing-sensitive passes of the principal run the cloud-sync client was stopped and an
unrelated workload was suspended. Background services kept running, and their load is recorded once
per second. The principal run was restarted once after an operating-system reboot interrupted its
scaling pass; the interrupted run was not analyzed (`docs/methodology.md`, section 11). During the
supplementary run the unrelated workload was again suspended, but the cloud-sync client and other
background programs kept running (median total CPU load 27%, including the run's own processes;
`results/supplement/raw/load.csv`). Its header timings therefore vary more from batch to batch than
the principal run's stage timings (interquartile range about 40% of the median against 2–7%); its
parse medians are 4–13% below the principal run's parse-stage medians, which time single calls.

## Limitations

- One platform (Windows 11 laptop, CPython). Absolute latencies will differ elsewhere; the paired,
  interleaved design protects only the comparisons.
- The workstation was not a dedicated benchmark host.
- One image corpus (DIV2K training set) and one password-KDF family (scrypt). Argon2id is not
  evaluated.
- The descriptive diagnostics and the SP 800-22 subset are descriptive. They cannot establish
  IND-CCA security or distinguish one sound AEAD from another.
- No proof of the composition, no FIPS 140 validation, and no side-channel evaluation. The kyber-py
  reference backend is not constant-time and is used only for known-answer tests.
- The GPU timings record medians without per-repeat samples or the repeat count, and Triton
  compilation time was not measured. The GPU environment was deleted after its run.
- The resume identity does not include the hardware; the stage and memory passes are not enumerated
  for completeness.
