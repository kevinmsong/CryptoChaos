# Reproduction guide

## Requirements

- Python 3.11 or newer (the published runs used CPython 3.13.14 on Windows 11).
- The DIV2K training high-resolution images (800 PNGs, `DIV2K_train_HR`), obtained from the
  original source (Agustsson and Timofte, NTIRE 2017). The images are not redistributed here;
  `configs/manifests/div2k_train_hr.csv` lists each file's SHA-256 and decoded-payload SHA-256.
- Optional: an NVIDIA GPU with a CUDA build of PyTorch and Triton for the GPU diagnostic kernels.
- Optional: a C compiler to rebuild the NIST STS 2.1.2 reference vectors.

## Windows (CPU)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
# exact versions used for the published runs:
python -m pip install -r requirements/lock-windows-py313.txt
$env:CRYPTOCHAOS_DIV2K_PATH = "C:\path\to\DIV2K_train_HR"
python -m pytest
python -m cryptochaos doctor
python -m cryptochaos benchmark --config configs/smoke.toml
python -m cryptochaos benchmark --config configs/principal.toml --resume
python -m cryptochaos report --run results/principal
python -m cryptochaos verify-artifact --run results/principal
```

## Linux / WSL2

```bash
export CRYPTOCHAOS_DIV2K_PATH=/path/to/DIV2K_train_HR
python -m pip install -e '.[test]'          # add ,gpu for the optional kernels
python -m cryptochaos doctor
python -m pytest
python -m cryptochaos benchmark --config configs/principal.toml --resume
```

If the images are on a cloud-synchronized folder (for example, OneDrive), make sure they are
stored locally first. `doctor` reports how many files are hydrated.

## What each command does

| Command | Purpose |
|---|---|
| `doctor` | Hardware, interpreter, crypto backends, GPU, dataset, power state, and actionable fixes |
| `manifest` | Rebuild a dataset manifest with file and decoded-payload hashes |
| `benchmark --config X [--resume] [--passes a,b]` | Run an experiment. A resume refuses any identity mismatch |
| `report --run DIR [--paper-dir DIR]` | Summaries, bootstrap intervals, figures, LaTeX tables and macros (written to `DIR/report/`; optionally copied to a manuscript directory) |
| `verify-artifact --run DIR` | Completeness against enumerated expected rows (all passes except stages and memory, which are reported with a warning), SHA-256 integrity of the run and of `report/`, code identity, and a scan of CSV column names and JSON keys for secret-looking names (values are not scanned). For a directory without a run configuration, such as `results/gpu`, it checks SHA-256 sums only |
| `benchmark-metrics --backend {cpu,torch,triton,all}` | CPU vs GPU diagnostic-kernel timings with parity checks |
| `analyze-maps` | Exhaustive analysis of the configuration D maps |

## Code identity of the published runs

Each run's `run.json` stores `measurement_source_sha256`, a SHA-256 over every module in
`src/cryptochaos/` except reporting, CLI, artifact checking, and the GPU kernels. `verify-artifact`
recomputes it for the current code. If the code has changed since the run, it searches this
repository's history for the commit whose measurement code has the recorded hash and names it. The
development commit recorded in `run.json` (`711dbf8`; `1b0a91a` for the GPU record) is not part of
this repository's history.

## Timing hygiene

Close other applications, keep the machine on AC power, and do not run other workloads during the
timing passes. The runner keeps Windows awake, records CPU load once per second, and runs all
CPU-parallel passes after the timing passes. Expect roughly 1-2 hours for `principal.toml` on an
8-core laptop.

## NIST STS reference vectors

`tests/vectors/sts/sts_reference.json` was produced with NIST STS 2.1.2 (`sts-2_1_2.zip`, SHA-256
`0238d2f1…9157`) compiled with `python -m ziglang cc -O2 -I include src/*.c -o assess.exe`, using
`scripts/build_sts_vectors.py`. Routine tests compare against the stored vectors and do not need the
binary.

## Optional GPU kernels

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python -m pip install -e ".[test]"
.\.venv-gpu\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
.\.venv-gpu\Scripts\python -m pip install triton-windows
.\.venv-gpu\Scripts\python -m pytest tests/test_gpu.py
.\.venv-gpu\Scripts\python -m cryptochaos benchmark-metrics --backend all --sizes 1048576,8388608,67108864,268435456 --out results/gpu/metrics.csv
python scripts/make_gpu_table.py --gpu-dir results/gpu --out paper/generated/gpu
```

The published GPU observations are `results/gpu/metrics.csv` (timings) and
`results/gpu/parity.json` (exact CPU/GPU agreement). `metrics.csv` holds one median per backend and
size; the number of repeats was not recorded (the command-line default is 20). `parity.json` and the
GPU `environment.json` were written by a session script that is not part of this repository; the
parity cases are those of `gpu_metrics.check_parity`, which `tests/test_gpu.py` also runs. Triton
compilation time was not measured: the first-call column reflects a warm compilation cache. The
`SHA256SUMS` in `results/gpu` was added on 2026-11-02, after the measurements.

The published GPU results used torch 2.15.0+cu132 and triton-windows 3.9.0.post30 on a GeForce RTX
5060 Ti. Choose the CUDA build that matches your driver. The GPU kernels compute only descriptive
reductions and run in a separate environment so that GPU libraries never load into the CPU timing
process. The environment is large; create it outside cloud-synchronized folders.
