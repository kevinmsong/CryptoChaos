"""Command-line interface: ``python -m cryptochaos <command>``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_doctor(_a) -> int:
    from . import environment
    from .dataset import is_hydrated, read_manifest
    from .kem import available_kem_backends

    env = environment.capture()
    print(f"CryptoChaos doctor  ({env['captured_at_utc']})")
    print(f"  platform      {env['platform']}")
    print(f"  cpu           {env['cpu']} ({env['cpu_physical_cores']} cores / {env['cpu_logical_cores']} threads)")
    print(f"  ram           {env['ram_bytes'] / 2**30:.1f} GiB")
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")
    b = env["backends"]
    print(f"  openssl       python: {b['python_openssl']}")
    for name, prov in b["kem_backends"].items():
        print(f"  kem {name:9s} {prov['library']} {prov['library_version']}  [{prov['implementation']}]")
    missing = [n for n in ("openssl", "pqcrypto", "kyber-py") if n not in available_kem_backends()]
    p = env["power"]
    print(f"  power         on AC: {p.get('on_ac_power')}  scheme: {p.get('power_scheme')}")
    s = env["source"]
    print(f"  source        commit {s['git_commit']} dirty={s['git_dirty']}")
    g = env["gpu"]
    if g["gpus"]:
        for d in g["gpus"]:
            print(f"  gpu           {d}")
    else:
        print("  gpu           no NVIDIA GPU visible to nvidia-smi")
    t = g.get("torch") or {}
    print(f"  torch         {t.get('version', t.get('unavailable'))}  cuda={t.get('cuda_available')}")
    print(f"  triton        {g.get('triton')}")
    actions = []
    if missing:
        actions.append(f"install KEM backends {missing}: python -m pip install -e .")
    data = os.environ.get("CRYPTOCHAOS_DIV2K_PATH")
    if data and Path(data).is_dir():
        pngs = sorted(Path(data).glob("*.png"))
        hydrated = sum(is_hydrated(x) for x in pngs)
        print(f"  dataset       {data}: {len(pngs)} PNG files, {hydrated} hydrated locally")
        man = Path("configs/manifests/div2k_train_hr.csv")
        if man.exists():
            names = {r.filename for r in read_manifest(man)}
            print(f"  manifest      {man}: {len(names)} images; missing locally: {len(names - {x.name for x in pngs})}")
        if hydrated < len(pngs):
            actions.append("hydrate the OneDrive dataset (Always keep on this device) before benchmarking")
    else:
        actions.append("set CRYPTOCHAOS_DIV2K_PATH to the DIV2K_train_HR directory")
    if not t.get("cuda_available"):
        if sys.platform == "win32":  # PyPI serves CPU-only torch wheels on Windows
            actions.append("optional GPU diagnostics: install a CUDA build of torch from the PyTorch index, then "
                           "triton-windows, in a separate environment (docs/reproduction.md, Optional GPU kernels)")
        else:
            actions.append("optional GPU diagnostics: python -m pip install -e .[gpu] with a CUDA build of torch")
    if p.get("on_ac_power") is False:
        actions.append("connect AC power before timing runs")
    print("  actions       " + ("none" if not actions else "\n                ".join(actions)))
    return 0


def cmd_manifest(a) -> int:
    from .dataset import build_manifest, write_manifest
    from .environment import repo_root

    recs = build_manifest(Path(a.dataset), repo_root())
    write_manifest(recs, Path(a.out))
    print(f"wrote {len(recs)} records to {a.out}")
    return 0


def cmd_benchmark(a) -> int:
    from .benchmark import run

    out = run(a.config, resume=a.resume, passes=a.passes.split(",") if a.passes else None)
    print(f"run directory: {out}")
    return 0


def cmd_report(a) -> int:
    from .reporting import build_report

    out = build_report(Path(a.run), paper_dir=Path(a.paper_dir) if a.paper_dir else None,
                       bootstrap=a.bootstrap)
    print(f"report written to {out}")
    return 0


def cmd_verify(a) -> int:
    from .artifact import verify_run

    report = verify_run(Path(a.run))
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


def cmd_benchmark_metrics(a) -> int:
    from .gpu_metrics import benchmark_backends

    rows = benchmark_backends(a.backend, sizes=[int(s) for s in a.sizes.split(",")], repeats=a.repeats,
                              out=Path(a.out) if a.out else None)
    for r in rows:
        print(r)
    return 0


def cmd_analyze_maps(a) -> int:
    from . import map_analysis

    res = {"legacy": map_analysis.legacy_analysis()}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text if not a.out else f"wrote {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cryptochaos", description="CryptoChaos experiments")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report capabilities and missing dependencies").set_defaults(fn=cmd_doctor)
    m = sub.add_parser("manifest", help="build a dataset manifest with hashes")
    m.add_argument("--dataset", required=True)
    m.add_argument("--out", default="configs/manifests/div2k_train_hr.csv")
    m.set_defaults(fn=cmd_manifest)
    b = sub.add_parser("benchmark", help="run an experiment configuration")
    b.add_argument("--config", required=True)
    b.add_argument("--resume", action="store_true")
    b.add_argument("--passes", default=None, help="comma-separated subset of passes")
    b.set_defaults(fn=cmd_benchmark)
    r = sub.add_parser("report", help="summaries, figures, and LaTeX tables from a run")
    r.add_argument("--run", required=True)
    r.add_argument("--paper-dir", default=None)
    r.add_argument("--bootstrap", type=int, default=10000)
    r.set_defaults(fn=cmd_report)
    v = sub.add_parser("verify-artifact", help="check completeness, identity, and hashes of a run")
    v.add_argument("--run", required=True)
    v.set_defaults(fn=cmd_verify)
    g = sub.add_parser("benchmark-metrics", help="CPU vs GPU diagnostic-kernel benchmark")
    g.add_argument("--backend", default="all", choices=["cpu", "torch", "triton", "all"])
    g.add_argument("--sizes", default="1048576,8388608,67108864")
    g.add_argument("--repeats", type=int, default=20)
    g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_benchmark_metrics)
    am = sub.add_parser("analyze-maps", help="exhaustive analysis of the configuration D maps")
    am.add_argument("--out", default=None)
    am.set_defaults(fn=cmd_analyze_maps)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
