"""Summaries, paired statistics, figures, and LaTeX tables generated directly from raw run data.

Statistical conventions (see docs/methodology.md):

* The image is the unit for corpus comparisons. Repeated timings of one image are summarized by
  their median; paired comparisons use per-image differences and ratios of those medians.
* Uncertainty: two-level (hierarchical) bootstrap that resamples images, then repeats within
  each image, 95 % percentile intervals, fixed seed.
* Corpus throughput is reported two ways, labelled: total bytes / total time (aggregate) and the
  median of per-image rates. Units: MiB = 2**20 bytes; times in milliseconds or microseconds.
* Pass rates use exact Clopper-Pearson 95 % intervals. No equivalence is claimed from overlap.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import metrics as M  # noqa: E402
from . import statistical_tests as ST  # noqa: E402
from .protocol import SUITES, envelope_overhead, header_length  # noqa: E402

MIB = 2**20
SUITE_ORDER = list("ABCDEF")
# Validated categorical order (dataviz reference palette, light surface; adjacent CVD dE >= 9.1).
# Okabe-Ito hues with Paul Tol's purple for E: every pair stays distinct under simulated protanopia and
# deuteranopia (worst OKLab delta-E x100 = 10.3); markers and line styles repeat the encoding for print.
SUITE_COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#D55E00", "E": "#AA3377", "F": "#56B4E9"}
SUITE_MARKERS = {"A": "o", "B": "s", "C": "^", "D": "v", "E": "D", "F": "P"}
SUITE_LINES = {"A": "-", "B": "--", "C": "-.", "D": ":", "E": "-", "F": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COMPARISONS = [("B", "A"), ("C", "A"), ("D", "C"), ("E", "C"), ("F", "E")]
# legend labels use the wording of the article's ablation table
SUITE_LABELS = {"A": "AES-256-GCM, pre-established key", "B": "ChaCha20-Poly1305, pre-established key",
                "C": "ML-KEM-1024 + HKDF-SHA256 + AES-256-GCM", "D": "C + truncated-seed chaotic value (negative control)",
                "E": "C + scrypt password key", "F": "E + chaotic value (CryptoChaos)"}
UNIFORMITY_MIN_SEQUENCES = 55  # SP 800-22 section 4.2.2: the uniformity check needs at least 55 sequences
MAP_LABELS = {"logistic": "Logistic", "chebyshev": "Chebyshev", "tent": "Tent", "henon": "H\\'enon"}


# ---------------------------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------------------------

def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def uniformity_or_nan(p: np.ndarray) -> float:
    """SP 800-22 4.2.2 p-value uniformity, or NaN below the 55-sequence minimum."""
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p)]
    return ST.uniformity_p_value(p) if p.size >= UNIFORMITY_MIN_SEQUENCES else float("nan")


def per_image_matrix(df: pd.DataFrame, suite: str, col: str, images: list[str]) -> np.ndarray:
    sub = df[df.suite == suite].pivot_table(index="image_id", columns="repeat", values=col, aggfunc="first")
    return sub.reindex(images).to_numpy(dtype=np.float64)


def hierarchical_bootstrap_paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int,
                                  chunk: int = 250) -> dict[str, Any]:
    """Paired comparison of suites a and b over images (rows) with repeats (columns).

    Statistics on per-image medians: median and mean difference (a - b) and geometric-mean
    ratio a / b. Resampling: images with replacement, then repeats within each image and suite.
    """
    ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    a, b = a[ok], b[ok]
    n, r = a.shape
    ma, mb = np.median(a, axis=1), np.median(b, axis=1)
    d = ma - mb
    lr = np.log(ma / mb)
    point = {"median_diff": float(np.median(d)), "mean_diff": float(np.mean(d)), "gm_ratio": float(np.exp(lr.mean())),
             "n_images": int(n), "repeats": int(r), "frac_a_slower": float(np.mean(d > 0))}
    rng = np.random.Generator(np.random.PCG64(seed))
    boots = {k: [] for k in ("median_diff", "mean_diff", "gm_ratio")}
    done = 0
    while done < n_boot:
        c = min(chunk, n_boot - done)
        img = rng.integers(0, n, (c, n))
        ra = rng.integers(0, r, (c, n, r))
        rb = rng.integers(0, r, (c, n, r))
        ba = np.median(np.take_along_axis(a[img], ra, axis=2), axis=2)
        bb = np.median(np.take_along_axis(b[img], rb, axis=2), axis=2)
        dd = ba - bb
        boots["median_diff"].append(np.median(dd, axis=1))
        boots["mean_diff"].append(dd.mean(axis=1))
        boots["gm_ratio"].append(np.exp(np.log(ba / bb).mean(axis=1)))
        done += c
    out = dict(point)
    for k, v in boots.items():
        v = np.concatenate(v)
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    out["n_boot"] = n_boot
    return out


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def median_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, x.size, (n_boot, x.size))
    med = np.median(x[idx], axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------------------------

def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$"))


def tex_table(path: Path, header: list[str], rows: list[list[str]], colspec: str, note: str = "") -> None:
    lines = [f"% generated by cryptochaos.reporting -- do not edit", r"\begin{tabular}{" + colspec + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(f"% {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:,.{digits}f}".replace(",", "{,}")


def fmt_ci(p: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{fmt(p, digits)} [{fmt(lo, digits)}, {fmt(hi, digits)}]"


_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def macro_name(text: str) -> str:
    """LaTeX command names allow letters only: spell out digits and drop other characters."""
    out = []
    for word in "".join(ch if ch.isalnum() else " " for ch in text).split():
        out.append("".join(_DIGITS[int(ch)] if ch.isdigit() else ch for ch in word[:1].upper() + word[1:]))
    return "".join(out)


def is_true(series: pd.Series) -> pd.Series:
    """Strict boolean parse of a CSV column: only the literal True counts (NaN, '', 2, ... do not)."""
    return series.astype(str).str.strip().str.lower().eq("true")


class Macros:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.items: dict[str, str] = {}

    def add(self, name: str, value: Any) -> None:
        name = macro_name(name)
        if name in self.items and self.items[name] != str(value):
            raise ValueError(f"macro name collision: {self.prefix}{name}")
        self.items[name] = str(value)

    def write(self, path: Path) -> None:
        out = ["% generated by cryptochaos.reporting -- do not edit"]
        for k, v in sorted(self.items.items()):
            out.append(f"\\newcommand{{\\{self.prefix}{k}}}{{{v}}}")
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
        "font.family": "serif", "pdf.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False, "figure.constrained_layout.use": True,
    })


def _save(fig, out: Path, name: str) -> list[str]:
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        paths.append(p.name)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------

def report_timing(raw: Path, out: Path, macros: Macros, n_boot: int, seed: int) -> dict[str, Any]:
    df = pd.read_csv(raw / "timing.csv")
    res: dict[str, Any] = {"rows": len(df), "errors": int(df["error"].notna().sum()),
                           "not_ok": int((~is_true(df["ok"])).sum())}
    df = df[is_true(df["ok"])]
    df["roundtrip_ns"] = df["seal_ns"] + df["open_ns"]
    images = sorted(df.image_id.astype(str).unique())
    df["image_id"] = df["image_id"].astype(str)
    rows = []
    summary = []
    for code in SUITE_ORDER:
        s = df[df.suite == code]
        if s.empty:
            continue
        per_img = s.groupby("image_id").agg(seal=("seal_ns", "median"), open=("open_ns", "median"),
                                             rt=("roundtrip_ns", "median"), nbytes=("payload_bytes", "first"))
        rec = {"suite": code, "n_images": int(per_img.shape[0]), "n_obs": int(len(s))}
        for op, col in (("seal", "seal_ns"), ("open", "open_ns"), ("roundtrip", "roundtrip_ns")):
            v = s[col].to_numpy() / 1e6
            rec[f"{op}_median_ms"] = float(np.median(per_img[op if op != "roundtrip" else "rt"]) / 1e6)
            rec[f"{op}_p95_ms"] = float(np.percentile(v, 95))
            rec[f"{op}_p99_ms"] = float(np.percentile(v, 99))
        rec["seal_mib_s_aggregate"] = float(s.payload_bytes.sum() / MIB / (s.seal_ns.sum() / 1e9))
        rec["open_mib_s_aggregate"] = float(s.payload_bytes.sum() / MIB / (s.open_ns.sum() / 1e9))
        rec["seal_mib_s_median_image"] = float(np.median(per_img.nbytes / MIB / (per_img.seal / 1e9)))
        rec["open_mib_s_median_image"] = float(np.median(per_img.nbytes / MIB / (per_img.open / 1e9)))
        rt_ratio = (s.roundtrip_ns / s.groupby("image_id").roundtrip_ns.transform("median")).to_numpy()
        rec["roundtrip_within_image_p95_ratio"] = float(np.percentile(rt_ratio, 95))
        rec["roundtrip_within_image_p99_ratio"] = float(np.percentile(rt_ratio, 99))
        if s["cold_seal_ns"].notna().any():
            cold = s[s["cold_seal_ns"].notna()]
            cold_ok = cold[is_true(cold["cold_ok"])]
            rec["cold_not_ok"] = int(len(cold) - len(cold_ok))
            rec["cold_seal_median_ms"] = float(cold_ok.groupby("image_id").cold_seal_ns.median().median() / 1e6)
            rec["cold_open_median_ms"] = float(cold_ok.groupby("image_id").cold_open_ns.median().median() / 1e6)
        rec["overhead_bytes"] = int(s.overhead_bytes.iloc[0])
        summary.append(rec)
        rows.append([code, fmt(rec["seal_median_ms"]), fmt(rec["open_median_ms"]), fmt(rec["roundtrip_p95_ms"]),
                     fmt(rec["seal_mib_s_aggregate"], 0), fmt(rec["open_mib_s_aggregate"], 0),
                     fmt(rec.get("cold_seal_median_ms", float("nan")))])
    res["suites"] = summary
    pd.DataFrame(summary).to_csv(out / "timing_summary.csv", index=False)
    tex_table(out / "tab_timing.tex",
              ["Config.", "Seal (ms)", "Open (ms)", "Round trip p95 (ms)", "Seal MiB/s", "Open MiB/s",
               "Cold seal (ms)"], rows, "lrrrrrr",
              "medians of per-image medians; MiB/s = total plaintext bytes / total time")
    comps = []
    crow = []
    for i, (a, b) in enumerate(COMPARISONS):
        if a not in df.suite.values or b not in df.suite.values:
            continue
        rec = {"a": a, "b": b}
        for op, col in (("seal", "seal_ns"), ("open", "open_ns"), ("roundtrip", "roundtrip_ns")):
            r = hierarchical_bootstrap_paired(per_image_matrix(df, a, col, images) / 1e6,
                                              per_image_matrix(df, b, col, images) / 1e6, n_boot, seed + i)
            rec[op] = r
        comps.append(rec)
        rt = rec["roundtrip"]
        # three decimals below 10 ms (ratios: four below 10), fewer above, so the table fits one column
        crow.append([f"{a} $-$ {b}",
                     fmt_ci(rt["median_diff"], rt["median_diff_lo"], rt["median_diff_hi"],
                            3 if abs(rt["median_diff"]) < 10 else 1),
                     fmt_ci(rt["mean_diff"], rt["mean_diff_lo"], rt["mean_diff_hi"],
                            3 if abs(rt["mean_diff"]) < 10 else 1),
                     fmt_ci(rt["gm_ratio"], rt["gm_ratio_lo"], rt["gm_ratio_hi"], 4 if rt["gm_ratio"] < 10 else 2),
                     str(rt["n_images"])])
        if (a, b) == ("F", "E"):
            for op in ("seal", "open", "roundtrip"):
                r = rec[op]
                macros.add(f"FE{op.capitalize()}MedDiffMs", fmt(r["median_diff"], 3))
                macros.add(f"FE{op.capitalize()}MedDiffLoMs", fmt(r["median_diff_lo"], 3))
                macros.add(f"FE{op.capitalize()}MedDiffHiMs", fmt(r["median_diff_hi"], 3))
                macros.add(f"FE{op.capitalize()}Ratio", fmt(r["gm_ratio"], 4))
                macros.add(f"FE{op.capitalize()}RatioLo", fmt(r["gm_ratio_lo"], 4))
                macros.add(f"FE{op.capitalize()}RatioHi", fmt(r["gm_ratio_hi"], 4))
            macros.add("FEFracSlower", fmt(100 * rec["roundtrip"]["frac_a_slower"], 1))
            macros.add("FERoundtripPct", fmt(100 * (rec["roundtrip"]["gm_ratio"] - 1), 1))
        if (a, b) == ("E", "C"):
            macros.add("ECRoundtripRatio", fmt(rec["roundtrip"]["gm_ratio"], 1))
            macros.add("ECRoundtripMedDiffMs", fmt(rec["roundtrip"]["median_diff"], 1))
        rt = rec["roundtrip"]
        macros.add(f"Cmp{a}{b}MedDiffMs", fmt(rt["median_diff"], 3))
        macros.add(f"Cmp{a}{b}MedDiffLoMs", fmt(rt["median_diff_lo"], 3))
        macros.add(f"Cmp{a}{b}MedDiffHiMs", fmt(rt["median_diff_hi"], 3))
        macros.add(f"Cmp{a}{b}Ratio", fmt(rt["gm_ratio"], 3))
        macros.add(f"Cmp{a}{b}FracSlower", fmt(100 * rt["frac_a_slower"], 1))
    res["comparisons"] = comps
    (out / "paired_comparisons.json").write_text(json.dumps(comps, indent=1), encoding="utf-8")
    n_images = {r[-1] for r in crow}
    if len(n_images) == 1:  # the same images in every comparison: state the count once, in the caption
        crow = [r[:-1] for r in crow]
        macros.add("PairedImages", n_images.pop())
        tex_table(out / "tab_paired.tex", ["Comparison", "Median diff. (ms)", "Mean diff. (ms)", "GM ratio"],
                  crow, "lrrr", "seal+open round trip; 95% hierarchical bootstrap CIs")
    else:
        macros.add("PairedImages", "the listed number of")
        tex_table(out / "tab_paired.tex", ["Comparison", "Median diff. (ms)", "Mean diff. (ms)", "GM ratio", "Images"],
                  crow, "lrrrr", "seal+open round trip; 95% hierarchical bootstrap CIs")
    for rec in summary:
        macros.add(f"Seal{rec['suite']}Ms", fmt(rec["seal_median_ms"]))
        macros.add(f"Open{rec['suite']}Ms", fmt(rec["open_median_ms"]))
        macros.add(f"SealMiBs{rec['suite']}", fmt(rec["seal_mib_s_aggregate"], 0))
        macros.add(f"OpenMiBs{rec['suite']}", fmt(rec["open_mib_s_aggregate"], 0))
        macros.add(f"RoundtripPNinetyFive{rec['suite']}Ms", fmt(rec["roundtrip_p95_ms"]))
        macros.add(f"RoundtripMedian{rec['suite']}Ms", fmt(rec["roundtrip_median_ms"]))
        macros.add(f"RoundtripWithinImagePNinetyFiveRatio{rec['suite']}",
                   fmt(rec["roundtrip_within_image_p95_ratio"], 2))
        macros.add(f"SealMiBsMedianImage{rec['suite']}", fmt(rec["seal_mib_s_median_image"], 0))
        macros.add(f"OpenMiBsMedianImage{rec['suite']}", fmt(rec["open_mib_s_median_image"], 0))
        if "cold_seal_median_ms" in rec:
            macros.add(f"ColdSeal{rec['suite']}Ms", fmt(rec["cold_seal_median_ms"]))
            macros.add(f"ColdOpen{rec['suite']}Ms", fmt(rec["cold_open_median_ms"]))
    macros.add("TimingImages", len(images))
    macros.add("TimingRepeats", int(df["repeat"].max()))
    macros.add("TimingObservations", f"{len(df):,}".replace(",", "{,}"))
    _fig_paired(comps, out)
    return res


def _fig_paired(comps: list[dict], out: Path) -> None:
    if not comps:
        return
    _style()
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    labels = []
    for i, rec in enumerate(comps):
        r = rec["roundtrip"]
        y = len(comps) - 1 - i
        ax.errorbar(r["gm_ratio"], y, xerr=[[r["gm_ratio"] - r["gm_ratio_lo"]], [r["gm_ratio_hi"] - r["gm_ratio"]]],
                    fmt="o", ms=4, color=SUITE_COLORS[rec["a"]], ecolor=INK2, elinewidth=1, capsize=2)
        label = f"{r['gm_ratio']:.3f}" if r["gm_ratio"] < 10 else f"{r['gm_ratio']:.1f}"
        ax.annotate(label, (r["gm_ratio"], y), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=INK2)
        labels.append((y, f"{rec['a']} vs {rec['b']}"))
    ax.set_xscale("log")
    ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_yticks([y for y, _ in labels], [t for _, t in labels])
    ax.set_xlabel("Round-trip latency ratio (log scale)")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_paired_ratios")


def report_scaling(raw: Path, out: Path, macros: Macros, n_boot: int, seed: int) -> dict[str, Any]:
    df = pd.read_csv(raw / "scaling.csv")
    df = df[is_true(df["ok"])]
    recs = []
    for (size, code, wl, pol), g in df.groupby(["payload_bytes", "suite", "workload", "password_policy"]):
        rec = {"payload_bytes": int(size), "suite": code, "workload": wl, "password_policy": pol, "n": len(g)}
        for op in ("seal", "open"):
            med, lo, hi = median_ci(g[f"{op}_ns"].to_numpy() / 1e6, n_boot // 5, seed)
            rec.update({f"{op}_ms": med, f"{op}_ms_lo": lo, f"{op}_ms_hi": hi,
                        f"{op}_mib_s": size / MIB / (med / 1e3)})
        recs.append(rec)
    sdf = pd.DataFrame(recs)
    sdf.to_csv(out / "scaling_summary.csv", index=False)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, op in zip(axes, ("seal", "open")):
        for code in SUITE_ORDER:
            g = sdf[(sdf.suite == code) & (sdf.workload == "warm") & (sdf.password_policy.isin(["none", "per_message"]))]
            if g.empty:
                continue
            g = g.sort_values("payload_bytes")
            x = g.payload_bytes / 1024
            ax.plot(x, g[f"{op}_ms"], color=SUITE_COLORS[code], marker=SUITE_MARKERS[code], ms=3.5,
                    ls=SUITE_LINES[code], lw=1.2, label=f"{code}: {SUITE_LABELS.get(code, SUITES[code].name)}")
            ax.fill_between(x, g[f"{op}_ms_lo"], g[f"{op}_ms_hi"], color=SUITE_COLORS[code], alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        sizes = sorted(sdf.payload_bytes.unique())
        ax.set_xticks([s / 1024 for s in sizes], [_size_label(int(s)) for s in sizes], rotation=30)
        ax.minorticks_off()
        ax.set_xlabel("Payload size (log scale)")
        ax.set_title("Seal (encrypt)" if op == "seal" else "Open (decrypt)", loc="left")
    axes[0].set_ylabel("Median latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    _save(fig, out, "fig_scaling")
    rows = []
    for size in sorted(sdf.payload_bytes.unique()):
        cells = []
        for code in SUITE_ORDER:
            g = sdf[(sdf.payload_bytes == size) & (sdf.suite == code) & (sdf.workload == "warm")
                    & sdf.password_policy.isin(["none", "per_message"])]
            cells.append(fmt(float(g.seal_ms.iloc[0]) + float(g.open_ms.iloc[0]), 3) if not g.empty else "--")
        rows.append([_size_label(int(size))] + cells)
    tex_table(out / "tab_scaling.tex", ["Payload"] + SUITE_ORDER, rows, "l" + "r" * len(SUITE_ORDER),
              "median seal + median open (ms), warm recipient, per-message password policy")
    sess = sdf[(sdf.password_policy == "session")]
    warm_c = sdf[(sdf.suite == "C") & (sdf.workload == "warm") & (sdf.password_policy == "none")]
    rows = []
    for size in sorted(sess.payload_bytes.unique()):
        g = warm_c[warm_c.payload_bytes == size]
        cells = [fmt(float(g.seal_ms.iloc[0]) + float(g.open_ms.iloc[0]), 3) if not g.empty else "--"]
        for code in ("D", "E", "F"):
            g = sess[(sess.payload_bytes == size) & (sess.suite == code)]
            cells.append(fmt(float(g.seal_ms.iloc[0]) + float(g.open_ms.iloc[0]), 3) if not g.empty else "--")
        rows.append([_size_label(int(size))] + cells)
    if rows:
        tex_table(out / "tab_scaling_session.tex", ["Payload", "C", "D", "E", "F"], rows, "lrrrr",
                  "median seal + open (ms); D-F with session-cached password material, C (no password) for "
                  "comparison")
    return {"rows": len(df)}


def _size_label(n: int) -> str:
    return f"{n // MIB} MiB" if n >= MIB else f"{n // 1024} KiB"


def report_kdf(raw: Path, out: Path, macros: Macros, n_boot: int, seed: int) -> dict[str, Any]:
    df = pd.read_csv(raw / "kdf_sensitivity.csv")
    df = df[is_true(df["ok"])]
    recs = []
    for (n, code), g in df.groupby(["log2_n", "suite"]):
        rt = (g.seal_ns + g.open_ns).to_numpy() / 1e6
        med, lo, hi = median_ci(rt, n_boot // 5, seed)
        recs.append({"log2_n": int(n), "suite": code, "roundtrip_ms": med, "lo": lo, "hi": hi, "n": len(g),
                     "memory_mib": 128 * 8 * 2**int(n) / MIB})
    kdf = pd.DataFrame(recs)
    kdf.to_csv(out / "kdf_summary.csv", index=False)
    _style()
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    for code in ("E", "F"):
        g = kdf[kdf.suite == code].sort_values("log2_n")
        if g.empty:
            continue
        ax.errorbar(g.log2_n + (0.06 if code == "F" else -0.06), g.roundtrip_ms,
                    yerr=[g.roundtrip_ms - g.lo, g.hi - g.roundtrip_ms], color=SUITE_COLORS[code],
                    marker=SUITE_MARKERS[code], ms=4, lw=1.2, ls=SUITE_LINES[code], capsize=2,
                    label={"E": "E: scrypt password key", "F": "F: E + chaotic value (CryptoChaos)"}[code])
    ax.set_yscale("log")
    ax.set_xticks(sorted(kdf.log2_n.unique()),
                  [f"$2^{{{n}}}$\n({int(128 * 8 * 2**n / MIB)} MiB)" for n in sorted(kdf.log2_n.unique())])
    ax.set_xlabel("scrypt N (core memory), r = 8, p = 1")
    ax.set_ylabel("Median seal + open (ms, log scale)")
    ax.legend(frameon=False, loc="lower right")  # latency rises with N: the lower right stays empty
    _save(fig, out, "fig_kdf_sensitivity")
    rows = []
    for n in sorted(kdf.log2_n.unique()):
        e = kdf[(kdf.log2_n == n) & (kdf.suite == "E")]
        f = kdf[(kdf.log2_n == n) & (kdf.suite == "F")]
        rows.append([f"$2^{{{n}}}$", fmt(128 * 8 * 2**n / MIB, 0),
                     fmt_ci(float(e.roundtrip_ms.iloc[0]), float(e.lo.iloc[0]), float(e.hi.iloc[0]), 1) if len(e) else "--",
                     fmt_ci(float(f.roundtrip_ms.iloc[0]), float(f.lo.iloc[0]), float(f.hi.iloc[0]), 1) if len(f) else "--"])
    tex_table(out / "tab_kdf.tex", ["$N$", "Memory (MiB)", "E (ms)", "F (ms)"], rows, "lrrr",
              "median seal+open, 1 MiB payload, per-message derivation; 95% bootstrap CI")
    return {"rows": len(df)}


STAGE_LABELS = {
    "kem_keygen": "ML-KEM keygen", "kem_encapsulate": "ML-KEM encaps.",
    "kem_decapsulate": "ML-KEM decaps.", "kem_load_public_key": "ML-KEM public-key load + check",
    "kem_load_secret_key": "ML-KEM secret-key load", "scrypt": "scrypt", "legacy_chaotic_key": "D chaotic value",
    "v2_chaos_value": "F chaotic value", "hkdf_aead_key": "HKDF key schedule",
    "blake3_shared_secret": "BLAKE3(ss)", "aead_seal": "AEAD seal (1 MiB)", "aead_open": "AEAD open (1 MiB)",
    "password_encode": "Password NFC/UTF-8", "envelope_parse": "Envelope parse (1 MiB)",
}


def _stage_label(stage: str, variant: str) -> str:
    """Figure label for a stage: plain words instead of the variant identifiers stored in the data."""
    label = STAGE_LABELS.get(stage, stage)
    if variant in ("v1-maps", "v2-maps", "blake3"):
        return label
    if variant.startswith("log2N="):
        return f"{label} ($N = 2^{{{variant[6:]}}}$)"
    if variant.endswith("-fields"):
        return f"HKDF, {variant[0]} key inputs"
    return f"{label} [{ {'openssl': 'OpenSSL'}.get(variant, variant)}]"


def report_stages(raw: Path, out: Path, macros: Macros) -> dict[str, Any]:
    df = pd.read_csv(raw / "stages.csv")
    g = df.groupby(["stage", "variant"]).ns.agg(["median", lambda x: np.percentile(x, 5),
                                                 lambda x: np.percentile(x, 95), "count"]).reset_index()
    g.columns = ["stage", "variant", "median_ns", "p5_ns", "p95_ns", "n"]
    g.to_csv(out / "stages_summary.csv", index=False)
    rows = []
    for _, r in g.iterrows():
        rows.append([tex_escape(STAGE_LABELS.get(r.stage, r.stage)),
                     tex_escape({'v1-maps': '8-bit maps', 'v2-maps': '64-bit maps'}.get(r.variant, r.variant)), fmt(r.median_ns / 1e3, 1),
                     fmt(r.p95_ns / 1e3, 1), str(int(r.n))])
    tex_table(out / "tab_stages.tex", ["Stage", "Variant", "Median ($\\mu$s)", "p95 ($\\mu$s)", "n"], rows, "llrrr",
              "isolated microbenchmarks; not summable into end-to-end latency")
    sel = g[~g.stage.isin(["envelope_parse", "password_encode", "kem_load_public_key", "kem_load_secret_key"])]
    order =["kem_keygen", "kem_encapsulate", "kem_decapsulate", "blake3_shared_secret", "scrypt",
             "legacy_chaotic_key", "v2_chaos_value", "hkdf_aead_key", "aead_seal", "aead_open"]
    sel = sel.assign(_o=sel.stage.map(lambda s: order.index(s) if s in order else len(order))).sort_values(
        ["_o", "variant"])
    _style()
    fig, ax = plt.subplots(figsize=(3.45, 3.6))
    # the chaotic-value stages are named by configuration; their map variant adds nothing for the reader
    labels = [_stage_label(s, v) for s, v in zip(sel.stage, sel.variant)]
    y = np.arange(len(sel))[::-1]
    ax.barh(y, sel.median_ns / 1e3, height=0.6, color=SUITE_COLORS["A"])
    ax.errorbar(sel.median_ns / 1e3, y, xerr=[np.zeros(len(sel)), (sel.p95_ns - sel.median_ns) / 1e3],
                fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Time per call ($\\mu$s, log scale)")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_stages")
    for _, r in g.iterrows():
        macros.add(f"Stage{macro_name(str(r.stage) + ' ' + str(r.variant))}Us", fmt(r.median_ns / 1e3, 1))
        if r.median_ns >= 1e6:
            macros.add(f"Stage{macro_name(str(r.stage) + ' ' + str(r.variant))}Ms", fmt(r.median_ns / 1e6, 1))
    # per-message key generation plus encapsulation, by backend
    med = g.set_index(["stage", "variant"]).median_ns
    for variant in ("kyber-py", "openssl"):
        if ("kem_keygen", variant) in med.index and ("kem_encapsulate", variant) in med.index:
            ms = (med["kem_keygen", variant] + med["kem_encapsulate", variant]) / 1e6
            macros.add(f"KeygenEncaps{macro_name(variant)}Ms", fmt(ms, 1 if ms >= 1 else 2))
    return {"stages": len(g)}


def report_diagnostics(raw: Path, out: Path, macros: Macros, n_boot: int, seed: int) -> dict[str, Any]:
    df = pd.read_csv(raw / "diagnostics.csv")
    res = {"rows": len(df), "errors": int(df["error"].notna().sum())}
    metrics_cols = [("shannon_entropy_bits_per_byte", 5), ("min_entropy_bits_per_byte", 4), ("chi2_uniform_p", 3),
                    ("bit_ones_fraction", 5), ("lag1_correlation", 5), ("reenc_unequal_byte_pct", 4),
                    ("reenc_mean_abs_diff_pct", 4), ("reenc_bit_diff_pct", 4), ("pt_ct_unequal_byte_pct", 4),
                    ("pt_ct_mean_abs_diff_pct", 3)]
    recs = []
    for code in ["P"] + SUITE_ORDER:
        s = df[df.suite == code]
        if s.empty:
            continue
        rec = {"suite": code, "n": len(s)}
        for col, _ in metrics_cols:
            if col in s and s[col].notna().any():
                v = s[col].to_numpy(dtype=np.float64)
                lo, hi = bootstrap_mean_ci(v, n_boot // 10, seed)
                rec.update({f"{col}_mean": float(np.nanmean(v)), f"{col}_sd": float(np.nanstd(v, ddof=1)),
                            f"{col}_lo": lo, f"{col}_hi": hi})
        recs.append(rec)
    ddf = pd.DataFrame(recs)
    ddf.to_csv(out / "diagnostics_summary.csv", index=False)
    rows = []
    for rec in recs:
        if rec["suite"] == "P":
            continue
        rows.append([rec["suite"], str(rec["n"]), fmt(rec["shannon_entropy_bits_per_byte_mean"], 6),
                     fmt(rec["reenc_unequal_byte_pct_mean"], 4), fmt(rec["reenc_mean_abs_diff_pct_mean"], 4),
                     fmt(rec["reenc_bit_diff_pct_mean"], 4), fmt(rec["pt_ct_unequal_byte_pct_mean"], 4),
                     fmt(rec["lag1_correlation_mean"] * 1e4, 2)])
    rows.append(["ref.", "", "8", fmt(M.UNEQUAL_BYTES_REF, 4), fmt(M.MEAN_ABS_DIFF_REF, 4), "50", "", "0"])
    tex_table(out / "tab_diagnostics.tex",
              ["Config.", "Images", "Entropy (bit/B)", "Re-enc. unequal B (\\%)", "Re-enc. |diff| (\\%)",
               "Re-enc. bits (\\%)", "PT/CT unequal B (\\%)", "Lag-1 corr. ($10^{-4}$)"], rows, "lrrrrrrr",
              "means over images; payload ciphertext only (framing and tag excluded)")
    # STS pass rates
    test_cols = [(n, lab) for n in ST.TEST_NAMES for lab in (("p1", "p2") if n == "serial" else ("p",))]
    srecs = []
    for code in ["P"] + SUITE_ORDER:
        s = df[df.suite == code]
        if s.empty:
            continue
        for name, lab in test_cols:
            status = s[f"sts_{name}_status"]
            ok = status == "ok"
            pv = s.loc[ok, f"sts_{name}_{lab}"].to_numpy(dtype=np.float64)
            k, n = int((pv >= ST.ALPHA).sum()), int(ok.sum())
            lo, hi = clopper_pearson(n - k, n)
            plo, phi = ST.proportion_interval(n) if n else (float("nan"), float("nan"))
            srecs.append({"suite": code, "test": name, "p": lab, "evaluated": n, "not_applicable": int((status == "not_applicable").sum()),
                          "prerequisite_failed": int((status == "prerequisite_failed").sum()),
                          "errors": int((status.isna() | s["error"].notna()).sum()), "rejected": n - k,
                          "rejection_rate": (n - k) / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
                          "pass_proportion": k / n if n else float("nan"), "sp80022_interval_lo": plo,
                          "sp80022_interval_hi": phi, "inside_interval": bool(plo <= k / n <= phi) if n else None,
                          "uniformity_p": uniformity_or_nan(pv)})
    sdf = pd.DataFrame(srecs)
    sdf.to_csv(out / "sts_summary.csv", index=False)
    rows = []
    for name, lab in test_cols:
        cells = []
        for code in SUITE_ORDER:
            r = sdf[(sdf.suite == code) & (sdf.test == name) & (sdf.p == lab)]
            cells.append("--" if r.empty or r.evaluated.iloc[0] == 0 else
                         f"{int(r.rejected.iloc[0])}/{int(r.evaluated.iloc[0])}")
        label = {"frequency": "Frequency", "block_frequency": "Block frequency", "runs": "Runs",
                 "longest_run": "Longest run", "spectral": "Spectral (DFT)",
                 "nonoverlapping_template": "Non-overl. template", "serial": "Serial"}[name]
        rows.append([label + (f" ($P_{lab[-1]}$)" if name == "serial" else "")] + cells)
    n_err = int(df[df.suite.isin(SUITE_ORDER)]["error"].notna().sum())
    tex_table(out / "tab_sts.tex", ["Test"] + SUITE_ORDER, rows, "l" + "r" * len(SUITE_ORDER),
              "rejections at alpha=0.01 / evaluated samples (first 1,000,000 payload-ciphertext bits per image)"
              + (f"; {n_err} samples failed with an error" if n_err else ""))
    # pooled over the ciphertext suites, per test output (SP 800-22 section 4.2 checks)
    pooled = []
    cdf_all = df[df.suite.isin(SUITE_ORDER)]
    for name, lab in test_cols:
        ok = cdf_all[f"sts_{name}_status"] == "ok"
        pv = cdf_all.loc[ok, f"sts_{name}_{lab}"].to_numpy(dtype=np.float64)
        n, rej = int(ok.sum()), int((pv < ST.ALPHA).sum())
        if not n:
            continue
        plo, phi = ST.proportion_interval(n)
        lo, hi = clopper_pearson(rej, n)
        pooled.append({"test": name, "p": lab, "evaluated": n, "rejected": rej, "rejection_rate": rej / n,
                       "ci_lo": lo, "ci_hi": hi, "pass_proportion": 1 - rej / n, "sp80022_interval_lo": plo,
                       "sp80022_interval_hi": phi, "inside_interval": bool(plo <= 1 - rej / n <= phi),
                       "uniformity_p": uniformity_or_nan(pv)})
    pd.DataFrame(pooled).to_csv(out / "sts_pooled_summary.csv", index=False)
    res["sts_pooled"] = pooled
    macros.add("DiagErrors", n_err)
    for r in pooled:
        key = macro_name(r["test"].replace("_", " ") + (" " + r["p"] if r["test"] == "serial" else ""))
        macros.add(f"StsPooled{key}Rejected", r["rejected"])
        macros.add(f"StsPooled{key}UniformityP", fmt(r["uniformity_p"], 3))
    if pooled:
        macros.add("StsPooledPerTest", f"{pooled[0]['evaluated']:,}".replace(",", "{,}"))
        macros.add("StsPooledOutsideInterval", sum(1 for r in pooled if not r["inside_interval"]))
    res["sts"] = srecs
    # the parameters each test applies to a full-length sample, recorded with the results (brief section 8)
    probe = ST.run_battery(ST.as_bits(np.random.default_rng(0).bytes(ST.SAMPLE_BITS // 8)))
    res["sts_parameters"] = {"alpha": ST.ALPHA, "sample_bits": ST.SAMPLE_BITS, **{r.test: r.params for r in probe}}
    # paired F vs E on per-image diagnostics
    pe = []
    for col, digits in metrics_cols[:8]:
        f = df[df.suite == "F"].set_index("image_id")[col]
        e = df[df.suite == "E"].set_index("image_id")[col]
        both = pd.concat([f, e], axis=1, keys=["F", "E"]).dropna()
        if both.empty:
            continue
        d = (both.F - both.E).to_numpy()
        lo, hi = bootstrap_mean_ci(d, n_boot // 10, seed)
        pe.append({"metric": col, "mean_diff_F_minus_E": float(d.mean()), "lo": lo, "hi": hi, "n": len(d)})
    pd.DataFrame(pe).to_csv(out / "diagnostics_F_minus_E.csv", index=False)
    res["F_minus_E"] = pe
    n_img = int(df[df.suite == "F"].shape[0]) if "F" in df.suite.values else 0
    macros.add("DiagImages", n_img)
    if pe:
        macros.add("DiagFEComparisons", len(pe))
        macros.add("DiagFEIntervalsContainZero", sum(1 for r in pe if r["lo"] <= 0 <= r["hi"]))
    cipher = sdf[sdf.suite.isin(SUITE_ORDER) & (sdf.evaluated > 0)]
    if not cipher.empty:
        macros.add("StsRejectMin", int(cipher.rejected.min()))
        macros.add("StsRejectMax", int(cipher.rejected.max()))
        macros.add("StsSamplesPerTest", int(cipher.evaluated.max()))
        macros.add("StsRejectTotal", f"{int(cipher.rejected.sum()):,}".replace(",", "{,}"))
        macros.add("StsEvaluatedTotal", f"{int(cipher.evaluated.sum()):,}".replace(",", "{,}"))
        macros.add("StsRejectPct", fmt(100 * cipher.rejected.sum() / cipher.evaluated.sum(), 2))
        tot_lo, tot_hi = clopper_pearson(int(cipher.rejected.sum()), int(cipher.evaluated.sum()))
        macros.add("StsRejectCiLoPct", fmt(100 * tot_lo, 2))
        macros.add("StsRejectCiHiPct", fmt(100 * tot_hi, 2))
        macros.add("StsCells", int(len(cipher)))
        macros.add("StsCellsInsideInterval", int(cipher.inside_interval.astype(bool).sum()))
        macros.add("StsMinUniformityP", fmt(float(cipher.uniformity_p.min()), 3))
        calib = raw / "calibration.csv"
        if calib.exists():  # ciphertext rejection rate versus the uniform ChaCha20 calibration streams
            cal = pd.read_csv(calib)
            cal = cal[cal.kind == "chacha20"]
            c_rej = c_n = 0
            for name, lab in test_cols:
                ok = cal[f"sts_{name}_status"] == "ok"
                c_n += int(ok.sum())
                c_rej += int((cal.loc[ok, f"sts_{name}_{lab}"] < ST.ALPHA).sum())
            if c_n:
                from scipy.stats import fisher_exact
                ct_rej, ct_n = int(cipher.rejected.sum()), int(cipher.evaluated.sum())
                _, fp = fisher_exact([[ct_rej, ct_n - ct_rej], [c_rej, c_n - c_rej]])
                macros.add("StsCalibRejectPct", fmt(100 * c_rej / c_n, 3))
                macros.add("StsVsCalibFisherP", fmt(float(fp), 2))
    for code in SUITE_ORDER:
        rec = next((r for r in recs if r["suite"] == code), None)
        if rec:
            macros.add(f"ReencUnequal{code}", fmt(rec["reenc_unequal_byte_pct_mean"], 4))
            macros.add(f"ReencAbsDiff{code}", fmt(rec["reenc_mean_abs_diff_pct_mean"], 4))
            macros.add(f"ReencBits{code}", fmt(rec["reenc_bit_diff_pct_mean"], 4))
    return res


def report_auth(raw: Path, out: Path, macros: Macros) -> dict[str, Any]:
    df = pd.read_csv(raw / "auth.csv")
    df["plaintext_released"] = is_true(df["plaintext_released"])
    table = df.groupby(["case", "suite"]).apply(
        lambda g: f"{int((g.outcome == 'rejected').sum())}/{len(g)}", include_groups=False).unstack(fill_value="--")
    case_order = ["valid", "wrong_recipient_key", "wrong_psk", "wrong_password", "legacy_collision_password",
                  "wrong_aad", "header_bitflip", "kem_ct_bitflip", "payload_bitflip", "tag_bitflip", "truncation",
                  "trailing_data", "bad_version", "unknown_suite", "suite_downgrade", "kdf_params_out_of_policy"]
    labels = {"valid": "Valid envelope (control)", "wrong_recipient_key": "Wrong recipient key",
              "wrong_psk": "Wrong pre-established key", "wrong_password": "Wrong password",
              "legacy_collision_password": "Colliding passphrase", "wrong_aad": "Wrong associated data",
              "header_bitflip": "Header bit flip", "kem_ct_bitflip": "KEM-ciphertext bit flip",
              "payload_bitflip": "Payload bit flip", "tag_bitflip": "Tag bit flip", "truncation": "Truncation",
              "trailing_data": "Trailing data", "bad_version": "Unsupported version", "unknown_suite": "Unknown configuration",
              "suite_downgrade": "Configuration downgrade to C",
              "kdf_params_out_of_policy": "scrypt cost outside allowed range"}
    rows = []
    for case in case_order:
        if case not in table.index:
            continue
        rows.append([labels[case]] + [table.loc[case].get(c, "--") for c in SUITE_ORDER])
    tex_table(out / "tab_auth.tex", ["Case (rejected/trials)"] + SUITE_ORDER, rows, "l" + "c" * len(SUITE_ORDER),
              "valid row counts rejections of correct envelopes (expected 0)")
    bad = df[(df.case != "valid") & (df.outcome == "accepted")]
    unexpected = bad[~((bad.suite == "D") & (bad.case == "legacy_collision_password"))]
    valid_rejected = df[(df.case == "valid") & (df.outcome != "accepted")]
    res = {"trials": len(df), "accepted_modified": len(bad), "unexpected_acceptances": len(unexpected),
           "valid_rejected": len(valid_rejected),
           "legacy_D_collision_accepted": int(((df.suite == "D") & (df.case == "legacy_collision_password")
                                               & (df.outcome == "accepted")).sum())}
    macros.add("AuthTrials", f"{len(df):,}".replace(",", "{,}"))
    macros.add("AuthUnexpectedAccept", len(unexpected))
    macros.add("AuthLegacyAccepted", res["legacy_D_collision_accepted"])
    return res


def report_keysens(raw: Path, out: Path, macros: Macros) -> dict[str, Any]:
    df = pd.read_csv(raw / "key_sensitivity.csv")
    g = df.groupby(["suite", "variation"]).agg(n=("key", "count"), unequal=("unequal_byte_pct", "mean"),
                                               bits=("bit_diff_pct", "mean"), min_bits=("differing_bits", "min"),
                                               max_bits=("differing_bits", "max")).reset_index()
    g.to_csv(out / "key_sensitivity_summary.csv", index=False)
    rows = []
    for _, r in g.iterrows():
        rows.append([r.suite, tex_escape(r.variation), str(int(r.n)), fmt(r.unequal, 3), fmt(r.bits, 3),
                     f"{int(r.min_bits)}--{int(r.max_bits)}"])
    tex_table(out / "tab_key_sensitivity.tex", ["Suite", "Varied input", "n", "Unequal B (\\%)", "Bits (\\%)",
                                                "Differing bits"], rows, "llrrrr",
              "fixture holds every other input fixed; synthetic data for the fixed key/nonce row")
    return {"rows": len(df)}


def report_memory(raw: Path, out: Path, macros: Macros) -> dict[str, Any]:
    df = pd.read_csv(raw / "memory.csv")
    df = df[is_true(df["ok"])]
    g = df.groupby(["payload_label", "suite", "mode"]).agg(
        payload_bytes=("payload_bytes", "first"), baseline=("baseline_rss_bytes", "median"),
        peak=("peak_bytes", "median"), delta=("peak_minus_baseline_rss_bytes", "median"),
        tm=("tracemalloc_peak_bytes", "median"), n=("key", "count"),
        raised=("ops_raised_peak", lambda x: bool(is_true(x).all()))).reset_index()
    g.to_csv(out / "memory_summary.csv", index=False)
    rows = []
    sizes = g.groupby("payload_label").payload_bytes.first()
    for label in sorted(sizes.index, key=lambda s: (not s.startswith("synthetic"), sizes[s], s)):
        n = int(sizes[label])
        shown = tex_escape(f"synthetic, {n // MIB} MiB" if label.startswith("synthetic") and n % MIB == 0
                           else f"{label.replace('-', ' ')}, {n / MIB:.2f} MiB")
        if rows:
            shown = r"\addlinespace " + shown  # one labelled group of suites per payload
        for code in SUITE_ORDER:
            r = g[(g.payload_label == label) & (g.suite == code) & (g["mode"] == "rss")]
            t = g[(g.payload_label == label) & (g.suite == code) & (g["mode"] == "tracemalloc")]
            if r.empty:
                continue
            delta = fmt(r.delta.iloc[0] / MIB, 1)
            if not r.raised.iloc[0]:
                delta = r"$\le$" + delta  # operations did not raise the high-water mark: upper bound
            rows.append([shown, code, fmt(r.baseline.iloc[0] / MIB, 1), fmt(r.peak.iloc[0] / MIB, 1),
                         delta, fmt(t.tm.iloc[0] / MIB, 1) if len(t) else "--"])
            shown = ""
    tex_table(out / "tab_memory.tex", ["Payload", "Config.", "Baseline", "Peak", "Peak $-$ baseline",
                                       "tracemalloc"], rows, "llrrrr",
              "MiB; isolated subprocess; peak = process peak working set (Windows) or ru_maxrss (Linux); "
              "<= marks an upper bound where the operations did not raise the high-water mark")
    return {"rows": len(df)}


def report_calibration(raw: Path, out: Path, macros: Macros) -> dict[str, Any]:
    df = pd.read_csv(raw / "calibration.csv")
    test_cols = [(n, lab) for n in ST.TEST_NAMES for lab in (("p1", "p2") if n == "serial" else ("p",))]
    recs = []
    for kind, g in df.groupby("kind"):
        for name, lab in test_cols:
            ok = g[f"sts_{name}_status"] == "ok"
            p = g.loc[ok, f"sts_{name}_{lab}"].to_numpy(dtype=np.float64)
            n = int(ok.sum())
            k = int((p < ST.ALPHA).sum())
            lo, hi = clopper_pearson(k, n)
            plo, phi = ST.proportion_interval(n) if n else (float("nan"), float("nan"))
            recs.append({"kind": kind, "expected": g.expected.iloc[0], "test": name, "p": lab, "streams": len(g),
                         "evaluated": n, "prerequisite_failed": int((g[f"sts_{name}_status"] == "prerequisite_failed").sum()),
                         "rejected": k, "rejection_rate": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
                         "pass_proportion": 1 - k / n if n else float("nan"), "sp80022_interval_lo": plo,
                         "sp80022_interval_hi": phi,
                         "uniformity_p": uniformity_or_nan(p),
                         "uniformity_status": "ok" if np.isfinite(p).sum() >= UNIFORMITY_MIN_SEQUENCES
                         else "not_applicable (fewer than 55 sequences)"})
    cdf = pd.DataFrame(recs)
    cdf.to_csv(out / "calibration_summary.csv", index=False)
    rows = []
    for name, lab in test_cols:
        r = cdf[(cdf.kind == "chacha20") & (cdf.test == name) & (cdf.p == lab)]
        if r.empty:
            continue
        r = r.iloc[0]
        label = name.replace("_", " ") + (f" ({lab})" if name == "serial" else "")
        rows.append([tex_escape(label), f"{int(r.rejected)}/{int(r.evaluated)}",
                     fmt_ci(100 * r.rejection_rate, 100 * r.ci_lo, 100 * r.ci_hi, 2), fmt(r.uniformity_p, 3)])
    tex_table(out / "tab_calibration.tex", ["Test", "Rejected", "Rate \\% [95\\% CI]", "Uniformity $P$"], rows,
              "lrrr", "uniform ChaCha20 streams; nominal rejection rate 1%")
    neg = cdf[cdf.expected == "reject"].groupby("kind").apply(
        lambda g: pd.Series({"streams": int(g.streams.iloc[0]),
                             "tests_rejecting_all": int((g.rejection_rate == 1.0).sum()),
                             "max_rejection_rate": float(g.rejection_rate.max())}), include_groups=False)
    distinct = df.groupby("kind").bits_sha256.nunique()
    neg["distinct_streams"] = distinct.reindex(neg.index).astype(int)
    neg.to_csv(out / "calibration_negative_controls.csv")
    _style()
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    u = cdf[cdf.kind == "chacha20"].reset_index(drop=True)
    x = np.arange(len(u))
    ax.errorbar(x, 100 * u.rejection_rate, yerr=[100 * (u.rejection_rate - u.ci_lo), 100 * (u.ci_hi - u.rejection_rate)],
                fmt="o", ms=3.5, color=SUITE_COLORS["A"], ecolor=INK2, elinewidth=0.9, capsize=2,
                label="ChaCha20 streams (Clopper-Pearson 95% CI)")
    ax.axhline(100 * ST.ALPHA, color=INK2, ls=":", lw=0.9, label="nominal $\\alpha$ = 1%")
    ax.set_xticks(x, [f"{t.replace('nonoverlapping_template', 'template').replace('block_frequency', 'block freq.').replace('longest_run', 'longest run')}"
                      + (f" {p}" if t == "serial" else "") for t, p in zip(u.test, u.p)], rotation=35, ha="right")
    ax.set_ylabel("Rejection rate (%)")
    ax.set_ylim(bottom=0)
    # above the axes: the interval whiskers can reach any height inside them
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    _save(fig, out, "fig_calibration")
    if not u.empty:
        macros.add("CalibStreams", int(u.streams.iloc[0]))
        macros.add("CalibRateMinPct", fmt(100 * u.rejection_rate.min(), 1))
        macros.add("CalibRateMaxPct", fmt(100 * u.rejection_rate.max(), 1))
        macros.add("CalibAllCoverNominal", "all" if bool(((u.ci_lo <= ST.ALPHA) & (ST.ALPHA <= u.ci_hi)).all())
                   else "not all")
        macros.add("CalibMinUniformityP", fmt(u.uniformity_p.min(), 3))
        inside = (u.sp80022_interval_lo <= u.pass_proportion) & (u.pass_proportion <= u.sp80022_interval_hi)
        macros.add("CalibAllInsideInterval", "all" if bool(inside.all()) else "not all")
        macros.add("CalibIntervalLo", fmt(float(u.sp80022_interval_lo.iloc[0]), 4))
        macros.add("CalibIntervalHi", fmt(float(u.sp80022_interval_hi.iloc[0]), 4))
    mt = cdf[cdf.kind == "mt19937"]
    if not mt.empty:
        macros.add("CalibMtStreams", int(mt.streams.iloc[0]))
        macros.add("CalibMtRejectMin", int(mt.rejected.min()))
        macros.add("CalibMtRejectMax", int(mt.rejected.max()))
    negs = cdf[cdf.expected == "reject"]
    if not negs.empty:
        caught = negs.groupby("kind").rejection_rate.max()
        macros.add("CalibNegativeKinds", int(caught.size))
        macros.add("CalibNegativeDetected", int((caught >= 0.5).sum()))
        if "lcg_low_bit" in distinct.index:
            macros.add("CalibLcgDistinctStreams", int(distinct["lcg_low_bit"]))
    return {"rows": len(df)}


def report_maps(raw: Path, out: Path, macros: Macros) -> dict[str, Any]:
    p = raw / "map_analysis.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    leg = data["legacy"]
    rows = []
    for name in ("logistic", "chebyshev", "tent", "henon"):
        r = leg[name]
        g = r["graph"]
        rows.append([MAP_LABELS[name], f"{r['seeds']:,}".replace(",", "{,}"), f"{r['distinct_outputs']:,}".replace(",", "{,}"),
                     fmt(r["shannon_bits"], 2), fmt(r["min_entropy_bits"], 2), str(g["cycles"]),
                     str(g["longest_cycle"]), str(g["max_transient"])])
    j = leg["joint"]
    rows.append(["Joint (40-bit seed)", "$2^{40}$", f"$2^{{{j['log2_distinct']:.2f}}}$",
                 fmt(j["shannon_bits_uniform_seed"], 2), fmt(j["min_entropy_bits_uniform_seed"], 2), "", "", ""])
    tex_table(out / "tab_legacy_maps.tex", ["Map", "Seeds", "Distinct 32-B outputs", "Shannon (bit)",
                                            "Min-entropy (bit)", "Cycles", "Longest cycle", "Max transient"],
              rows, "lrrrrrrr", "configuration D maps, exhaustive enumeration")
    cs = leg["collision_search"]
    macros.add("LegacyLogDistinct", fmt(j["log2_distinct"], 2))
    macros.add("LegacyMinEntropy", fmt(j["min_entropy_bits_uniform_seed"], 2))
    macros.add("LegacyShannon", fmt(j["shannon_bits_uniform_seed"], 2))
    macros.add("LegacyCollisionExpected", f"{j['expected_random_passphrases_to_first_collision']:,.0f}".replace(",", "{,}"))
    if "distinct_seed_key_collision" in cs:
        c = cs["distinct_seed_key_collision"]
        macros.add("LegacyCollisionA", c["passphrases"][0])
        macros.add("LegacyCollisionB", c["passphrases"][1])
        macros.add("LegacyCollisionFound", f"{c['found_after']:,}".replace(",", "{,}"))
    macros.add("LegacyTentZeroByte", leg["tent"]["all_zero_from_byte"])
    rows = []
    for w, maps_ in data.get("v2_reduced", {}).items():
        for name, r in maps_.items():
            g = r["graph"]
            rows.append([f"{w}", MAP_LABELS.get(name, name), f"{g['states']:,}".replace(",", "{,}"), str(g["cycles"]),
                         str(g["longest_cycle"]), str(g["max_transient"]),
                         f"{r['distinct_outputs']:,}".replace(",", "{,}") if "distinct_outputs" in r else "--"])
    for w, r in data.get("v2_reduced_henon", {}).items():
        g = r["graph"]
        rows.append([f"{w}", MAP_LABELS["henon"], f"{g['states']:,}".replace(",", "{,}"), str(g["cycles"]), str(g["longest_cycle"]),
                     str(g["max_transient"]), f"{r['distinct_outputs']:,}".replace(",", "{,}") if "distinct_outputs" in r else "--"])
    tex_table(out / "tab_v2_maps.tex", ["$W$", "Map", "States", "Cycles", "Longest cycle", "Max transient",
                                        "Distinct outputs"], rows, "rlrrrrr",
              "CryptoChaos maps at reduced word size W (exhaustive); the protocol uses W = 64")
    fw = data.get("v2_full_width", {})
    if fw:
        macros.add("VTwoSamples", fw["samples"])
        macros.add("VTwoCycleSteps", f"{fw['steps_searched']:,}".replace(",", "{,}"))
        macros.add("VTwoCyclesFound", sum(fw["orbits_cycling_within_limit"].values()))
    return {"legacy_log2_distinct": j["log2_distinct"]}


def report_overhead(out: Path, macros: Macros, median_payload: int | None) -> None:
    rows = []
    for code in SUITE_ORDER:
        oh = envelope_overhead(code)
        row = [code, str(header_length(code)), "16", str(oh), fmt(100 * oh / MIB, 3)]
        if median_payload:
            row.append(fmt(100 * oh / median_payload, 4))
        rows.append(row)
        macros.add(f"Overhead{code}", oh)
    header = ["Config.", "Header (B)", "Tag (B)", "Total (B)", "of 1 MiB (\\%)"]
    if median_payload:
        header.append("of median image (\\%)")
    tex_table(out / "tab_overhead.tex", header, rows, "lrrrr" + ("r" if median_payload else ""),
              "exact serialized envelope overhead; recipient-key provisioning is not per-message overhead")


def report_environment(run: Path, out: Path, macros: Macros) -> dict[str, Any]:
    env_path = run / "environment.json"
    if not env_path.exists():
        return {}
    env = json.loads(env_path.read_text(encoding="utf-8"))
    b = env["backends"]
    pk = b["packages"]
    kem = b["kem_backends"].get("openssl", {})
    cpu = " ".join(re.sub(r"\((?:R|TM)\)|^\d+th Gen ", "", env["cpu"]).split())  # drop trademark noise
    power = env.get("power", {})
    plan = re.search(r"\(([^()]+)\)\s*$", power.get("power_scheme") or "")
    ac = {True: "AC power", False: "battery"}.get(power.get("on_ac_power"), "power source unknown")
    rows = [["CPU", tex_escape(f"{cpu} ({env['cpu_physical_cores']} cores, {env['cpu_logical_cores']} threads)")],
            ["Memory", f"{env['ram_bytes'] / 2**30:.1f} GiB"],
            ["OS", tex_escape(env["platform"])],
            ["Python", tex_escape(b["python"].split()[0])],
            ["ML-KEM (primary)", tex_escape(f"cryptography {pk.get('cryptography')} / {kem.get('implementation', '')}")],
            ["ML-KEM (secondary)", tex_escape(f"pqcrypto {pk.get('pqcrypto')} (Rust backbone-ml-kem)")],
            ["ML-KEM (reference)", tex_escape(f"kyber-py {pk.get('kyber-py')} (pure Python)")],
            ["AEAD, HKDF, scrypt", tex_escape(f"cryptography {pk.get('cryptography')} (OpenSSL)")],
            ["NumPy / SciPy", tex_escape(f"{pk.get('numpy')} / {pk.get('scipy')}")],
            ["Power", tex_escape(f"{ac}; {plan.group(1)} plan" if plan else ac)]]
    tex_table(out / "tab_environment.tex", ["Item", "Value"], rows, ">{\\raggedright\\arraybackslash}l"
              ">{\\raggedright\\arraybackslash}p{0.6\\linewidth}")
    macros.add("CPU", tex_escape(env["cpu"]))
    macros.add("RAMGiB", f"{env['ram_bytes'] / 2**30:.0f}")
    return {"cpu": env["cpu"]}


def report_load(raw: Path, macros: Macros) -> dict[str, Any]:
    p = raw / "load_timing.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    res = {"samples": len(df), "cpu_total_median": float(df.cpu_percent_total.median()),
           "cpu_total_p95": float(df.cpu_percent_total.quantile(0.95))}
    macros.add("LoadCpuMedian", fmt(res["cpu_total_median"], 1))
    macros.add("LoadCpuPNinetyFive", fmt(res["cpu_total_p95"], 1))
    return res


def build_report(run: Path, paper_dir: Path | None = None, bootstrap: int = 10000) -> Path:
    run = run.resolve()
    raw = run / "raw"
    out = run / "report"
    out.mkdir(exist_ok=True)
    cfg = json.loads((run / "config.effective.json").read_text(encoding="utf-8"))
    seed = int(cfg["experiment"]["seed"])
    name = cfg["experiment"]["name"]
    prefix = "cc" + macro_name(name.replace("_", " "))
    macros = Macros(prefix)
    summary: dict[str, Any] = {"run": run.name, "experiment": name}
    summary["environment"] = report_environment(run, out, macros)
    median_payload = None
    if (raw / "timing.csv").exists():
        t = pd.read_csv(raw / "timing.csv", usecols=["image_id", "payload_bytes"]).drop_duplicates("image_id")
        median_payload = int(np.median(t.payload_bytes))
        macros.add("MedianPayloadMiB", fmt(median_payload / MIB, 2))
    report_overhead(out, macros, median_payload)
    sections = [("timing", lambda: report_timing(raw, out, macros, bootstrap, seed)),
                ("scaling", lambda: report_scaling(raw, out, macros, bootstrap, seed)),
                ("kdf_sensitivity", lambda: report_kdf(raw, out, macros, bootstrap, seed)),
                ("stages", lambda: report_stages(raw, out, macros)),
                ("diagnostics", lambda: report_diagnostics(raw, out, macros, bootstrap, seed)),
                ("auth", lambda: report_auth(raw, out, macros)),
                ("key_sensitivity", lambda: report_keysens(raw, out, macros)),
                ("memory", lambda: report_memory(raw, out, macros)),
                ("calibration", lambda: report_calibration(raw, out, macros)),
                ("map_analysis", lambda: report_maps(raw, out, macros))]
    for key, fn in sections:
        src = raw / ("map_analysis.json" if key == "map_analysis" else f"{key}.csv")
        if src.exists():
            summary[key] = fn()
    summary["load"] = report_load(raw, macros)
    macros.write(out / "macros.tex")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    from .runner import write_hashes
    write_hashes(out)  # report/SHA256SUMS: the derived outputs that feed the manuscript
    if paper_dir is not None:
        dest = paper_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        for p in out.iterdir():
            if p.suffix in (".tex", ".pdf"):
                shutil.copyfile(p, dest / p.name)
    return out
