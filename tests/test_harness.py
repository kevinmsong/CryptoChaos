"""End-to-end fixture (synthetic images -> raw results -> report -> verify), resume semantics, statistics."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptochaos import benchmark, map_analysis, reporting
from cryptochaos.artifact import verify_run
from cryptochaos.dataset import make_synthetic_dataset
from cryptochaos.errors import ConfigurationError
from cryptochaos.runner import CsvSink

E2E_PASSES = ["environment", "auth", "key_sensitivity", "stages", "timing", "diagnostics", "scaling",
              "kdf_sensitivity", "calibration"]


def write_config(tmp: Path, name: str, data_dir: Path, passes, seed: int = 11) -> Path:
    cfg = f"""
[experiment]
name = "{name}"
seed = {seed}
output_root = "{(tmp / 'runs').as_posix()}"
passes = {json.dumps(passes)}
prevent_sleep = false

[dataset]
manifest = ""
path = "{data_dir.as_posix()}"
selection = "all"

[timing]
repeats = 2
warmup_messages = 1
warmup_payload_bytes = 4096

[diagnostics]
workers = 1
sample_bits = 100000

[auth]
trials_per_class = 1

[key_sensitivity]
images = 1
synthetic_bytes = 4096

[stages]
repeats = 2
reference_repeats = 1
scrypt_log2_n = [14]
scrypt_repeats = 1
payload_bytes = 65536

[scaling]
sizes = [1024, 16384]
repeats = [2, 2]
cold_repeats_max = 1

[kdf_sensitivity]
log2_n = [14]
repeats = 1
payload_bytes = 4096

[calibration]
uniform_streams = 3
positive_control_streams = 1
negative_control_streams = 1
workers = 1
n_bits = 100000
"""
    path = tmp / f"{name}.toml"
    path.write_text(cfg, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    data = tmp / "images"
    make_synthetic_dataset(data, count=3)
    cfg = write_config(tmp, "e2e", data, E2E_PASSES)
    run_dir = benchmark.run(cfg)
    return tmp, data, cfg, run_dir


def test_e2e_rows_complete_and_correct(e2e_run):
    _, _, _, run = e2e_run
    t = pd.read_csv(run / "raw" / "timing.csv")
    assert len(t) == 3 * 2 * 6 and t.ok.all() and t.error.isna().all()
    assert set(t.suite) == set("ABCDEF")
    assert t[t.suite.isin(list("CDEF"))].cold_ok.all() and t[t.suite.isin(["A", "B"])].cold_seal_ns.isna().all()
    for _, r in t.iterrows():
        assert r.overhead_bytes == {"A": 45, "B": 45, "C": 1681, "D": 1681, "E": 1702, "F": 1702}[r.suite]
        assert r.envelope_bytes == r.payload_bytes + r.overhead_bytes
    # every image/repeat block contains each suite exactly once
    assert (t.groupby(["image_id", "repeat"]).suite.nunique() == 6).all()
    d = pd.read_csv(run / "raw" / "diagnostics.csv")
    assert len(d) == 3 * 7 and d[d.suite != "P"].ok.all()
    # synthetic images are shorter than the 100,000-bit sample only if tiny; record status explicitly
    assert d.sts_frequency_status.isin(["ok", "not_applicable"]).all()
    a = pd.read_csv(run / "raw" / "auth.csv")
    unexpected = a[(a.case != "valid") & (a.outcome == "accepted")
                   & ~((a.suite == "D") & (a.case == "legacy_collision_password"))]
    assert unexpected.empty
    assert (a[a.case == "valid"].outcome == "accepted").all()
    ks = pd.read_csv(run / "raw" / "key_sensitivity.csv")
    assert (ks[ks.variation == "plaintext_one_bit_fixed_key_nonce"].differing_bits == 1).all()
    assert (ks[(ks.suite == "D") & (ks.variation == "legacy_collision_password")].differing_bits == 0).all()


def test_e2e_report_units_and_assets(e2e_run):
    _, _, _, run = e2e_run
    out = reporting.build_report(run, bootstrap=60)
    for name in ("tab_timing.tex", "tab_paired.tex", "tab_auth.tex", "tab_overhead.tex", "tab_sts.tex",
                 "tab_calibration.tex", "fig_scaling.pdf", "fig_paired_ratios.pdf", "fig_stages.pdf",
                 "fig_calibration.pdf", "macros.tex", "summary.json"):
        assert (out / name).exists(), name
    t = pd.read_csv(run / "raw" / "timing.csv")
    s = pd.read_csv(out / "timing_summary.csv").set_index("suite")
    for code in "ABCDEF":
        sub = t[t.suite == code]
        expected = sub.payload_bytes.sum() / 2**20 / (sub.seal_ns.sum() / 1e9)  # MiB/s from raw bytes
        assert s.loc[code, "seal_mib_s_aggregate"] == pytest.approx(expected)
    tab = (out / "tab_timing.tex").read_text(encoding="utf-8")
    assert all(f"\n{c} &" in tab for c in "ABCDEF")
    assert "\\newcommand{\\ccETwoeTimingImages}{3}" in (out / "macros.tex").read_text(encoding="utf-8")


def test_e2e_verify_artifact(e2e_run):
    _, _, _, run = e2e_run
    rep = verify_run(run)
    assert rep["ok"], rep["problems"]
    assert rep["passes"]["timing"]["missing_rows"] == 0


def test_resume_rules(tmp_path):
    data = tmp_path / "images"
    make_synthetic_dataset(data, count=2)
    cfg = write_config(tmp_path, "resume", data, ["auth"])
    run = benchmark.run(cfg)
    rows = list(csv.DictReader((run / "raw" / "auth.csv").open(encoding="utf-8")))
    with pytest.raises(ConfigurationError):
        benchmark.run(cfg)  # exists and --resume not given
    # simulate an interrupted pass: drop the completion marker and the last three rows
    (run / "status" / "auth.json").unlink()
    with (run / "raw" / "auth.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows[:-3])
    benchmark.run(cfg, resume=True)
    again = list(csv.DictReader((run / "raw" / "auth.csv").open(encoding="utf-8")))
    assert len(again) == len(rows) and len({r["key"] for r in again}) == len(rows)
    changed = write_config(tmp_path, "resume", data, ["auth"], seed=12)
    with pytest.raises(ConfigurationError, match="identity mismatch"):
        benchmark.run(changed, resume=True)


def test_csv_sink_rejects_duplicates(tmp_path):
    sink = CsvSink(tmp_path / "x.csv", ["key", "v"])
    sink.write({"key": "a", "v": 1})
    with pytest.raises(ConfigurationError):
        sink.write({"key": "a", "v": 2})
    sink.close()
    with pytest.raises(ConfigurationError):
        CsvSink(tmp_path / "x.csv", ["key", "w"])  # schema change refused on resume


def test_csv_sink_repairs_torn_and_nul_tails(tmp_path):
    p = tmp_path / "t.csv"
    with CsvSink(p, ["key", "v", "ok"]) as s:
        s.write({"key": "a", "v": 1, "ok": True})
    with p.open("a", encoding="utf-8", newline="") as fh:
        fh.write("b,2")  # crash mid-row: no newline, missing column
    s = CsvSink(p, ["key", "v", "ok"])
    assert s.keys == {"a"} and s.repaired_bytes == 3  # the torn row is not treated as complete
    s.write({"key": "b", "v": 2, "ok": True})
    s.close()
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert [r["key"] for r in rows] == ["a", "b"] and all(None not in r for r in rows)
    p.write_bytes(p.read_bytes() + b"\x00" * 64)
    s = CsvSink(p, ["key", "v", "ok"])
    assert s.keys == {"a", "b"} and s.repaired_bytes == 64
    s.close()


def test_strict_bool_parsing_and_macro_names():
    s = pd.Series(["True", "False", "", "nan", "2", " true "])
    assert reporting.is_true(s).tolist() == [True, False, False, False, False, True]
    assert reporting.macro_name("scrypt log2N=14") == "ScryptLogTwoNOneFour"
    assert reporting.macro_name("scrypt log2N=16") != reporting.macro_name("scrypt log2N=14")
    m = reporting.Macros("cc")
    m.add("X1", 1)
    with pytest.raises(ValueError):
        m.add("X1", 2)


def test_memory_probe_subprocess():
    out = subprocess.run([sys.executable, "-m", "cryptochaos.memory_probe", "--suite", "E", "--payload-bytes", "65536",
                          "--operations", "1"], capture_output=True, text=True, check=True, timeout=300)
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    assert rec["ok"] and rec["peak_bytes"] >= rec["baseline_rss_bytes"] > 0
    assert rec["peak_bytes"] - rec["baseline_rss_bytes"] >= 15 * 2**20  # scrypt N=2^14, r=8 needs 16 MiB


def test_functional_graph_known_structure():
    # 0->1->2->0 (3-cycle), 3->0 (tail 1), 4->4 (fixed point), 5->3 (tail 2)
    g = map_analysis.functional_graph([1, 2, 0, 0, 4, 3])
    assert g["cycles"] == 2 and g["cycle_lengths"] == {1: 1, 3: 1}
    assert g["max_transient"] == 2 and g["states_on_cycles"] == 4
    assert map_analysis.v2_reduced(8)["logistic"]["graph"]["states"] == 256


def test_hierarchical_bootstrap_recovers_known_difference():
    rng = np.random.Generator(np.random.PCG64(1))
    base = rng.uniform(5, 15, size=(200, 1))
    a = base + 0.5 + rng.normal(0, 0.05, size=(200, 5))
    b = base + rng.normal(0, 0.05, size=(200, 5))
    r = reporting.hierarchical_bootstrap_paired(a, b, 400, 3)
    assert r["median_diff_lo"] <= 0.5 <= r["median_diff_hi"]
    assert r["gm_ratio_lo"] > 1.0 and r["frac_a_slower"] == 1.0


def test_clopper_pearson():
    from scipy.stats import beta

    k, n = 10, 1000
    lo, hi = reporting.clopper_pearson(k, n)
    assert lo == pytest.approx(beta.ppf(0.025, k, n - k + 1), rel=1e-9)
    assert hi == pytest.approx(beta.ppf(0.975, k + 1, n - k), rel=1e-9)
    assert reporting.clopper_pearson(0, 50)[0] == 0.0
