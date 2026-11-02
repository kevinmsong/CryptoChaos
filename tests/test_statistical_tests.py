"""SP 800-22 subset: NIST STS 2.1.2 agreement, worked examples, and applicability states."""
import json
import math

import numpy as np
import pytest

from cryptochaos import statistical_tests as T
from cryptochaos.calibration import StreamSpec, generate
from conftest import VECTORS

STS = json.loads((VECTORS / "sts" / "sts_reference.json").read_text(encoding="utf-8"))
TOL = 2e-6  # STS prints six decimals


def _case_bits(case):
    if case["kind"] == "sts_constant":
        packed = VECTORS / "sts" / (case["file"].replace(".", "_") + ".bin")
        return np.unpackbits(np.frombuffer(packed.read_bytes(), dtype=np.uint8))
    return generate(StreamSpec(case["kind"], case["index"], case["expected"], case["source"]), case["n_bits"])


@pytest.mark.parametrize("case", STS["cases"], ids=[f"{c['kind']}-{c.get('index', c.get('file'))}" for c in STS["cases"]])
def test_agreement_with_nist_sts(case):
    bits = _case_bits(case)
    assert bits.size == 1_000_000
    import hashlib
    assert hashlib.sha256(np.packbits(bits).tobytes()).hexdigest() == case["bits_sha256"]
    ours = {r.test: r for r in T.run_battery(bits)}
    for name, ref in case["sts_p_values"].items():
        r = ours[name]
        if name == "runs" and ref == [0.0] and r.status == "prerequisite_failed":
            continue  # STS reports 0.0 when the frequency prerequisite fails
        assert r.status == "ok", (name, r.reason)
        assert len(r.p_values) == len(ref)
        for mine, theirs in zip(r.p_values, ref):
            assert abs(mine - theirs) <= TOL, (name, mine, theirs)


@pytest.mark.parametrize("case", STS["cases"], ids=[f"{c['kind']}-{c.get('index', c.get('file'))}" for c in STS["cases"]])
def test_every_aperiodic_template_agrees_with_nist_sts(case):
    templates = T.aperiodic_templates(9)  # increasing binary order, as in STS templates/template9
    assert len(case["sts_template9_all"]) == len(templates) == 148
    bits = _case_bits(case)
    for template, theirs in zip(templates, case["sts_template9_all"]):
        r = T.non_overlapping_template(bits, template)
        assert r.status == "ok", (template, r.reason)
        assert abs(r.p_values[0] - theirs) <= TOL, (template, r.p_values[0], theirs)


EPS_100 = ("11001001000011111101101010100010001000010110100011"
           "00001000110100110001001100011001100010100010111000")
EPS_128 = ("11001100000101010110110001001100111000000000001001"
           "00110101010001000100111101011010000000110101111100"
           "1100111001101101100010110010")


def test_sp800_22_worked_examples():
    b = T.bits_from_string
    assert round(T.frequency(b("1011010101"), strict=False).p_values[0], 6) == 0.527089
    assert round(T.frequency(b(EPS_100)).p_values[0], 6) == 0.109599
    assert round(T.block_frequency(b("0110011010"), 3, strict=False).p_values[0], 6) == 0.801252
    assert round(T.block_frequency(b(EPS_100), 10, strict=False).p_values[0], 6) == 0.706438
    assert round(T.runs(b("1001101011"), strict=False).p_values[0], 6) == 0.147232
    assert round(T.runs(b(EPS_100)).p_values[0], 6) == 0.500798
    t = T.non_overlapping_template(b("10100100101110010110"), "001", N=2, strict=False)
    assert round(t.p_values[0], 6) == 0.344154
    s = T.serial(b("0011011101"), 3, strict=False)
    assert (round(s.p_values[0], 6), round(s.p_values[1], 6)) == (0.808792, 0.670320)


def test_longest_run_worked_example():
    bits = T.bits_from_string(EPS_128)
    assert bits.size == 128
    r = T.longest_run(bits)
    assert r.statistic["nu"] == [4, 9, 3, 0]  # N = 16 blocks of M = 8
    # STS 2.1.2 exact probabilities reproduce the published chi2 = 4.882457, P = 0.180609
    assert abs(r.statistic["chi2"] - 4.882457) < 5e-7
    assert round(r.p_values[0], 6) == 0.180609
    # the 4-digit rounded table printed in SP 800-22 3.4 gives 0.18059798
    doc = T.longest_run(bits, probabilities=(0.2148, 0.3672, 0.2305, 0.1875))
    assert abs(doc.p_values[0] - 0.18059798) < 1e-8


def test_longest_run_vectorization_matches_naive():
    rng = np.random.Generator(np.random.PCG64(5))
    blocks = (rng.random((40, 128)) < 0.6).astype(np.uint8)
    blocks[0] = 0
    blocks[1] = 1
    naive = []
    for row in blocks:
        best = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        naive.append(best)
    assert T.longest_run_per_block(blocks).tolist() == naive


def test_applicability_states_are_explicit():
    short = T.bits_from_string("10" * 20)
    assert T.frequency(short).status == "not_applicable"
    assert T.longest_run(short).status == "not_applicable"
    assert T.spectral(short).status == "not_applicable"
    bits = generate(StreamSpec("chacha20", 0, "uniform", ""), 1_000_000)
    assert T.serial(bits, 17).status == "not_applicable"
    assert T.block_frequency(bits, 128).status == "not_applicable"  # N = 7812 violates N < 100
    assert T.non_overlapping_template(bits, "101010101").status == "not_applicable"  # periodic template
    biased = np.ones(1000, dtype=np.uint8)
    r = T.runs(biased)
    assert r.status == "prerequisite_failed" and r.p_values == () and r.passed() is None
    row = T.battery_record([r])
    assert row["sts_runs_status"] == "prerequisite_failed" and row["sts_runs_p_pass"] is None
    assert math.isnan(row["sts_runs_p"])


def test_serial_reports_both_p_values_separately():
    bits = generate(StreamSpec("chacha20", 1, "uniform", ""), 1_000_000)
    s = T.serial(bits, 16)
    assert len(s.p_values) == 2
    row = T.battery_record([s])
    assert {"sts_serial_p1", "sts_serial_p2", "sts_serial_p1_pass", "sts_serial_p2_pass"} <= row.keys()


def test_aperiodic_template_counts_match_sts_files():
    assert len(T.aperiodic_templates(9)) == 148
    assert len(T.aperiodic_templates(10)) == 284
    assert T.TEMPLATE in T.aperiodic_templates(9)


def test_igamc_cephes_semantics():
    assert T.igamc(2.0, 0.0) == 1.0 and T.igamc(2.0, -3.0) == 1.0
    assert abs(T.igamc(1.0, 1.0) - math.exp(-1)) < 1e-15


def test_second_level_helpers():
    lo, hi = T.proportion_interval(1000)
    assert abs(lo - 0.980561) < 1e-5 and abs(hi - 0.999439) < 1e-5
    assert T.uniformity_p_value(np.linspace(0.0005, 0.9995, 1000)) > 0.99
    assert T.uniformity_p_value(np.full(1000, 0.05)) < 1e-6


def test_sample_policy():
    assert T.sample_bits(b"\x00" * 124_999) is None
    s = T.sample_bits(b"\xf0" * 200_000)
    assert s.size == 1_000_000 and s[:8].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
