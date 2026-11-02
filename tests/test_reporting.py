"""Report-side checks: the SP 800-22 section 4.2.2 minimum and checksum verification of output directories."""
import math

import numpy as np

from cryptochaos import artifact, reporting
from cryptochaos.runner import write_hashes


def test_uniformity_check_needs_55_sequences():
    assert math.isnan(reporting.uniformity_or_nan(np.full(54, 0.5)))
    assert math.isfinite(reporting.uniformity_or_nan(np.linspace(0.005, 0.995, 55)))


def test_hash_only_verification_detects_a_changed_file(tmp_path):
    (tmp_path / "metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    write_hashes(tmp_path)
    assert artifact.verify_run(tmp_path)["ok"]
    (tmp_path / "metrics.csv").write_text("a,b\n1,3\n", encoding="utf-8")
    report = artifact.verify_run(tmp_path)
    assert not report["ok"] and report["hash_mismatches"] == ["metrics.csv"]
