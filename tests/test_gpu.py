"""GPU diagnostic kernels: exact integer parity with the CPU reference (skipped without CUDA/Triton)."""
import numpy as np
import pytest

from cryptochaos import gpu_metrics as G
from cryptochaos.errors import BackendUnavailableError

pytestmark = pytest.mark.gpu


def _require_gpu():
    try:
        return G._triton_kernels()
    except BackendUnavailableError as exc:
        pytest.skip(f"GPU backend unavailable: {exc}")


def test_cpu_reference_matches_metrics_module():
    from cryptochaos import metrics as M

    rng = np.random.Generator(np.random.PCG64(4))
    a, b = rng.integers(0, 256, 10_001, dtype=np.uint8), rng.integers(0, 256, 10_001, dtype=np.uint8)
    unequal, absdiff, pc = G.cpu_diffs(a, b)
    assert 100 * unequal / a.size == pytest.approx(M.unequal_byte_pct(a, b))
    assert 100 * absdiff / (255 * a.size) == pytest.approx(M.mean_abs_diff_pct(a, b))
    assert 100 * pc / (8 * a.size) == pytest.approx(M.bit_diff_pct(a, b))
    assert np.array_equal(G.cpu_histogram(a), M.byte_histogram(a))


def test_triton_parity_exact_on_edge_cases():
    _require_gpu()
    results = G.check_parity()
    bad = [r for r in results if not (r["hist_equal"] and r["diffs_equal"])]
    assert not bad, bad


def test_torch_reference_parity():
    k = _require_gpu()
    torch = k["torch"]
    rng = np.random.Generator(np.random.PCG64(8))
    a, b = rng.integers(0, 256, 300_007, dtype=np.uint8), rng.integers(0, 256, 300_007, dtype=np.uint8)
    h, d = G.torch_reference(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda())
    assert np.array_equal(h.cpu().numpy(), G.cpu_histogram(a))
    assert tuple(int(x) for x in d.cpu().numpy()) == G.cpu_diffs(a, b)


def test_large_input_counter_widths():
    k = _require_gpu()
    torch = k["torch"]
    n = (1 << 31) + 12345  # beyond 32-bit element counts; int64 accumulators required
    try:
        a = torch.full((n,), 255, dtype=torch.uint8, device="cuda")
        b = torch.zeros((n,), dtype=torch.uint8, device="cuda")
    except RuntimeError as exc:  # insufficient device memory
        pytest.skip(f"cannot allocate 2 x {n} bytes on the device: {exc}")
    d = G.gpu_diffs(a, b).cpu().numpy()
    assert int(d[0]) == n and int(d[1]) == 255 * n and int(d[2]) == 8 * n
    h = G.gpu_histogram(a).cpu().numpy()
    assert int(h[255]) == n and int(h.sum()) == n
