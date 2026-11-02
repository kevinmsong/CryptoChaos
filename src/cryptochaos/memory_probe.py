"""Peak-memory probe executed in an isolated subprocess (one suite, one payload, N operations).

Windows: ``psutil`` ``peak_wset`` (peak working set, includes native allocations). Linux:
``getrusage(RUSAGE_SELF).ru_maxrss``. Both are process-lifetime high-water marks that cannot be
reset, so the probe imports only the protocol modules, records the baseline (current RSS and the
high-water mark after imports, key setup, and payload allocation), and reports whether the
operations raised the high-water mark. Only then is ``peak - baseline_rss`` attributable to the
operations; otherwise it is an upper bound set by earlier allocations. With ``--tracemalloc`` the
probe also reports the Python-allocation peak during the operations; that figure excludes native
(OpenSSL) allocations and is not total memory.
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import sys


def _peak_and_rss() -> tuple[int, int, str]:
    import psutil

    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):
        return int(mi.peak_wset), int(mi.rss), "windows_peak_working_set"
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, int(mi.rss), "linux_ru_maxrss"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--payload-bytes", type=int, required=True)
    ap.add_argument("--operations", type=int, default=3)
    ap.add_argument("--scrypt-log2n", type=int, default=14)
    ap.add_argument("--kem-backend", default="openssl")
    ap.add_argument("--seed", type=int, default=0, help="unused; kept for command-line compatibility")
    ap.add_argument("--image", default=None, help="raw RGB payload file (pre-decoded by the parent)")
    ap.add_argument("--tracemalloc", action="store_true")
    a = ap.parse_args(argv)

    from .kdf import ScryptParams
    from .kem import get_kem_backend
    from .protocol import SUITES, Receiver, Sender

    if a.image:
        with open(a.image, "rb") as fh:
            payload = fh.read()
    else:
        payload = os.urandom(a.payload_bytes)
    backend = get_kem_backend(a.kem_backend)
    kp = backend.generate_keypair()
    psk = os.urandom(32)
    password = base64.b64encode(os.urandom(15)).decode() if SUITES[a.suite].password else None
    aad = b"CryptoChaos-v2 benchmark"
    sender = Sender(a.suite, recipient_public_key=kp.public_key, kem_backend=backend, psk=psk,
                    scrypt=ScryptParams(a.scrypt_log2n))
    receiver = Receiver(recipient_secret_key=kp.secret_key, kem_backend=backend, psk=psk, accepted_suites={a.suite})
    gc.collect()
    base_peak, base_rss, metric = _peak_and_rss()
    tm_peak = None
    if a.tracemalloc:
        import tracemalloc

        tracemalloc.start()
    ok = True
    for _ in range(a.operations):
        env = sender.seal(payload, password=password, aad=aad)
        out = receiver.open(env, password=password, aad=aad)
        ok &= out == payload
        del env, out
    if a.tracemalloc:
        tm_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    peak, _, _ = _peak_and_rss()
    raised = peak > base_peak
    print(json.dumps({"baseline_rss_bytes": base_rss, "baseline_peak_bytes": base_peak, "peak_bytes": peak,
                      "peak_minus_baseline_rss_bytes": peak - base_rss, "ops_raised_peak": raised,
                      "tracemalloc_peak_bytes": tm_peak, "ok": bool(ok), "platform_metric": metric,
                      "payload_bytes": len(payload)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
