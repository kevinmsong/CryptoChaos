import json
import os
from pathlib import Path

import pytest

from cryptochaos import generate_recipient_keypair
from cryptochaos.kdf import ScryptParams, ScryptPolicy

ROOT = Path(__file__).resolve().parent.parent
VECTORS = Path(__file__).resolve().parent / "vectors"

# Low-cost scrypt for structural tests only; the protocol default policy rejects it.
FAST_SCRYPT = ScryptParams(10, 8, 1)
FAST_POLICY = ScryptPolicy(min_log2_n=10, max_log2_n=18)

# Passphrase pairs found by birthday search (map_analysis). Different 5-byte seeds, same configuration D key:
LEGACY_KEY_COLLISION = ("cryptochaos-demo-12996", "cryptochaos-demo-49085")
# Identical first five bytes of BLAKE3(passphrase):
LEGACY_SEED_COLLISION = ("cryptochaos-demo-1791738", "cryptochaos-demo-2065867")


@pytest.fixture(scope="session")
def keypair():
    return generate_recipient_keypair("openssl")


@pytest.fixture(scope="session")
def other_keypair():
    return generate_recipient_keypair("openssl")


@pytest.fixture(scope="session")
def psk():
    return os.urandom(32)




def load_acvp(kind: str) -> dict:
    return json.loads((VECTORS / "acvp" / f"ML-KEM-1024-{kind}.json").read_text(encoding="utf-8"))


def acvp_group(kind: str, function: str | None = None) -> list[dict]:
    groups = load_acvp(kind)["testGroups"]
    if function is not None:
        groups = [g for g in groups if g.get("function") == function]
    assert groups, f"no ACVP groups for {kind}/{function}"
    return [t for g in groups for t in g["tests"]]
