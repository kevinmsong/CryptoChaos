"""Regression test for the partitioning-oracle attack on non-committing AEADs.

AES-GCM is not key-committing: given k keys, one can solve a linear system over GF(2^128) for a
ciphertext of k-1 blocks and a tag that verifies under all k keys. An attacker who knows only the
recipient's public key can derive k candidate keys (one per guessed password), so without key
commitment a single envelope would open under k passwords and each receiver response would test
k guesses. CryptoChaos binds a 32-byte HKDF-derived commitment into the authenticated header and
checks it before decryption; this test builds the multicollision and shows it opens under at most
the one password whose commitment the attacker chose.
"""
import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import cryptochaos as cc
from cryptochaos import kdf
from cryptochaos.errors import DecryptionError
from cryptochaos.kem import get_kem_backend
from cryptochaos.maps import v2_chaos_value
from cryptochaos.protocol import SUITES, TAG_BYTES, Header, Receiver, build_aad
from conftest import FAST_POLICY, FAST_SCRYPT

R = 0xE1 << 120


def gmul(x: int, y: int) -> int:
    z, v = 0, y
    for i in range(128):
        if (x >> (127 - i)) & 1:
            z ^= v
        v = (v >> 1) ^ R if v & 1 else v >> 1
    return z


def gpow(x: int, e: int) -> int:
    r = 1 << 127
    while e:
        if e & 1:
            r = gmul(r, x)
        x = gmul(x, x)
        e >>= 1
    return r


def ginv(x: int) -> int:
    return gpow(x, (1 << 128) - 2)


def blocks(b: bytes) -> list[int]:
    b = b + bytes(-len(b) % 16)
    return [int.from_bytes(b[i:i + 16], "big") for i in range(0, len(b), 16)]


def aes_block(key: bytes, block: bytes) -> bytes:
    e = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return e.update(block) + e.finalize()


def ghash(h: int, xs: list[int]) -> int:
    y = 0
    for x in xs:
        y = gmul(y ^ x, h)
    return y


def gcm_tag(key: bytes, nonce: bytes, aad: bytes, ct: bytes) -> bytes:
    h = int.from_bytes(aes_block(key, bytes(16)), "big")
    ej0 = int.from_bytes(aes_block(key, nonce + b"\x00\x00\x00\x01"), "big")
    lb = (len(aad) * 8 << 64) | (len(ct) * 8)
    return (ghash(h, blocks(aad) + blocks(ct) + [lb]) ^ ej0).to_bytes(16, "big")


def solve(mat: list[list[int]], rhs: list[int]) -> list[int]:
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for c in range(n):
        p = next(r for r in range(c, n) if a[r][c])
        a[c], a[p] = a[p], a[c]
        inv = ginv(a[c][c])
        a[c] = [gmul(v, inv) for v in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [v ^ gmul(f, w) for v, w in zip(a[r], a[c])]
    return [a[i][n] for i in range(n)]


def multicollide(keys: list[bytes], nonce: bytes, aad: bytes) -> bytes:
    """Ciphertext of len(keys)-1 blocks plus a tag that verifies under every key."""
    k = len(keys)
    lb = (len(aad) * 8 << 64) | (16 * (k - 1) * 8)
    mat, rhs = [], []
    for key in keys:
        h = int.from_bytes(aes_block(key, bytes(16)), "big")
        ej0 = int.from_bytes(aes_block(key, nonce + b"\x00\x00\x00\x01"), "big")
        mat.append([1 << 127] + [gpow(h, k - j + 1) for j in range(1, k)])
        rhs.append(ej0 ^ gmul(ghash(h, blocks(aad)), gpow(h, k)) ^ gmul(lb, h))
    sol = solve(mat, rhs)
    return b"".join(c.to_bytes(16, "big") for c in sol[1:]) + sol[0].to_bytes(16, "big")


def test_ghash_model_matches_aesgcm():
    for n in (0, 1, 16, 33):
        k, nonce, aad, pt = os.urandom(32), os.urandom(12), os.urandom(n + 5), os.urandom(n)
        out = AESGCM(k).encrypt(nonce, pt, aad)
        assert gcm_tag(k, nonce, aad, out[:-16]) == out[-16:]


def test_multicollision_works_at_the_aead_layer():
    keys = [os.urandom(32) for _ in range(4)]
    nonce, aad = os.urandom(12), b"header"
    ct = multicollide(keys, nonce, aad)
    for key in keys:  # without key commitment every key accepts the same ciphertext
        AESGCM(key).decrypt(nonce, ct, aad)


def test_envelope_opens_under_at_most_one_candidate_password():
    k = 4
    candidates = [f"guess-{i}" for i in range(k)]
    kp = cc.generate_recipient_keypair("openssl")
    backend = get_kem_backend("openssl")
    for code in "EF":
        suite = SUITES[code]
        enc = backend.encapsulate(backend.load_public_key(kp.public_key))  # attacker: public key only
        hkdf_salt, pw_salt, nonce = os.urandom(32), os.urandom(16), os.urandom(12)
        keys, commits = [], []
        for pw in candidates:
            pw_key = kdf.derive_password_key(kdf.encode_password(pw), pw_salt, FAST_SCRYPT)
            fields = [(kdf.FIELD_KEM, kdf.kem_input(enc.shared_secret)), (kdf.FIELD_PW_KEY, pw_key)]
            if suite.password == "scrypt+chaos":
                fields.append((kdf.FIELD_CHAOS_V2, v2_chaos_value(pw_key)))
            key, commit = kdf.derive_key_and_commitment(cc.VERSION, suite.suite_id, fields, hkdf_salt)
            keys.append(key)
            commits.append(commit)
        header = Header(suite, nonce, 16 * (k - 1) + TAG_BYTES, enc.ciphertext, hkdf_salt, FAST_SCRYPT, pw_salt,
                        commits[0]).encode()
        env = header + multicollide(keys, nonce, build_aad(header, b"app"))
        accepted = []
        for pw in candidates:
            r = Receiver(recipient_secret_key=kp.secret_key, accepted_suites={code}, scrypt_policy=FAST_POLICY)
            try:
                r.open(env, password=pw, aad=b"app")
                accepted.append(pw)
            except DecryptionError:
                pass
        assert accepted == [candidates[0]], accepted  # one guess per envelope, not k
