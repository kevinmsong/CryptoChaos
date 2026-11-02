"""Finite-state analysis of the legacy maps (exhaustive) and the v2 maps (reduced precision + sampling).

These bounds concern the map component only. They are not an attack cost for the complete
KEM/AEAD system: in v1 they bound the diversity of the passphrase-derived value; in v2 the full
scrypt key enters the key schedule regardless of map behaviour.
"""
from __future__ import annotations

import collections
import json
import math
from typing import Any, Callable

import blake3
import numpy as np

from . import maps as M


def functional_graph(nxt: list[int]) -> dict[str, Any]:
    """Cycle and transient structure of x -> nxt[x] on states 0..N-1 (iterative, O(N))."""
    n = len(nxt)
    state = [0] * n  # 0 unvisited, 1 on current path, 2 finished
    tail = [0] * n
    cyc = [0] * n
    cycles: list[int] = []
    for start in range(n):
        if state[start]:
            continue
        path = []
        x = start
        while state[x] == 0:
            state[x] = 1
            path.append(x)
            x = nxt[x]
        if state[x] == 1:  # new cycle found at x
            i = path.index(x)
            length = len(path) - i
            cycles.append(length)
            for y in path[i:]:
                cyc[y], tail[y], state[y] = length, 0, 2
            path = path[:i]
        for y in reversed(path):
            z = nxt[y]
            cyc[y], tail[y], state[y] = cyc[z], tail[z] + 1, 2
    on_cycle = sum(1 for t in tail if t == 0)
    lengths = collections.Counter(cycles)
    return {"states": n, "cycles": len(cycles), "cycle_lengths": dict(sorted(lengths.items())),
            "longest_cycle": max(cycles), "states_on_cycles": on_cycle, "max_transient": max(tail),
            "mean_transient": float(np.mean(tail)), "mean_rho_length": float(np.mean(np.add(tail, cyc)))}


def output_distribution(outputs: list[bytes]) -> dict[str, Any]:
    c = collections.Counter(outputs)
    n = len(outputs)
    p = np.array(list(c.values()), dtype=np.float64) / n
    return {"seeds": n, "distinct_outputs": len(c), "shannon_bits": float(-(p * np.log2(p)).sum()),
            "min_entropy_bits": float(-math.log2(p.max())), "collision_probability": float((p * p).sum())}


def legacy_analysis() -> dict[str, Any]:
    out: dict[str, Any] = {}
    one_d = {"logistic": M.legacy_logistic_step, "chebyshev": M.legacy_chebyshev_step, "tent": M.legacy_tent_step}
    emit = {"logistic": M.legacy_logistic, "chebyshev": M.legacy_chebyshev, "tent": M.legacy_tent}
    for name, step in one_d.items():
        g = functional_graph([step(x) for x in range(256)])
        d = output_distribution([emit[name](32, s) for s in range(256)])
        out[name] = {**d, "graph": g}
    out["tent"]["all_zero_from_byte"] = 1 + max(
        next(i for i in range(32) if M.legacy_tent(32, s)[i:] == bytes(32 - i)) for s in range(256))
    nxt = [0] * 65536
    for x in range(256):
        for y in range(256):
            nx, ny = M.legacy_henon_step(x, y)
            nxt[(x << 8) | y] = (nx << 8) | ny
    out["henon"] = {**output_distribution([M.legacy_henon(32, x, y) for x in range(256) for y in range(256)]),
                    "graph": functional_graph(nxt)}
    joint_distinct = 1
    shannon = min_ent = 0.0
    coll = 1.0
    for name in ("logistic", "chebyshev", "tent", "henon"):
        joint_distinct *= out[name]["distinct_outputs"]
        shannon += out[name]["shannon_bits"]
        min_ent += out[name]["min_entropy_bits"]
        coll *= out[name]["collision_probability"]
    out["joint"] = {
        "seed_bits": 40,
        "distinct_chaotic_keys": joint_distinct,
        "log2_distinct": math.log2(joint_distinct),
        "shannon_bits_uniform_seed": shannon,
        "min_entropy_bits_uniform_seed": min_ent,
        "collision_probability": coll,
        "expected_random_passphrases_to_first_collision": math.sqrt(math.pi / (2 * coll)),
        "note": "Upper bounds on the diversity of the v1 passphrase-derived value; BLAKE3 assumed collision-free.",
    }
    out["collision_search"] = legacy_collision_search()
    return out


def legacy_collision_search(prefix: str = "cryptochaos-demo-", limit: int = 4_000_000) -> dict[str, Any]:
    """Birthday search for two passphrases with identical v1 chaotic keys (and identical seeds)."""
    def classes(fn, seeds):
        ids: dict[bytes, int] = {}
        return {s: ids.setdefault(fn(s), len(ids)) for s in seeds}

    c_l = classes(lambda s: M.legacy_logistic(32, s), range(256))
    c_c = classes(lambda s: M.legacy_chebyshev(32, s), range(256))
    c_t = classes(lambda s: M.legacy_tent(32, s), range(256))
    c_h = classes(lambda s: M.legacy_henon(32, s >> 8, s & 0xFF), range(65536))
    seen_key: dict[tuple, str] = {}
    seen_seed: dict[bytes, str] = {}
    key_pair = seed_pair = None
    tried = 0
    for i in range(limit):
        pw = f"{prefix}{i}"
        h = blake3.blake3(pw.encode()).digest()[:5]
        k = (c_l[h[0]], c_c[h[1]], c_t[h[2]], c_h[(h[3] << 8) | h[4]])
        if key_pair is None and k in seen_key and M.legacy_seed(seen_key[k]) != h:
            key_pair = (seen_key[k], pw, i + 1)
        seen_key.setdefault(k, pw)
        if seed_pair is None:
            if h in seen_seed:
                seed_pair = (seen_seed[h], pw, i + 1)
            else:
                seen_seed[h] = pw
        tried = i + 1
        if key_pair and seed_pair:
            break
    res: dict[str, Any] = {"passphrase_pattern": prefix + "<i>", "searched": tried}
    if key_pair:
        a, b, n = key_pair
        res["distinct_seed_key_collision"] = {"passphrases": [a, b], "found_after": n,
                                              "seeds": [M.legacy_seed(a).hex(), M.legacy_seed(b).hex()],
                                              "verified": M.legacy_chaotic_key(a) == M.legacy_chaotic_key(b)}
    if seed_pair:
        a, b, n = seed_pair
        res["identical_seed_collision"] = {"passphrases": [a, b], "found_after": n, "seed": M.legacy_seed(a).hex(),
                                           "verified": M.legacy_chaotic_key(a) == M.legacy_chaotic_key(b)}
    return res


def v2_reduced(word_bits: int) -> dict[str, Any]:
    m = M.V2Maps(word_bits)
    out: dict[str, Any] = {}
    steps: dict[str, Callable[[int], int]] = {"logistic": m.logistic, "tent": m.tent, "chebyshev": m.chebyshev}
    for name, step in steps.items():
        nxt = [step(x) for x in range(m.M)]
        res: dict[str, Any] = {"graph": functional_graph(nxt)}
        if word_bits <= 16:
            seeds = range(m.M) if name == "chebyshev" else (m.fix_seed_1d(x) for x in range(m.M))
            res.update(output_distribution([m.emit_1d(step, s) for s in seeds]))
        out[name] = res
    return out


def v2_reduced_henon(word_bits: int) -> dict[str, Any]:
    m = M.V2Maps(word_bits)
    n = m.M
    nxt = [0] * (n * n)
    for x in range(n):
        for y in range(n):
            nx, ny = m.henon(x, y)
            nxt[x * n + y] = nx * n + ny
    res: dict[str, Any] = {"graph": functional_graph(nxt)}
    if word_bits <= 8:
        res.update(output_distribution([m.emit_henon(x, y) for x in range(n) for y in range(n)]))
    return res


def brent_cycle(step: Callable[[Any], Any], x0: Any, limit: int) -> int | None:
    """Brent's cycle detection; returns the cycle length if found within ``limit`` steps."""
    power = lam = 1
    tortoise, hare = x0, step(x0)
    n = 1
    while tortoise != hare:
        if n >= limit:
            return None
        if power == lam:
            tortoise, power, lam = hare, power * 2, 0
        hare = step(hare)
        lam += 1
        n += 1
    return lam


def v2_full_width_samples(samples: int, limit: int, seed: int) -> dict[str, Any]:
    m = M.V2Maps(64)
    rng = np.random.Generator(np.random.PCG64([seed, 7]))
    found = collections.Counter()
    for _ in range(samples):
        w = [int(v) for v in rng.integers(0, 2**63, size=5, dtype=np.uint64)]
        w = [v | (int(rng.integers(0, 2)) << 63) for v in w]
        for name, step, x0 in (("logistic", m.logistic, m.fix_seed_1d(w[0])), ("tent", m.tent, m.fix_seed_1d(w[1])),
                               ("chebyshev", m.chebyshev, w[2])):
            if brent_cycle(step, x0, limit) is not None:
                found[name] += 1
        if brent_cycle(lambda s: m.henon(*s), (w[3], w[4]), limit) is not None:
            found["henon"] += 1
    return {"samples": samples, "steps_searched": limit,
            "orbits_cycling_within_limit": {k: found.get(k, 0) for k in ("logistic", "tent", "chebyshev", "henon")}}


def run(ctx) -> dict[str, Any]:
    cfg = ctx.cfg["map_analysis"]
    result = {"legacy": legacy_analysis(), "v2_reduced": {}, "v2_reduced_henon": {}}
    for w in cfg["v2_reduced_bits_1d"]:
        result["v2_reduced"][str(w)] = v2_reduced(w)
    for w in cfg["v2_reduced_bits_henon"]:
        result["v2_reduced_henon"][str(w)] = v2_reduced_henon(w)
    result["v2_full_width"] = v2_full_width_samples(cfg["v2_samples"], cfg["v2_cycle_search_steps"],
                                                    ctx.cfg["experiment"]["seed"])
    (ctx.raw / "map_analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    j = result["legacy"]["joint"]
    return {"legacy_log2_distinct": round(j["log2_distinct"], 3),
            "legacy_min_entropy_bits": round(j["min_entropy_bits_uniform_seed"], 3)}
