"""Mutation-operator pilot: the operators (offline, fast). The end-to-end run is in test_lenia.test_pool_end_to_end."""
from __future__ import annotations

import numpy as np
import pytest

from lenia import operators as ops
from lenia import pilot
from lenia.sim import Batch, Kernel, Rule


def full_rule(C=3, T=10.0):
    """One kernel per (source, target) pair, as Chan's 9-kernel 3-channel creatures."""
    ks = [Kernel(m=0.15 + 0.01 * i, s=0.02, b=(1.0,), r=0.8, h=0.5, c0=a, c1=b)
          for i, (a, b) in enumerate((a, b) for a in range(C) for b in range(C))]
    return Rule(ks, R=13, T=T, channels=C)


def wiring(r):
    return [(k.c0, k.c1) for k in r.kernels]


def test_valid_and_incoming():
    r = full_rule()
    assert ops.valid(r) and list(ops.incoming(r)) == [3, 3, 3]
    lonely = Rule([Kernel(0.15, 0.02, c0=0, c1=0)], R=13, channels=2)        # channel 1 is never written to
    assert not ops.valid(lonely)


def test_structural_edits():
    rng = np.random.default_rng(0)
    r = full_rule()
    for _ in range(20):
        a = ops.add_kernel(r, rng)
        assert len(a.kernels) == 10 and ops.valid(a) and a.kernels[-1].h <= 0.25 * 0.5 + 1e-9
        d = ops.remove_kernel(r, rng)
        assert len(d.kernels) == 8 and ops.valid(d)
        w = ops.rewire(r, rng)
        assert ops.valid(w) and sum(x != y for x, y in zip(wiring(w), wiring(r))) == 1
    assert wiring(r) == wiring(full_rule()) and len(r.kernels) == 9            # inputs untouched


def test_duplicate_keeps_weights_and_split_is_a_noop():
    rng = np.random.default_rng(1)
    r = full_rule()
    d = ops.duplicate_kernel(r, rng)
    assert len(d.kernels) == 10 and ops.valid(d)
    h = lambda rule, c: sum(k.h for k in rule.kernels if k.c1 == c)
    assert all(abs(h(d, c) - h(r, c)) < 1e-9 for c in range(3))
    # the premise: a kernel split into two identical halves is the same rule
    split = Rule([*r.kernels[:4], Kernel(**{**r.kernels[4].__dict__, "h": 0.25}),
                  *r.kernels[5:], Kernel(**{**r.kernels[4].__dict__, "h": 0.25})], R=13, T=10, channels=3)
    A = np.random.default_rng(2).random((1, 3, 48, 48)).astype(np.float32)
    assert np.allclose(Batch([r], (48, 48)).step(A), Batch([split], (48, 48)).step(A), atol=1e-5)


def test_edits_that_cannot_apply():
    rng = np.random.default_rng(0)
    diag = Rule([Kernel(0.15, 0.02, c0=0, c1=0), Kernel(0.2, 0.02, c0=1, c1=1)], R=13, channels=2)
    assert ops.remove_kernel(diag, rng) is None                                 # each channel's only writer
    for _ in range(10):                                                         # only sources can move
        assert [k.c1 for k in ops.rewire(diag, rng).kernels] == [0, 1]
    one = Rule([Kernel(0.15, 0.02)], R=13)
    assert ops.rewire(one, rng) is None and ops.crossover(one, one, rng) is None


def test_crossover_takes_whole_channels():
    rng = np.random.default_rng(3)
    a, b = full_rule(), full_rule()
    for k in b.kernels:
        k.m += 0.1
    for _ in range(20):
        c = ops.crossover(a, b, rng)
        src = {}
        for kc, ka in zip(c.kernels, a.kernels):
            src.setdefault(kc.c1, set()).add("a" if kc.m == ka.m else "b")
        assert all(len(v) == 1 for v in src.values()) and {"a", "b"} == set().union(*src.values())
    assert ops.crossover(a, full_rule(T=1.0), rng) is None                     # different layout


def test_pattern_operators():
    rng = np.random.default_rng(4)
    P = np.zeros((3, 40, 40), np.float32)
    P[:, 10:30, 12:28] = 0.8
    c = ops.cut(P, rng)
    assert c.shape == P.shape and 0.2 * P.sum() < c.sum() < 0.8 * P.sum()
    Q = np.zeros((3, 50, 30), np.float32)
    Q[:, 5:45, 5:25] = 0.5
    m = ops.merge(P, Q, rng)
    assert m.shape == (3, 50, 50) and m.max() == pytest.approx(0.8) and m.sum() >= P.sum()
    with pytest.raises(ValueError):
        ops.merge(P, Q[:2], rng)
    s = ops.reseed(full_rule(), rng, 64)
    assert s.shape == (3, 64, 64) and s.max() > 0
    assert P[:, 10:30, 12:28].min() == pytest.approx(0.8)                      # input untouched


def test_scaled_mutation_grows_with_scale():
    r = full_rule()
    step = lambda s: np.mean([abs(k2.m - k1.m) for seed in range(30)
                              for k1, k2 in zip(r.kernels, ops.scaled(r, np.random.default_rng(seed), s).kernels)])
    assert step(1) < step(2) < step(4)


def test_parents_and_children():
    r, P = full_rule(), np.ones((3, 20, 20), np.float32)
    el = [{"key": f"{m}:{i}", "map": m, "cell": f"{i},0", "rule": r, "pattern": P} for m in ("qa", "qb") for i in range(10)]
    el.append({"key": "qb:99", "map": "qb", "cell": "9,9", "rule": full_rule(T=1.0), "pattern": P})
    ps = pilot.pick_parents(el, 7, seed=0)
    assert len(ps) == 7 and sum(p["map"] == "qa" for p in ps) == 4 and len({p["key"] for p in ps}) == 7
    assert [p["key"] for p in pilot.pick_parents(el, 7, seed=0)] == [p["key"] for p in ps]
    rng = np.random.default_rng(0)
    for op in ops.OPERATORS:
        kids = pilot.make_children(op, ps, el, rng)
        assert len(kids) == 7 and all(k["rule"] is not None and k["start"] is not None for k in kids), op
        if op == "copy":
            assert all(k["rule"] is k["parent"]["rule"] and k["start"] is k["parent"]["pattern"] for k in kids)
        if op == "crossover":
            assert all(k["partner"]["key"] != k["parent"]["key"] and k["partner"]["rule"].T == 10.0 for k in kids)
    with pytest.raises(ValueError):
        pilot.make_children("nope", ps, el, rng)


def test_verdict_and_summary():
    assert pilot.verdict(0.2, 0.6) == "pass"
    assert pilot.verdict(0.05, 0.6) == "fail: too few viable"
    assert pilot.verdict(0.5, 0.4) == "fail: too near the pool"
    row = lambda op, stage, nn: {"op": op, "applicable": "True", "stage": stage, "nn": nn, "tries": "1",
                                 "d_nearest_parent": 0.3, "student": 1.0, "student_parent": 0.5}
    rows = ([row("mut4", "viable", 0.6)] * 2 + [row("mut4", "dead@100", "")] * 8
            + [row("reseed", "viable", 0.4)] * 5 + [row("reseed", "failed@2x", "")] * 5)
    s = {x["op"]: x for x in pilot.summarise(rows, pool_p90=0.53)}
    assert s["mut4"]["verdict"] == "pass" and s["mut4"]["viable_frac"] == 0.2 and s["mut4"]["beyond_pool_p90"] == 1.0
    assert s["reseed"]["verdict"] == "fail: too near the pool" and s["reseed"]["stages"]["failed@2x"] == 5
    assert s["mut4"]["student_delta_median"] == 0.5
