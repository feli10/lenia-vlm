"""Several duel questions in one request (judge.MultiQuery, gate 0e): request shape, answer parsing, cost booking,
cache keys and all-or-nothing caching. Offline. Runs under pytest or as a plain script (python3 tests/test_multi.py).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lenia.judge import Judge, MultiQuery, Query      # noqa: E402
from lenia.prompts import COMPLEXITY, SURPRISE         # noqa: E402

IMG = np.full((300, 300, 3), 200, np.uint8)
IMG2 = IMG.copy()
IMG2[100:200, 100:200] = 20
PIDS = (COMPLEXITY, SURPRISE)


def _judge(backend, d, **cfg):
    return Judge(backend, {"backend": backend, "model": "m", **cfg}, Path(d) / "c.sqlite", yes=True, log=lambda m: None)


def test_request_shape_and_parsing():
    seen = {}

    def fake_post(self, url, body, headers):
        seen[self.backend] = body
        return {"answers": {"q1": {"choice": "second", "probabilities": {"first": 0.2, "second": 0.8}},
                            "q2": {"choice": "first", "probabilities": {"first": 0.7, "second": 0.3}}},
                "usage": {"input_tokens": 1000, "cost": 0.0001}, "model": "jev-1.14.0"}
    old = Judge._post
    Judge._post = fake_post
    try:
        import os
        os.environ.setdefault("OPENROUTER_API_KEY", "x")
        os.environ.setdefault("TYPESAFE_API_KEY", "y")
        with tempfile.TemporaryDirectory() as d:
            for b in ("decisions", "jev"):
                vs = _judge(b, d)._call_multi(MultiQuery(PIDS, IMG, IMG2))
                assert [round(v.p_first, 6) for v in vs] == [0.2, 0.7] and [v.choice for v in vs] == ["second", "first"]
                assert vs[0].cost > 0 and vs[1].cost == 0                       # one request, booked once
                assert vs[0].snapshot == "jev-1.14.0"
    finally:
        Judge._post = old
    body = seen["jev"]
    assert len(body["assets"]) == 2 and set(body["questions"]) == {"q1", "q2"}
    assert "`first`" in body["questions"]["q1"]["instructions"]
    assert body["questions"]["q1"]["instructions"] != body["questions"]["q2"]["instructions"]
    assert all(set(q["criteria"]) == {"first", "second"} for q in body["questions"].values())
    assert set(seen["decisions"]["questions"]) == {"q1", "q2"} and len(seen["decisions"]["state"]) == 4


def test_keys():
    with tempfile.TemporaryDirectory() as d:
        j = _judge("jev", d)
        k = j.multi_keys(MultiQuery(PIDS, IMG, IMG2))
        kr = j.multi_keys(MultiQuery(PIDS[::-1], IMG, IMG2))
        single = {j.key(Query(p, IMG, IMG2)) for p in PIDS}
        assert len(set(k)) == 2 and not set(k) & single                # never served for a single question
        assert not set(k) & set(kr)                                    # nor for another order / combination
        assert k == j.multi_keys(MultiQuery(PIDS, IMG, IMG2))          # deterministic


def test_cache_all_or_nothing():
    with tempfile.TemporaryDirectory() as d:
        j = _judge("fake", d)
        qs = [MultiQuery(PIDS, IMG, IMG2), MultiQuery(PIDS, IMG2, IMG), MultiQuery(PIDS, IMG, IMG2)]   # one duplicate
        calls = []
        old = Judge._call_multi
        Judge._call_multi = lambda self, q: calls.append(1) or old(self, q)
        try:
            a = j.judge_multi(qs)
            assert len(calls) == 2 and len(a) == 3 and all(len(vs) == 2 and all(v.ok for v in vs) for vs in a)
            assert [v.p_first for v in a[0]] == [v.p_first for v in a[2]]
            b = j.judge_multi(qs)
            assert len(calls) == 2 and [[v.p_first for v in vs] for vs in a] == [[v.p_first for v in vs] for vs in b]
        finally:
            Judge._call_multi = old
        # fake answers per question equal the single-question fake answers (fake is independent by construction)
        assert abs(a[0][0].p_first - j.judge_many([Query(COMPLEXITY, IMG, IMG2)])[0].p_first) < 1e-12
        from lenia.judge import Verdict
        j2 = _judge("fake", d + "/x")
        Judge._call_multi = lambda self, q: [Verdict(0.6, "first", "{}", 0.001), Verdict(None, None, "parse error")]
        try:
            j2.judge_multi([MultiQuery(PIDS, IMG, IMG2)] * 1)
        finally:
            Judge._call_multi = old
        assert j2.cache.total_cost() == 0                                # partial answers are not cached


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"{name}: ok")
