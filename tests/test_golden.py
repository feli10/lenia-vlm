"""Golden values for everything the judge caches and saved runs depend on.

If any of these change, cached verdicts stop matching (the cache key hashes the prompt text and the picture
bytes) and saved runs (state.json rules, the fixed starting worlds) stop resuming or reproducing. Regenerate the
expected values only on purpose, when you mean to start fresh caches:

  PYTHONPATH=src:scripts python3 tests/test_golden.py --print

Runs under pytest or as a plain script (python3 tests/test_golden.py). Recorded before the 10-28 cleanup;
the cleanup moved code and prompts without changing any of these values.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lenia.judge import Judge, MultiQuery, Query, image_hash          # noqa: E402
from lenia.prompts import get_prompt                       # noqa: E402

from lenia import mace as M                                   # noqa: E402
from lenia.mace import mutate, texture                        # noqa: E402
from lenia.prompts import (INTEREST, NONE, SESSION_MATCH, STEER, SURPRISE,  # noqa: E402
                           attain_prompt, steer_prompt, verify_prompt, GEN, NAMER, REVIEW, REVIEW_R, REVIEW_S, GOAL_JUDGE,
                           CAPTION, NAMER_AIM, NAMER_AIM_V1, NAMER_AIM_V2)

GOAL = {"id": "golden", "sees": "a round body with a dark core", "push": "make the core split in two"}


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def compute() -> dict:
    out = {}
    out["prompt/complexity"] = _h(get_prompt(M.PROMPT_ID))
    out["prompt/surprise"] = _h(get_prompt(SURPRISE))
    out["prompt/steer"] = _h(get_prompt(steer_prompt(GOAL)))
    out["prompt/attain"] = _h(get_prompt(attain_prompt(GOAL)))
    out["prompt/interest"] = _h(get_prompt(INTEREST))
    out["prompt/verify"] = _h(get_prompt(verify_prompt(GOAL["sees"])))
    out["gen/namer"] = _h(GEN[NAMER])
    out["gen/review"] = _h(GEN[REVIEW])
    out["gen/review_refine"] = _h(GEN[REVIEW_R])
    out["gen/review_switch"] = _h(GEN[REVIEW_S])
    out["gen/goal_judge"] = _h(GEN[GOAL_JUDGE])
    out["gen/caption"] = _h(GEN[CAPTION])
    out["gen/namer_aim"] = _h(GEN[NAMER_AIM_V1])
    out["gen/namer_aim_v2"] = _h(GEN[NAMER_AIM_V2])
    out["gen/namer_aim_v3"] = _h(GEN[NAMER_AIM])
    out["menu/session_match"] = _h(SESSION_MATCH)                 # multiple-choice texts key the cache too
    out["menu/steer_groups"] = _h(STEER)
    out["menu/none"] = NONE
    rng = np.random.default_rng(0)
    frames = rng.random((M.FRAMES, 3, 40, 40)).astype(np.float32)
    pic = M.picture(frames)
    out["picture/shape"] = list(pic.shape)
    gap = M.TILE                                        # the 4-px separators between the 2x2 tiles
    out["picture/gaps"] = [int(pic[gap:gap + 4].min()), int(pic[gap:gap + 4].max()),
                           int(pic[:, gap:gap + 4].min()), int(pic[:, gap:gap + 4].max())]
    out["picture/stats"] = [float(x) for x in pic.reshape(-1, 3).mean(0)] + [float(x) for x in pic.reshape(-1, 3).std(0)]
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)   # fixed bytes: key format, independent of Pillow
    out["image_hash"] = image_hash(img)
    with tempfile.TemporaryDirectory() as d:
        j = Judge("jev", {"backend": "jev", "model": "jev-latest"}, Path(d) / "c.sqlite")
        out["cache_key"] = j.key(Query(M.PROMPT_ID, img, img[::-1].copy()))[:16]
        out["cache_key/multi"] = [k[:16] for k in j.multi_keys(MultiQuery((M.PROMPT_ID, SURPRISE), img, img[::-1].copy()))]
    rules, A0 = M.founders(Namespace(lineages=2, rule_seed=3, R=13, beta=2.0), (48, 48))
    out["founders"] = _h([r.to_dict() for r in rules])
    # rounded: np.exp differs in the last bit between x86 (Colab) and ARM (Mac), which changed an exact hash
    rnd = lambda o: ({k: rnd(v) for k, v in o.items()} if isinstance(o, dict) else [rnd(v) for v in o]
                     if isinstance(o, (list, tuple)) else round(o, 10) if isinstance(o, float) else o)
    out["mutate"] = _h(rnd(mutate(rules[0], np.random.default_rng(0), 3.0).to_dict()))
    final, films = M.simulate(rules, A0, 12, (48, 48), gap=4)
    out["simulate"] = [float(final.sum()), float((final ** 2).sum()), float(final[0, 0, 24, 24])]
    out["texture"] = float(texture(final[0]))
    return out


EXPECTED = {
    "prompt/complexity": "d38c09ab88d6d412",
    "prompt/surprise": "5c4c0e3708ac1496",
    "prompt/steer": "db0b97d7d5fe3f77",
    "prompt/attain": "46c8d4b2dfab9bfe",
    "picture/shape": [
        644,
        644,
        3
    ],
    "picture/gaps": [
        40,
        40,
        40,
        40
    ],
    "picture/stats": [
        126.7348867713437,
        126.44670344508313,
        128.22314918405925,
        50.023514836976716,
        49.82027713611228,
        50.27465218518456
    ],
    "image_hash": "6457bea750674dad26db",
    "cache_key": "2cb0284a9fa3e54f",
    "founders": "380698ada0694a32",
    "mutate": "8432a5d54ebb9323",          # 2026-10-29: now rounded to 10 decimals (x86 vs ARM last-bit np.exp); same rule
    "simulate": [
        6517.357421875,
        5264.0029296875,
        0.8589450120925903
    ],
    "texture": 0.021451515386685997,
    "prompt/interest": "9b858956b3910b81",
    "menu/session_match": "824bc47345e20218",
    "menu/steer_groups": "54b0d7e501852ce9",
    "menu/none": "none",
    "cache_key/multi": ["de49508527e84dd2", "3d301071e3a401d2"],  # added 2026-10-29 (judge.MultiQuery, gate 0e)
    "prompt/verify": "f5ccb529d327f9d9",                           # added 2026-10-29 (gate 0d)
    "gen/namer": "f57b109cd65cea2d",                               # added 2026-10-29 (lenia.describe, gate 0a)
    "gen/review": "b7339aa9d816035a",
    "gen/review_refine": "9e8f6554186c3c49",                       # added 2026-10-29 (N+R / N+S split after gate 0b)
    "gen/review_switch": "1c5d1362a9865626",
    "gen/goal_judge": "51c9fae75dfb3f8d",                          # added 2026-10-29 (goal check after arm L)
    "gen/caption": "f45934b78e738e80",                             # added 2026-11-02 (goal captions, Stage 1 v4)
    "gen/namer_aim": "5b9b6e0dca5eca58",                           # added 2026-11-04 (aim-conditioned namer)
    "gen/namer_aim_v2": "ebe4bf216e41b23c",                        # added 2026-11-04 (v2: the mass budget as a fact)
    "gen/namer_aim_v3": "33688547f4657c6a"                                     # added 2026-11-04 (v3: choose among distinctive features)
}   # recorded 2026-10-28 (cleanup); prompts, founders, mutate, simulate unchanged from before it; menus and interest added before the prompt move


def test_golden():
    got = compute()
    for k, v in EXPECTED.items():
        if k in ("simulate", "texture"):          # float maths: tolerate platform rounding, not real changes
            assert np.allclose(got[k], v, rtol=1e-4, atol=1e-6), (k, got[k], v)
        elif k == "picture/stats":                # Pillow's resampling can differ slightly between versions
            assert np.allclose(got[k], v, atol=0.5), (k, got[k], v)
        else:
            assert got[k] == v, (k, got[k], v)


if __name__ == "__main__":
    if "--print" in sys.argv:
        print(json.dumps(compute(), indent=1))
    else:
        test_golden()
        print("golden: all values match")
