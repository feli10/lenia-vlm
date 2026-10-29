"""Pin the Jev version: is the pinned name accepted, which snapshot answers, and does it answer like the model
behind our past results (`jev-latest` until 2026-10-28)?

Re-asks 20 duels from runs/res-check (full-size retest, asked under jev-latest on 10-28), both orders, under
each candidate name until one is accepted, and once more under jev-latest to see which snapshot that alias
points to now. About 120 calls, ~$0.015. Answers go to runs/pin-check/; nothing else is touched.

  python scripts/pin_check.py            # on the Mac (needs TYPESAFE_API_KEY in .env)
  python scripts/pin_check.py --judge fake --out /tmp/pin-dry    # offline dry run of the code path
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lenia.config import REPO_ROOT, load
from lenia.judge import FatalAPIError, Judge, Query, load_env
from lenia.prompts import SURPRISE


def ask(name: str, pairs: list, a, cfg, sample: int) -> dict:
    jc = {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "model": name, "max_run_usd": 0.05}
    judge = Judge(a.judge, jc, Path(a.out) / f"cache-{name}.sqlite", yes=True, log=lambda m: None)
    img = lambda p: np.asarray(Image.open(p).convert("RGB"))
    qs = [q for r in pairs for q in (Query(SURPRISE, img(r["a"]), img(r["b"]), sample),
                                     Query(SURPRISE, img(r["b"]), img(r["a"]), sample))]
    try:
        vs = judge.judge_many(qs, desc=name)
    except FatalAPIError as e:
        return {"accepted": False, "error": str(e)[:300]}
    x, y = np.array([v.p_first for v in vs[0::2]], float), np.array([v.p_first for v in vs[1::2]], float)
    ox, oy = np.array([r["x"] for r in pairs]), np.array([r["y"] for r in pairs])
    new, old = 0.5 * (x + 1 - y), 0.5 * (ox + 1 - oy)
    return {"accepted": True, "snapshots": judge.cache.snapshots(), "spent": round(judge.spent, 4),
            "mean_abs_diff": round(float(np.mean(np.abs(new - old))), 3),
            "max_abs_diff": round(float(np.max(np.abs(new - old))), 3),
            "same_winner": round(float(np.mean((new >= 0.5) == (old >= 0.5))), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="jev-1.14.0,jev-1.14", help="candidate pinned names, tried in order")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--judge", default="jev")
    ap.add_argument("--out", default="runs/pin-check")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    rows = json.loads(Path("runs/res-check/results.json").read_text())["retest/644"][:a.n]
    res = {}
    for name in a.models.split(","):
        res[name] = ask(name, rows, a, cfg, sample=0)
        print(name, json.dumps(res[name]), file=sys.stderr)
        if res[name]["accepted"]:
            break
    res["jev-latest"] = ask("jev-latest", rows, a, cfg, sample=2)
    print("jev-latest", json.dumps(res["jev-latest"]), file=sys.stderr)
    (Path(a.out) / "result.json").write_text(json.dumps(res, indent=1))
    ok = [k for k, v in res.items() if v.get("accepted") and k != "jev-latest"]
    pinned = cfg["judges"]["jev"]["model"]
    print(f"\npinned in configs/lenia.yaml: {pinned}; accepted name: {ok[0] if ok else 'none'}")
    if ok and ok[0] != pinned:
        print(f"-> set judges.jev.model to {ok[0]} in configs/lenia.yaml")
    if ok and res[ok[0]]["same_winner"] < 0.9:
        print("-> the pinned version answers differently from our past results: tell Claude before running Stage 1")


if __name__ == "__main__":
    main()
