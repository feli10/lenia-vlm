"""Gate 0a: which generative VLM names the trigger worlds? (project doc "VLM in the loop - escalation plan (core first)").

  PYTHONPATH=src:scripts python3 scripts/bakeoff.py ping                                    # 1 tiny call per model (~$0.01)
  PYTHONPATH=src:scripts python3 scripts/bakeoff.py names --first 2                         # 10 calls: tokens and cost
  PYTHONPATH=src:scripts python3 scripts/bakeoff.py names                                   # all 12 triggers x 5 models
  PYTHONPATH=src:scripts python3 scripts/bakeoff.py rate                                    # your blind ratings (Mac)
  PYTHONPATH=src:scripts python3 scripts/bakeoff.py verify                                  # Jev verify (after rating)
  PYTHONPATH=src:scripts python3 scripts/bakeoff.py report
  ... names --models fake --out /tmp/b0a --stage1 /tmp/b0a-s1                              # offline dry run

Setup: every model names the seed-0 triggers (runs/stage1-s0/triggers.json) in the trigger list's order, as arm N
will: the trigger world, 4 random map elites from the snapshot nearest its iteration (the same 4 for every model) and
the descriptions this model already gave to earlier triggers (lenia.prompts.NAMER -> {sees, push}). A model whose call
fails stops at that trigger; re-running continues from the cache. The 8 "typical" worlds of the original 0a design
are dropped: they served the `new: false` check, which moved to Stage 2.

Caches live in --stage1 (the seed's Stage 1 folder): describe_cache.sqlite for the VLM, judge_cache.sqlite for Jev, so
arm N later reuses the chosen model's names and their verify duels at no cost.

Judging (fixed 10-29, before any name was seen):
- primary: Felix, blind (model hidden, shuffled), per name: is `sees` right about the world (yes / partly / no), and
  is it what is notable about this world (yes / no). A model passes if >= 2/3 of its names are right (yes) and
  >= 1/2 notable.
- verify (gate 0d's frozen rule: the true world beats both random decoys), reported per model against gate 0d's
  calibration on statements Felix judged correct: 0.68 overall, 0.57 for statements Jev did not pick itself. Not a
  pass/fail bar (gate 0d: it misses ~1/3 of correct statements, mostly generic ones).
- tokens and cost per call, from the first --first calls and the whole run.
If an open-weights model passes, Tim's local model is worth asking for.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import stage1 as S1
from lenia.config import REPO_ROOT, load
from lenia.describe import Ask, Describer
from lenia.judge import Judge, Query, load_env
from lenia.prompts import NAMER, verify_prompt

MODELS = "gemini-pro,gemini-flash,qwen-8b,qwen-235b,gemma-31b"


def pic(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def describers(a, names: list, cache: Path) -> dict:
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    return {m: Describer(m, cfg["vlms"][m], cache, max_usd=a.max_vlm_usd) for m in names}


def ping(a) -> None:
    out = Path(a.out)
    small = [np.full((280, 280, 3), 30 * k, np.uint8) for k in range(5)]
    for m, d in describers(a, a.models.split(","), out / "ping_cache.sqlite").items():
        t0 = time.time()
        try:
            [x] = d.ask_many([Ask(NAMER, small, {"names": "(none yet)"})], desc="ping")
            print(f"{m:<14} {'ok ' if x.ok else 'BAD'} {time.time() - t0:5.1f}s version={x.version} usage={x.usage} "
                  f"cost=${x.cost:.5f} {'' if x.ok else x.raw[:200]}")
        except Exception as e:
            print(f"{m:<14} FAILED {type(e).__name__}: {str(e)[:300]}")


def names(a) -> None:
    out, s1, run = Path(a.out), Path(a.stage1), Path(a.run)
    out.mkdir(parents=True, exist_ok=True)
    T = json.loads((s1 / "triggers.json").read_text())["triggers"][:a.first or None]
    models = a.models.split(",")
    ds = describers(a, models, s1 / "describe_cache.sqlite")
    imgs = {t["id"]: [pic(s1 / "img" / t["image"])] + [pic(run / "img" / n) for n in S1.typical(run, t)] for t in T}
    res = {m: [] for m in models}
    stopped = set()
    lock = threading.Lock()

    def step(m, t):
        given = [x["sees"] for x in res[m]]
        fill = {"names": "\n".join(f"- {g}" for g in given) if given else "(none yet)"}
        [x] = ds[m].ask_many([Ask(NAMER, imgs[t["id"]], fill)], desc=t["id"])
        with lock:
            if x.ok:
                res[m].append({"trigger": t["id"], "sees": x.data["sees"], "push": x.data["push"], "usage": x.usage,
                               "cost": x.cost, "cached": x.cached, "version": x.version, "typical": S1.typical(run, t)})
            else:
                stopped.add(m)
                print(f"{m}: failed at {t['id']}: {x.raw[:200]}", file=sys.stderr)

    for t in T:
        with ThreadPoolExecutor(max_workers=len(models)) as ex:
            list(ex.map(lambda m: step(m, t), [m for m in models if m not in stopped]))
    prev = json.loads((out / "names.json").read_text()) if (out / "names.json").exists() else {}
    prev.update({m: v for m, v in res.items() if len(v) >= len(prev.get(m, []))})
    (out / "names.json").write_text(json.dumps(prev, indent=1))
    print("\nmodel           calls  input   output  thinking   $/call   (new calls only)")
    for m in models:
        new = [x for x in res[m] if not x["cached"]] or res[m]
        if new:
            u = lambda k: np.mean([x["usage"].get(k) or 0 for x in new])
            print(f"{m:<15} {len(res[m]):>5} {u('input'):>6.0f} {u('output'):>8.0f} {u('thinking'):>9.0f}  "
                  f"{np.mean([x['cost'] for x in new]):.5f}")
    print(f"\nnames saved to {out / 'names.json'}; not printed here, so your ratings stay blind "
          f"({', '.join(f'{m} {len(v)}' for m, v in res.items())} of {len(T)})")
    spent = {m: d.spent for m, d in ds.items()}
    print(f"\nspent this run: {json.dumps({m: round(v, 4) for m, v in spent.items()})}; VLM cache total "
          f"${next(iter(ds.values())).cache.total_cost():.4f}")


def verify(a) -> None:
    out, s1, run = Path(a.out), Path(a.stage1), Path(a.run)
    N = json.loads((out / "names.json").read_text())
    T = {t["id"]: t for t in json.loads((s1 / "triggers.json").read_text())["triggers"]}
    st = json.loads((run / "state.json").read_text())
    dec = {tid: S1.verify_decoys(st, t) for tid, t in T.items()}
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                  s1 / "judge_cache.sqlite", yes=True)
    qs, meta = [], []
    for m, xs in N.items():
        for x in xs:
            t = T[x["trigger"]]
            W = pic(s1 / "img" / t["image"])
            for kind, n in [("near", dec[t["id"]]["near"])] + [("rand", r) for r in dec[t["id"]]["rand"]]:
                if n is None:
                    continue
                D = pic(run / "img" / n)
                qs += [Query(verify_prompt(x["sees"]), W, D), Query(verify_prompt(x["sees"]), D, W)]
                meta.append((m, x["trigger"], kind))
    vs = judge.judge_many(qs, desc="0a verify")
    if any(not v.ok for v in vs):
        sys.exit("some calls failed (not cached); re-run to retry them")
    res = {}
    for i, (m, tid, kind) in enumerate(meta):
        r = res.setdefault(m, {}).setdefault(tid, {"near": None, "rand": []})
        p = 0.5 * (vs[2 * i].p_first + 1 - vs[2 * i + 1].p_first)
        if kind == "near":
            r["near"] = p
        else:
            r["rand"].append(p)
    (out / "verify.json").write_text(json.dumps({"decoys": dec, "p": res}, indent=1))
    print(f"judge spent ${judge.spent:.4f}; results in {out / 'verify.json'} (run `report` after rating)", file=sys.stderr)


def rate(a) -> None:
    """Blind page: one (trigger, name) at a time, model hidden, fixed shuffled order; answers to rate/ratings.csv."""
    from lenia import rating
    out, s1 = Path(a.out), Path(a.stage1)
    rdir = out / "rate"
    (rdir / "images").mkdir(parents=True, exist_ok=True)
    N = json.loads((out / "names.json").read_text())
    T = {t["id"]: t for t in json.loads((s1 / "triggers.json").read_text())["triggers"]}
    if not (rdir / "tasks.json").exists():
        items = [(m, x) for m, xs in N.items() for x in xs]
        order = np.random.default_rng(a.seed).permutation(len(items))
        tasks = []
        for k, j in enumerate(order):
            m, x = items[int(j)]
            img = T[x["trigger"]]["image"]
            if not (rdir / "images" / img).exists():
                Image.open(s1 / "img" / img).save(rdir / "images" / img)
            tasks.append({"id": k, "image": img, "trigger": x["trigger"], "model": m, "sees": x["sees"], "push": x["push"]})
        (rdir / "tasks.json").write_text(json.dumps(tasks, indent=1))
    rating.rate_names(rdir, port=a.port)


def report(a) -> None:
    out = Path(a.out)
    N = json.loads((out / "names.json").read_text())
    V = json.loads((out / "verify.json").read_text())["p"] if (out / "verify.json").exists() else {}
    R = {}
    if (out / "rate" / "ratings.csv").exists():
        tasks = {t["id"]: t for t in json.loads((out / "rate" / "tasks.json").read_text())}
        for r in csv.DictReader(open(out / "rate" / "ratings.csv")):
            t = tasks[int(r["id"])]
            R[(t["model"], t["trigger"])] = r                     # the last answer counts
    L = ["# Gate 0a: naming bake-off (seed-0 triggers)\n",
         "verify = the true world beats both random decoys (gate 0d rule; calibration on correct library statements: "
         "0.68, or 0.57 for statements Jev did not pick). Pass: right >= 2/3 and notable >= 1/2 (Felix, blind).\n",
         "| model | names | verify | right (yes / partly / no) | notable | pass | $/name | input tok | output+thinking tok |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, xs in N.items():
        vp = [all(p > 0.5 for p in V[m][x["trigger"]]["rand"]) for x in xs if m in V and x["trigger"] in V[m]]
        rs = [R[(m, x["trigger"])] for x in xs if (m, x["trigger"]) in R]
        right = [r["right"] for r in rs]
        nota = [r["notable"] for r in rs]
        ok = rs and right.count("yes") >= 2 / 3 * len(rs) and nota.count("yes") >= 0.5 * len(rs)
        L.append(f"| {m} | {len(xs)} | {np.mean(vp):.2f} ({sum(vp)}/{len(vp)}) | "
                 + (f"{right.count('yes')} / {right.count('partly')} / {right.count('no')} | {nota.count('yes')}/{len(nota)} | "
                    f"{'PASS' if ok else 'fail'}" if rs else "- | - | -") +
                 f" | {np.mean([x['cost'] for x in xs]):.4f} | {np.mean([x['usage'].get('input') or 0 for x in xs]):.0f} | "
                 f"{np.mean([(x['usage'].get('output') or 0) + (x['usage'].get('thinking') or 0) for x in xs]):.0f} |"
                 if xs else f"| {m} | 0 | - | - | - | - | - | - | - |")
    seen = [t for t in a.seen.split(",") if t]
    if seen and R:
        L += ["", f"Without {', '.join(seen)} (names shown with model labels before rating): right (yes) / notable"]
        for m, xs in N.items():
            rs = [R[(m, x["trigger"])] for x in xs if (m, x["trigger"]) in R and x["trigger"] not in seen]
            if rs:
                L.append(f"- {m}: {sum(r['right'] == 'yes' for r in rs)}/{len(rs)} / {sum(r['notable'] == 'yes' for r in rs)}/{len(rs)}")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ping", "names", "verify", "rate", "report"])
    ap.add_argument("--out", default="runs/bakeoff-0a")
    ap.add_argument("--stage1", default="runs/stage1-s0", help="triggers.json, trigger pictures and the shared caches")
    ap.add_argument("--run", default="runs/mace-qd5-sarch-s0", help="the base run (typical worlds, decoys)")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--first", type=int, default=0, help="names: only the first k triggers (k x models calls)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--judge", default="jev")
    ap.add_argument("--max-vlm-usd", type=float, default=3.0, help="hard cap on the --stage1 VLM cache's total spend")
    ap.add_argument("--max-usd", type=float, default=0.5, help="hard cap on the --stage1 Jev cache's total spend")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--seen", default="s0-L55,s0-L19", help="report: triggers whose names Felix saw unblinded")
    a = ap.parse_args()
    {"ping": ping, "names": names, "verify": verify, "rate": rate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
