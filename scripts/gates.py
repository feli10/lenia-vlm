"""Stage 0 gates of the escalation loop (project doc "VLM in the loop - escalation plan (core first)").

Gate 0e: several questions in one request. Jev bills the pictures (~97% of a duel) once per request and each extra
question only its text, so asking steer + floor + surprise about one child/parent pair in one request would cut
expedition calls by ~2/3, if the answers match asking each question on its own.

  PYTHONPATH=src:scripts python3 scripts/gates.py 0e --out runs/gate-0e --max-usd 0.10            # Mac, ~$0.045
  PYTHONPATH=src:scripts python3 scripts/gates.py 0e --out /tmp/g0e --judge fake                  # offline
  PYTHONPATH=src:scripts python3 scripts/gates.py 0e-report --out runs/gate-0e

Pairs (--pairs 40, half each kind, fixed seed) from the goal-session pilot (runs/goal-session), with the goal the
branch had, so the steer prompt is a real one:
- step: an accepted child and its parent (the expedition case; these passed steer at P >= 0.6 under jev-latest);
- cross: a world from another branch of the same elite against a branch parent, under that branch's goal, so P
  spreads around both thresholds.
Questions: steer (lenia.prompts.steer_prompt), the floor (COMPLEXITY) and the logged surprise question (SURPRISE),
each asked in both image orders, three ways: separate requests; one request in that order (A); one request in
reversed question order (B). P = 0.5 (x + 1 - y).

Pass (fixed before running, 10-29), for steer and floor, A vs separate and B vs separate:
  mean |dP| <= 0.03; >= 38 of 40 pairs on the same side of the threshold (steer 0.6, floor 0.4);
  position bias (mean P(first) over both orders) within 0.03 of separate.
Surprise is reported, not gated (it is only logged in expeditions). Reference noise: Jev re-asks at the pinned
version gave mean |dP| 0.014, max 0.03 (pin_check), and rank correlation 1.00 (gate 0f).
If A passes and B does not, combined requests are used only in one fixed question order.

From papers: nothing; this is an engineering check of the provider's multi-question requests.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, MultiQuery, Query, load_env
from lenia.prompts import COMPLEXITY, SURPRISE, steer_prompt

QNAMES = ("steer", "floor", "surprise")
THRESH = {"steer": 0.6, "floor": 0.4, "surprise": 0.5}
GATED = ("steer", "floor")


def gs_pairs(gs: Path, n: int, seed: int) -> list:
    st = json.loads((gs / "state.json").read_text())
    worlds = defaultdict(set)
    for b in st["branches"]:
        worlds[b["elite"]].update(h["image"] for h in b["history"])
    step, cross = [], []
    rng = np.random.default_rng(seed)
    for b in (b for b in st["branches"] if b["kind"] == "goal"):
        goals = {g["id"]: g for g in b["goals"]}
        H = b["history"]
        for i in range(1, len(H)):
            g = goals[H[i]["goal"]]
            g = {"id": g["id"], "sees": g["sees"], "push": g["push"]}
            step.append({"kind": "step", "child": H[i]["image"], "parent": H[i - 1]["image"], "goal": g})
            others = sorted(worlds[b["elite"]] - {H[i]["image"], H[i - 1]["image"]})
            if others:
                cross.append({"kind": "cross", "child": others[int(rng.integers(len(others)))], "parent": H[i - 1]["image"],
                              "goal": g})
    pick = lambda xs, k: [xs[int(j)] for j in sorted(rng.choice(len(xs), min(k, len(xs)), replace=False))]
    a = pick(step, n // 2)
    used = {(p["child"], p["parent"]) for p in a}
    cross = [p for p in cross if (p["child"], p["parent"]) not in used and (p["parent"], p["child"]) not in used]
    b = pick(cross, len(cross))                           # shuffled; keep the first n/2 distinct pairs
    out = []
    for p in b:
        k = (p["child"], p["parent"])
        if k not in used and k[::-1] not in used and len(out) < n - n // 2:
            used.add(k)
            out.append(p)
    return a + out


def g0e(a) -> None:
    out, gs = Path(a.out), Path(a.goal_session)
    out.mkdir(parents=True, exist_ok=True)
    pairs = gs_pairs(gs, a.pairs, a.seed)
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    names = sorted({p[k] for p in pairs for k in ("child", "parent")})
    pics = {n: np.asarray(Image.open(gs / "img" / n).convert("RGB")) for n in names}
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                  out / "judge_cache.sqlite", yes=True)
    pids = [(steer_prompt(p["goal"]), COMPLEXITY, SURPRISE) for p in pairs]

    # separate: one request per question and order
    qs = [Query(pid, *ims) for p, ps in zip(pairs, pids) for pid in ps
          for ims in ((pics[p["child"]], pics[p["parent"]]), (pics[p["parent"]], pics[p["child"]]))]
    vs = judge.judge_many(qs, desc="0e separate")
    res = {"separate": {q: [] for q in QNAMES}}
    cost = {"separate": sum(v.cost for v in vs), "requests_separate": len(qs)}
    for n in range(len(pairs)):
        for j, q in enumerate(QNAMES):
            x, y = vs[n * 6 + 2 * j], vs[n * 6 + 2 * j + 1]
            res["separate"][q].append(None if not (x.ok and y.ok) else [x.p_first, y.p_first])

    # combined: one request per order, questions in order (A) and reversed (B)
    for mode, rev in (("A", False), ("B", True)):
        mq = [MultiQuery(ps[::-1] if rev else ps, *ims) for p, ps in zip(pairs, pids)
              for ims in ((pics[p["child"]], pics[p["parent"]]), (pics[p["parent"]], pics[p["child"]]))]
        mv = judge.judge_multi(mq, desc=f"0e combined {mode}")
        cost[mode] = sum(v.cost for vv in mv for v in vv)
        cost[f"requests_{mode}"] = len(mq)
        res[mode] = {q: [] for q in QNAMES}
        for n in range(len(pairs)):
            x, y = mv[2 * n], mv[2 * n + 1]
            if rev:
                x, y = x[::-1], y[::-1]
            for j, q in enumerate(QNAMES):
                res[mode][q].append(None if not (x[j].ok and y[j].ok) else [x[j].p_first, y[j].p_first])
    if any(v is None for m in res.values() for q in m.values() for v in q):
        sys.exit("some calls failed (not cached); re-run to retry them")
    (out / "results.json").write_text(json.dumps({"pairs": len(pairs), "judge": judge.model, "cost": cost,
                                                   "snapshots": judge.cache.snapshots(), "p_first": res}, indent=1))
    print(f"judge spent ${judge.spent:.4f} (cache total ${judge.cache.total_cost():.4f})", file=sys.stderr)
    report(a)


def report(a) -> None:
    out = Path(a.out)
    R = json.loads((out / "results.json").read_text())
    res, cost = R["p_first"], R["cost"]
    P = {m: {q: np.array([0.5 * (x + 1 - y) for x, y in res[m][q]]) for q in QNAMES} for m in res}
    bias = {m: {q: float(np.mean(res[m][q])) for q in QNAMES} for m in res}
    kinds = np.array([p["kind"] for p in json.loads((out / "pairs.json").read_text())])
    n = len(kinds)
    L = [f"# Gate 0e: several questions in one request ({R['judge']}, {n} pairs, both orders)\n",
         "P = both-order win probability of the child. A = one request (steer, floor, surprise); B = one request in "
         "reversed question order. Pass for steer and floor: mean |dP| <= 0.03, >= 38/40 same side of the threshold, "
         "position bias within 0.03.\n",
         "| question | comparison | mean abs dP | max abs dP | corr | same side | bias (mean P first) |", "|---|---|---|---|---|---|---|"]
    verdict = {}
    for q in QNAMES:
        for m, ref in (("A", "separate"), ("B", "separate"), ("B", "A")):
            d = np.abs(P[m][q] - P[ref][q])
            side = int(((P[m][q] >= THRESH[q]) == (P[ref][q] >= THRESH[q])).sum())
            c = float(np.corrcoef(P[m][q], P[ref][q])[0, 1]) if P[ref][q].std() > 0 else float("nan")
            db = abs(bias[m][q] - bias[ref][q])
            ok = d.mean() <= 0.03 and side >= round(0.95 * n) and db <= 0.03
            if q in GATED and ref == "separate":
                verdict[(q, m)] = ok
            L.append(f"| {q} | {m} vs {ref} | {d.mean():.3f} | {d.max():.3f} | {c:.3f} | {side}/{n} | "
                     f"{bias[m][q]:.3f} vs {bias[ref][q]:.3f}{'' if q not in GATED or ref != 'separate' else (' PASS' if ok else ' FAIL')} |")
    L += ["", "| question | separate: mean P (step / cross) | share >= threshold (separate) |", "|---|---|---|"]
    for q in QNAMES:
        L.append(f"| {q} | {P['separate'][q][kinds == 'step'].mean():.2f} / {P['separate'][q][kinds == 'cross'].mean():.2f} | "
                 f"{(P['separate'][q] >= THRESH[q]).mean():.2f} |")
    if cost.get("separate"):
        ps = cost["separate"] / cost["requests_separate"]
        L += ["", f"Cost per request: separate ${ps:.6f}; combined A ${cost['A'] / cost['requests_A']:.6f} "
              f"({cost['A'] / cost['requests_A'] / ps:.2f}x one duel, for 3 questions); this gate ${sum(cost[m] for m in ('separate', 'A', 'B')):.4f}."]
    a_ok = all(verdict[(q, "A")] for q in GATED)
    b_ok = all(verdict[(q, "B")] for q in GATED)
    L += ["", "**Verdict:** " + ("PASS: combine questions, in any order." if a_ok and b_ok else
                                  "PASS in the fixed order only: combine, always in one question order." if a_ok else
                                  "FAIL: ask the questions separately.")]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("0e")
    e.add_argument("--out", default="runs/gate-0e")
    e.add_argument("--goal-session", default="runs/goal-session")
    e.add_argument("--pairs", type=int, default=40)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--judge", default="jev")
    e.add_argument("--config", default=None)
    e.add_argument("--max-usd", type=float, default=0.10, help="hard cap on this gate's spend (its own cache)")
    r = sub.add_parser("0e-report")
    r.add_argument("--out", default="runs/gate-0e")
    a = ap.parse_args()
    {"0e": g0e, "0e-report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
