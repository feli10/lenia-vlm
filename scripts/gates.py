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
Result (10-29, runs/gate-0e, $0.047): PASS in any order; no systematic shift (signed mean dP +0.002 steer, -0.001
floor); flips only at the threshold; 3 questions cost 1.14x one duel.

Gate 0d: verify. Given a description (`sees`) of one world, does Jev pick that world? The true world duels a near decoy
(a random elite of the same lineage) and two random elites of other lineages, both orders, under
lenia.prompts.verify_prompt. Calibrated here on library-v3 statements that Felix judged blind (runs/goal-groups-v3/
verify.jsonl: 20 mace-qd2 worlds, yes/no; "unsure" dropped). Half the statements were Jev's own picks for the world
(world -> text) and half random decoys, so the split by source shows how much Jev's own matching flatters it.

  PYTHONPATH=src:scripts python3 scripts/gates.py 0d --out runs/gate-0d --max-usd 0.15            # Mac, ~$0.08
  PYTHONPATH=src:scripts python3 scripts/gates.py 0d --out /tmp/g0d --judge fake                  # offline

Pass (fixed before running, 10-29):
  1. statements Felix said fit: the true world beats both random decoys in >= 70% (25% for a coin-flip judge);
  2. Jev's mean P over the three duels separates Felix's yes from no: AUC >= 0.70, permutation p < 0.05.
The near decoy is reported, not gated. The verify rule used later ("beats all three" or "beats both random") is the
one with the better balanced accuracy against Felix here, frozen before any VLM name is verified.
Ours: verifying a description by picking its world against near and random decoys.
Result (10-29, runs/gate-0d, $0.068): FAIL by two statements (criterion 1: 47/69 = 0.68); criterion 2 AUC 0.80 PASS.
Jev reads text (AUC 0.84-0.86 on statements it did not pick) but cannot catch its own matching errors. On the 60 bake-off
names (runs/bakeoff-0a) the binary rule passed 59/60, wrong names included (trigger worlds are distinctive, so even
vague names clear 0.5), while the continuous P still tracks Felix (AUC 0.83 random decoys, 0.85 near): log P, not pass.

Gate 0b, cut down (10-29): the review prompt (lenia.prompts.REVIEW) as a sanity check, not a gate. The planned test
(do "refine" calls fall on pilot branches that later lost interest?) is confounded: every pilot branch was re-matched to a
new goal at generation 5, so later interest reflects a goal the review never saw; and N+R is exploratory in Stage 1.
Here the chosen VLM reads each of the 30 goal-session branches at generation 5 (start world, current parent, the
phase-1 goal, how many generations produced an accepted child) and answers continue / refine.
  checks: valid answers; not one decision for every branch; refined pushes build on the start's `sees`.
  reported: does "refine" go with Luna's measured lack of goal progress at generation 5 (the pilot's 'progress' duel:
  the gen-5 parent vs its start, attain prompt), and does it add anything beyond the accepted count alone; later
  interest (gen 10 vs start) for reference only.

  PYTHONPATH=src:scripts python3 scripts/gates.py 0b --out runs/gate-0b                         # Mac, ~$0.2 (Gemini Pro)
  PYTHONPATH=src:scripts python3 scripts/gates.py 0b --prompt refine --out runs/gate-0b-refine   # N+R's prompt
  PYTHONPATH=src:scripts python3 scripts/gates.py 0b --prompt switch --out runs/gate-0b-switch   # N+S's prompt
Result (review_v1, runs/gate-0b): refine tracked lack of progress (AUC 0.73, p = 0.02; the accepted count alone 0.34), but
about half the "refines" switched to a new feature, so refine and switch became separate prompts (REVIEW_R, REVIEW_S).
For switch, also reported: switches on branches that were making progress (Luna progress >= 0.6), the over-switching
failure Felix flagged.
  PYTHONPATH=src:scripts python3 scripts/gates.py 0b --out /tmp/g0b --model fake               # offline
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
from lenia.prompts import COMPLEXITY, SURPRISE, steer_prompt, verify_prompt
from lenia.stats import auc

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


# ---------- gate 0d: verify ----------

def d_items(gg: Path, qd: Path, library: Path, seed: int) -> tuple:
    lib = {e["id"]: e for e in json.loads(library.read_text())}
    V = [json.loads(l) for l in open(gg / "verify.jsonl")]
    items = [{"world": v["world"], "gid": v["gid"], "group": v["group"], "source": v["source"], "label": v["answer"],
              "sees": lib[v["gid"]]["sees"]} for v in V if v["answer"] in ("yes", "no") and v["gid"] in lib]
    st = json.loads((qd / "state.json").read_text())
    el = [{"name": f"I{k}c{ck}_final.png", "image": e["image"], "root": e["root"]}
          for k, m in enumerate(st["islands"]) for ck, e in sorted(m.items())]
    by = {e["name"]: e for e in el}
    decoys = {}
    for i, w in enumerate(sorted({x["world"] for x in items})):
        rng = np.random.default_rng([seed, i])
        t = by[w]
        same = [e for e in el if e["root"] == t["root"] and e["name"] != w]
        other = [e for e in el if e["root"] != t["root"]]
        decoys[w] = {"near": same[int(rng.integers(len(same)))]["image"],
                     "rand": [other[int(j)]["image"] for j in rng.choice(len(other), 2, replace=False)]}
    return items, decoys


def g0d(a) -> None:
    out, gg, qd = Path(a.out), Path(a.goal_groups), Path(a.qd)
    out.mkdir(parents=True, exist_ok=True)
    items, decoys = d_items(gg, qd, Path(a.library), a.seed)
    pic = lambda path: np.asarray(Image.open(path).convert("RGB"))
    true = {w: pic(qd / "check" / "images" / w) for w in decoys}
    dp = {n: pic(qd / "img" / n) for d in decoys.values() for n in [d["near"], *d["rand"]]}
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                  out / "judge_cache.sqlite", yes=True)
    qs = []
    for x in items:
        pid, T = verify_prompt(x["sees"]), true[x["world"]]
        for n in [decoys[x["world"]]["near"], *decoys[x["world"]]["rand"]]:
            qs += [Query(pid, T, dp[n]), Query(pid, dp[n], T)]
    vs = judge.judge_many(qs, desc="0d verify")
    if any(not v.ok for v in vs):
        sys.exit("some calls failed (not cached); re-run to retry them")
    for i, x in enumerate(items):
        p = [0.5 * (vs[6 * i + 2 * j].p_first + 1 - vs[6 * i + 2 * j + 1].p_first) for j in range(3)]
        x.update({"p_near": p[0], "p_rand": p[1:], "p_first": [vs[6 * i + k].p_first for k in range(6)]})
    (out / "results.json").write_text(json.dumps({"judge": judge.model, "snapshots": judge.cache.snapshots(),
                                                   "decoys": decoys, "items": items}, indent=1))
    print(f"judge spent ${judge.spent:.4f} (cache total ${judge.cache.total_cost():.4f})", file=sys.stderr)
    report_d(a)


def report_d(a) -> None:
    out = Path(a.out)
    R = json.loads((out / "results.json").read_text())
    X = R["items"]
    y = np.array([x["label"] == "yes" for x in X], int)
    near = np.array([x["p_near"] for x in X])
    rnd = np.array([x["p_rand"] for x in X])
    mean3 = (near + rnd.sum(1)) / 3
    both_r = (rnd > 0.5).all(1)
    all3 = both_r & (near > 0.5)
    src = np.array([x["source"] for x in X])
    L = [f"# Gate 0d: verify ({R['judge']}, {len(X)} statements Felix judged, 20 worlds)\n",
         "Per statement: the true world vs a near decoy (same lineage) and two random elites (other lineages), both "
         "orders; P = both-order probability Jev picks the true world. Chance: 0.5 per duel, 25% for beating both random "
         "decoys, 12.5% for all three.\n",
         "| Felix | source | n | mean P near | mean P random | beats both random | beats all three |", "|---|---|---|---|---|---|---|"]
    for lab in ("yes", "no"):
        for s_ in ("jev", "decoy", "all"):
            m = (y == (lab == "yes")) & ((src == s_) if s_ != "all" else True)
            if m.sum():
                L.append(f"| {lab} | {s_} | {m.sum()} | {near[m].mean():.2f} | {rnd[m].mean():.2f} | "
                         f"{both_r[m].mean():.2f} | {all3[m].mean():.2f} |")
    L += ["", "| score | AUC (Felix yes vs no) | p |", "|---|---|---|"]
    aucs = {}
    for nm, v in (("mean of 3", mean3), ("mean random", rnd.mean(1)), ("near", near), ("min of 3", np.minimum(near, rnd.min(1)))):
        A_, p = auc(list(v), list(y), n_perm=5000)
        aucs[nm] = (A_, p)
        L.append(f"| {nm} | {A_:.2f} | {p:.4f} |")
    bal = {}
    for nm, acc in (("beats both random", both_r), ("beats all three", all3)):
        tpr, tnr = acc[y == 1].mean(), 1 - acc[y == 0].mean()
        bal[nm] = (tpr + tnr) / 2
        L.append(f"\nRule \"{nm}\": accepts {tpr:.2f} of Felix-yes, rejects {tnr:.2f} of Felix-no; balanced accuracy {bal[nm]:.2f}.")
    c1 = both_r[y == 1].mean() >= 0.70
    c2 = aucs["mean of 3"][0] >= 0.70 and aucs["mean of 3"][1] < 0.05
    rule = max(bal, key=bal.get)
    pos = np.array([x["p_first"] for x in X]).mean()
    L += ["", f"Position bias: mean P(first) {pos:.3f}.",
          f"\n**Criterion 1** (Felix-yes beat both random >= 70%): {both_r[y == 1].mean():.2f} {'PASS' if c1 else 'FAIL'}. "
          f"**Criterion 2** (AUC mean of 3 >= 0.70, p < 0.05): {aucs['mean of 3'][0]:.2f} {'PASS' if c2 else 'FAIL'}.",
          f"\n**Verdict:** {'PASS' if c1 and c2 else 'FAIL'}. Verify rule for later: \"{rule}\"."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- gate 0b (cut down): the review prompt ----------

def g0b(a) -> None:
    from lenia.describe import Ask, Describer
    from lenia.prompts import REVIEW, REVIEW_R, REVIEW_S
    pid = {"v1": REVIEW, "refine": REVIEW_R, "switch": REVIEW_S}[a.prompt]
    out, gs = Path(a.out), Path(a.goal_session)
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads((gs / "state.json").read_text())
    C = json.loads((gs / "check.json").read_text())
    br = [b for b in st["branches"] if b["kind"] == "goal"]
    luna = C["p_first"]["luna"]
    rows = lambda t: [(r, luna[i]) for i, r in enumerate(C["rows"]) if r[0] == t]
    prog, intr = rows("progress"), rows("interest_goal_vs_start")
    assert len(prog) == len(br) == len(intr), (len(prog), len(br), len(intr))
    ph, total, mut = int(st["args"]["phase"]), int(st["args"]["generations"]), int(st["args"]["mutants"])
    pic = lambda n: np.asarray(Image.open(gs / "img" / n).convert("RGB"))
    items, asks = [], []
    for b, (pr, pp), (ir, ip) in zip(br, prog, intr):
        H = b["history"]
        cur = [h for h in H if h["gen"] <= ph][-1]
        assert pr[2] == cur["image"], (b["name"], pr[2], cur["image"])            # the progress duel is about this parent
        k = sum(1 <= h["gen"] <= ph for h in H)
        g = b["goals"][0]
        prog_txt = (f"{k} of {ph} generations produced an accepted child (a mutant the cheap judge chose as a step toward "
                    "the goal); in the others no mutant was good enough and the parent stayed the same.")
        items.append({"branch": b["name"], "accepted": k, "sees": g["sees"], "push": g["push"], "start": H[0]["image"],
                      "current": cur["image"], "luna_progress": pp, "luna_interest_end": ip})
        fill = {"mutants": mut, "gens": ph, "total": total, "sees": g["sees"], "push": g["push"], "progress": prog_txt}
        if pid != REVIEW:
            fill["left"] = total - ph
        asks.append(Ask(pid, [pic(H[0]["image"]), pic(cur["image"])], fill))
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    d = Describer(a.model, cfg["vlms"][a.model], out / "describe_cache.sqlite", max_usd=a.max_vlm_usd)
    ans = d.ask_many(asks, desc="0b review")
    for x, r in zip(items, ans):
        x.update({"ok": r.ok, "decision": (r.data or {}).get("decision", "").strip().lower(),
                  "new_push": (r.data or {}).get("push"), "new_sees": (r.data or {}).get("sees"), "why": (r.data or {}).get("why"),
                  "cost": r.cost, "usage": r.usage})
    (out / "results.json").write_text(json.dumps({"model": d.model, "prompt": pid, "items": items}, indent=1))
    print(f"VLM spent ${d.spent:.4f} (cache total ${d.cache.total_cost():.4f})", file=sys.stderr)
    report_b(a)


def report_b(a) -> None:
    out = Path(a.out)
    R = json.loads((out / "results.json").read_text())
    X = R["items"]
    menu = ("continue", "refine", "switch")
    ok = [x for x in X if x["ok"] and x["decision"] in menu]
    dec = np.array([x["decision"] for x in ok])
    pg = np.array([x["luna_progress"] for x in ok])
    it = np.array([x["luna_interest_end"] for x in ok])
    k = np.array([x["accepted"] for x in ok])
    change = (dec != "continue").astype(int)
    L = [f"# Gate 0b (cut down): review prompt {R.get('prompt', 'review_v1')} on 30 pilot branches at generation 5 "
         f"({R['model']})\n",
         f"{len(ok)}/{len(X)} valid answers; " + ", ".join(f"{m} {int((dec == m).sum())}" for m in menu if (dec == m).any()) +
         ". Luna progress = P(the gen-5 parent beats its start on the branch's goal), both orders.\n"]
    for m in menu:
        if (dec == m).any():
            L.append(f"- {m}: mean Luna progress {pg[dec == m].mean():.2f} (n = {int((dec == m).sum())})")
    if 0 < change.sum() < len(change):
        A1, p1 = auc(list(-pg), list(change), n_perm=5000)
        A2, p2 = auc(list(-k.astype(float)), list(pg < 0.5), n_perm=5000)
        A3, p3 = auc(list(-it), list(change), n_perm=5000)
        L += [f"- any change (refine or switch) vs lower progress: AUC {A1:.2f}, p = {p1:.3f}",
              f"- baseline, accepted count alone: AUC (fewer accepted -> progress < 0.5) {A2:.2f}, p = {p2:.3f}",
              f"- for reference, later interest vs change: AUC {A3:.2f}, p = {p3:.3f} (confounded by the pilot's re-match)"]
    hi = pg >= 0.6
    if hi.any():
        L.append(f"- branches making progress (Luna >= 0.6, n = {int(hi.sum())}): " +
                 ", ".join(f"{m} {int(((dec == m) & hi).sum())}" for m in menu if ((dec == m) & hi).any()))
    L += ["", "| branch | accepted | Luna progress | decision | change | why |", "|---|---|---|---|---|---|"]
    for x in X:
        d = x["decision"] or "FAILED"
        ch = ""
        if d == "refine":
            ch = f"{x['push']} -> {x['new_push']}"
        elif d == "switch":
            ch = f"was: {x['push']} / now sees: {x['new_sees']} -> {x['new_push']}"
        L.append(f"| {x['branch']} | {x['accepted']} | {x['luna_progress']:.2f} | {d} | {ch} | {x['why'] or ''} |")
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
    d = sub.add_parser("0d")
    d.add_argument("--out", default="runs/gate-0d")
    d.add_argument("--goal-groups", default="runs/goal-groups-v3", help="Felix's blind yes/no on library statements")
    d.add_argument("--qd", default="runs/mace-qd2", help="the run the test worlds and decoys come from")
    d.add_argument("--library", default="data/goal_library_v3.json")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--judge", default="jev")
    d.add_argument("--config", default=None)
    d.add_argument("--max-usd", type=float, default=0.15, help="hard cap on this gate's spend (its own cache)")
    dr = sub.add_parser("0d-report")
    dr.add_argument("--out", default="runs/gate-0d")
    b = sub.add_parser("0b")
    b.add_argument("--out", default="runs/gate-0b")
    b.add_argument("--goal-session", default="runs/goal-session")
    b.add_argument("--model", default="gemini-pro")
    b.add_argument("--prompt", default="v1", choices=["v1", "refine", "switch"], help="review_v1 | REVIEW_R | REVIEW_S")
    b.add_argument("--config", default=None)
    b.add_argument("--max-vlm-usd", type=float, default=0.5)
    br = sub.add_parser("0b-report")
    br.add_argument("--out", default="runs/gate-0b")
    a = ap.parse_args()
    {"0e": g0e, "0e-report": report, "0d": g0d, "0d-report": report_d, "0b": g0b, "0b-report": report_b}[a.cmd](a)


if __name__ == "__main__":
    main()
