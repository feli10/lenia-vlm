"""Stage 1 of the escalation loop (project doc "VLM in the loop - escalation plan (core first)").

Step 1, the trigger list:

  PYTHONPATH=src:scripts python3 scripts/stage1.py triggers --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 --max-usd 0.25
  PYTHONPATH=src:scripts python3 scripts/stage1.py triggers --run runs/mace-qd5-sarch-s0 --out /tmp/s1-dry --judge fake
  PYTHONPATH=src:scripts python3 scripts/stage1.py sheet --out runs/stage1-s0          # redraw triggers.png only

Goal captions for the v4 CLIP shortlist (--arm N or L; hand captions for L's seed-0 goals in data/goal_library_v3_captions.json):

  PYTHONPATH=src:scripts python3 scripts/stage1.py captions --arm N --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0
  PYTHONPATH=src:scripts python3 scripts/stage1.py captions --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 \\
      --caption-vlm fake --dest /tmp/cap-dry

Caption check (v4 step 2; Colab, GPU, no API calls; regrows expedition v3's arm L children):

  python scripts/stage1.py capcheck --run RUNS/mace-qd5-sarch-s0 --out RUNS/stage1-s0 --backend auto --evals /content/capcheck-evals
  PYTHONPATH=src:scripts python3 scripts/stage1.py capcheck --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 --clip fake \\
      --limit 1 --steps 2 --size 48 --warmup 20 --render 12 --gap 4 --dry --evals /tmp/cc-dry

Expedition v4 (Colab: GPU and Jev; see colab/mace.ipynb section 5c). Smoke test first, then the arms:

  python scripts/stage1.py v4 --arm N --run RUNS/mace-qd5-sarch-s0 --out RUNS/stage1-s0 --backend auto --evals /content/evals \\
      --tag exp-v4-smoke --limit 2 --stop-after 3
  python scripts/stage1.py v4 --arm L --run RUNS/mace-qd5-sarch-s0 --out RUNS/stage1-s0 --backend auto --evals /content/evals
  ... then --arm N, NF, C. Results in <out>/exp-v4/<arm>/: state.json, duels.csv (every child), trace.jsonl (picks),
  sheet.png (start | halfway | end), summary.md.

Step 5 on, the expeditions (Colab: GPU and judge calls; see colab/mace.ipynb):

  python scripts/stage1.py expedition --arm L  --run RUNS/mace-qd5-sarch-s0 --out RUNS/stage1-s0 --backend auto
  python scripts/stage1.py check      --arm L  --run RUNS/mace-qd5-sarch-s0 --out RUNS/stage1-s0   # Luna; gate 0c for L
  ... then --arm N, NR, NS (NR and NS need N's goals; their review calls need Gemini: GOOGLE_CLOUD_PROJECT + ADC)
  Expeditions are v2 (fixed reference: the trigger's start) in <out>/exp-v2/; <out>/arms/L is v1 (see below).

Triggers: every surprise-archive member of the base run is re-scored against the run's final map, and each
lineage (seed root) contributes its best member. A member's stored score was measured against the map and archive
of its own iteration (early members against near-random founders); re-scoring puts every member against one
reference, what the run eventually produced.
- Opponents: one panel of --opponents final-map elites per lineage, drawn uniformly from all islands with a fixed
  seed. Members of one lineage face the same panel, so the pick within a lineage is a paired comparison; panels of
  different lineages are independent uniform draws. Archive members are not opponents (a lineage would meet its
  own siblings).
- Score: the surprise duel (lenia.prompts.SURPRISE), both orders; P = mean over the panel of 0.5 (x + 1 - y).
- Pick: the highest score per lineage. No score floor (fixed before seeing the scores, Felix 10-29); picks below
  0.5 are marked on the sheet, and Felix's glance drops loophole worlds.
- Complexity (lenia.prompts.COMPLEXITY, the map's entry prompt) against the same panel, for the picks only: logged,
  never used to select.
Outputs in --out: judge_cache.sqlite (the seed's shared Stage 1 cache: size later --max-usd for all arms),
rescore.csv (every member), triggers.json (the start of the Tier 1 record), triggers.png (contact sheet).

From papers: the candidates come from a novelty-search archive with an adaptive threshold (Lehman & Stanley 2011).
Ours: the pairwise surprise duel as the novelty score, re-scoring against one final pool, one trigger per lineage,
the per-lineage opponent panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mace_qd as QD
from lenia import backend
from lenia.config import REPO_ROOT, load
from lenia.judge import BudgetError, ChoiceQuery, Judge, MultiQuery, Query, load_env
from lenia.mace import TILE as ME_TILE, mutate
from lenia.prompts import (CAPTION, COMPLEXITY, NAMER, NONE, REVIEW_R, REVIEW_S, SESSION_MATCH, STEER, SURPRISE, attain_prompt,
                           steer_prompt)
from lenia.sim import Rule


def make_judge(a, out: Path) -> Judge:
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    return Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                 out / "judge_cache.sqlite", yes=True)


def duels(judge: Judge, prompt: str, pairs: list, pics: dict, desc: str) -> list:
    """pairs: [(world, opponent)] image names -> [P(world chosen)], both orders averaged; None if a call failed."""
    qs = [q for w, o in pairs for q in (Query(prompt, pics[w], pics[o]), Query(prompt, pics[o], pics[w]))]
    vs = judge.judge_many(qs, desc=desc)
    return [0.5 * (x.p_first + 1 - y.p_first) if x.ok and y.ok else None for x, y in zip(vs[::2], vs[1::2])]


def triggers(a) -> None:
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads((run / "state.json").read_text())
    sar = st.get("sarch", [])
    ref = [e["image"] for m in QD.islands_of(st) for _, e in sorted(m.items())]
    if not sar or len(ref) < a.opponents:
        sys.exit(f"{run}: {len(sar)} archive members, {len(ref)} map elites; nothing to re-score")
    if len(sar) >= QD.settings(st).sarch_max:
        print(f"WARNING: the archive is at its cap ({len(sar)}); its oldest members were dropped", file=sys.stderr)
    names = sorted({e["image"] for e in sar} | set(ref))
    missing = [n for n in names if not (run / "img" / n).exists()]
    if missing:
        sys.exit(f"{len(missing)} pictures missing, e.g. {missing[0]} (rsync the run's img folder)")
    pics = {n: np.asarray(Image.open(run / "img" / n).convert("RGB")) for n in names}
    roots = sorted({e["root"] for e in sar})
    rng = np.random.default_rng([a.seed, 1])
    panel = {r: [ref[int(j)] for j in rng.choice(len(ref), a.opponents, replace=False)] for r in roots}

    judge = make_judge(a, out)
    pairs = [(e["image"], o) for e in sar for o in panel[e["root"]]]
    ps = duels(judge, SURPRISE, pairs, pics, "re-score surprise")
    if any(p is None for p in ps):
        sys.exit(f"{sum(p is None for p in ps)} duels failed (not cached); re-run to retry them")
    k = a.opponents
    rows = []
    for i, e in enumerate(sar):
        v = ps[i * k:(i + 1) * k]
        rows.append({"image": e["image"], "root": e["root"], "it": e["it"], "island": e["island"],
                     "stored": e["surprise"], "surprise": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4),
                     "bodies": e["bodies"], "motion": e["motion"], "coverage": e["coverage"]})
    size = {r: sum(x["root"] == r for x in rows) for r in roots}
    best = {}
    for x in rows:
        if x["root"] not in best or x["surprise"] > best[x["root"]]["surprise"]:
            best[x["root"]] = x
    for x in rows:
        x["rank_in_lineage"] = 1 + sum(y["root"] == x["root"] and y["surprise"] > x["surprise"] for y in rows)
    picks = sorted(best.values(), key=lambda x: -x["surprise"])

    cs = duels(judge, COMPLEXITY, [(x["image"], o) for x in picks for o in panel[x["root"]]], pics, "trigger complexity")
    by_img = {e["image"]: e for e in sar}
    trig = []
    for n, x in enumerate(picks):
        v = [c for c in cs[n * k:(n + 1) * k] if c is not None]
        e = by_img[x["image"]]
        trig.append({**x, "id": f"s0-L{x['root']}", "lineage_members": size[x["root"]],
                     "complexity": round(float(np.mean(v)), 4) if v else None, "opponents": panel[x["root"]],
                     "parent": e.get("parent"), "rule": e["rule"]})

    with open(out / "rescore.csv", "w", newline="") as fh:
        cols = ["image", "root", "it", "island", "stored", "surprise", "sd", "rank_in_lineage", "bodies", "motion", "coverage"]
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["root"], x["rank_in_lineage"])))
    (out / "triggers.json").write_text(json.dumps(
        {"run": str(run), "run_iteration": st["it"] - 1, "prompt": SURPRISE, "complexity_prompt": COMPLEXITY,
         "opponents": a.opponents, "seed": a.seed, "judge": judge.model, "snapshots": judge.cache.snapshots(),
         "triggers": trig}, indent=1))
    (out / "img").mkdir(exist_ok=True)                 # the sheet and later steps read trigger pictures from --out
    for t in trig:
        if not (out / "img" / t["image"]).exists():
            Image.fromarray(pics[t["image"]]).save(out / "img" / t["image"])
    sheet(a)
    rho = np.corrcoef([x["stored"] for x in rows], [x["surprise"] for x in rows])[0, 1]
    print(f"\n{len(rows)} members re-scored; corr(stored, re-scored) = {rho:.2f}; judge spent ${judge.spent:.4f} "
          f"(cache total ${judge.cache.total_cost():.4f}); snapshots {judge.cache.snapshots()}")
    print("trigger      it  isl  members  surprise (sd)  stored  complexity  bodies  coverage")
    for t in trig:
        c = "-" if t["complexity"] is None else f"{t['complexity']:.2f}"
        print(f"{t['id']:<11} {t['it']:>3}  I{t['island']}  {t['lineage_members']:>7}  {t['surprise']:.2f} ({t['sd']:.2f})"
              f"    {t['stored']:.2f}   {c:>9}  {t['bodies']:>6}  {t['coverage']:.2f}")
    print(out / "triggers.png")


def verify_decoys(st: dict, trigger: dict, seed: int = 0) -> dict:
    """Gate 0d's decoys for a trigger, fixed per trigger id so every arm and the bake-off ask the same duels: a near
    decoy (a random final-map elite of the same lineage, if any) and two random elites of other lineages. The frozen
    verify rule (gate 0d) is "the true world beats both random decoys"; the near decoy is logged."""
    el = [e for m in QD.islands_of(st) for _, e in sorted(m.items())]
    rng = np.random.default_rng([seed, int(trigger["root"]), int(trigger["it"])])
    same = [e["image"] for e in el if e.get("root") == trigger["root"]]
    other = [e["image"] for e in el if e.get("root") != trigger["root"]]
    return {"near": same[int(rng.integers(len(same)))] if same else None,
            "rand": [other[int(j)] for j in rng.choice(len(other), 2, replace=False)]}


def typical(run: Path, trigger: dict, k: int = 4, seed: int = 0) -> list:
    """k random map elites from the snapshot nearest the trigger's iteration (the namer's "typical" contrast)."""
    snaps = sorted(int(p.stem[len("archive_it"):]) for p in (run / "snapshots").glob("archive_it*.json"))
    at = min(snaps, key=lambda x: abs(x - trigger["it"]))
    el = [e["image"] for m in QD.islands_of(json.loads((run / "snapshots" / f"archive_it{at:04d}.json").read_text()))
          for _, e in sorted(m.items())]
    rng = np.random.default_rng([seed, int(trigger["root"]), int(trigger["it"]), 7])
    return [el[int(j)] for j in rng.choice(len(el), k, replace=False)]


# ---------- expeditions (build step 5; v2 after arm L's first run, 10-29) ----------
#
# Every arm runs one expedition per trigger: EXP_STEPS steps of MUTANTS mutants of the current parent (lenia.mace.mutate
# + beta jitter, as the base run), each grown from the fixed start, grid noise dropped. Each child is judged against a
# FIXED reference, both orders, one combined request (gate 0e): goal progress (lenia.prompts.attain_prompt) against the
# goal's reference world, and complexity and surprise against the trigger's start. The child with the highest goal
# score is accepted if that score beats the parent's by >= MARGIN (the judge's noise band) and its complexity against the
# start is >= K_FLOOR. So progress accumulates on one scale and complexity cannot slide below where it began.
#
# Why v2 (arm L v1, runs/stage1-s0/arms/L): v1 accepted the child Jev preferred over its *parent* (steer >= 0.6) with a
# complexity floor also against the parent. Luna saw almost no goal progress (attain 0.59; the pilot: 0.53), because a
# best-of-16 duel against a moving parent mostly picks up the judge's quirks and nothing accumulates; and the parent
# floor let complexity slide one permitted step at a time (L55: 8 steps, Luna complexity 0.30 vs start). It also
# rejected 80% of goal-passing children, since moving toward these goals usually simplifies a world in Jev's eyes.
# Caveat: Jev now drives with the question Luna evaluates (attain); different models, and Felix's ratings stay independent.
#
# Arms differ only in the goal: L = a library-v3 goal matched by Jev; N = the VLM's {sees, push}; NR = N's goal plus one
# review after step REVIEW_AT (continue / refine); NS = the same plus "switch". A refine re-scores the parent under the
# new goal (reference: the start); a switch makes the step-REVIEW_AT parent the new goal's reference (complexity stays
# anchored to the start). Mutation randomness is seeded by (trigger index, step), so arms whose parents agree draw
# identical mutants; with the shared judge cache and the simulation cache (evals/), NR and NS replay N's first steps.
# From papers: steering an evolutionary search with a foundation model's goal (ASAL; the Picbreeder line). Ours: the
# fixed-reference guarded expedition, arms as treatments on identical starts with shared mutation randomness, the single
# scheduled review, and refine vs switch as separate options.

# v3 (10-30, after the heritability check, runs/stage1-s0/heritability): children of a v2 end parent kept little of its
# progress (mutation 3.0: child goal 0.46 vs parent 0.85), and the "beat the parent's score" bar stalled every
# expedition, because the parent's score was the best of 32 and so overrated (even the gentlest children sat ~0.15
# below it, flattening as mutation shrank). v3: gentle expedition mutation (scale 0.3, no ring edits: the most
# heritable setting measured; the base search keeps exploring at 3.0), and comma selection, the (1,lambda)-ES
# (Rechenberg, Schwefel): every step the best child that passes the complexity floor replaces the parent, whatever the
# parent's score. Progress is the mean goal score of each step's children (not inflated by selection); the best world
# found is kept for the record. v2 results stay in exp-v2/ (L, N), v1 in arms/ (L).

ARMS = ("L", "N", "NR", "NS")
EXP_STEPS, MUTANTS, REVIEW_AT = 10, 32, 5
MARGIN, K_FLOOR = 0.03, 0.45
MUT_SCALE, STRUCT = 0.3, 0.0             # expedition mutation (the base search uses its own mut_scale, 3.0, with ring edits)
EXP_DIR = "exp-v3"


def eval_settings(run: Path, a) -> argparse.Namespace:
    """The base run's simulation settings (so children are grown exactly like the triggers); no form axis."""
    v = QD.settings(json.loads((run / "state.json").read_text()))
    v.axes = []
    for k in ("size", "warmup", "render", "gap", "sim_batch"):        # overrides: offline dry runs only
        if getattr(a, k, None):
            setattr(v, k, getattr(a, k))
    return v


class Evals:
    """Simulation cache: a rule's picture and descriptors under the eval settings, keyed by the rule. Exact (the
    simulation is deterministic per rule), so replayed steps need no GPU."""
    def __init__(self, folder: Path, ea, keep: Path | None = None):
        import hashlib
        self.dir = folder
        self.keep = keep                     # the run folder's evals/: accepted worlds are copied there (they outlive a session)
        folder.mkdir(parents=True, exist_ok=True)
        self.ea = ea
        self.tag = json.dumps({k: getattr(ea, k) for k in ("size", "R", "warmup", "render", "gap", "start_seed")})
        self._h = hashlib.sha1

    def key(self, rule: Rule) -> str:
        return self._h((json.dumps(rule.to_dict(), sort_keys=True) + self.tag).encode()).hexdigest()[:20]

    def get_many(self, rules: list) -> list:
        keys = [self.key(r) for r in rules]
        out = [None] * len(rules)
        todo = []
        for i, k in enumerate(keys):
            if (self.dir / f"{k}.json").exists() and (self.dir / f"{k}.png").exists():
                d = json.loads((self.dir / f"{k}.json").read_text())
                out[i] = {**d, "pic": np.asarray(Image.open(self.dir / f"{k}.png").convert("RGB")), "image": f"evals/{k}.png",
                          "key": k}
            elif k not in {keys[j] for j in todo}:
                todo.append(i)
        if todo:
            ev = QD.evaluate([rules[i] for i in todo], self.ea)
            for i, e in zip(todo, ev):
                k = keys[i]
                Image.fromarray(e["pic"]).save(self.dir / f"{k}.png")
                d = {x: e[x] for x in ("bodies", "motion", "texture", "coverage")}
                (self.dir / f"{k}.json").write_text(json.dumps(d))
                out[i] = {**d, "pic": e["pic"], "image": f"evals/{k}.png", "key": k}
        for i, k in enumerate(keys):                          # duplicates within the batch
            if out[i] is None:
                j = keys.index(k)
                out[i] = out[j]
        return out

    def persist(self, e: dict) -> None:
        """Copy an accepted world's picture and descriptors into the run folder's evals/ (state.json points there)."""
        if self.keep is None or self.keep == self.dir:
            return
        import shutil
        self.keep.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "json"):
            if not (self.keep / f"{e['key']}.{ext}").exists():
                shutil.copy(self.dir / f"{e['key']}.{ext}", self.keep / f"{e['key']}.{ext}")


def goal_id(sees: str, push: str) -> str:
    import hashlib
    return "g" + hashlib.sha1(f"{sees}|{push}".encode()).hexdigest()[:12]


# Arm L's matching, lifted from scripts/goal_session.py (library, match, draw_goals) and trimmed to one goal per world.
def library(path: Path) -> dict:
    lib = json.loads(path.read_text())
    first = {}
    for g in lib:
        first.setdefault(g["sees"], g["id"])
    obs = {}
    for g in lib:
        obs.setdefault(first[g["sees"]], []).append(g)
    groups = {k: [o for o, gs in obs.items() if gs[0]["group"] == k] for k in STEER}       # no edge / leaps groups
    ctrl = [o for o, gs in obs.items() if gs[0]["group"] == "control"]
    return {"obs": obs, "groups": groups, "ctrl": ctrl}


def match(judge, L: dict, pics: list, rng, desc: str) -> list:
    """Per picture: {observation id or "none:<group>": P}; each group's menu (+ "none" + 1 control) in 2 orders."""
    cap = lambda t: t[:1].upper() + t[1:]
    qs, meta = [], []
    for i, pic in enumerate(pics):
        for k, ids in L["groups"].items():
            for _ in range(2):
                m = ids + [NONE, str(rng.choice(L["ctrl"]))]
                m = [m[j] for j in rng.permutation(len(m))]
                opts = {x: ("None of these fits." if x == NONE else cap(L["obs"][x][0]["sees"]) + ".") for x in m}
                qs.append(ChoiceQuery(pic, SESSION_MATCH.replace("{ABOUT}", STEER[k]), opts))
                meta.append((i, k))
    res = [dict() for _ in pics]
    probs = judge.choose_many(qs, desc=desc)
    if any(p is None for p in probs):
        sys.exit("some matching calls failed (not cached); re-run to retry them")
    for (i, k), p in zip(meta, probs):
        for x, v in p.items():
            x = f"{NONE}:{k}" if x == NONE else x
            res[i][x] = res[i].get(x, 0) + v / 2
    return res


def draw_goal(L: dict, probs: dict, rng, exclude: set) -> dict:
    """One observation: the top pick of each group (if it beats that group's "none"), sampled in proportion to Jev's
    match probability, skipping observations already used in this arm; one random direction."""
    top = []
    for k, ids in L["groups"].items():
        ok = [o for o in ids if o not in exclude]
        if ok:
            o = max(ok, key=lambda x: probs.get(x, 0))
            if probs.get(o, 0) > probs.get(f"{NONE}:{k}", 0) and probs.get(o, 0) > 0:
                top.append(o)
    cand = top or sorted((o for ids in L["groups"].values() for o in ids if o not in exclude), key=lambda x: -probs.get(x, 0))[:5]
    w = np.array([probs.get(o, 0) + 1e-3 for o in cand])
    o = cand[int(rng.choice(len(cand), p=w / w.sum()))]
    g = L["obs"][o][int(rng.integers(len(L["obs"][o])))]
    return {"id": g["id"], "obs": o, "group": g["group"], "sees": g["sees"], "push": g["push"], "source": "library",
            "p_match": round(float(probs.get(o, 0)), 4)}


def _judges(a, s1: Path) -> Judge:
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    return Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                 s1 / "judge_cache.sqlite", yes=True)


def _describer(a, s1: Path, cache_only: bool = False):
    from lenia.describe import Describer
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    return Describer(a.vlm, cfg["vlms"][a.vlm], s1 / "describe_cache.sqlite", max_usd=a.max_vlm_usd, cache_only=cache_only)


def _pic(s1: Path, rel: str) -> np.ndarray:
    return np.asarray(Image.open(s1 / rel).convert("RGB"))


def library_goals(a, judge, starts: list) -> list:
    """Arm L's goals: Jev matches each start to the library's observations; one goal drawn per trigger."""
    L = library(REPO_ROOT / a.library)
    probs = match(judge, L, starts, np.random.default_rng([a.seed, 101]), "L match")
    goals, used = [], set()
    for i, p in enumerate(probs):
        g = draw_goal(L, p, np.random.default_rng([a.seed, 102, i]), used)
        used.add(g["obs"])
        goals.append(g)
    return goals


def namer_goals(a, s1: Path, run: Path, T: list, starts: list, cache_only: bool = False) -> list:
    """Arm N's goals: the VLM (--vlm) names each trigger against 4 typical worlds of the run."""
    from lenia.describe import Ask
    d = _describer(a, s1, cache_only)
    goals, given = [], []
    for t, w in zip(T, starts):                  # sequential: each call sees the names given so far (as the bake-off)
        fill = {"names": "\n".join(f"- {x}" for x in given) if given else "(none yet)"}
        [r] = d.ask_many([Ask(NAMER, [w] + [_pic(run, f"img/{n}") for n in typical(run, t)], fill)], desc=t["id"])
        if not r.ok:
            sys.exit(f"naming failed at {t['id']}: {r.raw[:300]} (re-run to retry)")
        given.append(r.data["sees"])
        goals.append({"id": goal_id(r.data["sees"], r.data["push"]), "sees": r.data["sees"], "push": r.data["push"],
                      "source": f"namer:{d.model}", "cost": r.cost})
    return goals


# ---------- goal captions (v4 step 1, 11-02; project doc "Stage 1 direction - goal steering") ----------
#
# v4 shortlists each step's 48 children with CLIP (the 16 closest to a caption that pass the motion floor; Jev picks)
# only when the goal's end state can be pictured. CLIP scores one still, the bottom-right snapshot, so a caption
# (lenia.prompts.CAPTION) describes that still in 3-8 words, or is "none" for goals about motion, rhythm or change
# between snapshots (those get the motion floor plus a random 16). Each goal also gets a "start" caption of the same
# aspect of the start world, for directional scoring (the image change start -> child against the text change
# start -> end; Gal et al. 2021, StyleGAN-NADA), tested offline in step 2.
# Both arms are captioned the same way, one VLM call per goal (the goal's sees and push, and the trigger's start
# picture), so the caption writer is not a difference between arms (11-02: with hand captions for L, Gemini captioned
# 12/12 of N's goals and the hand captions 5/12 of L's, and the shortlist alone lifts Felix's win rate). The hand
# captions (CAPTIONS_L) stay as a check on the prompt's "none" threshold. The goals come from cached answers only:
# N's from the namer (cache_only), L's from Jev's library matching (a near-zero spend cap, so an uncached match stops
# the run instead of calling). Captions go through --caption-vlm (Gemini on Vertex by default; gemini-pro-or is the
# OpenRouter fallback) into their own cache, <dest>/caption_cache.sqlite; output <dest>/<arm>.json and .md.
# Ours: the caption step, the "none" route and the start caption. From papers: CLIP text targets for search (ASAL,
# Kumar et al. 2024), directional CLIP (Gal et al. 2021).

CAPTIONS_L = "data/goal_library_v3_captions.json"


def _caption(x: str) -> str:
    t = x.strip().strip('."\'').strip()
    return "none" if t.lower() in ("none", "n/a", "") else t


def load_captions(path: Path) -> dict:
    """{goal id: {"end": caption or "none", "start": ...}} from a captions file (a list of rows with id, end, start)."""
    return {r["id"]: {"end": r["end"], "start": r["start"]} for r in json.loads(Path(path).read_text())}


def captions(a) -> None:
    from lenia.describe import Ask, Describer
    s1, run = Path(a.out), Path(a.run)
    dest = Path(a.dest) if a.dest else s1 / "captions"
    dest.mkdir(parents=True, exist_ok=True)
    T = json.loads((s1 / "triggers.json").read_text())["triggers"][:a.limit or None]
    starts = [_pic(s1, f"img/{t['image']}") for t in T]
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    if a.arm == "N":
        goals = namer_goals(a, s1, run, T, starts, cache_only=True)
    else:                                    # cap 1e-9: any uncached matching call raises BudgetError, nothing is called
        judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": 1e-9},
                      s1 / "judge_cache.sqlite", yes=True)
        try:
            goals = library_goals(a, judge, starts)
        except BudgetError:                  # e.g. a partial copy of the folder: take the goals an earlier arm L recorded
            saved = [s1 / d / "L" / "state.json" for d in (EXP_DIR, "exp-v2", "arms")]
            saved = [x for x in saved if x.exists()]
            if not saved:
                sys.exit("arm L's library matching is not in the judge cache and no arm L state.json exists; "
                         "sync the folder from Drive or run the arm's setup first (it costs Jev calls)")
            byid = {tr["id"]: tr["goals"][0] for tr in json.loads(saved[0].read_text())["triggers"]}
            if any(t["id"] not in byid for t in T):
                sys.exit(f"{saved[0]} lacks some triggers")
            goals = [byid[t["id"]] for t in T]
            print(f"L's matching is not cached here; goals taken from {saved[0]} (same seed and library)", file=sys.stderr)
    d = Describer(a.caption_vlm, cfg["vlms"][a.caption_vlm], dest / "caption_cache.sqlite", max_usd=a.max_vlm_usd)
    ans = d.ask_many([Ask(CAPTION, [w], {"sees": g["sees"], "push": g["push"]}) for g, w in zip(goals, starts)],
                     desc=f"{a.arm} captions")
    rows, failed = [], []
    for t, g, r in zip(T, goals, ans):
        if not r.ok:
            failed.append(t["id"])
            continue
        end = _caption(r.data["end"])
        start = "none" if end == "none" else _caption(r.data["start"])
        flag = [] if end == "none" or 3 <= len(end.split()) <= 8 else ["end length"]
        flag += [] if end == "none" or start != "none" else ["no start caption"]
        rows.append({"trigger": t["id"], "id": g["id"], "sees": g["sees"], "push": g["push"], "end": end, "start": start,
                     "by": f"{a.caption_vlm}:{d.model}", "version": r.version, "cost": round(r.cost, 6), "flags": flag})
    if failed:
        sys.exit(f"{len(failed)} caption calls failed ({', '.join(failed)}); re-run to retry them (answers are cached)")
    (dest / f"{a.arm}.json").write_text(json.dumps(rows, indent=1))
    L = [f"# Arm {a.arm} goal captions, seed {a.seed} ({a.caption_vlm}, {CAPTION})\n",
         f"{sum(r['end'] != 'none' for r in rows)} of {len(rows)} goals captioned; spend ${sum(r['cost'] for r in rows):.4f}.\n",
         "| trigger | push | end caption | start caption | flags |", "|---|---|---|---|---|"]
    L += [f"| {r['trigger']} | {r['push']} | {r['end']} | {r['start']} | {', '.join(r['flags'])} |" for r in rows]
    (dest / f"{a.arm}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {dest / (a.arm + '.json')}", file=sys.stderr)


# ---------- caption check (v4 step 2, 11-03): which CLIP score predicts Jev's picks? ----------
#
# v4 shortlists children by CLIP before Jev chooses. Two ways to score a child against a goal caption:
# - target: cos(child, text(end)), what round 4's shortlist used (ASAL, Kumar et al. 2024);
# - change: cos(child - start, text(end) - text(start)), directional CLIP (Gal et al. 2021, StyleGAN-NADA).
# Offline and free of API calls: expedition v3's arm L children are regrown (their pictures were not kept: the
# simulation cache was on Colab's local disk), and each is scored both ways against the goal's captions; Jev's goal
# score for every child is already in duels.csv. Per (trigger, step): Spearman rho between each CLIP score and Jev's,
# over that step's judged children, and whether the child Jev picked (highest goal score passing the floor) lands in
# CLIP's top third (v4 keeps 16 of 48; chance = 1/3). Hand captions (CAPTIONS_L) are scored too, as a check on the
# captioner. Only captioned goals enter. Caveat: v3 mutated at 0.3, so siblings are close and every rho is range-
# restricted; the comparison between scores is fair, the absolute values understate what v4's 2.0 -> 0.8 would show.
# Reproduction is checked before anything is scored: every accepted child's simulation-cache key must equal the one in
# trace.jsonl (mutation reproduces bit for bit; x86 only, as the run), and the regrown pictures of accepted children
# are compared with the copies kept in <out>/evals (the simulation reproduces on this GPU).
# Ours: the check and its design. From papers: CLIP target scoring (ASAL), directional CLIP (Gal et al. 2021).

V3_SCALE, V3_STRUCT = 0.3, 0.0           # expedition v3's mutation, pinned here (v4 changes MUT_SCALE / STRUCT)


def _spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def capcheck(a) -> None:
    from lenia.embed import ClipScorer
    s1, run = Path(a.out), Path(a.run)
    adir = s1 / "exp-v3" / "L"
    dest = s1 / "capcheck"
    dest.mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    ea = eval_settings(run, a)
    st = json.loads((adir / "state.json").read_text())
    caps = {r["trigger"]: r for r in json.loads((s1 / "captions" / "L.json").read_text())}
    hand = load_captions(REPO_ROOT / CAPTIONS_L)
    trace = [json.loads(x) for x in (adir / "trace.jsonl").read_text().splitlines() if x.strip()]
    duels = list(csv.DictReader(open(adir / "duels.csv")))
    T = [tr for tr in st["triggers"] if caps.get(tr["id"], {}).get("end", "none") != "none"][:a.limit or None]
    steps = min(st["steps"], a.steps or st["steps"])
    M = st["mutants"]
    evals = Evals(Path(a.evals) if a.evals else Path("/tmp/capcheck-evals"), ea)
    clip = ClipScorer(a.clip)
    tile = lambda pic: np.asarray(pic)[-ME_TILE:, -ME_TILE:]
    rows, repro = [], {"keys": 0, "keys_ok": 0, "pics": 0, "pic_diff": []}
    for tr in T:
        c = caps[tr["id"]]
        texts = {"gemini": (c["end"], c["start"])}
        h = hand.get(tr["goals"][0]["id"])
        if h and h["end"] != "none":
            texts["hand"] = (h["end"], h["start"])
        start = _pic(s1, tr["start"])
        e0 = clip.images([tile(start)])[0]
        acc = {t["step"]: t for t in trace if t["trigger"] == tr["id"]}
        parent = tr["start_rule"]
        for step in range(1, steps + 1):
            rng = np.random.default_rng([st["seed"], tr["index"], step])
            kids = mutants_of(Rule.from_dict(parent), rng, M, ea, V3_SCALE, V3_STRUCT)
            ev = evals.get_many(kids)
            jev = {int(d["mutant"]): d for d in duels if d["trigger"] == tr["id"] and int(d["step"]) == step and d["goal"]}
            if step in acc:                    # reproduction: the accepted child is one of these, by key and by picture
                repro["keys"] += 1
                key = acc[step]["image"].split("/")[-1][:-4]
                hit = [j for j, v in enumerate(ev) if v["key"] == key]
                repro["keys_ok"] += bool(hit)
                kept = s1 / acc[step]["image"]
                if hit and kept.exists():
                    repro["pics"] += 1
                    repro["pic_diff"].append(float(np.abs(np.asarray(Image.open(kept).convert("RGB"), np.int16)
                                                          - ev[hit[0]]["pic"].astype(np.int16)).mean()))
            js = sorted(jev)
            if len(js) >= 4:
                E = clip.images([tile(ev[j]["pic"]) for j in js])
                G = [float(jev[j]["goal"]) for j in js]
                ok = [int(jev[j]["accepted"]) for j in js]
                pick = max((g, j) for g, j, o in zip(G, js, ok) if o)[1] if any(ok) else None
                r = {"trigger": tr["id"], "step": step, "n": len(js), "jev_sd": round(float(np.std(G)), 4)}
                for who, (end, beg) in texts.items():
                    te, ts = clip.text(end), clip.text(beg)
                    d = te - ts
                    d = d / max(np.linalg.norm(d), 1e-9)
                    dimg = E - e0
                    dimg = dimg / np.maximum(np.linalg.norm(dimg, axis=1, keepdims=True), 1e-9)
                    for kind, sc in (("target", E @ te), ("change", dimg @ d)):
                        r[f"{who}_{kind}_rho"] = _spearman(sc, G)
                        if pick is not None:
                            rank = int((sc > sc[js.index(pick)]).sum())
                            r[f"{who}_{kind}_top3"] = int(rank < max(1, round(len(js) / 3)))
                rows.append(r)
            if step in acc:
                parent = acc[step]["rule"]
        print(f"{tr['id']}: {steps} steps scored", file=sys.stderr)
    if repro["keys"] and repro["keys_ok"] < repro["keys"] and not a.dry:
        sys.exit(f"mutation did not reproduce: {repro['keys_ok']} of {repro['keys']} accepted children found by key "
                 "(run on Colab, x86, as expedition v3 did). Nothing was scored.")
    (dest / "results.json").write_text(json.dumps({"rows": rows, "repro": repro, "clip": a.clip}, indent=1))
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    L = ["# Caption check: which CLIP score predicts Jev's picks? (expedition v3, arm L, captioned goals)\n",
         f"Reproduction: {repro['keys_ok']} of {repro['keys']} accepted children found by key; regrown vs kept "
         f"pictures, mean |pixel difference| {mean(repro['pic_diff']):.2f} of 255 over {repro['pics']}.\n",
         f"{len(rows)} trigger-steps; Spearman rho with Jev's goal score among each step's children, and how often Jev's "
         "pick is in CLIP's top third (chance 0.33).\n",
         "| captions | score | mean rho | rho > 0 | pick in top third |", "|---|---|---|---|---|"]
    for who in ("gemini", "hand"):
        for kind in ("target", "change"):
            rh = [r[f"{who}_{kind}_rho"] for r in rows if r.get(f"{who}_{kind}_rho") is not None]
            t3 = [r[f"{who}_{kind}_top3"] for r in rows if f"{who}_{kind}_top3" in r]
            if rh:
                L.append(f"| {who} | {kind} | {mean(rh):+.3f} | {sum(x > 0 for x in rh)}/{len(rh)} | {mean(t3):.2f} ({len(t3)}) |")
    L += ["\nPer trigger (gemini captions, mean rho target / change):"]
    for tid in dict.fromkeys(r["trigger"] for r in rows):
        rr = [r for r in rows if r["trigger"] == tid]
        f = lambda k: mean([r[k] for r in rr if r.get(k) is not None])
        L.append(f"- {tid} ({caps[tid]['end']}): {f('gemini_target_rho'):+.3f} / {f('gemini_change_rho'):+.3f}")
    (dest / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- expedition v4 (11-03; project doc "Stage 1 direction - goal steering") ----------
#
# The step Felix rated best on the CLIP bench (jev-v4: 6 of 7 blind pairs against CLIP-driven search over 40 steps;
# the CLIP shortlist 9-3 against plain Jev over rounds 3-4), ported to the trigger expeditions. Every step, per trigger:
# 48 mutants of the parent (mutation 2.0 -> 0.8, geometric over 20 steps, no ring edits), each grown from the fixed
# noise start; grid-filling texture dropped. Then a shortlist of 16:
# - "clip" route (the goal has a picture caption): the 16 closest to the caption by CLIP (target score, the bench's
#   choice; directional scoring is untested, stage1.py capcheck can test it later), among children with motion >= 1.0;
# - "floor" route (caption "none"): 16 at random among children with motion >= 1.0 (bench jev-floor).
# Jev scores the 16 against the trigger's start in one combined request per order (goal progress = attain_prompt,
# unchanged from the bench; complexity; surprise), and the best goal score whose complexity is >= K_FLOOR replaces the
# parent (comma selection; if none passes, the parent stays).
# Arms (goals and captions from <out>/captions/<arm>.json; N's goals from the namer's cached answers):
#   L  = library goals;  N = VLM-written goals;
#   NF = N's goals forced onto the floor route, only on the triggers whose L goal has no caption, so N vs L can be
#        compared there with no CLIP help on either side (11-02: coverage differs, N 12/12 captioned, L 6/12);
#   C  = N's goals steered by CLIP alone (ASAL-style baseline, Kumar et al. 2024): the child closest to the caption (or
#        the push text when the caption is "none") among texture-passing children; no Jev, no motion floor.
# Mutation randomness is seeded by (trigger, step), so all arms draw the same step-1 children and share their
# simulations. Kept deliberately the same as the bench (untested changes wait): the attain prompt's wording, the
# motion floor of 1.0, target CLIP scoring. Added from the expeditions: the complexity floor against the start.
# Logged per step: CLIP's mean over all valid children (unselected, so not inflated), mean motion, Jev's mean over the
# judged 16 (shortlisted, so biased upward under the clip route), and the pick.
# Ours: the trigger expedition, the routes, arms NF and C. From papers: comma selection ((1,lambda)-ES; Rechenberg,
# Schwefel), CLIP text targets for search (ASAL), foundation-model goals steering evolution (ASAL; the Picbreeder line).

V4_DIR, V4_ARMS = "exp-v4", ("L", "N", "NF", "C")
V4_STEPS, V4_CHILDREN, V4_JUDGED = 20, 48, 16
V4_SCHED, V4_MOTION = (2.0, 0.8), 1.0


def v4_scale(step: int, steps: int) -> float:
    s0, s1 = V4_SCHED
    return float(s0 * (s1 / s0) ** ((step - 1) / max(1, steps - 1)))


def v4_setup(a, s1: Path, run: Path, judge) -> dict:
    T = json.loads((s1 / "triggers.json").read_text())["triggers"]
    starts = [_pic(s1, f"img/{t['image']}") for t in T]
    src = "L" if a.arm == "L" else "N"
    goals = library_goals(a, judge, starts) if src == "L" else namer_goals(a, s1, run, T, starts, cache_only=True)
    capf = s1 / "captions" / f"{src}.json"
    if not capf.exists():
        sys.exit(f"{capf} is missing: run `stage1.py captions --arm {src}` first")
    caps = {r["trigger"]: r for r in json.loads(capf.read_text())}
    keep = list(range(len(T)))
    if a.arm == "NF":                        # only where L's goal has no picture caption
        lcaps = {r["trigger"]: r for r in json.loads((s1 / "captions" / "L.json").read_text())}
        keep = [i for i, t in enumerate(T) if lcaps[t["id"]]["end"] == "none"]
    keep = keep[:a.limit or None]
    trs = []
    for i in keep:
        t, g = T[i], goals[i]
        c = caps.get(t["id"])
        if c is None or c["id"] != g["id"]:
            sys.exit(f"{capf}: no caption for {t['id']}'s goal {g['id']} (re-run the captions command)")
        route = "clip" if a.arm == "C" else "floor" if a.arm == "NF" else ("clip" if c["end"] != "none" else "floor")
        target = c["end"] if c["end"] != "none" else g["push"]
        trs.append({"id": t["id"], "index": i, "start": f"img/{t['image']}", "start_rule": t["rule"],
                    "start_motion": t.get("motion"), "goal": g, "caption": {"end": c["end"], "start": c["start"]},
                    "route": route, "target": target, "parent_rule": t["rule"], "parent": f"img/{t['image']}",
                    "g": 0.5, "k": 0.5, "s": 0.5, "history": [], "stats": []})
    return {"arm": a.arm, "version": V4_DIR, "step": 0, "seed": a.seed, "steps": a.steps, "children": V4_CHILDREN,
            "judged": V4_JUDGED, "sched": list(V4_SCHED), "motion_floor": a.motion_floor, "k_floor": K_FLOOR, "triggers": trs}


def expedition_v4(a) -> None:
    from lenia.embed import ClipScorer
    s1, run = Path(a.out), Path(a.run)
    adir = s1 / a.tag / a.arm
    adir.mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    ea = eval_settings(run, a)
    judge = None if a.arm == "C" else _judges(a, s1)
    if (adir / "state.json").exists():
        st = json.loads((adir / "state.json").read_text())
        print(f"arm {a.arm}: resuming after step {st['step']}", file=sys.stderr)
    else:
        st = v4_setup(a, s1, run, judge or _judges(a, s1))
        QD.save_state(adir, st)
        for tr in st["triggers"]:
            print(f"{tr['id']} [{tr['route']}] {tr['goal']['push']}  ->  CLIP target: {tr['target']}", file=sys.stderr)
    evals = Evals(Path(a.evals) if a.evals else s1 / "evals", ea, keep=s1 / "evals")
    if st["step"] == 0:
        t0 = json.loads((s1 / "triggers.json").read_text())["triggers"][0]    # the same trigger for every arm
        same_platform(s1, evals, st, ea, tag=f"{V4_DIR} sched={V4_SCHED[0]}-{V4_SCHED[1]} struct=0", scale=V4_SCHED[0],
                      struct=0.0, tr={"start_rule": t0["rule"], "index": 0})
    clip = ClipScorer(a.clip)
    tile = lambda pic: np.asarray(pic)[-ME_TILE:, -ME_TILE:]
    starts = {tr["index"]: _pic(s1, tr["start"]) for tr in st["triggers"]}
    stop = min(st["steps"], a.stop_after or st["steps"])
    while st["step"] < stop:
        step, t0 = st["step"] + 1, time.time()
        scale = v4_scale(step, st["steps"])
        rules, owner = [], []
        for tr in st["triggers"]:
            rng = np.random.default_rng([st["seed"], tr["index"], step])      # the same draws in every arm
            for m in mutants_of(Rule.from_dict(tr["parent_rule"]), rng, st["children"], ea, scale, 0.0):
                rules.append(m)
                owner.append(tr["index"])
        ev = evals.get_many(rules)
        emb = clip.images([tile(v["pic"]) for v in ev])
        byi = {tr["index"]: tr for tr in st["triggers"]}
        short, csc, rows = {}, {}, []
        for i, tr in byi.items():
            ks = [j for j, o in enumerate(owner) if o == i]
            cs = emb[ks] @ clip.text(tr["target"])
            csc.update(zip(ks, cs.tolist()))
            ok = [ea.max_texture <= 0 or ev[j]["texture"] <= ea.max_texture for j in ks]
            if a.arm == "C":
                short[i] = [ks[n] for n in np.argsort(-cs) if ok[n]][:1]
                continue
            fl = [o and ev[j]["motion"] >= st["motion_floor"] for o, j in zip(ok, ks)]
            if tr["route"] == "clip":
                order = [ks[n] for n in np.argsort(-cs) if fl[n]]
            else:
                rr = np.random.default_rng([st["seed"], tr["index"], step, 7]).permutation(len(ks))
                order = [ks[n] for n in rr if fl[n]]
            short[i] = order[:st["judged"]]
        jv = {}
        if a.arm != "C":
            mq, meta = [], []
            for i, js in short.items():
                pids = (attain_prompt(byi[i]["goal"]), COMPLEXITY, SURPRISE)
                for j in js:
                    mq += [MultiQuery(pids, ev[j]["pic"], starts[i]), MultiQuery(pids, starts[i], ev[j]["pic"])]
                    meta.append(j)
            vs = judge.judge_multi(mq, desc=f"v4 {a.arm} step {step}")
            if any(not all(v.ok for v in vv) for vv in vs):
                sys.exit("some judge requests failed (not cached); re-run to retry them (the step restarts)")
            both = lambda x, y, q: 0.5 * (x[q].p_first + 1 - y[q].p_first)
            for n, j in enumerate(meta):
                x, y = vs[2 * n], vs[2 * n + 1]
                jv[j] = (both(x, y, 0), both(x, y, 1), both(x, y, 2))
        for i, tr in byi.items():
            ks = [j for j, o in enumerate(owner) if o == i]
            if a.arm == "C":
                pick = short[i][0] if short[i] else None
            else:
                cand = [(jv[j][0], jv[j][1], j) for j in short[i] if jv[j][1] >= st["k_floor"]]
                pick = max(cand)[2] if cand else None
            valid = [j for j in ks if ea.max_texture <= 0 or ev[j]["texture"] <= ea.max_texture]
            rec = {"step": step, "scale": round(scale, 4), "valid": len(valid), "floor": sum(ev[j]["motion"] >= st["motion_floor"]
                   for j in valid), "judged": len(short[i]) if a.arm != "C" else 0,
                   "clip_mean": round(float(np.mean([csc[j] for j in valid])), 4) if valid else None,
                   "motion_mean": round(float(np.mean([ev[j]["motion"] for j in valid])), 4) if valid else None,
                   "jev_mean_judged": round(float(np.mean([jv[j][0] for j in short[i]])), 4) if jv and short[i] else None,
                   "moved": pick is not None}
            tr["stats"].append(rec)
            for j in ks:
                g, k, s_ = jv.get(j, (None, None, None))
                rows.append([step, tr["id"], ks.index(j), round(ev[j]["texture"], 4), round(ev[j]["motion"], 4),
                             round(csc[j], 4), int(j in short[i]), *(None if v is None else round(v, 4) for v in (g, k, s_)),
                             int(j == pick)])
            if pick is not None:
                evals.persist(ev[pick])
                tr["parent_rule"], tr["parent"] = rules[pick].to_dict(), ev[pick]["image"]
                if a.arm != "C":
                    tr["g"], tr["k"], tr["s"] = (round(v, 4) for v in jv[pick])
                h = {"step": step, "image": ev[pick]["image"], "goal_score": tr["g"] if a.arm != "C" else None,
                     "complexity": tr["k"] if a.arm != "C" else None, "surprise": tr["s"] if a.arm != "C" else None,
                     "clip": round(csc[pick], 4), "motion": round(ev[pick]["motion"], 4), "bodies": ev[pick]["bodies"]}
                tr["history"].append(h)
                tr["_trace"] = {"arm": a.arm, "trigger": tr["id"], **h, "rule": rules[pick].to_dict()}
        st["step"] = step
        QD.save_state(adir, st)                # state first, then the logs (a crash re-runs the step from its cache)
        new = not (adir / "duels.csv").exists()
        with open(adir / "duels.csv", "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["step", "trigger", "child", "texture", "motion", "clip", "shortlisted", "goal", "complexity",
                            "surprise", "picked"])
            w.writerows(rows)
        with open(adir / "trace.jsonl", "a") as fh:
            for tr in st["triggers"]:
                if tr.get("_trace", {}).get("step") == step:
                    fh.write(json.dumps(tr["_trace"]) + "\n")
        print(f"v4 {a.arm} step {step}/{st['steps']} (scale {scale:.2f}): {len(rules)} children in {time.time() - t0:.0f}s; "
              f"moved {sum(tr['stats'][-1]['moved'] for tr in st['triggers'])}/{len(st['triggers'])}; CLIP mean "
              f"{np.nanmean([tr['stats'][-1]['clip_mean'] or np.nan for tr in st['triggers']]):.3f}", file=sys.stderr)
    for tr in st["triggers"]:
        tr.pop("_trace", None)
    QD.save_state(adir, st)
    if st["step"] >= stop:
        v4_sheet(s1, st, adir / "sheet.png")
        v4_summary(st, adir / "summary.md")
        print(f"arm {a.arm}: {st['step']} of {st['steps']} steps; {adir / 'sheet.png'}, {adir / 'summary.md'}", file=sys.stderr)


def v4_sheet(s1: Path, st: dict, path: Path, px: int = 260) -> None:
    """One row per trigger: start | the parent halfway | the end, with the goal, route and CLIP target."""
    head, cols = 42, 3
    im = Image.new("RGB", (cols * (px + 8) + 8, len(st["triggers"]) * (px + head + 10) + 10), (250, 250, 250))
    d = ImageDraw.Draw(im)
    f1, f2 = _font(14), _font(11)
    for n, tr in enumerate(st["triggers"]):
        y = 10 + n * (px + head + 10)
        hs = tr["history"]
        mid = next((h["image"] for h in reversed(hs) if h["step"] <= st["step"] // 2), tr["start"])
        d.text((8, y), f"{tr['id']} [{tr['route']}]  moved {len(hs)}/{st['step']} steps  "
               + (f"goal {tr['g']:.2f}  complexity {tr['k']:.2f}" if st["arm"] != "C" else "CLIP only"),
               fill="black", font=f1)
        d.text((8, y + 18), (tr["goal"]["push"][:70] + " | " + tr["target"])[:120], fill=(70, 70, 70), font=f2)
        for c, rel in enumerate((tr["start"], mid, tr["parent"])):
            im.paste(Image.open(s1 / rel).convert("RGB").resize((px, px), Image.BILINEAR), (8 + c * (px + 8), y + head))
    im.save(path)


def v4_summary(st: dict, path: Path) -> None:
    L = [f"# Expedition v4, arm {st['arm']}, seed {st['seed']}: {st['step']} of {st['steps']} steps\n",
         "Jev's goal and complexity scores are the end parent against the start (0.5 = no change). CLIP mean = all "
         "valid children's mean CLIP score against the target (unselected); Jev mean = the judged shortlist's (biased "
         "upward under the clip route).\n",
         "| trigger | route | moved | goal | complexity | CLIP mean, step 1 -> last | motion, step 1 -> last | Jev mean judged, 1 -> last |",
         "|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.2f}"
    for tr in st["triggers"]:
        a, b = tr["stats"][0], tr["stats"][-1]
        L.append(f"| {tr['id']} | {tr['route']} | {len(tr['history'])}/{st['step']} | "
                 + (f"{tr['g']:.2f} | {tr['k']:.2f}" if st["arm"] != "C" else "- | -")
                 + f" | {f(a['clip_mean'])} -> {f(b['clip_mean'])} | {f(a['motion_mean'])} -> {f(b['motion_mean'])} | "
                   f"{f(a['jev_mean_judged'])} -> {f(b['jev_mean_judged'])} |")
    path.write_text("\n".join(L) + "\n")


# ---------- v4's blind rating session (11-03) ----------
#
# Felix rates on the Mac; worlds are shown as films (the snapshot window, every --every steps) above their four stills
# (the stored picture Jev saw; regrown for the controls, whose pictures were not kept). Two questions:
# - goal ("which shows the wanted change more clearly?", the goal's sees and push shown): only between worlds that
#   pursued the same goal. Groups: end vs start for N and L (does steering show?), control vs start (a random
#   unselected step-1 child: what an unsteered change scores; 6 triggers with N's goal, 6 with L's), N vs C (does Jev
#   add to CLIP alone?), N vs NF (does the CLIP shortlist help on the same goals?), end vs step 10 (do steps 11-20 help?);
# - interest ("which is more interesting as artificial life?", no goal shown): between arms whose goals differ, the
#   Stage 1 question. N vs L on L's 6 captioned triggers, NF vs L on L's 6 uncaptioned ones (no CLIP on either side).
# Sides and order random; images named by hash; arm names stay in tasks.json and never reach the page.
# Replay check: each regrown world's stills are compared with the stored picture. A world whose replay differs
# (mean |pixel difference| > REPLAY_TOL; 11-03: 7 of the 12 trigger starts, stored by the base run on other hardware)
# shows its regrown film and regrown stills, so both sides of every pair have a film and the stills match it; the
# rest show their stored stills (what Jev saw). Listed in replay.json either way.

REPLAY_TOL = 6.0


def v4_films(rules: list, ea, every: int) -> list:
    """Grow rules from the fixed start as QD.evaluate does; per rule: (frames of the snapshot window, picture)."""
    from lenia import mace as ME, seeds
    shape = (ea.size, ea.size)
    A0 = seeds.smooth_noise(ea.R, shape, 3, seed=ea.start_seed, size_r=3.0)
    steps = ea.warmup + ea.render
    gap = min(ea.gap, steps // (ME.FRAMES - 1))
    keep = {steps - i * gap for i in range(ME.FRAMES)}
    out = []
    for k in range(0, len(rules), ea.sim_batch):
        rr = rules[k:k + ea.sim_batch]
        b = backend.make_batch(rr, shape)
        X = backend.to_device(np.stack([A0] * len(rr)))
        films, snaps = [[] for _ in rr], [[] for _ in rr]
        for t in range(1, steps + 1):
            X = b.step(X)
            film_t = t >= ea.warmup and (t - ea.warmup) % every == 0
            if film_t or t in keep:
                x = backend.to_numpy(X).copy()
                for j in range(len(rr)):
                    if film_t:
                        films[j].append(x[j])
                    if t in keep:
                        snaps[j].append(x[j])
        for f, sn in zip(films, snaps):
            out.append((f, ME.picture(np.stack(sn)), np.stack(sn)))
        print(f"films: {min(k + ea.sim_batch, len(rules))}/{len(rules)} worlds", file=sys.stderr)
    return out


def _gif(frames: list, snaps: np.ndarray, path: Path, px: int, fps: int) -> None:
    from lenia import render
    scale = max(float(np.percentile(snaps, 99.5)), 1e-6)       # the stills' brightness, so film and stills match
    ims = [Image.fromarray(render.colorize(np.clip(f / scale, 0, 1))).resize((px, px), Image.BILINEAR)
           .quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)


def v4rate(a) -> None:
    import hashlib
    s1 = Path(a.out)
    rdir = s1 / "rate-v4"
    if a.step == "rate":
        from lenia.rating import rate_pairs
        return rate_pairs(rdir, port=a.port)
    if a.step == "report":
        return v4rate_report(rdir)
    if a.step == "refilm":
        return v4rate_refilm(a, s1, rdir)
    if (rdir / "tasks.json").exists() and not a.force:
        sys.exit(f"{rdir / 'tasks.json'} exists; the session is built (--force rebuilds it: same seed, same pairs, "
                 "but only before any rating)")
    if a.force and (rdir / "ratings.csv").exists():
        sys.exit(f"{rdir / 'ratings.csv'} exists: ratings were made against this build; not rebuilding")
    run = Path(a.run)
    backend.configure(a.backend)
    ea = eval_settings(run, a)
    T = json.loads((s1 / "triggers.json").read_text())["triggers"]
    S, TR = {}, {}
    for arm in V4_ARMS:
        f = s1 / a.tag / arm / "state.json"
        if f.exists():
            S[arm] = {tr["id"]: tr for tr in json.loads(f.read_text())["triggers"]}
            S[arm]["_st"] = json.loads(f.read_text())
            TR[arm] = [json.loads(x) for x in (s1 / a.tag / arm / "trace.jsonl").read_text().splitlines() if x.strip()]
        else:
            print(f"arm {arm}: not run, its pairs are skipped", file=sys.stderr)
    rng = np.random.default_rng([a.seed, 404])
    W = {}                                   # world name -> {"rule", "still" (stored picture, relative) or None}

    def world(arm, tid, which):
        name = f"{arm}:{tid}:{which}"
        if name not in W:
            tr = S[arm][tid]
            if which == "end":
                W[name] = {"rule": tr["parent_rule"], "still": tr["parent"]}
            else:                            # the parent halfway, as on the sheet
                mid = S[arm]["_st"]["step"] // 2
                h = next((x for x in reversed(TR[arm]) if x["trigger"] == tid and x["step"] <= mid), None)
                W[name] = ({"rule": h["rule"], "still": h["image"]} if h else {"rule": tr["start_rule"], "still": tr["start"]})
        return name

    def start(t):
        W.setdefault(f"start:{t['id']}", {"rule": t["rule"], "still": f"img/{t['image']}"})
        return f"start:{t['id']}"

    def control(t, i):                       # a random unselected step-1 child (the same children in every arm)
        name = f"ctrl:{t['id']}"
        if name not in W:
            arm = "N" if "N" in S else next(iter(S))
            st = S[arm]["_st"]
            rows = [r for r in csv.DictReader(open(s1 / a.tag / arm / "duels.csv"))
                    if r["trigger"] == t["id"] and r["step"] == "1" and r["picked"] == "0"
                    and (ea.max_texture <= 0 or float(r["texture"]) <= ea.max_texture)]
            j = int(rows[int(rng.integers(len(rows)))]["child"])
            kids = mutants_of(Rule.from_dict(t["rule"]), np.random.default_rng([st["seed"], i, 1]), st["children"], ea,
                              v4_scale(1, st["steps"]), 0.0)
            W[name] = {"rule": kids[j].to_dict(), "still": None, "child": j}
        return name

    goal = lambda arm, tid: S[arm][tid]["goal"]
    tasks = []

    def add(group, q, tid, x, y, g=None):
        if W[x]["rule"] == W[y]["rule"]:
            return                           # the same world: nothing to rate
        tasks.append({"group": group, "question": q, "trigger": tid, "a": x, "b": y,
                      "sees": g["sees"] if g else None, "push": g["push"] if g else None})
    have = lambda *arms, tid=None: all(x in S and (tid is None or tid in S[x]) for x in arms)
    cap = {tid: S["L"][tid]["route"] for tid in S["L"] if tid != "_st"} if "L" in S else {}
    for i, t in enumerate(T):
        tid = t["id"]
        for arm in ("N", "L"):
            if have(arm, tid=tid):
                add(f"end-vs-start:{arm}", "goal", tid, world(arm, tid, "end"), start(t), goal(arm, tid))
        if have("N", "C", tid=tid):
            add("N-vs-C", "goal", tid, world("N", tid, "end"), world("C", tid, "end"), goal("N", tid))
        if have("N", "NF", tid=tid):
            add("N-vs-NF", "goal", tid, world("N", tid, "end"), world("NF", tid, "end"), goal("N", tid))
        if have("N", "L", tid=tid) and cap.get(tid) == "clip":
            add("interest:N-vs-L", "interest", tid, world("N", tid, "end"), world("L", tid, "end"))
        elif have("NF", "L", tid=tid):
            add("interest:NF-vs-L", "interest", tid, world("NF", tid, "end"), world("L", tid, "end"))
    ids = [t["id"] for t in T]
    order = [ids[k] for k in rng.permutation(len(ids))]
    for n, tid in enumerate(order):          # controls: half with N's goal, half with L's
        arm = ("N" if n % 2 == 0 else "L") if have("N", "L") else next(iter(S))
        t = T[ids.index(tid)]
        if have(arm, tid=tid):
            add(f"control:{arm}", "goal", tid, control(t, ids.index(tid)), start(t), goal(arm, tid))
    for arm, tids in (("N", order[:6]), ("L", order[6:12])):  # end vs halfway: 6 triggers per arm, disjoint
        for tid in tids:
            if have(arm, tid=tid):
                add(f"end-vs-mid:{arm}", "goal", tid, world(arm, tid, "end"), world(arm, tid, "mid"), goal(arm, tid))
    names = sorted(W)
    print(f"{len(tasks)} pairs over {len(names)} worlds; growing their films", file=sys.stderr)
    res = v4_films([Rule.from_dict(W[n]["rule"]) for n in names], ea, a.every)
    (rdir / "images").mkdir(parents=True, exist_ok=True)
    salt = rng.integers(1 << 30)
    replay = {}
    for n, (frames, pic, snaps) in zip(names, res):
        h = hashlib.sha1(f"{n}|{salt}".encode()).hexdigest()[:12]
        W[n]["file"] = h
        stored = W[n]["still"]
        d = None
        if stored and (s1 / stored).exists():
            ref = np.asarray(Image.open(s1 / stored).convert("RGB"), np.int16)
            d = float(np.abs(ref - pic.astype(np.int16)).mean()) if ref.shape == pic.shape else float("inf")
            replay[n] = round(d, 2)
        if d is not None and d <= REPLAY_TOL:
            Image.open(s1 / stored).convert("RGB").save(rdir / "images" / f"{h}.png")
        else:                                # no stored picture, or the replay diverged: the regrown stills
            Image.fromarray(pic).save(rdir / "images" / f"{h}.png")
        _gif(frames, snaps, rdir / "images" / f"{h}.gif", a.px, a.fps)
    out = []
    for k in rng.permutation(len(tasks)):
        t = tasks[k]
        l, r = (t["a"], t["b"]) if rng.random() < 0.5 else (t["b"], t["a"])
        out.append({"id": len(out), **t, "left": W[l]["file"], "right": W[r]["file"], "left_world": l, "right_world": r})
    (rdir / "tasks.json").write_text(json.dumps(out, indent=1))
    bad = {n: d for n, d in replay.items() if d > REPLAY_TOL}
    (rdir / "replay.json").write_text(json.dumps({"tol": REPLAY_TOL, "diff": replay, "diverged": bad,
                                                   "controls": {n: W[n]["child"] for n in W if n.startswith("ctrl:")}}, indent=1))
    print(f"wrote {rdir}: {len(out)} pairs ({dict(Counter(t['group'] for t in out))}); replay: {len(replay)} stored "
          f"pictures compared, median diff {np.median(list(replay.values())) if replay else float('nan'):.2f}, "
          f"{len(bad)} diverged (shown regrown)", file=sys.stderr)


def v4rate_refilm(a, s1: Path, rdir: Path) -> None:
    """Give every world the replay check flagged a film and matching stills, regrown here (11-03: the 7 diverged trigger
    starts of a build that showed them as stills only). Writes over their files in place (same hashed names), so the
    pairs are unchanged; refused once ratings exist. Starts only: every other world replayed exactly."""
    if (rdir / "ratings.csv").exists():
        sys.exit(f"{rdir / 'ratings.csv'} exists: ratings were made against these files; not changing them")
    backend.configure(a.backend)
    ea = eval_settings(Path(a.run), a)
    rp = json.loads((rdir / "replay.json").read_text())
    files = {}
    for t in json.loads((rdir / "tasks.json").read_text()):
        files[t["left_world"]], files[t["right_world"]] = t["left"], t["right"]
    rules = {f"start:{t['id']}": t["rule"] for t in json.loads((s1 / "triggers.json").read_text())["triggers"]}
    todo = sorted(n for n in rp["diverged"] if n in files)
    if any(n not in rules for n in todo):
        sys.exit(f"only starts can be refilmed here: {[n for n in todo if n not in rules]}")
    for n, (frames, pic, snaps) in zip(todo, v4_films([Rule.from_dict(rules[n]) for n in todo], ea, a.every)):
        Image.fromarray(pic).save(rdir / "images" / f"{files[n]}.png")
        _gif(frames, snaps, rdir / "images" / f"{files[n]}.gif", a.px, a.fps)
    rp["refilmed"] = {"worlds": todo, "backend": a.backend, "time": time.strftime("%Y-%m-%d %H:%M")}
    (rdir / "replay.json").write_text(json.dumps(rp, indent=1))
    print(f"refilmed {len(todo)} worlds: {', '.join(todo)}", file=sys.stderr)


def _binom_greater(k: int, n: int) -> float:
    """One-sided sign test: P(X >= k), X ~ Binomial(n, 1/2)."""
    from math import comb
    return float(sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else float("nan")


def v4rate_report(rdir: Path) -> None:
    from lenia.stats import fisher_greater
    tasks = {t["id"]: t for t in json.loads((rdir / "tasks.json").read_text())}
    R = {int(r["id"]): r["answer"] for r in csv.DictReader(open(rdir / "ratings.csv"))}
    by = {}
    for i, ans in R.items():
        t = tasks[i]
        win = None if ans == "neither" else (t["left_world"] if ans == "left" else t["right_world"]) == t["a"]
        by.setdefault(t["group"], []).append(win)
    rows, L = {}, [f"# Expedition v4: blind ratings ({len(R)} of {len(tasks)} pairs rated)\n",
                   "First-named world = the one counted as a win (end, N, NF ...). Neither = can't tell.\n",
                   "| group | wins | losses | neither | win rate (decided) | one-sided p |", "|---|---|---|---|---|---|"]
    for g in sorted(by):
        v = by[g]
        w, l_ = sum(x is True for x in v), sum(x is False for x in v)
        rows[g] = (w, l_)
        L.append(f"| {g} | {w} | {l_} | {sum(x is None for x in v)} | {w / max(1, w + l_):.2f} | {_binom_greater(w, w + l_):.3f} |")
    L.append("\np: sign test against a coin flip, except below. Does steering beat an unsteered change "
             "(end vs start against control vs start, Fisher one-sided)?")
    for arm in ("N", "L"):
        e = rows.get(f"end-vs-start:{arm}")
        cc = tuple(map(sum, zip(*[rows[g] for g in ("control:N", "control:L") if g in rows]))) if any(
            g in rows for g in ("control:N", "control:L")) else None
        if e and cc:
            L.append(f"- {arm}: end beats start {e[0]}/{sum(e)}, controls {cc[0]}/{sum(cc)}: "
                     f"p = {fisher_greater(e[0], e[1], cc[0], cc[1]):.3f}")
    sides = list(R.values())
    L.append(f"\nSides: {sides.count('left')} left, {sides.count('right')} right, {sides.count('neither')} neither.")
    (rdir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def setup_arm(a, s1: Path, run: Path, T: list, judge) -> dict:
    """The arm's goal per trigger, and the starting state."""
    starts = [_pic(s1, f"img/{t['image']}") for t in T]
    if a.arm == "L":
        goals = library_goals(a, judge, starts)
    elif a.arm == "N":
        goals = namer_goals(a, s1, run, T, starts)
    else:
        nst = s1 / EXP_DIR / "N" / "state.json"
        if not nst.exists():
            sys.exit("arm N's goals are needed first: run --arm N (at least its setup)")
        goals = [tr["goals"][0] for tr in json.loads(nst.read_text())["triggers"]]
    return {"arm": a.arm, "version": EXP_DIR, "step": 0, "seed": a.seed, "mutants": a.mutants, "steps": a.steps, "triggers": [
        {"id": t["id"], "index": i, "start": f"img/{t['image']}", "start_rule": t["rule"],
         "goals": [{**g, "from_step": 1, "ref": f"img/{t['image']}"}],
         "parent_rule": t["rule"], "parent": f"img/{t['image']}", "g": 0.5, "k": 0.5, "s": 0.5,
         "history": [], "accepted": 0, "k_rejected": 0}
        for i, (t, g) in enumerate(zip(T, goals))]}


def do_review(a, s1: Path, st: dict, judge) -> None:
    from lenia.describe import Ask
    d = _describer(a, s1)
    pid = REVIEW_R if a.arm == "NR" else REVIEW_S
    allowed = {"continue", "refine"} | ({"switch"} if a.arm == "NS" else set())
    asks = []
    for tr in st["triggers"]:
        g = tr["goals"][-1]
        cm = [c["mean_goal"] for c in tr.get("child_mean", [])]
        trend = (f"the average mutant's goal score went from {cm[0]:.2f} in generation 1 to {cm[-1]:.2f} in generation "
                 f"{len(cm)}" if len(cm) > 1 else "")
        prog = (f"The judge's goal score of the current parent against the start world is {tr['g']:.2f} (0.5 = no change "
                f"toward the goal, 1 = clearly further); {trend}. The current parent's complexity against the start is "
                f"{tr['k']:.2f} (0.5 = as complex as the start); {tr['k_rejected']} mutants that moved toward the goal "
                "were rejected because they lost complexity.")
        asks.append(Ask(pid, [_pic(s1, tr["start"]), _pic(s1, tr["parent"])],
                        {"mutants": st["mutants"], "gens": REVIEW_AT, "total": st["steps"], "left": st["steps"] - REVIEW_AT,
                         "sees": g["sees"], "push": g["push"], "progress": prog}))
    ans = d.ask_many(asks, desc=f"{a.arm} review")
    if any(not r.ok for r in ans):
        sys.exit("some review calls failed (not cached); re-run to retry them")
    rescore = []
    for tr, r in zip(st["triggers"], ans):
        dec = r.data["decision"].strip().lower()
        eff = dec if dec in allowed else "continue"
        g = tr["goals"][-1]
        new = None
        if eff == "refine":                   # same target, new direction: the reference stays the start; re-score
            new = {"sees": g["sees"], "push": r.data["push"], "source": "refine", "ref": tr["start"]}
        elif eff == "switch":                 # a new target seen in the current parent: it is the new reference
            new = {"sees": r.data["sees"], "push": r.data["push"], "source": "switch", "ref": tr["parent"]}
        tr["review"] = {"decision": dec, "applied": eff, "why": r.data.get("why"), "raw": r.data, "cost": r.cost,
                        "parent": tr["parent"], "g_before": tr["g"]}
        if new:
            tr["goals"].append({**new, "id": goal_id(new["sees"], new["push"]), "from_step": REVIEW_AT + 1})
            if eff == "switch":
                tr["g"] = 0.5
            else:
                rescore.append(tr)
    if rescore:                               # the parent's score under the refined goal, against the start
        mq = []
        for tr in rescore:
            P, S = _pic(s1, tr["parent"]), _pic(s1, tr["start"])
            pid = attain_prompt(tr["goals"][-1])
            mq += [MultiQuery((pid,), P, S), MultiQuery((pid,), S, P)]
        vs = judge.judge_multi(mq, desc=f"{a.arm} re-score refined goals")
        if any(not all(v.ok for v in vv) for vv in vs):
            sys.exit("some re-scoring requests failed (not cached); re-run to retry them")
        for n, tr in enumerate(rescore):
            tr["g"] = round(0.5 * (vs[2 * n][0].p_first + 1 - vs[2 * n + 1][0].p_first), 4)
    st["reviewed"] = True


def mutants_of(par: Rule, rng, n: int, ea, scale: float = MUT_SCALE, struct: float = STRUCT) -> list:
    """n expedition mutants of a parent: lenia.mace.mutate at the expedition's strength, beta jitter scaled with it."""
    out = []
    for _ in range(n):
        m = mutate(par, rng, scale, struct)
        m.flow = "mace"
        m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1 * min(1.0, scale / 3.0))), 0.5, 4.0))
        out.append(m)
    return out


def same_platform(s1: Path, evals: "Evals", st: dict, ea, tag: str | None = None, scale: float = MUT_SCALE,
                  struct: float = STRUCT, tr: dict | None = None) -> None:
    """Shared mutation randomness only holds if every arm draws bit-identical mutants (np.exp differs in the last bit
    between x86 and ARM; NumPy may change between Colab images). The first arm records the first trigger's first 8
    step-1 mutants (their simulation-cache keys) in <out>/mutation_fingerprint.json; every later arm must match."""
    import hashlib
    tr = tr or st["triggers"][0]
    ms = mutants_of(Rule.from_dict(tr["start_rule"]), np.random.default_rng([st["seed"], tr["index"], 1]), 8, ea,
                    scale, struct)
    fp = hashlib.sha1("".join(evals.key(m) for m in ms).encode()).hexdigest()[:16]
    tag = tag or f"{EXP_DIR} scale={MUT_SCALE} struct={STRUCT}"
    f = s1 / "mutation_fingerprint.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    if tag not in known:
        known[tag] = fp
        f.write_text(json.dumps(known, indent=1))
    elif known[tag] != fp:
        sys.exit("this machine draws different mutants than the arms already run (another CPU type or NumPy version), so "
                 "the arms would not share mutation randomness. Run every arm of a seed in the same environment (Colab).")


def expedition(a) -> None:
    s1, run = Path(a.out), Path(a.run)
    adir = s1 / EXP_DIR / a.arm
    adir.mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    ea = eval_settings(run, a)
    T = json.loads((s1 / "triggers.json").read_text())["triggers"][:a.limit or None]
    judge = _judges(a, s1)
    if (adir / "state.json").exists():
        st = json.loads((adir / "state.json").read_text())
        print(f"arm {a.arm}: resuming after step {st['step']}", file=sys.stderr)
    else:
        st = setup_arm(a, s1, run, T, judge)
        QD.save_state(adir, st)
        for tr in st["triggers"]:
            g = tr["goals"][0]
            print(f"{tr['id']}: {g['sees']} -> {g['push']}", file=sys.stderr)
    evals = Evals(Path(a.evals) if a.evals else s1 / "evals", ea, keep=s1 / "evals")
    M = st["mutants"]
    if st["step"] == 0:
        same_platform(s1, evals, st, ea)
    fh = open(adir / "duels.csv", "a", newline="")
    log = csv.writer(fh)
    if fh.tell() == 0:
        log.writerow(["step", "trigger", "mutant", "texture", "goal", "complexity", "surprise", "accepted", "goal_id"])
    trace = open(adir / "trace.jsonl", "a")
    while st["step"] < st["steps"]:
        step = st["step"] + 1
        t0 = time.time()
        if a.arm in ("NR", "NS") and step == REVIEW_AT + 1 and not st.get("reviewed"):
            do_review(a, s1, st, judge)
            QD.save_state(adir, st)
            print(f"review: {dict(Counter(tr['review']['applied'] for tr in st['triggers']))}", file=sys.stderr)
        rules, owner = [], []
        for tr in st["triggers"]:
            rng = np.random.default_rng([st["seed"], tr["index"], step])      # the same draws in every arm
            for m in mutants_of(Rule.from_dict(tr["parent_rule"]), rng, M, ea):
                rules.append(m)
                owner.append(tr["index"])
        ev = evals.get_many(rules)
        byi = {tr["index"]: tr for tr in st["triggers"]}
        starts = {i: _pic(s1, byi[i]["start"]) for i in byi}
        refs = {i: _pic(s1, byi[i]["goals"][-1]["ref"]) for i in byi}
        mq, meta = [], []                      # meta: (trigger, child, layout); layout 1 = one request, 2 = two
        for j, (i, v) in enumerate(zip(owner, ev)):
            g = byi[i]["goals"][-1]
            if ea.max_texture > 0 and v["texture"] > ea.max_texture:
                log.writerow([step, byi[i]["id"], j % M, round(v["texture"], 4), "", "", "", 0, g["id"]])
                continue
            if g["ref"] == byi[i]["start"]:
                pids = (attain_prompt(g), COMPLEXITY, SURPRISE)
                mq += [MultiQuery(pids, v["pic"], starts[i]), MultiQuery(pids, starts[i], v["pic"])]
                meta.append((i, j, 1))
            else:                              # after a switch: goal against the switch point, the rest against the start
                pg = (attain_prompt(g),)
                mq += [MultiQuery(pg, v["pic"], refs[i]), MultiQuery(pg, refs[i], v["pic"]),
                       MultiQuery((COMPLEXITY, SURPRISE), v["pic"], starts[i]), MultiQuery((COMPLEXITY, SURPRISE), starts[i], v["pic"])]
                meta.append((i, j, 2))
        vs = judge.judge_multi(mq, desc=f"{a.arm} step {step}")
        if any(not all(v.ok for v in vv) for vv in vs):
            sys.exit("some judge requests failed (not cached); re-run to retry them (the step restarts)")
        best, pos, gs = {}, 0, {}
        both = lambda x, y, q: 0.5 * (x[q].p_first + 1 - y[q].p_first)
        for i, j, lay in meta:
            if lay == 1:
                x, y = vs[pos], vs[pos + 1]
                G, K, S = both(x, y, 0), both(x, y, 1), both(x, y, 2)
                pos += 2
            else:
                G = both(vs[pos], vs[pos + 1], 0)
                K, S = both(vs[pos + 2], vs[pos + 3], 0), both(vs[pos + 2], vs[pos + 3], 1)
                pos += 4
            tr = byi[i]
            gs.setdefault(i, []).append(G)
            acc = K >= K_FLOOR                 # comma selection: the best child passing the floor replaces the parent
            if not acc and G >= tr["g"] + MARGIN and step <= REVIEW_AT:
                tr["k_rejected"] += 1
            if acc and (G, K) > best.get(i, (-1.0, 0.0, None, None))[:2]:
                best[i] = (G, K, j, S)
            log.writerow([step, tr["id"], j % M, round(ev[j]["texture"], 4), round(G, 4), round(K, 4), round(S, 4), int(acc),
                          tr["goals"][-1]["id"]])
        for i, tr in byi.items():               # the unselected progress signal: this step's children's mean goal score
            if gs.get(i):
                tr.setdefault("child_mean", []).append({"step": step, "mean_goal": round(float(np.mean(gs[i])), 4), "n": len(gs[i])})
        for i, (G, K, j, S) in best.items():
            tr = byi[i]
            evals.persist(ev[j])
            tr["parent_rule"], tr["parent"] = rules[j].to_dict(), ev[j]["image"]
            tr["g"], tr["k"], tr["s"] = round(G, 4), round(K, 4), round(S, 4)
            if step <= REVIEW_AT:
                tr["accepted"] += 1
            if tr["g"] > tr.get("best", {}).get("g", -1.0):
                tr["best"] = {"step": step, "image": ev[j]["image"], "g": tr["g"], "k": tr["k"], "s": tr["s"]}
            h = {"step": step, "image": ev[j]["image"], "goal_score": tr["g"], "complexity": tr["k"], "surprise": tr["s"],
                 "child_mean_goal": tr["child_mean"][-1]["mean_goal"],
                 "goal": tr["goals"][-1]["id"], "bodies": ev[j]["bodies"], "motion": round(ev[j]["motion"], 4)}
            tr["history"].append(h)
            trace.write(json.dumps({"arm": a.arm, "trigger": tr["id"], **h, "rule": rules[j].to_dict()}) + "\n")
        st["step"] = step
        QD.save_state(adir, st)
        fh.flush()
        trace.flush()
        cm = [np.mean(v) for v in gs.values() if v]
        print(f"{a.arm} step {step}: children's mean goal score {np.mean(cm) if cm else float('nan'):.2f}, "
              f"{len(best)}/{len(byi)} triggers moved, {len(mq)} requests, "
              f"{time.time() - t0:.0f}s, judge spent ${judge.spent:.4f} (cache ${judge.cache.total_cost():.2f})", file=sys.stderr)
    fh.close()
    trace.close()
    print(f"arm {a.arm} done: {adir}", file=sys.stderr)


def check(a) -> None:
    """Held-out Luna, end vs start per trigger: attain (the original goal), complexity, surprise; for a changed goal
    also attain of the new goal, end vs the parent it was set at. Gate 0c (arm L): mean attain > 0.6 and mean complexity
    >= 0.5. Writes arms/<arm>/check.json, report.md and check.png (start | end per trigger)."""
    s1 = Path(a.out)
    adir = s1 / (a.dir or EXP_DIR) / a.arm
    st = json.loads((adir / "state.json").read_text())
    if st["step"] < st["steps"]:
        sys.exit(f"arm {a.arm} has run {st['step']} of {st['steps']} steps")
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    luna = Judge(a.luna, {**cfg["judges"]["defaults"], **cfg["judges"][a.luna], "max_run_usd": a.max_luna_usd},
                 s1 / "luna_cache.sqlite", yes=True)
    qs, meta = [], []
    for tr in st["triggers"]:
        S, E = _pic(s1, tr["start"]), _pic(s1, tr["parent"])
        g0 = tr["goals"][0]
        for name, pid, x, y in [("attain", attain_prompt(g0), E, S), ("complexity", COMPLEXITY, E, S), ("surprise", SURPRISE, E, S)] + (
                [("attain_new", attain_prompt(tr["goals"][-1]), E, _pic(s1, tr["goals"][-1].get("ref", tr["review"]["parent"])))]
                if len(tr["goals"]) > 1 else []):
            qs += [Query(pid, x, y), Query(pid, y, x)]
            meta.append((tr["id"], name))
    vs = luna.judge_many(qs, desc=f"{a.arm} check (Luna)")
    if any(not v.ok for v in vs):
        sys.exit("some Luna calls failed (not cached); re-run to retry them")
    res = {}
    for n, (tid, name) in enumerate(meta):
        res.setdefault(tid, {})[name] = round(0.5 * (vs[2 * n].p_first + 1 - vs[2 * n + 1].p_first), 4)
    (adir / "check.json").write_text(json.dumps(res, indent=1))
    m = {k: float(np.mean([r[k] for r in res.values()])) for k in ("attain", "complexity", "surprise")}
    L = [f"# Arm {a.arm}, seed {st['seed']}: end vs start (Luna, held out)\n",
         f"P = probability Luna prefers the end world, both orders. {len(res)} triggers, {st['steps']} steps x {st['mutants']} mutants.\n",
         "| trigger | steps taken | goal | attain | complexity | surprise | change |", "|---|---|---|---|---|---|---|"]
    for tr in st["triggers"]:
        r = res[tr["id"]]
        g = tr["goals"][0]
        ch = ""
        if len(tr["goals"]) > 1:
            ch = f"{tr['review']['applied']}: {tr['goals'][-1]['push']} (attain new {r['attain_new']:.2f})"
        elif tr.get("review"):
            ch = "continue"
        L.append(f"| {tr['id']} | {len(tr['history'])} | {g['sees']} -> {g['push']} | {r['attain']:.2f} | {r['complexity']:.2f} | "
                 f"{r['surprise']:.2f} | {ch} |")
    L.append(f"\n**Means:** attain {m['attain']:.2f}, complexity {m['complexity']:.2f}, surprise {m['surprise']:.2f}.")
    cms = [tr["child_mean"] for tr in st["triggers"] if tr.get("child_mean")]
    if cms:                                  # v3: the unselected progress signal (Jev, goal vs start)
        by = {}
        for cm in cms:
            for c in cm:
                by.setdefault(c["step"], []).append(c["mean_goal"])
        L.append("\nMean goal score of each step's children (Jev, vs the start; not inflated by selection): " +
                 ", ".join(f"{k}: {np.mean(v):.2f}" for k, v in sorted(by.items())))
    if a.arm == "L":
        ok = m["attain"] > 0.6 and m["complexity"] >= 0.5
        L.append(f"\n**Gate 0c** (mean attain > 0.6 and mean complexity >= 0.5): {'PASS' if ok else 'FAIL'}.")
    (adir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"Luna spent ${luna.spent:.4f}", file=sys.stderr)
    sheet_arm(s1, st, res, adir / "check.png")


def sheet_arm(s1: Path, st: dict, res: dict, path: Path, px: int = 300) -> None:
    """One row per trigger: start | end, with the goal and Luna's end-vs-start P."""
    head = 40
    im = Image.new("RGB", (2 * px + 24, len(st["triggers"]) * (px + head + 10) + 10), (250, 250, 250))
    d = ImageDraw.Draw(im)
    f1, f2 = _font(15), _font(12)
    for k, tr in enumerate(st["triggers"]):
        y = 10 + k * (px + head + 10)
        r = res[tr["id"]]
        g = tr["goals"][-1]
        d.text((8, y), f"{tr['id']}  {len(tr['history'])} steps  attain {r['attain']:.2f}  complexity {r['complexity']:.2f}  "
               f"surprise {r['surprise']:.2f}", fill="black", font=f1)
        d.text((8, y + 19), (g["push"][:95] + ("..." if len(g["push"]) > 95 else "")), fill=(70, 70, 70), font=f2)
        for c, rel in enumerate((tr["start"], tr["parent"])):
            im.paste(Image.open(s1 / rel).convert("RGB").resize((px, px), Image.BILINEAR), (8 + c * (px + 8), y + head))
    im.save(path)


# ---------- heritability: do children keep their parent's progress? (after arm N, 10-29) ----------
#
# Arms L and N (v2) stalled after 1-3 steps: once a child scoring ~0.85 on its goal (vs the start) was accepted, its own
# children averaged ~0.45, no better than mutants of the start. The mutation operator was inherited from the base search
# (mut_scale 3.0 plus discrete ring edits whose chance does not depend on the scale), built to explore, not to develop a
# feature. Here every expedition's end parent gets MUTANTS children under each (scale, struct) variant, judged exactly as
# in an expedition (goal, complexity, surprise vs the start, both orders). Reported per variant: how much of the parent's
# goal score the children keep, and how many would be accepted (goal >= parent + MARGIN, complexity >= K_FLOOR).
VARIANTS = ((3.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.3, 0.0))       # (mut_scale, struct): the first is the current one


def heritability(a) -> None:
    s1, run = Path(a.out), Path(a.run)
    hdir = s1 / "heritability"
    hdir.mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    ea = eval_settings(run, a)
    evals = Evals(Path(a.evals) if a.evals else s1 / "evals", ea)
    judge = _judges(a, s1)
    parents = []
    for arm in a.arms.split(","):
        st = json.loads((s1 / EXP_DIR / arm / "state.json").read_text())
        parents += [(arm, tr) for tr in st["triggers"] if tr["history"]]
    rules, meta = [], []
    for v, (scale, struct) in enumerate(VARIANTS):
        for arm, tr in parents:
            rng = np.random.default_rng([a.seed, tr["index"], 900 + v, ARMS.index(arm)])
            par = Rule.from_dict(tr["parent_rule"])
            for c in range(a.mutants):
                m = mutate(par, rng, scale, struct)
                m.flow = "mace"
                m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1 * min(1.0, scale / 3.0))), 0.5, 4.0))
                rules.append(m)
                meta.append((v, arm, tr))
    t0 = time.time()
    ev = evals.get_many(rules)
    print(f"{len(rules)} children grown in {time.time() - t0:.0f}s", file=sys.stderr)
    starts = {}
    mq, keep = [], []
    for k, ((v, arm, tr), e) in enumerate(zip(meta, ev)):
        if ea.max_texture > 0 and e["texture"] > ea.max_texture:
            continue
        S = starts.setdefault(tr["start"], _pic(s1, tr["start"]))
        pids = (attain_prompt(tr["goals"][-1]), COMPLEXITY, SURPRISE)
        mq += [MultiQuery(pids, e["pic"], S), MultiQuery(pids, S, e["pic"])]
        keep.append(k)
    vs = judge.judge_multi(mq, desc="heritability")
    if any(not all(x.ok for x in xx) for xx in vs):
        sys.exit("some judge requests failed (not cached); re-run to retry them")
    rows = []
    for n, k in enumerate(keep):
        v, arm, tr = meta[k]
        x, y = vs[2 * n], vs[2 * n + 1]
        G, K, S_ = (0.5 * (x[q].p_first + 1 - y[q].p_first) for q in range(3))
        rows.append({"variant": v, "arm": arm, "trigger": tr["id"], "parent_g": tr["g"], "G": round(G, 4), "K": round(K, 4),
                     "S": round(S_, 4)})
    n_tex = {v: sum(1 for (vv, *_), e in zip(meta, ev) if vv == v and ea.max_texture > 0 and e["texture"] > ea.max_texture)
             for v in range(len(VARIANTS))}
    (hdir / "results.json").write_text(json.dumps({"variants": VARIANTS, "mutants": a.mutants, "rows": rows, "texture": n_tex}, indent=1))
    L = [f"# Heritability: children of each expedition's end parent ({len(parents)} parents, {a.mutants} children per variant)\n",
         "Judged as in an expedition: goal, complexity, surprise vs the start, both orders. 'keeps' = goal score within 0.1 of "
         f"the parent's; 'accepted' = goal >= parent + {MARGIN} and complexity >= {K_FLOOR}. Parents' mean goal score "
         f"{np.mean([tr['g'] for _, tr in parents]):.2f}.\n",
         "| mut_scale | ring edits | grid noise | child goal mean | child - parent (median) | keeps | accepted | complexity >= 0.45 | surprise mean |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v, (scale, struct) in enumerate(VARIANTS):
        rr = [r for r in rows if r["variant"] == v]
        d = np.array([r["G"] - r["parent_g"] for r in rr])
        acc = np.mean([(r["G"] >= r["parent_g"] + MARGIN and r["K"] >= K_FLOOR) for r in rr])
        L.append(f"| {scale} | {'yes' if struct else 'no'}{' (current)' if v == 0 else ''} | {n_tex[v]} | {np.mean([r['G'] for r in rr]):.2f} | "
                 f"{np.median(d):+.2f} | {np.mean(d >= -0.1):.0%} | {acc:.1%} | {np.mean([r['K'] >= K_FLOOR for r in rr]):.0%} | "
                 f"{np.mean([r['S'] for r in rr]):.2f} |")
    (hdir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"judge spent ${judge.spent:.4f}", file=sys.stderr)


# ---------- goal check: who judges goal progress right? (after arm L, 10-29) ----------
#
# Gate 0c used Luna's attain as ground truth, but Luna was never validated as a judge of goal progress (pilot: Felix saw
# the goal branch show its goal 13:1, Luna 0.53). Here Felix rates blind, start vs end with sides shuffled and no scores,
# every expedition that moved, in the given expedition folders; Jev (attain), Luna (from each folder's check.json) and the
# VLM (GOAL_JUDGE, both image orders) judge the same pairs. Reported: each judge's agreement with Felix's decisive
# answers, and Felix's "end shows it" rate per folder. Goals are split, by a keyword rule fixed before rating, into
# motion goals (hard to see in four snapshots) and static ones.
DYNAMIC = ("rhythm", "heartbeat", "orbit", "circulat", "swarm", "crawl", "curl", "wave", "swim", "travel", "move",
           "chase", "rotat", "puls", "drift")


def goalcheck(a) -> None:
    s1 = Path(a.out)
    gdir = s1 / "goalcheck"
    (gdir / "images").mkdir(parents=True, exist_ok=True)
    if a.step == "build":
        items = []
        for d in a.dirs.split(","):
            st = json.loads((s1 / d / a.arm / "state.json").read_text())
            for tr in st["triggers"]:
                if tr["history"]:
                    items.append({"dir": d, "trigger": tr["id"], "start": tr["start"], "end": tr["parent"],
                                  "sees": tr["goals"][0]["sees"], "push": tr["goals"][0]["push"]})
        rng = np.random.default_rng(a.seed)
        tasks = []
        for k, j in enumerate(rng.permutation(len(items))):
            x = items[int(j)]
            start_left = bool(rng.integers(2))
            for side, rel in (("a", x["start"] if start_left else x["end"]), ("b", x["end"] if start_left else x["start"])):
                Image.open(s1 / rel).convert("RGB").save(gdir / "images" / f"{k:02d}{side}.png")
            dyn = any(w in x["push"].lower() for w in DYNAMIC)
            tasks.append({"id": k, "left": f"{k:02d}a.png", "right": f"{k:02d}b.png", "start_side": "left" if start_left else "right",
                          "motion_goal": dyn, **x})
        (gdir / "tasks.json").write_text(json.dumps(tasks, indent=1))
        print(f"{len(tasks)} pairs ({sum(t['motion_goal'] for t in tasks)} motion goals) -> {gdir}; next: rate, then judge")
    elif a.step == "rate":
        from lenia.rating import rate_goal_pairs
        rate_goal_pairs(gdir, port=a.port)
    elif a.step == "judge":
        from lenia.describe import Ask
        from lenia.prompts import GOAL_JUDGE
        T = json.loads((gdir / "tasks.json").read_text())
        judge = _judges(a, s1)
        pid = lambda t: attain_prompt({"id": goal_id(t["sees"], t["push"]), "sees": t["sees"], "push": t["push"]})
        qs = []
        for t in T:
            E, S = _pic(s1, t["end"]), _pic(s1, t["start"])
            qs += [Query(pid(t), E, S), Query(pid(t), S, E)]
        vs = judge.judge_many(qs, desc="goal check: Jev attain")
        d = _describer(a, s1)
        asks = []
        for t in T:
            E, S = _pic(s1, t["end"]), _pic(s1, t["start"])
            asks += [Ask(GOAL_JUDGE, [E, S], {"sees": t["sees"], "push": t["push"]}),
                     Ask(GOAL_JUDGE, [S, E], {"sees": t["sees"], "push": t["push"]})]
        ans = d.ask_many(asks, desc="goal check: VLM")
        if any(not v.ok for v in vs):
            sys.exit("some Jev calls failed (not cached); re-run to retry them")
        res, miss = {}, 0
        for n, t in enumerate(T):
            res[t["id"]] = {"jev": round(0.5 * (vs[2 * n].p_first + 1 - vs[2 * n + 1].p_first), 4), "vlm": None}
            x, y = ans[2 * n], ans[2 * n + 1]
            if x.ok and y.ok:                 # a pair the VLM could not answer in both orders is left out, not guessed
                ch = [str(x.data["choice"]).strip(), str(y.data["choice"]).strip()]
                res[t["id"]].update(vlm=0.5 * ((ch[0] == "1") + (ch[1] == "2")), vlm_why=[x.data.get("why"), y.data.get("why")])
            else:
                miss += 1
        if miss:
            print(f"WARNING: the VLM is missing {miss} of {len(T)} pairs (re-run judge later to fill them in)", file=sys.stderr)
        (gdir / "judges.json").write_text(json.dumps(res, indent=1))
        print(f"Jev ${judge.spent:.4f}, VLM ${d.spent:.4f}; next: report", file=sys.stderr)
    else:
        goalcheck_report(s1, gdir)


def goalcheck_report(s1: Path, gdir: Path) -> None:
    T = json.loads((gdir / "tasks.json").read_text())
    J = {int(k): v for k, v in json.loads((gdir / "judges.json").read_text()).items()} if (gdir / "judges.json").exists() else {}
    F = {int(r["id"]): r["answer"] for r in csv.DictReader(open(gdir / "ratings.csv"))} if (gdir / "ratings.csv").exists() else {}
    luna = {}
    for d in {t["dir"] for t in T}:
        cj = s1 / d / "L" / "check.json"
        if cj.exists():
            for tid, r in json.loads(cj.read_text()).items():
                luna[(d, tid)] = r["attain"]
    rows = []
    for t in T:
        f = F.get(t["id"])
        fe = None if f in (None, "neither") else (f != t["start_side"])          # Felix: the end shows it
        rows.append({**t, "felix": f, "felix_end": fe, "jev": J.get(t["id"], {}).get("jev"),
                     "vlm": J.get(t["id"], {}).get("vlm"), "luna": luna.get((t["dir"], t["trigger"]))})
    L = ["# Goal check: start vs end of the arm L expeditions that moved\n",
         "Felix blind (sides shuffled, no scores); judges: P(the end shows the wanted change more clearly), both orders.\n",
         "| folder | pairs | Felix: end / start / neither |", "|---|---|---|"]
    for d in sorted({r["dir"] for r in rows}):
        rr = [r for r in rows if r["dir"] == d and r["felix"]]
        L.append(f"| {d} | {len(rr)} | {sum(r['felix_end'] is True for r in rr)} / {sum(r['felix_end'] is False for r in rr)} / "
                 f"{sum(r['felix'] == 'neither' for r in rr)} |")
    dec = [r for r in rows if r["felix_end"] is not None]
    L += ["", f"Agreement with Felix on his {len(dec)} decisive pairs (judge P > 0.5 = end):", "",
          "| judge | agree | mean P when Felix says end | mean P when Felix says start |", "|---|---|---|---|"]
    for j in ("jev", "luna", "vlm"):
        dd = [r for r in dec if r[j] is not None]
        if dd:
            ag = sum((r[j] > 0.5) == r["felix_end"] for r in dd)
            pe = [r[j] for r in dd if r["felix_end"]]
            ps = [r[j] for r in dd if not r["felix_end"]]
            L.append(f"| {j} | {ag}/{len(dd)} | {np.mean(pe) if pe else float('nan'):.2f} | {np.mean(ps) if ps else float('nan'):.2f} |")
    for lab, m in (("motion goals", True), ("static goals", False)):
        rr = [r for r in rows if r["motion_goal"] == m and r["felix"]]
        if rr:
            L.append(f"\n{lab}: Felix end {sum(r['felix_end'] is True for r in rr)}, start {sum(r['felix_end'] is False for r in rr)}, "
                     f"neither {sum(r['felix'] == 'neither' for r in rr)}")
    L += ["", "| folder | trigger | goal | Felix | Jev | Luna | VLM |", "|---|---|---|---|---|---|---|"]
    f2 = lambda v: "-" if v is None else f"{v:.2f}"
    for r in sorted(rows, key=lambda r: (r["dir"], r["trigger"])):
        fl = {True: "end", False: "start", None: r["felix"] or "-"}[r["felix_end"]]
        L.append(f"| {r['dir']} | {r['trigger']} | {r['push']}{' (motion)' if r['motion_goal'] else ''} | {fl} | {f2(r['jev'])} | "
                 f"{f2(r['luna'])} | {f2(r['vlm'])} |")
    (gdir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def sheet(a, cols: int = 4, px: int = 420) -> Path:
    """triggers.png: one 2x2 picture per trigger, highest re-scored surprise first."""
    out = Path(a.out)
    T = json.loads((out / "triggers.json").read_text())["triggers"]
    head = 46
    rows = (len(T) + cols - 1) // cols
    im = Image.new("RGB", (cols * (px + 8) + 8, rows * (px + head + 8) + 8), (250, 250, 250))
    d = ImageDraw.Draw(im)
    f1, f2 = _font(17), _font(14)
    for i, t in enumerate(T):
        x, y = 8 + (i % cols) * (px + 8), 8 + (i // cols) * (px + head + 8)
        low = t["surprise"] < 0.5
        d.text((x, y), f"{t['id']}  it{t['it']} I{t['island']}  ({t['lineage_members']} in lineage)", fill="black", font=f1)
        c = "-" if t["complexity"] is None else f"{t['complexity']:.2f}"
        d.text((x, y + 22), f"surprise {t['surprise']:.2f} (stored {t['stored']:.2f})  complexity {c}  "
               f"bodies {t['bodies']}  cov {t['coverage']:.2f}", fill=(190, 30, 30) if low else (60, 60, 60), font=f2)
        im.paste(Image.open(out / "img" / t["image"]).convert("RGB").resize((px, px), Image.BILINEAR), (x, y + head))
    p = out / "triggers.png"
    im.save(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triggers")
    t.add_argument("--run", required=True, help="the base run (mace_qd.py --sarch)")
    t.add_argument("--out", required=True, help="this seed's Stage 1 folder (shared judge cache)")
    t.add_argument("--judge", default="jev")
    t.add_argument("--config", default=None)
    t.add_argument("--opponents", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-usd", type=float, default=0.25, help="hard cap on the --out cache's total spend, across resumes")
    s = sub.add_parser("sheet")
    s.add_argument("--out", required=True)
    e = sub.add_parser("expedition")
    c = sub.add_parser("check")
    for x in (e, c):
        x.add_argument("--arm", required=True, choices=ARMS)
        x.add_argument("--run", required=True, help="the base run")
        x.add_argument("--out", required=True, help="this seed's Stage 1 folder")
        x.add_argument("--config", default=None)
    e.add_argument("--backend", default="numpy", help="numpy | auto | cuda | mps")
    e.add_argument("--judge", default="jev")
    e.add_argument("--vlm", default="gemini-pro", help="namer / reviewer (arms N, NR, NS)")
    e.add_argument("--library", default="data/goal_library_v3.json")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--mutants", type=int, default=MUTANTS)
    e.add_argument("--steps", type=int, default=EXP_STEPS)
    e.add_argument("--limit", type=int, default=0, help="only the first k triggers (dry runs)")
    e.add_argument("--max-usd", type=float, default=6.0, help="hard cap on the --out Jev cache's total spend (all arms)")
    e.add_argument("--evals", default=None, help="simulation cache folder (default <out>/evals; on Colab use local disk, "
                   "e.g. /content/evals: accepted worlds are still copied to <out>/evals)")
    e.add_argument("--max-vlm-usd", type=float, default=3.0, help="hard cap on the --out VLM cache's total spend")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        e.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    h = sub.add_parser("heritability")
    h.add_argument("--run", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--arms", default="L,N")
    h.add_argument("--evals", default=None, help="simulation cache folder (default <out>/evals; a local disk is faster on Colab)")
    h.add_argument("--backend", default="numpy")
    h.add_argument("--judge", default="jev")
    h.add_argument("--config", default=None)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--mutants", type=int, default=16)
    h.add_argument("--max-usd", type=float, default=6.0)
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        h.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    g = sub.add_parser("goalcheck")
    g.add_argument("step", choices=["build", "rate", "judge", "report"])
    g.add_argument("--out", required=True)
    g.add_argument("--arm", default="L")
    g.add_argument("--dirs", default="exp-v2,arms", help="expedition folders to pool (v2, v1)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--port", type=int, default=8772)
    g.add_argument("--config", default=None)
    g.add_argument("--judge", default="jev")
    g.add_argument("--vlm", default="gemini-pro")
    g.add_argument("--max-usd", type=float, default=6.0)
    g.add_argument("--max-vlm-usd", type=float, default=3.0)
    c.add_argument("--max-luna-usd", type=float, default=0.5)
    c.add_argument("--luna", default="luna", help="the held-out judge (fake for dry runs)")
    c.add_argument("--dir", default=None, help=f"the expedition folder (default {EXP_DIR}; arms = v1)")
    k = sub.add_parser("captions", help="goal captions for the v4 CLIP shortlist (step 1)")
    k.add_argument("--arm", default="N", choices=("L", "N"))
    k.add_argument("--run", required=True, help="the base run (the namer's typical worlds)")
    k.add_argument("--out", required=True, help="this seed's Stage 1 folder (triggers, the namer's cache)")
    k.add_argument("--dest", default=None, help="where N.json, N.md and the caption cache go (default <out>/captions)")
    k.add_argument("--config", default=None)
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--vlm", default="gemini-pro", help="the namer whose cached answers are N's goals (no new calls)")
    k.add_argument("--judge", default="jev", help="the matcher whose cached answers give L's goals (no new calls)")
    k.add_argument("--library", default="data/goal_library_v3.json")
    k.add_argument("--caption-vlm", default="gemini-pro", help="writes the captions (gemini-pro-or: the same model via OpenRouter; fake for a dry run)")
    k.add_argument("--limit", type=int, default=0, help="only the first k triggers")
    k.add_argument("--max-vlm-usd", type=float, default=0.25, help="hard cap on the caption cache's total spend")
    q = sub.add_parser("capcheck", help="v4 step 2: target vs change CLIP scoring against Jev's picks (Colab; no API calls)")
    q.add_argument("--run", required=True, help="the base run (simulation settings)")
    q.add_argument("--out", required=True, help="this seed's Stage 1 folder (exp-v3/L, captions/L.json)")
    q.add_argument("--backend", default="numpy", help="numpy | auto | cuda | mps")
    q.add_argument("--evals", default=None, help="simulation cache for the regrown children (default /tmp/capcheck-evals; "
                   "on Colab /content/capcheck-evals); nothing is written to <out>/evals")
    q.add_argument("--clip", default="openai/clip-vit-base-patch32", help="CLIP model (fake for a dry run)")
    q.add_argument("--limit", type=int, default=0, help="only the first k captioned triggers")
    q.add_argument("--steps", type=int, default=0, help="only the first k steps")
    q.add_argument("--dry", action="store_true", help="score even if mutation does not reproduce (offline dry runs)")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        q.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    v = sub.add_parser("v4", help="expedition v4: 48 children, CLIP or floor shortlist of 16, Jev picks (arms L, N, NF, C)")
    v.add_argument("--arm", required=True, choices=V4_ARMS)
    v.add_argument("--run", required=True, help="the base run")
    v.add_argument("--out", required=True, help="this seed's Stage 1 folder")
    v.add_argument("--tag", default=V4_DIR, help=f"results folder under --out (default {V4_DIR}; e.g. exp-v4-smoke)")
    v.add_argument("--config", default=None)
    v.add_argument("--backend", default="numpy", help="numpy | auto | cuda | mps")
    v.add_argument("--judge", default="jev")
    v.add_argument("--clip", default="openai/clip-vit-base-patch32", help="CLIP model (fake for a dry run)")
    v.add_argument("--vlm", default="gemini-pro", help="the namer whose cached answers are N's goals (no new calls)")
    v.add_argument("--library", default="data/goal_library_v3.json")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--steps", type=int, default=V4_STEPS, help="the schedule's length (fixed at setup)")
    v.add_argument("--motion-floor", type=float, default=V4_MOTION, help="motion floor for the shortlist (fixed at setup; "
                   "the bench's 1.0, uncalibrated; 0 for offline dry runs, whose tiny worlds barely move)")
    v.add_argument("--stop-after", type=int, default=0, help="stop after k steps (smoke tests; resumable)")
    v.add_argument("--limit", type=int, default=0, help="only the first k of the arm's triggers")
    v.add_argument("--evals", default=None, help="simulation cache folder (on Colab: /content/evals; picks are copied "
                   "to <out>/evals)")
    v.add_argument("--max-usd", type=float, default=6.5, help="hard cap on the --out Jev cache's TOTAL spend (all arms, "
                   "earlier expeditions included: $3.70 on 11-03)")
    v.add_argument("--max-vlm-usd", type=float, default=0.0)
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        v.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    r4 = sub.add_parser("v4rate", help="v4's blind session: build (Colab: films), rate (Mac page), report")
    r4.add_argument("step", choices=["build", "refilm", "rate", "report"])
    r4.add_argument("--out", required=True, help="this seed's Stage 1 folder (the session goes to <out>/rate-v4)")
    r4.add_argument("--run", default=None, help="the base run (build and refilm)")
    r4.add_argument("--tag", default=V4_DIR)
    r4.add_argument("--backend", default="numpy")
    r4.add_argument("--seed", type=int, default=0)
    r4.add_argument("--every", type=int, default=30, help="simulation steps per film frame")
    r4.add_argument("--px", type=int, default=280)
    r4.add_argument("--fps", type=int, default=15)
    r4.add_argument("--port", type=int, default=8773)
    r4.add_argument("--force", action="store_true", help="rebuild an existing session (refused once ratings exist)")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        r4.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    a = ap.parse_args()
    {"v4rate": v4rate, "v4": expedition_v4, "capcheck": capcheck, "captions": captions, "triggers": triggers, "sheet": sheet, "expedition": expedition, "check": check, "goalcheck": goalcheck,
     "heritability": heritability}[a.cmd](a)


if __name__ == "__main__":
    main()
