"""Stage 1 of the escalation loop (project doc "VLM in the loop - escalation plan (core first)").

Step 1, the trigger list:

  PYTHONPATH=src:scripts python3 scripts/stage1.py triggers --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 --max-usd 0.25
  PYTHONPATH=src:scripts python3 scripts/stage1.py triggers --run runs/mace-qd5-sarch-s0 --out /tmp/s1-dry --judge fake
  PYTHONPATH=src:scripts python3 scripts/stage1.py sheet --out runs/stage1-s0          # redraw triggers.png only

Goal captions for the v4 CLIP shortlist (arm N; arm L's are hand-written in data/goal_library_v3_captions.json):

  PYTHONPATH=src:scripts python3 scripts/stage1.py captions --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 --max-vlm-usd 0.25
  PYTHONPATH=src:scripts python3 scripts/stage1.py captions --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 \\
      --caption-vlm fake --dest /tmp/cap-dry

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
from lenia.judge import ChoiceQuery, Judge, MultiQuery, Query, load_env
from lenia.mace import mutate
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
# Arm L: hand-written captions for the library goals it draws (CAPTIONS_L, checked by Felix). Arm N: one VLM call per
# goal (the goal's sees and push, and the trigger's start picture). N's goals come from the namer's cached answers
# only (cache_only: no new naming calls, since those define the goals); the captions go through --caption-vlm (Gemini
# on Vertex by default; gemini-pro-or is the OpenRouter fallback) into their own cache, <dest>/caption_cache.sqlite.
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
    goals = namer_goals(a, s1, run, T, starts, cache_only=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    d = Describer(a.caption_vlm, cfg["vlms"][a.caption_vlm], dest / "caption_cache.sqlite", max_usd=a.max_vlm_usd)
    ans = d.ask_many([Ask(CAPTION, [w], {"sees": g["sees"], "push": g["push"]}) for g, w in zip(goals, starts)],
                     desc="N captions")
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
    (dest / "N.json").write_text(json.dumps(rows, indent=1))
    L = [f"# Arm N goal captions, seed {a.seed} ({a.caption_vlm}, {CAPTION})\n",
         f"{sum(r['end'] != 'none' for r in rows)} of {len(rows)} goals captioned; spend ${sum(r['cost'] for r in rows):.4f}.\n",
         "| trigger | push | end caption | start caption | flags |", "|---|---|---|---|---|"]
    L += [f"| {r['trigger']} | {r['push']} | {r['end']} | {r['start']} | {', '.join(r['flags'])} |" for r in rows]
    (dest / "N.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {dest / 'N.json'}", file=sys.stderr)


def setup_arm(a, s1: Path, run: Path, T: list, judge) -> dict:
    """The arm's goal per trigger, and the starting state."""
    starts = [_pic(s1, f"img/{t['image']}") for t in T]
    if a.arm == "L":
        L = library(REPO_ROOT / a.library)
        probs = match(judge, L, starts, np.random.default_rng([a.seed, 101]), "L match")
        goals, used = [], set()
        for i, p in enumerate(probs):
            g = draw_goal(L, p, np.random.default_rng([a.seed, 102, i]), used)
            used.add(g["obs"])
            goals.append(g)
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


def same_platform(s1: Path, evals: "Evals", st: dict, ea) -> None:
    """Shared mutation randomness only holds if every arm draws bit-identical mutants (np.exp differs in the last bit
    between x86 and ARM; NumPy may change between Colab images). The first arm records the first trigger's first 8
    step-1 mutants (their simulation-cache keys) in <out>/mutation_fingerprint.json; every later arm must match."""
    import hashlib
    tr = st["triggers"][0]
    ms = mutants_of(Rule.from_dict(tr["start_rule"]), np.random.default_rng([st["seed"], tr["index"], 1]), 8, ea)
    fp = hashlib.sha1("".join(evals.key(m) for m in ms).encode()).hexdigest()[:16]
    tag = f"{EXP_DIR} scale={MUT_SCALE} struct={STRUCT}"
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
    k = sub.add_parser("captions", help="arm N's goal captions (v4 step 1); arm L's are hand-written in " + CAPTIONS_L)
    k.add_argument("--run", required=True, help="the base run (the namer's typical worlds)")
    k.add_argument("--out", required=True, help="this seed's Stage 1 folder (triggers, the namer's cache)")
    k.add_argument("--dest", default=None, help="where N.json, N.md and the caption cache go (default <out>/captions)")
    k.add_argument("--config", default=None)
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--vlm", default="gemini-pro", help="the namer whose cached answers are N's goals (no new calls)")
    k.add_argument("--caption-vlm", default="gemini-pro", help="writes the captions (gemini-pro-or: the same model via OpenRouter; fake for a dry run)")
    k.add_argument("--limit", type=int, default=0, help="only the first k triggers")
    k.add_argument("--max-vlm-usd", type=float, default=0.25, help="hard cap on the caption cache's total spend")
    a = ap.parse_args()
    {"captions": captions, "triggers": triggers, "sheet": sheet, "expedition": expedition, "check": check, "goalcheck": goalcheck,
     "heritability": heritability}[a.cmd](a)


if __name__ == "__main__":
    main()
