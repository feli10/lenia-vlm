"""Full open-ended run, seed 1 (project doc "Handoff - full run build (seed 1)", 11-05): one aim, end to end, in rounds.

  python scripts/fullrun.py run --out RUNS/full-s1-smoke --backend auto --evals /content/evals-smoke \\
      --rounds 1 --iters 2 --triggers 2 --end 6                                  # smoke test of round 1 first (~$0.35)
  python scripts/fullrun.py run --out RUNS/full-s1 --backend auto --evals /content/evals --baseline RUNS/base-s1
  python scripts/mace_qd.py run --out RUNS/base-s1 --backend auto --iterations 100   # the baseline: continues the fork
  python scripts/fullrun.py gallery --run RUNS/base-s1 --out RUNS/base-s1             # the baseline's gallery
  python scripts/fullrun.py report --out RUNS/full-s1 --baseline RUNS/base-s1         # report.md (no calls)
  python scripts/fullrun.py ledger --out RUNS/full-s1                                 # ledger.csv (no calls)
  PYTHONPATH=src:scripts python3 scripts/fullrun.py run --out ~/fr-dry --dry          # offline: fake judges, tiny worlds

`run` resumes after a disconnect (re-run the same command): the driver's state says which round and phase it is in,
the map search resumes from its own state, and every judge and VLM answer is cached.

Each round (--rounds 5):
1. Map search: --iters (20) more iterations of mace_qd.py in <out>/map, resumed round to round. Seed 1, the aim's entry
   duel (--aim, lenia.prompts.map_prompt), seed 0's settings (4 islands, bodies x form x motion, 64 children, surprise
   archive), plus --surprise-px 484 (gate 0f) and --ledger. After round 1's map search, --baseline copies the map folder:
   the map-only baseline shares the full run's first 20 iterations exactly and continues alone to 100.
2. Triggers: candidates are every current map elite and every surprise-archive member (the archive only takes map
   rejects, so archive-only triggers would shut out worlds that are both good and surprising). Each is re-scored with
   the surprise duel (484 px) against a panel of --opponents (6): its --memory-k (2) nearest explored worlds (past
   expedition starts, switch stepping stones and ends; nearest by the map's DINOv2 form embedding of the last snapshot)
   and current-map elites for the rest. That is the memory that discourages repeats: novelty relative to what has been
   explored, as in novelty search's k-nearest neighbours, with no hard limits (a world that is itself an explored world
   meets itself and scores 0.5 there, without a call). The elite part of a panel is chosen per lineage by rendezvous
   hashing, so a lineage's members face the same elites (a paired pick within the lineage) and unchanged pairs hit the
   cache in later rounds. Best per lineage, the top --triggers (10); fewer lineages than that: the next best fill in.
3. Goals: namer_aim_v3 per trigger (frozen), with 4 random current-map elites as the typical worlds and the "names
   already given" list carried across rounds; then goal_caption_v1 for each goal's route.
4. Expeditions: arm NS-aim's design, unchanged: 10 steps (--end) of the 20-step mutation schedule (2.0 -> 0.8), 48
   children, the CLIP or motion-floor shortlist of 16, Jev's combined attain / complexity / surprise request against the
   reference, comma selection with the complexity floor 0.45 (COMPLEXITY), and one review_aim_v1 review after step 5
   (continue, refine, or switch to one of the 3 most surprising unchosen judged children of steps 3-5). Unlike NS-aim
   nothing is copied: every expedition runs its 10 steps.
5. Feedback: each expedition's end world (if it moved) enters the map in its trigger's island through the aim entry duel
   against its cell's elite (P >= 0.6, both orders), or takes the cell if empty. Its picture is kept either way
   (map/img/fb_r<round>_<rank>.png). Felix 11-04: the surprise-archive maps had less interesting elites than the first
   MAP-Elites runs, possibly because archive parents starve refinement; feedback adds refined worlds back.
Ledger: map/ledger.jsonl (every judged map world: rule, descriptors, the entry duel's opponent and P, placed or not;
every surprise judgment; every feedback duel), rounds/r*/rescore.csv (trigger re-scores), rounds/r*/exp/judged.jsonl
(every judged expedition child with its rule and scores). `ledger` joins them into one ledger.csv, one row per world.
Pictures of judged worlds that were never kept (map rejects outside the archive, unpicked expedition children) are
regrowable from their rule: the simulation is deterministic per rule and platform.

Gallery (`gallery`, also the last step of `run`):
- a Bradley-Terry rating per world from every aim entry duel in the ledger (both orders averaged into one fractional
  win; Hunter's MM algorithm, with one virtual half-won game against a fixed reference per world, so ratings stay finite);
- the cell champions (the best-rated world with a picture in each island cell) play a 4-round Swiss tournament of aim
  duels, which connects the ratings across cells; the ratings are refitted on everything;
- the top worlds, diversified: at most 2 per lineage and 1 per cell (across islands), 24 in all;
- Gemini (gallery_pick_v1) picks the final 8 from that shortlist, shown in random order, judging the set as a whole
  (in the spirit of Sam's Picbreeder score tracking). The same command on the baseline's folder gives its gallery.

From papers: MAP-Elites (Mouret & Clune 2015), novelty search's archive and k-nearest-neighbour novelty (Lehman & Stanley
2011), Bradley-Terry (Hunter 2004's MM algorithm), Swiss pairing, Picbreeder (Secretan et al. 2011). Ours: the rounds
loop, memory in the trigger panel, feedback of expedition ends into the map, the score ledger and the gallery pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mace_qd as QD
import stage1 as S1
from lenia import backend
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, MultiQuery, Query, load_env
from lenia.mace import TILE
from lenia.prompts import (AIM_ALIFE, CAPTION, COMPLEXITY, GALLERY, NAMER_AIM, REVIEW_AIM, SURPRISE, attain_prompt)
from lenia.sim import Rule

SCRIPTS = Path(__file__).resolve().parent
# fixed at setup (state.json "args"); --dry swaps in DRY's values
DEFAULTS = dict(rounds=5, iters=20, triggers=10, opponents=6, memory_k=2, surprise_px=484, end=S1.V4_AIM_END,
                review_at=S1.V4_REVIEW_AT, children=S1.V4_CHILDREN, judged=S1.V4_JUDGED, motion_floor=S1.V4_MOTION,
                k_floor=S1.K_FLOOR, review_kids=S1.V4_REVIEW_KIDS, judge="jev", vlm="gemini-pro",
                clip="openai/clip-vit-base-patch32", seed=1, aim=AIM_ALIFE)
DRY = dict(rounds=2, iters=2, triggers=3, children=8, judged=4, motion_floor=0.0, judge="fake", vlm="fake", clip="fake")
MAP_ARGS = ["--islands", "4", "--axes", "bodies,form,motion", "--body-edges", "3,9", "--motion-edges", "0.75,1.0,1.25",
            "--sarch", "--ledger"]                     # with mace_qd.py's defaults: seed 0's base-run settings
MAP_DRY = ["--embed-model", "fake", "--batch", "8", "--init", "8", "--size", "96", "--warmup", "200", "--render", "120",
           "--gap", "40"]
PHASES = ("map", "triggers", "goals", "exp", "feedback")
GALLERY_ROUNDS, GALLERY_SHORT, GALLERY_K, PER_ROOT = 4, 24, 8, 2


# ---------- shared bits ----------

def pic_of(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def tile(pic) -> np.ndarray:
    return np.asarray(pic)[-TILE:, -TILE:]


def read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def ledger_rows(run: Path) -> list:
    """map/ledger.jsonl, last row per (id, kind) (a crash mid-iteration can repeat rows on resume)."""
    last = {}
    for r in read_jsonl(run / "ledger.jsonl"):
        last[(r["id"], r["kind"])] = r
    return list(last.values())


class Ctx:
    """Settings, folders and lazily made judges, VLM, CLIP, embedder and simulation cache of one full run."""
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        self.F = Path(a.out)
        self.MAP = self.F / "map"
        self._j = self._d = self._clip = self._ev = None

    def mstate(self) -> dict:
        return json.loads((self.MAP / "state.json").read_text())

    def judge(self) -> Judge:
        if self._j is None:
            c = load(self.a.config)
            load_env(REPO_ROOT / ".env")
            self._j = Judge(self.cfg.judge, {**c["judges"]["defaults"], **c["judges"][self.cfg.judge],
                                             "max_run_usd": self.a.max_usd}, self.F / "judge_cache.sqlite", yes=True)
        return self._j

    def vlm(self):
        if self._d is None:
            from lenia.describe import Describer
            c = load(self.a.config)
            load_env(REPO_ROOT / ".env")
            self._d = Describer(self.cfg.vlm, c["vlms"][self.cfg.vlm], self.F / "describe_cache.sqlite",
                                max_usd=self.a.max_vlm_usd)
        return self._d

    def clip(self):
        if self._clip is None:
            from lenia.embed import ClipScorer
            self._clip = ClipScorer(self.cfg.clip)
        return self._clip

    def ea(self):
        return S1.eval_settings(self.MAP, argparse.Namespace(sim_batch=self.a.sim_batch))

    def evals(self) -> "S1.Evals":
        if self._ev is None:
            self._ev = S1.Evals(Path(self.a.evals) if self.a.evals else self.F / "evals", self.ea(), keep=self.F / "evals")
        return self._ev


def embed_tiles(C: Ctx, items: list) -> dict:
    """{key: unit vector}: the map's form embedder (DINOv2) on each picture's last snapshot; items [(key, path)].
    Cached in <out>/emb.npz by key (keys are paths relative to <out>, which never change content)."""
    f = C.F / "emb.npz"
    have = dict(np.load(f)) if f.exists() else {}
    todo = [(k, p) for k, p in items if k not in have]
    if todo:
        model = QD.embedder(QD.settings(C.mstate()).embed_model)
        vs = model([Image.fromarray(tile(pic_of(p))) for _, p in todo])
        for (k, _), v in zip(todo, vs):
            v = np.asarray(v, np.float64)
            have[k] = v / max(np.linalg.norm(v), 1e-12)
        np.savez(f, **have)
    return {k: have[k] for k, _ in items}


def pick_duels(judge, prompt: str, pairs: list, pics: dict, desc: str) -> list:
    """[(world, opponent)] -> P(world chosen), both orders; identical ids score 0.5 without a call."""
    todo = [(w, o) for w, o in pairs if w != o]
    got = iter(S1.duels(judge, prompt, todo, pics, desc) if todo else [])
    return [0.5 if w == o else next(got) for w, o in pairs]


def sheet(rows: list, path: Path, cols: int = 5, px: int = 240, lines: int = 2) -> None:
    """A contact sheet: rows = [(picture path, [label lines])]."""
    if not rows:
        return
    f = S1._font(12)
    h = px + 6 + 15 * lines
    im = Image.new("RGB", (cols * (px + 8) + 8, ((len(rows) + cols - 1) // cols) * (h + 8) + 8), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for n, (p, lab) in enumerate(rows):
        x, y = 8 + (n % cols) * (px + 8), 8 + (n // cols) * (h + 8)
        for i, t in enumerate(lab[:lines]):
            d.text((x, y + 15 * i), str(t)[:44], fill="black" if i == 0 else (90, 90, 90), font=f)
        im.paste(Image.open(p).convert("RGB").resize((px, px), Image.BILINEAR), (x, y + 15 * lines + 4))
    im.save(path)


# ---------- 1. map search ----------

def map_phase(C: Ctx, st: dict, r: int) -> None:
    cfg, a = C.cfg, C.a
    target = r * cfg.iters
    if (C.MAP / "state.json").exists() and C.mstate()["it"] > target:
        print(f"round {r}: the map search is already past iteration {target}", file=sys.stderr)
    else:
        cmd = [sys.executable, str(SCRIPTS / "mace_qd.py"), "run", "--out", str(C.MAP), "--iterations", str(target),
               "--backend", a.backend, "--judge", cfg.judge, "--max-usd", str(a.map_max_usd)]
        cmd += ["--sim-batch", str(a.sim_batch)] if a.sim_batch else []
        cmd += ["--config", a.config] if a.config else []
        if not (C.MAP / "state.json").exists():
            cmd += MAP_ARGS + ["--seed", str(cfg.seed), "--aim", cfg.aim, "--surprise-px", str(cfg.surprise_px)]
            cmd += MAP_DRY if cfg.dry else []
        print(f"round {r}: map search to iteration {target}", file=sys.stderr)
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode:
            sys.exit("the map search stopped; re-run this command to resume it")
    if r == 1 and a.baseline:
        B = Path(a.baseline)
        if (B / "state.json").exists():
            print(f"baseline {B} already exists; not forked again", file=sys.stderr)
        else:
            shutil.copytree(C.MAP, B, dirs_exist_ok=True)
            (B / "FORKED.md").write_text(f"Forked from {C.MAP} after round 1's map search (iteration {target}), before any "
                                         "trigger, expedition or feedback. Continue it alone with mace_qd.py run "
                                         f"--out {B} --iterations {cfg.rounds * cfg.iters}.\n")
            st["baseline"] = {"path": str(B), "forked_at": target}
            print(f"baseline forked to {B} at iteration {target}", file=sys.stderr)


# ---------- 2. triggers ----------

def candidates(sm: dict) -> list:
    out = []
    for k, m in enumerate(QD.islands_of(sm)):
        for ck, e in sorted(m.items()):
            out.append({"wid": Path(e["image"]).stem, "image": e["image"], "root": e.get("root"), "island": k, "cell": ck,
                        "src": e.get("source", "map"), "it": e["it"], "rule": e["rule"], "texture": e["texture"],
                        "bodies": e["bodies"], "motion": e["motion"], "coverage": e.get("coverage")})
    for e in sm.get("sarch", []):
        out.append({"wid": Path(e["image"]).stem, "image": e["image"], "root": e.get("root"), "island": e["island"],
                    "cell": None, "src": "archive", "it": e["it"], "rule": e["rule"], "texture": e["texture"],
                    "bodies": e["bodies"], "motion": e["motion"], "coverage": e.get("coverage"), "stored": e["surprise"]})
    return out


def rendezvous(pool: list, key: str, n: int, exclude: str) -> list:
    order = sorted(pool, key=lambda x: hashlib.sha1(f"{key}|{x}".encode()).hexdigest())
    return [x for x in order if x != exclude][:n]


def triggers_phase(C: Ctx, st: dict, r: int, R: Path) -> None:
    cfg = C.cfg
    sm = C.mstate()
    am = QD.settings(sm)
    cands = candidates(sm)
    elites = [c["wid"] for c in cands if c["src"] != "archive"]
    mem = st["memory"]
    paths = {c["wid"]: C.MAP / "img" / c["image"] for c in cands}
    paths.update({m["id"]: C.F / m["image"] for m in mem})
    vec = embed_tiles(C, [(f"map/img/{c['image']}", paths[c["wid"]]) for c in cands] +
                      [(m["image"], C.F / m["image"]) for m in mem]) if mem else {}
    M = np.stack([vec[m["image"]] for m in mem]) if mem else None
    panels = {}
    for c in cands:
        near = []
        if mem:
            sims = M @ vec[f"map/img/{c['image']}"]
            near = [mem[int(j)]["id"] for j in np.argsort(-sims)[:min(cfg.memory_k, len(mem))]]
        panels[c["wid"]] = near + rendezvous(elites, f"{cfg.seed}|{c['root']}", cfg.opponents - len(near), c["wid"])
    need = {c["wid"] for c in cands} | {o for p in panels.values() for o in p}
    pics = {w: QD.small(pic_of(paths[w]), cfg.surprise_px) for w in need}
    pairs = [(c["wid"], o) for c in cands for o in panels[c["wid"]]]
    ps = pick_duels(C.judge(), SURPRISE, pairs, pics, f"round {r} triggers")
    if any(p is None for p in ps):
        sys.exit(f"{sum(p is None for p in ps)} trigger duels failed (not cached); re-run to retry them")
    memids = {m["id"] for m in mem}
    rows, n = [], 0
    for c in cands:
        v = ps[n:n + len(panels[c["wid"]])]
        n += len(v)
        nm = [p for o, p in zip(panels[c["wid"]], v) if o in memids]
        rows.append({**c, "surprise": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4),
                     "vs_memory": round(float(np.mean(nm)), 4) if nm else None, "panel": panels[c["wid"]]})
    order = sorted(rows, key=lambda x: -x["surprise"])
    best = {}
    for x in order:
        best.setdefault(x["root"], x)
    picks = sorted(best.values(), key=lambda x: -x["surprise"])[:cfg.triggers]
    if len(picks) < cfg.triggers:                          # fewer lineages than triggers: the next best fill in
        chosen = {x["wid"] for x in picks}
        picks += [x for x in order if x["wid"] not in chosen][:cfg.triggers - len(picks)]
    (C.F / "img").mkdir(exist_ok=True)
    T = []
    for rank, x in enumerate(picks):
        if not (C.F / "img" / x["image"]).exists():
            shutil.copy(paths[x["wid"]], C.F / "img" / x["image"])
        T.append({**x, "id": f"r{r}-{rank:02d}-L{x['root']}", "rank": rank, "start": f"img/{x['image']}"})
    with open(R / "rescore.csv", "w", newline="") as fh:
        cols = ["wid", "src", "root", "island", "cell", "it", "surprise", "sd", "vs_memory", "texture", "bodies", "motion"]
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(order)
    arch = [c for c in cands if c["src"] == "archive"]
    stats = {"map_it": sm["it"] - 1, "elites": len(elites), "lineages": QD.lineages(sm), "archive": len(arch),
             "archive_texture_median": round(float(np.median([c["texture"] for c in arch])), 4) if arch else None,
             "elite_texture_median": round(float(np.median([c["texture"] for c in cands if c["src"] != "archive"])), 4),
             "trigger_texture_median": round(float(np.median([t["texture"] for t in T])), 4),
             "trigger_sources": dict(Counter(t["src"] for t in T)), "memory": len(mem),
             "trigger_surprise_mean": round(float(np.mean([t["surprise"] for t in T])), 4),
             "candidates": len(cands), "duels": len(pairs), "map_threshold": sm.get("sarch_thr")}
    (R / "triggers.json").write_text(json.dumps({"round": r, "prompt": SURPRISE, "px": cfg.surprise_px,
                                                 "stats": stats, "triggers": T}, indent=1))
    sheet([(C.F / t["start"], [f"{t['id']} {t['src']}", f"surprise {t['surprise']:.2f}  tex {t['texture']:.2f}"])
           for t in T], R / "triggers.png")
    st["rounds"].setdefault(str(r), {})["triggers"] = stats
    print(f"round {r}: {len(T)} triggers from {len(cands)} candidates ({stats['archive']} in the archive, archive texture "
          f"median {stats['archive_texture_median']}, triggers {stats['trigger_texture_median']}); sources "
          f"{stats['trigger_sources']}; judge spent ${C.judge().spent:.4f}", file=sys.stderr)


# ---------- 3. goals ----------

def goals_phase(C: Ctx, st: dict, r: int, R: Path) -> None:
    from lenia.describe import Ask
    cfg = C.cfg
    T = json.loads((R / "triggers.json").read_text())["triggers"]
    sm = C.mstate()
    elites = [e["image"] for m in QD.islands_of(sm) for _, e in sorted(m.items())]
    d = C.vlm()
    names = list(st["names"])                 # saved only when the phase ends, so a resumed phase asks the same
    goals = []
    for t in T:                               # sequential: each call sees the names given so far (across rounds)
        rng = np.random.default_rng([cfg.seed, r, t["rank"], 7])
        others = [x for x in elites if x != t["image"]]
        typ = [others[int(j)] for j in rng.choice(len(others), 4, replace=False)]
        fill = {"names": "\n".join(f"- {x}" for x in names) if names else "(none yet)", "aim": cfg.aim}
        [ans] = d.ask_many([Ask(NAMER_AIM, [pic_of(C.F / t["start"])] + [pic_of(C.MAP / "img" / x) for x in typ], fill)],
                           desc=t["id"])
        if not ans.ok:
            sys.exit(f"naming failed at {t['id']}: {ans.raw[:300]} (re-run to retry)")
        names.append(ans.data["sees"])
        g = {"id": S1.goal_id(ans.data["sees"], ans.data["push"]), "sees": ans.data["sees"], "push": ans.data["push"],
             "source": f"namer:{d.model}", "prompt": NAMER_AIM, "aim": cfg.aim, "cost": round(ans.cost, 6), "typical": typ}
        if isinstance(ans.data.get("features"), list):
            g["features"] = [str(x) for x in ans.data["features"]]
        goals.append(g)
    caps = d.ask_many([Ask(CAPTION, [pic_of(C.F / t["start"])], {"sees": g["sees"], "push": g["push"]})
                       for t, g in zip(T, goals)], desc=f"round {r} captions")
    if any(not c.ok for c in caps):
        sys.exit("some caption calls failed (not cached); re-run to retry them")
    rows = []
    for t, g, c in zip(T, goals, caps):
        end = S1._caption(c.data["end"])
        cap = {"end": end, "start": "none" if end == "none" else S1._caption(c.data["start"])}
        rows.append({"trigger": t["id"], **g, "caption": cap, "route": "clip" if end != "none" else "floor",
                     "target": end if end != "none" else g["push"], "flags": S1._goal_flags(g)})
    (R / "goals.json").write_text(json.dumps(rows, indent=1))
    L = [f"# Round {r} goals ({NAMER_AIM}, {CAPTION}, {cfg.vlm})\n", f"Aim: \"{cfg.aim}\"\n",
         "| trigger | push | sees | route: target | flags |", "|---|---|---|---|---|"]
    L += [f"| {x['trigger']} | {x['push']} | {x['sees']} | {x['route']}: {x['target']} | {', '.join(x['flags'])} |" for x in rows]
    (R / "goals.md").write_text("\n".join(L) + "\n")
    st["names"] = names
    print("\n".join(L), file=sys.stderr)


# ---------- 4. expeditions (arm NS-aim's step and review, stage1.py v4) ----------

def exp_setup(C: Ctx, r: int, R: Path) -> dict:
    cfg = C.cfg
    T = json.loads((R / "triggers.json").read_text())["triggers"]
    G = {g["trigger"]: g for g in json.loads((R / "goals.json").read_text())}
    trs = []
    for t in T:
        g = G[t["id"]]
        trs.append({"id": t["id"], "index": t["rank"], "rank": t["rank"], "island": t["island"], "root": t["root"],
                    "start_id": t["wid"], "src": t["src"], "start": t["start"], "start_rule": t["rule"],
                    "goal": {k: g[k] for k in ("id", "sees", "push", "source", "prompt", "aim") if k in g},
                    "caption": g["caption"], "route": g["route"], "target": g["target"], "parent_rule": t["rule"],
                    "parent": t["start"], "g": 0.5, "k": 0.5, "s": 0.5, "history": [], "stats": []})
    return {"arm": "full", "version": "fullrun", "round": r, "step": 0, "seed": cfg.seed, "steps": S1.V4_STEPS,
            "end": cfg.end, "aim": cfg.aim, "namer": NAMER_AIM, "children": cfg.children, "judged": cfg.judged,
            "sched": list(S1.V4_SCHED), "motion_floor": cfg.motion_floor, "k_floor": cfg.k_floor,
            "review_at": cfg.review_at, "reviewed": False, "triggers": trs}


def exp_rng(est: dict, tr: dict, step: int, *extra) -> np.random.Generator:
    return np.random.default_rng([est["seed"], 1000 + est["round"], tr["rank"], step, *extra])


def exp_step(C: Ctx, est: dict, E: Path) -> None:
    """One step of every expedition: v4's step (stage1.expedition_v4) with its arm-C branches removed."""
    ea, evals, clip, judge = C.ea(), C.evals(), C.clip(), C.judge()
    step, t0 = est["step"] + 1, time.time()
    scale = S1.v4_scale(step, est["steps"])
    rules, owner = [], []
    for tr in est["triggers"]:
        for m in S1.mutants_of(Rule.from_dict(tr["parent_rule"]), exp_rng(est, tr, step), est["children"], ea, scale, 0.0):
            rules.append(m)
            owner.append(tr["rank"])
    ev = evals.get_many(rules)
    emb = clip.images([tile(v["pic"]) for v in ev])
    byi = {tr["rank"]: tr for tr in est["triggers"]}
    refs = {i: pic_of(C.F / tr.get("ref", tr["start"])) for i, tr in byi.items()}
    short, csc = {}, {}
    for i, tr in byi.items():
        ks = [j for j, o in enumerate(owner) if o == i]
        cs = emb[ks] @ clip.text(tr["target"])
        csc.update(zip(ks, cs.tolist()))
        fl = [(ea.max_texture <= 0 or ev[j]["texture"] <= ea.max_texture) and ev[j]["motion"] >= est["motion_floor"] for j in ks]
        if tr["route"] == "clip":
            order = [ks[n] for n in np.argsort(-cs) if fl[n]]
        else:
            rr = exp_rng(est, tr, step, 7).permutation(len(ks))
            order = [ks[n] for n in rr if fl[n]]
        short[i] = order[:est["judged"]]
    mq, meta = [], []
    for i, js in short.items():
        pids = (attain_prompt(byi[i]["goal"]), COMPLEXITY, SURPRISE)
        for j in js:
            mq += [MultiQuery(pids, ev[j]["pic"], refs[i]), MultiQuery(pids, refs[i], ev[j]["pic"])]
            meta.append(j)
    vs = judge.judge_multi(mq, desc=f"round {est['round']} step {step}") if mq else []
    if any(not all(v.ok for v in vv) for vv in vs):
        sys.exit("some judge requests failed (not cached); re-run to retry them (the step restarts)")
    both = lambda x, y, q: 0.5 * (x[q].p_first + 1 - y[q].p_first)
    jv = {j: tuple(both(vs[2 * n], vs[2 * n + 1], q) for q in range(3)) for n, j in enumerate(meta)}
    rows, judged = [], []
    for i, tr in byi.items():
        ks = [j for j, o in enumerate(owner) if o == i]
        cand = [(jv[j][0], jv[j][1], j) for j in short[i] if jv[j][1] >= est["k_floor"]]
        pick = max(cand)[2] if cand else None
        valid = [j for j in ks if ea.max_texture <= 0 or ev[j]["texture"] <= ea.max_texture]
        tr["stats"].append({"step": step, "scale": round(scale, 4), "valid": len(valid),
                            "floor": sum(ev[j]["motion"] >= est["motion_floor"] for j in valid), "judged": len(short[i]),
                            "clip_mean": round(float(np.mean([csc[j] for j in valid])), 4) if valid else None,
                            "motion_mean": round(float(np.mean([ev[j]["motion"] for j in valid])), 4) if valid else None,
                            "jev_mean_judged": round(float(np.mean([jv[j][0] for j in short[i]])), 4) if short[i] else None,
                            "moved": pick is not None})
        for j in ks:
            g, k, s_ = jv.get(j, (None, None, None))
            rows.append([step, tr["id"], ks.index(j), round(ev[j]["texture"], 4), round(ev[j]["motion"], 4), round(csc[j], 4),
                         int(j in short[i]), *(None if v is None else round(v, 4) for v in (g, k, s_)), int(j == pick)])
            if j in short[i]:
                judged.append({"round": est["round"], "step": step, "trigger": tr["id"], "child": ks.index(j),
                               "key": ev[j]["key"], "rule": rules[j].to_dict(), "goal": round(g, 4), "complexity": round(k, 4),
                               "surprise": round(s_, 4), "picked": int(j == pick), "clip": round(csc[j], 4),
                               "bodies": ev[j]["bodies"], "motion": round(ev[j]["motion"], 4),
                               "texture": round(ev[j]["texture"], 4)})
        if pick is not None:
            evals.persist(ev[pick])
            tr["parent_rule"], tr["parent"] = rules[pick].to_dict(), ev[pick]["image"]
            tr["g"], tr["k"], tr["s"] = (round(v, 4) for v in jv[pick])
            h = {"step": step, "image": ev[pick]["image"], "goal_score": tr["g"], "complexity": tr["k"], "surprise": tr["s"],
                 "clip": round(csc[pick], 4), "motion": round(ev[pick]["motion"], 4), "bodies": ev[pick]["bodies"]}
            tr["history"].append(h)
            tr["_trace"] = {"round": est["round"], "trigger": tr["id"], **h, "rule": rules[pick].to_dict()}
    est["step"] = step
    traces = [tr.pop("_trace") for tr in est["triggers"] if "_trace" in tr]
    QD.save_state(E, est)                     # state first, then the logs (a crash re-runs the step from its caches)
    new = not (E / "duels.csv").exists()
    with open(E / "duels.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["step", "trigger", "child", "texture", "motion", "clip", "shortlisted", "goal", "complexity",
                        "surprise", "picked"])
        w.writerows(rows)
    with open(E / "judged.jsonl", "a") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in judged)
    with open(E / "trace.jsonl", "a") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in traces)
    print(f"round {est['round']} step {step}/{est['end']} (scale {scale:.2f}): {len(rules)} children in "
          f"{time.time() - t0:.0f}s; moved {sum(tr['stats'][-1]['moved'] for tr in est['triggers'])}/{len(est['triggers'])}; "
          f"judge spent ${judge.spent:.4f}", file=sys.stderr)


def exp_review(C: Ctx, est: dict, E: Path) -> None:
    """The one review_aim_v1 review (stage1.v4_review's decisions, minus NS-aim's copying of N-aim's continues)."""
    from lenia.describe import Ask
    at = est["review_at"]
    judged = {}
    for x in read_jsonl(E / "judged.jsonl"):           # last row per child (a repeated step rewrites its rows)
        judged[(x["trigger"], x["step"], x["child"])] = x
    ev_all = C.evals()
    kids, asks, live = {}, [], []
    for tr in est["triggers"]:
        cand = [x for x in judged.values() if x["trigger"] == tr["id"] and at - 2 <= x["step"] <= at and not x["picked"]]
        top = sorted(cand, key=lambda x: -x["surprise"])[:C.cfg.review_kids]
        if len(top) < C.cfg.review_kids:
            tr["review"] = {"decision": "none", "applied": "continue", "image": None,
                            "why": f"not reviewed: only {len(top)} judged unchosen children in steps {at - 2}-{at}"}
            continue
        got = ev_all.get_many([Rule.from_dict(x["rule"]) for x in top])
        for e in got:
            ev_all.persist(e)
        kids[tr["id"]] = (top, got)
        byk = {h["step"]: h for h in tr["history"]}
        prog = "; ".join(f"generation {k}: {byk[k]['goal_score']:.2f}" if k in byk else f"generation {k}: no child passed"
                         for k in range(1, at + 1))
        asks.append(Ask(REVIEW_AIM, [pic_of(C.F / tr["start"]), pic_of(C.F / tr["parent"])] + [e["pic"] for e in got],
                        {"aim": est["aim"], "children": est["children"], "judged": est["judged"], "gens": at,
                         "total": est["end"], "left": est["end"] - at, "sees": tr["goal"]["sees"],
                         "push": tr["goal"]["push"], "progress": prog}))
        live.append(tr)
    d = C.vlm()
    ans = d.ask_many(asks, desc=f"round {est['round']} review") if asks else []
    if any(not x.ok for x in ans):
        sys.exit("some review calls failed (not cached); re-run to retry them")
    todo = []
    forced = os.environ.get("FULLRUN_FAKE_REVIEW", "").split(",") if C.cfg.vlm == "fake" else [""]
    for n, (tr, x) in enumerate(zip(live, ans)):
        top, got = kids[tr["id"]]
        data = dict(x.data)
        if forced != [""]:                    # dry runs only: e.g. FULLRUN_FAKE_REVIEW=switch:4,refine,continue
            f = forced[n % len(forced)].split(":")
            data.update(decision=f[0], image=f[1] if len(f) > 1 else None, push="make it " + f[0])
        dec = data["decision"].strip().lower()
        img = data.get("image")
        img = int(img) if str(img).strip() in ("3", "4", "5") else None
        eff = dec if dec in ("continue", "refine") or (dec == "switch" and img) else "continue"
        tr["review"] = {"decision": dec, "applied": eff, "image": img, "why": data.get("why"), "sees": data["sees"],
                        "push": data["push"], "cost": round(x.cost, 6), "parent": tr["parent"], "g_before": tr["g"],
                        "children": [{"step": c["step"], "child": c["child"], "surprise": c["surprise"], "image": e["image"]}
                                     for c, e in zip(top, got)]}
        if eff == "continue":
            continue
        sees = tr["goal"]["sees"] if eff == "refine" else data["sees"]
        g = {"id": S1.goal_id(sees, data["push"]), "sees": sees, "push": data["push"], "source": f"review:{eff}",
             "from_step": at + 1}
        tr["goals_before"] = tr.get("goals_before", []) + [tr["goal"]]
        tr["goal"] = g
        if eff == "switch":
            c, e = top[img - 3], got[img - 3]
            tr["parent_rule"], tr["parent"] = c["rule"], e["image"]
            tr["ref"], tr["ref_rule"], tr["ref_key"] = e["image"], c["rule"], e["key"]
            tr["g"], tr["k"], tr["s"] = 0.5, 0.5, 0.5
        todo.append((tr, g))
    if todo:                                  # captions for the new goals: they set the route and the CLIP target
        cs = d.ask_many([Ask(CAPTION, [pic_of(C.F / tr["parent"])], {"sees": g["sees"], "push": g["push"]}) for tr, g in todo],
                        desc=f"round {est['round']} review captions")
        if any(not c.ok for c in cs):
            sys.exit("some caption calls failed (not cached); re-run to retry them")
        for (tr, g), c in zip(todo, cs):
            end = S1._caption(c.data["end"])
            tr["caption"] = {"end": end, "start": "none" if end == "none" else S1._caption(c.data["start"])}
            tr["route"] = "clip" if end != "none" else "floor"
            tr["target"] = end if end != "none" else g["push"]
    est["reviewed"] = True
    L = [f"# Round {est['round']} review after step {at} ({REVIEW_AIM}, {C.cfg.vlm})\n",
         "| trigger | decision | image | goal before | goal after | why |", "|---|---|---|---|---|---|"]
    for tr in est["triggers"]:
        v = tr["review"]
        before = (tr.get("goals_before") or [tr["goal"]])[-1]["push"]
        L.append(f"| {tr['id']} | {v['applied']}{'' if v['applied'] == v['decision'] else ' (asked ' + v['decision'] + ')'} | "
                 f"{v['image'] or '-'} | {before} | {tr['goal']['push'] if v['applied'] != 'continue' else '='} | {v['why']} |")
    (E / "review.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), file=sys.stderr)


def exp_phase(C: Ctx, st: dict, r: int, R: Path) -> None:
    E = R / "exp"
    E.mkdir(exist_ok=True)
    if (E / "state.json").exists():
        est = json.loads((E / "state.json").read_text())
        print(f"round {r}: expeditions resume after step {est['step']}", file=sys.stderr)
    else:
        est = exp_setup(C, r, R)
        QD.save_state(E, est)
        for tr in est["triggers"]:
            print(f"{tr['id']} [{tr['route']}] {tr['goal']['push']}  ->  CLIP target: {tr['target']}", file=sys.stderr)
    while est["step"] < est["end"]:
        if est["step"] == est["review_at"] and not est["reviewed"]:
            exp_review(C, est, E)
            QD.save_state(E, est)
        exp_step(C, est, E)
    S1.v4_sheet(C.F, est, E / "sheet.png")
    S1.v4_summary(est, E / "summary.md")
    st["rounds"].setdefault(str(r), {})["expeditions"] = {
        "moved": sum(bool(tr["history"]) for tr in est["triggers"]),
        "review": dict(Counter(tr.get("review", {}).get("applied", "-") for tr in est["triggers"])),
        "routes": dict(Counter(tr["route"] for tr in est["triggers"])),
        "goal_mean": round(float(np.mean([tr["g"] for tr in est["triggers"]])), 4),
        "complexity_mean": round(float(np.mean([tr["k"] for tr in est["triggers"]])), 4)}


# ---------- 5. feedback ----------

def feedback_phase(C: Ctx, st: dict, r: int, R: Path) -> None:
    est = json.loads((R / "exp" / "state.json").read_text())
    sm = C.mstate()
    am = QD.settings(sm)
    done = r in sm.get("feedback_rounds", [])
    if done:                                  # the map already took this round's feedback (a crash after its save)
        fb = json.loads((R / "feedback.json").read_text())
    else:
        pid, arches = QD.entry_prompt(am), QD.islands_of(sm)
        ends = [tr for tr in sorted(est["triggers"], key=lambda t: t["rank"]) if tr["history"]]
        evs = C.evals().get_many([Rule.from_dict(tr["parent_rule"]) for tr in ends]) if ends else []
        if "form" in am.axes and evs:
            model = QD.embedder(am.embed_model)
            mu, pc = np.asarray(sm["form"]["mean"]), np.asarray(sm["form"]["pc"])
            for e, v in zip(evs, model([Image.fromarray(tile(e["pic"])) for e in evs])):
                e["form"] = float((np.asarray(v, np.float64) - mu) @ pc)
        fb, lrows = [], []
        for tr in sorted(est["triggers"], key=lambda t: t["rank"]):
            if not tr["history"]:
                fb.append({"trigger": tr["id"], "outcome": "never moved"})
                continue
        for tr, e in zip(ends, evs):
            name = f"fb_r{r}_{tr['rank']:02d}"
            Image.fromarray(e["pic"]).save(C.MAP / "img" / f"{name}.png")
            k = tr["island"]
            arch = arches[k]
            rec = {"trigger": tr["id"], "id": name, "image": f"map/img/{name}.png", "island": k, "key": e["key"],
                   "bodies": e["bodies"], "motion": round(e["motion"], 4), "texture": round(e["texture"], 4)}
            if am.max_texture > 0 and e["texture"] > am.max_texture:
                fb.append({**rec, "outcome": "grid noise"})
                continue
            ck = QD.cell_of(e, am, sm)
            if ck in arch:
                inc = arch[ck]["image"]
                [p] = S1.duels(C.judge(), pid, [(name, inc)], {name: e["pic"], inc: pic_of(C.MAP / "img" / inc)},
                               f"round {r} feedback {tr['id']}")
                if p is None:
                    sys.exit("a feedback duel failed (not cached); re-run to retry it")
                acc, kind, opp = p >= am.threshold, "duel", Path(inc).stem
            else:
                p, acc, kind, opp = None, True, "new", None
            prev = arch.get(ck)
            if acc:
                arch[ck] = {"rule": tr["parent_rule"], "image": f"{name}.png", "it": sm["it"] - 1, "parent": f"fb:{tr['id']}",
                            "root": tr["root"], "bodies": e["bodies"], "motion": round(e["motion"], 4),
                            "texture": round(e["texture"], 4), "coverage": round(e["coverage"], 4),
                            "p": None if p is None else round(p, 4),
                            "replaced": 0 if prev is None else prev.get("replaced", 0) + 1, "source": "feedback",
                            "trigger": tr["id"], "round": r, **({"form": round(e["form"], 4)} if "form" in e else {})}
            fb.append({**rec, "cell": ck, "outcome": "new cell" if kind == "new" else ("replaced" if acc else "lost"),
                       "p": None if p is None else round(p, 4), "opponent": opp})
            lrows.append({"id": name, "source": "feedback", "it": sm["it"] - 1, "island": k, "n": tr["rank"], "kind": kind,
                          "cell": ck, "opponent": opp, "p": None if p is None else round(p, 4), "accepted": int(acc),
                          "placed": int(acc), "parent": f"fb:{tr['id']}", "parent_id": tr["start_id"], "root": tr["root"],
                          "rule": tr["parent_rule"], "bodies": e["bodies"], "motion": round(e["motion"], 4),
                          "texture": round(e["texture"], 4), "coverage": round(e["coverage"], 4), "round": r,
                          "trigger": tr["id"], **({"form": round(e["form"], 4)} if "form" in e else {})})
        (R / "feedback.json").write_text(json.dumps(fb, indent=1))
        QD.write_ledger(C.MAP, lrows)
        sm["feedback_rounds"] = sm.get("feedback_rounds", []) + [r]
        (C.MAP / "snapshots" / f"feedback_r{r}.json").write_text(json.dumps(QD.islands_of(sm)))
        QD.save_state(C.MAP, sm)
    ids = {m["id"] for m in st["memory"]}     # memory: what this round explored
    fbby = {x["trigger"]: x for x in fb if "id" in x}
    for tr in est["triggers"]:
        new = [{"id": tr["start_id"], "image": tr["start"], "kind": "start"}]
        if tr.get("ref"):
            new.append({"id": tr["ref_key"], "image": tr["ref"], "kind": "switch"})
        if tr["id"] in fbby:
            new.append({"id": fbby[tr["id"]]["id"], "image": fbby[tr["id"]]["image"], "kind": "end"})
        for m in new:
            if m["id"] not in ids:
                st["memory"].append({**m, "round": r, "trigger": tr["id"]})
                ids.add(m["id"])
    st["rounds"].setdefault(str(r), {})["feedback"] = dict(Counter(x["outcome"] for x in fb))
    print(f"round {r} feedback: {dict(Counter(x['outcome'] for x in fb))}; memory {len(st['memory'])} worlds",
          file=sys.stderr)


# ---------- the loop ----------

def resolve(a) -> argparse.Namespace:
    base = {**DEFAULTS, **(DRY if a.dry else {})}
    return argparse.Namespace(**{k: getattr(a, k) if getattr(a, k, None) is not None else v for k, v in base.items()},
                              dry=a.dry)


def run(a) -> None:
    F = Path(a.out)
    F.mkdir(parents=True, exist_ok=True)
    if (F / "state.json").exists():
        st = json.loads((F / "state.json").read_text())
        cfg = argparse.Namespace(**st["args"])
        print(f"resuming: round {st['round']}, phase {st['phase']}", file=sys.stderr)
    else:
        cfg = resolve(a)
        st = {"args": vars(cfg), "round": 1, "phase": "map", "names": [], "memory": [], "rounds": {},
              "started": time.strftime("%Y-%m-%d %H:%M")}
        QD.save_state(F, st)
    backend.configure(a.backend)
    C = Ctx(a, cfg)
    fns = {"map": lambda r, R: map_phase(C, st, r), "triggers": lambda r, R: triggers_phase(C, st, r, R),
           "goals": lambda r, R: goals_phase(C, st, r, R), "exp": lambda r, R: exp_phase(C, st, r, R),
           "feedback": lambda r, R: feedback_phase(C, st, r, R)}
    while st["round"] <= cfg.rounds:
        r, ph = st["round"], st["phase"]
        R = F / "rounds" / f"r{r}"
        R.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        fns[ph](r, R)
        st["rounds"].setdefault(str(r), {}).setdefault("sec", {})[ph] = round(time.time() - t0, 1)
        nxt = PHASES.index(ph) + 1
        st["round"], st["phase"] = (r, PHASES[nxt]) if nxt < len(PHASES) else (r + 1, PHASES[0])
        QD.save_state(F, st)
        if a.stop_after and a.stop_after == f"{r}:{ph}":
            print(f"stopped after round {r}, phase {ph} (--stop-after)", file=sys.stderr)
            return
    if not a.no_gallery:
        gallery(argparse.Namespace(**{**vars(a), "run": str(C.MAP), "out": str(F)}), cfg)
    report(argparse.Namespace(out=str(F), baseline=a.baseline))


# ---------- gallery ----------

def bt_fit(edges: list, ids: list, prior: float = 1.0, iters: int = 5000, tol: float = 1e-9) -> dict:
    """Bradley-Terry log-strengths by Hunter's (2004) MM algorithm. edges: [(i, j, p)], p = P(i preferred over j) from one
    both-order duel, counted as a fractional win. Every world also plays `prior` virtual games against a fixed reference
    of strength 1 and wins half of them, so a world with one result (or none) gets a finite, shrunk rating."""
    idx = {x: n for n, x in enumerate(ids)}
    n = len(ids)
    I = np.array([idx[i] for i, _, _ in edges], int)
    J = np.array([idx[j] for _, j, _ in edges], int)
    P = np.array([p for *_, p in edges], float)
    W = np.full(n, prior / 2)
    np.add.at(W, I, P)
    np.add.at(W, J, 1 - P)
    pi = np.ones(n)
    for _ in range(iters):
        d = 1.0 / (pi[I] + pi[J])
        D = prior / (pi + 1.0)
        np.add.at(D, I, d)
        np.add.at(D, J, d)
        new = W / D
        done = np.max(np.abs(np.log(new) - np.log(pi))) < tol
        pi = new
        if done:
            break
    return {x: float(np.log(pi[idx[x]])) for x in ids}


def swiss(S: list, rating: dict, rounds: int, rng, play) -> list:
    """A Swiss tournament over S: round 1 random pairs, later rounds pair neighbours by points (then rating), no rematch
    where avoidable. play([(x, y)]) -> [P(x preferred) or None]. Returns [(x, y, p, round)]."""
    pts, met, out = {x: 0.0 for x in S}, set(), []
    for t in range(rounds):
        order = [S[int(j)] for j in rng.permutation(len(S))] if t == 0 else sorted(S, key=lambda x: (-pts[x], -rating[x]))
        pairs, free = [], list(order)
        while len(free) >= 2:
            x = free.pop(0)
            j = next((n for n, y in enumerate(free) if frozenset((x, y)) not in met), 0)
            y = free.pop(j)
            pairs.append((x, y))
            met.add(frozenset((x, y)))
        for (x, y), p in zip(pairs, play(pairs, t + 1)):
            if p is not None:
                pts[x] += p
                pts[y] += 1 - p
                out.append((x, y, p, t + 1))
    return out


def ancestry(rows: list):
    """A function: world id -> the fed-back expedition ends among its ancestors (nearest first), via parent_id."""
    par = {x["id"]: x.get("parent_id") for x in rows if x["kind"] != "surprise"}

    def fb(i):
        seen, out = set(), []
        while i and i not in seen:
            seen.add(i)
            if str(i).startswith("fb_"):
                out.append(i)
            i = par.get(i)
        return out
    return fb


def parse_picks(data: dict | None, show: list, k: int) -> tuple:
    """Gemini's picks (a list of image numbers, or a string of them) -> (world ids in its order, reasons)."""
    import re
    picks = []
    if data is None:
        return [], []
    for s in re.findall(r"\d+", json.dumps(data.get("picks", ""))):
        j = int(s) - 1
        if 0 <= j < len(show) and show[j] not in picks:
            picks.append(show[j])
    return picks[:k], [w.strip() for w in str(data.get("why", "")).split(";")]


def gallery(a, cfg=None) -> None:
    from lenia.describe import Ask, Describer
    run, dest = Path(a.run), Path(a.out) / "gallery"
    dest.mkdir(parents=True, exist_ok=True)
    sm = json.loads((run / "state.json").read_text())
    am = QD.settings(sm)
    pid, aim = QD.entry_prompt(am), am.aim or AIM_ALIFE
    jname, vname = (cfg.judge, cfg.vlm) if cfg else (a.judge, a.vlm)
    seed = cfg.seed if cfg else am.seed
    k_pick, n_short, n_swiss = getattr(a, "k", None) or GALLERY_K, getattr(a, "short", None) or GALLERY_SHORT, \
        getattr(a, "swiss", None) or GALLERY_ROUNDS
    rows = ledger_rows(run)
    if not rows:
        sys.exit(f"{run / 'ledger.jsonl'} is missing or empty: the map search needs --ledger")
    info = {x["id"]: x for x in rows if x["kind"] != "surprise"}
    edges = [(x["id"], x["opponent"], float(x["p"])) for x in rows
             if x["kind"] in ("duel", "duel_new") and x.get("p") is not None and x.get("opponent")]
    has = lambda i: (run / "img" / f"{i}.png").exists()
    ids = sorted({i for e in edges for i in e[:2]} | {i for i in info if has(i)})
    r0 = bt_fit(edges, ids)
    champ = {}
    for i in ids:                             # the cell champions: best-rated world with a picture per island cell
        x = info.get(i)
        if x is None or not has(i) or x.get("cell") in (None, ""):
            continue
        c = (x["island"], x["cell"])
        if c not in champ or r0[i] > r0[champ[c]]:
            champ[c] = i
    S = sorted(champ.values())
    c = load(getattr(a, "config", None))
    load_env(REPO_ROOT / ".env")
    judge = Judge(jname, {**c["judges"]["defaults"], **c["judges"][jname], "max_run_usd": a.gallery_max_usd},
                  dest / "judge_cache.sqlite", yes=True)
    pics = {i: pic_of(run / "img" / f"{i}.png") for i in S}
    play = lambda pairs, t: S1.duels(judge, pid, pairs, pics, f"gallery swiss round {t}")
    games = swiss(S, r0, n_swiss, np.random.default_rng([seed, 31]), play)
    r1 = bt_fit(edges + [(x, y, p) for x, y, p, _ in games], ids)
    ranked = sorted(S, key=lambda i: -r1[i])
    short, per_root, cells = [], Counter(), set()
    for i in ranked:                          # diversify: at most PER_ROOT per lineage, one per cell across islands
        x = info[i]
        if per_root[x.get("root")] >= PER_ROOT or x["cell"] in cells:
            continue
        short.append(i)
        per_root[x.get("root")] += 1
        cells.add(x["cell"])
        if len(short) == n_short:
            break
    show = [short[int(j)] for j in np.random.default_rng([seed, 41]).permutation(len(short))]   # no rank order shown
    d = Describer(vname, c["vlms"][vname], dest / "describe_cache.sqlite", max_usd=a.gallery_max_vlm_usd)
    k = min(k_pick, len(show))
    [ans] = d.ask_many([Ask(GALLERY, [pics[i] for i in show], {"aim": aim, "n": len(show), "k": k})], desc="gallery pick")
    data = ans.data if ans.ok else None
    if data is not None and vname == "fake" and os.environ.get("FULLRUN_FAKE_PICKS"):     # dry runs: test the parsing
        data = {**data, "picks": json.loads(os.environ["FULLRUN_FAKE_PICKS"])}
    picks, why = parse_picks(data, show, k)
    filled = 0
    for i in short:                           # a short or failed answer: fill from the ratings, and say so
        if len(picks) >= k:
            break
        if i not in picks:
            picks.append(i)
            filled += 1
    fb = ancestry(rows)
    ent = lambda i: {"id": i, "image": f"img/{i}.png", "rating": round(r1[i], 4), "rating_before_swiss": round(r0[i], 4),
                     "island": info[i]["island"], "cell": info[i]["cell"], "root": info[i].get("root"),
                     "source": info[i].get("source"), "it": info[i]["it"], "feedback_ancestors": fb(i),
                     "texture": info[i].get("texture"), "motion": info[i].get("motion"), "bodies": info[i].get("bodies")}
    out = {"run": str(run), "aim": aim, "entry_prompt": pid, "vlm": f"{vname}:{d.model}", "nodes": len(ids),
           "entry_duels": len(edges), "champions": len(S), "swiss_games": len(games), "shown_order": show,
           "vlm_answer": ans.data if ans.ok else ans.raw[:500], "filled_from_ratings": filled,
           "spend": {"jev": round(judge.cache.total_cost(), 4), "vlm": round(d.cache.total_cost(), 4)},
           "picks": [{**ent(i), "why": why[n] if n < len(why) and n < k - filled else None} for n, i in enumerate(picks)],
           "shortlist": [ent(i) for i in short]}
    (dest / "gallery.json").write_text(json.dumps(out, indent=1))
    with open(dest / "bt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "rating", "rating_before_swiss", "champion", "island", "cell", "root", "source", "it", "picture"])
        champs = set(S)
        for i in sorted(ids, key=lambda i: -r1[i]):
            x = info.get(i, {})
            w.writerow([i, round(r1[i], 4), round(r0[i], 4), int(i in champs), x.get("island"), x.get("cell"),
                        x.get("root"), x.get("source"), x.get("it"), int(has(i))])
    with open(dest / "tournament.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "first", "second", "p_first"])
        w.writerows([[t, x, y, round(p, 4)] for x, y, p, t in games])
    lab = lambda e: [f"{e['id']} {e['source']}", f"L{e['root']} cell {e['cell']}  BT {e['rating']:.2f}"]
    sheet([(run / e["image"], lab(e)) for e in out["picks"]], dest / "gallery.png", cols=4, px=300)
    sheet([(run / e["image"], lab(e)) for e in out["shortlist"]], dest / "shortlist.png", cols=6, px=220)
    print(f"gallery: {len(ids)} worlds, {len(edges)} entry duels, {len(S)} cell champions, {len(games)} Swiss games; "
          f"{len(short)} shortlisted, {len(picks)} picked ({filled} filled from the ratings); Jev ${judge.spent:.4f}, "
          f"VLM ${d.spent:.4f}; {dest / 'gallery.png'}", file=sys.stderr)


# ---------- report and ledger (no calls) ----------

def spent(path: Path, table: str) -> float:
    import sqlite3
    if not path.exists():
        return 0.0
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return float(db.execute(f"SELECT COALESCE(SUM(cost), 0) FROM {table}").fetchone()[0])
    finally:
        db.close()


def map_stats(run: Path) -> dict:
    sm = json.loads((run / "state.json").read_text())
    el = [e for m in QD.islands_of(sm) for e in m.values()]
    sar = sm.get("sarch", [])
    return {"it": sm["it"] - 1, "elites": len(el), "lineages": QD.lineages(sm), "archive": len(sar),
            "archive_texture": round(float(np.median([e["texture"] for e in sar])), 3) if sar else None,
            "elite_texture": round(float(np.median([e["texture"] for e in el])), 3) if el else None,
            "elite_texture_above_0.25": round(float(np.mean([e["texture"] > 0.25 for e in el])), 3) if el else None,
            "feedback_elites": sum(e.get("source") == "feedback" for e in el),
            "jev": round(spent(run / "judge_cache.sqlite", "verdicts"), 3)}


def report(a) -> None:
    F = Path(a.out)
    st = json.loads((F / "state.json").read_text())
    MAP = F / "map"
    rows = ledger_rows(MAP)
    fb = ancestry(rows)
    L = [f"# Full run {F.name}: round {min(st['round'], st['args']['rounds'])} of {st['args']['rounds']}"
         f"{'' if st['round'] > st['args']['rounds'] else ', phase ' + st['phase']}\n",
         f"Aim: \"{st['args']['aim']}\"; seed {st['args']['seed']}; {st['args']['iters']} map iterations, "
         f"{st['args']['triggers']} triggers and {st['args']['end']}-step expeditions per round.\n",
         "| round | map it | elites / lineages | archive (texture) | elite texture | trigger texture | trigger sources | "
         "from feedback | moved | review | feedback |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in range(1, st["args"]["rounds"] + 1):
        x = st["rounds"].get(str(r), {})
        t = x.get("triggers")
        if not t:
            continue
        R = F / "rounds" / f"r{r}"
        T = json.loads((R / "triggers.json").read_text())["triggers"]
        desc = sum(bool(fb(tr["wid"])) for tr in T)
        e = x.get("expeditions", {})
        L.append(f"| {r} | {t['map_it']} | {t['elites']} / {t['lineages']} | {t['archive']} ({t['archive_texture_median']}) | "
                 f"{t['elite_texture_median']} | {t['trigger_texture_median']} | {t['trigger_sources']} | {desc}/{len(T)} | "
                 f"{e.get('moved', '-')} | {e.get('review', '-')} | {x.get('feedback', '-')} |")
    ms = map_stats(MAP) if (MAP / "state.json").exists() else {}
    jev = {"map": ms.get("jev", 0.0), "triggers + expeditions + feedback": round(spent(F / "judge_cache.sqlite", "verdicts"), 3),
           "gallery": round(spent(F / "gallery" / "judge_cache.sqlite", "verdicts"), 3)}
    gem = {"goals, captions, reviews": round(spent(F / "describe_cache.sqlite", "answers"), 3),
           "gallery": round(spent(F / "gallery" / "describe_cache.sqlite", "answers"), 3)}
    L += ["", f"Map now: {ms}", "",
          f"Spend: Jev ${sum(jev.values()):.2f} ({jev}); Gemini ${sum(gem.values()):.2f} ({gem}).", ""]
    gj = F / "gallery" / "gallery.json"
    if gj.exists():
        g = json.loads(gj.read_text())
        L += [f"## Gallery ({len(g['picks'])} picked by {g['vlm']} from {len(g['shortlist'])}; {g['champions']} cell "
              f"champions, {g['swiss_games']} Swiss games, {g['entry_duels']} entry duels)", "",
              "| world | source | lineage | cell | BT | feedback ancestors | why |", "|---|---|---|---|---|---|---|"]
        L += [f"| {p['id']} | {p['source']} | L{p['root']} | {p['cell']} | {p['rating']:.2f} | "
              f"{len(p['feedback_ancestors'])} | {p['why'] or '(filled from ratings)'} |" for p in g["picks"]]
        n_fb = sum(bool(p["feedback_ancestors"]) for p in g["picks"])
        L += ["", f"{n_fb} of {len(g['picks'])} gallery worlds descend from a fed-back expedition end "
              f"({sum(bool(p['feedback_ancestors']) for p in g['shortlist'])} of {len(g['shortlist'])} shortlisted).", ""]
    if getattr(a, "baseline", None) and (Path(a.baseline) / "state.json").exists():
        B = Path(a.baseline)
        L += [f"## Baseline {B.name} (map search only)", "", f"Map: {map_stats(B)}"]
        bg = B / "gallery" / "gallery.json"
        if bg.exists():
            g = json.loads(bg.read_text())
            L += [f"Gallery: {', '.join(p['id'] for p in g['picks'])}; spend {g['spend']}"]
    (F / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def ledger(a) -> None:
    import math
    F = Path(a.out)
    st = json.loads((F / "state.json").read_text())
    MAP, iters = F / "map", st["args"]["iters"]
    W = {}
    for x in ledger_rows(MAP):
        w = W.setdefault(x["id"], {"id": x["id"]})
        if x["kind"] == "surprise":
            w.update(archive_surprise=x["score"], archive_added=x["added"])
            continue
        w.update(source=x["source"], round=x.get("round") or max(1, math.ceil(x["it"] / iters)), it=x["it"],
                 island=x["island"], root=x.get("root"), parent_id=x.get("parent_id"), cell=x.get("cell"), kind=x["kind"],
                 entry_p=x.get("p"), entry_opponent=x.get("opponent"), placed=x.get("placed"), bodies=x.get("bodies"),
                 motion=x.get("motion"), texture=x.get("texture"), rule=json.dumps(x["rule"]),
                 picture=f"map/img/{x['id']}.png" if (MAP / "img" / f"{x['id']}.png").exists() else "")
    for R in sorted((F / "rounds").glob("r*")):
        r = int(R.name[1:])
        if (R / "rescore.csv").exists():
            for x in csv.DictReader(open(R / "rescore.csv")):
                w = W.setdefault(x["wid"], {"id": x["wid"]})
                w["rescore"] = (w.get("rescore", "") + f" r{r}:{x['surprise']}").strip()
        jd = {}
        for x in read_jsonl(R / "exp" / "judged.jsonl"):
            jd[(x["trigger"], x["step"], x["child"])] = x
        for x in jd.values():
            i = f"x_{x['trigger']}_s{x['step']:02d}_c{x['child']:02d}"
            W[i] = {"id": i, "source": "expedition", "round": r, "trigger": x["trigger"], "step": x["step"],
                    "goal": x["goal"], "complexity": x["complexity"], "exp_surprise": x["surprise"], "picked": x["picked"],
                    "bodies": x["bodies"], "motion": x["motion"], "texture": x["texture"], "rule": json.dumps(x["rule"]),
                    "picture": f"evals/{x['key']}.png" if (F / "evals" / f"{x['key']}.png").exists() else ""}
    cols = ["id", "source", "round", "it", "island", "root", "parent_id", "cell", "kind", "entry_p", "entry_opponent",
            "placed", "archive_surprise", "archive_added", "rescore", "trigger", "step", "goal", "complexity",
            "exp_surprise", "picked", "bodies", "motion", "texture", "picture", "rule"]
    with open(F / "ledger.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(W.values())
    print(f"{F / 'ledger.csv'}: {len(W)} worlds ({dict(Counter(x.get('source', '?') for x in W.values()))}); "
          "an empty picture means regrow it from its rule", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", default="numpy", help="numpy | auto | cuda | mps")
    r.add_argument("--evals", default=None, help="simulation cache (on Colab /content/evals; picks go to <out>/evals)")
    r.add_argument("--baseline", default=None, help="fork the map here after round 1's map search (the map-only baseline)")
    r.add_argument("--config", default=None)
    r.add_argument("--sim-batch", type=int, default=0, help="worlds simulated together (map and expeditions; free to raise)")
    r.add_argument("--max-usd", type=float, default=5.0, help="cap on <out>/judge_cache.sqlite's total (triggers, "
                   "expeditions, feedback; ~$2.70 expected)")
    r.add_argument("--map-max-usd", type=float, default=4.0, help="cap on the map search's own cache (~$2.40 at 484 px)")
    r.add_argument("--max-vlm-usd", type=float, default=3.0, help="cap on <out>/describe_cache.sqlite (~$1.70 expected)")
    r.add_argument("--gallery-max-usd", type=float, default=0.5)
    r.add_argument("--gallery-max-vlm-usd", type=float, default=0.5)
    r.add_argument("--stop-after", default=None, help="ROUND:PHASE, e.g. 1:exp (phases: " + ", ".join(PHASES) + ")")
    r.add_argument("--no-gallery", action="store_true")
    r.add_argument("--dry", action="store_true", help="offline: fake judges, VLM and CLIP, tiny worlds, 2 short rounds")
    for k, v in DEFAULTS.items():             # fixed at setup (kept in state.json); defaults are the agreed design
        r.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=None,
                       help=f"default {v}")
    g = sub.add_parser("gallery")
    g.add_argument("--run", required=True, help="a map-search folder with ledger.jsonl (<full>/map, or the baseline)")
    g.add_argument("--out", required=True, help="the gallery goes to <out>/gallery")
    g.add_argument("--judge", default="jev")
    g.add_argument("--vlm", default="gemini-pro")
    g.add_argument("--config", default=None)
    g.add_argument("--k", type=int, default=GALLERY_K, help="worlds in the final gallery")
    g.add_argument("--short", type=int, default=GALLERY_SHORT, help="the diversified shortlist Gemini picks from")
    g.add_argument("--swiss", type=int, default=GALLERY_ROUNDS, help="Swiss rounds among the cell champions")
    g.add_argument("--gallery-max-usd", type=float, default=0.5)
    g.add_argument("--gallery-max-vlm-usd", type=float, default=0.5)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default=None)
    lg = sub.add_parser("ledger")
    lg.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "gallery": gallery, "report": report, "ledger": ledger}[a.cmd](a)


if __name__ == "__main__":
    main()
