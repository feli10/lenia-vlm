"""MAP-Elites over MaCE-Lenia worlds, with Jev duels as the quality signal.

  python scripts/mace_qd.py run   --out RUNS/mace-qd --backend auto     # v1 (bodies x motion, one map)
  python scripts/mace_qd.py run   --out RUNS/mace-qd2 --backend auto --islands 4 --axes bodies,form,motion \\
      --body-edges 3,9 --motion-edges 0.75,1.0,1.25                     # v2: islands + a form axis
  python scripts/mace_qd.py map   --out RUNS/mace-qd                    # map.png + summary
  python scripts/mace_qd.py check --out RUNS/mace-qd                    # final elites vs elites at a snapshot
  python scripts/mace_qd.py rate  --out RUNS/mace-qd                    # your ratings of the check pairs (Mac)
  python scripts/mace_qd.py run   --out RUNS/mace-qd5-sarch-s0-aim --like RUNS/mace-qd5-sarch-s0 --backend auto \\
      --aim "more interesting as artificial life: richer structure or behaviour, more like something alive"   # map-prompt test
  python scripts/mace_qd.py run   --out /tmp/qd-dry --judge fake --embed-model fake --iterations 3 --batch 8 \\
      --init 8 --size 96 --warmup 200 --render 120 --gap 40             # offline dry run

Why MAP-Elites (project doc "MaCE-Lenia Jev-guided evolution results"): the hill climber (mace_evo.py, now in archive/)
duels each mutant only against its own parent and, after ~60 generations, moved sideways. Here a candidate
must beat the elite already holding its niche. v1 fixed that treadmill (held-out Luna preferred later elites
33/43) but one seed's lineage colonised all 56 cells: bodies x motion can be traversed by a single body
plan. v2 adds two things against that:
- islands (--islands K): K separate maps, each seeded with every K-th seed and evolved from its own elites
  (--batch / K mutants each); every --migrate-every iterations each island receives one random elite of
  the next island (ring), which must win its cell like any challenger. One lineage can still win an island
  but no longer the whole run.
- a form axis ("form" in --axes): the first principal component of a frozen DINOv2 embedding of the final
  frame, fitted on the seeds at iteration 0 and binned at their quantiles (--form-bins), so looking
  different earns a separate cell.

Evaluation: every rule is grown from the same smoothed-noise start (--start-seed) for --warmup steps and
filmed for --render more; the judge sees four snapshots --gap steps apart (2x2 image). No world is
inherited, so elite and challenger are always the same age.
- Constraint: the grid-noise rule only (texture of the final frame <= --max-texture).
- Axes (any of): bodies (components >= 9 cells, --body-edges), motion (|final - first snapshot| / mass
  over --render steps, --motion-edges; frozen ring patterns ~0.5, visibly rearranging worlds 1.3-1.6),
  form (above).
- Insertion: a candidate takes an empty cell; for an occupied cell it duels the elite (Jev, both orders,
  P = mean) and replaces it at P >= --threshold. Several candidates for one cell in one iteration: the first
  takes an empty cell, the others duel it; the highest P >= threshold wins.
- Variation: kernel mutation at --mut-scale plus beta x exp(N(0, 0.1)) in [0.5, 4], as in the hill climber (lenia.mace.mutate).
- Seeds: the 12 founders (lenia.mace.founders) plus --init more draws of the same sampler. Every elite records the seed
  it descends from ("root").
- Parent selection: uniform from the island's elites (plus the surprise archive's members with --sarch).
  Surprise-weighted parent selection (v3, --parent-select surprise) was retired on 2026-10-28 (null on the
  replicate); the full script with it is archive/scripts/mace_qd_full.py.
- Surprise archive (--sarch, v4): surprise decides what is *kept*, not only what is mutated. After the map
  insertions of an iteration, --sarch-candidates random challengers per island that did not get into the map
  (lost the complexity duel or the competition for their cell; grid noise excluded) are judged with the surprise
  duel against --surprise-opponents random members of the map and the surprise archive (all islands), both
  orders. Those scoring >= the current threshold join the surprise archive (kept even though the map rejected
  them; at most --sarch-max, oldest dropped first). A share --sarch-parents of each island's parents is drawn
  uniformly from that island's archive members, so surprising worlds become stepping stones whose children
  compete for the map like any other. This is novelty search's archive (Lehman & Stanley 2011) with the
  validated surprise duel as the novelty measure, including their adaptive threshold: +0.02 when more than
  2 enter in one iteration, -0.02 after 5 iterations with none, within [0.5, 0.9], from --sarch-threshold.
Saves state after every iteration (resumes) and a snapshot of the maps every --snapshot-every iterations.
Full run (11-05, scripts/fullrun.py): --surprise-px 484 shrinks the surprise duel's pictures (gate 0f); --ledger logs
every judged world with its rule and its duel's opponent (ledger.jsonl), for the gallery's Bradley-Terry ratings.
Both are off by default, so earlier runs and commands are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lenia import mace as ME
from lenia import backend, render, seeds
from lenia.mace import bodies as label_bodies
from lenia.config import REPO_ROOT, load
from lenia.mace import mutate, texture
from lenia.judge import Judge, Query, load_env
from lenia.sample import sample
from lenia.sim import Rule, with_flow

PROMPT_ID = ME.PROMPT_ID


def entry_prompt(a) -> str:
    """The map's entry duel: COMPLEXITY, or with --aim the same prompt with the aim as its goal (lenia.prompts.map_prompt;
    map-prompt test, 11-04). Runs from before --aim have no aim and keep COMPLEXITY."""
    from lenia.prompts import map_prompt
    return map_prompt(a.aim) if getattr(a, "aim", None) else PROMPT_ID
BODY_EDGES = [2, 3, 5, 9, 17, 33]                     # v1 bins: 1 | 2 | 3-4 | 5-8 | 9-16 | 17-32 | 33+


# ---------- run settings (v1 runs predate some options) ----------

def settings(st: dict) -> argparse.Namespace:
    a = argparse.Namespace(**{"axes": ["bodies", "motion"], "body_edges": BODY_EDGES, "islands": 1,
                              "migrate_every": 10, "form_bins": 4, "embed_model": "dinov2_vits14",
                              "parent_select": "uniform", "surprise_opponents": 6, "surprise_every": 20,
                              "surprise_mix": 0.5, "surprise_temp": 0.1, "surprise_ref": "all", "sarch": False,
                              "sarch_candidates": 3, "sarch_parents": 0.25, "sarch_threshold": 0.6, "sarch_max": 200,
                              "aim": None, "like": None, "surprise_px": 0, "ledger": False, **st["args"]})
    if isinstance(a.axes, str):
        a.axes = a.axes.split(",")
    return a


def islands_of(st_or_snapshot) -> list:
    """The list of island maps of a state or snapshot (v1: a single map)."""
    if isinstance(st_or_snapshot, list):
        return st_or_snapshot
    if "islands" in st_or_snapshot:
        return st_or_snapshot["islands"]
    if "archive" in st_or_snapshot:
        return [st_or_snapshot["archive"]]
    return [st_or_snapshot]                            # a v1 snapshot file is the archive itself


def edges(axis: str, a, st: dict) -> list:
    return {"bodies": list(a.body_edges), "motion": list(a.motion_edges)}.get(axis) or st["form"]["edges"]


def labels(axis: str, a, st: dict) -> list:
    e = edges(axis, a, st)
    if axis == "bodies":
        lo = [1] + list(e)
        return [f"{lo[i]}" if i < len(e) and e[i] - lo[i] == 1 else (f"{lo[i]}-{e[i] - 1}" if i < len(e) else f"{lo[i]}+")
                for i in range(len(e) + 1)]
    if axis == "form":
        return [f"f{i}" for i in range(len(e) + 1)]
    lo = [0.0] + list(e)
    return [f"{lo[i]:.2f}-{e[i]:.2f}" if i < len(e) else f">{lo[i]:.2f}" for i in range(len(e) + 1)]


def cell_of(e: dict, a, st: dict) -> str:
    out = []
    for ax in a.axes:
        side = "right" if ax == "bodies" else "left"
        out.append(int(np.searchsorted(edges(ax, a, st), e[ax], side=side)))
    return "-".join(map(str, out))


# ---------- evaluation ----------

def n_bodies(A: np.ndarray, min_cells: int = 9) -> int:
    m = A.sum(0)
    mask = m > 0.1 * max(float(np.percentile(m, 99.5)), 1e-9)
    if not mask.any():
        return 0
    lab = label_bodies(mask)
    _, area = np.unique(lab[mask], return_counts=True)
    return int((area >= min_cells).sum())


def coverage(A: np.ndarray) -> float:
    m = A.sum(0)
    return float((m > 0.05 * max(float(m.max()), 1e-9)).mean())


_EMB = {}


def embedder(name: str):
    if name not in _EMB:
        from lenia.embed import Embedder
        _EMB[name] = Embedder(name)
    return _EMB[name]


def evaluate(rules: list, a) -> list:
    """Grow every rule from the fixed start; per rule: picture, descriptors, texture (and an embedding of
    the final frame if the form axis is on)."""
    shape = (a.size, a.size)
    A0 = seeds.smooth_noise(a.R, shape, 3, seed=a.start_seed, size_r=3.0)
    out = []
    for k in range(0, len(rules), a.sim_batch):
        rr = rules[k:k + a.sim_batch]
        _, films = ME.simulate(rr, np.stack([A0] * len(rr)), a.warmup + a.render, shape, a.gap)
        for f in films:
            first, last = f[0], f[-1]
            mass = max(float(last.sum()), 1e-9)
            out.append({"pic": ME.picture(f), "bodies": n_bodies(last), "motion": float(np.abs(last - first).sum() / mass),
                        "texture": float(texture(last)), "coverage": coverage(last)})
    if "form" in a.axes and out:
        tiles = [Image.fromarray(e["pic"][-ME.TILE:, -ME.TILE:]) for e in out]   # the final snapshot
        for e, v in zip(out, embedder(a.embed_model)(tiles)):
            e["emb"] = np.asarray(v, np.float64)
    return out


def fit_form(evals: list, a) -> dict:
    """PCA of the seeds' embeddings (texture-passing ones): mean, first component, quantile bin edges."""
    X = np.stack([e["emb"] for e in evals if a.max_texture <= 0 or e["texture"] <= a.max_texture])
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    pc = vt[0]
    proj = (X - mu) @ pc
    q = np.quantile(proj, np.linspace(0, 1, a.form_bins + 1)[1:-1])
    return {"mean": mu.tolist(), "pc": pc.tolist(), "edges": [float(x) for x in q]}


def add_form(evals: list, st: dict) -> None:
    if "form" not in st:
        return
    mu, pc = np.asarray(st["form"]["mean"]), np.asarray(st["form"]["pc"])
    for e in evals:
        e["form"] = float((e["emb"] - mu) @ pc)


# ---------- insertion ----------

def save_state(out: Path, st: dict) -> None:
    tmp = out / "state.tmp.json"
    tmp.write_text(json.dumps(st))
    tmp.replace(out / "state.json")


def insert(st: dict, k: int, cands: list, evals: list, judge, out: Path, it: int, log, a, pids: list | None = None,
           ledger: list | None = None) -> dict:
    """Into island k. cands: [(rule, parent label or None, root seed)]. Returns counts. With ledger (a list), one row
    per candidate is appended to it (--ledger): its id (its picture's name without .png, saved or not), rule, the duel's
    opponent id and P, and whether it was placed; pids: the parents' ids, parallel to cands."""
    arch = islands_of(st)[k]
    imgdir = out / "img"
    tag = f"I{k}_" if a.islands > 1 else ""
    by_cell: dict = {}
    n_tex = 0

    lrows = {}

    def wid(n):                              # the name place() gives the picture, without .png
        mig = "m" if str(cands[n][1] or "").startswith("migrant") else ""
        return f"it{it:04d}_{tag}{mig}{n:03d}"

    def row(n, ck, kind, p, acc, opp=None):
        e = evals[n]
        if ledger is not None:
            rule, parent, root = cands[n]
            lrows[n] = {"id": wid(n), "source": "migrant" if str(parent or "").startswith("migrant") else "map", "it": it,
                        "island": k, "n": n, "kind": kind, "cell": ck, "opponent": opp,
                        "p": None if p is None else round(p, 4), "accepted": int(acc), "parent": parent,
                        "parent_id": pids[n] if pids else None, "root": root, "rule": rule.to_dict(),
                        "bodies": e["bodies"], "motion": round(e["motion"], 4), "texture": round(e["texture"], 4),
                        "coverage": round(e.get("coverage", float("nan")), 4),
                        **({"form": round(e["form"], 4)} if "form" in e else {})}
        log.writerow([it, k, n, cands[n][1] or "", ck, kind, "" if p is None else round(p, 4), int(acc), e["bodies"],
                      round(e["motion"], 4), round(e.get("form", float("nan")), 4), round(e["texture"], 4),
                      round(cands[n][0].beta, 3), cands[n][2]])

    for n, e in enumerate(evals):
        if a.max_texture > 0 and e["texture"] > a.max_texture:
            n_tex += 1
            row(n, "", "texture", None, 0)
            continue
        by_cell.setdefault(cell_of(e, a, st), []).append(n)
    duels, first_new, opps = [], {}, []
    for ck, ns in by_cell.items():
        if ck in arch:
            inc = np.asarray(Image.open(imgdir / arch[ck]["image"]).convert("RGB"))
            duels += [(ck, n, inc) for n in ns]
            opps += [Path(arch[ck]["image"]).stem] * len(ns)
        else:
            first_new[ck] = ns[0]
            duels += [(ck, n, evals[ns[0]]["pic"]) for n in ns[1:]]
            opps += [wid(ns[0])] * (len(ns) - 1)
    qs = []
    for ck, n, inc in duels:
        qs += [Query(entry_prompt(a), evals[n]["pic"], inc), Query(entry_prompt(a), inc, evals[n]["pic"])]
    vs = judge.judge_many(qs, desc=f"it {it} {tag.rstrip('_') or 'map'}") if qs else []
    best = {}
    for d, (ck, n, _) in enumerate(duels):
        va, vb = vs[2 * d], vs[2 * d + 1]
        p = None if not (va.ok and vb.ok) else 0.5 * (va.p_first + 1 - vb.p_first)
        acc = p is not None and p >= a.threshold
        if acc and p > best.get(ck, (0.0, None))[0]:
            best[ck] = (p, n)
        row(n, ck, "duel" if ck in arch else "duel_new", p, acc, opps[d])

    def place(ck, n, p):
        rule, parent, root = cands[n]
        e = evals[n]
        # migrants get their own names: a migrant and the island's mutant n of the same iteration used to share
        # it{it}_I{k}_{n}.png, and the existence check below gave one of them the other's picture (fixed 2026-10-29)
        mig = "m" if str(parent or "").startswith("migrant") else ""
        name = f"it{it:04d}_{tag}{mig}{n:03d}.png"
        if not (imgdir / name).exists():
            Image.fromarray(e["pic"]).save(imgdir / name)
        prev = arch.get(ck)
        arch[ck] = {"rule": rule.to_dict(), "image": name, "it": it, "parent": parent, "root": root,
                    "bodies": e["bodies"], "motion": round(e["motion"], 4), "texture": round(e["texture"], 4),
                    "coverage": round(e["coverage"], 4), "p": None if p is None else round(p, 4),
                    "replaced": 0 if prev is None else prev.get("replaced", 0) + 1,
                    **({"form": round(e["form"], 4)} if "form" in e else {})}

    new_cells = replaced = 0
    placed = {n for ck, n in first_new.items() if ck not in best} | {n for _, n in best.values()}
    for ck, n in first_new.items():
        if ck not in best:
            place(ck, n, None)
            new_cells += 1
            row(n, ck, "new", None, 1)
    for ck, (p, n) in best.items():
        if ck in arch:
            replaced += 1
        else:
            new_cells += 1
        place(ck, n, p)
    outside = [n for ns in by_cell.values() for n in ns if n not in placed]
    if ledger is not None:
        ledger += [{**r, "placed": int(n in placed)} for n, r in sorted(lrows.items())]
    return {"texture_fail": n_tex, "new": new_cells, "replaced": replaced, "duels": len(duels), "outside": outside}


def lineages(st: dict) -> int:
    return len({e.get("root") for arch in islands_of(st) for e in arch.values()})


def small(pic: np.ndarray, px: int) -> np.ndarray:
    """The picture shrunk to px per side with a box filter (--surprise-px): gate 0f (scripts/res_check.py) validated 484 px
    for the surprise duel only; Jev bills by input size, so it costs about 44% less per duel."""
    return pic if not px or px >= pic.shape[0] else np.asarray(Image.fromarray(pic).resize((px, px), Image.BOX))


def write_ledger(out: Path, rows: list) -> None:
    """--ledger: one JSON line per judged world and per surprise judgment (ledger.jsonl). Written before the state is
    saved, so a crash mid-iteration can repeat an iteration's rows on resume: readers keep the last row per (id, kind)."""
    if rows:
        with open(out / "ledger.jsonl", "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)


# ---------- surprise archive (v4) ----------

def sarch_step(st: dict, outside: list, judge, out: Path, rng, a, it: int, slog, ledger: list | None = None) -> dict:
    """outside: [(island, cand, eval)] challengers the map rejected this iteration. Judges up to
    --sarch-candidates per island with the surprise duel against random members of map + surprise archive and
    adds those at or above the threshold. Returns counts."""
    from lenia.prompts import SURPRISE
    sar = st.setdefault("sarch", [])
    pick = []
    for k in range(a.islands):
        mine = [x for x in outside if x[0] == k]
        pick += [mine[int(j)] for j in rng.choice(len(mine), min(a.sarch_candidates, len(mine)), replace=False)]
    ref = [islands_of(st)[k][ck]["image"] for k, m in enumerate(islands_of(st)) for ck in m] + [e["image"] for e in sar]
    if not pick or len(ref) < 2:
        return {}
    cache = {}
    sp = getattr(a, "surprise_px", 0) or 0

    def pic(name):
        if name not in cache:
            cache[name] = small(np.asarray(Image.open(out / "img" / name).convert("RGB")), sp)
        return cache[name]

    qs, opp_ids = [], []
    for k, c, ev in pick:
        mine = small(ev["pic"], sp)
        js = rng.choice(len(ref), min(a.surprise_opponents, len(ref)), replace=False)
        opp_ids.append([Path(ref[int(j)]).stem for j in js])
        for j in js:
            o = pic(ref[int(j)])
            qs += [Query(SURPRISE, mine, o), Query(SURPRISE, o, mine)]
    vs = judge.judge_many(qs, desc=f"it {it} surprise archive")
    per = len(qs) // len(pick)
    thr = st.get("sarch_thr", a.sarch_threshold)
    added = 0
    tag = lambda k: f"I{k}_" if a.islands > 1 else ""
    for i, (k, (rule, parent, root, n), ev) in enumerate(pick):
        v = [0.5 * (x.p_first + 1 - y.p_first) for x, y in zip(vs[i * per:(i + 1) * per:2], vs[i * per + 1:(i + 1) * per:2])
             if x.ok and y.ok]
        sc = float(np.mean(v)) if v else None
        ok = sc is not None and sc >= thr
        slog.writerow([it, k, n, parent or "", "" if sc is None else round(sc, 4), round(thr, 3), int(ok), ev["bodies"],
                       round(ev["motion"], 4), root])
        if ledger is not None:
            ledger.append({"id": f"it{it:04d}_{tag(k)}{n:03d}", "kind": "surprise", "it": it, "island": k,
                           "score": None if sc is None else round(sc, 4), "threshold": round(thr, 3), "added": int(ok),
                           "opponents": opp_ids[i], "px": sp or ev["pic"].shape[0]})
        if ok:
            name = f"it{it:04d}_{tag(k)}{n:03d}.png"
            if not (out / "img" / name).exists():
                Image.fromarray(ev["pic"]).save(out / "img" / name)
            sar.append({"rule": rule.to_dict(), "image": name, "it": it, "parent": parent, "root": root, "island": k,
                        "surprise": round(sc, 4), "bodies": ev["bodies"], "motion": round(ev["motion"], 4),
                        "texture": round(ev["texture"], 4), "coverage": round(ev["coverage"], 4),
                        **({"form": round(ev["form"], 4)} if "form" in ev else {})})
            added += 1
    if len(sar) > a.sarch_max:
        del sar[:len(sar) - a.sarch_max]
    dry = 0 if added else st.get("sarch_dry", 0) + 1                  # novelty search's adaptive threshold
    if added > 2:
        thr = min(0.9, thr + 0.02)
    elif dry >= 5:
        thr, dry = max(0.5, thr - 0.02), 0
    st["sarch_thr"], st["sarch_dry"] = round(thr, 4), dry
    return {"sarch_judged": len(pick), "sarch_added": added, "sarch_calls": len(qs)}


def run(a) -> None:
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                  out / "judge_cache.sqlite", yes=True)
    like_form = None
    if getattr(a, "like", None) and not (out / "state.json").exists():   # copy another run's settings (all but these)
        base = json.loads((Path(a.like) / "state.json").read_text())
        own = ("out", "backend", "judge", "config", "sim_batch", "max_usd", "aim", "like", "cmd")
        for k, v in base["args"].items():
            if k not in own:
                setattr(a, k, v)
        like_form = base.get("form")         # the same form-axis fit, so a cell means the same thing in both maps
        print(f"settings and form fit copied from {a.like}", file=sys.stderr)
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        keep = ("iterations", "backend", "judge", "out", "config", "sim_batch", "max_usd")
        if "parent_select" not in st["args"]:                    # older runs: uniform parent selection
            st["args"]["parent_select"] = "uniform"
        if st["args"]["parent_select"] == "surprise":
            sys.exit("this run used --parent-select surprise (v3), retired on 2026-10-28: resume it with "
                     "archive/scripts/mace_qd_full.py")
        st["args"].setdefault("sarch", False)
        st["args"].update({k: getattr(a, k) for k in keep})
        a = settings(st)
        print(f"resuming at iteration {st['it']} ({sum(len(m) for m in islands_of(st))} cells filled)", file=sys.stderr)
    else:
        st = {"it": 0, "args": {**vars(a), "axes": a.axes}, "prompt": ME.PROMPTS[entry_prompt(a)], "progress": []}
        if a.islands > 1:
            st["islands"] = [{} for _ in range(a.islands)]
        else:
            st["archive"] = {}
    K = a.islands
    fh = open(out / "log.csv", "a", newline="")
    log = csv.writer(fh)
    if fh.tell() == 0:
        log.writerow(["it", "island", "cand", "parent", "cell", "kind", "p", "accepted", "bodies", "motion", "form",
                      "texture", "beta", "root"])
    if a.sarch:
        sfh = open(out / "sarch.csv", "a", newline="")
        slog = csv.writer(sfh)
        if sfh.tell() == 0:
            slog.writerow(["it", "island", "cand", "parent", "surprise", "threshold", "added", "bodies", "motion", "root"])
    while st["it"] <= a.iterations:
        it = st["it"]
        t0 = time.time()
        rng = np.random.default_rng([a.seed, it])
        if it == 0:
            found = ME.founders(argparse.Namespace(lineages=12, rule_seed=3, R=a.R, beta=a.beta, size=a.size), (a.size, a.size))[0]
            extra = [with_flow(r, "mace", a.beta) for r in
                     sample("multichannel", a.init, seed=a.init_seed, R=a.R, channels=3, self_k=1, cross_k=1)]
            seeds_ = found + extra
            groups = [[(r, None, s) for s, r in enumerate(seeds_) if s % K == k] for k in range(K)]
            pid_groups = [[None] * len(g) for g in groups]
        else:
            groups, pid_groups = [], []
            for k, arch in enumerate(islands_of(st)):
                keys = sorted(arch)
                g, pg = [], []
                sk = [j for j, e in enumerate(st.get("sarch", [])) if e["island"] == k] if a.sarch else []
                for _ in range(a.batch // K if keys else 0):
                    if sk and rng.random() < a.sarch_parents:          # a stepping stone from the surprise archive
                        j = sk[int(rng.integers(len(sk)))]
                        se = st["sarch"][j]
                        par = Rule.from_dict(se["rule"])
                        m = mutate(par, rng, a.mut_scale)
                        m.flow = "mace"
                        m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                        g.append((m, f"I{k}:S:{se['image']}", se.get("root")))
                        pg.append(Path(se["image"]).stem)
                        continue
                    pk = keys[int(rng.integers(len(keys)))]
                    par = Rule.from_dict(arch[pk]["rule"])
                    m = mutate(par, rng, a.mut_scale)
                    m.flow = "mace"
                    m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                    g.append((m, f"I{k}:{pk}" if K > 1 else pk, arch[pk].get("root")))
                    pg.append(Path(arch[pk]["image"]).stem)
                groups.append(g)
                pid_groups.append(pg)
        flat = [c for g in groups for c in g]
        evals = evaluate([c[0] for c in flat], a)
        if it == 0 and "form" in a.axes:
            st["form"] = like_form or fit_form(evals, a)
        add_form(evals, st)
        res = Counter()
        pos = 0
        outside = []
        lrows = [] if a.ledger else None
        for k, g in enumerate(groups):
            r = insert(st, k, g, evals[pos:pos + len(g)], judge, out, it, log, a, pids=pid_groups[k], ledger=lrows)
            outside += [(k, (*g[n], n), evals[pos + n]) for n in r.pop("outside")]
            res.update(r)
            pos += len(g)
        if K > 1 and it > 0 and it % a.migrate_every == 0:              # ring migration: island k gets one of k+1's
            maps = islands_of(st)
            picks = []
            for k in range(K):
                src = maps[(k + 1) % K]
                if src:
                    ck = sorted(src)[int(rng.integers(len(src)))]
                    picks.append((k, (k + 1) % K, ck, dict(src[ck])))
            for k, s, ck, e in picks:
                ev = {"pic": np.asarray(Image.open(out / "img" / e["image"]).convert("RGB")), "bodies": e["bodies"],
                      "motion": e["motion"], "texture": e["texture"], "coverage": e["coverage"],
                      **({"form": e["form"]} if "form" in e else {})}
                r = insert(st, k, [(Rule.from_dict(e["rule"]), f"migrant I{s}:{ck}", e.get("root"))], [ev], judge, out, it, log, a,
                           pids=[Path(e["image"]).stem], ledger=lrows)
                res["migrated"] += r["new"] + r["replaced"]
        if a.sarch and it > 0:
            res.update(sarch_step(st, outside, judge, out, np.random.default_rng([a.seed, it, 11]), a, it, slog, lrows))
            sfh.flush()
        fh.flush()
        filled = sum(len(m) for m in islands_of(st))
        st["progress"].append({"it": it, "filled": filled, "lineages": lineages(st), **dict(res), "sec": round(time.time() - t0, 1)})
        if it % a.snapshot_every == 0:
            snap = st["islands"] if K > 1 else st["archive"]
            (out / "snapshots" / f"archive_it{it:04d}.json").write_text(json.dumps(snap))
        st["it"] = it + 1
        if lrows:
            write_ledger(out, lrows)
        save_state(out, st)
        print(f"it {it}: {filled} cells filled, {lineages(st)} seed lineages, {res['new']} new, {res['replaced']} replaced, "
              f"{res['duels']} duels, {res['texture_fail']} grid-noise"
              + (f", {res['migrated']} migrants settled" if K > 1 and it % a.migrate_every == 0 and it else "")
              + (f", surprise archive +{res['sarch_added']}/{res['sarch_judged']} -> {len(st.get('sarch', []))} "
                 f"(threshold {st.get('sarch_thr', a.sarch_threshold):.2f})" if a.sarch and it else "")
              + f", {time.time() - t0:.0f}s, judge spent ${judge.spent:.4f}", file=sys.stderr)
    fh.close()
    draw_map(out)
    if a.sarch:
        sfh.close()
        draw_sarch(out)


# ---------- outputs ----------

def draw_map(out: Path, path: Path | None = None, maps: list | None = None, title: str | None = None) -> Path:
    """One panel per island: columns = the last axis, rows = every combination of the other axes."""
    st = json.loads((out / "state.json").read_text())
    a = settings(st)
    maps = maps if maps is not None else islands_of(st)
    ax_rows, ax_col = a.axes[:-1], a.axes[-1]
    rl = [labels(x, a, st) for x in ax_rows]
    cl = labels(ax_col, a, st)
    combos = [[]]
    for L in rl:
        combos = [c + [i] for c in combos for i in range(len(L))]
    T = 150 if len(combos) * len(maps) <= 8 else 96
    pad, left, top = 3, 64, 40
    pw = len(cl) * (T + pad) + 12
    W, H = left + len(maps) * pw, top + 18 + len(combos) * (T + pad) + 24
    im = Image.new("RGB", (W, H), (12, 13, 20))
    d = ImageDraw.Draw(im)
    n_cells = len(combos) * len(cl)
    d.text((8, 8), title or (f"{out.name}: {sum(len(m) for m in maps)}/{n_cells * len(maps)} cells, "
                             f"{lineages({'islands': maps})} seed lineages, iteration {st['it'] - 1}; rows "
                             f"{' x '.join(ax_rows)}, columns {ax_col}"), fill=(225, 228, 235))
    for r, c in enumerate(combos):
        d.text((6, top + 18 + r * (T + pad) + T // 2 - 6), " ".join(rl[x][i] for x, i in enumerate(c)), fill=(160, 165, 180))
    for k, arch in enumerate(maps):
        x0 = left + k * pw
        if len(maps) > 1:
            roots = Counter(e.get("root") for e in arch.values())
            d.text((x0, top), f"island {k}: lineages {dict(roots.most_common(3))}", fill=(200, 205, 215))
        for j in range(len(cl)):
            d.text((x0 + j * (T + pad) + 2, H - 20), cl[j], fill=(160, 165, 180))
        for r, c in enumerate(combos):
            for j in range(len(cl)):
                e = arch.get("-".join(map(str, c + [j])))
                x, y = x0 + j * (T + pad), top + 18 + r * (T + pad)
                if e is None:
                    d.rectangle([x, y, x + T, y + T], outline=(40, 42, 55))
                    continue
                pic = np.asarray(Image.open(out / "img" / e["image"]).convert("RGB"))
                im.paste(Image.fromarray(pic[-ME.TILE:, -ME.TILE:]).resize((T, T), Image.BILINEAR), (x, y))
    path = path or out / "map.png"
    im.save(path)
    return path


def draw_sarch(out: Path, cols: int = 10, px: int = 160) -> Path | None:
    """sarch.png: every surprise-archive member (2x2 picture), oldest first, labelled iteration / island / score."""
    st = json.loads((out / "state.json").read_text())
    sar = st.get("sarch", [])
    if not sar:
        return None
    rows = (len(sar) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * px, rows * (px + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for i, e in enumerate(sar):
        x, y = (i % cols) * px, (i // cols) * (px + 14)
        sheet.paste(Image.open(out / "img" / e["image"]).convert("RGB").resize((px, px)), (x, y + 14))
        d.text((x + 3, y + 1), f"it{e['it']} I{e['island']} s{e['surprise']:.2f}", fill="black")
    path = out / "sarch.png"
    sheet.save(path)
    return path


def show_map(a) -> None:
    out = Path(a.out)
    p = draw_map(out)
    st = json.loads((out / "state.json").read_text())
    pr = st["progress"]
    if pr:
        tail = pr[-10:]
        print(f"iteration {st['it'] - 1}: {sum(len(m) for m in islands_of(st))} cells, {lineages(st)} seed lineages; "
              f"last 10 iterations: {sum(x.get('new', 0) for x in tail)} new, {sum(x.get('replaced', 0) for x in tail)} "
              f"replaced, {sum(x.get('duels', 0) for x in tail)} duels, {np.mean([x['sec'] for x in tail]):.0f}s per iteration")
    print(p)
    if st.get("sarch"):
        print(f"surprise archive: {len(st['sarch'])} members, threshold {st.get('sarch_thr'):.2f}")
        print(draw_sarch(out))


def check(a) -> None:
    """Treadmill check: for every cell whose elite changed since snapshot --at, the final elite vs the
    elite it had then (same island, same cell), judged by --judges in both orders."""
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    snaps = sorted(int(p.stem[len("archive_it"):]) for p in (out / "snapshots").glob("archive_it*.json"))
    last = st["it"] - 1
    at = a.at if a.at is not None else min(snaps, key=lambda s: abs(s - last // 2))
    old_maps = islands_of(json.loads((out / "snapshots" / f"archive_it{at:04d}.json").read_text()))
    new_maps = islands_of(st)
    K = len(new_maps)
    cdir = out / "check"
    (cdir / "images").mkdir(parents=True, exist_ok=True)
    pairs, pics = [], {}
    for k, (old, new) in enumerate(zip(old_maps, new_maps)):
        pre = f"I{k}" if K > 1 else ""
        for ck, e in sorted(new.items()):
            if ck in old and old[ck]["image"] != e["image"]:
                x, y = f"{pre}c{ck}_final", f"{pre}c{ck}_it{at:04d}"
                for nm, src in ((x, e["image"]), (y, old[ck]["image"])):
                    shutil.copy(out / "img" / src, cdir / "images" / f"{nm}.png")
                    pics[nm] = np.asarray(Image.open(cdir / "images" / f"{nm}.png").convert("RGB"))
                pairs.append([x, y])
    (cdir / "pairs.json").write_text(json.dumps(pairs))
    draw_map(out, cdir / f"map_it{at:04d}.png", old_maps, f"{out.name}: maps at iteration {at}")
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    res = {}
    for jn in a.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, cdir / "judge_cache.sqlite", yes=True)
        pid = entry_prompt(settings(st))
        qs = [q for x, y in pairs for q in (Query(pid, pics[x], pics[y]), Query(pid, pics[y], pics[x]))]
        vs = judge.judge_many(qs, desc=f"check {jn}") if qs else []
        res[jn] = [None if not (vs[2 * k].ok and vs[2 * k + 1].ok) else 0.5 * (vs[2 * k].p_first + 1 - vs[2 * k + 1].p_first)
                   for k in range(len(pairs))]
    json.dump({"at": at, "final_it": last, "pairs": pairs, "p_first_of_pair": res}, open(cdir / "check.json", "w"), indent=1)
    lines = [f"# {out.name}: does a niche's elite improve? (iteration {last} vs {at})\n",
             f"{len(pairs)} cells changed elite since iteration {at} (of {sum(len(m) for m in old_maps)} filled then, "
             f"{sum(len(m) for m in new_maps)} now; {lineages(st)} seed lineages now). "
             "P = probability the final elite is chosen as the parent, both orders.\n",
             "| judge | mean P(final) | final wins |", "|---|---|---|"]
    for jn, ps in res.items():
        v = [p for p in ps if p is not None]
        lines.append(f"| {jn} | {np.mean(v):.2f} | {sum(p > 0.5 for p in v)}/{len(v)} |" if v else f"| {jn} | - | - |")
    (cdir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def rate(a) -> None:
    from lenia import rating
    cdir = Path(a.out) / "check"
    n = len(json.loads((cdir / "pairs.json").read_text()))
    rating.rate(cdir, n=n, source="pairs.json",
                 question="Which world would you choose as the parent, to evolve toward more complex, life-like structure?",
                 description="Each image is one whole world at four times, far enough apart to show movement, in reading order.")


def main() -> None:
    floats = lambda s: [float(x) for x in s.split(",")]
    ints = lambda s: [int(x) for x in s.split(",")]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", default="numpy", help="numpy | auto | cuda | mps")
    r.add_argument("--judge", default="jev")
    r.add_argument("--config", default=None)
    r.add_argument("--iterations", type=int, default=150)
    r.add_argument("--batch", type=int, default=64, help="mutants per iteration (split evenly over islands)")
    r.add_argument("--init", type=int, default=52, help="random rules added to the 12 founders at iteration 0")
    r.add_argument("--init-seed", type=int, default=4)
    r.add_argument("--sim-batch", type=int, default=64, help="worlds simulated together (lower if the GPU runs out of memory)")
    r.add_argument("--size", type=int, default=160)
    r.add_argument("--R", type=float, default=13)
    r.add_argument("--beta", type=float, default=2.0)
    r.add_argument("--warmup", type=int, default=1500)
    r.add_argument("--render", type=int, default=1200)
    r.add_argument("--gap", type=int, default=400)
    r.add_argument("--start-seed", type=int, default=0)
    r.add_argument("--axes", type=lambda s: s.split(","), default=["bodies", "motion"],
                   help="comma list of bodies, form, motion; the last one is the columns of map.png")
    r.add_argument("--body-edges", type=ints, default=BODY_EDGES)
    r.add_argument("--motion-edges", type=floats, default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
    r.add_argument("--form-bins", type=int, default=4)
    r.add_argument("--embed-model", default="dinov2_vits14", help="dinov2_vits14 | fake (tests)")
    r.add_argument("--islands", type=int, default=1)
    r.add_argument("--migrate-every", type=int, default=10)
    r.add_argument("--max-texture", type=float, default=0.3)
    r.add_argument("--threshold", type=float, default=0.6)
    r.add_argument("--mut-scale", type=float, default=3.0)
    r.add_argument("--snapshot-every", type=int, default=10)
    r.add_argument("--surprise-opponents", type=int, default=6)
    r.add_argument("--sarch", action="store_true", help="keep surprising map rejects in a surprise archive (v4)")
    r.add_argument("--sarch-candidates", type=int, default=3, help="rejected challengers judged per island per iteration")
    r.add_argument("--sarch-parents", type=float, default=0.25, help="share of each island's parents drawn from its archive members")
    r.add_argument("--sarch-threshold", type=float, default=0.6, help="initial surprise score needed to join (adapts)")
    r.add_argument("--sarch-max", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--surprise-px", type=int, default=0, help="shrink the surprise duel's pictures to this many px per side "
                   "(box filter; 484 validated in gate 0f; 0 = full size, as every run before seed 1's full run)")
    r.add_argument("--ledger", action="store_true", help="write ledger.jsonl: every judged world (id, rule, descriptors, "
                   "the duel's opponent and P, placed or not) and every surprise judgment (full run, 11-05)")
    r.add_argument("--aim", default=None, help="the user's aim as the entry duel's goal (lenia.prompts.map_prompt) "
                   "instead of COMPLEXITY's \"complex, life-like\" (map-prompt test, 11-04)")
    r.add_argument("--like", default=None, help="copy every setting and the form-axis fit from this run's state.json "
                   "(all but --out, --backend, --judge, --config, --sim-batch, --max-usd, --aim): a one-change comparison")
    r.add_argument("--max-usd", type=float, default=6.0, help="hard cap on this run's judge spend, across resumes "
                   "(a 100-iteration --sarch run costs ~$3); raise it and re-run to continue")
    m = sub.add_parser("map")
    m.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--at", type=int, default=None, help="snapshot iteration to compare with (default: the one nearest half the run)")
    c.add_argument("--judges", default="luna,jev")
    c.add_argument("--config", default=None)
    t = sub.add_parser("rate")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        backend.configure(a.backend)
        assert all(x in ("bodies", "form", "motion") for x in a.axes), a.axes
    {"run": run, "map": show_map, "check": check, "rate": rate}[a.cmd](a)


if __name__ == "__main__":
    main()
