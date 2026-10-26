"""MAP-Elites over MaCE-Lenia worlds, with Jev duels as the quality signal.

  python scripts/mace_qd.py run   --out RUNS/mace-qd --backend auto     # v1 (bodies x motion, one map)
  python scripts/mace_qd.py run   --out RUNS/mace-qd2 --backend auto --islands 4 --axes bodies,form,motion \\
      --body-edges 3,9 --motion-edges 0.75,1.0,1.25                     # v2: islands + a form axis
  python scripts/mace_qd.py map   --out RUNS/mace-qd                    # map.png + summary
  python scripts/mace_qd.py check --out RUNS/mace-qd                    # final elites vs elites at a snapshot
  python scripts/mace_qd.py rate  --out RUNS/mace-qd                    # your ratings of the check pairs (Mac)
  python scripts/mace_qd.py run   --out /tmp/qd-dry --judge fake --embed-model fake --iterations 3 --batch 8 \\
      --init 8 --size 96 --warmup 200 --render 120 --gap 40             # offline dry run

Why MAP-Elites (project doc "MaCE-Lenia Jev-guided evolution results"): the hill climber (mace_evo.py)
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
- Variation: kernel mutation at --mut-scale plus beta x exp(N(0, 0.1)) in [0.5, 4], as in mace_evo.
- Seeds: the 12 mace_evo founders plus --init more draws of the same sampler. Every elite records the seed
  it descends from ("root").
- Parent selection (--parent-select): "uniform" (v1, v2) picks parents uniformly from the island's elites.
  "surprise" (v3) scores every elite with Jev's surprise duel (surprise_check.py, validated against Felix's
  ratings: "which world shows something more unusual and interesting ..."), each elite against
  --surprise-opponents random other elites of the current archive (all islands), both orders; new elites are
  scored when they enter and the whole archive is re-scored every --surprise-every iterations, so "unusual"
  always means unusual relative to what the archive holds now. A share --surprise-mix of parents is still drawn
  uniformly; the rest with probability proportional to exp(score / --surprise-temp). Surprise only decides where
  to search; entry into a cell is still the complexity duel against the incumbent. --surprise-ref other-lineages
  draws the opponents only from other seed lineages, so a lineage that spreads over many cells is not made
  "typical" by its own relatives and keeps being refined.
Saves state after every iteration (resumes) and a snapshot of the maps every --snapshot-every iterations.
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

import mace_evo as ME
from lenia import backend, render, seeds
from lenia.alive import bodies as label_bodies
from lenia.config import REPO_ROOT, load
from lenia.explore import mutate, texture
from lenia.judge import Judge, Query, load_env
from lenia.sample import sample
from lenia.sim import Rule, with_flow

PROMPT_ID = ME.PROMPT_ID
BODY_EDGES = [2, 3, 5, 9, 17, 33]                     # v1 bins: 1 | 2 | 3-4 | 5-8 | 9-16 | 17-32 | 33+


# ---------- run settings (v1 runs predate some options) ----------

def settings(st: dict) -> argparse.Namespace:
    a = argparse.Namespace(**{"axes": ["bodies", "motion"], "body_edges": BODY_EDGES, "islands": 1,
                              "migrate_every": 10, "form_bins": 4, "embed_model": "dinov2_vits14",
                              "parent_select": "uniform", "surprise_opponents": 6, "surprise_every": 20,
                              "surprise_mix": 0.5, "surprise_temp": 0.1, "surprise_ref": "all", **st["args"]})
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
        from lenia.distill import Embedder
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


def insert(st: dict, k: int, cands: list, evals: list, judge, out: Path, it: int, log, a) -> dict:
    """Into island k. cands: [(rule, parent label or None, root seed)]. Returns counts."""
    arch = islands_of(st)[k]
    imgdir = out / "img"
    tag = f"I{k}_" if a.islands > 1 else ""
    by_cell: dict = {}
    n_tex = 0

    def row(n, ck, kind, p, acc):
        e = evals[n]
        log.writerow([it, k, n, cands[n][1] or "", ck, kind, "" if p is None else round(p, 4), int(acc), e["bodies"],
                      round(e["motion"], 4), round(e.get("form", float("nan")), 4), round(e["texture"], 4),
                      round(cands[n][0].beta, 3), cands[n][2]])

    for n, e in enumerate(evals):
        if a.max_texture > 0 and e["texture"] > a.max_texture:
            n_tex += 1
            row(n, "", "texture", None, 0)
            continue
        by_cell.setdefault(cell_of(e, a, st), []).append(n)
    duels, first_new = [], {}
    for ck, ns in by_cell.items():
        if ck in arch:
            inc = np.asarray(Image.open(imgdir / arch[ck]["image"]).convert("RGB"))
            duels += [(ck, n, inc) for n in ns]
        else:
            first_new[ck] = ns[0]
            duels += [(ck, n, evals[ns[0]]["pic"]) for n in ns[1:]]
    qs = []
    for ck, n, inc in duels:
        qs += [Query(PROMPT_ID, evals[n]["pic"], inc), Query(PROMPT_ID, inc, evals[n]["pic"])]
    vs = judge.judge_many(qs, desc=f"it {it} {tag.rstrip('_') or 'map'}") if qs else []
    best = {}
    for d, (ck, n, _) in enumerate(duels):
        va, vb = vs[2 * d], vs[2 * d + 1]
        p = None if not (va.ok and vb.ok) else 0.5 * (va.p_first + 1 - vb.p_first)
        acc = p is not None and p >= a.threshold
        if acc and p > best.get(ck, (0.0, None))[0]:
            best[ck] = (p, n)
        row(n, ck, "duel" if ck in arch else "duel_new", p, acc)

    def place(ck, n, p):
        rule, parent, root = cands[n]
        e = evals[n]
        name = f"it{it:04d}_{tag}{n:03d}.png"
        if not (imgdir / name).exists():
            Image.fromarray(e["pic"]).save(imgdir / name)
        prev = arch.get(ck)
        arch[ck] = {"rule": rule.to_dict(), "image": name, "it": it, "parent": parent, "root": root,
                    "bodies": e["bodies"], "motion": round(e["motion"], 4), "texture": round(e["texture"], 4),
                    "coverage": round(e["coverage"], 4), "p": None if p is None else round(p, 4),
                    "replaced": 0 if prev is None else prev.get("replaced", 0) + 1,
                    **({"form": round(e["form"], 4)} if "form" in e else {})}

    new_cells = replaced = 0
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
    return {"texture_fail": n_tex, "new": new_cells, "replaced": replaced, "duels": len(duels)}


def lineages(st: dict) -> int:
    return len({e.get("root") for arch in islands_of(st) for e in arch.values()})


# ---------- surprise-weighted parent selection (v3) ----------

def score_surprise(st: dict, which: list, judge, out: Path, rng, a, desc: str) -> int:
    """Jev surprise score for the elites `which` ([(island, cell)]) against random other elites of the current
    archive; stored as e["surprise"]. Returns the number of judge calls."""
    from surprise_check import SURPRISE
    allk = [(k, ck) for k, m in enumerate(islands_of(st)) for ck in m]
    if len(allk) < 2 or not which:
        return 0
    cache = {}

    def pic(k, ck):
        name = islands_of(st)[k][ck]["image"]
        if name not in cache:
            cache[name] = np.asarray(Image.open(out / "img" / name).convert("RGB"))
        return cache[name]

    qs, meta = [], []
    for k, ck in which:
        others = [x for x in allk if x != (k, ck)]
        if a.surprise_ref == "other-lineages":            # judge a lineage only against other lineages, so its own
            root = islands_of(st)[k][ck].get("root")      # relatives filling the map do not make it "typical"
            diff = [x for x in others if islands_of(st)[x[0]][x[1]].get("root") != root]
            others = diff if len(diff) >= a.surprise_opponents else others
        for j in rng.choice(len(others), min(a.surprise_opponents, len(others)), replace=False):
            o = others[int(j)]
            qs += [Query(SURPRISE, pic(k, ck), pic(*o)), Query(SURPRISE, pic(*o), pic(k, ck))]
            meta.append((k, ck))
    vs = judge.judge_many(qs, desc=desc)
    acc = {}
    for d, key in enumerate(meta):
        x, y = vs[2 * d], vs[2 * d + 1]
        if x.ok and y.ok:
            acc.setdefault(key, []).append(0.5 * (x.p_first + 1 - y.p_first))
    it = st["it"]
    for (k, ck), v in acc.items():
        e = islands_of(st)[k][ck]
        e["surprise"], e["surprise_it"] = round(float(np.mean(v)), 4), it
    return len(qs)


def pick_parent(arch: dict, keys: list, rng, a) -> str:
    if a.parent_select != "surprise" or rng.random() < a.surprise_mix:
        return keys[int(rng.integers(len(keys)))]
    sc = np.array([arch[k].get("surprise", np.nan) for k in keys], float)
    sc = np.where(np.isnan(sc), np.nanmedian(sc) if np.isfinite(sc).any() else 0.5, sc)
    w = np.exp((sc - sc.max()) / a.surprise_temp)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def run(a) -> None:
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge]}, out / "judge_cache.sqlite", yes=True)
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        keep = ("iterations", "backend", "judge", "out", "config", "sim_batch")
        if "parent_select" not in st["args"]:                    # older runs: uniform parent selection
            st["args"]["parent_select"] = "uniform"
        st["args"].update({k: getattr(a, k) for k in keep})
        a = settings(st)
        print(f"resuming at iteration {st['it']} ({sum(len(m) for m in islands_of(st))} cells filled)", file=sys.stderr)
    else:
        st = {"it": 0, "args": {**vars(a), "axes": a.axes}, "prompt": ME.PROMPTS[PROMPT_ID], "progress": []}
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
        else:
            groups = []
            for k, arch in enumerate(islands_of(st)):
                keys = sorted(arch)
                g = []
                for _ in range(a.batch // K if keys else 0):
                    pk = pick_parent(arch, keys, rng, a)
                    par = Rule.from_dict(arch[pk]["rule"])
                    m = mutate(par, rng, a.mut_scale)
                    m.flow = "mace"
                    m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                    g.append((m, f"I{k}:{pk}" if K > 1 else pk, arch[pk].get("root")))
                groups.append(g)
        flat = [c for g in groups for c in g]
        evals = evaluate([c[0] for c in flat], a)
        if it == 0 and "form" in a.axes:
            st["form"] = fit_form(evals, a)
        add_form(evals, st)
        res = Counter()
        pos = 0
        for k, g in enumerate(groups):
            res.update(insert(st, k, g, evals[pos:pos + len(g)], judge, out, it, log, a))
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
                r = insert(st, k, [(Rule.from_dict(e["rule"]), f"migrant I{s}:{ck}", e.get("root"))], [ev], judge, out, it, log, a)
                res["migrated"] += r["new"] + r["replaced"]
        if a.parent_select == "surprise":
            srng = np.random.default_rng([a.seed, it, 7])
            if it % a.surprise_every == 0:
                which = [(k, ck) for k, m in enumerate(islands_of(st)) for ck in m]
            else:
                which = [(k, ck) for k, m in enumerate(islands_of(st)) for ck, e in m.items() if "surprise" not in e]
            res["surprise_calls"] += score_surprise(st, which, judge, out, srng, a, f"it {it} surprise")
            sv = [e["surprise"] for m in islands_of(st) for e in m.values() if "surprise" in e]
            res["surprise_mean_x1000"] = int(1000 * np.mean(sv)) if sv else 0
        fh.flush()
        filled = sum(len(m) for m in islands_of(st))
        st["progress"].append({"it": it, "filled": filled, "lineages": lineages(st), **dict(res), "sec": round(time.time() - t0, 1)})
        if it % a.snapshot_every == 0:
            snap = st["islands"] if K > 1 else st["archive"]
            (out / "snapshots" / f"archive_it{it:04d}.json").write_text(json.dumps(snap))
        st["it"] = it + 1
        save_state(out, st)
        print(f"it {it}: {filled} cells filled, {lineages(st)} seed lineages, {res['new']} new, {res['replaced']} replaced, "
              f"{res['duels']} duels, {res['texture_fail']} grid-noise"
              + (f", {res['migrated']} migrants settled" if K > 1 and it % a.migrate_every == 0 and it else "")
              + (f", archive surprise {res['surprise_mean_x1000'] / 1000:.2f}" if a.parent_select == "surprise" else "")
              + f", {time.time() - t0:.0f}s, judge spent ${judge.spent:.4f}", file=sys.stderr)
    fh.close()
    draw_map(out)


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
        qs = [q for x, y in pairs for q in (Query(PROMPT_ID, pics[x], pics[y]), Query(PROMPT_ID, pics[y], pics[x]))]
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
    from lenia import vlmtest
    cdir = Path(a.out) / "check"
    n = len(json.loads((cdir / "pairs.json").read_text()))
    vlmtest.rate(cdir, n=n, source="pairs.json",
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
    r.add_argument("--parent-select", default="uniform", choices=["uniform", "surprise"])
    r.add_argument("--surprise-opponents", type=int, default=6)
    r.add_argument("--surprise-every", type=int, default=20, help="re-score the whole archive every N iterations")
    r.add_argument("--surprise-mix", type=float, default=0.5, help="share of parents still drawn uniformly")
    r.add_argument("--surprise-temp", type=float, default=0.1)
    r.add_argument("--surprise-ref", default="all", choices=["all", "other-lineages"],
                   help="opponents for the surprise duel: any elite (v3), or only elites of other seed lineages")
    r.add_argument("--seed", type=int, default=0)
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
