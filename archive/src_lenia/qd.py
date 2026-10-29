"""Phase 5: quality-diversity search (MAP-Elites) against a reward model, under the alive constraint.

Why: the greedy search (rsearch) let both rewards win mostly by making creatures bigger, the teacher's
first preference, and collapsed lineages onto one body plan. Here size and speed (the teacher's first and
second preferences) are the map's axes, so a creature can only improve its cell by being better at the
same size and speed: search pressure goes into whatever else the reward learned. Comparisons inside a cell
are size-matched by construction (bins are about 1.2x wide), where the student is weakest.

run     (Colab, GPU) the map: `bins` x `bins` cells over log size (radius of gyration, R units) and log
        speed (R per time unit), ranges from the pool. Seeded with every pool creature (rendered again,
        scored, placed). Each iteration draws `batch` parents uniformly from occupied cells, mutates them
        (started from the parent's pattern), runs the frozen alive check (alive v2, as in rsearch),
        renders and scores the viable mutants, and offers each to the cell its size and speed fall in:
        it takes the cell if the cell is empty or it scores higher than the elite, and is still alive at
        200 units. Quality: `student` (distilled image reward) or `features` (five-statistic reward).
        Every insertion is stored (image, rule, start pattern, both scores, the student's ensemble spread, the
        distance to the nearest pool creature in embedding space, statistics).
        Resumable.
check   (Mac, API keys) the teacher and a second judge on:
          within-cell: each improved cell's first occupant vs its elite at checkpoint iterations (the RQ1
            curve at matched size and speed, plus win rate against student gain at the end: high-gain cells
            are where over-optimisation should show first)
          across the map: random pairs of final elites (does the student still rank creatures it was never
            trained on, from far corners of the map, as the teacher does?)
        and pairs for you to rate (first occupant vs final elite).
draw    map.png (score per cell, seed vs final), elites.png (each cell's elite in place), progress.png.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .alive import Thresholds
from .describe import Space
from .explore import evaluate, mutate
from .rollout import Horizon
from .rsearch import STATS, _camera, _render, _save, viable_mutants
from .sim import Rule

CHECKPOINTS = [5, 10, 20, 40, 80, 120, 160]
AXES = {"size": "size_rg", "speed": "speed", "change": "change", "mass": "mass"}
# map ranges (log-spaced); values outside fall into the end cells
RANGES = {"size": (0.35, 2.5), "speed": (0.003, 0.5), "change": (0.03, 0.8), "mass": (0.4, 12.0)}


class Grid:
    def __init__(self, axes=("size", "speed"), bins: int = 10, ranges: dict | None = None):
        self.axes, self.bins = list(axes), int(bins)
        rg = {**RANGES, **(ranges or {})}
        self.edges = [np.geomspace(*rg[a], self.bins + 1)[1:-1] for a in self.axes]

    def cell(self, row: dict):
        v = [row.get(AXES[a]) for a in self.axes]
        if any(x is None or not np.isfinite(x) for x in v):
            return None
        return ",".join(str(int(np.digitize(max(x, 1e-9), e))) for x, e in zip(v, self.edges))

    def to_json(self) -> dict:
        return {"axes": self.axes, "bins": self.bins, "edges": [e.tolist() for e in self.edges]}


class _Scorer:
    """Both rewards, the student's ensemble spread, and how far a creature is from the pool the student was
    trained on (cosine distance to the nearest pool creature, in the student's standardised embedding)."""

    def __init__(self, student_dir: Path, pool_dir: Path, reward: str):
        from .distill import Student
        self.st, self.reward = Student(student_dir), reward
        p = Path(pool_dir) / f"embed_{self.st.model}.npz"
        self.ref = None
        if p.exists():
            R = (np.load(p)["emb"].astype(np.float64) - self.st.mu) / self.st.sd
            self.ref = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)

    def __call__(self, images: list, rows: list) -> dict:
        n = len(images)
        f = self.st.score_features(rows) if rows else np.zeros(0)
        try:
            if self.st._embedder is None:
                from .distill import Embedder
                self.st._embedder = Embedder(self.st.model)
            E = self.st._embedder(images) if n else np.zeros((0, len(self.st.mu)))
            mean, spread = self.st.score_embeddings(E) if n else (np.zeros(0), np.zeros(0))
            nn = np.full(n, np.nan)
            if self.ref is not None and n:
                X = (np.asarray(E, np.float64) - self.st.mu) / self.st.sd
                X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
                nn = 1 - (X @ self.ref.T).max(1)
        except ImportError:                 # no PyTorch: only the five-statistic reward can drive the search
            if self.reward == "student":
                raise
            mean = spread = nn = np.full(n, np.nan)
            E = None
        main = mean if self.reward == "student" else f
        return {"main": main, "student": mean, "spread": spread, "features": f, "nn": nn, "emb": E}

    def unit(self, E) -> np.ndarray:
        """Embeddings in the student's standardised space, unit length (cosine distance = 1 - dot)."""
        X = (np.asarray(E, np.float64) - self.st.mu) / self.st.sd
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def _event(it: int, cell: str, name: str, rule: Rule, pkey: str, sc: dict, j: int, row: dict, kind: str, parent=None, extra=None):
    e = {"it": it, "cell": cell, "image": name, "rule": rule.to_json(), "pattern": pkey, "kind": kind, "parent": parent,
         **{k: float(sc[k][j]) for k in ("main", "student", "spread", "features", "nn")},
         **{s: (None if row[s] is None else float(row[s])) for s in STATS}}
    e.update(extra or {})
    return e


def _qd_stats(state: dict) -> dict:
    el = [state["events"][v[-1]] for v in state["cells"].values()]
    sc = np.array([e["main"] for e in el], float)
    base = state["config"]["score_floor"]
    return {"cells": len(el), "mean_elite": round(float(np.nanmean(sc)), 4) if len(sc) else None,
            "qd_score": round(float(np.nansum(sc - base)), 3)}


def run(out: Path, pool_dir: Path, survey: Path, archive: Path, student_dir: Path, reward: str = "student",
        iterations: int = 80, batch_size: int = 64, axes=("size", "speed"), bins: int = 10, mut_scale: float = 1.0,
        seed: int = 0, shape=(256, 256), th: Thresholds = None, batch: int = 64, ranges=None, log=print) -> dict:
    """ranges: None (the fixed RANGES), {axis: (lo, hi)}, or "pool" (the pool's 5th-95th percentiles)."""
    from .found import calibration
    from .pool import _item_of
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    th = th or Thresholds()
    cal = calibration(survey)
    space = Space(np.load(Path(survey) / "alive.npz")["desc"])
    max_dist, max_loss = cal["rescale_max_dist"], cal.get("texture_max_loss")
    scorer = _Scorer(student_dir, pool_dir, reward)
    camera = _camera(pool_dir)
    hz1 = Horizon(time=100.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    hz2 = Horizon(time=200.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    t0 = time.time()
    if (out / "state.json").exists():
        state = json.loads((out / "state.json").read_text())
        pz = np.load(out / "patterns.npz")
        patterns = {k: pz[k] for k in pz.files}
        grid = Grid(state["config"]["axes"], state["config"]["bins"])
        grid.edges = [np.asarray(e) for e in state["config"]["grid"]["edges"]]
        log(f"resuming at iteration {state['it']} ({len(state['cells'])} cells)")
    else:
        rows = list(csv.DictReader(open(Path(pool_dir) / "creatures.csv")))
        if ranges == "pool":
            ranges = {a: tuple(np.quantile([max(float(r[AXES[a]]), 1e-4) for r in rows], [0.05, 0.95])) for a in axes}
        grid = Grid(axes, bins, ranges)
        cache = {}
        items = [_item_of(r, survey, archive, cache) for r in rows]
        log(f"seeding the map with {len(items)} pool creatures")
        rend = _render([it[2] for it in items], [it[3] for it in items], shape, camera=camera)
        sc = scorer([r[0] for r in rend], [r[1] for r in rend])
        from .alive import ALIVE
        state = {"it": 0, "cells": {}, "events": [], "history": [],
                 "config": {"reward": reward, "student": str(student_dir), "pool": str(pool_dir), "camera": camera,
                            "axes": list(axes), "bins": bins, "grid": grid.to_json(), "iterations": iterations,
                            "batch_size": batch_size, "mut_scale": mut_scale, "seed": seed, "rescale_max_dist": max_dist,
                            "texture_max_loss": max_loss,
                            "score_floor": float(np.nanmin(sc["main"])) if len(sc["main"]) else 0.0}}
        patterns, best = {}, {}
        for j, (row, it, (img, st, _)) in enumerate(zip(rows, items, rend)):
            c = grid.cell(st)
            if st["class"] not in ALIVE or c is None:
                continue
            if c not in best or sc["main"][j] > sc["main"][best[c]]:
                best[c] = j
        for c, j in sorted(best.items()):
            name = f"seed_{rows[j]['id']}.png"
            rend[j][0].save(out / "images" / name)
            pkey = f"e{len(state['events']):05d}"
            patterns[pkey] = np.asarray(items[j][3], np.float32)
            state["events"].append(_event(0, c, name, items[j][2], pkey, sc, j, rend[j][1], "seed",
                                          extra={"pool_id": rows[j]["id"], "source": rows[j]["source"]}))
            state["cells"][c] = [len(state["events"]) - 1]
        state["history"].append({"it": 0, **_qd_stats(state), "seconds": round(time.time() - t0, 1)})
        _save(out, state, patterns)
        log(f"seeded: {len(state['cells'])} of {bins ** len(axes)} cells (reward: {reward})")
    cfg = state["config"]
    for it in range(state["it"] + 1, iterations + 1):
        rng = np.random.default_rng([seed, it])
        occupied = sorted(state["cells"])
        pc = [occupied[k] for k in rng.integers(0, len(occupied), batch_size)]
        pe = [state["events"][state["cells"][c][-1]] for c in pc]
        rules = [mutate(Rule.from_dict(json.loads(e["rule"])), rng, mut_scale) for e in pe]
        starts = [patterns[e["pattern"]] for e in pe]
        r1, ok, viable, rend = viable_mutants(rules, starts, shape, th, batch, space, max_dist, max_loss, hz1, camera)
        sc = scorer([r[0] for r in rend], [r[1] for r in rend]) if rend else None
        offers = {}
        for j, k in enumerate(viable):
            c = grid.cell(rend[j][1])
            if c is None or not np.isfinite(sc["main"][j]):
                continue
            cur = state["events"][state["cells"][c][-1]]["main"] if c in state["cells"] else -np.inf
            if sc["main"][j] > cur and (c not in offers or sc["main"][j] > sc["main"][offers[c]]):
                offers[c] = j
        cand = sorted(offers.items())
        conf = evaluate([rules[viable[j]] for _, j in cand], [r1[viable[j]][1] for _, j in cand], shape, hz2, th, None, batch) if cand else []
        new = imp = 0
        for (c, j), (_, pat2, _) in zip(cand, conf):
            if pat2 is None:
                continue
            k = viable[j]
            eid = len(state["events"])
            name, pkey = f"i{it:03d}_e{eid:05d}.png", f"e{eid:05d}"
            rend[j][0].save(out / "images" / name)
            patterns[pkey] = np.asarray(pat2, np.float32)
            kind = "improve" if c in state["cells"] else "new"
            if kind == "improve":                  # replaced elites keep their pattern too (a few KB each), so any
                imp += 1                           # creature in a cell's history can be replayed later
            else:
                new += 1
            state["events"].append(_event(it, c, name, rules[k], pkey, sc, j, rend[j][1], kind, parent=pc[k]))
            state["cells"].setdefault(c, []).append(eid)
        state["history"].append({"it": it, "mutants": len(rules), "alive": len(ok), "viable": len(viable),
                                 "offers": len(cand), "new": new, "improved": imp, **_qd_stats(state),
                                 "seconds": round(time.time() - t0, 1)})
        state["it"] = it
        _save(out, state, patterns)
        h = state["history"][-1]
        log(f"it {it}: {h['mutants']} mutants, alive {h['alive']}, viable {h['viable']}, new cells {new}, improved {imp}; "
            f"{h['cells']} cells, mean elite {h['mean_elite']:+.3f}, QD score {h['qd_score']:.1f}; {h['seconds']:.0f}s")
    draw(out)
    return state


# ---------- pictures ----------

def elite_at(state: dict, cell: str, it: int):
    ev = [state["events"][k] for k in state["cells"][cell] if state["events"][k]["it"] <= it]
    return ev[-1] if ev else None


def _matrix(state: dict, it: int, key: str = "main") -> np.ndarray:
    b = state["config"]["bins"]
    M = np.full((b, b), np.nan)
    for c in state["cells"]:
        e = elite_at(state, c, it)
        if e is not None:
            i, j = (int(x) for x in c.split(","))
            M[i, j] = e[key]
    return M


def draw(out: Path, px: int = 128) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    st = json.loads((out / "state.json").read_text())
    cfg, b = st["config"], st["config"]["bins"]
    ax_names = cfg["axes"]
    edges = cfg["grid"]["edges"]
    lab = lambda e: [f"<{e[0]:.2g}"] + [f"{lo:.2g}-{hi:.2g}" for lo, hi in zip(e[:-1], e[1:])] + [f">{e[-1]:.2g}"]
    paths = []
    # map: score per cell, seed vs final
    M0, M1 = _matrix(st, 0), _matrix(st, st["it"])
    lo, hi = np.nanmin([np.nanmin(M0), np.nanmin(M1)]), np.nanmax([np.nanmax(M0), np.nanmax(M1)])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, M, t in zip(axs, (M0, M1), (f"seed: pool creatures ({np.isfinite(M0).sum()} cells)",
                                       f"after {st['it']} iterations ({np.isfinite(M1).sum()} cells)")):
        im = a.imshow(M, origin="lower", cmap="Blues", vmin=lo, vmax=hi)
        a.set_title(t, fontsize=10, loc="left")
        a.set_ylabel(ax_names[0]); a.set_xlabel(ax_names[1])
        a.set_yticks(range(b)); a.set_yticklabels(lab(edges[0]), fontsize=7)
        a.set_xticks(range(b)); a.set_xticklabels(lab(edges[1]), fontsize=7, rotation=60)
    fig.colorbar(im, ax=axs, shrink=0.8, label=f"{cfg['reward']} score of the cell's elite")
    fig.suptitle(f"MAP-Elites ({cfg['reward']} reward): {ax_names[0]} x {ax_names[1]}", x=0.01, ha="left")
    fig.savefig(out / "map.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    paths.append(out / "map.png")
    # elites in place (rows: first axis, largest at the top; columns: second axis)
    W = b * (px + 4) + 70
    H = b * (px + 4) + 60
    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), f"Elites after {st['it']} iterations ({cfg['reward']}): rows {ax_names[0]} (up = larger), "
                   f"columns {ax_names[1]} (right = larger); label = score", fill=(20, 20, 20))
    for c in st["cells"]:
        e = elite_at(st, c, st["it"])
        i, j = (int(x) for x in c.split(","))
        x, y = 66 + j * (px + 4), 24 + (b - 1 - i) * (px + 4)
        im = Image.open(out / "images" / e["image"])
        im = im.crop((0, 0, im.height, im.height)) if im.width > im.height * 1.5 else im   # the motion view only
        sheet.paste(im.resize((px, px), Image.BILINEAR), (x, y))
        d.text((x + 2, y + 2), f"{e['main']:+.2f}" + ("" if e["kind"] == "seed" else " *"), fill=(255, 255, 255))
    for i in range(b):
        d.text((4, 24 + (b - 1 - i) * (px + 4) + px // 2), lab(edges[0])[i], fill=(60, 60, 60))
    for j in range(b):
        d.text((66 + j * (px + 4), H - 30), lab(edges[1])[j], fill=(60, 60, 60))
    d.text((6, H - 14), "* = found by the search (else a pool creature)", fill=(60, 60, 60))
    sheet.save(out / "elites.png")
    paths.append(out / "elites.png")
    # progress
    h = st["history"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, k, t in zip(axs, ("cells", "mean_elite", "qd_score"), ("cells filled", "mean elite score", "QD score")):
        a.plot([r["it"] for r in h], [r[k] for r in h], color="#2a78d6", lw=2)
        a.set_title(t, fontsize=10, loc="left"); a.set_xlabel("iteration")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "progress.png", dpi=130, facecolor="white"); plt.close(fig)
    paths.append(out / "progress.png")
    return paths


# ---------- check ----------

def _sig(x):
    return 1 / (1 + math.exp(-x))


def check(out: Path, judges: dict, prompt: str = "curate_dual", n_random: int = 150, max_cells: int | None = None,
          seed: int = 0, log=print) -> dict:
    """judges: {name: Judge}. Within-cell (first occupant vs elite at checkpoints) and across-map pairs."""
    from .judge import Query
    out = Path(out)
    st = json.loads((out / "state.json").read_text())
    G = st["it"]
    ev = st["events"]
    cps = [c for c in CHECKPOINTS if c < G] + [G]
    rng = np.random.default_rng(seed)
    cells = [c for c in sorted(st["cells"]) if len(st["cells"][c]) > 1]
    if max_cells and len(cells) > max_cells:
        cells = sorted(rng.choice(cells, max_cells, replace=False).tolist())
    within = []                                  # (cell, checkpoint, first, elite)
    for c in cells:
        f = ev[st["cells"][c][0]]
        for cp in cps:
            e = elite_at(st, c, cp)
            if e is not None and e["image"] != f["image"]:
                within.append((c, cp, f, e))
    final = [elite_at(st, c, G) for c in sorted(st["cells"])]
    across = []
    if len(final) > 1:
        for _ in range(n_random):
            a, b = rng.choice(len(final), 2, replace=False)
            across.append((final[a], final[b]))
    load = lambda n: np.asarray(Image.open(out / "images" / n).convert("RGB"))
    P = {}
    for name, j in judges.items():
        pairs = [(e, f) for _, _, f, e in within] + across
        qs = []
        for A, B in pairs:
            qs += [Query(prompt, load(A["image"]), load(B["image"])), Query(prompt, load(B["image"]), load(A["image"]))]
        vs = j.judge_many(qs, desc=f"qd check {name}") if qs else []
        for k, (A, B) in enumerate(pairs):
            v1, v2 = vs[2 * k], vs[2 * k + 1]
            P[(name, A["image"], B["image"])] = (v1.p_first + 1 - v2.p_first) / 2 if v1.ok and v2.ok else np.nan
    rows = []
    for c, cp, f, e in within:
        for name in judges:
            rows.append({"part": "within", "judge": name, "cell": c, "checkpoint": cp, "first": f["image"], "elite": e["image"],
                         "elite_it": e["it"], "replacements": sum(ev[k]["it"] <= cp for k in st["cells"][c]) - 1,
                         "p_elite_beats_first": P[(name, e["image"], f["image"])],
                         "student_gain": e["student"] - f["student"], "features_gain": e["features"] - f["features"],
                         "spread": e["spread"], "nn": e["nn"],
                         **{f"{s}_first": f[s] for s in STATS}, **{s: e[s] for s in STATS}})
    for A, B in across:
        for name in judges:
            rows.append({"part": "across", "judge": name, "cell": f"{A['cell']}|{B['cell']}", "first": B["image"], "elite": A["image"],
                         "p_elite_beats_first": P[(name, A["image"], B["image"])],
                         "student_gain": A["student"] - B["student"], "features_gain": A["features"] - B["features"],
                         "same_size_row": A["cell"].split(",")[0] == B["cell"].split(",")[0]})
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "check.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    res = {"iterations": G, "reward": st["config"]["reward"], "cells_checked": len(cells), "curve": [], "by_gain": [], "across": {}}
    for cp in cps:
        rr = [r for r in rows if r["part"] == "within" and r["checkpoint"] == cp]
        row = {"checkpoint": cp, "cells": len({r["cell"] for r in rr})}
        if rr:
            one = [r for r in rr if r["judge"] == list(judges)[0]]
            row["student_gain"] = round(float(np.mean([r["student_gain"] for r in one])), 3)
            row["features_gain"] = round(float(np.mean([r["features_gain"] for r in one])), 3)
            row["student_predicts"] = round(float(np.mean([_sig(r["student_gain"]) for r in one])), 3)
            for name in judges:
                ps = [r["p_elite_beats_first"] for r in rr if r["judge"] == name]
                row[f"{name}_p"] = round(float(np.nanmean(ps)), 3)
        res["curve"].append(row)
    fin = [r for r in rows if r["part"] == "within" and r["checkpoint"] == G]
    if fin:
        g = np.array([r["student_gain"] for r in fin if r["judge"] == list(judges)[0]])
        qs = np.quantile(g, [0, 1 / 3, 2 / 3, 1]) if len(g) >= 6 else np.array([g.min(), g.max()])
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = [r for r in fin if lo <= r["student_gain"] <= hi]
            row = {"gain_range": f"{lo:+.2f} to {hi:+.2f}", "cells": len({r["cell"] for r in sel}),
                   "student_predicts": round(float(np.mean([_sig(r["student_gain"]) for r in sel])), 3)}
            for name in judges:
                row[f"{name}_p"] = round(float(np.nanmean([r["p_elite_beats_first"] for r in sel if r["judge"] == name])), 3)
            res["by_gain"].append(row)
    for name in judges:
        rr = [r for r in rows if r["part"] == "across" and r["judge"] == name and np.isfinite(r["p_elite_beats_first"])]
        for lab, sel in (("all", rr), ("same size row", [r for r in rr if r["same_size_row"]])):
            if sel:
                res["across"].setdefault(name, {})[lab] = {
                    "n": len(sel),
                    "student_agrees": round(float(np.mean([(r["student_gain"] > 0) == (r["p_elite_beats_first"] > 0.5) for r in sel])), 3),
                    "features_agrees": round(float(np.mean([(r["features_gain"] > 0) == (r["p_elite_beats_first"] > 0.5) for r in sel])), 3)}
    # pairs for you: first occupant vs final elite, up to 24 improved cells
    hp = []
    for c in cells[:]:
        f, e = ev[st["cells"][c][0]], elite_at(st, c, G)
        if e["image"] == f["image"]:
            continue
        a, b = f["image"][:-4], e["image"][:-4]
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        hp.append({"a": a, "b": b, "left": left, "right": right, "kind": "first_vs_elite", "cell": c})
    rng.shuffle(hp)
    (out / "human_pairs.json").write_text(json.dumps(hp[:24], indent=1))
    hum = [r for r in csv.DictReader(open(out / "human.csv"))] if (out / "human.csv").exists() else []
    won = [r["winner"] == r["b"] for r in hum if r["choice"] in ("left", "right")]
    res["you_elite_beats_first"] = (round(float(np.mean(won)), 3), len(won)) if won else None
    (out / "check.json").write_text(json.dumps(res, indent=2))
    _md(out, res, st, list(judges))
    log((out / "check.md").read_text())
    return res


def _md(out: Path, r: dict, st: dict, judges: list) -> None:
    cfg = st["config"]
    L = [f"# MAP-Elites ({cfg['reward']} reward, {' x '.join(cfg['axes'])}): teacher check", "",
         f"{r['iterations']} iterations x {cfg['batch_size']} mutants; {len(st['cells'])} cells filled; "
         f"{r['cells_checked']} improved cells checked. p = P(elite beats the cell's first occupant), both orders.", "",
         "## Within cells (matched size and speed), by iteration", "",
         "| iteration | cells | student gain | student predicts | features gain | " + " | ".join(f"{j} p" for j in judges) + " |",
         "|---" * (5 + len(judges)) + "|"]
    for c in r["curve"]:
        if "student_gain" in c:
            L.append(f"| {c['checkpoint']} | {c['cells']} | {c['student_gain']:+.3f} | {c['student_predicts']:.3f} | "
                     f"{c['features_gain']:+.3f} | " + " | ".join(f"{c[f'{j}_p']:.3f}" for j in judges) + " |")
    L += ["", "## Within cells at the end, by student gain (where over-optimisation should show first)", "",
          "| student gain | cells | student predicts | " + " | ".join(f"{j} p" for j in judges) + " |",
          "|---" * (3 + len(judges)) + "|"]
    for c in r["by_gain"]:
        L.append(f"| {c['gain_range']} | {c['cells']} | {c['student_predicts']:.3f} | " + " | ".join(f"{c[f'{j}_p']:.3f}" for j in judges) + " |")
    L += ["", "## Across the map (random pairs of final elites)", "",
          "| judge | pairs | n | student agrees | features agree |", "|---|---|---|---|---|"]
    for j, d in r["across"].items():
        for lab, v in d.items():
            L.append(f"| {j} | {lab} | {v['n']} | {v['student_agrees']:.3f} | {v['features_agrees']:.3f} |")
    if r.get("you_elite_beats_first"):
        L += ["", f"Your ratings: elite beats first occupant {r['you_elite_beats_first'][0]} ({r['you_elite_beats_first'][1]} cells)."]
    L += ["", "Pictures: map.png, elites.png, progress.png."]
    (out / "check.md").write_text("\n".join(L) + "\n")
