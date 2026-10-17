"""Early VLM test: do VLM judges prefer real creatures, and can they be fooled?

build   pick creatures in five groups, re-simulate each, render a 2x2 grid of four snapshots
        (3 time units apart) and choose random pairs:
          catalogue  Chan's species from their own patterns (the human-curated reference)
          coherent   our multikernel search with the resolution check, shown at 2x resolution
          classic    our classic search (no resolution check)
          speckle    the most novel finds of the multikernel search without the check (pixel-scale noise)
          failure    random rules that froze, died, spread or exploded
        Every creature is simulated where it was accepted and drawn with smooth interpolation and one
        colour map. One fixed camera per creature across its four snapshots (2 time units apart),
        cropped tightly around its whole path, with a faint grid every kernel radius: shape change
        and motion both show (whole world, no grid, for spreading or exploding failures).
run     ask each judge about every pair in both orders, per prompt (cached; re-running is free)
rate    a local page to rate pairs yourself (same images the judges saw); writes human.csv
report  a prompt comparison table, then per judge and prompt: order consistency, group-vs-group win
        rates, Bradley-Terry scores, what the scores correlate with, agreement with the other judge,
        the other prompts and your own ratings; contact sheets of each ranking's top and bottom 12
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import render, seeds as seedbank
from .alive import ALIVE, centroid, recentre, torus_delta
from .explore import load_archive, rescaled, upsample
from .sim import Rule

GROUPS = ["catalogue", "coherent", "classic", "speckle", "failure"]
FRAME_PX, GAP = 320, 8
SNAP_DT = 2.0          # time units between snapshots
SWEEP = ["appeal", "appeal_bare", "beautiful", "curate", "interesting", "lifelike"]
HUMAN_QUESTION = "Which creature do you find more visually appealing?"


def roughness(F) -> float:
    """Edge length per unit mass (catalogue species: median 0.58, p99 1.07)."""
    F = np.asarray(F, np.float32)
    F = F.sum(0) if F.ndim == 3 else F
    return float((np.abs(np.diff(F, axis=0)).sum() + np.abs(np.diff(F, axis=1)).sum()) / max(F.sum(), 1e-9))


def fixed_window(frames: list, R: float, occupied: float = 0.05) -> list:
    """Crop all frames with ONE camera: centred on the creature's mean position over the frames and
    big enough to hold its whole path, so translation stays visible (a glider shifts across the
    panels; a frozen blob doesn't). Square window, Chebyshev extent plus a margin of 0.2R per side."""
    F = [np.asarray(f, np.float32) for f in frames]
    F = [f if f.ndim == 3 else f[None] for f in F]
    H, W = F[0].shape[-2:]
    c0 = centroid(F[0].sum(0))
    cs = [centroid(f.sum(0)) for f in F]
    dy = np.mean([torus_delta(c0[0], c[0], H) for c in cs])
    dx = np.mean([torus_delta(c0[1], c[1], W) for c in cs])
    m = ((c0[0] + dy) % H, (c0[1] + dx) % W)
    rc = [recentre(f, m) for f in F]
    ext = 0.0
    for f in rc:
        occ = np.argwhere(f.sum(0) > occupied)
        if len(occ):
            ext = max(ext, float(np.abs(occ - np.array([H / 2, W / 2])).max()))
    n = int(min(H, W, max(8, round(2 * ext + 0.4 * R))))
    y0, x0 = (H - n) // 2, (W - n) // 2
    return [f[:, y0:y0 + n, x0:x0 + n] for f in rc]


def _grid_lines(tile: np.ndarray, n_cells: int, step_cells: float) -> np.ndarray:
    """Faint grid fixed to the window, painted only on empty background (never over the creature)."""
    t = tile.copy()
    px = t.shape[0]
    bg = t.min(-1) > 245
    pos = [int(round(k * step_cells * px / n_cells)) for k in range(1, int(n_cells / step_cells) + 1)]
    for p in pos:
        if 0 < p < px:
            for sl in (np.s_[p, :], np.s_[:, p]):
                m = bg[sl]
                t[sl][m] = 232
    return t


def grid_image(frames: list, R: float, whole_world: bool, cmap: str = "chan") -> np.ndarray:
    """Four (C,H,W) frames -> one RGB image: a 2x2 grid of snapshots seen by one fixed camera (so motion
    shows), with a faint grid every kernel radius; bilinear upscale. whole_world: no crop, no grid."""
    if whole_world:
        crops = [np.asarray(f, np.float32) for f in frames]
    else:
        crops = fixed_window(frames, R)
    n = crops[0].shape[-1]
    tiles = []
    for a in crops:
        t = np.asarray(Image.fromarray(render.colorize(a, cmap)).resize((FRAME_PX, FRAME_PX), Image.BILINEAR))
        tiles.append(t if whole_world else _grid_lines(t, n, R))
    side = 2 * FRAME_PX + GAP
    out = Image.new("RGB", (side, side), (205, 205, 205))
    for k, t in enumerate(tiles):
        out.paste(Image.fromarray(t), ((k % 2) * (FRAME_PX + GAP), (k // 2) * (FRAME_PX + GAP)))
    return np.asarray(out)


# ---------- build ----------

def _pick_catalogue(catalogue: list, n: int, rng) -> list:
    idx = rng.choice(len(catalogue), min(n, len(catalogue)), replace=False)
    return [("catalogue", f"chan:{catalogue[i]['code']}", catalogue[i]["rule"], ("cells", catalogue[i]["cells"]), 1)
            for i in sorted(idx)]


def _pick_archive(run: Path, group: str, n_top: int, n_rand: int, rng, scale: int, rough_min: float | None = None) -> list:
    kids = [a for a in load_archive(run) if a["parent"] >= 0]
    if rough_min is not None:
        kids = [a for a in kids if roughness(a["final"]) > rough_min]
    kids.sort(key=lambda a: -a["dist_cat"])
    chosen = kids[:n_top]
    rest = kids[n_top:]
    if n_rand and rest:
        chosen += [rest[i] for i in sorted(rng.choice(len(rest), min(n_rand, len(rest)), replace=False))]
    out = []
    for a in chosen:
        f = a["final"] if a["final"].ndim == 3 else a["final"][None]
        rule, f = (rescaled(a["rule"], scale), upsample(f, scale)) if scale > 1 else (a["rule"], f)
        out.append((group, f"{run.name}#{a['id']}", rule, ("cells", np.asarray(f, np.float32)), scale))
    return out


def _pick_failures(surveys: list, per_class: int, rng) -> list:
    want = {"frozen": [], "dying": [], "spread": [], "exploded": []}
    for run in surveys:
        rules = {int(r["rule"]): r for r in csv.DictReader(open(Path(run) / "rules.csv"))}
        for r in csv.DictReader(open(Path(run) / "rollouts.csv")):
            if r["class"] in want and rules[int(r["rule"])]["alive_seeds"] == "0":
                want[r["class"]].append((Path(run).name, int(r["rule"]), int(r["seed"]), rules[int(r["rule"])]["json"]))
    out = []
    for cls, items in want.items():
        for k in sorted(rng.choice(len(items), min(per_class, len(items)), replace=False)) if items else []:
            name, i, j, js = items[k]
            out.append(("failure", f"{name}#{i}/seed{j}:{cls}", Rule.from_dict(json.loads(js)), ("bank", j), 1))
    return out


def _simulate(items: list, shape, time: float, log) -> list:
    from .show import simulate
    every = int(round(SNAP_DT * items[0][2].T)) if items else 30
    sims = simulate([(src, rule, spec) for _, src, rule, spec, _ in items], shape, time=time, every=every)
    out = []
    for (group, src, rule, spec, scale), (_, _, frames, st) in zip(items, sims):
        fr = [np.asarray(f, np.float32) for _, f in frames[-4:]]
        out.append({"group": group, "source": src, "rule": rule, "scale": scale, "frames": fr, "stats": st})
    log(f"  simulated {len(items)} ({items[0][0] if items else '-'})")
    return out


def build(out: Path, runs: Path, catalogue: list, n: dict | None = None, seed: int = 0, pairs_per: int = 3, log=print) -> dict:
    """Pick, simulate and render the creatures; write images/, creatures.csv, pairs.json, sheet_<group>.png."""
    n = {"catalogue": 40, "coherent_top": 15, "coherent_rand": 15, "classic": 20, "speckle": 20, "failure_per_class": 5,
         **(n or {})}
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sims = []
    sims += _simulate(_pick_catalogue(catalogue, n["catalogue"], rng), (256, 256), 50.0, log)
    sims += _simulate(_pick_archive(runs / "explore-multikernel", "coherent", n["coherent_top"], n["coherent_rand"], rng, 2),
                      (256, 256), 30.0, log)
    sims += _simulate(_pick_archive(runs / "explore-classic", "classic", 0, n["classic"], rng, 1), (128, 128), 30.0, log)
    sims += _simulate(_pick_archive(runs / "explore-multikernel-norescale", "speckle", n["speckle"], 0, rng, 1, rough_min=1.07),
                      (128, 128), 30.0, log)
    sims += _simulate(_pick_failures([runs / "survey-multikernel", runs / "survey-classic"], n["failure_per_class"], rng),
                      (128, 128), 100.0, log)

    rows, dropped = [], []
    for s in sims:
        cls = s["stats"]["class"]
        alive = cls in ALIVE
        if (s["group"] == "failure") == alive:      # a 'failure' that lived, or a creature that didn't
            dropped.append((s["group"], s["source"], cls))
            continue
        cid = f"c{len(rows):03d}"
        whole = s["group"] == "failure" and cls in ("spread", "exploded")
        img = grid_image(s["frames"], s["rule"].R, whole)
        Image.fromarray(img).save(out / "images" / f"{cid}.png")
        st = s["stats"]
        rows.append({"id": cid, "group": s["group"], "source": s["source"], "class": cls, "R": s["rule"].R,
                     "scale": s["scale"], "mass": round(float(st.get("mass_end", np.nan)), 4),
                     "size_rg": round(float(st.get("rg_max_w", np.nan)), 4), "speed": round(float(st.get("speed_w", np.nan)), 4),
                     "change": round(float(st.get("change_w", np.nan)), 4), "roughness": round(roughness(s["frames"][-1]) * s["scale"], 3),
                     "rule": s["rule"].to_json()})
    with open(out / "creatures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # random pairs: every creature meets pairs_per random others (unordered, no repeats)
    ids = [r["id"] for r in rows]
    pairs = set()
    for a in ids:
        for b in rng.choice([x for x in ids if x != a], pairs_per, replace=False):
            pairs.add(tuple(sorted((a, str(b)))))
    pairs = sorted(pairs)
    (out / "pairs.json").write_text(json.dumps(pairs))

    for g in GROUPS:
        items = [(r["id"] + " " + r["source"].split(":")[-1][:18], out / "images" / f"{r['id']}.png") for r in rows if r["group"] == g]
        if items:
            _contact(items, out / f"sheet_{g}.png", f"{g} ({len(items)})")
    summary = {"creatures": len(rows), "pairs": len(pairs), "per_group": {g: sum(r["group"] == g for r in rows) for g in GROUPS},
               "dropped": dropped, "seed": seed}
    (out / "build.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: summary[k] for k in ("creatures", "pairs", "per_group")}))
    return summary


def _contact(items: list, path: Path, title: str, cols: int = 6, px: int = 160) -> None:
    rows_n = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (px + 6) + 6, rows_n * (px + 20) + 26), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill=(20, 20, 20))
    for k, (lab, p) in enumerate(items):
        r, c = divmod(k, cols)
        x, y = 6 + c * (px + 6), 26 + r * (px + 20)
        sheet.paste(Image.open(p).resize((px, px), Image.BILINEAR), (x, y + 14))
        d.text((x, y), str(lab)[:26], fill=(40, 40, 40))
    sheet.save(path)


# ---------- run ----------

def run(out: Path, judges: dict, prompts: list, limit: int | None = None, log=print) -> Path:
    """judges: {name: Judge}. Writes verdicts.csv (one row per pair x judge x prompt, both orders)."""
    from .judge import Query
    out = Path(out)
    pairs = json.loads((out / "pairs.json").read_text())[:limit]
    imgs = {}
    load = lambda i: imgs.setdefault(i, np.asarray(Image.open(out / "images" / f"{i}.png").convert("RGB")))
    path = out / "verdicts.csv"
    rows = []
    for jname, judge in judges.items():
        for pid in prompts:
            qs = [Query(pid, load(a), load(b)) for a, b in pairs] + [Query(pid, load(b), load(a)) for a, b in pairs]
            vs = judge.judge_many(qs, desc=f"{jname}/{pid}")
            m = len(pairs)
            for k, (a, b) in enumerate(pairs):
                v1, v2 = vs[k], vs[m + k]
                rows.append({"judge": jname, "prompt": pid, "a": a, "b": b,
                             "p_a_when_first": v1.p_first, "p_a_when_second": None if v2.p_first is None else 1 - v2.p_first})
            log(f"{jname}/{pid}: spent ${judge.spent:.4f} so far")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


# ---------- report ----------

def bradley_terry(ids: list, games: list, iters: int = 500, prior: float = 0.5) -> dict:
    """Soft Bradley-Terry by MM. games: (a, b, s) with s = P(a beats b). A prior of `prior` virtual draws
    against an average opponent keeps scores finite. Returns log-strengths (mean 0)."""
    idx = {i: k for k, i in enumerate(ids)}
    w = np.ones(len(ids))
    wins = np.full(len(ids), prior)
    for a, b, s in games:
        wins[idx[a]] += s
        wins[idx[b]] += 1 - s
    for _ in range(iters):
        den = 2 * prior / (w + 1.0)
        for a, b, _ in games:
            ia, ib = idx[a], idx[b]
            t = 1.0 / (w[ia] + w[ib])
            den[ia] += t
            den[ib] += t
        w = wins / den
        w /= np.exp(np.mean(np.log(w)))
    s = np.log(w)
    return {i: float(s[idx[i]]) for i in ids}


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def report(out: Path, log=print) -> dict:
    out = Path(out)
    cre = {r["id"]: r for r in csv.DictReader(open(out / "creatures.csv"))}
    ids = sorted(cre)
    rows = list(csv.DictReader(open(out / "verdicts.csv")))
    fl = lambda v: float(v) if v not in (None, "", "None") else np.nan
    combos = sorted({(r["judge"], r["prompt"]) for r in rows})
    res, scores = {}, {}
    for jp in combos:
        rr = [r for r in rows if (r["judge"], r["prompt"]) == jp]
        p1 = np.array([fl(r["p_a_when_first"]) for r in rr])
        p2 = np.array([fl(r["p_a_when_second"]) for r in rr])
        ok = np.isfinite(p1) & np.isfinite(p2)
        pav = (p1 + p2) / 2
        first_rate = float(np.mean(np.concatenate([p1[np.isfinite(p1)] > 0.5, (1 - p2[np.isfinite(p2)]) > 0.5])))
        games = [(r["a"], r["b"], float(p)) for r, p, o in zip(rr, pav, ok) if o]
        bt = bradley_terry(ids, games)
        scores[jp] = bt
        # group vs group: mean P(row group beats column group)
        mat = {g: {h: [] for h in GROUPS} for g in GROUPS}
        for (a, b, s) in games:
            ga, gb = cre[a]["group"], cre[b]["group"]
            mat[ga][gb].append(s)
            mat[gb][ga].append(1 - s)
        res[jp] = {"pairs": int(ok.sum()), "failed": int((~ok).sum()),
                   "order_agreement": round(float(np.mean((p1[ok] > 0.5) == (p2[ok] > 0.5))), 3),
                   "first_slot_rate": round(first_rate, 3),
                   "mean_confidence": round(float(np.mean(np.abs(pav[ok] - 0.5) * 2)), 3),
                   "group_mean_score": {g: round(float(np.mean([bt[i] for i in ids if cre[i]["group"] == g])), 2)
                                        for g in GROUPS if any(cre[i]["group"] == g for i in ids)},
                   "group_vs_group": {g: {h: (round(float(np.mean(v)), 2), len(v)) for h, v in mat[g].items() if v}
                                      for g in GROUPS},
                   "correlates_alive_only": {k: round(_spearman([bt[i] for i in ids if cre[i]["class"] in ALIVE],
                                                               [fl(cre[i][k]) for i in ids if cre[i]["class"] in ALIVE]), 2)
                                             for k in ("mass", "size_rg", "speed", "change", "roughness")}}
    # agreement between judges / prompts: same pair, order-averaged winner; and score rank correlation
    agree = {}
    pav_of = {jp: {(r["a"], r["b"]): (fl(r["p_a_when_first"]) + fl(r["p_a_when_second"])) / 2
                   for r in rows if (r["judge"], r["prompt"]) == jp} for jp in combos}
    for x, y in combinations(combos, 2):
        common = [k for k in pav_of[x] if np.isfinite(pav_of[x][k]) and np.isfinite(pav_of[y].get(k, np.nan))]
        if common:
            agree[f"{x[0]}/{x[1]} vs {y[0]}/{y[1]}"] = {
                "pair_agreement": round(float(np.mean([(pav_of[x][k] > 0.5) == (pav_of[y][k] > 0.5) for k in common])), 3),
                "score_spearman": round(_spearman([scores[x][i] for i in ids], [scores[y][i] for i in ids]), 3)}
    human = _human_summary(out, cre, pav_of)
    for jp in combos:
        h = human.get("agreement", {}).get(f"{jp[0]}/{jp[1]}")
        res[jp]["human_agreement"] = h
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "source", "class"] + [f"{j}/{p}" for j, p in combos])
        for i in ids:
            w.writerow([i, cre[i]["group"], cre[i]["source"], cre[i]["class"]] + [round(scores[jp][i], 3) for jp in combos])
    for jp in combos:
        ranked = sorted(ids, key=lambda i: -scores[jp][i])
        for tag, sel in (("top", ranked[:12]), ("bottom", ranked[-12:])):
            _contact([(f"{i} {cre[i]['group']} {scores[jp][i]:+.2f}", out / "images" / f"{i}.png") for i in sel],
                     out / f"{tag}_{jp[0]}_{jp[1]}.png", f"{jp[0]} / {jp[1]}: {tag} 12")
    summary = {"per_judge_prompt": {f"{j}/{p}": v for (j, p), v in res.items()}, "agreement": agree,
               "human": {k: v for k, v in human.items() if k != "agreement"}}
    (out / "report.json").write_text(json.dumps(summary, indent=2))
    _markdown(out, summary)
    log((out / "report.md").read_text())
    return summary


def _markdown(out: Path, s: dict) -> None:
    L = ["# VLM test: report", "",
         "Scores are Bradley-Terry log-strengths (0 = average); win rates are P(row group beats column group), order-averaged.", ""]
    L += _comparison(s)
    for name, r in s["per_judge_prompt"].items():
        L += [f"## {name}", "",
              f"- pairs {r['pairs']} (failed {r['failed']}); order agreement {r['order_agreement']}; "
              f"first-slot rate {r['first_slot_rate']}; mean confidence {r['mean_confidence']}",
              "- mean score by group: " + ", ".join(f"{g} {v:+.2f}" for g, v in r["group_mean_score"].items()),
              "- score vs creature statistics (Spearman, living creatures only): "
              + ", ".join(f"{k} {v:+.2f}" for k, v in r["correlates_alive_only"].items()), "",
              "| row beats column | " + " | ".join(GROUPS) + " |", "|---" * (len(GROUPS) + 1) + "|"]
        for g in GROUPS:
            if g in r["group_vs_group"]:
                L.append(f"| {g} | " + " | ".join(f"{r['group_vs_group'][g][h][0]:.2f} ({r['group_vs_group'][g][h][1]})"
                                                   if h in r["group_vs_group"][g] else "-" for h in GROUPS) + " |")
        L.append("")
    if s["agreement"]:
        L += ["## Agreement", "", "| | same winner | score rank corr. |", "|---|---|---|"]
        L += [f"| {k} | {v['pair_agreement']} | {v['score_spearman']} |" for k, v in s["agreement"].items()]
    (out / "report.md").write_text("\n".join(L) + "\n")


def _wr(r: dict, g: str, h: str):
    v = r["group_vs_group"].get(g, {}).get(h)
    return f"{v[0]:.2f}" if v else "-"


def _comparison(s: dict) -> list:
    """One row per judge x prompt: the numbers used to pick a prompt."""
    per = s["per_judge_prompt"]
    L = ["## Prompt comparison", "",
         "Order agr. = same winner in both orders; 1st slot = P(first image wins); win rates are order-averaged; "
         "r(size), r(rough) = Spearman of score with size relative to the kernel and with roughness (living creatures); "
         "Luna~Jev = rank correlation of the two judges' scores under that prompt; human = agreement with your picks.", "",
         "| judge / prompt | order agr. | 1st slot | catalogue > ours | ours > speckle | ours > failure | r(size) | r(rough) | Luna~Jev | human |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    cross = {}
    for k, v in s["agreement"].items():
        x, y = k.split(" vs ")
        jx, px = x.split("/")
        jy, py = y.split("/")
        if px == py and jx != jy:
            cross[px] = v["score_spearman"]
    for name, r in sorted(per.items(), key=lambda kv: (kv[0].split("/")[1], kv[0])):
        prompt = name.split("/")[1]
        h = r.get("human_agreement")
        L.append(f"| {name} | {r['order_agreement']:.2f} | {r['first_slot_rate']:.2f} | {_wr(r, 'catalogue', 'coherent')} | "
                 f"{_wr(r, 'coherent', 'speckle')} | {_wr(r, 'coherent', 'failure')} | {r['correlates_alive_only']['size_rg']:+.2f} | "
                 f"{r['correlates_alive_only']['roughness']:+.2f} | {cross.get(prompt, float('nan')):.2f} | "
                 + (f"{h[0]:.2f} ({h[1]})" if h else "-") + " |")
    hs = s.get("human") or {}
    if hs.get("rated"):
        L += ["", f"Your ratings: {hs['rated']} pairs ({hs['skipped']} skipped). Your group win rates: "
              + ", ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in hs["group_wins"].items()) + "."]
    return L + [""]


# ---------- your own ratings ----------

def _human_rows(out: Path) -> list:
    p = Path(out) / "human.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def _human_summary(out: Path, cre: dict, pav_of: dict) -> dict:
    rows = [r for r in _human_rows(out) if r["choice"] in ("left", "right")]
    skipped = sum(r["choice"] == "skip" for r in _human_rows(out))
    if not rows:
        return {}
    agreement = {}
    for (j, p), pav in pav_of.items():
        hits = [(pav.get((r["a"], r["b"]), np.nan) > 0.5) == (r["winner"] == r["a"]) for r in rows
                if np.isfinite(pav.get((r["a"], r["b"]), np.nan))]
        if hits:
            agreement[f"{j}/{p}"] = (round(float(np.mean(hits)), 3), len(hits))
    wins = {}
    for g, h in (("catalogue", "coherent"), ("coherent", "speckle"), ("coherent", "failure"), ("catalogue", "failure")):
        ws = [r["winner"] == (r["a"] if cre[r["a"]]["group"] == g else r["b"]) for r in rows
              if {cre[r["a"]]["group"], cre[r["b"]]["group"]} == {g, h}]
        if ws:
            wins[f"{g} > {h}"] = (round(float(np.mean(ws)), 2), len(ws))
    return {"rated": len(rows), "skipped": skipped, "agreement": agreement, "group_wins": wins}


RATE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rate creatures</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f2;color:#222}
 header{padding:14px 20px;display:flex;justify-content:space-between;align-items:baseline}
 h1{font-size:18px;margin:0} #prog{color:#666}
 .q{padding:0 20px 8px;color:#444}
 .pair{display:flex;gap:16px;justify-content:center;padding:8px 20px}
 .pair figure{margin:0;background:#fff;border:3px solid transparent;border-radius:8px;padding:6px;cursor:pointer;flex:1;max-width:46vw}
 .pair figure:hover{border-color:#999}
 .pair img{display:block;margin:auto;max-width:100%;max-height:calc(100vh - 210px);width:auto;height:auto}
 .pair figcaption{text-align:center;color:#666;font-size:13px;padding-top:4px}
 .bar{display:flex;gap:10px;justify-content:center;padding:12px}
 button{font-size:15px;padding:8px 16px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
 #done{display:none;padding:40px;text-align:center;font-size:18px}
</style></head><body>
<header><h1>QUESTION</h1><span id="prog"></span></header>
<div class="q">Each image shows one creature at four moments a few time units apart, in reading order. Go with your first impression.</div>
<div id="ui"><div class="pair">
 <figure id="L"><img id="li"><figcaption>left &larr;</figcaption></figure>
 <figure id="R"><img id="ri"><figcaption>right &rarr;</figcaption></figure></div>
 <div class="bar"><button id="bl">&larr; Left</button><button id="bs">Can't decide (space)</button>
 <button id="br">Right &rarr;</button><button id="bu">Undo (backspace)</button></div></div>
<div id="done">All done. Your ratings are in runs/vlmtest/human.csv. You can close this tab and stop the server (Ctrl-C).</div>
<script>
let todo=[], total=0, busy=false;
async function load(){const r=await (await fetch('/pairs')).json(); todo=r.todo; total=r.total; show();}
function show(){
 document.getElementById('prog').textContent=(total-todo.length)+' / '+total+' rated';
 if(!todo.length){document.getElementById('ui').style.display='none';document.getElementById('done').style.display='block';return;}
 document.getElementById('ui').style.display='block';document.getElementById('done').style.display='none';
 const p=todo[0]; document.getElementById('li').src='/img/'+p.left+'.png'; document.getElementById('ri').src='/img/'+p.right+'.png';}
async function vote(choice){ if(busy||!todo.length) return; busy=true; const p=todo[0];
 await fetch('/vote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...p,choice})});
 todo.shift(); busy=false; show();}
async function undo(){ if(busy) return; busy=true; await fetch('/undo',{method:'POST'}); busy=false; load();}
document.getElementById('L').onclick=()=>vote('left'); document.getElementById('R').onclick=()=>vote('right');
document.getElementById('bl').onclick=()=>vote('left'); document.getElementById('br').onclick=()=>vote('right');
document.getElementById('bs').onclick=()=>vote('skip'); document.getElementById('bu').onclick=undo;
document.addEventListener('keydown',e=>{ if(e.key==='ArrowLeft')vote('left'); else if(e.key==='ArrowRight')vote('right');
 else if(e.key===' '){e.preventDefault();vote('skip');} else if(e.key==='Backspace'){e.preventDefault();undo();}});
load();
</script></body></html>"""


def human_pairs(out: Path, n: int = 60, seed: int = 1) -> list:
    """The pairs you rate: a fixed random subset of pairs.json with random left/right sides (human_pairs.json)."""
    out = Path(out)
    p = out / "human_pairs.json"
    if p.exists():
        return json.loads(p.read_text())
    pairs = json.loads((out / "pairs.json").read_text())
    rng = np.random.default_rng(seed)
    pick = [pairs[i] for i in sorted(rng.choice(len(pairs), min(n, len(pairs)), replace=False))]
    order = rng.permutation(len(pick))
    hp = []
    for k in order:
        a, b = pick[k]
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        hp.append({"a": a, "b": b, "left": left, "right": right})
    p.write_text(json.dumps(hp, indent=1))
    return hp


def make_server(out: Path, n: int = 60, port: int = 8765):
    """The rating server (localhost only); every click is appended to human.csv straight away."""
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    out = Path(out)
    hp = human_pairs(out, n)
    path = out / "human.csv"
    lock = threading.Lock()
    fields = ["time", "a", "b", "left", "right", "choice", "winner"]

    def rated() -> set:
        return {(r["a"], r["b"]) for r in _human_rows(out)}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, RATE_PAGE.replace("QUESTION", HUMAN_QUESTION).encode(), "text/html; charset=utf-8")
            if self.path == "/pairs":
                done = rated()
                return self._send(200, json.dumps({"todo": [p for p in hp if (p["a"], p["b"]) not in done], "total": len(hp)}).encode(),
                                  "application/json")
            if self.path.startswith("/img/") and self.path.endswith(".png"):
                name = Path(self.path).name
                f = out / "images" / name
                if f.parent == out / "images" and f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/vote":
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if d.get("choice") not in ("left", "right", "skip"):
                    return self._send(400, b"bad choice", "text/plain")
                winner = d["left"] if d["choice"] == "left" else d["right"] if d["choice"] == "right" else ""
                with lock:
                    new = not path.exists()
                    with open(path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow({"time": round(_time.time(), 1), "a": d["a"], "b": d["b"], "left": d["left"],
                                    "right": d["right"], "choice": d["choice"], "winner": winner})
                return self._send(200, b"ok", "text/plain")
            if self.path == "/undo":
                with lock:
                    rows = _human_rows(out)
                    if rows:
                        with open(path, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fields)
                            w.writeheader()
                            w.writerows(rows[:-1])
                return self._send(200, b"ok", "text/plain")
            self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.pairs_left = len(hp) - len(rated() & {(p["a"], p["b"]) for p in hp})
    srv.pairs_total = len(hp)
    return srv


def rate(out: Path, n: int = 60, port: int = 8765, open_browser: bool = True, log=print) -> None:
    import webbrowser
    srv = make_server(out, n, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    log(f"rating page: {url}  ({srv.pairs_left} of {srv.pairs_total} pairs left; answers go to "
        f"{Path(out) / 'human.csv'}; Ctrl-C to stop, progress is kept)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
