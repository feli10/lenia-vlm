"""Phase 2: the labelling pool. Living creatures under the frozen alive check (alive v2), rendered the way
the judges see them, paired, labelled by a VLM, and checked before anything is distilled.

build   (Colab, GPU) pick creatures:
          chan     Chan's living found/ creatures from a calibrated survey that also pass its 2x and
                   texture-loss checks (the same checks the search applies)
          search   descendants from a novelty-search archive, at most `per_lineage` per lineage, so no
                   single family dominates
        re-simulate each from its stored pattern, render the fixed-camera 2x2 image (vlmtest.grid_image),
        hold out a fraction of the creatures (stratified by group), and make random pairs within each
        split. Held-out pairs are only between held-out creatures, so the student (Phase 3) is tested on
        creatures it never saw.
label   (your Mac, API keys) ask a judge about pairs in both orders: Luna on every pair (training labels
        and held-out labels), Jev on held-out pairs (a second opinion). One file per judge and prompt
        (labels_<judge>_<prompt>.csv); the judge cache makes re-runs free.
rate    you rate held-out pairs (same images, same question as the judge).
report  the judge check on this pool, before distilling: order agreement, position bias, confidence,
        agreement with the other judge and with you, chan vs search win rates, what the scores follow
        (size, mass, speed, change, texture), top and bottom creatures. Writes report.md / report.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .alive import ALIVE
from .sim import Rule

SNAP_EVERY = 2.0          # time units between the four snapshots (vlmtest.SNAP_DT)
QUESTION = ("You are curating a gallery of the most remarkable creatures discovered in Lenia. "
            "Which creature would you rather include?")
DESCRIPTION_DUAL = ("Each image shows one creature in two views, both at the same scale for every creature. Left: its "
                    "movement over the last 16 time units as multiple exposures (fainter = earlier). Right: four "
                    "close-ups 2 time units apart, in reading order.")
DESCRIPTION_DUAL_BODY = ("Each image shows one creature in two views, each drawn relative to the creature's own size (so "
                         "the images do not show which creature is bigger). Left: its movement over the last 16 time units "
                         "as multiple exposures, in a window four body lengths wide (fainter = earlier). Right: four "
                         "close-ups 2 time units apart, in reading order.")


# ---------- build ----------

def _chan_items(survey: Path, n: int, rng, max_share: float = 0.4) -> list:
    """Living, check-passing creatures of a calibrated survey; at most max_share of n from one file."""
    survey = Path(survey)
    cal = json.loads((survey / "calibration.json").read_text())
    z = np.load(survey / "alive.npz")
    pat = {str(c): z[f"p{k}"] for k, c in enumerate(z["codes"])}
    rows = [r for r in csv.DictReader(open(survey / "creatures.csv")) if r["code"] in pat]

    def ok(r):
        try:
            return (float(r["rescale_dist"]) <= cal["rescale_max_dist"]
                    and float(r["texture_loss"]) <= (cal.get("texture_max_loss") or np.inf))
        except (KeyError, ValueError):
            return False
    rows = [r for r in rows if ok(r)]
    order = rng.permutation(len(rows))
    cap = max(1, int(max_share * n))
    per, out = {}, []
    for k in order:
        r = rows[k]
        if per.get(r["file"], 0) >= cap:
            continue
        per[r["file"]] = per.get(r["file"], 0) + 1
        out.append(("chan", f"chan:{r['code']}", Rule.from_dict(json.loads(r["rule"])), np.asarray(pat[r["code"]], np.float32)))
        if len(out) >= n:
            break
    return out


def _search_items(run: Path, per_lineage: int, rng) -> list:
    from .explore import load_archive
    run = Path(run)
    kids = [a for a in load_archive(run) if a["parent"] >= 0]
    by = {}
    for a in kids:
        by.setdefault(a["root"], []).append(a)
    out = []
    for root, xs in sorted(by.items()):
        for k in sorted(rng.permutation(len(xs))[:per_lineage]):
            a = xs[k]
            f = np.asarray(a["final"], np.float32)
            out.append(("search", f"{run.name}#{a['id']}", a["rule"], f if f.ndim == 3 else f[None]))
    return out


def _simulate(items: list, shape, time: float, chunk: int, log, snap: float = SNAP_EVERY, keep: int = 4) -> list:
    """Run each item; keep its last four frames, `snap` time units apart. Rules differ in T (steps
    per time unit), so items are run in groups of equal T, in chunks (GPU memory)."""
    from .show import simulate
    out = [None] * len(items)
    by_T = {}
    for k, it in enumerate(items):
        by_T.setdefault(it[2].T, []).append(k)
    done = 0
    for T, ks in sorted(by_T.items()):
        every = max(1, int(round(snap * T)))
        for c in range(0, len(ks), chunk):
            part = ks[c:c + chunk]
            sims = simulate([(items[k][1], items[k][2], ("cells", items[k][3])) for k in part], shape, time=time, every=every)
            for k, (_, _, frames, st) in zip(part, sims):
                group, src, rule, _ = items[k]
                out[k] = {"group": group, "source": src, "rule": rule, "frames": [np.asarray(f, np.float32) for _, f in frames[-keep:]],
                          "stats": st}
            done += len(part)
            log(f"  simulated {done}/{len(items)}")
    return out


def _pairs(ids: list, per: int, rng) -> list:
    s = set()
    for a in ids:
        others = [x for x in ids if x != a]
        for b in rng.choice(others, min(per, len(others)), replace=False):
            s.add(tuple(sorted((a, str(b)))))
    return sorted(s)


def build(out: Path, survey: Path, archive: Path, shape=(256, 256), n_chan: int = 300, per_lineage: int = 10,
          heldout: float = 0.2, pairs_train: int = 8, pairs_heldout: int = 8, time: float = 30.0, chunk: int = 64,
          seed: int = 0, snap: float = SNAP_EVERY, log=print) -> dict:
    """Write images/, creatures.csv, pairs_train.json, pairs_heldout.json, pairs.json (both), sheet_<group>.png."""
    from .explore import texture
    from .vlmtest import _contact, grid_image, roughness
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = _chan_items(survey, n_chan, rng) + _search_items(archive, per_lineage, rng)
    log(f"pool: {sum(i[0] == 'chan' for i in items)} chan + {sum(i[0] == 'search' for i in items)} search creatures; simulating")
    sims = _simulate(items, shape, max(time, 3 * snap), chunk, log, snap)
    rows, dropped = [], []
    for s in sims:
        cls = s["stats"]["class"]
        if cls not in ALIVE:                  # should be rare: these all passed the alive check before
            dropped.append((s["group"], s["source"], cls))
            continue
        cid = f"p{len(rows):04d}"
        Image.fromarray(grid_image(s["frames"], s["rule"].R, False)).save(out / "images" / f"{cid}.png")
        st = s["stats"]
        rows.append({"id": cid, "group": s["group"], "source": s["source"], "class": cls, "R": s["rule"].R,
                     "channels": s["rule"].channels, "mass": round(float(st.get("mass_end", np.nan)), 4),
                     "size_rg": round(float(st.get("rg_max_w", np.nan)), 4), "speed": round(float(st.get("speed_w", np.nan)), 4),
                     "change": round(float(st.get("change_w", np.nan)), 4), "roughness": round(roughness(s["frames"][-1]), 3),
                     "texture": round(texture(s["frames"][-1]), 4), "split": "", "rule": s["rule"].to_json()})
    # creature-level split, stratified by group
    for g in ("chan", "search"):
        idx = [k for k, r in enumerate(rows) if r["group"] == g]
        held = set(rng.choice(idx, int(round(heldout * len(idx))), replace=False).tolist()) if idx else set()
        for k in idx:
            rows[k]["split"] = "heldout" if k in held else "train"
    with open(out / "creatures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tr = _pairs([r["id"] for r in rows if r["split"] == "train"], pairs_train, rng)
    ho = _pairs([r["id"] for r in rows if r["split"] == "heldout"], pairs_heldout, rng)
    (out / "pairs_train.json").write_text(json.dumps(tr))
    (out / "pairs_heldout.json").write_text(json.dumps(ho))
    (out / "pairs.json").write_text(json.dumps(tr + ho))
    for g in ("chan", "search"):
        sel = [r for r in rows if r["group"] == g]
        sel = [sel[i] for i in sorted(rng.permutation(len(sel))[:48])]
        if sel:
            _contact([(f"{r['id']} {r['source'].split(':')[-1][:16]}", out / "images" / f"{r['id']}.png") for r in sel],
                     out / f"sheet_{g}.png", f"{g}: 48 of {sum(r['group'] == g for r in rows)}")
    summary = {"creatures": len(rows), "per_group": {g: sum(r["group"] == g for r in rows) for g in ("chan", "search")},
               "heldout_creatures": sum(r["split"] == "heldout" for r in rows), "pairs_train": len(tr), "pairs_heldout": len(ho),
               "dropped": dropped, "seed": seed, "snap": snap, "survey": str(survey), "archive": str(archive)}
    (out / "build.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: v for k, v in summary.items() if k not in ("dropped", "survey", "archive")}) + f"; dropped {len(dropped)}")
    return summary


def _item_of(row: dict, survey: Path, archive: Path, cache: dict):
    """Rebuild a pool creature's (group, source, rule, start pattern) from where build took it."""
    rule = Rule.from_dict(json.loads(row["rule"]))
    src = row["source"]
    if src.startswith("chan:"):
        if "chan" not in cache:
            z = np.load(Path(survey) / "alive.npz")
            cache["chan"] = {str(c): z[f"p{k}"] for k, c in enumerate(z["codes"])}
        cells = cache["chan"][src[5:]]
    else:
        if "search" not in cache:
            cache["search"] = np.load(Path(archive) / "patterns.npz")
        cells = cache["search"][f"final_{src.rsplit('#', 1)[1]}"]
    cells = np.asarray(cells, np.float32)
    return (row["group"], src, rule, cells if cells.ndim == 3 else cells[None])


def rerender(src: Path, out: Path, survey: Path, archive: Path, snap: float, split: str = "heldout",
             shape=(256, 256), time: float | None = None, chunk: int = 64, camera: str = "fixed", upscale: int = 1,
             log=print) -> dict:
    """The same creatures and pairs as pool `src`, re-rendered with `snap` time units between snapshots,
    into a new folder (same creature ids, same held-out pairs, same pairs for you to rate), so labels and
    ratings on the two image versions can be compared pair by pair. split: 'heldout' or 'all'.
    camera: 'fixed' (vlmtest.grid_image: one camera holding the whole path) or 'track' (vlmtest.tracking_image:
    centred on the creature, same scale for every creature, world-fixed grid sliding underneath), 'dual'
    (vlmtest.dual_image) or 'dual_body' (the same, each creature relative to its own size).
    upscale: simulate every creature at upscale x R (rule rescaled, pattern upsampled, world upscale x larger)
    for the pictures only, so small creatures show real detail instead of blurred pixels. Every pool creature
    passed the 2x coherence check, so at 2 it is the same creature."""
    import shutil
    from .vlmtest import dual_image, grid_image, tracking_image
    draw = {"track": lambda fr, R: tracking_image(fr, R), "dual": lambda fr, R: dual_image(fr, R),
            "dual_body": lambda fr, R: dual_image(fr, R, scale="body"),
            "fixed": lambda fr, R: grid_image(fr, R, False)}[camera]
    n_keep = 9 if camera.startswith("dual") else 4   # dual: 16 time units of motion at 2-unit steps
    if camera.startswith("dual"):
        snap = 2.0
    src, out = Path(src), Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src / "creatures.csv")))
    keep = [r for r in rows if split == "all" or r["split"] == split]
    cache = {}
    items = [_item_of(r, survey, archive, cache) for r in keep]
    if upscale > 1:
        from .explore import rescaled, upsample
        items = [(g, s_, rescaled(rule, upscale), upsample(c, upscale)) for g, s_, rule, c in items]
        shape = tuple(int(x * upscale) for x in shape)
    time = time or max(30.0, 3 * snap + 16.0)
    log(f"re-rendering {len(items)} creatures, {snap:g} time units between snapshots ({time:g} units)")
    sims = _simulate(items, shape, time, chunk, log, snap, n_keep)
    dropped = []
    for r, s_ in zip(keep, sims):
        if s_["stats"]["class"] not in ALIVE:
            dropped.append((r["id"], s_["stats"]["class"]))
        Image.fromarray(draw(s_["frames"], s_["rule"].R)).save(out / "images" / f"{r['id']}.png")
    with open(out / "creatures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0]))
        w.writeheader()
        w.writerows(keep)
    for name in ("pairs_heldout.json", "human_pairs.json") + (("pairs_train.json", "pairs.json") if split == "all" else ()):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    if split != "all":
        (out / "pairs_train.json").write_text("[]")
        shutil.copy(out / "pairs_heldout.json", out / "pairs.json")
    summary = {"from": str(src), "snap": snap, "camera": camera, "upscale": upscale, "time": time, "creatures": len(keep), "split": split,
               "not_alive_in_render": dropped}
    (out / "build.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: v for k, v in summary.items() if k != "not_alive_in_render"}) + f"; not alive in render: {len(dropped)}")
    return summary


FEATURES = ("speed", "change", "size_rg", "mass", "texture")


def _logit_weights(pairs: list, cre: dict, target) -> dict:
    """What drives a set of pairwise choices, each feature holding the others fixed: logistic regression of
    P(a chosen) on the differences of the creatures' standardised (log) features, a minus b. target(a, b)
    gives the observed P(a chosen) (soft labels allowed). Coefficients are per standard deviation of the
    pool; positive = prefers more of it. Fitted by Newton's method with a small ridge."""
    if len(pairs) < 10:
        return {}
    fl = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    ids = sorted(cre)
    raw = {f: np.array([fl(cre[i][f]) for i in ids]) for f in FEATURES}
    for f in ("speed", "change", "mass"):
        raw[f] = np.log(np.maximum(raw[f], 1e-4))
    z = {f: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9) for f, v in raw.items()}
    idx = {i: k for k, i in enumerate(ids)}
    X = np.array([[z[f][idx[a]] - z[f][idx[b]] for f in FEATURES] for a, b in pairs])
    y = np.array([target(a, b) for a, b in pairs], float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - 0.1 * w
        Hm = X.T @ (X * (p * (1 - p))[:, None]) + 0.1 * np.eye(len(w))
        w = w + np.linalg.solve(Hm, g)
    return {f: round(float(v), 2) for f, v in zip(FEATURES, w)}


def compare(a: Path, b: Path, judge: str = "luna", prompt: str = "curate", prompt_a: str | None = None, log=print) -> dict:
    """Two image versions of the same pairs (e.g. pool and pool-snap8): the judge's agreement with your
    ratings on each, its order agreement, how often it picks the faster / more-changing / larger creature,
    what drives its choices with the features held against each other (`weights`, logistic regression;
    the same for your ratings), and how often the two versions give the same winner. prompt is the prompt
    used on b; prompt_a the one used on a (default: the same)."""
    from .vlmtest import _human_rows
    out = {}
    fl = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    labs = {}
    for name, d, pr in (("a", Path(a), prompt_a or prompt), ("b", Path(b), prompt)):
        cre = {r["id"]: r for r in csv.DictReader(open(d / "creatures.csv"))}
        L = {(r["a"], r["b"]): (fl(r["p_a_when_first"]), fl(r["p_a_when_second"]))
             for r in csv.DictReader(open(d / f"labels_{judge}_{pr}.csv")) if r["split"] == "heldout"}
        L = {k: v for k, v in L.items() if np.isfinite(v[0]) and np.isfinite(v[1])}
        labs[name] = {k: (v[0] + v[1]) / 2 for k, v in L.items()}
        hum = [r for r in _human_rows(d) if r["choice"] in ("left", "right")]
        res = {"folder": str(d), "prompt": pr, "pairs": len(L),
               "order_agreement": round(float(np.mean([(v[0] > 0.5) == (v[1] > 0.5) for v in L.values()])), 3)}
        hits = [(labs[name][(r["a"], r["b"])] > 0.5) == (r["winner"] == r["a"]) for r in hum if (r["a"], r["b"]) in labs[name]]
        res["agrees_with_you"] = (round(float(np.mean(hits)), 3), len(hits)) if hits else None
        for stat in ("speed", "change", "size_rg", "mass", "texture"):
            res[f"picks_higher_{stat}"] = round(float(np.mean([(fl(cre[x][stat]) > fl(cre[y][stat])) == (p > 0.5)
                                                               for (x, y), p in labs[name].items()])), 3)
        lab_ = labs[name]
        res["weights"] = _logit_weights(list(lab_), cre, lambda x, y: lab_[(x, y)])
        if hum:
            res["you_pick_higher"] = {stat: round(float(np.mean([(fl(cre[r["a"]][stat]) > fl(cre[r["b"]][stat])) == (r["winner"] == r["a"])
                                                                for r in hum])), 3) for stat in ("speed", "change", "size_rg", "texture")}
            win = {(r["a"], r["b"]): float(r["winner"] == r["a"]) for r in hum}
            res["your_weights"] = _logit_weights(list(win), cre, lambda x, y: win[(x, y)])
        out[name] = res
    common = [k for k in labs["a"] if k in labs["b"]]
    out["same_winner_across_versions"] = round(float(np.mean([(labs["a"][k] > 0.5) == (labs["b"][k] > 0.5) for k in common])), 3)
    log(json.dumps(out, indent=1))
    return out


# ---------- label ----------

def label(out: Path, judge, prompt: str, split: str = "all", limit: int | None = None, log=print) -> Path:
    """Ask `judge` about every pair of the split in both orders; write labels_<judge>_<prompt>.csv with
    p_a_when_first, p_a_when_second and their mean p_a (order-averaged P(a preferred))."""
    from .judge import Query
    out = Path(out)
    name = {"all": "pairs.json", "train": "pairs_train.json", "heldout": "pairs_heldout.json"}[split]
    pairs = [tuple(p) for p in json.loads((out / name).read_text())][:limit]
    held = {tuple(p) for p in json.loads((out / "pairs_heldout.json").read_text())}
    imgs = {}
    load = lambda i: imgs.setdefault(i, np.asarray(Image.open(out / "images" / f"{i}.png").convert("RGB")))
    qs = [Query(prompt, load(a), load(b)) for a, b in pairs] + [Query(prompt, load(b), load(a)) for a, b in pairs]
    vs = judge.judge_many(qs, desc=f"{judge.name}/{prompt}/{split}")
    m = len(pairs)
    path = out / f"labels_{judge.name}_{prompt}.csv"
    old = {}
    if path.exists():                          # keep labels of other splits from earlier runs
        old = {(r["a"], r["b"]): r for r in csv.DictReader(open(path))}
    for k, (a, b) in enumerate(pairs):
        p1 = vs[k].p_first
        p2 = None if vs[m + k].p_first is None else 1 - vs[m + k].p_first
        pa = (p1 + p2) / 2 if p1 is not None and p2 is not None else None
        old[(a, b)] = {"a": a, "b": b, "split": "heldout" if (a, b) in held else "train",
                       "p_a_when_first": p1, "p_a_when_second": p2, "p_a": pa}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "split", "p_a_when_first", "p_a_when_second", "p_a"])
        w.writeheader()
        w.writerows(old.values())
    log(f"{judge.name}/{prompt}: {m} pairs labelled ({sum(v.ok for v in vs)}/{2 * m} calls ok); spent ${judge.spent:.4f}; -> {path.name}")
    return path


# ---------- report ----------

def _labels(out: Path) -> dict:
    fl = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    res = {}
    for p in sorted(Path(out).glob("labels_*_*.csv")):
        _, judge, prompt = p.stem.split("_", 2)
        res[(judge, prompt)] = {(r["a"], r["b"]): (fl(r["p_a_when_first"]), fl(r["p_a_when_second"]), r["split"])
                                for r in csv.DictReader(open(p))}
    return res


def report(out: Path, log=print) -> dict:
    from .vlmtest import _contact, _human_rows, _spearman, bradley_terry
    out = Path(out)
    cre = {r["id"]: r for r in csv.DictReader(open(out / "creatures.csv"))}
    ids = sorted(cre)
    labs = _labels(out)
    fl = lambda v: float(v) if v not in (None, "", "nan") else np.nan
    per, scores, pav = {}, {}, {}
    for jp, L in labs.items():
        keys = list(L)
        p1 = np.array([L[k][0] for k in keys])
        p2 = np.array([L[k][1] for k in keys])
        ok = np.isfinite(p1) & np.isfinite(p2)
        pa = (p1 + p2) / 2
        pav[jp] = {k: v for k, v, o in zip(keys, pa, ok) if o}
        games = [(a, b, float(p)) for (a, b), p, o in zip(keys, pa, ok) if o]
        bt = bradley_terry(ids, games)
        scores[jp] = bt
        cs = [s if cre[a]["group"] == "chan" else 1 - s for a, b, s in games if {cre[a]["group"], cre[b]["group"]} == {"chan", "search"}]
        alive_ids = [i for i in ids if any(i in k for k in pav[jp])]
        per[f"{jp[0]}/{jp[1]}"] = {
            "pairs": int(ok.sum()), "failed": int((~ok).sum()),
            "heldout_pairs": int(sum(L[k][2] == "heldout" for k, o in zip(keys, ok) if o)),
            "order_agreement": round(float(np.mean((p1[ok] > 0.5) == (p2[ok] > 0.5))), 3) if ok.any() else None,
            "first_slot_rate": round(float(np.mean(np.concatenate([p1[np.isfinite(p1)] > 0.5, p2[np.isfinite(p2)] < 0.5]))), 3),
            "mean_confidence": round(float(np.mean(np.abs(pa[ok] - 0.5) * 2)), 3) if ok.any() else None,
            "decisive_share": round(float(np.mean(np.abs(pa[ok] - 0.5) > 0.25)), 3) if ok.any() else None,
            "chan_beats_search": (round(float(np.mean(cs)), 3), len(cs)) if cs else None,
            "score_vs": {k: round(_spearman([bt[i] for i in alive_ids], [fl(cre[i][k]) for i in alive_ids]), 2)
                         for k in ("size_rg", "mass", "speed", "change", "roughness", "texture")}}
    agree = {}
    keys_ = list(labs)
    for i in range(len(keys_)):
        for j in range(i + 1, len(keys_)):
            x, y = keys_[i], keys_[j]
            common = [k for k in pav[x] if k in pav[y]]
            if common:
                agree[f"{x[0]}/{x[1]} vs {y[0]}/{y[1]}"] = {
                    "pairs": len(common),
                    "same_winner": round(float(np.mean([(pav[x][k] > 0.5) == (pav[y][k] > 0.5) for k in common])), 3),
                    "same_winner_decisive": round(float(np.mean([(pav[x][k] > 0.5) == (pav[y][k] > 0.5) for k in common
                                                                 if abs(pav[x][k] - 0.5) > 0.25 and abs(pav[y][k] - 0.5) > 0.25] or [np.nan])), 3)}
    hum = [r for r in _human_rows(out) if r["choice"] in ("left", "right")]
    human = {"rated": len(hum), "skipped": sum(r["choice"] == "skip" for r in _human_rows(out))}
    for jp in labs:
        hits = [(pav[jp][(r["a"], r["b"])] > 0.5) == (r["winner"] == r["a"]) for r in hum if (r["a"], r["b"]) in pav[jp]]
        if hits:
            per[f"{jp[0]}/{jp[1]}"]["human_agreement"] = (round(float(np.mean(hits)), 3), len(hits))
    if hum:
        cs = [r["winner"] == (r["a"] if cre[r["a"]]["group"] == "chan" else r["b"]) for r in hum
              if {cre[r["a"]]["group"], cre[r["b"]]["group"]} == {"chan", "search"}]
        human["chan_beats_search"] = (round(float(np.mean(cs)), 3), len(cs)) if cs else None
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "split", "source"] + [f"{j}/{p}" for j, p in labs])
        for i in ids:
            w.writerow([i, cre[i]["group"], cre[i]["split"], cre[i]["source"]] + [round(scores[jp][i], 3) for jp in labs])
    for jp in labs:
        rated = [i for i in ids if any(i in k for k in pav[jp])]
        ranked = sorted(rated, key=lambda i: -scores[jp][i])
        for tag, sel in (("top", ranked[:12]), ("bottom", ranked[-12:])):
            _contact([(f"{i} {cre[i]['group']} {scores[jp][i]:+.2f}", out / "images" / f"{i}.png") for i in sel],
                     out / f"{tag}_{jp[0]}_{jp[1]}.png", f"{jp[0]} / {jp[1]}: {tag} 12")
    summary = {"per_judge_prompt": per, "agreement": agree, "human": human}
    (out / "report.json").write_text(json.dumps(summary, indent=2))
    _markdown(out, summary)
    log((out / "report.md").read_text())
    return summary


def _markdown(out: Path, s: dict) -> None:
    L = ["# Labelling pool: judge check", "",
         "Order agr. = same winner in both orders; 1st slot = P(first image wins); confidence = mean |p - 0.5| x 2; "
         "decisive = share of pairs with p < 0.25 or > 0.75; chan > search = order-averaged win rate of Chan's creatures "
         "against search descendants; r(.) = Spearman of the Bradley-Terry score with that statistic.", "",
         "| judge / prompt | pairs | order agr. | 1st slot | confidence | decisive | chan > search | human | r(size) | r(mass) | r(speed) | r(change) | r(texture) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s["per_judge_prompt"].items():
        cs, h, v = r.get("chan_beats_search"), r.get("human_agreement"), r["score_vs"]
        L.append(f"| {k} | {r['pairs']} ({r['heldout_pairs']} held out) | {r['order_agreement']} | {r['first_slot_rate']} | "
                 f"{r['mean_confidence']} | {r['decisive_share']} | " + (f"{cs[0]} ({cs[1]})" if cs else "-") + " | "
                 + (f"{h[0]} ({h[1]})" if h else "-") + f" | {v['size_rg']:+.2f} | {v['mass']:+.2f} | {v['speed']:+.2f} | "
                 f"{v['change']:+.2f} | {v['texture']:+.2f} |")
    if s["agreement"]:
        L += ["", "| agreement | pairs | same winner | same winner (both decisive) |", "|---|---|---|---|"]
        L += [f"| {k} | {v['pairs']} | {v['same_winner']} | {v['same_winner_decisive']} |" for k, v in s["agreement"].items()]
    h = s["human"]
    if h.get("rated"):
        L += ["", f"Your ratings: {h['rated']} pairs ({h['skipped']} skipped)"
              + (f"; chan > search {h['chan_beats_search'][0]} ({h['chan_beats_search'][1]})" if h.get("chan_beats_search") else "") + "."]
    (Path(out) / "report.md").write_text("\n".join(L) + "\n")
