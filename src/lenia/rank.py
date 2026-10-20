"""Rank Chan's catalogues by the student and by the teacher: what does each consider most worth including?

Also a held-out test of the student with no search involved: most of these creatures were never in its
training pool, and the classic catalogue is a different kind of creature altogether.

build   (Colab, GPU) two sets, each rendered exactly as the pool (two-view image, R = 20, 256 x 256):
          found    every living creature of the calibrated found/ survey (alive at 100 units), flagged
                   with whether it is in the labelling pool (train or held-out split) and whether it
                   passes the calibrated 2x and texture checks
          classic  Chan's classic catalogue (animals.json; one channel, one kernel), brought to R = 20,
                   run from its own pattern, kept if alive at 100 units. One channel goes through a
                   colour map: `gray` (default) puts it on black like the multi-channel creatures, but it
                   is still monochrome, so a gap here mixes creature type with image style
        scores every creature with the student (mean and ensemble spread), the five-statistic reward and
        the distance to the nearest pool creature (cosine, the student's standardised embedding, as qd's
        `nn`), and makes random pairs within each set (`pairs_per` partners per creature, so each
        creature is in about twice that many comparisons) for the teacher.
label   (your Mac, API keys) pool.label on the pairs, both orders: labels_<judge>_<prompt>.csv.
report  (anywhere) per set: Bradley-Terry scores from each judge's labels; the teacher's split-half
        reliability (the ceiling for any agreement with it); agreement of the student, the
        five-statistic reward and "larger wins" with each judge (rank correlation over creatures and
        winner agreement on decisive pairs), for found creatures by pool status (train / held-out /
        never in the pool); the same with size taken out (residuals after regressing each score on log
        size); top and bottom 12 sheets per scorer, raw and size-adjusted; leaderboard.csv. Works before
        any labels exist (student-only rankings).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .alive import ALIVE, Thresholds
from .rollout import Horizon
from .sim import Rule

DECISIVE = 0.25            # |p - 0.5| above this: a decisive pair
STATS = ("speed", "change", "size_rg", "mass", "texture")
FIELDS = ["id", "set", "code", "name", "file", "in_pool", "pool_split", "passes_checks", "R0", "class", *STATS,
          "student", "spread", "features", "nn", "rule"]


# ---------- build ----------

def found_items(survey: Path, pool_dir: Path | None) -> list:
    """Every living creature of a found/ survey: (meta, rule, final pattern)."""
    survey = Path(survey)
    cal = json.loads((survey / "calibration.json").read_text()) if (survey / "calibration.json").exists() else {}
    z = np.load(survey / "alive.npz")
    pat = {str(c): z[f"p{k}"] for k, c in enumerate(z["codes"])}
    rows = {}
    for r in csv.DictReader(open(survey / "creatures.csv")):     # the last living row of a code is the one
        if r["code"] in pat and r["class"] in ALIVE:              # whose pattern alive.npz kept
            rows[r["code"]] = r
    split = {}
    if pool_dir is not None and (Path(pool_dir) / "creatures.csv").exists():
        for r in csv.DictReader(open(Path(pool_dir) / "creatures.csv")):
            if r["source"].startswith("chan:"):
                c = r["source"][5:]
                split[c] = "train" if "train" in (split.get(c), r["split"]) else r["split"]

    def passes(r):
        try:
            return (float(r["rescale_dist"]) <= cal["rescale_max_dist"]
                    and float(r["texture_loss"]) <= (cal.get("texture_max_loss") or np.inf))
        except (KeyError, ValueError, TypeError):
            return False
    out = []
    for c in sorted(rows):
        r = rows[c]
        P = np.asarray(pat[c], np.float32)
        out.append(({"code": c, "name": r["name"], "file": r["file"], "in_pool": c in split, "pool_split": split.get(c, "new"),
                     "passes_checks": passes(r), "R0": r.get("R0", r["R"])},
                    Rule.from_dict(json.loads(r["rule"])), P if P.ndim == 3 else P[None]))
    return out


def classic_items(catalogue_path: Path, R: float, shape, th: Thresholds, batch: int, max_frac: float = 0.5,
                  log=print) -> tuple:
    """Chan's classic species at kernel radius R, run from their own pattern for 100 units; the living
    ones as (meta, rule, final pattern), plus how many were tried / too big / dead."""
    from .catalogue import load
    from .explore import evaluate
    from .found import scale_to
    sp = load(catalogue_path, max_R=1e9, T=None, single_kernel=True, max_size=10 ** 9)
    keep, big = [], 0
    for s in sp:
        rule, cells = scale_to(s["rule"], np.asarray(s["cells"], np.float32)[None], R)
        if max(cells.shape[-2:]) > max_frac * min(shape):
            big += 1
            continue
        keep.append((s, rule, cells, s["rule"].R))
    hz = Horizon(time=100.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    res = evaluate([k[1] for k in keep], [k[2] for k in keep], shape, hz, th, None, batch)
    out = [({"code": s["code"], "name": s["name"], "file": "animals", "in_pool": False, "pool_split": "new",
             "passes_checks": "", "R0": R0}, rule, np.asarray(pat, np.float32))
           for (s, rule, _, R0), (_, pat, _) in zip(keep, res) if pat is not None]
    info = {"species": len(sp), "too_big": big, "run": len(keep), "alive": len(out)}
    log(f"classic: {info['alive']} alive of {info['run']} run ({info['species']} species, {big} too big at R = {R:g})")
    return out, info


def build(out: Path, survey: Path, pool_dir: Path, student_dir: Path, catalogue: Path | None = None,
          sets=("found", "classic"), shape=(256, 256), th: Thresholds = None, batch: int = 64, pairs_per: int = 8,
          classic_cmap: str = "gray", seed: int = 0, log=print) -> dict:
    from .found import calibration
    from .pool import _pairs
    from .qd import _Scorer
    from .rsearch import _camera, _render
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    th = th or Thresholds()
    rng = np.random.default_rng(seed)
    camera = _camera(pool_dir)
    cal = calibration(survey) or {}
    R = cal.get("R") or 20.0
    scorer = _Scorer(student_dir, pool_dir, "student")
    rows, pairs, info, embs = [], [], {"camera": camera, "R": R, "pairs_per": pairs_per, "seed": seed,
                                      "classic_cmap": classic_cmap, "survey": str(survey), "pool": str(pool_dir),
                                      "student": str(student_dir)}, []
    for name in sets:
        if name == "found":
            items = found_items(survey, pool_dir)
            log(f"found: {len(items)} living creatures ({sum(m['in_pool'] for m, _, _ in items)} in the pool)")
            cmap = "chan"
        elif name == "classic":
            if catalogue is None or not Path(catalogue).exists():
                log(f"classic: no catalogue at {catalogue}; skipped")
                continue
            items, info["classic"] = classic_items(catalogue, R, shape, th, batch, log=log)
            cmap = classic_cmap
        else:
            raise ValueError(f"unknown set {name!r}")
        rend = []
        for k in range(0, len(items), 128):               # in parts: the frames of hundreds of creatures add up
            part = items[k:k + 128]
            rend += [(im, st) for im, st, _ in _render([it[1] for it in part], [it[2] for it in part], shape, camera=camera, cmap=cmap)]
            log(f"  {name}: rendered {min(k + 128, len(items))}/{len(items)}")
        ok = [j for j, (_, st) in enumerate(rend) if st["class"] in ALIVE]
        info.setdefault("dropped_in_render", {})[name] = len(items) - len(ok)
        if not ok:
            continue
        sc = scorer([rend[j][0] for j in ok], [rend[j][1] for j in ok])
        ids = []
        for m, j in enumerate(ok):
            meta, rule, _ = items[j]
            cid = f"{name[0]}{j:04d}"
            rend[j][0].save(out / "images" / f"{cid}.png")
            st = rend[j][1]
            rows.append({"id": cid, "set": name, **meta, "class": st["class"],
                         **{s: (None if st[s] is None else round(float(st[s]), 5)) for s in STATS},
                         **{k: round(float(sc[k][m]), 5) for k in ("student", "spread", "features", "nn")},
                         "rule": rule.to_json()})
            ids.append(cid)
        if sc.get("emb") is not None:
            embs.append((ids, np.asarray(sc["emb"], np.float32)))
        pairs += _pairs(ids, pairs_per, rng) if len(ids) > 1 else []
        log(f"{name}: {len(ids)} creatures scored; {len(pairs)} pairs so far")
    with open(out / "creatures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "pairs.json").write_text(json.dumps(pairs))
    (out / "pairs_heldout.json").write_text("[]")         # pool.label expects it; nothing is held out here
    if embs:
        np.savez_compressed(out / f"embed_{scorer.st.model}.npz", ids=np.array(sum([e[0] for e in embs], [])),
                            emb=np.concatenate([e[1] for e in embs]), model=scorer.st.model)
    info.update({"creatures": {s: sum(r["set"] == s for r in rows) for s in sets}, "pairs": len(pairs),
                 "calls_per_judge": 2 * len(pairs)})
    (out / "build.json").write_text(json.dumps(info, indent=2))
    log(f"{len(rows)} creatures, {len(pairs)} pairs ({2 * len(pairs)} judge calls per judge) -> {out}")
    report(out, log=log)
    return info


# ---------- report ----------

def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _games(path: Path) -> list:
    """(a, b, p_a) for pairs judged in both orders."""
    out = []
    for r in csv.DictReader(open(path)):
        p1, p2 = _f(r["p_a_when_first"]), _f(r["p_a_when_second"])
        if np.isfinite(p1) and np.isfinite(p2):
            out.append((r["a"], r["b"], (p1 + p2) / 2))
    return out


def split_half(ids: list, games: list, reps: int = 5, seed: int = 0) -> float | None:
    """Spearman between Bradley-Terry scores fitted on two random halves of the pairs (mean over reps):
    how well the judge's ranking agrees with itself at half the data. Spearman-Brown gives the full-data
    reliability: 2r / (1 + r)."""
    from .vlmtest import _spearman, bradley_terry
    if len(games) < 20:
        return None
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(reps):
        m = rng.random(len(games)) < 0.5
        a = bradley_terry(ids, [g for g, k in zip(games, m) if k])
        b = bradley_terry(ids, [g for g, k in zip(games, m) if not k])
        rs.append(_spearman([a[i] for i in ids], [b[i] for i in ids]))
    return float(np.nanmean(rs))


def residual(y: np.ndarray, size: np.ndarray) -> np.ndarray:
    """y with its linear trend in log size removed (NaN where either is missing)."""
    y, x = np.asarray(y, float), np.log(np.maximum(np.asarray(size, float), 1e-6))
    ok = np.isfinite(y) & np.isfinite(x)
    out = np.full(len(y), np.nan)
    if ok.sum() < 3:
        return out
    A = np.stack([x[ok], np.ones(ok.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    out[ok] = y[ok] - A @ coef
    return out


def pair_agreement(games: list, score: dict, decisive: float = DECISIVE) -> tuple:
    """Share of decisive judged pairs where the scorer prefers the judge's winner (ties count half), and n."""
    hits = []
    for a, b, p in games:
        if abs(p - 0.5) <= decisive or a not in score or b not in score:
            continue
        d = score[a] - score[b]
        if not np.isfinite(d):
            continue
        hits.append(0.5 if d == 0 else float((d > 0) == (p > 0.5)))
    return (round(float(np.mean(hits)), 3), len(hits)) if hits else (None, 0)


def _groups(rows: list) -> dict:
    g = {}
    for r in rows:
        g.setdefault(r["set"], []).append(r["id"])
        if r["set"] == "found":
            g.setdefault(f"found: {'never in pool' if r['pool_split'] == 'new' else 'pool ' + r['pool_split']}", []).append(r["id"])
    return g


def report(out: Path, log=print) -> dict:
    from .vlmtest import _contact, _spearman, bradley_terry
    out = Path(out)
    rows = list(csv.DictReader(open(out / "creatures.csv")))
    cre = {r["id"]: r for r in rows}
    labs = {}
    for p in sorted(out.glob("labels_*_*.csv")):
        _, j, pr = p.stem.split("_", 2)
        labs[f"{j}/{pr}"] = _games(p)
    sets = list(dict.fromkeys(r["set"] for r in rows))
    size = {i: _f(cre[i]["size_rg"]) for i in cre}
    scorers = {"student": {i: _f(cre[i]["student"]) for i in cre}, "features": {i: _f(cre[i]["features"]) for i in cre},
               "larger wins": size}
    judge_scores = {}
    res = {"sets": {}, "groups": {}, "judges": list(labs)}
    for s in sets:
        ids = [r["id"] for r in rows if r["set"] == s]
        sr = {"creatures": len(ids)}
        for jn, games in labs.items():
            g = [x for x in games if x[0] in cre and x[1] in cre and cre[x[0]]["set"] == s and cre[x[1]]["set"] == s]
            if not g:
                continue
            bt = bradley_terry(ids, g)
            judge_scores.setdefault(jn, {}).update(bt)
            sh = split_half(ids, g)
            sr[jn] = {"pairs": len(g), "decisive": round(float(np.mean([abs(p - 0.5) > DECISIVE for _, _, p in g])), 3),
                      "split_half": None if sh is None else round(sh, 3),
                      "reliability": None if sh is None else round(2 * sh / (1 + sh), 3),
                      "r_size": round(_spearman([bt[i] for i in ids], [size[i] for i in ids]), 3)}
        sr["student_r_size"] = round(_spearman([scorers["student"][i] for i in ids], [size[i] for i in ids]), 3)
        sr["student_nn_median"] = round(float(np.nanmedian([_f(cre[i]["nn"]) for i in ids])), 3)
        res["sets"][s] = sr
    # agreement per group of creatures (pairs with both creatures in the group)
    for gname, ids in _groups(rows).items():
        idset = set(ids)
        gr = {"creatures": len(ids), "nn_median": round(float(np.nanmedian([_f(cre[i]["nn"]) for i in ids])), 3)}
        for jn, games in labs.items():
            if jn not in judge_scores:
                continue
            g = [x for x in games if x[0] in idset and x[1] in idset]
            J = [judge_scores[jn].get(i, np.nan) for i in ids]
            Jr = residual(J, [size[i] for i in ids])
            d = {}
            for sn, sc in scorers.items():
                S = [sc[i] for i in ids]
                d[sn] = {"rank_r": round(_spearman(S, J), 3), "pairs": pair_agreement(g, sc)}
                if sn != "larger wins":
                    d[sn]["rank_r_size_out"] = round(_spearman(residual(S, [size[i] for i in ids]), Jr), 3)
            gr[jn] = d
        res["groups"][gname] = gr
    # leaderboard and sheets
    cols = {"student": scorers["student"], **judge_scores}
    with open(out / "leaderboard.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "set", "code", "name", "pool_split", "size_rg"] + [f"{c}" for c in cols] + [f"{c} (size out)" for c in cols])
        adj = {}
        for s in sets:
            ids = [r["id"] for r in rows if r["set"] == s]
            for c, sc in cols.items():
                for i, v in zip(ids, residual([sc.get(i, np.nan) for i in ids], [size[i] for i in ids])):
                    adj[(c, i)] = v
        for s in sets:
            key = cols[next(iter(judge_scores), "student")]          # sorted by the first judge, else the student
            ids = sorted([r["id"] for r in rows if r["set"] == s], key=lambda i: -np.nan_to_num(key.get(i, np.nan), nan=-1e9))
            for i in ids:
                w.writerow([i, s, cre[i]["code"], cre[i]["name"], cre[i]["pool_split"], cre[i]["size_rg"]]
                           + [round(cols[c].get(i, np.nan), 4) for c in cols] + [round(adj[(c, i)], 4) for c in cols])
    sheets = []
    for s in sets:
        ids = [r["id"] for r in rows if r["set"] == s]
        for c, sc in cols.items():
            for tag, vals in (("", [sc.get(i, np.nan) for i in ids]), ("_size_out", residual([sc.get(i, np.nan) for i in ids], [size[i] for i in ids]))):
                ranked = [i for _, i in sorted(((v, i) for v, i in zip(vals, ids) if np.isfinite(v)), reverse=True)]
                if len(ranked) < 2:
                    continue
                k = min(12, len(ranked) // 2)
                lab = lambda i, v: f"{cre[i]['code'].split(':')[-1][:12]} {v:+.2f}"
                vv = dict(zip(ids, vals))
                name = f"rank_{s}_{c.replace('/', '_')}{tag}.png"
                _contact([(lab(i, vv[i]), out / "images" / f"{i}.png") for i in ranked[:k] + ranked[-k:]], out / name,
                         f"{s}, {c}{' with size taken out' if tag else ''}: top {k} (first rows), bottom {k} (last rows)")
                sheets.append(name)
    res["sheets"] = sheets
    (out / "report.json").write_text(json.dumps(res, indent=2))
    _markdown(out, res)
    log((out / "report.md").read_text())
    return res


def _markdown(out: Path, r: dict) -> None:
    v = lambda x, f=".3f": "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, f)
    L = ["# Catalogue rankings: student vs teacher", "",
         "Each set is ranked separately (pairs only within a set). Teacher scores: Bradley-Terry on order-averaged "
         "labels. Split-half = Spearman between rankings from two random halves of the pairs; reliability = "
         "Spearman-Brown for all the pairs, the ceiling for agreement with that judge.", "",
         "| set | creatures | median distance to pool | student r(size) | " + " | ".join(
             f"{j}: pairs / decisive / split-half / reliability / r(size)" for j in r["judges"]) + " |",
         "|---|---|---|---|" + "---|" * len(r["judges"])]
    for s, d in r["sets"].items():
        L.append(f"| {s} | {d['creatures']} | {v(d['student_nn_median'])} | {v(d['student_r_size'], '+.2f')} | " + " | ".join(
            (f"{d[j]['pairs']} / {d[j]['decisive']:.0%} / {v(d[j]['split_half'])} / {v(d[j]['reliability'])} / {v(d[j]['r_size'], '+.2f')}"
             if j in d else "-") for j in r["judges"]) + " |")
    for j in r["judges"]:
        L += ["", f"## Agreement with {j}", "",
              "rank r = Spearman over creatures; pairs = winner agreement on decisive pairs within the group (n); "
              "size out = Spearman after removing each score's trend in log size.", "",
              "| group | creatures | distance to pool | student rank r | student pairs | student size out | features rank r | "
              "features pairs | features size out | larger wins rank r | larger wins pairs |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, d in r["groups"].items():
            if j not in d:
                continue
            a = d[j]
            pr = lambda x: f"{v(x['pairs'][0])} ({x['pairs'][1]})"
            L.append(f"| {g} | {d['creatures']} | {v(d['nn_median'])} | {v(a['student']['rank_r'], '+.3f')} | {pr(a['student'])} | "
                     f"{v(a['student']['rank_r_size_out'], '+.3f')} | {v(a['features']['rank_r'], '+.3f')} | {pr(a['features'])} | "
                     f"{v(a['features']['rank_r_size_out'], '+.3f')} | {v(a['larger wins']['rank_r'], '+.3f')} | {pr(a['larger wins'])} |")
    if not r["judges"]:
        L += ["", "No teacher labels yet: student rankings only (run `lenia rank label` on your Mac, then `lenia rank report`)."]
    L += ["", "Sheets: " + ", ".join(r["sheets"]) + ". leaderboard.csv: every creature, all scores, raw and size-adjusted."]
    (Path(out) / "report.md").write_text("\n".join(L) + "\n")
