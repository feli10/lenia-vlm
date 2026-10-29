"""Chan's `found/` collections: ~3,000 2D Lenia creatures with several kernels and channels.

github.com/Chakazul/Lenia, Python/found/ (same licence as the catalogue). File names encode the
setup: 2DCK, e.g. 233 = 2D, 3 channels, kernel layout 3 (15 kernels); 213 = 1 channel, 3 kernels.
Each file is a sequence of concatenated JSON objects {code, name, cname, params, cells}: params is
Chan's list of kernels (R, T, b, m, s, h, r, c = [source, target channel]), cells one RLE string per
channel.

  download   fetch the files into data/found/
  load       parse them into {code, name, file, rule, cells (C, h, w)}, optionally brought to a common
             kernel radius (pattern resampled, rule R changed; Lenia is scale-free up to the grid)
  survey     run every creature from its own pattern, classify, describe; write the living ones
             (roots and reference set for `lenia explore --from-found`) and colour contact sheets;
             then calibrate the coherence checks on the living ones (calibration.json)

Common R (the config's world.R; 20 in configs/found.yaml): Chan's creatures range from R = 8 to 41.
At a common R they are comparable in size and resolution, and fine structure is resolved by several
cells. (In the first Experiment A run, the most novel family descended from the one R = 12 root and
had stripes 1-2 cells wide; re-run at R = 26 the stripes widened with R, so they were real, but at
R = 12 a picture cannot tell real fine structure from grid noise.)

The default files are the multi-channel ones (2 or 3 channels), so every creature renders as colour on
black; the single-channel files (212, 213) render on white and would confound VLM comparisons.
"""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from .alive import ALIVE, Thresholds
from .rle import rle2arr
from .sim import Rule

BASE = "https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/found/"
# 2D files; channels / kernels / creatures (as of 2026-10)
FILES = {"212": "1 ch, 2 kernels, 44", "213": "1 ch, 3 kernels, 88", "221": "2 ch, 4 kernels, 40",
         "222": "2 ch, 6 kernels, 9", "223": "2 ch, 8 kernels, 21", "231": "3 ch, 9 kernels, 23",
         "231b": "3 ch, 9 kernels, 752", "232": "3 ch, 12 kernels, 63", "233": "3 ch, 15 kernels, 954",
         "233b": "3 ch, 15 kernels, 42", "233c": "3 ch, 15 kernels, 642", "233d": "3 ch, 15 kernels, 41",
         "233s": "3 ch, 15 kernels, 166", "241": "4 ch, 16 kernels, 48", "242": "4 ch, 20 kernels, 12",
         "243": "4 ch, 24 kernels, 19", "261": "6 ch, 36 kernels, 16"}
DEFAULT = ["221", "223", "231", "231b", "232", "233", "233c", "233s"]
CALIBRATION_QUANTILES = {"rescale": 95, "texture_loss": 99}   # 2x: same rule as the classic calibration
# Real creatures keep their fine structure at 2x up to a few hundredths (pulsing, phase), so their p99 can
# be very tight; grid noise loses most of it (0.1-0.3). The threshold is never below this floor, about
# the whole fine structure of a median creature (texture 0.03), so ordinary variation is not rejected.
TEXTURE_LOSS_FLOOR = 0.03


def download(names: list, dest: Path, log=print) -> list:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = dest / f"{n}.json"
        if not p.exists():
            log(f"  downloading {n}.json")
            with urllib.request.urlopen(BASE + f"{n}.json", timeout=120) as r:
                p.write_bytes(r.read())
        out.append(p)
    return out


def parse(path: Path) -> list:
    s = Path(path).read_text()
    dec = json.JSONDecoder()
    i, out = 0, []
    while i < len(s):
        while i < len(s) and s[i] in " \n\r\t,":
            i += 1
        if i >= len(s):
            break
        try:
            o, i = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        if isinstance(o, dict) and "params" in o and "cells" in o:
            out.append(o)
    return out


def _cells(c) -> np.ndarray:
    arrs = [rle2arr(x) for x in (c if isinstance(c, list) else [c])]
    h, w = max(a.shape[0] for a in arrs), max(a.shape[1] for a in arrs)
    A = np.zeros((len(arrs), h, w), np.float32)
    for k, a in enumerate(arrs):
        A[k, :a.shape[0], :a.shape[1]] = a
    return np.clip(A, 0, 1)


def scale_to(rule: Rule, cells: np.ndarray, R: float) -> tuple:
    """The same creature at kernel radius R: rule R changed, pattern resampled by R / rule.R."""
    from .explore import rescaled, upsample
    f = float(R) / float(rule.R)
    if abs(f - 1) < 1e-6:
        return rule, cells
    return rescaled(rule, f), upsample(cells, f)


def load(folder: Path, names: list, max_size: int = 128, target_R: float | None = None, log=print) -> list:
    """Creatures from the given files; drops duplicates, malformed records and patterns wider than
    max_size (after scaling to target_R, if given; the original R is kept as R0)."""
    out, seen, bad = [], set(), 0
    for n in names:
        p = Path(folder) / f"{n}.json"
        if not p.exists():
            raise SystemExit(f"{p} missing: run `lenia found download` first")
        for k, o in enumerate(parse(p)):
            key = hashlib.sha1(json.dumps([o["params"], o["cells"]], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            try:
                rule = Rule.from_chan(o["params"])
                cells = _cells(o["cells"])
            except Exception:
                bad += 1
                continue
            if cells.shape[0] != rule.channels:
                bad += 1
                continue
            R0 = float(rule.R)
            if target_R:
                rule, cells = scale_to(rule, cells, target_R)
            if max(cells.shape[1:]) > max_size:
                bad += 1
                continue
            code = str(o.get("code") or "").strip() or f"#{k}"
            out.append({"code": f"{n}:{code}", "name": str(o.get("name") or ""), "file": n, "rule": rule, "cells": cells,
                        "R0": R0})
    log(f"found: {len(out)} creatures from {', '.join(names)} ({bad} skipped: malformed or too big)"
        + (f", all at R = {target_R:g}" if target_R else ""))
    return out


def survey(creatures: list, out: Path, shape, th: Thresholds = None, batch: int = 16, ex=None, time_units: float = 100.0,
           sheet_per_file: int = 48, calibrate: bool = True, rescale_time: float = 50.0, log=print) -> dict:
    """Run each creature from its own pattern; write creatures.csv, alive.npz (final patterns + descriptors
    of the living ones, the roots and reference set for explore) and sheet_<file>.png. With calibrate,
    also run every living creature at 2x resolution and write calibration.json: the thresholds
    `lenia explore --from-found` then uses (2x descriptor distance: p95 of the creatures alive at both
    scales; texture loss at 2x: p99), so the checks are calibrated on the creatures being searched."""
    from . import render
    from .describe import Space
    from .explore import evaluate, rescale_check, texture, texture_loss
    from .rollout import Horizon
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    th = th or Thresholds()
    hz = Horizon(time=time_units, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    res = evaluate([c["rule"] for c in creatures], [c["cells"] for c in creatures], shape, hz, th, ex, batch)
    rows, pats, descs = [], {}, {}
    for c, (st, pat, d) in zip(creatures, res):
        st.pop("desc_half_gap", None)
        st.pop("_trace", None)
        rows.append({"code": c["code"], "name": c["name"], "file": c["file"], "channels": c["rule"].channels,
                     "kernels": len(c["rule"].kernels), "R": c["rule"].R, "R0": c.get("R0", c["rule"].R), "class": st["class"],
                     "speed": st.get("speed_w"), "change": st.get("change_w"), "mass": st.get("mass_end"),
                     "texture": "", "rescale_class": "", "rescale_dist": "", "texture_loss": "", "rule": c["rule"].to_json()})
        if pat is not None:
            pats[c["code"]] = np.asarray(pat, np.float16)
            descs[c["code"]] = d
    codes = sorted(pats)
    calib = None
    if calibrate and codes:
        by_code = {c["code"]: c for c in creatures}
        tex = {k: texture(pats[k]) for k in codes}
        log(f"calibrating on {len(codes)} living creatures: 2x resolution for {rescale_time:g} units ...")
        r2 = rescale_check([by_code[k]["rule"] for k in codes], [np.asarray(pats[k], np.float32) for k in codes],
                           shape, th, ex, batch, 2, rescale_time, with_pattern=True)
        space = Space(np.array([descs[k] for k in codes]))
        dist, loss = {}, {}
        for k, (cl, d2, p2) in zip(codes, r2):
            ok = cl in ALIVE and d2 is not None
            dist[k] = (cl, float(space.dist(descs[k], np.asarray(d2)[None])[0]) if ok else np.inf)
            if ok:
                loss[k] = texture_loss(pats[k], p2)
        for r in rows:
            if r["code"] in dist:
                r["texture"] = round(tex[r["code"]], 5)
                r["rescale_class"], r["rescale_dist"] = dist[r["code"]][0], round(dist[r["code"]][1], 4)
                r["texture_loss"] = round(loss[r["code"]], 5) if r["code"] in loss else ""
        both = np.array([v for cl, v in dist.values() if np.isfinite(v)])
        tv = np.array(list(loss.values()))
        qr, qt = CALIBRATION_QUANTILES["rescale"], CALIBRATION_QUANTILES["texture_loss"]
        calib = {"R": float(creatures[0]["rule"].R) if len({c["rule"].R for c in creatures}) == 1 else None,
                 "living": len(codes), "alive_at_2x": int(len(both)),
                 "rescale_max_dist": round(float(np.percentile(both, qr)), 3) if len(both) else None,
                 "rescale_dist_quantiles": {f"p{q}": round(float(np.percentile(both, q)), 3) for q in (50, 90, 95, 99)} if len(both) else None,
                 "texture_max_loss": round(max(float(np.percentile(tv, qt)), TEXTURE_LOSS_FLOOR), 4) if len(tv) else None,
                 "texture_loss_floor": TEXTURE_LOSS_FLOOR,
                 "texture_loss_quantiles": {f"p{q}": round(float(np.percentile(tv, q)), 4) for q in (1, 50, 90, 95, 99)} if len(tv) else None,
                 "texture_quantiles": {f"p{q}": round(float(np.percentile(list(tex.values()), q)), 4) for q in (50, 90, 99)},
                 "rule": f"2x: p{qr} of descriptor distance among creatures alive at both scales; "
                         f"texture loss at 2x: p{qt} among the same, at least {TEXTURE_LOSS_FLOOR}"}
        (out / "calibration.json").write_text(json.dumps(calib, indent=2))
        R = {c["code"]: c["rule"].R for c in creatures}
        if loss:      # what the texture-loss threshold is about: most fine structure, most lost at 2x
            items = [(f"{k.split(':', 1)[1][:8]} t{tex[k]:.2f} l{loss.get(k, np.nan):+.2f}",
                      render.frame_image(pats[k], R[k], "chan", 112, None))
                     for k in sorted(codes, key=lambda k: -tex[k])[:12] + sorted(loss, key=lambda k: -loss[k])[:12]]
            render.sheet(items, cols=6, title=f"top 2 rows: most fine structure (t = texture); bottom 2: most lost at 2x "
                                              f"(l = loss; threshold {calib['texture_max_loss']:.3f})").save(out / "sheet_texture.png")
        log("calibration: " + json.dumps({k: calib[k] for k in ("living", "alive_at_2x", "rescale_max_dist", "texture_max_loss")}))
    with open(out / "creatures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "alive.npz", codes=np.array(codes), desc=np.array([descs[c] for c in codes], np.float32),
                        **{f"p{k}": pats[c] for k, c in enumerate(codes)})
    by_file = {}
    for r in rows:
        f = by_file.setdefault(r["file"], {"creatures": 0, "alive": 0, "classes": {}})
        f["creatures"] += 1
        f["alive"] += r["class"] in ALIVE
        f["classes"][r["class"]] = f["classes"].get(r["class"], 0) + 1
    for n, f in by_file.items():
        f["alive_fraction"] = round(f["alive"] / f["creatures"], 3)
        alive = [r for r in rows if r["file"] == n and r["class"] in ALIVE][:sheet_per_file]
        if alive:
            R = {r["code"]: r["R"] for r in alive}
            items = [(f"{r['code'].split(':', 1)[1][:10]} {r['class']}",
                      render.frame_image(pats[r["code"]], R[r["code"]], "chan", 112, None)) for r in alive]
            render.sheet(items, cols=6, title=f"found {n} ({FILES.get(n, '')}): living, final frame").save(out / f"sheet_{n}.png")
    summary = {"creatures": len(rows), "alive": len(codes), "per_file": by_file, "shape": list(shape), "time": time_units,
               "calibration": calib}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({"creatures": len(rows), "alive": len(codes),
                    "per_file": {n: f"{f['alive']}/{f['creatures']}" for n, f in by_file.items()}}))
    return summary


def calibration(run: Path) -> dict | None:
    p = Path(run) / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


def roots_from_survey(run: Path, n: int = 64, seed: int = 0) -> tuple:
    """n diverse living creatures (farthest-point sampling on descriptors) as explore roots, plus the
    descriptors of every living creature as the reference set. Roots: [(label, rule, own pattern)].
    With a calibrated survey, roots are drawn only from creatures that pass the calibrated 2x checks
    (farthest-point sampling otherwise favours outliers, which are the ones that fail them)."""
    from .describe import Space
    run = Path(run)
    z = np.load(run / "alive.npz")
    codes, desc = list(z["codes"]), z["desc"]
    rows = {r["code"]: r for r in csv.DictReader(open(run / "creatures.csv"))}
    rules = {c: Rule.from_dict(json.loads(r["rule"])) for c, r in rows.items()}
    space = Space(desc)
    pool = list(range(len(codes)))
    cal = calibration(run)
    if cal and cal.get("rescale_max_dist") is not None:
        def passes(r):
            try:
                return (float(r["rescale_dist"]) <= cal["rescale_max_dist"]
                        and float(r["texture_loss"]) <= (cal.get("texture_max_loss") or np.inf))
            except (KeyError, ValueError):
                return False
        pool = [k for k in pool if passes(rows[str(codes[k])])] or pool
    rng = np.random.default_rng(seed)
    chosen = [pool[int(rng.integers(len(pool)))]]
    dmin = space.dist(desc[chosen[0]], desc[pool])
    while len(chosen) < min(n, len(pool)):
        k = pool[int(np.argmax(dmin))]
        chosen.append(k)
        dmin = np.minimum(dmin, space.dist(desc[k], desc[pool]))
    roots = [(f"found {codes[k]}", rules[str(codes[k])], np.asarray(z[f"p{k}"], np.float32)) for k in chosen]
    return roots, desc
