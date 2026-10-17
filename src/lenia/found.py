"""Chan's `found/` collections: ~3,000 2D Lenia creatures with several kernels and channels.

github.com/Chakazul/Lenia, Python/found/ (same licence as the catalogue). File names encode the
setup: 2DCK, e.g. 233 = 2D, 3 channels, kernel layout 3 (15 kernels); 213 = 1 channel, 3 kernels.
Each file is a sequence of concatenated JSON objects {code, name, cname, params, cells}: params is
Chan's list of kernels (R, T, b, m, s, h, r, c = [source, target channel]), cells one RLE string per
channel.

  download   fetch the files into data/found/
  load       parse them into {code, name, file, rule, cells (C, h, w)}
  survey     run every creature from its own pattern, classify, describe; write the living ones
             (roots and reference set for `lenia explore --from-found`) and colour contact sheets
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
DEFAULT = ["212", "213", "221", "223", "231", "231b", "232", "233", "233c", "233s"]


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


def load(folder: Path, names: list, max_size: int = 128, log=print) -> list:
    """Creatures from the given files; drops duplicates, malformed records and patterns wider than max_size."""
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
            if cells.shape[0] != rule.channels or max(cells.shape[1:]) > max_size:
                bad += 1
                continue
            code = str(o.get("code") or "").strip() or f"#{k}"
            out.append({"code": f"{n}:{code}", "name": str(o.get("name") or ""), "file": n, "rule": rule, "cells": cells})
    log(f"found: {len(out)} creatures from {', '.join(names)} ({bad} skipped: malformed or too big)")
    return out


def survey(creatures: list, out: Path, shape, th: Thresholds = None, batch: int = 16, ex=None, time_units: float = 100.0,
           sheet_per_file: int = 48, log=print) -> dict:
    """Run each creature from its own pattern; write creatures.csv, alive.npz (final patterns + descriptors
    of the living ones, the roots and reference set for explore) and sheet_<file>.png."""
    from . import render
    from .explore import evaluate
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
                     "kernels": len(c["rule"].kernels), "R": c["rule"].R, "class": st["class"],
                     "speed": st.get("speed_w"), "change": st.get("change_w"), "mass": st.get("mass_end"),
                     "rule": c["rule"].to_json()})
        if pat is not None:
            pats[c["code"]] = np.asarray(pat, np.float16)
            descs[c["code"]] = d
    with open(out / "creatures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    codes = sorted(pats)
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
    summary = {"creatures": len(rows), "alive": len(codes), "per_file": by_file, "shape": list(shape), "time": time_units}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({"creatures": len(rows), "alive": len(codes),
                    "per_file": {n: f"{f['alive']}/{f['creatures']}" for n, f in by_file.items()}}))
    return summary


def roots_from_survey(run: Path, n: int = 64, seed: int = 0) -> tuple:
    """n diverse living creatures (farthest-point sampling on descriptors) as explore roots, plus the
    descriptors of every living creature as the reference set. Roots: [(label, rule, own pattern)]."""
    from .describe import Space
    run = Path(run)
    z = np.load(run / "alive.npz")
    codes, desc = list(z["codes"]), z["desc"]
    rules = {r["code"]: Rule.from_dict(json.loads(r["rule"])) for r in csv.DictReader(open(run / "creatures.csv"))}
    space = Space(desc)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(codes)))]
    dmin = space.dist(desc[chosen[0]], desc)
    while len(chosen) < min(n, len(codes)):
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, space.dist(desc[k], desc))
    roots = [(f"found {codes[k]}", rules[str(codes[k])], np.asarray(z[f"p{k}"], np.float32)) for k in chosen]
    return roots, desc
