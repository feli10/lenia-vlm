"""Watch rules: animated GIFs of one rule or a grid of many, with a camera that follows the creature.

What to show is a list of "clips", each a rule plus a starting pattern and a label:
  from a survey     a rule id from <run>/rules.csv, started from its best seed (or a chosen one)
  from the catalogue a species code, started from its own catalogued pattern
  a rule given inline as JSON (Rule.to_json format or Chan's params), started from a seed of the bank
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import render, seeds as seedbank
from .alive import Thresholds, Tracker
from .rle import rle2arr
from .backend import make_batch, to_device, to_numpy
from .sim import Rule, place


def clips_from_run(run: Path, ids: list | None = None, alive_only: bool = False, seed: int | None = None,
                   limit: int = 20) -> list:
    """[(label, rule, init_spec)] from a survey folder. ids=None means every rule (or every alive one)."""
    rows = list(csv.DictReader(open(Path(run) / "rules.csv")))
    rej = Path(run) / "summary_rejudged.json"
    alive_ids = set(json.loads(rej.read_text())["alive_rule_ids"]) if rej.exists() else None
    out = []
    for r in rows:
        i = int(r["rule"])
        if ids is not None and i not in ids:
            continue
        is_alive = (i in alive_ids) if alive_ids is not None else int(r["alive_seeds"]) > 0
        if ids is None and alive_only and not is_alive:
            continue
        j = int(r["best_seed"]) if seed is None else seed
        out.append((f"#{i}", Rule.from_dict(json.loads(r["json"])), ("bank", j)))
    return out[:limit] if ids is None else out


def clips_from_catalogue(path: Path, codes: list) -> list:
    by = {x["code"]: x for x in json.loads(Path(path).read_text()) if "cells" in x and "params" in x}
    out = []
    for c in codes:
        if c not in by:
            raise SystemExit(f"no species {c} in {path}")
        out.append((c, Rule.from_chan(by[c]["params"]), ("cells", rle2arr(by[c]["cells"]))))
    return out


def clip_from_json(text: str, seed: int = 0) -> tuple:
    d = json.loads(text)
    rule = Rule.from_dict(d) if "kernels" in d else Rule.from_chan(d)
    return ("rule", rule, ("bank", seed))


def _init(spec, rule: Rule, shape) -> np.ndarray:
    kind, val = spec
    if kind == "bank":
        return seedbank.make(seedbank.DEFAULT_BANK[val], rule.R, shape, rule.channels)
    return place(val, shape, rule.channels)


def simulate(clips: list, shape=(128, 128), time: float = 50.0, every: int = 2, th: Thresholds = None) -> list:
    """Run every clip; returns [(label, rule, frames [(t, (C,H,W) float16)], stats)]. Clips are batched
    by structure; each keeps a frame every `every` steps."""
    th = th or Thresholds()
    groups = {}
    for k, (lab, rule, spec) in enumerate(clips):
        groups.setdefault((rule.structure(), rule.T), []).append(k)
    out = [None] * len(clips)
    for ks in groups.values():
        rules = [clips[k][1] for k in ks]
        T = rules[0].T
        b = make_batch(rules, shape)
        A = to_device(np.stack([_init(clips[k][2], clips[k][1], shape) for k in ks]).astype(np.float32))
        trackers = [Tracker(r.R, shape, th) for r in rules]
        frames = [[] for _ in ks]
        steps = int(round(time * T))
        rec = max(1, int(round(0.5 * T)))
        for s in range(steps + 1):
            if s % every == 0 or s % rec == 0:
                An = to_numpy(A)
            if s % every == 0:
                for i in range(len(ks)):
                    frames[i].append((s / T, An[i].astype(np.float16)))
            if s % rec == 0:
                for i, tr in enumerate(trackers):
                    tr.add(s / T, An[i], count_components=(s == steps))
            if s < steps:
                A = b.step(A)
        for i, k in enumerate(ks):
            out[k] = (clips[k][0], clips[k][1], frames[i], trackers[i].summary())
    return out


def _tile(frame, R, cmap, px, n, follow: bool) -> Image.Image:
    if follow:
        a = render.crop(frame, R, 0, n)
    else:
        a = np.asarray(frame, np.float32)
    return Image.fromarray(render.colorize(a, cmap)).resize((px, px), Image.NEAREST if a.shape[-1] <= px else Image.BILINEAR)


def gif(sims: list, path: Path, px: int = 192, cols: int | None = None, cmap: str = "chan", follow: bool = True,
        crop_r: float = 3.0, fps: int = 20, label: bool = True, start: float = 0.0, bg=(245, 245, 245)) -> Path:
    """Animate one or more simulations side by side. follow=True: a camera centred on the creature, sized
    to hold its largest frame (from `start` on); follow=False: the whole world."""
    n_clips = len(sims)
    cols = cols or min(n_clips, 5)
    rows = (n_clips + cols - 1) // cols
    lh = 14 if label else 0
    pad = 4
    sizes = []
    for lab, rule, frames, st in sims:
        H = min(np.asarray(frames[0][1]).shape[-2:])
        use = [f for f in frames if f[0] >= start] or frames
        sizes.append(render.crop_size(use[:: max(1, len(use) // 40)], rule.R, crop_r, H) if follow else None)
    n_frames = min(len(f) for _, _, f, _ in sims)
    first = next((i for i, f in enumerate(sims[0][2]) if f[0] >= start), 0)
    images = []
    for fi in range(first, n_frames):
        canvas = Image.new("RGB", (cols * (px + pad) + pad, rows * (px + lh + pad) + pad + lh), bg)
        d = ImageDraw.Draw(canvas)
        for k, (lab, rule, frames, st) in enumerate(sims):
            r, c = divmod(k, cols)
            x, y = pad + c * (px + pad), pad + r * (px + lh + pad)
            if label:
                d.text((x, y), f"{lab} {st['class']}", fill=(30, 30, 30))
            canvas.paste(_tile(frames[fi][1], rule.R, cmap, px, sizes[k], follow), (x, y + lh))
        if label:
            d.text((pad, canvas.height - 12), f"t = {sims[0][2][fi][0]:.1f}", fill=(90, 90, 90))
        images.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path
