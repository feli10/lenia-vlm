"""Bert Chan's Lenia catalogue (animals.json, MIT licence): the human-curated reference set.

Get it once:  curl -L -o data/animals.json \\
              https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json

Two uses:
  own   run each species from its own catalogued pattern: the classifier should call most of them
        alive (a check on the classifier), and the pictures are the reference for Test 2
  bank  run each species' rule from the seed bank instead: how often does the bank find a living
        creature under rules known to have one? (a check on the seed bank)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import render, seeds as seedbank
from .alive import CLASSES, Thresholds
from .rle import rle2arr
from .rollout import Horizon, run_batch
from .sim import Rule, place


def load(path, max_R: float = 20, T: float | None = 10, single_kernel: bool = True, max_size: int = 96) -> list:
    """Species with params and cells; by default classic ones with R <= max_R whose pattern is at most
    max_size cells wide (they fit a 128 world with room to move)."""
    out = []
    for x in json.loads(Path(path).read_text()):
        if "cells" not in x or "params" not in x:
            continue
        p = x["params"]
        if single_kernel and isinstance(p, list):
            continue
        rule = Rule.from_chan(p)
        if rule.R > max_R or (T is not None and rule.T != T):
            continue
        cells = rle2arr(x["cells"])
        if max(cells.shape) > max_size:
            continue
        out.append({"code": x["code"], "name": x["name"], "rule": rule, "cells": cells})
    return out


def run(species: list, start: str, out: Path, shape=(128, 128), hz: Horizon = None, th: Thresholds = None,
        bank=None, cmap="chan", px=96, crop_r=3.0, batch=16, max_sheet=60, seed=0, log=print) -> dict:
    hz, th = hz or Horizon(), th or Thresholds()
    bank = bank or seedbank.DEFAULT_BANK
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if start == "own":
        jobs = [(k, -1) for k in range(len(species))]
    else:
        jobs = [(k, j) for k in range(len(species)) for j in range(len(bank))]
    results = {}
    groups = {}
    for k, j in jobs:   # species differ in kernel core / growth function: batch like with like
        groups.setdefault((species[k]["rule"].structure(), species[k]["rule"].T), []).append((k, j))
    chunks = [g[i:i + batch] for g in groups.values() for i in range(0, len(g), batch)]
    done = 0
    for ch in chunks:
        rules = [species[k]["rule"] for k, _ in ch]
        inits = np.stack([place(species[k]["cells"], shape) if j < 0 else
                          seedbank.make(bank[j], species[k]["rule"].R, shape, 1) for k, j in ch])
        for (k, j), r in zip(ch, run_batch(rules, inits, shape, hz, th, [j for _, j in ch])):
            results[(k, j)] = r
        done += len(ch)
        log(f"  {done}/{len(jobs)}")
    rows, items = [], []
    for k, sp in enumerate(species):
        rs = [results[(k, j)] for kk, j in jobs if kk == k]
        alive = [r for r in rs if r.stats["alive"]]
        best = max(alive, key=lambda r: r.stats["change_w"]) if alive else rs[0]
        rows.append({"code": sp["code"], "name": sp["name"], "alive_starts": len(alive), "starts": len(rs),
                     "class": best.stats["class"], "classes": "|".join(r.stats["class"] for r in rs)})
        items.append((f"{sp['code']} {best.stats['class']}", render.filmstrip(best.film, sp["rule"].R, cmap, px, crop_r)))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.permutation(len(items))[:max_sheet])
    render.sheet([items[i] for i in pick], cols=2, title=f"Chan's catalogue, start={start}").save(out / f"catalogue_{start}.png")
    (out / f"catalogue_{start}.json").write_text(json.dumps(rows, indent=1))
    n_alive = sum(r["alive_starts"] > 0 for r in rows)
    summ = {"start": start, "species": len(rows), "alive": n_alive, "alive_fraction": round(n_alive / max(1, len(rows)), 3),
            "classes": {c: sum(r["class"] == c for r in rows) for c in CLASSES}}
    log(json.dumps(summ))
    return summ
