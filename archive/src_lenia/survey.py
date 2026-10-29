"""Test 1 and Test 2: run many rules from the seed bank, classify every rollout, draw contact sheets.

Outputs (in the run folder):
  rollouts.csv    one row per (rule, seed): class and statistics
  rules.csv       one row per rule: its classes over the bank, how many seeds gave a living creature,
                  the best seed, and the rule itself (JSON)
  summary.json    counts per class (per rollout and per rule) and timing
  sheet_alive.png filmstrips of living creatures (one per rule, its best seed)
  sheet_<class>.png filmstrips of a sample of each failure class
  grid.png        (grid mode) the (m, s) plane coloured by outcome
"""
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import render, seeds as seedbank
from .alive import ALIVE, CLASSES, Thresholds, classify
from .rollout import Horizon, run_batch
from .sim import Rule

STAT_KEYS = ["mass_end", "mass_min_w", "mass_max_w", "mass_start_w", "occ_max_w", "rg_max_w", "speed_w", "change_w",
             "bodies", "growth_w", "rg_growth_w", "t_end"]


def _work(args):
    rule_dicts, jobs, bank, shape, hz, th, keep_film = args
    rules = {i: Rule.from_dict(d) for i, d in rule_dicts.items()}
    rr = [rules[i] for i, _ in jobs]
    inits = np.stack([seedbank.make(bank[j], rules[i].R, shape, rules[i].channels) for i, j in jobs])
    res = run_batch(rr, inits, shape, hz, th, [j for _, j in jobs])
    out = []
    for (i, j), r in zip(jobs, res):
        film = r.film if keep_film else None
        out.append((i, j, r.stats, film, np.clip(np.nan_to_num(r.trace, nan=np.nan, posinf=6e4), -6e4, 6e4).astype(np.float16), r.thumbs))
    return out


def survey(rules: list, out: Path, bank=None, shape=(128, 128), hz: Horizon = None, th: Thresholds = None,
           workers: int = 0, batch: int = 16, cmap: str = "chan", px: int = 96, crop_r: float = 3.0,
           max_alive: int = 120, max_per_class: int = 24, rng_seed: int = 0, grid_shape=None, log=print) -> dict:
    bank = bank or seedbank.DEFAULT_BANK
    hz, th = hz or Horizon(), th or Thresholds()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rule_dicts = [r.to_dict() for r in rules]
    jobs = [(i, j) for i in range(len(rules)) for j in range(len(bank))]
    # group by structure so a batch shares kernels' wiring; keep T uniform per batch
    groups = {}
    for i, j in jobs:
        groups.setdefault((rules[i].structure(), rules[i].T), []).append((i, j))
    chunks = [g[k:k + batch] for g in groups.values() for k in range(0, len(g), batch)]
    t0 = time.time()
    stats = {}
    films = {}
    traces, thumbs = {}, {}
    done = 0
    args = [({i: rule_dicts[i] for i, _ in ch}, ch, bank, tuple(shape), hz, th, True) for ch in chunks]
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    it = ex.map(_work, args) if ex else map(_work, args)
    rng = np.random.default_rng(rng_seed)
    for res in it:
        for i, j, s, film, tr, th_ in res:
            stats[(i, j)] = s
            traces[(i, j)] = tr
            # keep films and thumbnails for alive outcomes and a random subset of the rest (memory)
            if film is not None and (s["alive"] or rng.random() < max(0.02, 4 * max_per_class / max(1, len(jobs)))):
                films[(i, j)] = film
                thumbs[(i, j)] = th_
        done += len(res)
        if done % max(1, len(jobs) // 10) < len(res):
            log(f"  {done}/{len(jobs)} rollouts, {time.time() - t0:.0f}s")
    if ex:
        ex.shutdown()
    secs = time.time() - t0

    # per-rollout table
    with open(out / "rollouts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "seed", "seed_label", "class", "alive"] + STAT_KEYS)
        for (i, j), s in sorted(stats.items()):
            w.writerow([i, j, seedbank.label(bank[j]), s["class"], int(s["alive"])] + [s.get(k) for k in STAT_KEYS])

    # full statistics time series of every rollout (t, mass, occupancy, spread, speed, change), and
    # 32x32 thumbnails every few units of time for the rollouts we keep pictures of
    np.savez_compressed(out / "traces.npz", **{f"{i}_{j}": v for (i, j), v in traces.items()})
    np.savez_compressed(out / "thumbs.npz", **{f"{i}_{j}": np.stack([x for _, x in v]) for (i, j), v in thumbs.items() if v},
                        **{f"t_{i}_{j}": np.array([t for t, _ in v], np.float32) for (i, j), v in thumbs.items() if v})

    # per-rule table
    rule_rows = []
    for i, r in enumerate(rules):
        cls = [stats[(i, j)]["class"] for j in range(len(bank))]
        alive_js = [j for j in range(len(bank)) if stats[(i, j)]["alive"]]
        # best seed: alive and fastest-changing (most 'lively'), else the first
        best = max(alive_js, key=lambda j: stats[(i, j)]["change_w"]) if alive_js else 0
        verdict = stats[(i, best)]["class"] if alive_js else max(set(cls), key=cls.count)
        rule_rows.append({"rule": i, "verdict": verdict, "alive_seeds": len(alive_js), "best_seed": best,
                          "classes": "|".join(cls), "json": r.to_json()})
    with open(out / "rules.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rule_rows[0]))
        w.writeheader()
        w.writerows(rule_rows)

    per_rollout = {c: sum(s["class"] == c for s in stats.values()) for c in CLASSES}
    per_rule = {c: sum(rr["verdict"] == c for rr in rule_rows) for c in CLASSES}
    alive_rules = sum(rr["alive_seeds"] > 0 for rr in rule_rows)
    summary = {"rules": len(rules), "seeds": len(bank), "rollouts": len(jobs), "seconds": round(secs, 1),
               "alive_rules": alive_rules, "alive_rule_fraction": round(alive_rules / len(rules), 4),
               "alive_rollout_fraction": round(sum(s["alive"] for s in stats.values()) / len(jobs), 4),
               "per_rollout": per_rollout, "per_rule_verdict": per_rule,
               "seed_alive_rate": {seedbank.label(b): round(np.mean([stats[(i, j)]["alive"] for i in range(len(rules))]), 4)
                                   for j, b in enumerate(bank)},
               "shape": list(shape), "horizon": vars(hz), "thresholds": vars(th)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # contact sheets
    R = rules[0].R
    alive_items = []
    for rr in rule_rows:
        key = (rr["rule"], rr["best_seed"])
        if rr["alive_seeds"] and key in films:
            s = stats[key]
            lab = f"#{rr['rule']} {s['class']} seeds={rr['alive_seeds']}/{len(bank)} v={s['speed_w']:.2f}"
            alive_items.append((lab, render.filmstrip(films[key], R, cmap, px, crop_r)))
    if alive_items:
        idx = rng.permutation(len(alive_items))[:max_alive]
        render.sheet([alive_items[k] for k in sorted(idx)], cols=2,
                     title=f"alive ({len(alive_items)} rules), best seed, last {hz.film_span:g} time units").save(out / "sheet_alive.png")
    for c in CLASSES:
        if c in ALIVE:
            continue
        keys = [k for k in films if stats[k]["class"] == c]
        if not keys:
            continue
        keys = [keys[k] for k in rng.permutation(len(keys))[:max_per_class]]
        localized = c in ("frozen", "dying", "growing", "multi")
        items = [(f"#{i} {seedbank.label(bank[j])}", render.filmstrip(films[(i, j)], R, cmap, px, crop_r if localized else None))
                 for i, j in sorted(keys)]
        render.sheet(items, cols=2, title=f"{c} (sample)").save(out / f"sheet_{c}.png")

    if grid_shape:
        _grid_plot(rules, rule_rows, grid_shape, out / "grid.png")
    log(json.dumps({k: summary[k] for k in ("rules", "alive_rules", "alive_rule_fraction", "per_rule_verdict", "seconds")}))
    return summary


COLORS = {"fizzled": "#dddddd", "exploded": "#7a0019", "spread": "#e8853a", "dying": "#bbbbbb",
          "growing": "#f2c14e", "multi": "#8e7cc3", "frozen": "#4f6d7a", "mover": "#2a9d4b", "pulser": "#7fd18b"}


def _grid_plot(rules, rule_rows, grid_shape, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ny, nx = grid_shape
    ms = sorted({r.kernels[0].m for r in rules})
    ss = sorted({r.kernels[0].s for r in rules})
    img = np.ones((len(ss), len(ms), 3))
    for r, row in zip(rules, rule_rows):
        y, x = ss.index(r.kernels[0].s), ms.index(r.kernels[0].m)
        img[y, x] = matplotlib.colors.to_rgb(COLORS[row["verdict"]])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(img, origin="lower", aspect="auto", extent=[ms[0], ms[-1], 0, len(ss)])
    ticks = np.linspace(0, len(ss) - 1, 6).astype(int)
    ax.set_yticks(ticks + 0.5)
    ax.set_yticklabels([f"{ss[t]:.3f}" for t in ticks])
    ax.set_xlabel("growth centre m")
    ax.set_ylabel("growth width s (log spaced)")
    ax.set_title("classic Lenia, one ring: outcome per rule (alive if any seed lives)")
    ax.legend(handles=[Patch(color=v, label=k) for k, v in COLORS.items()], bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rejudge(run: Path, th: Thresholds, log=print) -> dict:
    """Re-classify a finished survey from rollouts.csv with new thresholds (no simulation, no pictures)."""
    run = Path(run)
    rows = list(csv.DictReader(open(run / "rollouts.csv")))
    by_rule = {}
    for r in rows:
        st = {k: (float(r[k]) if r.get(k) not in (None, "") else None) for k in STAT_KEYS}
        if st["mass_start_w"] is None:          # older runs: no 'growing' check possible
            st["mass_start_w"] = st["mass_end"]
        r["class"] = classify(st, th)
        by_rule.setdefault(int(r["rule"]), []).append(r["class"])
    per_rollout = {c: sum(r["class"] == c for r in rows) for c in CLASSES}
    verdict = {}
    for i, cls in by_rule.items():
        alive = [c for c in cls if c in ALIVE]
        verdict[i] = alive[0] if alive else max(set(cls), key=cls.count)
    alive_rules = sum(any(c in ALIVE for c in cls) for cls in by_rule.values())
    out = {"rules": len(by_rule), "alive_rules": alive_rules, "alive_rule_fraction": round(alive_rules / len(by_rule), 4),
           "per_rollout": per_rollout, "per_rule_verdict": {c: sum(v == c for v in verdict.values()) for c in CLASSES},
           "thresholds": vars(th), "alive_rule_ids": sorted(i for i, cls in by_rule.items() if any(c in ALIVE for c in cls))}
    (run / "summary_rejudged.json").write_text(json.dumps(out, indent=2))
    log(json.dumps({k: out[k] for k in ("rules", "alive_rules", "alive_rule_fraction", "per_rule_verdict")}))
    return out
