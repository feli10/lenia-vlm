"""Plot what happened to rollouts over time, from a run's traces.npz (and thumbs.npz if there).

Survey folders key rollouts by "rule_seed" (default seed: the rule's best seed); explore folders by
archive id. Each rollout gets one row: mass, occupancy, spread and speed over time, the shaded
judging window, and (survey runs) the 32x32 thumbnails taken every few units of time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def keys_for(run: Path, ids: list, seed: int | None) -> list:
    run = Path(run)
    if (run / "rules.csv").exists():
        rows = {int(r["rule"]): r for r in csv.DictReader(open(run / "rules.csv"))}
        ro = {(int(r["rule"]), int(r["seed"])): r["class"] for r in csv.DictReader(open(run / "rollouts.csv"))}
        out = []
        for i in ids:
            j = seed if seed is not None else int(rows[i]["best_seed"])
            out.append((f"#{i} seed {j}: {ro.get((i, j), '?')}  (rule over all seeds: {rows[i]['verdict']})", f"{i}_{j}"))
        return out
    rows = {int(r["id"]): r for r in csv.DictReader(open(run / "archive.csv"))}
    return [(f"#{i} ({rows[i]['class']})", str(i)) for i in ids]


def _horizon(run: Path, default: float) -> float:
    """The run's full rollout length (survey: horizon.time; explore: confirm_time or child_time)."""
    p = Path(run) / "summary.json"
    if not p.exists():
        return default
    s = json.loads(p.read_text())
    if "horizon" in s:
        return float(s["horizon"]["time"])
    return float(s.get("confirm_time") or s.get("child_time") or default)


def plot(run: Path, ids: list, out: Path, seed: int | None = None, th=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    tr = np.load(run / "traces.npz")
    thumbs = np.load(run / "thumbs.npz") if (run / "thumbs.npz").exists() else None
    keys = keys_for(run, ids, seed)
    n = len(keys)
    has_th = thumbs is not None and any(k in thumbs.files for _, k in keys)
    fig, axes = plt.subplots(n, 2 if has_th else 1, figsize=(13 if has_th else 8, 2.4 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.2, 1]} if has_th else None)
    for row, (lab, k) in enumerate(keys):
        ax = axes[row, 0]
        if k not in tr.files:
            ax.set_title(f"{lab}: no trace")
            continue
        T = tr[k].astype(np.float32)
        t = T[:, 0]
        ax.plot(t, T[:, 1], label="mass / R²")
        ax.plot(t, T[:, 3], label="spread / R")
        ax.plot(t, T[:, 2] * 10, label="occupancy x10")
        ax.plot(t, np.nan_to_num(T[:, 4]), label="speed (R / time)", alpha=0.7)
        end = t[-1]
        horizon = _horizon(run, end)
        if end >= horizon - 1e-6:
            ax.axvspan(end / 2, end, color="#eeeeee", zorder=0)      # the judging window
        else:
            ax.axvline(end, color="#999999", ls="--", lw=1)
            ax.text(end, ax.get_ylim()[1] * 0.9, " stopped early", fontsize=7, color="#666666")
        if th is not None:
            ax.axhline(th.max_rg_r, color="#e8853a", ls=":", lw=1)
        ax.set_title(lab, fontsize=9, loc="left")
        ax.set_xlabel("time")
        if row == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper left")
        if has_th:
            a2 = axes[row, 1]
            a2.axis("off")
            if k in thumbs.files:
                im = thumbs[k]
                ts = thumbs[f"t_{k}"]
                strip = np.concatenate([np.pad(x, 1, constant_values=200) for x in im], 1)
                a2.imshow(255 - strip, cmap="magma_r", vmin=0, vmax=255)
                a2.set_title("thumbnails at t = " + ", ".join(f"{x:g}" for x in ts), fontsize=7, loc="left")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
