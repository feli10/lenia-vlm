"""ASAL's open-endedness score on long MaCE runs (11-13; Sam: "Have we tried running the ASAL CLIP-distance-over-time
metric out of the box? Presumably it would optimize for sims that keep changing in appearance over time.").

Sanity check before any search: grow the 20 named qd2 worlds to 10x the search's length, embed frames along each run
with CLIP (OpenAI ViT-B/32, ASAL's model) and score them with ASAL's open-endedness loss, exactly as published:
the mean over frames of each frame's highest similarity to any earlier frame (lower = keeps looking new). Long runs
(lenia-runs/longrun) give the expectation: Closing ring stays active to 100k steps, Budding hub settles into one
churning blob, Looping worm freezes at ~13.5k. If the score doesn't rank those three that way, searching on it is moot.

  python scripts/oecheck.py build --out RUNS/oe-check --backend auto          # Colab GPU (section 11), ~10 min
  python scripts/oecheck.py report --out RUNS/oe-check                        # anywhere, no GPU
  Offline dry run (device shell): build --out ~/oe-dry --worlds RUNS/oe-check/worlds.json --steps 600 --every 100 --clip fake

Windows: the search's (steps 1,500-2,700), the whole run, and the late half (13,500-27,000). Each world's frames share
one brightness scale (99.5th percentile from step 1,500 on), as in the long-run GIFs. Pixel activity (change between
consecutive frames / mass) and its large-scale part (8x8-cell blocks) are reported beside it, to see what CLIP adds.
From papers: the score (ASAL, Kumar et al. 2024, calc_open_endedness_score). Ours: the windows, the comparisons.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lenia import backend, render, seeds
from lenia.sim import Rule

WINDOWS = {"search": (1500, 2750), "whole": (0, None), "late": (13500, None)}


def asal_oe(z: np.ndarray) -> float:
    """ASAL's open-endedness loss: mean_t max_{t'<t} <z_t, z_t'> (strict lower triangle; row 0 is all zeros)."""
    k = np.tril(z @ z.T, k=-1)
    return float(k.max(axis=1).mean())


def coarse(x: np.ndarray, k: int = 8) -> np.ndarray:
    C, H, W = x.shape
    return x[:, :H // k * k, :W // k * k].reshape(C, H // k, k, W // k, k).mean((2, 4))


def grow(rules, s, times, steps, batch):
    shape = (s["size"], s["size"])
    A0 = seeds.smooth_noise(s["R"], shape, 3, seed=s["start_seed"], size_r=3.0)
    want = set(times)
    for k in range(0, len(rules), batch):
        rr = rules[k:k + batch]
        b = backend.make_batch(rr, shape)
        X = backend.to_device(np.stack([A0] * len(rr)))
        films = [[A0.astype(np.float32)] if 0 in want else [] for _ in rr]
        for t in range(1, steps + 1):
            X = b.step(X)
            if t in want:
                x = backend.to_numpy(X)
                for j in range(len(rr)):
                    films[j].append(np.asarray(x[j], np.float32).copy())
            if t % 2700 == 0:
                print(f"  worlds {k + 1}-{k + len(rr)}: step {t:,}/{steps:,}", file=sys.stderr, flush=True)
        for f in films:
            yield np.stack(f)


def build(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = json.loads(Path(a.worlds or out / "worlds.json").read_text())
    s, W = D["settings"], D["worlds"]
    times = list(range(0, a.steps + 1, a.every))
    backend.configure(a.backend)
    from lenia.embed import ClipScorer
    clip = ClipScorer("fake" if a.clip == "fake" else "openai/clip-vit-base-patch32")
    res, strips = [], []
    for w, film in zip(W, grow([Rule.from_dict(w["rule"]) for w in W], s, times, a.steps, a.batch)):
        tt = np.array(times)
        scale = max(float(np.percentile(film[tt >= min(1500, a.steps // 2)], 99.5)), 1e-6)
        rgb = [render.colorize(np.clip(f / scale, 0, 1)) for f in film]
        z = clip.images(rgb)
        mass = film.sum((1, 2, 3))
        act = np.array([np.abs(film[i + 1] - film[i]).sum() / max(mass[i], 1e-9) for i in range(len(film) - 1)])
        big = np.array([np.abs(coarse(film[i + 1]) - coarse(film[i])).sum() * 64 / max(mass[i], 1e-9)
                        for i in range(len(film) - 1)])
        r = {"name": w["name"], "set": w.get("set"), "every": a.every, "steps": a.steps}
        for wn, (lo, hi) in WINDOWS.items():
            m = (tt >= lo) & (tt <= (hi if hi is not None else a.steps))
            if m.sum() >= 3:
                r[f"oe_{wn}"] = asal_oe(z[m])
                mm = m[1:] & m[:-1]
                r[f"act_{wn}"] = float(act[mm].mean())
                r[f"big_{wn}"] = float(big[mm].mean())
        res.append(r)
        np.savez_compressed(out / f"emb_{len(res) - 1:02d}.npz", z=z.astype(np.float16), t=tt, act=act, big=big)
        pick = [int(np.argmin(np.abs(tt - x))) for x in np.linspace(0, a.steps, 8)]
        strips.append((w["name"], [Image.fromarray(rgb[i]).resize((120, 120)) for i in pick], [times[i] for i in pick]))
        print(f"{w['name']}: OE loss search {r.get('oe_search', float('nan')):.3f}, late {r.get('oe_late', float('nan')):.3f}",
              file=sys.stderr, flush=True)
    (out / "oe.json").write_text(json.dumps({"settings": s, "steps": a.steps, "every": a.every, "clip": clip.model,
                                             "results": res}, indent=1))
    sheet = Image.new("RGB", (8 * 122 + 150, len(strips) * 122), (20, 20, 20))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, (n, ims, ts) in enumerate(strips):
        d.text((4, i * 122 + 50), n[:22], fill=(230, 230, 230))
        for j, im in enumerate(ims):
            sheet.paste(im, (150 + j * 122, i * 122))
    sheet.save(out / "strips.png")
    report(a)


def report(a) -> None:
    out = Path(a.out)
    O = json.loads((out / "oe.json").read_text())
    R = sorted(O["results"], key=lambda r: r.get("oe_late", 9))
    L = [f"# ASAL open-endedness score on {len(R)} MaCE worlds, {O['steps']:,} steps (frame every {O['every']}; {O['clip']})\n",
         "OE loss = ASAL's score: mean over frames of the highest CLIP similarity to any earlier frame. **Lower = keeps "
         "looking new.** Activity = change between consecutive frames / mass; large-scale = the same on 8x8-cell blocks.\n",
         "| world | OE search (1.5k-2.7k) | OE whole | OE late (13.5k-27k) | activity late | large-scale late |",
         "|---|---|---|---|---|---|"]
    for r in R:
        L.append(f"| {r['name']} | {r.get('oe_search', float('nan')):.3f} | {r.get('oe_whole', float('nan')):.3f} | "
                 f"{r.get('oe_late', float('nan')):.3f} | {r.get('act_late', float('nan')):.3f} | {r.get('big_late', float('nan')):.3f} |")
    by = {r["name"]: r for r in R}
    ref = ["Closing ring", "Budding hub", "Looping worm"]
    if all(n in by and "oe_late" in by[n] for n in ref):
        v = [by[n]["oe_late"] for n in ref]
        ok = v[0] < v[1] < v[2]
        L.append(f"\n**Sanity (long runs: ring stays active, hub settles, worm freezes):** OE late "
                 + ", ".join(f"{n} {x:.3f}" for n, x in zip(ref, v)) + f": {'ranked as expected' if ok else 'NOT ranked as expected'}.")

    def rank(x):
        return np.argsort(np.argsort(x))
    for wn in ("search", "late"):
        rs = [r for r in R if f"oe_{wn}" in r]
        if len(rs) > 3:
            o = -np.array([r[f"oe_{wn}"] for r in rs])                   # novelty: higher = more open-ended
            for k, lab in (("act", "pixel activity"), ("big", "large-scale activity")):
                c = np.corrcoef(rank(o), rank(np.array([r[f"{k}_{wn}"] for r in rs])))[0, 1]
                L.append(f"- {wn} window: rank correlation of CLIP novelty with {lab}: {c:.2f}")
    s, l = [r.get("oe_search") for r in R], [r.get("oe_late") for r in R]
    if all(x is not None for x in s + l):
        c = np.corrcoef(rank(-np.array(s)), rank(-np.array(l)))[0, 1]
        L.append(f"- Does the search window predict the late window? rank correlation {c:.2f}")
    (out / "oe.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "report"):
        sub.add_parser(c).add_argument("--out", required=True)
    b = sub.choices["build"]
    b.add_argument("--worlds", default=None, help="worlds.json (default <out>/worlds.json)")
    b.add_argument("--backend", default="numpy")
    b.add_argument("--steps", type=int, default=27000)
    b.add_argument("--every", type=int, default=250, help="steps between embedded frames")
    b.add_argument("--batch", type=int, default=20)
    b.add_argument("--clip", default="openai", help="openai (ASAL's ViT-B/32) or fake (offline dry runs)")
    a = ap.parse_args()
    {"build": build, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
