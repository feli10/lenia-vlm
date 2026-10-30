"""CLIP targeted-search bench: which search settings actually steer MaCE-Lenia worlds toward a text target?
(Meeting 10-30, direction 1; project doc "Stage 1 - expedition v3 (heritability and comma selection)".)

Stage 1's expeditions barely climbed (Jev's unselected child mean 0.46 -> 0.52 over 10 steps) and there was no
objective, held-out score to tell why. CLIP gives one: free, local and absolute. Each run evolves a start world toward a
text prompt, scoring every child by the cosine similarity between CLIP's embedding of its final snapshot and the
prompt's. Nothing here calls an API.

  python scripts/clipbench.py run    --run RUNS/mace-qd5-sarch-s0 --out RUNS/clipbench --backend auto   # Colab, ~1 h
  python scripts/clipbench.py report --out RUNS/clipbench
  python scripts/clipbench.py run    --run runs/mace-qd5-sarch-s0 --out /tmp/cb --clip fake --size 48 --warmup 40 \\
      --render 40 --gap 10 --inherit-steps 40 --steps 3 --lam 4 --starts 1 --targets 2                    # offline

Configurations (CONFIGS), each run on every (target, start) pair with the same budget (--steps x --lam children):
- es-*: our expedition loop, comma selection (the (1,lambda)-ES, as expedition v3): each step the best child (grid noise
  excluded) replaces the parent. Mutation scale 0.3 (v3) or 1.0, no ring edits.
- *-regrow: every child grows from the same smoothed-noise start (the base search and all of Stage 1 so far).
- *-inherit: a child continues from its parent's final world state (--inherit-steps), so the phenotype is carried
  over: the heritability question from Stage 1 v3.
- local-1: control, best-of-N local search. Every step mutates the START again (no lineage), so any advantage of the
  es-* runs over it is accumulated progress, not just more samples.
- cma: separable CMA-ES (ASAL's optimiser; `pip install cma`) over each kernel's continuous parameters (m, log s, r, h)
  and log beta; ring structure fixed. Regrow, as in ASAL.
Starts: --starts random elites of the base run's final map (fixed seed, grid noise excluded). Targets: TARGETS[:--targets].

Reported per configuration: climb (mean CLIP score of the children at the last 3 steps minus at step 1: progress not
inflated by selection), gain (best world minus the start), and how often it beats the local control on gain.
From papers: CLIP-guided target search on Lenia and sep-CMA-ES (ASAL, Kumar et al. 2024); the (1,lambda)-ES (Rechenberg,
Schwefel). Ours: the configurations compared (world inheritance vs regrowth, the local control) on MaCE-Lenia.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mace_qd as QD
from lenia import backend, seeds
from lenia import mace as ME
from lenia.embed import ClipScorer
from lenia.mace import mutate, texture
from lenia.sim import Rule

TARGETS = ("a biological cell", "a worm", "a ring", "two cells", "a starfish", "a cluster of many small cells")
CONFIGS = {
    "es-regrow-0.3": {"opt": "es", "world": "regrow", "scale": 0.3},
    "es-inherit-0.3": {"opt": "es", "world": "inherit", "scale": 0.3},
    "es-regrow-1": {"opt": "es", "world": "regrow", "scale": 1.0},
    "es-inherit-1": {"opt": "es", "world": "inherit", "scale": 1.0},
    "local-1": {"opt": "local", "world": "regrow", "scale": 1.0},
    "cma": {"opt": "cma", "world": "regrow", "scale": 1.0},
}
SD = (0.01, 0.08, 0.03, 0.05)          # per-kernel mutation sd at scale 1: m, log s, r, h (as lenia.mace.mutate_kernel)
BOUNDS = ((0.05, 0.6), (np.log(0.003), np.log(0.2)), (0.3, 1.0), (0.05, 1.0))


def mutants(par: Rule, rng, n: int, scale: float) -> list:
    out = []
    for _ in range(n):
        m = mutate(par, rng, scale, 0.0)
        m.flow = "mace"
        m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1 * min(1.0, scale / 3.0))), 0.5, 4.0))
        out.append(m)
    return out


def encode(r: Rule) -> np.ndarray:
    d = r.to_dict()
    return np.array([v for k in d["kernels"] for v in (k["m"], np.log(k["s"]), k["r"], k["h"])] + [np.log(d["beta"])])


def decode(x: np.ndarray, template: Rule) -> Rule:
    d = template.to_dict()
    for i, k in enumerate(d["kernels"]):
        m, ls, r, h = (float(np.clip(x[4 * i + j], *BOUNDS[j])) for j in range(4))
        k.update(m=m, s=float(np.exp(ls)), r=r, h=h)
    d["beta"] = float(np.clip(np.exp(x[-1]), 0.5, 4.0))
    return Rule.from_dict(d)


def grow(rules: list, worlds: list, steps: int, a) -> list:
    """Simulate; per rule: the 2x2 picture, its final snapshot (the CLIP input), the final state, the texture."""
    shape, out = (a.size, a.size), []
    for k in range(0, len(rules), a.sim_batch):
        final, films = ME.simulate(rules[k:k + a.sim_batch], np.stack(worlds[k:k + a.sim_batch]), steps, shape, a.gap)
        for st, f in zip(final, films):
            pic = ME.picture(f)
            out.append({"pic": pic, "tile": pic[-ME.TILE:, -ME.TILE:], "state": np.asarray(st, np.float32),
                        "texture": float(texture(st))})
    return out


def setup(a) -> dict:
    st = json.loads((Path(a.run) / "state.json").read_text())
    qa = QD.settings(st)
    el = [e for m in QD.islands_of(st) for _, e in sorted(m.items()) if e.get("texture", 0) <= qa.max_texture]
    rng = np.random.default_rng(a.seed)
    starts = [el[int(j)] for j in rng.choice(len(el), a.starts, replace=False)]
    A0 = seeds.smooth_noise(qa.R, (a.size, a.size), 3, seed=qa.start_seed, size_r=3.0)
    ev = grow([Rule.from_dict(e["rule"]) for e in starts], [A0] * len(starts), a.warmup + a.render, a)
    clip = ClipScorer(a.clip)
    for t in TARGETS[:a.targets]:                # sanity check: real CLIP cosines for these pictures sit around 0.15-0.30
        print(f"start scores for '{t}': " + ", ".join(f"{x:.3f}" for x in clip.score([v["tile"] for v in ev], t)), file=sys.stderr)
    runs = []
    for name in a.configs.split(","):
        c = CONFIGS[name]
        for t in TARGETS[:a.targets]:
            for si, (e, v) in enumerate(zip(starts, ev)):
                s0 = float(clip.score([v["tile"]], t)[0])
                r = {"config": name, **c, "target": t, "start": si, "start_image": e["image"], "start_rule": e["rule"],
                     "parent": e["rule"], "state": v["state"], "score0": s0, "parent_score": s0, "steps": [],
                     "best": {"score": s0, "step": 0, "pic": v["pic"]}, "start_pic": v["pic"]}
                if c["opt"] == "cma":
                    import cma                                   # pip install cma
                    x0 = encode(Rule.from_dict(e["rule"]))
                    stds = [SD[j % 4] * c["scale"] for j in range(len(x0) - 1)] + [0.1 * c["scale"]]
                    r["es"] = cma.CMAEvolutionStrategy(x0, 1.0, {"popsize": a.lam, "CMA_diagonal": True, "CMA_stds": stds,
                                                                 "seed": a.seed + 1 + len(runs), "verbose": -9})
                runs.append(r)
    return {"step": 0, "A0": A0, "runs": runs, "args": {k: v for k, v in vars(a).items() if k != "cmd"}}


def run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    ck = out / "checkpoint.pkl"
    S = pickle.loads(ck.read_bytes()) if ck.exists() else setup(a)
    if ck.exists():
        print(f"resuming after step {S['step']}", file=sys.stderr)
    clip = ClipScorer(a.clip)
    runs = S["runs"]
    while S["step"] < a.steps:
        step, t0 = S["step"] + 1, time.time()
        kids = []                                                # (run index, rule, initial world, x for cma)
        for ri, r in enumerate(runs):
            rng = np.random.default_rng([a.seed, ri, step])
            if r["opt"] == "cma":
                X = r["es"].ask()
                rr = [decode(np.asarray(x), Rule.from_dict(r["start_rule"])) for x in X]
            else:
                base = Rule.from_dict(r["start_rule"] if r["opt"] == "local" else r["parent"])
                rr, X = mutants(base, rng, a.lam, r["scale"]), [None] * a.lam
            for m, x in zip(rr, X):
                kids.append((ri, m, r["state"] if r["world"] == "inherit" else S["A0"], x))
        ev = [None] * len(kids)
        for mode, steps in (("regrow", a.warmup + a.render), ("inherit", a.inherit_steps)):
            idx = [k for k, (ri, *_) in enumerate(kids) if runs[ri]["world"] == mode]
            if idx:
                for k, e in zip(idx, grow([kids[k][1] for k in idx], [kids[k][2] for k in idx], steps, a)):
                    ev[k] = e
        emb = clip.images([e["tile"] for e in ev])
        for ri, r in enumerate(runs):
            ks = [k for k, kid in enumerate(kids) if kid[0] == ri]
            sc = emb[ks] @ clip.text(r["target"])
            ok = np.array([ev[k]["texture"] <= a.max_texture for k in ks])
            valid = np.where(ok, sc, -np.inf)
            j = int(np.argmax(valid)) if ok.any() else None
            if r["opt"] == "cma":
                r["es"].tell([kids[k][3] for k in ks], list(np.where(ok, -sc, 1.0)))
            if j is not None and r["opt"] == "es":
                k = ks[j]
                r["parent"], r["state"], r["parent_score"] = kids[k][1].to_dict(), ev[k]["state"], float(sc[j])
            if j is not None and float(sc[j]) > r["best"]["score"]:
                r["best"] = {"score": float(sc[j]), "step": step, "pic": ev[ks[j]]["pic"], "rule": kids[ks[j]][1].to_dict()}
            r["steps"].append({"step": step, "mean": float(sc[ok].mean()) if ok.any() else None,
                               "max": float(sc[ok].max()) if ok.any() else None, "valid": int(ok.sum()),
                               "parent": r["parent_score"]})
        S["step"] = step
        tmp = out / "checkpoint.tmp"
        tmp.write_bytes(pickle.dumps(S))
        tmp.replace(ck)
        print(f"step {step}/{a.steps}: {len(kids)} worlds in {time.time() - t0:.0f}s; " + ", ".join(
            f"{c} {np.mean([r['steps'][-1]['mean'] or np.nan for r in runs if r['config'] == c]):.3f}"
            for c in dict.fromkeys(r["config"] for r in runs)), file=sys.stderr)
    save(out, S)
    report(a)


def save(out: Path, S: dict) -> None:
    (out / "img").mkdir(exist_ok=True)
    rows = []
    for k, r in enumerate(S["runs"]):
        Image.fromarray(r["best"]["pic"]).save(out / "img" / f"best_{k:03d}.png")
        Image.fromarray(r["start_pic"]).save(out / "img" / f"start_{r['start']}.png")
        rows.append({x: r[x] for x in ("config", "opt", "world", "scale", "target", "start", "start_image", "score0")} |
                    {"steps": r["steps"], "best_score": r["best"]["score"], "best_step": r["best"]["step"],
                     "best_rule": r["best"].get("rule"), "best_image": f"img/best_{k:03d}.png"})
    (out / "results.json").write_text(json.dumps({"args": S["args"], "runs": rows}, indent=1))


def report(a) -> None:
    out = Path(a.out)
    R = json.loads((out / "results.json").read_text())["runs"]
    cfgs = list(dict.fromkeys(r["config"] for r in R))
    mean = lambda xs: float(np.nanmean([x for x in xs if x is not None])) if any(x is not None for x in xs) else float("nan")
    climb = lambda r: mean([s["mean"] for s in r["steps"][-3:]]) - mean([r["steps"][0]["mean"]])
    gain = lambda r: r["best_score"] - r["score0"]
    local = {(r["target"], r["start"]): gain(r) for r in R if r["config"] == "local-1"}
    L = ["# CLIP targeted-search bench\n",
         f"{len(R)} runs: {len(cfgs)} configurations x {len({r['target'] for r in R})} targets x {len({r['start'] for r in R})} "
         f"starts; {len(R[0]['steps'])} steps x {R[0]['steps'][0]['valid'] if R[0]['steps'] else '?'}+ children each. Scores are "
         "CLIP cosine similarities (final snapshot vs the prompt); differences of ~0.01 are visible, ~0.03 large.\n",
         "| configuration | climb (children, last 3 steps - step 1) | gain (best - start) | beats local control on gain |",
         "|---|---|---|---|"]
    for c in cfgs:
        rr = [r for r in R if r["config"] == c]
        beats = [gain(r) > local[(r["target"], r["start"])] for r in rr if (r["target"], r["start"]) in local]
        L.append(f"| {c} | {mean([climb(r) for r in rr]):+.4f} | {mean([gain(r) for r in rr]):+.4f} | "
                 f"{'-' if c == 'local-1' or not beats else f'{sum(beats)}/{len(beats)}'} |")
    L += ["", "Gain per target (mean over starts):", "", "| target | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    for t in dict.fromkeys(r["target"] for r in R):
        L.append(f"| {t} | " + " | ".join(f"{mean([gain(r) for r in R if r['config'] == c and r['target'] == t]):+.4f}"
                                          for c in cfgs) + " |")
    L += ["", "Children's mean score per step (all targets and starts):", ""]
    for c in cfgs:
        rr = [r for r in R if r["config"] == c]
        n = max(len(r["steps"]) for r in rr)
        L.append(f"- {c}: " + ", ".join(f"{mean([r['steps'][i]['mean'] for r in rr if len(r['steps']) > i]):.3f}" for i in range(n)))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sheet(out, R, cfgs)


def sheet(out: Path, R: list, cfgs: list, px: int = 150) -> None:
    """One row per (target, start): the start's final snapshot, then each configuration's best world."""
    keys = list(dict.fromkeys((r["target"], r["start"]) for r in R))
    head = 34
    im = Image.new("RGB", ((len(cfgs) + 1) * (px + 6) + 6, len(keys) * (px + head) + 30), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for c, name in enumerate(["start"] + cfgs):
        d.text((6 + c * (px + 6), 8), name, fill="black")
    for k, (t, s) in enumerate(keys):
        y = 30 + k * (px + head)
        rr = {r["config"]: r for r in R if r["target"] == t and r["start"] == s}
        r0 = next(iter(rr.values()))
        d.text((6, y), f"'{t}', start {s}", fill="black")
        tile = lambda p: Image.open(out / p).convert("RGB").crop((ME.TILE + 4, ME.TILE + 4, 2 * ME.TILE + 4, 2 * ME.TILE + 4)).resize((px, px))
        im.paste(tile(f"img/start_{s}.png"), (6, y + 16))
        d.text((6, y + 16 + px - 12), f"{r0['score0']:.3f}", fill="white")
        for c, name in enumerate(cfgs, 1):
            if name in rr:
                im.paste(tile(rr[name]["best_image"]), (6 + c * (px + 6), y + 16))
                d.text((6 + c * (px + 6), y + 16 + px - 12), f"{rr[name]['best_score']:.3f}", fill="white")
    im.save(out / "sheet.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True, help="the base run (start worlds)")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", default="numpy")
    r.add_argument("--clip", default="openai/clip-vit-base-patch32", help="a transformers CLIP model, or fake")
    r.add_argument("--configs", default=",".join(CONFIGS))
    r.add_argument("--targets", type=int, default=4, help="the first k of TARGETS")
    r.add_argument("--starts", type=int, default=2)
    r.add_argument("--steps", type=int, default=12)
    r.add_argument("--lam", type=int, default=16, help="children per step")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--size", type=int, default=160)
    r.add_argument("--warmup", type=int, default=1500)
    r.add_argument("--render", type=int, default=1200)
    r.add_argument("--gap", type=int, default=400)
    r.add_argument("--inherit-steps", type=int, default=1500, help="steps a child runs from its parent's final state")
    r.add_argument("--max-texture", type=float, default=0.3)
    r.add_argument("--sim-batch", type=int, default=64)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
