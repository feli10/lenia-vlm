"""Jev-guided evolution of mass-conserving (MaCE) Lenia worlds: the first, deliberately simple experiment.

  uv run python scripts/mace_evo.py run   --out runs/mace-evo-jev            # judge calls: run on the Mac
  uv run python scripts/mace_evo.py run   --out runs/mace-evo-dry --judge fake --generations 3   # offline dry run
  uv run python scripts/mace_evo.py sheet --out runs/mace-evo-jev            # lineages x checkpoints

Setup
- Founders: the 12 random 3-channel rules of the MaCE sweep (sampler seed 3, 9 kernels each), R = 13,
  beta = 2, each grown from smoothed noise for `--warmup` steps in a 160x160 world. The unit is the whole
  world (several bodies allowed), not one creature.
- Each generation, every lineage makes `--mutants` children: the parent's rule mutated (explore.mutate:
  m, s, r, h, rings of about half the kernels) plus beta x exp(N(0, 0.1)) clipped to [0.5, 4], started from
  the parent's current world and run for `--steps` steps.
- The only constraint is the grid-noise rule: a child whose final world has more than `--max-texture` of
  its spatial power at wavelengths under 4 cells is discarded (grid noise is not valid continuum
  structure). --max-texture 0 switches it off (the no-filter arm).
- The first child that passes duels its parent: Jev, both orders, P(child) = mean of the two. It replaces
  the parent if P(child) >= `--threshold`. Mass is conserved, so nothing can fizzle or explode.
Everything is saved after each generation; re-running the same command resumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lenia import backend, render, seeds
from lenia.config import REPO_ROOT, load
from lenia.explore import mutate, texture
from lenia.judge import Judge, Query, load_env
from lenia.prompts import PROMPTS
from lenia.sample import sample
from lenia.sim import Rule, with_flow

PROMPT_ID = "evolve_complexity_world"
PROMPTS.setdefault(PROMPT_ID, {
    "instructions": ("You are shown two worlds from a mass-conserving variant of Lenia, a continuous cellular automaton "
                     "used in artificial-life research. Each image shows one whole world as four snapshots taken a few "
                     "moments apart, in reading order (top-left, top-right, bottom-left, bottom-right); the colours are "
                     "three interacting chemical channels. The two images are {A} and {B}. We are running an "
                     "evolutionary process whose goal is to reach more complex, life-like structure over many "
                     "generations. Which world would you choose as the parent for the next generation?"),
    "first": "I would choose the world in {A} as the parent.",
    "second": "I would choose the world in {B} as the parent.",
})

FRAMES, FRAME_GAP, TILE = 4, 50, 320         # 4 snapshots 50 steps apart, each drawn 320 px (image 640x640)


# ---------- simulation and pictures ----------

def simulate(rules: list, worlds: np.ndarray, steps: int, shape) -> tuple[np.ndarray, list]:
    """Run one batch; return final worlds and, per world, the last FRAMES snapshots FRAME_GAP steps apart."""
    b = backend.make_batch(rules, shape)
    A = backend.to_device(worlds)
    keep = {steps - i * FRAME_GAP for i in range(FRAMES)}
    film = []
    for t in range(1, steps + 1):
        A = b.step(A)
        if t in keep:
            film.append(backend.to_numpy(A).copy())
    film = np.stack(film, 1)                                  # (B, FRAMES, C, H, W)
    return film[:, -1], [f for f in film]


def picture(frames: np.ndarray) -> np.ndarray:
    """(FRAMES, C, H, W) -> 2x2 composite (uint8); one brightness scale shared by the four snapshots."""
    scale = max(float(np.percentile(frames, 99.5)), 1e-6)
    tiles = [np.asarray(Image.fromarray(render.colorize(np.clip(f / scale, 0, 1))).resize((TILE, TILE), Image.BILINEAR))
             for f in frames]
    gap = np.full((TILE, 4, 3), 40, np.uint8)
    rows = [np.concatenate([tiles[0], gap, tiles[1]], 1), np.concatenate([tiles[2], gap, tiles[3]], 1)]
    return np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), 40, np.uint8), rows[1]], 0)


# ---------- state ----------

def save(out: Path, st: dict, worlds: np.ndarray) -> None:
    np.save(out / "worlds.tmp.npy", worlds.astype(np.float32))
    (out / "worlds.tmp.npy").replace(out / "worlds.npy")
    (out / "state.tmp.json").write_text(json.dumps(st))
    (out / "state.tmp.json").replace(out / "state.json")


def founders(args, shape):
    rules = sample("multichannel", args.lineages, seed=args.rule_seed, R=args.R, channels=3, self_k=1, cross_k=1)
    rules = [with_flow(r, "mace", args.beta) for r in rules]
    A0 = np.stack([seeds.smooth_noise(args.R, shape, 3, seed=i, size_r=3.0) for i in range(len(rules))])
    return rules, A0


def run(args) -> None:
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    shape = (args.size, args.size)
    cfg = load(args.config)
    load_env(REPO_ROOT / ".env")
    jcfg = {**cfg["judges"]["defaults"], **cfg["judges"][args.judge]}
    judge = Judge(args.judge, jcfg, out / "judge_cache.sqlite", yes=True)
    rng = np.random.default_rng(args.seed)

    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        worlds = np.load(out / "worlds.npy")
        rng = np.random.default_rng([args.seed, st["gen"]])
        print(f"resuming at generation {st['gen']}", file=sys.stderr)
    else:
        rules, A0 = founders(args, shape)
        t0 = time.time()
        worlds, films = simulate(rules, A0, args.warmup, shape)
        print(f"founders grown in {time.time() - t0:.0f}s", file=sys.stderr)
        st = {"gen": 0, "args": vars(args), "prompt": PROMPTS[PROMPT_ID], "lineages": []}
        for i, (r, f) in enumerate(zip(rules, films)):
            name = f"L{i:02d}_g000.png"
            Image.fromarray(picture(f)).save(out / "img" / name)
            st["lineages"].append({"rule": r.to_dict(), "image": name, "texture": round(texture(f[-1]), 4),
                                   "history": [{"gen": 0, "image": name, "texture": round(texture(f[-1]), 4)}]})
        save(out, st, worlds)

    log_path = out / "duels.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["gen", "lineage", "mutants", "passed_texture", "child_texture", "p_child_first",
                        "p_parent_first", "p_child", "accepted", "beta"])
        while st["gen"] < args.generations:
            g = st["gen"] + 1
            t0 = time.time()
            L = st["lineages"]
            kids, starts, owner = [], [], []
            for i, lin in enumerate(L):
                parent = Rule.from_dict(lin["rule"])
                for _ in range(args.mutants):
                    k = mutate(parent, rng, args.mut_scale)
                    k.flow = "mace"
                    k.beta = float(np.clip(parent.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                    kids.append(k); starts.append(worlds[i]); owner.append(i)
            finals, films = simulate(kids, np.stack(starts), args.steps, shape)
            tex = [texture(f) for f in finals]
            # first texture-passing child per lineage
            pick, n_pass = {}, {}
            for j, i in enumerate(owner):
                ok = args.max_texture <= 0 or tex[j] <= args.max_texture
                n_pass[i] = n_pass.get(i, 0) + ok
                if ok and i not in pick:
                    pick[i] = j
            pics = {j: picture(films[j]) for j in pick.values()}
            parent_pics = {i: np.asarray(Image.open(out / "img" / L[i]["image"]).convert("RGB")) for i in pick}
            qs = []
            for i, j in pick.items():
                qs += [Query(PROMPT_ID, pics[j], parent_pics[i]), Query(PROMPT_ID, parent_pics[i], pics[j])]
            vs = judge.judge_many(qs, desc=f"gen {g}")
            for n, (i, j) in enumerate(pick.items()):
                a, b = vs[2 * n], vs[2 * n + 1]
                if not (a.ok and b.ok):
                    continue
                p = 0.5 * (a.p_first + (1 - b.p_first))
                acc = p >= args.threshold
                w.writerow([g, i, args.mutants, n_pass[i], round(tex[j], 4), round(a.p_first, 4), round(b.p_first, 4),
                            round(p, 4), int(acc), round(kids[j].beta, 3)])
                if acc:
                    name = f"L{i:02d}_g{g:03d}.png"
                    Image.fromarray(pics[j]).save(out / "img" / name)
                    L[i].update(rule=kids[j].to_dict(), image=name, texture=round(tex[j], 4))
                    L[i]["history"].append({"gen": g, "image": name, "texture": round(tex[j], 4), "p": round(p, 4)})
                    worlds[i] = finals[j]
            fh.flush()
            st["gen"] = g
            save(out, st, worlds)
            acc_n = sum(1 for lin in L if lin["history"][-1]["gen"] == g)
            print(f"gen {g}: {len(pick)}/{len(L)} lineages had a texture-passing child, {acc_n} accepted, "
                  f"{time.time() - t0:.0f}s, judge spent ${judge.spent:.4f}", file=sys.stderr)


def sheet(args) -> None:
    out = Path(args.out)
    st = json.loads((out / "state.json").read_text())
    cps = sorted({0, *[int(c) for c in args.checkpoints.split(",")], st["gen"]})
    cps = [c for c in cps if c <= st["gen"]]
    items = []
    for i, lin in enumerate(st["lineages"]):
        for c in cps:
            h = [e for e in lin["history"] if e["gen"] <= c][-1]
            im = Image.open(out / "img" / h["image"]).convert("RGB").resize((240, 240), Image.BILINEAR)
            items.append((f"L{i:02d} gen {c} (from g{h['gen']}, tex {h['texture']:.2f})", im))
    path = out / "lineages.png"
    render.sheet(items, cols=len(cps), title=f"{out.name}: lineages x generation").save(path)
    print(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="runs/mace-evo-jev")
    r.add_argument("--judge", default="jev", help="jev | luna | fake (names in configs/lenia.yaml judges)")
    r.add_argument("--config", default=None)
    r.add_argument("--backend", default="numpy", help="numpy | auto | mps | cuda")
    r.add_argument("--generations", type=int, default=40)
    r.add_argument("--lineages", type=int, default=12)
    r.add_argument("--mutants", type=int, default=2)
    r.add_argument("--steps", type=int, default=600, help="steps a child runs from its parent's world")
    r.add_argument("--warmup", type=int, default=1500, help="steps founders grow from noise")
    r.add_argument("--size", type=int, default=160)
    r.add_argument("--R", type=float, default=13)
    r.add_argument("--beta", type=float, default=2.0)
    r.add_argument("--max-texture", type=float, default=0.3, help="grid-noise rule; 0 = off")
    r.add_argument("--threshold", type=float, default=0.6)
    r.add_argument("--mut-scale", type=float, default=1.0)
    r.add_argument("--rule-seed", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sheet")
    s.add_argument("--out", default="runs/mace-evo-jev")
    s.add_argument("--checkpoints", default="10,20,30")
    a = ap.parse_args()
    if a.cmd == "run":
        backend.configure(a.backend)
        run(a)
    else:
        sheet(a)


if __name__ == "__main__":
    main()
