"""Jev-guided evolution of mass-conserving (MaCE) Lenia worlds: the first, deliberately simple experiment.

  uv run python scripts/mace_evo.py run     [--out runs/mace-evo-jev-v3]   # Jev-guided run (judge calls: on the Mac)
  uv run python scripts/mace_evo.py control [--out ...]                    # time-only control (simulation only)
  uv run python scripts/mace_evo.py check   [--out ...]                    # founder / final / control, Luna + Jev
  uv run python scripts/mace_evo.py rate    [--out ...]                    # your ratings of the check pairs
  uv run python scripts/mace_evo.py sheet   [--out ...]                    # lineages x checkpoints
  uv run python scripts/mace_evo.py run --out runs/mace-evo-dry --judge fake --generations 3   # offline dry run

Setup
- Founders: the 12 random 3-channel rules of the MaCE sweep (sampler seed 3, 9 kernels each), R = 13,
  beta = 2, each grown from smoothed noise for `--warmup` steps in a 160x160 world. The unit is the whole
  world (several bodies allowed), not one creature.
- Each generation, every lineage runs its own rule on from its current world for `--steps` steps (the
  null continuation) next to `--mutants` children: the rule mutated (explore.mutate at `--mut-scale`:
  m, s, r, h, rings of about half the kernels) plus beta x exp(N(0, 0.1)) clipped to [0.5, 4], from the same
  world for the same steps. A duel therefore differs only by the mutation, not by age.
- The only constraint is the grid-noise rule: a child whose final world has more than `--max-texture` of
  its spatial power at wavelengths under 4 cells is discarded (grid noise is not valid continuum
  structure). --max-texture 0 switches it off (the no-filter arm).
- Every passing child duels the null continuation: Jev, both orders, P(child) = mean of the two. The best
  child with P >= `--threshold` becomes the lineage; otherwise the null continuation does (so every
  lineage ages at the same rate). For the first `--aging-check` generations the null is also judged
  against the lineage's previous picture, which measures whether the judge simply prefers younger worlds.
  Mass is conserved, so nothing can fizzle or explode.
v3 (default): children run 1200 steps and the judge sees snapshots 400 steps apart, long enough to
show movement (v2: 600 steps, 50 apart, so it judged near-stills).
v1 (runs/mace-evo-jev) dueled one child against the parent's stored, 600-steps-younger picture and accepted 4%.
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
                     "time apart, long enough to show movement and change, in reading order (top-left, top-right, "
                     "bottom-left, bottom-right); the colours are "
                     "three interacting chemical channels. The two images are {A} and {B}. We are running an "
                     "evolutionary process whose goal is to reach more complex, life-like structure over many "
                     "generations. Which world would you choose as the parent for the next generation?"),
    "first": "I would choose the world in {A} as the parent.",
    "second": "I would choose the world in {B} as the parent.",
})

FRAMES, TILE = 4, 320                        # 4 snapshots, each drawn 320 px (image 640x640)


# ---------- simulation and pictures ----------

def simulate(rules: list, worlds: np.ndarray, steps: int, shape, gap: int = 400) -> tuple[np.ndarray, list]:
    """Run one batch; return final worlds and, per world, the last FRAMES snapshots `gap` steps apart."""
    gap = min(gap, steps // (FRAMES - 1))
    b = backend.make_batch(rules, shape)
    A = backend.to_device(worlds)
    keep = {steps - i * gap for i in range(FRAMES)}
    film = [np.asarray(worlds, np.float32).copy()] if 0 in keep else []
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
        worlds, films = simulate(rules, A0, args.warmup, shape, args.frame_gap)
        np.save(out / "worlds_g000.npy", worlds.astype(np.float32))
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
            w.writerow(["gen", "lineage", "kind", "mutant", "texture", "p_a_first", "p_b_first", "p_a", "accepted", "beta"])
        while st["gen"] < args.generations:
            g = st["gen"] + 1
            t0 = time.time()
            L = st["lineages"]
            M = args.mutants
            # per lineage: slot 0 = the parent's own rule run on (the null continuation), slots 1..M = mutants,
            # all from the same world for the same number of steps, so a duel differs only by the mutation
            rules, starts = [], []
            for i, lin in enumerate(L):
                parent = Rule.from_dict(lin["rule"])
                rules.append(parent); starts.append(worlds[i])
                for _ in range(M):
                    k = mutate(parent, rng, args.mut_scale)
                    k.flow = "mace"
                    k.beta = float(np.clip(parent.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                    rules.append(k); starts.append(worlds[i])
            finals, films = simulate(rules, np.stack(starts), args.steps, shape, args.frame_gap)
            tex = [texture(f) for f in finals]
            slot = lambda i, m: i * (M + 1) + m
            pics = {}
            qs, meta = [], []
            for i in range(len(L)):
                n0 = slot(i, 0)
                pics[n0] = picture(films[n0])
                for m in range(1, M + 1):
                    j = slot(i, m)
                    if args.max_texture > 0 and tex[j] > args.max_texture:
                        continue
                    pics[j] = picture(films[j])
                    qs += [Query(PROMPT_ID, pics[j], pics[n0]), Query(PROMPT_ID, pics[n0], pics[j])]
                    meta.append((i, "mutant", m, j))
                if g <= args.aging_check:          # does Jev prefer the younger picture of the same rule?
                    old = np.asarray(Image.open(out / "img" / L[i]["image"]).convert("RGB"))
                    qs += [Query(PROMPT_ID, pics[n0], old), Query(PROMPT_ID, old, pics[n0])]
                    meta.append((i, "aging", 0, n0))
            vs = judge.judge_many(qs, desc=f"gen {g}")
            best = {}
            for n, (i, kind, m, j) in enumerate(meta):
                a, b = vs[2 * n], vs[2 * n + 1]
                if not (a.ok and b.ok):
                    continue
                p = 0.5 * (a.p_first + (1 - b.p_first))
                acc = kind == "mutant" and p >= args.threshold
                if acc and p > best.get(i, (0, None))[0]:
                    best[i] = (p, j)
                w.writerow([g, i, kind, m, round(tex[j], 4), round(a.p_first, 4), round(b.p_first, 4), round(p, 4),
                            int(acc), round(rules[j].beta, 3)])
            for i, lin in enumerate(L):
                p, j = best.get(i, (None, slot(i, 0)))
                kind = "null" if p is None else "mutant"
                name = f"L{i:02d}_g{g:03d}.png"
                Image.fromarray(pics[j]).save(out / "img" / name)
                lin.update(rule=rules[j].to_dict(), image=name, texture=round(tex[j], 4))
                lin["history"].append({"gen": g, "image": name, "texture": round(tex[j], 4), "kind": kind,
                                       **({"p": round(p, 4), "rule": rules[j].to_dict()} if p is not None else {})})
                worlds[i] = finals[j]
            fh.flush()
            st["gen"] = g
            save(out, st, worlds)
            if g % 10 == 0:
                np.save(out / f"worlds_g{g:03d}.npy", worlds.astype(np.float32))
            print(f"gen {g}: {len(best)}/{len(L)} lineages took a mutant, {len(qs)} judge calls, "
                  f"{time.time() - t0:.0f}s, judge spent ${judge.spent:.4f}", file=sys.stderr)


# ---------- time-only control, held-out check, human rating ----------

def run_args(out: Path) -> argparse.Namespace:
    st = json.loads((out / "state.json").read_text())
    a = argparse.Namespace(**{"frame_gap": 50, **st["args"]})        # v2 runs predate --frame-gap (50 steps)
    return a


def founder_worlds(out: Path, a, shape) -> tuple[list, np.ndarray]:
    """The founders' rules and worlds: saved by runs from v3 on, otherwise regrown the same way."""
    rules, A0 = founders(a, shape)
    if (out / "worlds_g000.npy").exists():
        return rules, np.load(out / "worlds_g000.npy")
    print("regrowing founders (this run did not save them)", file=sys.stderr)
    return rules, simulate(rules, A0, a.warmup, shape, a.frame_gap)[0]


def control(args) -> None:
    """Each founder's own rule, never mutated, run for as many steps as the evolution took (gens x steps)."""
    out = Path(args.out); a = run_args(out); st = json.loads((out / "state.json").read_text())
    shape = (a.size, a.size)
    cdir = out / "control"; cdir.mkdir(exist_ok=True)
    rules, W = founder_worlds(out, a, shape)
    total, done, t0 = st["gen"] * a.steps, 0, time.time()
    chunk = 10 * a.steps                                   # save at the same checkpoints as the run (every 10 gens)
    while done < total:
        n = min(chunk, total - done)
        W, _ = simulate(rules, W, n, shape, a.frame_gap)
        done += n
        np.save(cdir / f"worlds_g{done // a.steps:03d}.npy", W.astype(np.float32))
        print(f"control: {done}/{total} steps ({time.time() - t0:.0f}s)", file=sys.stderr)


def rules_at(out: Path, gens: list) -> dict:
    """Each lineage's rule at the given generations. Runs from this version on store every accepted rule in
    their history; older runs only kept the current rule, so the run is replayed: the founders are rebuilt,
    every generation's mutants are redrawn from the run's random stream (mutation is the only thing that
    draws from it) and the winning mutant is taken from duels.csv. The replay is checked mutant by mutant
    against the betas logged in duels.csv and must end on the saved current rules."""
    st = json.loads((out / "state.json").read_text()); a = run_args(out)
    gens = sorted(set(gens))
    if all("rule" in h for lin in st["lineages"] for h in lin["history"] if h.get("kind") == "mutant") and \
            any("rule" in h for lin in st["lineages"] for h in lin["history"]):
        found, _ = founders(a, (a.size, a.size))
        res = {}
        for g in gens:
            res[g] = []
            for f, lin in zip(found, st["lineages"]):
                hs = [h for h in lin["history"] if h["gen"] <= g and "rule" in h]
                res[g].append(Rule.from_dict(hs[-1]["rule"]) if hs else f)
        return res
    rows = list(csv.DictReader(open(out / "duels.csv")))
    win, beta_log = {}, {}
    for r in rows:
        if r["kind"] != "mutant":
            continue
        key = (int(r["gen"]), int(r["lineage"]))
        beta_log[key + (int(r["mutant"]),)] = float(r["beta"])
        if r["accepted"] == "1" and float(r["p_a"]) > win.get(key, (0.0, 0))[0]:
            win[key] = (float(r["p_a"]), int(r["mutant"]))
    rules, _ = founders(a, (a.size, a.size))
    rng = np.random.default_rng(a.seed)
    res = {0: list(rules)} if 0 in gens else {}
    for g in range(1, st["gen"] + 1):
        new = []
        for i, par in enumerate(rules):
            kids = []
            for m in range(1, a.mutants + 1):
                k = mutate(par, rng, a.mut_scale)
                k.flow = "mace"
                k.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                logged = beta_log.get((g, i, m))
                if logged is not None and abs(round(k.beta, 3) - logged) > 1.5e-3:
                    sys.exit(f"replay diverged at generation {g}, lineage {i}, mutant {m} (was the run resumed? "
                             f"a resume reseeds the random stream)")
                kids.append(k)
            new.append(Rule.from_dict(json.loads((kids[win[(g, i)][1] - 1] if (g, i) in win else par).to_json())))
        rules = new
        if g in gens:
            res[g] = list(rules)
    flat = lambda d: np.array([v for k in d["kernels"] for v in (k["m"], k["s"], k["r"], k["h"], *k["b"])] + [d["beta"]])
    for r, lin in zip(rules, st["lineages"]):
        x, y = flat(json.loads(r.to_json())), flat(lin["rule"])
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-9, atol=1e-12):
            sys.exit("replay does not end on the saved rules; not using it")
    print(f"replayed {st['gen']} generations; matches the saved rules", file=sys.stderr)
    return res


def check(args) -> None:
    """Founder, a mid-run checkpoint, the evolved final and the time-only control (at the end and at the
    checkpoint) of every lineage, all re-rendered the same way (snapshots `--frame-gap` apart over
    `--render-steps`), judged by a held-out judge and the driver, both orders."""
    out = Path(args.out); a = run_args(out); st = json.loads((out / "state.json").read_text())
    shape = (a.size, a.size)
    cdir = out / "check"; (cdir / "images").mkdir(parents=True, exist_ok=True)
    G = st["gen"]
    mid = args.mid if args.mid is not None else (G // 2) // 10 * 10
    cpath = out / "control" / f"worlds_g{G:03d}.npy"
    if not cpath.exists():
        sys.exit(f"no {cpath}: run `control` first")
    f_rules, f_W = founder_worlds(out, a, shape)
    e_rules = [Rule.from_dict(l["rule"]) for l in st["lineages"]]
    worlds = {"founder": (f_rules, f_W), "final": (e_rules, np.load(out / "worlds.npy")), "control": (f_rules, np.load(cpath))}
    groups = [("final beats founder", "final", "founder"), ("final beats time-only control", "final", "control"),
              ("control beats founder (time alone)", "control", "founder")]
    mid_ok = 0 < mid < G and (out / f"worlds_g{mid:03d}.npy").exists() and (out / "control" / f"worlds_g{mid:03d}.npy").exists()
    if mid_ok:
        m_rules = rules_at(out, [mid])[mid]
        worlds[f"g{mid:03d}"] = (m_rules, np.load(out / f"worlds_g{mid:03d}.npy"))
        worlds[f"control_g{mid:03d}"] = (f_rules, np.load(out / "control" / f"worlds_g{mid:03d}.npy"))
        groups += [(f"gen {mid} beats founder", f"g{mid:03d}", "founder"),
                   (f"gen {mid} beats control at gen {mid}", f"g{mid:03d}", f"control_g{mid:03d}"),
                   (f"final (gen {G}) beats gen {mid}", "final", f"g{mid:03d}")]
    elif args.mid is not None:
        sys.exit(f"no saved worlds for generation {mid} (runs save every 10th generation; control must have reached it)")
    pics = {}
    for kind, (rules, W) in worlds.items():
        _, films = simulate(rules, W, args.render_steps, shape, args.frame_gap)
        for i, f in enumerate(films):
            name = f"L{i:02d}_{kind}"
            pics[name] = picture(f)
            Image.fromarray(pics[name]).save(cdir / "images" / f"{name}.png")
    n = len(e_rules)
    pairs = [[f"L{i:02d}_{x}", f"L{i:02d}_{y}"] for _, x, y in groups for i in range(n)]
    (cdir / "pairs.json").write_text(json.dumps(pairs))
    cfg = load(args.config); load_env(REPO_ROOT / ".env")
    res = {}
    for jn in args.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, cdir / "judge_cache.sqlite", yes=True)
        qs = [q for x, y in pairs for q in (Query(PROMPT_ID, pics[x], pics[y]), Query(PROMPT_ID, pics[y], pics[x]))]
        vs = judge.judge_many(qs, desc=f"check {jn}")
        res[jn] = [None if not (vs[2 * k].ok and vs[2 * k + 1].ok) else 0.5 * (vs[2 * k].p_first + 1 - vs[2 * k + 1].p_first)
                   for k in range(len(pairs))]
    json.dump({"pairs": pairs, "p_first_of_pair": res, "mid": mid if mid_ok else None}, open(cdir / "check.json", "w"), indent=1)
    lines = [f"# {out.name}: held-out check\n",
             f"Each pair is judged in both orders; P = probability the first-named world is chosen as the parent.\n",
             "| comparison | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for lab, x0, y0 in groups:
        row = []
        for jn, ps in res.items():
            sel = [p for (x, y), p in zip(pairs, ps) if p is not None and x[4:] == x0 and y[4:] == y0]
            row.append(f"{np.mean(sel):.2f} (wins {sum(p > 0.5 for p in sel)}/{len(sel)})" if sel else "-")
        lines.append(f"| {lab} | " + " | ".join(row) + " |")
    lines += ["", "Per pair:", "", "| pair | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for k, (x, y) in enumerate(pairs):
        lines.append(f"| {x} vs {y} | " + " | ".join("-" if res[j][k] is None else f"{res[j][k]:.2f}" for j in res) + " |")
    (cdir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(groups)]))
    print(cdir / "report.md")


def rate(args) -> None:
    """Your own ratings of the check pairs, in the browser (same page as `lenia vlmtest rate`)."""
    from lenia import vlmtest
    cdir = Path(args.out) / "check"
    pairs = json.loads((cdir / "pairs.json").read_text()); n = len(pairs)
    hp_path = cdir / "human_pairs.json"
    if hp_path.exists():                       # check grew (e.g. the mid-run checkpoint): add the new pairs
        hp = json.loads(hp_path.read_text()); have = {(p["a"], p["b"]) for p in hp}
        rng = np.random.default_rng(len(hp))
        for a_, b_ in pairs:
            if (a_, b_) not in have:
                l_, r_ = (a_, b_) if rng.random() < 0.5 else (b_, a_)
                hp.insert(int(rng.integers(len(hp) + 1)), {"a": a_, "b": b_, "left": l_, "right": r_})
        hp_path.write_text(json.dumps(hp, indent=1)); n = len(hp)
    vlmtest.rate(cdir, n=n, source="pairs.json",
                 question="Which world would you choose as the parent, to evolve toward more complex, life-like structure?",
                 description="Each image is one whole world at four times, far enough apart to show movement, in reading order.")


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
    r.add_argument("--out", default="runs/mace-evo-jev-v3")
    r.add_argument("--judge", default="jev", help="jev | luna | fake (names in configs/lenia.yaml judges)")
    r.add_argument("--config", default=None)
    r.add_argument("--backend", default="numpy", help="numpy | auto | mps | cuda")
    r.add_argument("--generations", type=int, default=30)
    r.add_argument("--lineages", type=int, default=12)
    r.add_argument("--mutants", type=int, default=4)
    r.add_argument("--steps", type=int, default=1200, help="steps a child runs from its parent's world")
    r.add_argument("--frame-gap", type=int, default=400, help="steps between the four snapshots the judge sees")
    r.add_argument("--warmup", type=int, default=1500, help="steps founders grow from noise")
    r.add_argument("--size", type=int, default=160)
    r.add_argument("--R", type=float, default=13)
    r.add_argument("--beta", type=float, default=2.0)
    r.add_argument("--max-texture", type=float, default=0.3, help="grid-noise rule; 0 = off")
    r.add_argument("--threshold", type=float, default=0.6)
    r.add_argument("--mut-scale", type=float, default=3.0)
    r.add_argument("--aging-check", type=int, default=3, help="generations that also judge null vs previous picture")
    r.add_argument("--rule-seed", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sheet")
    s.add_argument("--out", default="runs/mace-evo-jev-v3")
    s.add_argument("--checkpoints", default="10,20,30")
    for name, fn in (("control", "time-only control: founders' own rules for as many steps as the run"),
                     ("check", "re-render founder / final / control and judge them (held-out judge + driver)"),
                     ("rate", "rate the check pairs yourself in the browser")):
        c = sub.add_parser(name, help=fn)
        c.add_argument("--out", default="runs/mace-evo-jev-v3")
        c.add_argument("--backend", default="numpy")
        if name == "check":
            c.add_argument("--judges", default="luna,jev")
            c.add_argument("--config", default=None)
            c.add_argument("--render-steps", type=int, default=1200)
            c.add_argument("--mid", type=int, default=None, help="mid-run checkpoint generation (default: half the run, rounded to 10)")
            c.add_argument("--frame-gap", type=int, default=400)
    a = ap.parse_args()
    if hasattr(a, "backend"):
        backend.configure(a.backend)
    {"run": run, "sheet": sheet, "control": control, "check": check, "rate": rate}[a.cmd](a)


if __name__ == "__main__":
    main()
