"""CLIP targeted-search bench: which search settings actually steer MaCE-Lenia worlds toward a text target?
(Meeting 10-30, direction 1; project doc "Stage 1 - expedition v3 (heritability and comma selection)".)

Stage 1's expeditions barely climbed (Jev's unselected child mean 0.46 -> 0.52 over 10 steps) and there was no
objective, held-out score to tell why. CLIP gives one: free, local and absolute. Each run evolves a start world toward a
text prompt, scoring every child by the cosine similarity between CLIP's embedding of its final snapshot and the
prompt's. Nothing here calls an API.

  python scripts/clipbench.py run    --run RUNS/mace-qd5-sarch-s0 --out RUNS/clipbench --backend auto   # Colab, ~1 h
  python scripts/clipbench.py report --out RUNS/clipbench
  python scripts/clipbench.py gifs   --run RUNS/mace-qd5-sarch-s0 --out RUNS/clipbench --backend auto  # before/after films
  python scripts/clipbench.py run --run RUNS/mace-qd5-sarch-s0 --out RUNS/clipbench-jev --backend auto \
      --configs es-regrow-1,jev-regrow-1,local-1                      # round 2: can Jev steer? (~$0.35, ~40 min)
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
Starts: --starts random elites of the base run's final map (fixed seed, grid noise excluded). Targets: TARGETS[:--targets], or any text with --prompts.

Round 1 (10-30, runs/clipbench): selection climbs (es-* at scale 1 beat the local control 7/8; scale 0.3 barely
climbs: expedition v3 was too gentle); world inheritance adds a little (+0.004) but can spread into grid-filling
textures; sep-CMA-ES climbs less but in every run. Round 2: jev-* steer by Jev (the expedition's goal score vs the
start) while CLIP, held out, scores every child; mutation randomness is shared across configurations (seeded by
target, start and step), so jev-regrow-1 vs es-regrow-1 is paired.
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
from lenia import backend, render, seeds
from lenia import mace as ME
from lenia.embed import ClipScorer
from lenia.mace import mutate, texture
from lenia.sim import Rule

TARGETS = ("a biological cell", "a worm", "a ring", "two cells", "a starfish", "a cluster of many small cells")


def targets_of(a) -> list:
    return [t.strip() for t in a.prompts.split("|")] if getattr(a, "prompts", None) else list(TARGETS[:a.targets])


def target_id(t: str) -> int:
    """Seed component for a target: its TARGETS index (round 1-2 seeds unchanged), else a stable hash of the text."""
    import zlib
    return TARGETS.index(t) if t in TARGETS else 100 + zlib.crc32(t.encode()) % 10 ** 6
CONFIGS = {
    "es-regrow-0.3": {"opt": "es", "world": "regrow", "scale": 0.3},
    "es-inherit-0.3": {"opt": "es", "world": "inherit", "scale": 0.3},
    "es-regrow-1": {"opt": "es", "world": "regrow", "scale": 1.0},
    "es-inherit-1": {"opt": "es", "world": "inherit", "scale": 1.0},
    "local-1": {"opt": "local", "world": "regrow", "scale": 1.0},
    "cma": {"opt": "cma", "world": "regrow", "scale": 1.0},
    # Jev steering, CLIP held out (round 2, 10-30): selection by Jev's goal score against the start (the expedition's
    # rule), with the target as the goal; CLIP still scores every child but never selects.
    "jev-regrow-1": {"opt": "es", "world": "regrow", "scale": 1.0, "judge": "jev"},
    "jev-inherit-1": {"opt": "es", "world": "inherit", "scale": 1.0, "judge": "jev"},
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


def motion(film) -> float:
    """mace_qd's motion: |last - first snapshot| / mass over the rendered window (frozen ring patterns ~0.5, visibly
    rearranging worlds 1.3-1.6). CLIP sees only the last snapshot, so nothing in its score rewards movement."""
    first, last = np.asarray(film[0]), np.asarray(film[-1])
    return float(np.abs(last - first).sum() / max(float(last.sum()), 1e-9))


def grow(rules: list, worlds: list, steps: int, a) -> list:
    """Simulate; per rule: the 2x2 picture, its final snapshot (the CLIP input), the final state, the texture."""
    shape, out = (a.size, a.size), []
    for k in range(0, len(rules), a.sim_batch):
        final, films = ME.simulate(rules[k:k + a.sim_batch], np.stack(worlds[k:k + a.sim_batch]), steps, shape, a.gap)
        for st, f in zip(final, films):
            pic = ME.picture(f)
            out.append({"pic": pic, "tile": pic[-ME.TILE:, -ME.TILE:], "state": np.asarray(st, np.float32),
                        "texture": float(texture(st)), "motion": motion(f)})
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
    for t in targets_of(a):                # sanity check: real CLIP cosines for these pictures sit around 0.15-0.30
        print(f"start scores for '{t}': " + ", ".join(f"{x:.3f}" for x in clip.score([v["tile"] for v in ev], t)), file=sys.stderr)
    runs = []
    for name in a.configs.split(","):
        c = CONFIGS[name]
        for t in targets_of(a):
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
            rng = np.random.default_rng([a.seed, target_id(r["target"]), r["start"], step])   # shared across configs
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
        jev = jev_scores(a, out, runs, kids, ev) if any(r.get("judge") == "jev" for r in runs) else {}
        for ri, r in enumerate(runs):
            ks = [k for k, kid in enumerate(kids) if kid[0] == ri]
            clip_sc = emb[ks] @ clip.text(r["target"])
            ok = np.array([ev[k]["texture"] <= a.max_texture and ev[k]["motion"] >= a.min_motion for k in ks])
            sc = np.array([jev.get(k, np.nan) for k in ks]) if r.get("judge") == "jev" else clip_sc
            valid = np.where(ok, sc, -np.inf)
            j = int(np.argmax(valid)) if ok.any() else None
            if r["opt"] == "cma":
                r["es"].tell([kids[k][3] for k in ks], list(np.where(ok, -sc, 1.0)))
            if j is not None and r["opt"] == "es":
                k = ks[j]
                r["parent"], r["state"], r["parent_score"] = kids[k][1].to_dict(), ev[k]["state"], float(clip_sc[j])
            if j is not None and float(clip_sc[j]) > r["best"]["score"]:   # the record is always in CLIP's terms
                r["best"] = {"score": float(clip_sc[j]), "step": step, "pic": ev[ks[j]]["pic"], "rule": kids[ks[j]][1].to_dict(),
                             "motion": ev[ks[j]]["motion"]}
            rec = {"step": step, "mean": float(clip_sc[ok].mean()) if ok.any() else None,
                   "max": float(clip_sc[ok].max()) if ok.any() else None, "valid": int(ok.sum()), "parent": r["parent_score"],
                   "motion": float(np.mean([ev[k]["motion"] for k, o in zip(ks, ok) if o])) if ok.any() else None}
            if r.get("judge") == "jev" and ok.sum() > 2:
                rec["jev_mean"] = float(np.mean(sc[ok]))
                rec["rho"] = spearman(sc[ok], clip_sc[ok])      # do Jev and CLIP rank these siblings alike?
            r["steps"].append(rec)
        S["step"] = step
        tmp = out / "checkpoint.tmp"
        tmp.write_bytes(pickle.dumps(S))
        tmp.replace(ck)
        print(f"step {step}/{a.steps}: {len(kids)} worlds in {time.time() - t0:.0f}s; " + ", ".join(
            f"{c} {np.mean([r['steps'][-1]['mean'] or np.nan for r in runs if r['config'] == c]):.3f}"
            for c in dict.fromkeys(r["config"] for r in runs)), file=sys.stderr)
    save(out, S)
    report(a)


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else float("nan")


_JUDGE = {}


def jev_scores(a, out: Path, runs: list, kids: list, ev: list) -> dict:
    """Jev's goal score against the start for every child of a Jev-steered run: the attain prompt with the target as
    the goal, both orders (as an expedition, minus the complexity floor). Returns {child index: P}."""
    from lenia.config import REPO_ROOT, load
    from lenia.judge import Judge, MultiQuery, load_env
    from lenia.prompts import attain_prompt
    if "j" not in _JUDGE:
        cfg = load(None)
        load_env(REPO_ROOT / ".env")
        _JUDGE["j"] = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                            out / "judge_cache.sqlite", yes=True)
    mq, idx = [], []
    for k, (ri, *_) in enumerate(kids):
        r = runs[ri]
        if r.get("judge") != "jev" or ev[k]["texture"] > a.max_texture or ev[k]["motion"] < a.min_motion:
            continue
        pid = attain_prompt({"id": "bench_" + r["target"].replace(" ", "_"), "sees": "an ordinary world from this search",
                             "push": f"make it look like {r['target']}"})
        mq += [MultiQuery((pid,), ev[k]["pic"], r["start_pic"]), MultiQuery((pid,), r["start_pic"], ev[k]["pic"])]
        idx.append(k)
    vs = _JUDGE["j"].judge_multi(mq, desc="bench Jev")
    if any(not all(v.ok for v in vv) for vv in vs):
        sys.exit("some Jev requests failed (not cached); re-run to resume")
    return {k: 0.5 * (vs[2 * n][0].p_first + 1 - vs[2 * n + 1][0].p_first) for n, k in enumerate(idx)}


def save(out: Path, S: dict) -> None:
    (out / "img").mkdir(exist_ok=True)
    rows = []
    for k, r in enumerate(S["runs"]):
        Image.fromarray(r["best"]["pic"]).save(out / "img" / f"best_{k:03d}.png")
        Image.fromarray(r["start_pic"]).save(out / "img" / f"start_{r['start']}.png")
        rows.append({x: r[x] for x in ("config", "opt", "world", "scale", "target", "start", "start_image", "score0")} |
                    {"steps": r["steps"], "best_score": r["best"]["score"], "best_step": r["best"]["step"],
                     "best_rule": r["best"].get("rule"), "best_motion": r["best"].get("motion"), "best_image": f"img/best_{k:03d}.png"})
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
    jr = [r for r in R if any("rho" in s for s in r["steps"])]
    if jr:
        L += ["", "Jev-steered runs: Spearman correlation between Jev's goal scores and CLIP's among siblings, per step "
              "(1 = same ranking, 0 = unrelated):", ""]
        for c in dict.fromkeys(r["config"] for r in jr):
            rr = [r for r in jr if r["config"] == c]
            L.append(f"- {c}: mean rho {mean([s.get('rho') for r in rr for s in r['steps']]):+.3f}; by step " + ", ".join(
                f"{mean([r['steps'][i].get('rho') for r in rr if len(r['steps']) > i]):+.2f}" for i in range(len(rr[0]['steps']))))
    if any("motion" in s for r in R for s in r["steps"]):
        L += ["", "Motion (mace_qd's measure; frozen ~0.5, rearranging 1.3-1.6), children's mean at step 1 -> last step, and "
              "the best world's (CLIP sees one frame, so nothing here rewards movement):", ""]
        for c in cfgs:
            rr = [r for r in R if r["config"] == c]
            L.append(f"- {c}: children {mean([r['steps'][0].get('motion') for r in rr]):.2f} -> "
                     f"{mean([r['steps'][-1].get('motion') for r in rr]):.2f}; best worlds {mean([r.get('best_motion') for r in rr]):.2f}")
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


def gifs(a) -> None:
    """Before/after films: per target, one GIF with a row per start world and a column per setting (the start, then
    each configuration's best world), every world growing from the same smoothed-noise start, as it was scored.
    Regrow configurations replay exactly. Inherit worlds can't (they depend on the whole lineage of states), so for
    those the checkpoint's final parent world is continued instead, labelled as such. Prints a replay check: CLIP's
    score of each replayed final picture vs the score recorded in the run."""
    from PIL import ImageFont
    out = Path(a.out)
    backend.configure(a.backend)
    res = json.loads((out / "results.json").read_text())
    R, A = res["runs"], res["args"]
    st = json.loads((Path(a.run) / "state.json").read_text())
    qa = QD.settings(st)
    by_img = {e["image"]: e["rule"] for m in QD.islands_of(st) for e in m.values() if "image" in e}
    shape, steps = (A["size"], A["size"]), A["warmup"] + A["render"]
    A0 = seeds.smooth_noise(qa.R, shape, 3, seed=qa.start_seed, size_r=3.0)
    ck = out / "checkpoint.pkl"
    CK = pickle.loads(ck.read_bytes())["runs"] if ck.exists() else []
    cfgs = [c for c in a.configs.split(",") if any(r["config"] == c for r in R)]
    panels = []                                   # (target, start, column label, rule, world, recorded score or None)
    for t in dict.fromkeys(r["target"] for r in R):
        for s in sorted({r["start"] for r in R if r["target"] == t}):
            rr = {r["config"]: r for r in R if r["target"] == t and r["start"] == s}
            r0 = next(iter(rr.values()))
            panels.append((t, s, "start", by_img[r0["start_image"]], A0, r0["score0"]))
            for c in cfgs:
                r = rr.get(c)
                if r is None:
                    continue
                if CONFIGS[c]["world"] == "inherit":
                    k = next((x for x in CK if x["config"] == c and x["target"] == t and x["start"] == s), None)
                    if k is not None:
                        panels.append((t, s, f"{c.replace('es-', '')} (continued)", k["parent"], k["state"], None))
                elif r.get("best_rule"):
                    panels.append((t, s, c, r["best_rule"], A0, r["best_score"]))
    keep_pic = {steps - i * A["gap"] for i in range(ME.FRAMES)}
    films, pics = [[] for _ in panels], [[] for _ in panels]
    for k0 in range(0, len(panels), a.sim_batch):
        P = panels[k0:k0 + a.sim_batch]
        b = backend.make_batch([Rule.from_dict(p[3]) for p in P], shape)
        X = backend.to_device(np.stack([np.asarray(p[4], np.float32) for p in P]))
        for t in range(steps + 1):
            if t:
                X = b.step(X)
            if t % a.every == 0 or t in keep_pic:
                x = backend.to_numpy(X).copy()
                for j in range(len(P)):
                    if t % a.every == 0:
                        films[k0 + j].append((t, x[j]))
                    if t in keep_pic:
                        pics[k0 + j].append(x[j])
        print(f"simulated {min(k0 + a.sim_batch, len(panels))}/{len(panels)} worlds", file=sys.stderr)
    clip = ClipScorer(a.clip)
    print("replay check (CLIP score of the replayed world vs recorded):", file=sys.stderr)
    for p, pc in zip(panels, pics):
        sc = float(clip.score([ME.picture(np.stack(pc))[-ME.TILE:, -ME.TILE:]], p[0])[0])
        p_rec = f"{p[5]:.3f}" if p[5] is not None else "  -  "
        print(f"  {p[0]:<20} start {p[1]} {p[2]:<26} recorded {p_rec}  replayed {sc:.3f}  motion {motion(pc):.2f}", file=sys.stderr)
    (out / "gifs").mkdir(exist_ok=True)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    px, pad, head = a.px, 6, 34
    for t in dict.fromkeys(p[0] for p in panels):
        idx = [i for i, p in enumerate(panels) if p[0] == t]
        rows = sorted({panels[i][1] for i in idx})
        cols = list(dict.fromkeys(panels[i][2] for i in idx))
        W, H = pad + len(cols) * (px + pad), head + len(rows) * (px + 16 + pad)
        scale = {i: max(float(np.percentile(np.stack([f for _, f in films[i]][-max(1, len(films[i]) // 2):]), 99.5)), 1e-6)
                 for i in idx}                    # one brightness per world, from its second half (noise excluded)
        frames = []
        for n in range(len(films[idx[0]])):
            im = Image.new("RGB", (W, H), (24, 24, 24))
            d = ImageDraw.Draw(im)
            tt = films[idx[0]][n][0]
            d.text((pad, 4), f"'{t}'  step {tt}/{steps}" + ("  (scored at the end)" if tt == steps else ""),
                   fill="white", font=font)
            for i in idx:
                r_, c_ = rows.index(panels[i][1]), cols.index(panels[i][2])
                x0, y0 = pad + c_ * (px + pad), head + r_ * (px + 16 + pad)
                if r_ == 0:
                    d.text((x0, head - 16), panels[i][2], fill=(200, 200, 200), font=font)
                f = films[i][n][1]
                tile = Image.fromarray(render.colorize(np.clip(f / scale[i], 0, 1))).resize((px, px), Image.BILINEAR)
                im.paste(tile, (x0, y0))
                lab = f"start {panels[i][1]}" + (f"   CLIP {panels[i][5]:.3f}" if panels[i][5] is not None else "")
                d.text((x0, y0 + px + 2), lab, fill=(200, 200, 200), font=font)
            frames.append(im.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE))
        frames += [frames[-1]] * a.hold
        f_out = out / "gifs" / f"{t.replace(' ', '_')}.gif"
        frames[0].save(f_out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
        print(f"wrote {f_out} ({f_out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def cross(a) -> None:
    """Is CLIP's steering target-specific? Score every run's start and best world against every target of the bench
    (plus a neutral prompt). Specific gain = the best world's gain on its own target minus its mean gain on the other
    targets: ~0 means the search made worlds more CLIP-friendly in general (sharper, more object-like), not more
    like the target. Also: how often the best world's top-scoring target is its own (chance = 1 / number of targets)."""
    out = Path(a.out)
    R = json.loads((out / "results.json").read_text())["runs"]
    T = list(dict.fromkeys(r["target"] for r in R))
    P = T + [x for x in a.neutral.split("|") if x]
    clip = ClipScorer(a.clip)
    tile = lambda f: np.asarray(Image.open(out / f).convert("RGB"))[-ME.TILE:, -ME.TILE:]
    txt = np.stack([clip.text(p) for p in P], 1)
    L = ["# Cross-scores: every start and best world against every target\n",
         "Rows: runs (config, own target, start). Cells: best world's score, (gain over the start's score for the same "
         "prompt). * = the prompt CLIP ranks highest among the targets for this world.\n",
         "| config | target | start | " + " | ".join(P) + " | specific gain |", "|---" * (len(P) + 4) + "|"]
    spec, hits = {}, {}
    for k, r in enumerate(R):
        e = clip.images([tile(f"img/start_{r['start']}.png"), tile(r["best_image"])])
        s0, s1 = e[0] @ txt, e[1] @ txt
        g = s1 - s0
        own = T.index(r["target"])
        sg = float(g[own] - np.mean([g[j] for j in range(len(T)) if j != own]))
        top = int(np.argmax(s1[:len(T)]))
        spec.setdefault(r["config"], []).append(sg)
        hits.setdefault(r["config"], []).append(top == own)
        L.append(f"| {r['config']} | {r['target']} | {r['start']} | " + " | ".join(
            f"{s1[j]:.3f}{'*' if j == top else ''} ({g[j]:+.3f})" for j in range(len(P))) + f" | {sg:+.4f} |")
    L += ["", f"Chance of the own target ranking top: {1 / len(T):.2f}.", ""]
    for c in spec:
        L.append(f"- {c}: specific gain {np.mean(spec[c]):+.4f} (mean gain on own target "
                 f"{np.mean([r['best_score'] - r['score0'] for r in R if r['config'] == c]):+.4f}); "
                 f"own target top {sum(hits[c])}/{len(hits[c])}")
    (out / "cross.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


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
    r.add_argument("--prompts", default=None, help='custom targets, "|"-separated (overrides --targets), e.g. "a starfish|a pepperoni pizza"')
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
    r.add_argument("--min-motion", type=float, default=0.0, help="exclude children below this motion (0 = off; logged either way)")
    r.add_argument("--sim-batch", type=int, default=64)
    r.add_argument("--judge", default="jev")
    r.add_argument("--max-usd", type=float, default=1.0, help="hard cap on this bench's Jev spend (its own cache)")
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    g = sub.add_parser("gifs", help="before/after films of the best worlds")
    g.add_argument("--run", required=True, help="the base run (start worlds)")
    g.add_argument("--out", required=True, help="a finished bench folder")
    g.add_argument("--backend", default="numpy")
    g.add_argument("--clip", default="openai/clip-vit-base-patch32")
    g.add_argument("--configs", default="es-regrow-1,cma,es-inherit-1")
    g.add_argument("--every", type=int, default=30, help="simulation steps per GIF frame")
    g.add_argument("--px", type=int, default=200)
    g.add_argument("--fps", type=int, default=15)
    g.add_argument("--hold", type=int, default=20, help="frames to hold the last (scored) frame")
    g.add_argument("--sim-batch", type=int, default=64)
    x = sub.add_parser("cross", help="is the steering target-specific? every world vs every target")
    x.add_argument("--out", required=True)
    x.add_argument("--clip", default="openai/clip-vit-base-patch32")
    x.add_argument("--neutral", default="an abstract colorful pattern", help='"|"-separated extra prompts')
    a = ap.parse_args()
    {"run": run, "report": report, "gifs": gifs, "cross": cross}[a.cmd](a)


if __name__ == "__main__":
    main()
