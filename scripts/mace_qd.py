"""MAP-Elites over MaCE-Lenia worlds, with Jev duels as the quality signal.

  python scripts/mace_qd.py run   --out RUNS/mace-qd --backend auto     # Colab: section 12 of the notebook
  python scripts/mace_qd.py map   --out RUNS/mace-qd                    # map.png + summary
  python scripts/mace_qd.py check --out RUNS/mace-qd                    # final elites vs elites at a snapshot
  python scripts/mace_qd.py rate  --out RUNS/mace-qd                    # your ratings of the check pairs (Mac)
  python scripts/mace_qd.py run   --out /tmp/qd-dry --judge fake --iterations 3 --batch 8 --init 8 --size 96 \
      --warmup 200 --render 120 --gap 40                                # offline dry run

Why MAP-Elites (see the project doc "MaCE-Lenia Jev-guided evolution results"): the hill climber
(mace_evo.py) duels each mutant only against its own parent, and after ~60 generations it moved sideways
(none of Luna, Jev or Felix preferred gen 120 to gen 60), drifting toward fragmentation and world-filling
texture. Here a candidate must beat the elite already holding its niche, and the two behaviours the
judge over-rewarded become the map's axes instead of directions to drift in.

Evaluation: every rule is grown from the same smoothed-noise start (seed --start-seed) for --warmup
steps and filmed for --render more; the judge sees four snapshots --gap steps apart (2x2 image). No world
is inherited, so every elite and challenger is the same age and a rule's world is reproducible.
- Constraint: the grid-noise rule only (texture of the final frame <= --max-texture).
- Axes: bodies (connected bodies of >= 9 cells in the final frame; bins 1, 2, 3-4, 5-8, 9-16, 17-32, 33+)
  x motion (|final - first snapshot| / mass, i.e. change over --render steps; --motion-edges, calibrated
  on the 64 seeds: frozen ring patterns score ~0.5, visibly rearranging worlds 1.3-1.6).
- Insertion: a candidate takes an empty cell; for an occupied cell it duels the elite (Jev, both orders,
  P = mean) and replaces it if P >= --threshold. Several candidates for one cell in one iteration: the
  first takes an empty cell, the others duel it; the highest P >= threshold wins.
- Variation: parents drawn uniformly from filled cells; kernel mutation at --mut-scale plus
  beta x exp(N(0, 0.1)) in [0.5, 4], as in mace_evo.
- Seeds: the 12 mace_evo founders plus --init more draws of the same sampler (another seed).
Saves state after every iteration (resumes) and an archive snapshot every --snapshot-every iterations.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mace_evo as ME
from lenia import backend, render, seeds
from lenia.alive import bodies as label_bodies
from lenia.config import REPO_ROOT, load
from lenia.explore import mutate, texture
from lenia.judge import Judge, Query, load_env
from lenia.sample import sample
from lenia.sim import Rule, with_flow

PROMPT_ID = ME.PROMPT_ID
BODY_EDGES = [2, 3, 5, 9, 17, 33]                     # bins: 1 | 2 | 3-4 | 5-8 | 9-16 | 17-32 | 33+
BODY_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]


# ---------- evaluation ----------

def n_bodies(A: np.ndarray, min_cells: int = 9) -> int:
    m = A.sum(0)
    mask = m > 0.1 * max(float(np.percentile(m, 99.5)), 1e-9)
    if not mask.any():
        return 0
    lab = label_bodies(mask)
    _, area = np.unique(lab[mask], return_counts=True)
    return int((area >= min_cells).sum())


def coverage(A: np.ndarray) -> float:
    m = A.sum(0)
    return float((m > 0.05 * max(float(m.max()), 1e-9)).mean())


def evaluate(rules: list, a) -> list:
    """Grow every rule from the fixed start; return per rule: picture, descriptors, texture."""
    shape = (a.size, a.size)
    A0 = seeds.smooth_noise(a.R, shape, 3, seed=a.start_seed, size_r=3.0)
    out = []
    for k in range(0, len(rules), a.sim_batch):
        rr = rules[k:k + a.sim_batch]
        _, films = ME.simulate(rr, np.stack([A0] * len(rr)), a.warmup + a.render, shape, a.gap)
        for f in films:
            first, last = f[0], f[-1]
            mass = max(float(last.sum()), 1e-9)
            out.append({"pic": ME.picture(f), "bodies": n_bodies(last), "motion": float(np.abs(last - first).sum() / mass),
                        "texture": float(texture(last)), "coverage": coverage(last)})
    return out


def cell_of(e: dict, a) -> tuple[int, int]:
    return int(np.searchsorted(BODY_EDGES, e["bodies"], side="right")), int(np.searchsorted(a.motion_edges, e["motion"]))


# ---------- archive ----------

def key(c) -> str:
    return f"{c[0]}-{c[1]}"


def save_state(out: Path, st: dict) -> None:
    tmp = out / "state.tmp.json"
    tmp.write_text(json.dumps(st))
    tmp.replace(out / "state.json")


def insert(st: dict, cands: list, evals: list, judge, out: Path, it: int, log, a) -> dict:
    """cands: [(rule, parent_key or None)]. Returns counts."""
    arch = st["archive"]
    imgdir = out / "img"
    by_cell: dict = {}
    n_tex = 0
    for n, ((rule, parent), e) in enumerate(zip(cands, evals)):
        if a.max_texture > 0 and e["texture"] > a.max_texture:
            n_tex += 1
            log.writerow([it, n, parent or "", "", "texture", "", 0, e["bodies"], round(e["motion"], 4), round(e["texture"], 4),
                          round(rule.beta, 3)])
            continue
        by_cell.setdefault(key(cell_of(e, a)), []).append(n)
    new_cells, duels = 0, []                     # duels: (cell, challenger n, incumbent picture, incumbent label)
    first_new = {}
    for ck, ns in by_cell.items():
        if ck in arch:
            inc = np.asarray(Image.open(imgdir / arch[ck]["image"]).convert("RGB"))
            duels += [(ck, n, inc) for n in ns]
        else:
            first_new[ck] = ns[0]
            duels += [(ck, n, evals[ns[0]]["pic"]) for n in ns[1:]]
    qs = []
    for ck, n, inc in duels:
        qs += [Query(PROMPT_ID, evals[n]["pic"], inc), Query(PROMPT_ID, inc, evals[n]["pic"])]
    vs = judge.judge_many(qs, desc=f"it {it}") if qs else []
    best = {}
    for d, (ck, n, _) in enumerate(duels):
        va, vb = vs[2 * d], vs[2 * d + 1]
        p = None if not (va.ok and vb.ok) else 0.5 * (va.p_first + 1 - vb.p_first)
        acc = p is not None and p >= a.threshold
        if acc and p > best.get(ck, (0.0, None))[0]:
            best[ck] = (p, n)
        e = evals[n]
        log.writerow([it, n, cands[n][1] or "", ck, "duel" if ck in arch else "duel_new", "" if p is None else round(p, 4),
                      int(acc), e["bodies"], round(e["motion"], 4), round(e["texture"], 4), round(cands[n][0].beta, 3)])

    def place(ck, n, p):
        rule, parent = cands[n]
        e = evals[n]
        name = f"it{it:04d}_{n:03d}.png"
        Image.fromarray(e["pic"]).save(imgdir / name)
        prev = arch.get(ck)
        arch[ck] = {"rule": rule.to_dict(), "image": name, "it": it, "parent": parent,
                    "bodies": e["bodies"], "motion": round(e["motion"], 4), "texture": round(e["texture"], 4),
                    "coverage": round(e["coverage"], 4), "p": None if p is None else round(p, 4),
                    "replaced": 0 if prev is None else prev.get("replaced", 0) + 1}

    for ck, n in first_new.items():
        if ck not in best:
            place(ck, n, None)
            new_cells += 1
            e = evals[n]
            log.writerow([it, n, cands[n][1] or "", ck, "new", "", 1, e["bodies"], round(e["motion"], 4),
                          round(e["texture"], 4), round(cands[n][0].beta, 3)])
    replaced = 0
    for ck, (p, n) in best.items():
        if ck not in arch:
            new_cells += 1                       # a later candidate beat the first one for a new cell
        else:
            replaced += 1
        place(ck, n, p)
    return {"texture_fail": n_tex, "new": new_cells, "replaced": replaced, "duels": len(duels)}


def run(a) -> None:
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge]}, out / "judge_cache.sqlite", yes=True)
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        keep = ("iterations", "backend", "judge", "out", "config", "sim_batch")
        st["args"].update({k: getattr(a, k) for k in keep})
        a = argparse.Namespace(**{**vars(a), **st["args"]})
        print(f"resuming at iteration {st['it']} ({len(st['archive'])} cells filled)", file=sys.stderr)
    else:
        st = {"it": 0, "args": vars(a), "prompt": ME.PROMPTS[PROMPT_ID], "archive": {}, "progress": []}
    fh = open(out / "log.csv", "a", newline="")
    log = csv.writer(fh)
    if fh.tell() == 0:
        log.writerow(["it", "cand", "parent", "cell", "kind", "p", "accepted", "bodies", "motion", "texture", "beta"])
    while st["it"] <= a.iterations:
        it = st["it"]
        t0 = time.time()
        rng = np.random.default_rng([a.seed, it])
        if it == 0:
            found = ME.founders(argparse.Namespace(lineages=12, rule_seed=3, R=a.R, beta=a.beta, size=a.size), (a.size, a.size))[0]
            extra = [with_flow(r, "mace", a.beta) for r in
                     sample("multichannel", a.init, seed=a.init_seed, R=a.R, channels=3, self_k=1, cross_k=1)]
            cands = [(r, None) for r in found + extra]
        else:
            keys = sorted(st["archive"])
            cands = []
            for _ in range(a.batch):
                pk = keys[int(rng.integers(len(keys)))]
                par = Rule.from_dict(st["archive"][pk]["rule"])
                k = mutate(par, rng, a.mut_scale)
                k.flow = "mace"
                k.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                cands.append((k, pk))
        evals = evaluate([c[0] for c in cands], a)
        res = insert(st, cands, evals, judge, out, it, log, a)
        fh.flush()
        st["progress"].append({"it": it, "filled": len(st["archive"]), **res, "sec": round(time.time() - t0, 1)})
        if it % a.snapshot_every == 0:
            (out / "snapshots" / f"archive_it{it:04d}.json").write_text(json.dumps(st["archive"]))
        st["it"] = it + 1
        save_state(out, st)
        print(f"it {it}: {len(st['archive'])} cells filled, {res['new']} new, {res['replaced']} replaced, "
              f"{res['duels']} duels, {res['texture_fail']} grid-noise, {time.time() - t0:.0f}s, "
              f"judge spent ${judge.spent:.4f}", file=sys.stderr)
    fh.close()
    draw_map(out)


# ---------- outputs ----------

def draw_map(out: Path, path: Path | None = None, archive: dict | None = None, title: str | None = None) -> Path:
    st = json.loads((out / "state.json").read_text())
    a = argparse.Namespace(**st["args"])
    arch = archive if archive is not None else st["archive"]
    T, pad, left, top = 150, 4, 70, 40
    nb, nm = len(BODY_LABELS), len(a.motion_edges) + 1
    W, H = left + nm * (T + pad), top + nb * (T + pad) + 30
    im = Image.new("RGB", (W, H), (12, 13, 20))
    d = ImageDraw.Draw(im)
    d.text((8, 8), title or f"{out.name}: elites by bodies (rows) x motion (columns), {len(arch)}/{nb * nm} cells, iteration {st['it'] - 1}",
           fill=(225, 228, 235))
    me = [0.0] + list(a.motion_edges)
    for j in range(nm):
        lab = f">{me[j]:.2f}" if j == nm - 1 else f"{me[j]:.2f}-{me[j + 1]:.2f}"
        d.text((left + j * (T + pad) + 4, H - 24), lab, fill=(160, 165, 180))
    for i in range(nb):
        d.text((8, top + i * (T + pad) + T // 2 - 6), BODY_LABELS[i], fill=(160, 165, 180))
        for j in range(nm):
            e = arch.get(f"{i}-{j}")
            x, y = left + j * (T + pad), top + i * (T + pad)
            if e is None:
                d.rectangle([x, y, x + T, y + T], outline=(40, 42, 55))
                continue
            pic = np.asarray(Image.open(out / "img" / e["image"]).convert("RGB"))
            im.paste(Image.fromarray(pic[-320:, -320:]).resize((T, T), Image.BILINEAR), (x, y))
    path = path or out / "map.png"
    im.save(path)
    return path


def show_map(a) -> None:
    out = Path(a.out)
    p = draw_map(out)
    st = json.loads((out / "state.json").read_text())
    pr = st["progress"]
    if pr:
        tail = pr[-10:]
        print(f"iteration {st['it'] - 1}: {len(st['archive'])} cells; last 10 iterations: "
              f"{sum(x['new'] for x in tail)} new, {sum(x['replaced'] for x in tail)} replaced, "
              f"{sum(x['duels'] for x in tail)} duels, {np.mean([x['sec'] for x in tail]):.0f}s per iteration")
    print(p)


def check(a) -> None:
    """Treadmill check: for every cell whose elite changed since snapshot --at, the final elite vs the
    elite it had then (same cell, same behaviour bin), judged by --judges in both orders."""
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    snaps = sorted(int(p.stem[len("archive_it"):]) for p in (out / "snapshots").glob("archive_it*.json"))
    last = st["it"] - 1
    at = a.at if a.at is not None else min(snaps, key=lambda s: abs(s - last // 2))
    old = json.loads((out / "snapshots" / f"archive_it{at:04d}.json").read_text())
    cdir = out / "check"
    (cdir / "images").mkdir(parents=True, exist_ok=True)
    pairs, pics = [], {}
    for ck, e in sorted(st["archive"].items()):
        if ck in old and old[ck]["image"] != e["image"]:
            x, y = f"c{ck}_final", f"c{ck}_it{at:04d}"
            for nm, src in ((x, e["image"]), (y, old[ck]["image"])):
                shutil.copy(out / "img" / src, cdir / "images" / f"{nm}.png")
                pics[nm] = np.asarray(Image.open(cdir / "images" / f"{nm}.png").convert("RGB"))
            pairs.append([x, y])
    (cdir / "pairs.json").write_text(json.dumps(pairs))
    draw_map(out, cdir / f"map_it{at:04d}.png", old, f"{out.name}: archive at iteration {at}")
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    res = {}
    for jn in a.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, cdir / "judge_cache.sqlite", yes=True)
        qs = [q for x, y in pairs for q in (Query(PROMPT_ID, pics[x], pics[y]), Query(PROMPT_ID, pics[y], pics[x]))]
        vs = judge.judge_many(qs, desc=f"check {jn}") if qs else []
        res[jn] = [None if not (vs[2 * k].ok and vs[2 * k + 1].ok) else 0.5 * (vs[2 * k].p_first + 1 - vs[2 * k + 1].p_first)
                   for k in range(len(pairs))]
    json.dump({"at": at, "final_it": last, "pairs": pairs, "p_first_of_pair": res}, open(cdir / "check.json", "w"), indent=1)
    lines = [f"# {out.name}: does a niche's elite improve? (iteration {last} vs {at})\n",
             f"{len(pairs)} cells changed elite since iteration {at} (of {len(old)} filled then, {len(st['archive'])} now). "
             "P = probability the final elite is chosen as the parent, both orders.\n",
             "| judge | mean P(final) | final wins |", "|---|---|---|"]
    for jn, ps in res.items():
        v = [p for p in ps if p is not None]
        lines.append(f"| {jn} | {np.mean(v):.2f} | {sum(p > 0.5 for p in v)}/{len(v)} |" if v else f"| {jn} | - | - |")
    (cdir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def rate(a) -> None:
    from lenia import vlmtest
    cdir = Path(a.out) / "check"
    n = len(json.loads((cdir / "pairs.json").read_text()))
    vlmtest.rate(cdir, n=n, source="pairs.json",
                 question="Which world would you choose as the parent, to evolve toward more complex, life-like structure?",
                 description="Each image is one whole world at four times, far enough apart to show movement, in reading order.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", default="numpy", help="numpy | auto | cuda | mps")
    r.add_argument("--judge", default="jev")
    r.add_argument("--config", default=None)
    r.add_argument("--iterations", type=int, default=150)
    r.add_argument("--batch", type=int, default=64, help="mutants per iteration")
    r.add_argument("--init", type=int, default=52, help="random rules added to the 12 founders at iteration 0")
    r.add_argument("--init-seed", type=int, default=4)
    r.add_argument("--sim-batch", type=int, default=64, help="worlds simulated together (lower if the GPU runs out of memory)")
    r.add_argument("--size", type=int, default=160)
    r.add_argument("--R", type=float, default=13)
    r.add_argument("--beta", type=float, default=2.0)
    r.add_argument("--warmup", type=int, default=1500)
    r.add_argument("--render", type=int, default=1200)
    r.add_argument("--gap", type=int, default=400)
    r.add_argument("--start-seed", type=int, default=0)
    r.add_argument("--motion-edges", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
    r.add_argument("--max-texture", type=float, default=0.3)
    r.add_argument("--threshold", type=float, default=0.6)
    r.add_argument("--mut-scale", type=float, default=3.0)
    r.add_argument("--snapshot-every", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("map")
    m.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--at", type=int, default=None, help="snapshot iteration to compare with (default: the one nearest half the run)")
    c.add_argument("--judges", default="luna,jev")
    c.add_argument("--config", default=None)
    t = sub.add_parser("rate")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        backend.configure(a.backend)
    {"run": run, "map": show_map, "check": check, "rate": rate}[a.cmd](a)


if __name__ == "__main__":
    main()
