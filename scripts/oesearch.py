"""Search on ASAL's open-endedness score (11-13; Sam: "how far could we push in that direction"), with a control.

In three sentences: an elitist evolutionary search starts from random MaCE worlds and keeps the worlds whose look keeps
changing over a long run, by ASAL's CLIP score (oecheck passed its sanity check: ring > hub > frozen worm). The same
search on plain large-scale activity is the control, because the score correlates 0.72 with it; only a head-to-head
shows whether CLIP adds anything. Felix then rates the winners blind, on films twice as long as the window searched,
against random starting worlds and the Jev-found catalog worlds (that step is built once the runs are done).

  python scripts/oesearch.py run --out RUNS/oe-search-clip --objective clip --backend auto     # Colab section 12
  python scripts/oesearch.py run --out RUNS/oe-search-act  --objective act  --backend auto --founders-from RUNS/oe-search-clip
  (next, after both runs: a check that grows the winners to 2x the searched length, and a blind rating page)
  python scripts/oesearch.py report --out RUNS/oe-search-clip                                   # anywhere
  Offline dry run: run --out ~/oes-dry --objective clip --clip fake --size 64 --steps 600 --window 300 --every 100
                   --founders 8 --mu 4 --lam 4 --gens 2

Search: (mu + lambda) with elitism (the parent survives unless children beat it; seed 1's comma selection lost progress).
Generation 0 evaluates --founders random rules (lenia.mace.founders' sampler, beta 2) and keeps the best --mu; each
generation makes --lam children of uniformly chosen survivors (stage1.mutants_of at --strength) and keeps the best --mu
of parents + children. Every world grows from the map's fixed start for --steps; the score uses frames every --every
steps from --window on (default 9,000-18,000: the second half, where the worm-like freezes show). Both scores are
computed for every world (CLIP novelty = 1 - ASAL's loss; activity = large-scale change between frames / mass), so
each arm also shows how the other measure moved. The grid-noise rule (texture of the last frame <= 0.3) is kept: it
blocks a known numerical loophole, nothing else. Resumes from <out>/state.json.
From papers: the score (ASAL, Kumar et al. 2024). Ours: the search, the control, the long held-out check.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import stage1 as S1
from lenia import backend, render, seeds
from lenia.mace import texture
from lenia.sample import sample
from lenia.sim import Rule, with_flow


def asal_oe(z: np.ndarray) -> float:
    k = np.tril(z @ z.T, k=-1)
    return float(k.max(axis=1).mean())


def coarse(x: np.ndarray, k: int = 8) -> np.ndarray:
    C, H, W = x.shape
    return x[:, :H // k * k, :W // k * k].reshape(C, H // k, k, W // k, k).mean((2, 4))


class Evaluator:
    def __init__(self, a, s: dict):
        self.a, self.s = a, s
        from lenia.embed import ClipScorer
        self.clip = ClipScorer("fake" if a.clip == "fake" else "openai/clip-vit-base-patch32")
        self.times = list(range(a.window, a.steps + 1, a.every))

    def films(self, rules: list, steps: int, times: list):
        shape = (self.s["size"], self.s["size"])
        A0 = seeds.smooth_noise(self.s["R"], shape, 3, seed=self.s["start_seed"], size_r=3.0)
        want = set(times)
        for k in range(0, len(rules), self.a.batch):
            rr = rules[k:k + self.a.batch]
            b = backend.make_batch(rr, shape)
            X = backend.to_device(np.stack([A0] * len(rr)))
            films = [[A0.astype(np.float32)] if 0 in want else [] for _ in rr]
            for t in range(1, steps + 1):
                X = b.step(X)
                if t in want:
                    x = backend.to_numpy(X)
                    for j in range(len(rr)):
                        films[j].append(np.asarray(x[j], np.float32).copy())
            for f in films:
                yield np.stack(f)

    def measure(self, film: np.ndarray) -> dict:
        scale = max(float(np.percentile(film, 99.5)), 1e-6)
        z = self.clip.images([render.colorize(np.clip(f / scale, 0, 1)) for f in film])
        mass = film.sum((1, 2, 3))
        big = [np.abs(coarse(film[i + 1]) - coarse(film[i])).sum() * 64 / max(mass[i], 1e-9) for i in range(len(film) - 1)]
        return {"clip": round(1 - asal_oe(z), 5), "act": round(float(np.mean(big)), 5),
                "texture": round(float(texture(film[-1])), 4)}

    def evaluate(self, rules: list) -> list:
        return [self.measure(f) for f in self.films(rules, self.a.steps, self.times)]


def settings(a) -> dict:
    s = json.loads(Path(a.worlds).read_text())["settings"] if a.worlds else {"size": 160, "R": 13, "start_seed": 0}
    if a.size:                                   # dry runs only
        s = {**s, "size": a.size, "R": max(3.0, s["R"] * a.size / 160)}
    return s


def fitness(e: dict, a) -> float:
    return -1.0 if e["texture"] > a.max_texture else e[a.objective]


def run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sf = out / "state.json"
    s = settings(a)
    backend.configure(a.backend)
    ev = Evaluator(a, s)
    logf = out / "log.csv"
    new_log = not logf.exists()
    log = csv.writer(open(logf, "a", newline=""))
    if new_log:
        log.writerow(["gen", "id", "parent", "root", "clip", "act", "texture", "fit", "kept"])
    if sf.exists():
        st = json.loads(sf.read_text())
        print(f"resuming {out} at generation {st['gen'] + 1}", file=sys.stderr)
    else:
        rules = sample("multichannel", a.founders, seed=a.rule_seed, R=s["R"], channels=3, self_k=1, cross_k=1)
        rules = [with_flow(r, "mace", 2.0) for r in rules]
        t0 = time.time()
        other = Path(a.founders_from) / "state.json" if a.founders_from else None
        if other and other.exists():             # the other arm already measured both scores on the same founders
            F = json.loads(other.read_text())["founders"]
            same = [Rule.from_dict(p["rule"]).to_dict() for p in F] == [r.to_dict() for r in rules]
            if not same or json.loads(other.read_text())["args"]["steps"] != a.steps:
                sys.exit(f"{other}: different founders or settings; drop --founders-from")
            es = [{k: p[k] for k in ("clip", "act", "texture")} for p in F]
            print(f"generation 0 reused from {other.parent}", file=sys.stderr)
        else:
            es = ev.evaluate(rules)
        pop = [{"id": f"g00_{i:03d}", "rule": r.to_dict(), "root": i, "parent": None, "gen": 0, **e, "fit": fitness(e, a)}
               for i, (r, e) in enumerate(zip(rules, es))]
        keep = sorted(pop, key=lambda p: -p["fit"])[:a.mu]
        for p in pop:
            log.writerow([0, p["id"], "", p["root"], p["clip"], p["act"], p["texture"], p["fit"], int(p in keep)])
        st = {"args": vars(a), "settings": s, "gen": 0, "founders": pop, "pop": keep, "history": []}
        st["history"].append(summary(0, keep, len(pop), time.time() - t0))
        sf.write_text(json.dumps(st))
        print(line(st["history"][-1]), file=sys.stderr, flush=True)
    while st["gen"] < a.gens:
        g = st["gen"] + 1
        rng = np.random.default_rng([a.seed, g])
        pop = st["pop"]
        t0 = time.time()
        pars = [pop[int(rng.integers(len(pop)))] for _ in range(a.lam)]
        kids_r = [S1.mutants_of(Rule.from_dict(p["rule"]), rng, 1, None, a.strength)[0] for p in pars]
        es = ev.evaluate(kids_r)
        kids = [{"id": f"g{g:02d}_{i:03d}", "rule": r.to_dict(), "root": p["root"], "parent": p["id"], "gen": g, **e,
                 "fit": fitness(e, a)} for i, (p, r, e) in enumerate(zip(pars, kids_r, es))]
        keep = sorted(pop + kids, key=lambda p: -p["fit"])[:a.mu]
        ids = {p["id"] for p in keep}
        for p in kids:
            log.writerow([g, p["id"], p["parent"], p["root"], p["clip"], p["act"], p["texture"], p["fit"], int(p["id"] in ids)])
        st["pop"], st["gen"] = keep, g
        st["history"].append(summary(g, keep, len(kids), time.time() - t0, kids))
        sf.write_text(json.dumps(st))
        print(line(st["history"][-1]), file=sys.stderr, flush=True)
    report(a)


def summary(g, keep, n, sec, kids=None) -> dict:
    h = {"gen": g, "evaluated": n, "seconds": round(sec), "best_fit": keep[0]["fit"],
         "mean_fit": round(float(np.mean([p["fit"] for p in keep])), 5),
         "mean_clip": round(float(np.mean([p["clip"] for p in keep])), 5),
         "mean_act": round(float(np.mean([p["act"] for p in keep])), 5),
         "lineages": len({p["root"] for p in keep})}
    if kids is not None:
        h["kids_kept"] = sum(p["id"] in {q["id"] for q in keep} for p in kids)
    return h


def line(h) -> str:
    return (f"gen {h['gen']:>2}: best {h['best_fit']:.4f}, survivors' mean clip {h['mean_clip']:.4f} act {h['mean_act']:.3f}, "
            f"{h['lineages']} lineages" + (f", {h['kids_kept']} children kept" if "kids_kept" in h else "")
            + f" ({h['seconds']} s)")


def report(a) -> None:
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    ar = st["args"]
    L = [f"# Search on {ar['objective']} ({'ASAL CLIP novelty' if ar['objective'] == 'clip' else 'large-scale activity'}), "
         f"{ar['founders']} random founders, mu {ar['mu']} + lambda {ar['lam']}, strength {ar['strength']}, "
         f"{ar['steps']:,} steps (scored from {ar['window']:,})\n",
         "| gen | best fitness | survivors: mean CLIP novelty | mean activity | lineages | children kept | s |",
         "|---|---|---|---|---|---|---|"]
    for h in st["history"]:
        L.append(f"| {h['gen']} | {h['best_fit']:.4f} | {h['mean_clip']:.4f} | {h['mean_act']:.3f} | {h['lineages']} | "
                 f"{h.get('kids_kept', '-')} | {h['seconds']} |")
    f = st["founders"]
    L.append(f"\nFounders (generation 0, all {len(f)}): mean CLIP novelty {np.mean([p['clip'] for p in f]):.4f}, "
             f"mean activity {np.mean([p['act'] for p in f]):.3f}; "
             f"{sum(p['texture'] > ar['max_texture'] for p in f)} failed the grid-noise rule.")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--objective", choices=["clip", "act"], required=True)
    r.add_argument("--worlds", default=None, help="a worlds.json with settings (default: size 160, R 13, start 0)")
    r.add_argument("--backend", default="numpy")
    r.add_argument("--clip", default="openai", help="openai (ASAL's ViT-B/32) or fake (dry runs)")
    r.add_argument("--founders", type=int, default=64)
    r.add_argument("--founders-from", default=None, help="reuse generation 0 from the other arm's run folder")
    r.add_argument("--rule-seed", type=int, default=1113)
    r.add_argument("--mu", type=int, default=16)
    r.add_argument("--lam", type=int, default=32)
    r.add_argument("--gens", type=int, default=20)
    r.add_argument("--strength", type=float, default=1.0)
    r.add_argument("--steps", type=int, default=18000)
    r.add_argument("--window", type=int, default=9000)
    r.add_argument("--every", type=int, default=250)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--max-texture", type=float, default=0.3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--size", type=int, default=0, help="dry runs only")
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
