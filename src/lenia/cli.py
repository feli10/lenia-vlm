"""lenia: is Lenia a good domain for VLM-judged aesthetics? (Phase 0-1: no VLM calls yet)

  orbium     sanity check: Orbium from its catalogued pattern should glide (class 'mover')
  catalogue  run Chan's species from their own pattern (--start own) or from the seed bank (--start bank)
  survey     Test 1 + 2: sample rules, run each from the seed bank, classify, draw contact sheets
             modes: classic | grid | multikernel | multichannel
  rejudge    re-classify a finished survey with the current thresholds (configs/lenia.yaml: alive)
  explore    mobility test: novelty search over rules (no VLM) from a survey's survivors; does it
             move away from where it starts, and outside Chan's catalogue?
  trace      plot rollouts over time (mass, size, speed; thumbnails) from a run's traces.npz
  show       animated GIF of rules: from a survey (--run, --rules or --alive), from Chan's catalogue
             (--chan O2u,S1s) or a rule given as JSON (--rule); the camera follows the creature

Run `lenia <command> -h` for options.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load, resolve


def _common(cfg):
    from .alive import Thresholds
    from .rollout import Horizon
    return tuple(cfg["world"]["size"]), Horizon.from_dict(cfg["horizon"]), Thresholds.from_dict(cfg["alive"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lenia", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML merged over configs/lenia.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("orbium")
    p.add_argument("--out")

    p = sub.add_parser("catalogue")
    p.add_argument("--start", choices=["own", "bank"], default="own")
    p.add_argument("--n", type=int, default=0, help="random subset of species (0 = all that fit)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")

    p = sub.add_parser("survey")
    p.add_argument("--mode", choices=["classic", "grid", "multikernel", "multichannel"], required=True)
    p.add_argument("--n", type=int, default=500, help="rules (grid: about n points)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int)
    p.add_argument("--out")

    p = sub.add_parser("rejudge")
    p.add_argument("--run", required=True)

    p = sub.add_parser("explore")
    p.add_argument("--from", dest="from_runs", help="survey folder(s), comma-separated: roots = their alive rules")
    p.add_argument("--evals", type=int, default=3000)
    p.add_argument("--children", type=int, default=32, help="children per generation")
    p.add_argument("--child-time", type=float, default=100.0, help="units of time each child is run for")
    p.add_argument("--confirm-time", type=float, default=200.0, help="archive candidates must also be alive after this")
    p.add_argument("--mut-scale", type=float, default=1.0, help="mutation step size multiplier")
    p.add_argument("--rescale-time", type=float, default=50.0,
                   help="archive candidates (and roots) must also live this long at a finer grid (0 = no check)")
    p.add_argument("--rescale-factor", type=int, default=2, help="how much finer: R -> factor*R, pattern upsampled")
    p.add_argument("--rescale-max-dist", type=float, help="... and stay within this descriptor distance of itself "
                                                          "(default: calibrated on the catalogue, see explore.RESCALE_MAX_DIST)")
    p.add_argument("--no-catalogue", action="store_true", help="skip the comparison with Chan's catalogue")
    p.add_argument("--replot", help="instead of searching: redraw a finished run's pictures and lineage GIF")
    p.add_argument("--confirm", help="instead of searching: re-run a finished run's most novel and random members "
                                     "for 30/100/200 time units (are they really alive?)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int)
    p.add_argument("--out")

    p = sub.add_parser("trace")
    p.add_argument("--run", required=True, help="a survey or explore folder")
    p.add_argument("--rules", required=True, help="comma-separated rule ids (survey) or archive ids (explore)")
    p.add_argument("--seed", type=int, help="survey: which seed of the bank (default: the rule's best)")
    p.add_argument("--out")

    p = sub.add_parser("show")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="a survey folder (uses its rules.csv)")
    src.add_argument("--chan", help="comma-separated catalogue codes, e.g. O2u,S1s (needs paths.catalogue)")
    src.add_argument("--rule", help="a rule as JSON: Rule.to_json() format (as in rules.csv) or Chan's params")
    p.add_argument("--rules", help="with --run: comma-separated rule ids (default: all, or all alive with --alive)")
    p.add_argument("--alive", action="store_true", help="with --run: only rules judged alive")
    p.add_argument("--limit", type=int, default=20, help="with --run and no --rules: at most this many")
    p.add_argument("--seed", type=int, help="start from this seed of the bank (default: the rule's best seed, or 0)")
    p.add_argument("--time", type=float, default=50.0, help="units of time to simulate")
    p.add_argument("--start", type=float, default=10.0, help="skip the first units of time (start-up transients)")
    p.add_argument("--every", type=int, default=2, help="steps per frame")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--px", type=int, help="tile size (default 256 for one clip, 128 for a grid)")
    p.add_argument("--cols", type=int)
    p.add_argument("--full", action="store_true", help="show the whole world instead of following the creature")
    p.add_argument("--cmap", help="override render.cmap")
    p.add_argument("--out", help="output .gif")

    a = ap.parse_args(argv)
    cfg = load(a.config)
    shape, hz, th = _common(cfg)
    rc = cfg["render"]
    runs = resolve(cfg, "runs_dir")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if a.cmd == "orbium":
        import numpy as np
        from . import render, seeds
        from .rle import rle2arr
        from .rollout import run_batch
        from .sim import Rule, place
        d = seeds.donors()["O2u"]
        rule = Rule.from_chan({"R": 13, "T": 10, "b": "1", "m": 0.15, "s": 0.015, "kn": 1, "gn": 1})
        res = run_batch([rule], place(rle2arr(d["cells"]), shape)[None], shape, hz, th, [0])[0]
        out = Path(a.out or runs / "orbium")
        out.mkdir(parents=True, exist_ok=True)
        render.filmstrip(res.film, rule.R, rc["cmap"], rc["px"], rc["crop_r"]).save(out / "orbium.png")
        print(json.dumps(res.stats, indent=2))
        print(f"picture: {out / 'orbium.png'}")
        return 0 if res.stats["class"] == "mover" else 1

    if a.cmd == "catalogue":
        import numpy as np
        from . import catalogue
        path = resolve(cfg, "catalogue")
        if not path.exists():
            print(f"no catalogue at {path}; download it first (see src/lenia/catalogue.py)")
            return 1
        sp = catalogue.load(path)
        if a.n:
            rng = np.random.default_rng(a.seed)
            sp = [sp[i] for i in sorted(rng.choice(len(sp), min(a.n, len(sp)), replace=False))]
        print(f"{len(sp)} species")
        out = Path(a.out or runs / f"catalogue-{a.start}-{stamp}")
        catalogue.run(sp, a.start, out, shape, hz, th, cmap=rc["cmap"], px=rc["px"], crop_r=rc["crop_r"],
                      batch=cfg["run"]["batch"])
        print(f"results: {out}")
        return 0

    if a.cmd == "rejudge":
        from .survey import rejudge
        rejudge(Path(a.run), th)
        return 0

    if a.cmd == "explore" and a.confirm:
        from .explore import confirm
        confirm(Path(a.confirm), th=th, shape=shape)
        return 0

    if a.cmd == "explore" and a.replot:
        from .explore import replot
        cat = None
        if not a.no_catalogue and resolve(cfg, "catalogue").exists():
            from . import catalogue
            cat = catalogue.load(resolve(cfg, "catalogue"))
        replot(Path(a.replot), cat, shape, th, rc["cmap"])
        return 0

    if a.cmd == "explore":
        if not a.from_runs:
            print("explore needs --from (or --replot)")
            return 1
        from .explore import explore, roots_from_survey
        roots = [r for f in a.from_runs.split(",") for r in roots_from_survey(Path(f))]
        cat = None
        path = resolve(cfg, "catalogue")
        if not a.no_catalogue:
            if path.exists():
                from . import catalogue
                cat = catalogue.load(path)
            else:
                print(f"(no catalogue at {path}: skipping the catalogue comparison)")
        name = "-".join(Path(f).name for f in a.from_runs.split(","))
        out = Path(a.out or runs / f"explore-{name}-{stamp}")
        print(f"{len(roots)} roots -> {out}")
        explore(roots, out, cat, shape, th, evals=a.evals, children_per_gen=a.children, child_time=a.child_time,
                confirm_time=a.confirm_time, rescale_time=a.rescale_time, rescale_factor=a.rescale_factor,
                **({"rescale_max_dist": a.rescale_max_dist} if a.rescale_max_dist is not None else {}),
                mut_scale=a.mut_scale, workers=a.workers if a.workers is not None else cfg["run"]["workers"],
                batch=cfg["run"]["batch"], seed=a.seed, cmap=rc["cmap"])
        print(f"results: {out}")
        return 0

    if a.cmd == "trace":
        from .trace import plot
        ids = [int(x) for x in a.rules.split(",")]
        out = Path(a.out or Path(a.run) / f"trace-{a.rules.replace(',', '_')}.png")
        print(f"plot: {plot(Path(a.run), ids, out, a.seed, th)}")
        return 0

    if a.cmd == "show":
        from . import show
        if a.run:
            ids = [int(x) for x in a.rules.split(",")] if a.rules else None
            clips = show.clips_from_run(Path(a.run), ids, a.alive, a.seed, a.limit)
            name = f"{Path(a.run).name}-{'alive' if a.alive and not ids else (a.rules or 'all').replace(',', '_')}"
        elif a.chan:
            path = resolve(cfg, "catalogue")
            if not path.exists():
                print(f"no catalogue at {path}; download it first (see src/lenia/catalogue.py)")
                return 1
            clips = show.clips_from_catalogue(path, a.chan.split(","))
            name = "chan-" + a.chan.replace(",", "_")
        else:
            clips = [show.clip_from_json(a.rule, a.seed or 0)]
            name = "rule"
        if not clips:
            print("nothing to show")
            return 1
        print(f"simulating {len(clips)} clip(s) for {a.time:g} time units ...")
        sims = show.simulate(clips, shape, a.time, a.every, th)
        for lab, rule, _, st in sims:
            print(f"  {lab}: {st['class']}  speed={st['speed_w']:.3f} R/t  change={st['change_w']:.3f}/t  mass={st['mass_end']:.3f} R^2")
        px = a.px or (256 if len(sims) == 1 else 128)
        out = Path(a.out or runs / "show" / f"{name}-{stamp}.gif")
        show.gif(sims, out, px=px, cols=a.cols, cmap=a.cmap or rc["cmap"], follow=not a.full,
                 crop_r=rc["crop_r"], fps=a.fps, start=a.start)
        print(f"gif: {out}")
        return 0

    if a.cmd == "survey":
        from .sample import sample
        from .survey import survey
        w, s = cfg["world"], cfg["sample"]
        rules = sample(a.mode, a.n, a.seed, R=w["R"], T=w["T"], kernels=s["kernels"], channels=s["channels"],
                       self_k=s["self_k"], cross_k=s["cross_k"])
        out = Path(a.out or runs / f"survey-{a.mode}-{stamp}")
        grid_shape = None
        if a.mode == "grid":
            k = int(round(len(rules) ** 0.5))
            grid_shape = (k, k)
        print(f"{len(rules)} rules x {len(__import__('lenia.seeds', fromlist=['x']).DEFAULT_BANK)} seeds -> {out}")
        survey(rules, out, shape=shape, hz=hz, th=th, workers=a.workers if a.workers is not None else cfg["run"]["workers"],
               batch=cfg["run"]["batch"], cmap=rc["cmap"], px=rc["px"], crop_r=rc["crop_r"], rng_seed=a.seed,
               grid_shape=grid_shape)
        print(f"results: {out}")
        return 0
    return 1


def entry() -> int:
    return main(sys.argv[1:])
