"""lenia: is Lenia a good domain for VLM-judged aesthetics?

  orbium     sanity check: Orbium from its catalogued pattern should glide (class 'mover')
  catalogue  run Chan's species from their own pattern (--start own) or from the seed bank (--start bank)
  survey     Test 1 + 2: sample rules, run each from the seed bank, classify, draw contact sheets
             modes: classic | grid | multikernel | multichannel
  rejudge    re-classify a finished survey with the current thresholds (configs/lenia.yaml: alive)
  explore    mobility test: novelty search over rules (no VLM) from a survey's survivors; does it
             move away from where it starts, and outside Chan's catalogue?
  trace      plot rollouts over time (mass, size, speed; thumbnails) from a run's traces.npz
  vlmtest    early VLM test: build a creature set, ask judges (Luna, Jev) about pairs under several
             prompts, rate pairs yourself, report (build | run | rate | report); `run` needs API keys in .env
  guided     VLM-guided search: a judge (Luna, prompt 'curate') accepts mutations it prefers, under the
             alive + resolution constraints (run | check | rate | gif); needs API keys in .env
  bench      check the GPU simulator against numpy and time both (use with --backend auto)
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
    ap.add_argument("--backend", help="numpy | auto | cuda | mps | cpu (default: configs/lenia.yaml run.backend)")
    ap.add_argument("--runs", help="results folder (default: configs/lenia.yaml paths.runs_dir), e.g. on Google Drive")
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
    p.add_argument("--from-found", help="a `lenia found survey` folder: roots = diverse living found/ creatures "
                                        "(own patterns), reference set = all of them")
    p.add_argument("--roots", type=int, default=64, help="with --from-found: how many roots")
    p.add_argument("--evals", type=int, default=3000)
    p.add_argument("--children", type=int, default=32, help="children per generation")
    p.add_argument("--child-time", type=float, default=100.0, help="units of time each child is run for")
    p.add_argument("--confirm-time", type=float, default=200.0, help="archive candidates must also be alive after this")
    p.add_argument("--mut-scale", type=float, default=1.0, help="mutation step size multiplier")
    p.add_argument("--rescale-time", type=float, default=50.0,
                   help="archive candidates (and roots) must also live this long at a finer grid (0 = no check)")
    p.add_argument("--rescale-factor", type=int, default=2, help="how much finer: R -> factor*R, pattern upsampled")
    p.add_argument("--rescale-max-dist", type=float, help="... and stay within this descriptor distance of itself "
                                                          "(default: from --from-found's calibration.json, else "
                                                          "calibrated on the catalogue, see explore.RESCALE_MAX_DIST)")
    p.add_argument("--texture-max-loss", type=float, help="at the 2x check, also reject creatures that lose more fine "
                                                          "structure than this (default: from --from-found's "
                                                          "calibration.json, else off; negative = off)")
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

    p = sub.add_parser("vlmtest")
    p.add_argument("action", choices=["build", "run", "rate", "report"])
    p.add_argument("--out", help="test folder (default runs/vlmtest)")
    p.add_argument("--judges", default="luna,jev", help="comma-separated names from configs/lenia.yaml: judges")
    p.add_argument("--prompts", default=",".join(["appeal", "appeal_bare", "beautiful", "curate", "interesting", "lifelike"]),
                   help="comma-separated prompt ids (src/lenia/prompts.py); default: the prompt sweep")
    p.add_argument("--n", type=int, default=60, help="rate: how many pairs you rate")
    p.add_argument("--port", type=int, default=8765, help="rate: local port")
    p.add_argument("--limit", type=int, help="run: only the first N pairs (smoke test)")
    p.add_argument("--yes", action="store_true", help="run: proceed even if the estimate exceeds the budget")
    p.add_argument("--verbose", action="store_true", help="run: print every failed call")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("guided")
    p.add_argument("action", choices=["run", "check", "rate", "gif"])
    p.add_argument("--out", help="run folder (default runs/guided)")
    p.add_argument("--from", dest="from_run", help="explore run to take founders from (default runs/explore-multikernel)")
    p.add_argument("--lineages", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--judge", help="driving judge (configs/lenia.yaml: judges)")
    p.add_argument("--prompt")
    p.add_argument("--threshold", type=float)
    p.add_argument("--heldout", help="check: comma-separated judges (default: driver + held-out)")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("found")                 # Chan's found/ collections (multi-kernel, multi-channel)
    p.add_argument("action", choices=["download", "survey"])
    p.add_argument("--files", help="comma-separated file names (default: found.DEFAULT)")
    p.add_argument("--n-per-file", type=int, default=0, help="survey: random subset per file (0 = all)")
    p.add_argument("--target-R", type=float, help="survey: bring every creature to this kernel radius (default: the "
                                                   "config's world.R, 20 in configs/found.yaml; 0 = keep Chan's own R)")
    p.add_argument("--no-calibrate", action="store_true", help="survey: skip the 2x calibration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")

    p = sub.add_parser("bench")                 # GPU check: correctness against numpy, then speed

    a = ap.parse_args(argv)
    cfg = load(a.config)
    from . import backend
    backend.configure(a.backend or cfg["run"].get("backend", "numpy"))
    if a.runs:
        cfg["paths"]["runs_dir"] = a.runs
    if backend.is_gpu():                        # one process, big batches (worker processes would each grab the GPU)
        cfg["run"]["workers"] = 1
        cfg["run"]["batch"] = int(cfg["run"].get("batch_gpu", 64))
        if getattr(a, "workers", None):
            a.workers = 1
    if a.cmd != "bench":
        print(f"[lenia] backend: {backend.info()}", file=sys.stderr)
    shape, hz, th = _common(cfg)
    rc = cfg["render"]
    runs = resolve(cfg, "runs_dir")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if a.cmd == "found":
        from . import found
        names = a.files.split(",") if a.files else found.DEFAULT
        folder = resolve(cfg, "catalogue").parent / "found"
        if a.action == "download":
            found.download(names, folder)
            print(f"files in {folder}")
            return 0
        import numpy as np
        cs = found.load(folder, names, max_size=min(shape) // 2, target_R=(cfg["world"]["R"] if a.target_R is None else a.target_R) or None)
        if a.n_per_file:
            rng = np.random.default_rng(a.seed)
            keep = []
            for n in names:
                idx = [i for i, c in enumerate(cs) if c["file"] == n]
                keep += [idx[i] for i in sorted(rng.choice(len(idx), min(a.n_per_file, len(idx)), replace=False))]
            cs = [cs[i] for i in sorted(keep)]
        out = Path(a.out or runs / "found-survey")
        from concurrent.futures import ProcessPoolExecutor
        w = cfg["run"]["workers"] or max(1, (__import__("os").cpu_count() or 2) - 1)
        ex = ProcessPoolExecutor(w) if w > 1 else None
        print(f"{len(cs)} creatures -> {out}")
        found.survey(cs, out, shape, th, batch=cfg["run"]["batch"], ex=ex, calibrate=not a.no_calibrate)
        if ex:
            ex.shutdown()
        return 0

    if a.cmd == "bench":
        from .bench import run as bench
        return 0 if bench() else 1

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
        if not a.from_runs and not a.from_found:
            print("explore needs --from or --from-found (or --replot)")
            return 1
        from .explore import explore, roots_from_survey
        ref_desc = None
        checks = {}
        if a.from_found:
            from .found import calibration, roots_from_survey as found_roots
            roots, ref_desc = found_roots(Path(a.from_found), a.roots, seed=a.seed)
            a.no_catalogue = True                   # the found creatures are the reference set
            cal = calibration(Path(a.from_found))
            if cal:
                if cal.get("rescale_max_dist") is not None:
                    checks["rescale_max_dist"] = cal["rescale_max_dist"]
                checks["texture_max_loss"] = cal.get("texture_max_loss")
                print(f"checks calibrated on {a.from_found}: 2x max distance {checks.get('rescale_max_dist')}, "
                      f"max texture loss {checks.get('texture_max_loss')}")
        else:
            roots = [r for f in a.from_runs.split(",") for r in roots_from_survey(Path(f))]
        if a.rescale_max_dist is not None:
            checks["rescale_max_dist"] = a.rescale_max_dist
        if a.texture_max_loss is not None:
            checks["texture_max_loss"] = a.texture_max_loss if a.texture_max_loss >= 0 else None
        cat = None
        path = resolve(cfg, "catalogue")
        if not a.no_catalogue:
            if path.exists():
                from . import catalogue
                cat = catalogue.load(path)
            else:
                print(f"(no catalogue at {path}: skipping the catalogue comparison)")
        name = "-".join(Path(f).name for f in (a.from_runs or a.from_found).split(","))
        out = Path(a.out or runs / f"explore-{name}-{stamp}")   # give --out to resume an interrupted run
        print(f"{len(roots)} roots -> {out}")
        explore(roots, out, cat, shape, th, evals=a.evals, children_per_gen=a.children, child_time=a.child_time,
                confirm_time=a.confirm_time, rescale_time=a.rescale_time, rescale_factor=a.rescale_factor,
                **checks, mut_scale=a.mut_scale, workers=a.workers if a.workers is not None else cfg["run"]["workers"],
                batch=cfg["run"]["batch"], seed=a.seed, cmap=rc["cmap"], ref_desc=ref_desc)
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

    if a.cmd == "guided":
        from . import guided
        g = cfg["guided"]
        out = Path(a.out or runs / "guided")
        if a.action == "rate":
            from .vlmtest import rate
            if not (out / "human_pairs.json").exists():
                print("run `lenia guided check` first (it picks the pairs to rate)")
                return 1
            rate(out, port=8765)
            return 0
        if a.action == "gif":
            print(f"gif: {guided.gif(out, shape)}")
            return 0
        from .config import REPO_ROOT
        from .judge import Judge, load_env
        load_env(REPO_ROOT / ".env")
        jc = cfg["judges"]
        mk = lambda n: Judge(n, {**jc.get("defaults", {}), **jc[n]}, resolve(cfg, "judge_cache"), yes=a.yes)
        prompt = a.prompt or g["prompt"]
        if a.action == "run":
            import numpy as np
            src = Path(a.from_run or runs / g["founders_from"])
            n_lin = a.lineages or g["lineages"]
            founders = guided.pick_founders(src, n_lin + 8, seed=a.seed)     # spares: some may fail the 2x check now
            guided.run(out, founders, mk(a.judge or g["judge"]), prompt, np.load(src / "catalogue_desc.npy"),
                       generations=a.generations or g["generations"], threshold=a.threshold or g["threshold"],
                       mut_scale=g.get("mut_scale", 1.0), seed=a.seed, shape=shape,
                       workers=a.workers if a.workers is not None else cfg["run"]["workers"], th=th,
                       n_lineages=n_lin, attempts=g.get("attempts", 3))
            try:
                print(f"gif: {guided.gif(out, shape)}")
            except Exception as e:          # pictures are a bonus; never lose a run over them
                print(f"(gif failed: {e})")
            print(f"done: {out}; next: lenia guided check")
            return 0
        if a.action == "check":
            import csv as _csv
            js = __import__("json").loads((out / "state.json").read_text())
            names = (a.heldout or f"{js['config']['judge']},{g['heldout']}").split(",")
            vt = runs / "vlmtest"
            cat = [(r["id"], vt / "images" / f"{r['id']}.png") for r in _csv.DictReader(open(vt / "creatures.csv"))
                   if r["group"] == "catalogue"]
            guided.check(out, {n: mk(n) for n in names}, js["config"]["prompt"], cat, seed=a.seed)
            return 0

    if a.cmd == "vlmtest":
        from . import vlmtest
        out = Path(a.out or runs / "vlmtest")
        if a.action == "build":
            from . import catalogue
            cat = catalogue.load(resolve(cfg, "catalogue"), max_R=36, max_size=128)
            vlmtest.build(out, runs, cat, seed=a.seed)
            print(f"built: {out} (look at sheet_*.png before running the judges)")
            return 0
        if a.action == "rate":
            vlmtest.rate(out, n=a.n, port=a.port)
            return 0
        if a.action == "run":
            from .config import REPO_ROOT
            from .judge import Judge, load_env
            load_env(REPO_ROOT / ".env")
            jc = cfg["judges"]
            judges = {n: Judge(n, {**jc.get("defaults", {}), **jc[n]}, resolve(cfg, "judge_cache"), yes=a.yes,
                               verbose=a.verbose) for n in a.judges.split(",")}
            path = vlmtest.run(out, judges, a.prompts.split(","), limit=a.limit)
            print(f"verdicts: {path}; total spent ${sum(j.spent for j in judges.values()):.4f}")
            if a.limit:
                return 0
        vlmtest.report(out)
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
