"""Mobility test: can a simple search move through living-creature space, away from where it starts?

Novelty search over rules, no VLM. Each child is a mutated copy of an archive member's rule, started
from that member's final creature (the pattern carries over, as in Chan's own interactive search), so a
lineage can morph gradually through the thin viable bands instead of re-growing from scratch. A child
that is alive after child_time (default 100 units) and novel (its mean descriptor distance to its 5
nearest archive members exceeds a threshold that adapts so that roughly 5-25% of living children get
in) is a candidate; candidates are re-run to confirm_time (default 200), then run once more at double
resolution (rule R -> 2R, creature upsampled; rescale_time, default 50) and join the archive only if
alive in both and still novel. The archive keeps the confirmed run's final creature. The resolution
check removes 'speckle': patterns whose structure lives at the scale of single cells, which novelty
search otherwise favours and which freeze or die on a finer grid. Roots get the same check.

Measured: how often mutants stay alive; how far lineages drift from their roots (descriptor and
parameter space); and how much of the archive lies outside the spread of Chan's catalogue, as
measured by these descriptors (a crude proxy for 'new'; a learned image embedding would be better).
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import render, seeds as seedbank
from .alive import ALIVE, Thresholds, centroid, recentre
from .describe import NAMES, Space, bins, describe
from .rollout import Horizon, run_batch
from .sim import Kernel, Rule, place


# ---------- mutation ----------

def _twelfths(x: float) -> float:
    return round(min(1.0, max(0.0, x)) * 12) / 12


def mutate_kernel(k: Kernel, rng, multi: bool, scale: float = 1.0) -> Kernel:
    k = copy.deepcopy(k)
    k.m = float(np.clip(k.m + rng.normal(0, 0.01 * scale), 0.05, 0.6))
    k.s = float(np.clip(k.s * np.exp(rng.normal(0, 0.08 * scale)), 0.003, 0.2))
    b = list(k.b)
    u = rng.random()
    if u < 0.15:                                            # nudge one ring by a twelfth
        i = int(rng.integers(len(b)))
        b[i] = _twelfths(b[i] + rng.choice([-1, 1]) / 12)
    elif u < 0.20 and len(b) < 3:                           # add a ring
        b.insert(int(rng.integers(len(b) + 1)), _twelfths(rng.random()))
    elif u < 0.25 and len(b) > 1:                           # drop a ring
        b.pop(int(rng.integers(len(b))))
    if max(b) < 1:
        b[int(np.argmax(b))] = 1.0
    k.b = tuple(b)
    if multi:
        k.r = float(np.clip(k.r + rng.normal(0, 0.03 * scale), 0.3, 1.0))
        k.h = float(np.clip(k.h + rng.normal(0, 0.05 * scale), 0.05, 1.0))
    return k


def mutate(rule: Rule, rng, scale: float = 1.0) -> Rule:
    multi = len(rule.kernels) > 1
    ks = [mutate_kernel(k, rng, multi, scale) if (not multi or rng.random() < 0.5) else copy.deepcopy(k)
          for k in rule.kernels]
    return Rule(ks, R=rule.R, T=rule.T, channels=rule.channels, kn=rule.kn, gn=rule.gn)


def param_vector(rule: Rule) -> np.ndarray:
    """m, log s, mean ring height, number of rings (+ r, h) per kernel: for 'how far did the rule move'."""
    v = []
    for k in rule.kernels:
        v += [k.m / 0.05, np.log(k.s) / 0.3, float(np.mean(k.b)) / 0.25, len(k.b)]
        if len(rule.kernels) > 1:
            v += [k.r / 0.1, k.h / 0.2]
    return np.array(v, np.float32)


# ---------- evaluation (in worker processes) ----------

def crop_pattern(A: np.ndarray, R: float, margin_r: float = 0.6) -> np.ndarray:
    """The creature cut out of the world (C, n, n), re-centred, with a margin, as float16."""
    A = np.asarray(A, np.float32)
    n = render.crop_size([(0, A)], R, 0, min(A.shape[-2:]))
    n = int(min(min(A.shape[-2:]), n + margin_r * R))
    return render.crop(A, R, 0, n).astype(np.float16)


def _eval(args):
    rule_dicts, inits, shape, hz, th = args
    rules = [Rule.from_dict(d) for d in rule_dicts]
    A0 = np.stack([place(np.asarray(p, np.float32), shape, r.channels) for p, r in zip(inits, rules)])
    res = run_batch(rules, A0, shape, hz, th, [0] * len(rules))
    out = []
    for r, rule in zip(res, rules):
        final = np.asarray(r.film[-1][1], np.float32)
        alive = r.stats["class"] in ALIVE
        d = None
        if alive:   # average over the filmstrip frames so a pulser's phase does not count as novelty
            ds = np.array([describe(np.asarray(f, np.float32), r.stats, rule.R) for _, f in r.film])
            d = ds.mean(0)
            r.stats["desc_half_gap"] = ds   # kept for the noise-floor estimate (roots only)
        r.stats["_trace"] = np.clip(np.nan_to_num(r.trace, nan=np.nan, posinf=6e4), -6e4, 6e4).astype(np.float16)
        out.append((r.stats, crop_pattern(final, rule.R) if alive else None, d))
    return out


def evaluate(rules: list, inits: list, shape, hz: Horizon, th: Thresholds, ex, batch: int = 16) -> list:
    groups = {}
    for i, r in enumerate(rules):
        groups.setdefault((r.structure(), r.T), []).append(i)
    chunks = [g[k:k + batch] for g in groups.values() for k in range(0, len(g), batch)]
    args = [([rules[i].to_dict() for i in ch], [inits[i] for i in ch], tuple(shape), hz, th) for ch in chunks]
    res = list(ex.map(_eval, args)) if ex else [_eval(a) for a in args]
    out = [None] * len(rules)
    for ch, rr in zip(chunks, res):
        for i, x in zip(ch, rr):
            out[i] = x
    return out


# ---------- coherence: does it survive at a finer grid? ----------

# largest descriptor distance between a creature and its 2x-resolution self that still counts as 'the
# same creature'. The p95 over Chan's catalogue species alive at both scales (194 of 202; median 0.68,
# p99 2.59), from scripts/calibrate_rescale.py. For scale: neighbouring catalogue species are a
# median 1.44 apart, and a creature vs itself 5 time units later about 0.5-0.8.
RESCALE_MAX_DIST = 1.98

def upsample(P: np.ndarray, f: int = 2) -> np.ndarray:
    """(C, h, w) -> (C, f*h, f*w) by bilinear interpolation (cell centres aligned), values kept in [0, 1]."""
    P = np.asarray(P, np.float32)
    P = P[None] if P.ndim == 2 else P

    def along(P, ax):
        n = P.shape[ax]
        x = np.clip((np.arange(n * f) + 0.5) / f - 0.5, 0, n - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        w = (x - i0).reshape([-1 if a == ax % P.ndim else 1 for a in range(P.ndim)])
        return np.take(P, i0, axis=ax) * (1 - w) + np.take(P, i1, axis=ax) * w

    return np.clip(along(along(P, -2), -1), 0, 1)


def rescaled(rule: Rule, f: int = 2) -> Rule:
    """The same rule on an f-times finer grid: kernel radius R -> f*R (rings, widths, T unchanged)."""
    return Rule(copy.deepcopy(rule.kernels), R=rule.R * f, T=rule.T, channels=rule.channels, kn=rule.kn, gn=rule.gn)


def rescale_check(rules: list, patterns: list, shape, th: Thresholds, ex, batch: int = 16, f: int = 2,
                  time: float = 50.0) -> list:
    """Run each creature at f-times the resolution (rule rescaled, pattern upsampled, world f-times larger)
    and return [(class, descriptor or None)] there. A Lenia creature is a continuum object: it should
    survive and look and move the same (descriptors are in units of R). Structure that lives at the scale
    of single cells (speckle) does not: it freezes, dies, or relaxes into a different, smoother creature."""
    if not rules:
        return []
    big = tuple(int(s * f) for s in shape)
    hz = Horizon(time=time, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    res = evaluate([rescaled(r, f) for r in rules], [upsample(p, f) for p in patterns], big, hz, th, ex, batch)
    return [(st["class"], d) for st, _, d in res]


# ---------- the search ----------

def roots_from_survey(run: Path) -> list:
    """[(label, rule, seed_index)] for every rule judged alive in a survey (summary_rejudged.json)."""
    run = Path(run)
    rows = {int(r["rule"]): r for r in csv.DictReader(open(run / "rules.csv"))}
    rej = run / "summary_rejudged.json"
    ids = json.loads(rej.read_text())["alive_rule_ids"] if rej.exists() else [i for i, r in rows.items() if int(r["alive_seeds"])]
    return [(f"{run.name}#{i}", Rule.from_dict(json.loads(rows[i]["json"])), int(rows[i]["best_seed"])) for i in ids]


def _cells3(c) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return c if c.ndim == 3 else c[None]


def _root_init(j, rule: Rule, shape) -> np.ndarray:
    """A root starts from a seed of the bank (j = index) or from its own pattern (j = array)."""
    if isinstance(j, np.ndarray):
        return _cells3(j)
    return seedbank.make(seedbank.DEFAULT_BANK[j], rule.R, shape, rule.channels)


def coherent_with(space, d1, res2, max_dist):
    c, d2 = res2
    if c not in ALIVE or d2 is None:
        return False, np.inf
    dist = float(space.dist(d1, np.asarray(d2)[None])[0])
    return dist <= max_dist, dist


def _setup(roots, catalogue, shape, hz_root, hz_child, th, ex, batch, rescale_time, rescale_factor, ref_desc,
           coherent_factory, log):
    """Roots (alive and, with rescale_time, coherent) and the descriptor space (reference set + roots)."""
    rr = evaluate([r for _, r, _ in roots], [_root_init(j, r, shape) for _, r, j in roots], shape, hz_root, th, ex, batch)
    arch = []
    for (lab, rule, j), (st, pat, d) in zip(roots, rr):
        if pat is None:
            continue
        arch.append({"id": len(arch), "parent": -1, "root": len(arch), "gen": 0, "label": lab, "rule": rule,
                     "trace": st.pop("_trace", None), "init": _root_init(j, rule, shape), "final": pat,
                     "stats": st, "desc": d, "novelty": np.nan, "rescale_dist": np.nan})
    n_roots_alive = len(arch)
    root_scale = (rescale_check([a["rule"] for a in arch], [a["final"] for a in arch], shape, th, ex, batch,
                                rescale_factor, rescale_time) if rescale_time and arch else None)
    cat_desc = np.zeros((0, len(NAMES)), np.float32)
    if ref_desc is not None:
        cat_desc = np.asarray(ref_desc, np.float32).reshape(-1, len(NAMES))
        log(f"reference set: {len(cat_desc)} precomputed descriptors")
    elif catalogue:
        cr = evaluate([s["rule"] for s in catalogue], [_cells3(s["cells"]) for s in catalogue], shape, hz_child, th, ex, batch)
        cat_desc = np.array([d for _, _, d in cr if d is not None], np.float32).reshape(-1, len(NAMES))
        log(f"reference set: {len(cat_desc)} of {len(catalogue)} alive and described")
    for a in arch:      # roots: noise floor = distance between the descriptor over [40, 45] and over [45, 50]
        ds = a["stats"].pop("desc_half_gap")
        a["desc_halves"] = (ds[:6].mean(0), ds[6:].mean(0))
    if not arch:
        raise SystemExit("no living roots")
    space = Space(np.concatenate([cat_desc, np.array([a["desc"] for a in arch])]) if len(cat_desc) else
                  np.array([a["desc"] for a in arch]))
    coherent = coherent_factory(space)
    if root_scale is not None:
        keep = []
        for a, r2 in zip(arch, root_scale):
            ok, a["rescale_dist"] = coherent(a["desc"], r2)
            a["stats"]["rescale_class"] = r2[0]
            if ok:
                keep.append(a)
        for i, a in enumerate(keep):
            a["id"] = a["root"] = i
        arch = keep
    log(f"roots: {n_roots_alive} of {len(roots)} alive under current thresholds"
        + (f", {len(arch)} of them also coherent at {rescale_factor}x resolution" if rescale_time else ""))
    if not arch:
        raise SystemExit("no living roots that are coherent at the finer grid")
    space.cat_desc, space.n_roots_alive = cat_desc, n_roots_alive
    return arch, space


def _save_checkpoint(path: Path, state: dict) -> None:
    import pickle
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict | None:
    import pickle
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def explore(roots: list, out: Path, catalogue: list | None = None, shape=(128, 128), th: Thresholds = None,
            evals: int = 3000, children_per_gen: int = 32, child_time: float = 100.0, confirm_time: float = 200.0, mut_scale: float = 1.0,
            rescale_time: float = 50.0, rescale_factor: int = 2, rescale_max_dist: float = RESCALE_MAX_DIST,
            k: int = 5, workers: int = 0, batch: int = 16, seed: int = 0, cmap: str = "chan",
            checkpoint_every_s: float = 120.0, ref_desc: np.ndarray | None = None, log=print) -> dict:
    """ref_desc: descriptors of a reference set (e.g. Chan's found/ creatures from `lenia found survey`), used
    instead of simulating `catalogue` for the descriptor scale and the 'beyond the reference' measures."""
    th = th or Thresholds()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    t0 = time.time()
    hz_child = Horizon(time=child_time, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    hz_confirm = Horizon(time=confirm_time, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    n_late = [0]
    n_rescale_fail = [0]
    rescale_cost = rescale_time * rescale_factor ** 2 / child_time    # one check, in child-equivalents
    hz_root = Horizon(time=max(child_time, 50.0), record_every=0.5, frames=12, film_span=10.0, thumb_every=0)

    def coherent(d1, res2):
        """Alive at the finer grid, and the same creature there: descriptor distance <= rescale_max_dist."""
        c, d2 = res2
        if c not in ALIVE or d2 is None:
            return False, np.inf
        dist = float(space.dist(d1, np.asarray(d2)[None])[0])
        return dist <= rescale_max_dist, dist

    ckpt = out / "checkpoint.pkl"
    state = _load_checkpoint(ckpt)
    if state is not None:
        g = state
        arch, hist, n_eval, gen, alive_children = g["arch"], g["hist"], g["n_eval"], g["gen"], g["alive_children"]
        tau, noise, nn0, cat_desc, space, rng = g["tau"], g["noise"], g["nn0"], g["cat_desc"], g["space"], g["rng"]
        n_late, n_rescale_fail, n_roots, n_roots_alive = g["n_late"], g["n_rescale_fail"], g["n_roots"], g["n_roots_alive"]
        t0 -= g["elapsed"]
        log(f"resuming from {ckpt.name}: generation {gen}, {round(n_eval)} evals, archive {len(arch)}")
    else:
        arch, space = _setup(roots, catalogue, shape, hz_root, hz_child, th, ex, batch, rescale_time, rescale_factor, ref_desc,
                             coherent_factory=lambda sp: (lambda d1, r2: coherent_with(sp, d1, r2, rescale_max_dist)), log=log)
        cat_desc = space.cat_desc
        n_roots_alive, n_roots = space.n_roots_alive, len(arch)
        D = lambda: np.array([a["desc"] for a in arch])
        nn0 = [np.sort(space.dist(a["desc"], D()))[1] for a in arch] if n_roots > 1 else [1.0]
        tau = 0.5 * float(np.median(nn0))
        noise = float(np.median([space.dist(a["desc_halves"][0], a["desc_halves"][1][None])[0] for a in arch]))
        log(f"initial novelty threshold {tau:.2f}; noise floor (a root vs itself 5 time units later) {noise:.2f}")
        hist, n_eval, gen, alive_children = [], 0, 0, []
    last_save = time.time()

    def save():
        _save_checkpoint(ckpt, {"arch": arch, "hist": hist, "n_eval": n_eval, "gen": gen, "alive_children": alive_children,
                                "tau": tau, "noise": noise, "nn0": nn0, "cat_desc": cat_desc, "space": space, "rng": rng,
                                "n_late": n_late, "n_rescale_fail": n_rescale_fail, "n_roots": n_roots,
                                "n_roots_alive": n_roots_alive, "elapsed": time.time() - t0})
    if state is None:
        save()
    D = lambda: np.array([a["desc"] for a in arch])
    # 3. novelty search (alive_children: every living child, archived or not, for 'where does mutation go')
    while n_eval < evals:
        gen += 1
        parents = [arch[i] for i in rng.integers(len(arch), size=children_per_gen)]
        kids = [mutate(p["rule"], rng, mut_scale) for p in parents]
        res = evaluate(kids, [p["final"] for p in parents], shape, hz_child, th, ex, batch)
        n_eval += len(kids)
        n_alive = n_added = n_cand = 0
        classes = {}
        cands = []
        for p, kr, (st, pat, d) in zip(parents, kids, res):
            classes[st["class"]] = classes.get(st["class"], 0) + 1
            st.pop("desc_half_gap", None)
            st.pop("_trace", None)
            if pat is None:
                continue
            n_alive += 1
            alive_children.append({"desc": d, "root": p["root"], "gen": gen})
            if space.knn_mean(d, D(), k) >= tau:
                cands.append((p, kr))
        # second stage: candidates must also be alive after confirm_time (catches late failures);
        # the archive keeps the confirmed run's final creature, statistics and descriptor
        n_cand = len(cands)
        n_failed_scale = 0
        if cands:
            conf = evaluate([kr for _, kr in cands], [p["final"] for p, _ in cands], shape, hz_confirm, th, ex, batch)
            # third stage: the confirmed creature must also live at a finer grid (not speckle)
            ok = [i for i, (_, pat, _) in enumerate(conf) if pat is not None]
            scl = [None] * len(cands)
            if rescale_time:
                for i, c in zip(ok, rescale_check([cands[i][1] for i in ok], [conf[i][1] for i in ok], shape, th, ex,
                                                  batch, rescale_factor, rescale_time)):
                    scl[i] = c
                n_eval += len(ok) * rescale_cost
            for (p, kr), (st, pat, d), sc in zip(cands, conf, scl):
                st.pop("desc_half_gap", None)
                trace = st.pop("_trace", None)
                if pat is None:
                    n_late[0] += 1
                    continue
                rdist = np.nan
                if rescale_time:
                    ok_, rdist = coherent(d, sc)
                    st["rescale_class"] = sc[0]
                    if not ok_:
                        n_rescale_fail[0] += 1
                        n_failed_scale += 1
                        continue
                nov = space.knn_mean(d, D(), k)
                if nov >= tau:
                    arch.append({"id": len(arch), "parent": p["id"], "root": p["root"], "gen": gen, "label": "",
                                 "trace": trace, "rule": kr, "init": p["final"], "final": pat, "stats": st, "desc": d,
                                 "novelty": nov, "rescale_dist": rdist})
                    n_added += 1
            n_eval += len(cands) * confirm_time / child_time     # count confirmation runs in child-equivalents
        # keep roughly 5-25% of living children
        if n_alive and n_added > 0.25 * n_alive:
            tau *= 1.1
        elif n_added == 0:
            tau *= 0.95
        hist.append({"gen": gen, "evals": round(n_eval), "alive": n_alive, "candidates": n_cand, "added": n_added,
                     "failed_rescale": n_failed_scale,
                     "archive": len(arch),
                     "tau": round(tau, 3), "classes": classes})
        if time.time() - last_save > checkpoint_every_s or n_eval >= evals:
            save()
            last_save = time.time()
        if gen % 10 == 0 or n_eval >= evals:
            log(f"  gen {gen}: {round(n_eval)} evals, alive {n_alive}/{len(kids)}, candidates {n_cand}, failed {rescale_factor}x {n_failed_scale}, added {n_added}, archive {len(arch)}, "
                f"tau {tau:.2f}, {time.time() - t0:.0f}s")
    if ex:
        ex.shutdown()

    # 4. metrics
    Dall = D()
    root_desc = {a["root"]: a["desc"] for a in arch if a["parent"] < 0}
    root_par = {a["root"]: param_vector(a["rule"]) for a in arch if a["parent"] < 0}
    for a in arch:
        a["dist_root"] = float(space.dist(a["desc"], root_desc[a["root"]][None])[0])
        pv, rv = param_vector(a["rule"]), root_par[a["root"]]
        a["param_root"] = float(np.linalg.norm(pv - rv)) if pv.shape == rv.shape else np.nan
        a["dist_other_roots"] = space.nearest(a["desc"], np.array([root_desc[r] for r in root_desc if r != a["root"]])) if n_roots > 1 else np.nan
        a["dist_cat"] = space.nearest(a["desc"], cat_desc) if len(cat_desc) else np.nan
    summary = {"roots": n_roots, "evals": round(n_eval), "generations": gen, "archive": len(arch),
               "children_alive_fraction": round(sum(h["alive"] for h in hist) / max(1, gen * children_per_gen), 4),
               "archived_fraction_of_alive": round((len(arch) - n_roots) / max(1, len(alive_children)), 4),
               "seconds": round(time.time() - t0, 1), "child_time": child_time, "mut_scale": mut_scale,
               "noise_floor": round(noise, 2), "initial_tau": round(0.5 * float(np.median(nn0)), 2),
               "confirm_time": confirm_time, "candidates": int(sum(h["candidates"] for h in hist)),
               "candidates_failed_confirmation": int(n_late[0]),
               "rescale": {"factor": rescale_factor, "time": rescale_time, "max_dist": rescale_max_dist, "roots_alive": n_roots_alive,
                           "roots_failed": n_roots_alive - n_roots, "candidates_failed": int(n_rescale_fail[0])}
               if rescale_time else None}
    kids_ = [a for a in arch if a["parent"] >= 0]
    if kids_:
        dr = np.array([a["dist_root"] for a in kids_])
        summary["dist_from_root"] = {"median": round(float(np.median(dr)), 2), "p90": round(float(np.percentile(dr, 90)), 2),
                                     "max": round(float(dr.max()), 2)}
        pr = np.array([a["param_root"] for a in kids_])
        summary["param_dist_from_root"] = {"median": round(float(np.nanmedian(pr)), 2), "max": round(float(np.nanmax(pr)), 2)}
        # roots per lineage: how many lineages produced anything
        summary["lineages_with_descendants"] = len({a["root"] for a in kids_})
    if len(cat_desc) > 1:
        # how far is a creature from its nearest catalogue species, compared with how far catalogue
        # species are from each other? (the catalogue's own nearest-neighbour distances, as quantiles)
        cat_nn = np.array([np.sort(space.dist(c, cat_desc))[1] for c in cat_desc])
        qs = {q: float(np.percentile(cat_nn, q)) for q in (50, 75, 90)}
        frac = lambda xs, v: round(float(np.mean([x["dist_cat"] > v for x in xs])), 3) if xs else None
        rts = [a for a in arch if a["parent"] < 0]
        summary["catalogue"] = {"species_described": int(len(cat_desc)),
                                "nn_quantiles": {f"p{q}": round(v, 2) for q, v in qs.items()},
                                "roots_beyond": {f"p{q}": frac(rts, v) for q, v in qs.items()},
                                "children_beyond": {f"p{q}": frac(kids_, v) for q, v in qs.items()}}
    cov = lambda ds: {bins(d) for d in ds}
    c_roots, c_arch = cov([a["desc"] for a in arch if a["parent"] < 0]), cov(Dall)
    c_cat = cov(cat_desc)
    summary["coverage_cells"] = {"roots": len(c_roots), "archive": len(c_arch), "catalogue": len(c_cat),
                                 "archive_not_in_catalogue": len(c_arch - c_cat) if len(cat_desc) else None,
                                 "archive_not_in_roots": len(c_arch - c_roots)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "history.json").write_text(json.dumps(hist))

    # 5. files: archive table, patterns, pictures
    with open(out / "archive.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "parent", "root", "gen", "label", "class", "speed_w", "change_w", "mass_end", "novelty",
                    "dist_root", "param_root", "dist_cat", "rescale_class", "rescale_dist", "rule"] + NAMES)
        for a in arch:
            s = a["stats"]
            w.writerow([a["id"], a["parent"], a["root"], a["gen"], a["label"], s["class"], s["speed_w"], s["change_w"],
                        s["mass_end"], a["novelty"], a["dist_root"], a["param_root"], a["dist_cat"], s.get("rescale_class", ""), a.get("rescale_dist", np.nan),
                        a["rule"].to_json()]
                       + [float(x) for x in a["desc"]])
    np.save(out / "catalogue_desc.npy", cat_desc)
    np.savez_compressed(out / "traces.npz", **{str(a["id"]): a["trace"] for a in arch if a.get("trace") is not None})
    np.savez_compressed(out / "patterns.npz", **{f"init_{a['id']}": np.asarray(a["init"], np.float16) for a in arch},
                        **{f"final_{a['id']}": a["final"] for a in arch})
    _pictures(arch, alive_children, cat_desc, space, hist, out, cmap, shape)
    log(json.dumps(summary))
    return summary


def _tile(pat, R, cmap, px=96):
    return render.frame_image(np.asarray(pat, np.float32), R, cmap, px, crop_r=0,
                              n=max(np.asarray(pat).shape[-2:]))


def _pictures(arch, alive_children, cat_desc, space, hist, out, cmap, shape):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = arch[0]["rule"].R
    # most distant from the catalogue (or from their root when there is no catalogue)
    kids = [a for a in arch if a["parent"] >= 0]
    key = (lambda a: a["dist_cat"]) if len(cat_desc) else (lambda a: a["dist_root"])
    top = sorted(kids, key=key, reverse=True)[:40]
    items = [(f"#{a['id']} {a['dist_cat']:.1f}", _tile(a["final"], R, cmap)) for a in top]
    render.sheet(items, cols=5, title="farthest from Chan's catalogue (label: id, descriptor distance)").save(out / "sheet_novel.png")
    roots = [a for a in arch if a["parent"] < 0]
    render.sheet([(f"root {a['label']}", _tile(a["final"], R, cmap)) for a in roots], cols=6,
                 title="roots (survey survivors)").save(out / "sheet_roots.png")
    # lineages: root -> ... -> its farthest descendant (up to 7 steps shown)
    by_id = {a["id"]: a for a in arch}
    strips = []
    for r in roots:
        desc = [a for a in kids if a["root"] == r["id"]]
        if not desc:
            continue
        leaf = max(desc, key=lambda a: a["dist_root"])
        path = [leaf]
        while path[-1]["parent"] >= 0:
            path.append(by_id[path[-1]["parent"]])
        path = path[::-1]
        idx = np.unique(np.linspace(0, len(path) - 1, min(7, len(path))).round().astype(int))
        tiles = [_tile(path[i]["final"], R, cmap, 80) for i in idx]
        from PIL import Image
        strip = Image.new("RGB", (len(tiles) * 82, 80), (230, 230, 230))
        for i, t in enumerate(tiles):
            strip.paste(t, (i * 82, 0))
        strips.append((leaf["dist_root"], f"root {r['label']} -> #{leaf['id']} ({len(path) - 1} steps, d_root={leaf['dist_root']:.1f})", strip))
    strips.sort(key=lambda x: -x[0])
    if strips:
        render.sheet([(lab, im) for _, lab, im in strips[:12]], cols=1, title="lineages: root to farthest descendant").save(out / "sheet_lineages.png")
    # descriptor map: PCA of catalogue + roots + archive
    allD = [cat_desc] if len(cat_desc) else []
    Dr = np.array([a["desc"] for a in roots])
    Dk = np.array([a["desc"] for a in kids]) if kids else np.zeros((0, Dr.shape[1]))
    med = np.median(np.concatenate(allD + [Dr]), 0)
    zc = lambda X: np.clip((np.asarray(X) - med) / space.scale, -4, 4)
    Z = zc(np.concatenate(allD + [Dr, Dk]))
    mu = Z.mean(0)
    U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    proj = lambda X: ((zc(X) - mu) @ Vt[:2].T) if len(X) else np.zeros((0, 2))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    if len(cat_desc):
        pc = proj(cat_desc)
        ax[0].scatter(pc[:, 0], pc[:, 1], s=10, c="#bbbbbb", label="Chan's catalogue")
    if len(Dk):
        pk = proj(Dk)
        sc = ax[0].scatter(pk[:, 0], pk[:, 1], s=10, c=[a["gen"] for a in kids], cmap="viridis", label="archive (colour = generation)")
        fig.colorbar(sc, ax=ax[0], label="generation")
    pr = proj(Dr)
    ax[0].scatter(pr[:, 0], pr[:, 1], s=40, marker="*", c="#d62728", label="roots")
    ax[0].set_title("descriptor space (PCA of 20 hand-made descriptors)", fontsize=10)
    ax[0].legend(fontsize=8)
    g = [h["gen"] for h in hist]
    ax[1].plot(g, [h["archive"] for h in hist], label="archive size")
    ax2 = ax[1].twinx()
    ax2.plot(g, [h["alive"] / max(1, sum(h["classes"].values())) for h in hist], color="#2a9d4b", alpha=0.6, label="children alive")
    ax2.set_ylim(0, 1)
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("archive size")
    ax2.set_ylabel("fraction of children alive")
    ax[1].set_title("progress")
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=120)
    plt.close(fig)


def load_archive(run: Path) -> list:
    run = Path(run)
    pats = np.load(run / "patterns.npz")
    arch = []
    for r in csv.DictReader(open(run / "archive.csv")):
        i = int(r["id"])
        fl = lambda k: float(r[k]) if r[k] not in ("", "nan") else np.nan
        arch.append({"id": i, "parent": int(r["parent"]), "root": int(r["root"]), "gen": int(r["gen"]), "label": r["label"],
                     "rule": Rule.from_dict(json.loads(r["rule"])), "init": pats[f"init_{i}"], "final": pats[f"final_{i}"],
                     "stats": {"class": r["class"], "speed_w": fl("speed_w"), "change_w": fl("change_w"), "mass_end": fl("mass_end")},
                     "desc": np.array([float(r[n]) for n in NAMES], np.float32), "novelty": fl("novelty"),
                     "dist_root": fl("dist_root"), "param_root": fl("param_root"), "dist_cat": fl("dist_cat"),
                     "rescale_class": r.get("rescale_class", ""),
                     "rescale_dist": float(r["rescale_dist"]) if r.get("rescale_dist") not in (None, "", "nan") else np.nan})
    return arch


def replot(run: Path, catalogue: list | None = None, shape=(128, 128), th: Thresholds = None, cmap: str = "chan",
           gif_lineages: int = 6, child_time: float = 30.0, log=print):
    """Redraw the pictures of a finished run, and animate the roots next to their farthest descendants."""
    from . import show
    run = Path(run)
    arch = load_archive(run)
    hist = json.loads((run / "history.json").read_text())
    cpath = run / "catalogue_desc.npy"
    if cpath.exists():
        cat_desc = np.load(cpath)
    elif catalogue:
        hz = Horizon(time=child_time, record_every=0.5, frames=6, film_span=5.0)
        cr = evaluate([s["rule"] for s in catalogue], [s["cells"][None] for s in catalogue], shape, hz, th or Thresholds(), None)
        cat_desc = np.array([d for _, _, d in cr if d is not None], np.float32)
        np.save(cpath, cat_desc)
    else:
        cat_desc = np.zeros((0, len(NAMES)), np.float32)
    roots = [a for a in arch if a["parent"] < 0]
    space = Space(np.concatenate([cat_desc, np.array([a["desc"] for a in roots])]) if len(cat_desc) else np.array([a["desc"] for a in roots]))
    _pictures(arch, [], cat_desc, space, hist, run, cmap, shape)
    # GIF: each row = a root (left) and its farthest descendant (right), both from their stored starts
    kids = [a for a in arch if a["parent"] >= 0]
    best = {}
    for a in kids:
        if a["root"] not in best or a["dist_root"] > best[a["root"]]["dist_root"]:
            best[a["root"]] = a
    pairs = sorted(best.values(), key=lambda a: -a["dist_root"])[:gif_lineages]
    by_id = {a["id"]: a for a in arch}
    clips = []
    for leaf in pairs:
        root = by_id[leaf["root"]]
        clips.append((f"root {root['label'].split('#')[-1]}", root["rule"], ("cells", np.asarray(root["init"], np.float32))))
        clips.append((f"#{leaf['id']} d={leaf['dist_root']:.1f}", leaf["rule"], ("cells", np.asarray(leaf["init"], np.float32))))
    if clips:
        sims = show.simulate(clips, shape, time=40.0, every=2, th=th)
        show.gif(sims, run / "lineages.gif", px=128, cols=4, cmap=cmap, start=15.0)
    log(f"redrawn: {run}")


def confirm(run: Path, n: int = 40, times=(30, 100, 200), shape=(128, 128), th: Thresholds = None, seed: int = 0,
            log=print) -> dict:
    """Re-run archive members for longer: the n most novel (farthest from the catalogue, else from
    their root) and n random ones. Did they stay alive, or were they slow explosions?"""
    from collections import Counter
    from . import show
    run = Path(run)
    arch = load_archive(run)
    kids = [a for a in arch if a["parent"] >= 0]
    has_cat = not all(np.isnan(a["dist_cat"]) for a in kids)
    top = sorted(kids, key=lambda a: -(a["dist_cat"] if has_cat else a["dist_root"]))[:n]
    rng = np.random.default_rng(seed)
    rnd = [kids[i] for i in rng.choice(len(kids), min(n, len(kids)), replace=False)]
    out = {}
    for name, group in (("most_novel", top), ("random", rnd)):
        clips = [(f"#{a['id']}", a["rule"], ("cells", np.asarray(a["init"], np.float32))) for a in group]
        out[name] = {}
        for T in times:
            sims = show.simulate(clips, shape, time=T, every=50, th=th)
            c = Counter(st["class"] for *_, st in sims)
            out[name][str(T)] = {"alive": sum(c[k] for k in ALIVE), "of": len(group), "classes": dict(c)}
            log(f"  {name} @ {T} time units: alive {out[name][str(T)]['alive']}/{len(group)}  {dict(c)}")
    (run / "confirm.json").write_text(json.dumps(out, indent=2))
    return out
