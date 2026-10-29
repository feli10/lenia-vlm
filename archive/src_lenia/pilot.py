"""Mutation-operator pilot: which variation operators reach genuinely new creatures and stay viable?

Why: the MAP-Elites runs (qd.py) found creatures about as far from the pool as pool creatures are from
each other (median cosine distance 0.35 vs 0.31 in the student's DINOv2 embedding): siblings, not new
species. `explore.mutate` only nudges parameters; even so, half its children die. Before a new search, this
pilot measures each operator in operators.py on the same parents, with no API calls.

run     (Colab, GPU) draws `n_parents` final elites from the MAP-Elites maps (stratified by map), applies
        every operator once to each parent (operators needing a partner draw one from all the maps'
        elites: same channel count for `merge`, same layout for `crossover`), and runs each child through
        the frozen alive check (alive v2: alive at 100 units, coherent and keeping its texture at 2x,
        alive in the render; then alive at 200 units, as an elite must be). Structural children first get a
        short re-tune: up to `retune_rounds` x `retune_k` small m/s nudges of the new rule (the unchanged
        edit first), and the first one alive at 100 units goes on. Viable children are rendered in the
        pool's image format and embedded with the student's image model; recorded: cosine distance (in the
        student's standardised embedding, as qd's `nn`) to the nearest parent and to the nearest pool
        creature, the student's and the five-statistic rewards, and their statistics. Results are saved
        after each operator; a stopped run resumes.
report  (anywhere) the table and the decision rule, written before running (docs, "Next: mutation-operator
        pilot"): an operator is worth using if at least 10% of its children are viable and the median
        viable child is beyond 0.5 from the pool. Writes report.md, summary.json, sheet.png.
"""
from __future__ import annotations

import csv
import json
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import operators as ops
from .alive import Thresholds
from .describe import Space
from .explore import evaluate
from .rollout import Horizon
from .rsearch import STATS, _camera, _render, viable_mutants
from .sim import Rule

MIN_VIABLE = 0.10          # decision rule: at least this share of children viable ...
MIN_POOL_DIST = 0.5        # ... and the median viable child farther than this from the nearest pool creature
FIELDS = ["op", "child", "parent", "partner", "parent_cell", "applicable", "stage", "tries", "image", "pattern",
          "d_parent", "d_partner", "d_nearest_parent", "nn", "student", "student_parent", "features", "spread",
          *STATS, "kernels", "rule"]


def _hz(t: float) -> Horizon:
    return Horizon(time=t, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)


# ---------- parents ----------

def load_elites(maps: list) -> list:
    """Every final elite of the MAP-Elites runs: dicts with key, map folder, cell, rule, pattern, image path."""
    out = []
    for m in maps:
        m = Path(m)
        st = json.loads((m / "state.json").read_text())
        pz = np.load(m / "patterns.npz")
        for c in sorted(st["cells"]):
            eid = st["cells"][c][-1]
            e = st["events"][eid]
            P = np.asarray(pz[e["pattern"]], np.float32)
            out.append({"key": f"{m.name}:{eid}", "map": m.name, "cell": c, "rule": Rule.from_dict(json.loads(e["rule"])),
                        "pattern": P if P.ndim == 3 else P[None], "image": m / "images" / e["image"],
                        "nn": e.get("nn"), "student": e.get("student")})
    return out


def pick_parents(elites: list, n: int, seed: int = 0) -> list:
    """n elites, split evenly across the maps (the remainder to the first maps), without replacement."""
    rng = np.random.default_rng([seed, 1])
    maps = sorted({e["map"] for e in elites})
    by = {m: [e for e in elites if e["map"] == m] for m in maps}
    want = {m: n // len(maps) + (k < n % len(maps)) for k, m in enumerate(maps)}
    out = []
    for m in maps:
        k = min(want[m], len(by[m]))
        out += [by[m][i] for i in sorted(rng.choice(len(by[m]), k, replace=False))]
    return out


def _partner(p: dict, elites: list, rng, ok) -> dict | None:
    cand = [e for e in elites if e["key"] != p["key"] and ok(e)]
    return cand[int(rng.integers(len(cand)))] if cand else None


def make_children(op: str, parents: list, elites: list, rng) -> list:
    """One child per parent: {parent, partner, rule (None = operator does not apply), start, structural}."""
    out = []
    for p in parents:
        rule, start, q = p["rule"], p["pattern"], None
        if op == "copy":
            pass
        elif op in ("mut1", "mut2", "mut4"):
            rule = ops.scaled(rule, rng, float(op[3:]))
        elif op == "reseed":
            start = ops.reseed(rule, rng, max(start.shape[-2:]))
        elif op == "cut":
            start = ops.cut(start, rng)
        elif op == "merge":
            q = _partner(p, elites, rng, lambda e: e["pattern"].shape[0] == start.shape[0])
            start = ops.merge(start, q["pattern"], rng) if q else None
            rule = rule if q else None
        elif op in ops.STRUCTURAL:
            rule = ops.STRUCTURAL[op](rule, rng)
            rule = rule if rule is not None and ops.valid(rule) else None
        elif op == "crossover":
            q = _partner(p, elites, rng, lambda e: ops.layout(e["rule"]) == ops.layout(rule))
            rule = ops.crossover(rule, q["rule"], rng) if q else None
        else:
            raise ValueError(f"unknown operator {op!r}; choose from {ops.OPERATORS}")
        out.append({"parent": p, "partner": q, "rule": rule, "start": start, "structural": op in ops.STRUCTURAL})
    return out


# ---------- one operator ----------

def retune(rules: list, starts: list, rng, shape, th, batch, hz1, rounds: int = 2, k: int = 3, scale: float = 0.5):
    """For each rule (None = skip): up to rounds x k candidates (the rule itself, then small m/s nudges of it),
    run to hz1; the first alive one is kept. Returns (kept rules or None, their evaluate() results, tries)."""
    n = len(rules)
    kept, res, tries = [None] * n, [None] * n, [0] * n
    for rnd in range(rounds):
        todo = [i for i in range(n) if rules[i] is not None and kept[i] is None]
        if not todo:
            break
        cand, owner = [], []
        for i in todo:
            for j in range(k):
                cand.append(rules[i] if (rnd == 0 and j == 0) else ops.nudge_ms(rules[i], rng, scale))
                owner.append(i)
        r = evaluate(cand, [starts[i] for i in owner], shape, hz1, th, None, batch)
        for rule, i, x in zip(cand, owner, r):
            tries[i] += 1
            if kept[i] is None and x[1] is not None:
                kept[i], res[i] = rule, x
    return kept, res, tries


def run_operator(op: str, parents: list, elites: list, ctx: dict, log=print) -> tuple:
    """Children of every parent under `op`, through the alive check. Returns (rows, images, patterns)."""
    rng = np.random.default_rng([ctx["seed"], zlib.crc32(op.encode())])
    kids = make_children(op, parents, elites, rng)
    n = len(kids)
    shape, th, batch, hz1 = ctx["shape"], ctx["th"], ctx["batch"], _hz(100.0)
    rows = [{"op": op, "child": f"{op}_{i:03d}", "parent": c["parent"]["key"], "parent_cell": c["parent"]["cell"],
             "partner": c["partner"]["key"] if c["partner"] else "", "applicable": c["rule"] is not None,
             "stage": "n/a" if c["rule"] is None else "dead@100", "tries": 0, "kernels": None if c["rule"] is None else len(c["rule"].kernels),
             "rule": "" if c["rule"] is None else c["rule"].to_json()} for i, c in enumerate(kids)]
    app = [i for i in range(n) if kids[i]["rule"] is not None]
    starts = [kids[i]["start"] for i in app]
    if op in ops.STRUCTURAL:
        kept, r1s, tries = retune([kids[i]["rule"] for i in app], starts, rng, shape, th, batch, hz1,
                                  ctx["retune_rounds"], ctx["retune_k"], ctx["retune_scale"])
        for i, t, rule in zip(app, tries, kept):
            rows[i]["tries"] = t
            if rule is not None:
                rows[i]["rule"], rows[i]["kernels"] = rule.to_json(), len(rule.kernels)
        live = [j for j, rule in enumerate(kept) if rule is not None]
        idx = [app[j] for j in live]
        rules, starts, r1 = [kept[j] for j in live], [starts[j] for j in live], [r1s[j] for j in live]
    else:
        for i in app:
            rows[i]["tries"] = 1
        idx, rules, r1 = app, [kids[i]["rule"] for i in app], None
    if not rules:
        return rows, {}, {}
    r1, ok, viable, rend = viable_mutants(rules, starts, shape, th, batch, ctx["space"], ctx["max_dist"], ctx["max_loss"],
                                          hz1, ctx["camera"], r1=r1)
    for k in ok:
        rows[idx[k]]["stage"] = "failed@2x"
    conf = evaluate([rules[k] for k in viable], [r1[k][1] for k in viable], shape, _hz(200.0), th, None, batch) if viable else []
    keep = [(j, k) for j, (k, (_, pat2, _)) in enumerate(zip(viable, conf)) if pat2 is not None]
    for k in viable:
        rows[idx[k]]["stage"] = "dead@200"
    images, patterns = {}, {}
    if keep:
        sc = ctx["scorer"]([rend[j][0] for j, _ in keep], [rend[j][1] for j, _ in keep])
        U = ctx["scorer"].unit(sc["emb"]) if sc.get("emb") is not None else None
        for m, (j, k) in enumerate(keep):
            i = idx[k]
            row, kid = rows[i], kids[i]
            row["stage"] = "viable"
            row["image"], row["pattern"] = f"{row['child']}.png", row["child"]
            images[row["image"]] = rend[j][0]
            patterns[row["child"]] = np.asarray(conf[j][1], np.float32)
            row.update({s: (None if rend[j][1][s] is None else float(rend[j][1][s])) for s in STATS})
            for key in ("student", "features", "spread", "nn"):
                row[key] = float(sc[key][m])
            row["student_parent"] = kid["parent"].get("student_now")
            if U is not None:
                dp = 1 - float(U[m] @ ctx["elite_unit"][kid["parent"]["key"]])
                dq = 1 - float(U[m] @ ctx["elite_unit"][kid["partner"]["key"]]) if kid["partner"] else None
                row["d_parent"], row["d_partner"] = dp, dq
                row["d_nearest_parent"] = dp if dq is None else min(dp, dq)
    nv = sum(r["stage"] == "viable" for r in rows)
    log(f"  {op}: {len(app)}/{n} applicable, {len(ok)} alive at 100, {len(viable)} past 2x, {nv} viable")
    return rows, images, patterns


# ---------- the run ----------

def run(out: Path, maps: list, pool_dir: Path, survey: Path, student_dir: Path, operators=ops.OPERATORS,
        n_parents: int = 64, retune_rounds: int = 2, retune_k: int = 3, retune_scale: float = 0.5, seed: int = 0,
        shape=(256, 256), th: Thresholds = None, batch: int = 64, log=print) -> dict:
    from .found import calibration
    from .qd import _Scorer
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    th = th or Thresholds()
    cal = calibration(survey)
    scorer = _Scorer(student_dir, pool_dir, "student")
    elites = load_elites(maps)
    parents = pick_parents(elites, n_parents, seed)
    sp = out / "state.json"
    state = json.loads(sp.read_text()) if sp.exists() else {"done": [], "seconds": {}}
    if state.get("parents") and state["parents"] != [p["key"] for p in parents]:
        raise ValueError(f"{out} was started with other parents (maps, n_parents or seed changed); use another --out")
    # every elite embedded once: parents' and partners' distances, and the parents' baseline
    t0 = time.time()
    sc = scorer([Image.open(e["image"]).convert("RGB") for e in elites], [])
    U = scorer.unit(sc["emb"])
    for e, u, s in zip(elites, U, sc["student"]):
        e["student_now"] = float(s)
    elite_unit = {e["key"]: u for e, u in zip(elites, U)}
    loo = None
    if scorer.ref is not None:
        S = scorer.ref @ scorer.ref.T
        np.fill_diagonal(S, -np.inf)
        loo = 1 - S.max(1)
    pk = {p["key"] for p in parents}
    state.update({"parents": [p["key"] for p in parents],
                  "config": {"maps": [str(m) for m in maps], "pool": str(pool_dir), "student": str(student_dir),
                             "survey": str(survey), "n_parents": n_parents, "retune_rounds": retune_rounds,
                             "retune_k": retune_k, "retune_scale": retune_scale, "seed": seed, "camera": _camera(pool_dir),
                             "rescale_max_dist": cal["rescale_max_dist"], "texture_max_loss": cal.get("texture_max_loss")},
                  "baseline": {"parents_nn": [float(sn) for e, sn in zip(elites, sc["nn"]) if e["key"] in pk],
                               "elites_nn": [float(x) for x in sc["nn"]],
                               "pool_loo_nn": None if loo is None else [float(x) for x in loo]}})
    log(f"{len(parents)} parents from {len(elites)} elites in {', '.join(Path(m).name for m in maps)}; embedded in {time.time() - t0:.0f}s")
    ctx = {"seed": seed, "shape": tuple(shape), "th": th, "batch": batch, "space": Space(np.load(Path(survey) / "alive.npz")["desc"]),
           "max_dist": cal["rescale_max_dist"], "max_loss": cal.get("texture_max_loss"), "camera": _camera(pool_dir),
           "scorer": scorer, "elite_unit": elite_unit, "retune_rounds": retune_rounds, "retune_k": retune_k,
           "retune_scale": retune_scale}
    rp = out / "results.csv"
    pz = out / "patterns.npz"
    if rp.exists():                         # drop rows of an operator that was interrupted before it was marked done
        old = [r for r in csv.DictReader(open(rp)) if r["op"] in state["done"]]
        with open(rp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(old)
    patterns = dict(np.load(pz)) if pz.exists() else {}
    for op in operators:
        if op in state["done"]:
            log(f"  {op}: done already")
            continue
        t1 = time.time()
        rows, images, pats = run_operator(op, parents, elites, ctx, log)
        for name, im in images.items():
            im.save(out / "images" / name)
        patterns.update(pats)
        np.savez_compressed(out / "patterns.tmp.npz", **patterns)
        (out / "patterns.tmp.npz").replace(pz)
        new = not rp.exists()
        with open(rp, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows([{k: r.get(k) for k in FIELDS} for r in rows])
        state["done"].append(op)
        state["seconds"][op] = round(time.time() - t1, 1)
        sp.with_suffix(".tmp").write_text(json.dumps(state))
        sp.with_suffix(".tmp").replace(sp)
        log(f"  {op}: {state['seconds'][op]:.0f}s")
    return report(out, log=log)


# ---------- report ----------

def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else np.nan


def verdict(viable_frac: float, median_nn: float) -> str:
    a, b = viable_frac >= MIN_VIABLE, np.isfinite(median_nn) and median_nn > MIN_POOL_DIST
    return "pass" if a and b else "fail: " + " and ".join(x for x, bad in (("too few viable", not a), ("too near the pool", not b)) if bad)


def summarise(rows: list, pool_p90: float | None = None) -> list:
    out = []
    for op in [o for o in ops.OPERATORS if any(r["op"] == o for r in rows)] + sorted({r["op"] for r in rows} - set(ops.OPERATORS)):
        rr = [r for r in rows if r["op"] == op]
        app = [r for r in rr if str(r["applicable"]) == "True"]
        v = [r for r in app if r["stage"] == "viable"]
        frac = len(v) / len(app) if app else 0.0
        nn = [_f(r["nn"]) for r in v]
        sd = [_f(r["student"]) - _f(r["student_parent"]) for r in v]
        s = {"op": op, "children": len(rr), "applicable": len(app), "viable": len(v), "viable_frac": round(frac, 3),
             "stages": {k: sum(r["stage"] == k for r in app) for k in ("dead@100", "failed@2x", "dead@200", "viable")},
             "tries_per_child": round(float(np.mean([_f(r["tries"]) for r in app])), 2) if app else None,
             "d_parent_median": round(_med([_f(r["d_nearest_parent"]) for r in v]), 3),
             "nn_median": round(_med(nn), 3),
             "beyond_min_dist": round(float(np.mean([x > MIN_POOL_DIST for x in nn if np.isfinite(x)])), 3) if v else None,
             "student_delta_median": round(_med(sd), 3)}
        if pool_p90 is not None and v:
            s["beyond_pool_p90"] = round(float(np.mean([x > pool_p90 for x in nn if np.isfinite(x)])), 3)
        s["verdict"] = "baseline" if op == "copy" else verdict(frac, s["nn_median"]) if app else "n/a"
        out.append(s)
    return out


def report(out: Path, log=print) -> dict:
    out = Path(out)
    rows = list(csv.DictReader(open(out / "results.csv")))
    st = json.loads((out / "state.json").read_text())
    b = st.get("baseline", {})
    loo = [x for x in (b.get("pool_loo_nn") or []) if np.isfinite(x)]
    p90 = float(np.quantile(loo, 0.9)) if loo else None
    summ = summarise(rows, p90)
    res = {"rule": f"pass = viable >= {MIN_VIABLE:.0%} of children and median viable child > {MIN_POOL_DIST} from the pool",
           "operators": summ, "passing": [s["op"] for s in summ if s["verdict"] == "pass"],
           "baseline": {"parents_nn_median": round(_med(b.get("parents_nn", [])), 3),
                        "elites_nn_median": round(_med(b.get("elites_nn", [])), 3),
                        "pool_loo_nn_median": round(_med(loo), 3) if loo else None,
                        "pool_loo_nn_p90": round(p90, 3) if p90 else None},
           "config": st.get("config"), "seconds": st.get("seconds")}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    _markdown(out, res)
    sheet(out, rows)
    log((out / "report.md").read_text())
    return res


def _markdown(out: Path, r: dict) -> None:
    b, c = r["baseline"], r.get("config") or {}
    L = ["# Mutation-operator pilot", "",
         f"{c.get('n_parents')} parents from {', '.join(Path(m).name for m in c.get('maps', []))}; one child per parent per "
         f"operator. Viable = alive at 100 units, coherent and keeping texture at 2x, alive in the render, alive at 200. "
         f"Structural operators: up to {c.get('retune_rounds')} x {c.get('retune_k')} m/s re-tune tries.", "",
         f"Distances: cosine, student's standardised DINOv2 embedding. Pool leave-one-out nearest neighbour: median "
         f"{b['pool_loo_nn_median']}, p90 {b['pool_loo_nn_p90']}; the parents (final elites): median {b['parents_nn_median']}.", "",
         f"Decision rule (written before running): {r['rule']}.", "",
         "| operator | what | viable | dead@100 / 2x / dead@200 | tries | to parent | to pool | > 0.5 | > pool p90 | student delta | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    num = lambda x, f=".3f": "-" if x is None or not np.isfinite(x) else format(x, f)
    for s in r["operators"]:
        g = s["stages"]
        if not s["applicable"]:
            L.append(f"| {s['op']} | {ops.DESCRIPTION.get(s['op'], '')} | 0/0 | - | - | - | - | - | - | - | n/a (applies to no parent) |")
            continue
        L.append(f"| {s['op']} | {ops.DESCRIPTION.get(s['op'], '')} | {s['viable']}/{s['applicable']} ({s['viable_frac']:.0%}) | "
                 f"{g['dead@100']} / {g['failed@2x']} / {g['dead@200']} | {s['tries_per_child']} | {num(s['d_parent_median'])} | "
                 f"{num(s['nn_median'])} | {num(s['beyond_min_dist'], '.2f')} | {num(s.get('beyond_pool_p90'), '.2f')} | "
                 f"{num(s['student_delta_median'], '+.3f')} | {s['verdict']} |")
    L += ["", f"Passing: {', '.join(r['passing']) or 'none'}.", "",
          "Columns: tries = 100-unit rollouts per child (re-tune cost); to parent = median distance of viable children to "
          "the nearer of their parents; to pool = median distance to the nearest pool creature; > 0.5 and > pool p90 = share "
          "of viable children beyond those; student delta = median student score of the child minus its parent's.", "",
          "sheet.png: per operator, the viable children farthest from the pool, each next to its parent."]
    if r.get("seconds"):
        L += ["", "Time per operator (s): " + ", ".join(f"{k} {v:.0f}" for k, v in r["seconds"].items())]
    (out / "report.md").write_text("\n".join(L) + "\n")


def _motion_view(path: Path, px: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.width > im.height * 1.5:
        im = im.crop((0, 0, im.height, im.height))
    return im.resize((px, px), Image.BILINEAR)


def sheet(out: Path, rows: list | None = None, per_op: int = 4, px: int = 96) -> Path | None:
    """Rows = operators; per row the `per_op` viable children farthest from the pool, each as parent | child."""
    out = Path(out)
    rows = rows if rows is not None else list(csv.DictReader(open(out / "results.csv")))
    st = json.loads((out / "state.json").read_text())
    maps = {Path(m).name: Path(m) for m in st.get("config", {}).get("maps", [])}
    events = {}
    for name, m in maps.items():
        try:
            events[name] = json.loads((m / "state.json").read_text())["events"]
        except OSError:
            pass
    op_list = [o for o in dict.fromkeys(r["op"] for r in rows)]
    if not op_list:
        return None
    lw, gap = 110, 6
    W = lw + per_op * (2 * px + gap + 2)
    H = 22 + len(op_list) * (px + 18)
    im = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.text((6, 4), "farthest viable children from the pool: parent | child (label: distance to pool)", fill=(20, 20, 20))
    for i, op in enumerate(op_list):
        y = 22 + i * (px + 18)
        v = sorted([r for r in rows if r["op"] == op and r["stage"] == "viable"], key=lambda r: -_f(r["nn"]))[:per_op]
        d.text((6, y + px // 2 - 6), f"{op} ({sum(r['op'] == op and r['stage'] == 'viable' for r in rows)})", fill=(40, 40, 40))
        for j, r in enumerate(v):
            x = lw + j * (2 * px + gap + 2)
            mname, eid = r["parent"].split(":")
            try:
                im.paste(_motion_view(maps[mname] / "images" / events[mname][int(eid)]["image"], px), (x, y))
            except (OSError, KeyError, IndexError):
                pass                         # parent picture not reachable here (e.g. map folder elsewhere)
            im.paste(_motion_view(out / "images" / r["image"], px), (x + px + 2, y))
            d.text((x + px + 4, y + px + 2), f"{_f(r['nn']):.2f}", fill=(40, 40, 40))
    im.save(out / "sheet.png")
    return out / "sheet.png"
