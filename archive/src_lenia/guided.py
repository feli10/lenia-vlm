"""VLM-guided search: does a VLM's taste, used directly as the objective, close the gap to Chan's species?

run    16 lineages, each founded by a different coherent creature from the multikernel search. Every
       generation each lineage makes one child: a mutated copy of its rule, started from its current
       creature. The child must pass the hard constraints, cheapest first:
         1. alive after 100 time units (the 'alive' classifier)
         2. coherent: run again at 2x resolution for 50 units, still alive and the same creature there
            (descriptor distance <= explore.RESCALE_MAX_DIST); this run also gives the picture
         (a lineage whose mutant fails 1 or 2 tries another, up to `attempts` per generation: about half
         of mutants die, so this keeps most lineages in play; it costs simulation, never judge calls)
         3. the judge (Luna, prompt 'curate') prefers it to its parent: P(child) averaged over both
            orders >= threshold (0.6)
         4. still alive after 200 time units
       Only then does it replace its parent. Lineages never merge. State is saved after every
       generation, so a stopped run resumes where it left off.
check  held-out evaluation (judges that never drove the search, plus the driver for comparison):
       founder vs each checkpoint, and founder vs final against the same catalogue species; how size,
       speed, liveliness and roughness drifted; pairs for you to rate (`lenia guided rate`).
"""
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .alive import ALIVE, Thresholds
from .describe import Space, describe
from .explore import RESCALE_MAX_DIST, evaluate, load_archive, mutate, rescaled, upsample
from .rollout import Horizon
from .sim import Rule
from .vlmtest import SNAP_DT, _contact, bradley_terry, grid_image, roughness

CHECKPOINTS = [0, 10, 25, 50]


# ---------- founders ----------

def pick_founders(run: Path, n: int = 16, seed: int = 0, margin: float = 0.75) -> list:
    """n diverse coherent creatures from an explore run (farthest-point sampling in descriptor space).
    Only creatures comfortably inside the coherence constraint (2x distance <= margin x the threshold):
    the search piled up just under the threshold, and mutants of those edge cases mostly fall over it."""
    arch = [a for a in load_archive(run)
            if not np.isfinite(a.get("rescale_dist", np.nan)) or a["rescale_dist"] <= margin * RESCALE_MAX_DIST]
    cat = np.load(Path(run) / "catalogue_desc.npy")
    space = Space(cat if len(cat) else np.array([a["desc"] for a in arch]))
    D = np.array([a["desc"] for a in arch])
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(arch)))]
    dmin = space.dist(D[chosen[0]], D)
    while len(chosen) < min(n, len(arch)):
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, space.dist(D[k], D))
    return [{"label": f"{Path(run).name}#{arch[k]['id']}", "rule": arch[k]["rule"],
             "pattern": np.asarray(arch[k]["final"], np.float32), "desc": arch[k]["desc"]} for k in chosen]


# ---------- 2x simulation: coherence check + picture ----------

def _sim2x(args):
    from .show import simulate
    rule_dicts, pats, shape, sim_time, every = args
    clips = [("c", Rule.from_dict(d), ("cells", np.asarray(p, np.float32))) for d, p in zip(rule_dicts, pats)]
    out = []
    # a frame every time unit: the descriptor averages the last 6 (5 units, as in explore's calibrated
    # check); the picture uses 4 frames SNAP_DT apart ending at the last step
    step = int(round(SNAP_DT))
    for _, rule, frames, st in simulate(clips, tuple(shape), time=sim_time, every=every):
        fr = [np.asarray(f, np.float32) for _, f in frames]
        d = np.mean([describe(f, st, rule.R) for f in fr[-6:]], axis=0) if st["class"] in ALIVE else None
        out.append((st, fr[-1 - 3 * step::step], d))
    return out


def sim2x(rules: list, patterns: list, shape, ex, sim_time: float = 50.0, batch: int = 8) -> list:
    """Each creature at double resolution: [(stats, last four frames 2 units apart, descriptor or None)]."""
    if not rules:
        return []
    from . import backend
    if backend.is_gpu():
        batch = max(batch, 64)
    big = [rescaled(r) for r in rules]
    pats = [upsample(p) for p in patterns]
    groups = {}
    for i, r in enumerate(big):
        groups.setdefault((r.structure(), r.T), []).append(i)
    chunks = [g[k:k + batch] for g in groups.values() for k in range(0, len(g), batch)]
    every = int(round(big[0].T))            # one frame per time unit
    args = [([big[i].to_dict() for i in ch], [pats[i] for i in ch], tuple(s * 2 for s in shape), sim_time, every)
            for ch in chunks]
    res = list(ex.map(_sim2x, args)) if ex else [_sim2x(a) for a in args]
    out = [None] * len(rules)
    for ch, rr in zip(chunks, res):
        for i, x in zip(ch, rr):
            out[i] = x
    return out


def _stats_row(st: dict) -> dict:
    return {k: float(st.get(k, np.nan)) for k in ("mass_end", "rg_max_w", "speed_w", "change_w")}


# ---------- state ----------

def _save_state(out: Path, state: dict) -> None:
    lin = state["lineages"]
    js = {"gen": state["gen"], "config": state["config"], "history": state["history"],
          "lineages": [{k: v for k, v in L.items() if k not in ("pattern", "rule", "desc", "image")} |
                       {"rule": L["rule"].to_json(), "desc": [float(x) for x in L["desc"]]} for L in lin]}
    tmp = out / "state.json.tmp"
    tmp.write_text(json.dumps(js))
    np.savez_compressed(out / "state_patterns.tmp.npz", **{str(i): L["pattern"] for i, L in enumerate(lin)})
    os.replace(out / "state_patterns.tmp.npz", out / "state_patterns.npz")
    os.replace(tmp, out / "state.json")


def _load_state(out: Path) -> dict | None:
    if not (out / "state.json").exists():
        return None
    js = json.loads((out / "state.json").read_text())
    pats = np.load(out / "state_patterns.npz")
    for i, L in enumerate(js["lineages"]):
        L["rule"] = Rule.from_dict(json.loads(L["rule"]))
        L["desc"] = np.array(L["desc"], np.float32)
        L["pattern"] = pats[str(i)]
        L["image"] = np.asarray(Image.open(out / "images" / L["image_file"]).convert("RGB"))
    return js


# ---------- run ----------

def run(out: Path, founders: list, judge, prompt: str, cat_desc: np.ndarray, generations: int = 50,
        threshold: float = 0.6, mut_scale: float = 1.0, seed: int = 0, shape=(128, 128), workers: int = 0,
        th: Thresholds = None, n_lineages: int = 16, attempts: int = 3, log=print) -> dict:
    """founders: candidates in order of preference; the first n_lineages that are coherent now are used."""
    from .judge import Query
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    th = th or Thresholds()
    space = Space(cat_desc)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    hz1 = Horizon(time=100.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    hz_confirm = Horizon(time=200.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    t0 = time.time()

    state = _load_state(out)
    if state is None:
        # founders: picture at 2x; keep those that are still alive and coherent there
        r2 = sim2x([f["rule"] for f in founders], [f["pattern"] for f in founders], shape, ex)
        lin = []
        for f, (st, fr, d2) in zip(founders, r2):
            if len(lin) >= n_lineages:
                break
            if d2 is None or float(space.dist(f["desc"], d2[None])[0]) > RESCALE_MAX_DIST:
                log(f"  founder {f['label']} not coherent at 2x now; skipped")
                continue
            k = len(lin)
            img = grid_image(fr, 2 * f["rule"].R, False)
            name = f"L{k:02d}_g000.png"
            Image.fromarray(img).save(out / "images" / name)
            lin.append({"id": k, "founder": f["label"], "rule": f["rule"], "pattern": f["pattern"], "desc": f["desc"],
                        "image": img, "image_file": name, "accepted": 0,
                        "events": [{"gen": 0, "image": name, "rule": f["rule"].to_json(), "p": None, **_stats_row(st)}]})
        np.savez_compressed(out / "founders.npz", **{str(L["id"]): np.asarray(L["pattern"], np.float32) for L in lin})
        state = {"gen": 0, "lineages": lin, "history": [],
                 "config": {"prompt": prompt, "judge": judge.name, "model": judge.model, "threshold": threshold,
                            "mut_scale": mut_scale, "seed": seed, "generations": generations}}
        _save_state(out, state)
        log(f"{len(lin)} lineages founded")
    lin = state["lineages"]
    rng = np.random.default_rng([seed, state["gen"]])

    for gen in range(state["gen"] + 1, generations + 1):
        kids, r1, r2, imgs = [None] * len(lin), [None] * len(lin), {}, {}
        alive, coherent, n_tries = set(), [], 0
        for _ in range(attempts):        # lineages still without a viable child try another mutant
            todo = [i for i in range(len(lin)) if i not in imgs]
            if not todo:
                break
            ks = [mutate(lin[i]["rule"], rng, mut_scale) for i in todo]
            n_tries += len(ks)
            res = evaluate(ks, [lin[i]["pattern"] for i in todo], shape, hz1, th, ex)
            ok = [(i, k, x) for i, k, x in zip(todo, ks, res) if x[1] is not None]
            res2 = sim2x([k for _, k, _ in ok], [x[1] for _, _, x in ok], shape, ex)
            for (i, k, x), y in zip(ok, res2):
                alive.add(i)
                st2, fr, d2 = y
                if d2 is not None and float(space.dist(x[2], d2[None])[0]) <= RESCALE_MAX_DIST:
                    kids[i], r1[i], r2[i] = k, x, y
                    imgs[i] = grid_image(fr, 2 * k.R, False)
                    coherent.append(i)
        qs = []
        for i in coherent:
            qs += [Query(prompt, imgs[i], lin[i]["image"]), Query(prompt, lin[i]["image"], imgs[i])]
        vs = judge.judge_many(qs, desc=f"gen {gen}") if qs else []
        p_child = {}
        for k, i in enumerate(coherent):
            v1, v2 = vs[2 * k], vs[2 * k + 1]
            if v1.ok and v2.ok:
                p_child[i] = (v1.p_first + (1 - v2.p_first)) / 2
        won = [i for i, p in p_child.items() if p >= threshold]
        conf = evaluate([kids[i] for i in won], [lin[i]["pattern"] for i in won], shape, hz_confirm, th, ex) if won else []
        accepted = []
        for i, (st, pat, d) in zip(won, conf):
            if pat is None:
                continue
            L = lin[i]
            name = f"L{i:02d}_g{gen:03d}.png"
            Image.fromarray(imgs[i]).save(out / "images" / name)
            L.update(rule=kids[i], pattern=pat, desc=d, image=imgs[i], image_file=name, accepted=L["accepted"] + 1)
            L["events"].append({"gen": gen, "image": name, "rule": kids[i].to_json(), "p": round(p_child[i], 4),
                                **_stats_row(r2[i][0])})
            accepted.append(i)
        state["history"].append({"gen": gen, "mutants_tried": n_tries, "lineages_with_alive_child": len(alive),
                                 "coherent": len(coherent), "judged": len(p_child),
                                 "won_duel": len(won), "accepted": len(accepted),
                                 "mean_p_child": round(float(np.mean(list(p_child.values()))), 3) if p_child else None,
                                 "spent": round(judge.spent, 4), "seconds": round(time.time() - t0, 1)})
        state["gen"] = gen
        _save_state(out, state)
        h = state["history"][-1]
        log(f"gen {gen}: {h['mutants_tried']} mutants tried, {h['coherent']}/{len(lin)} lineages with a viable child, judge-preferred {h['won_duel']}, "
            f"accepted {h['accepted']}; mean P(child) {h['mean_p_child']}; ${judge.spent:.3f}; {h['seconds']:.0f}s")
    if ex:
        ex.shutdown()
    draw_lineages(out)
    return state


def state_at(L: dict, gen: int) -> dict:
    """The lineage's creature at a generation: its last accepted event at or before it."""
    return [e for e in L["events"] if e["gen"] <= gen][-1]


def draw_lineages(out: Path, checkpoints: list | None = None) -> Path:
    out = Path(out)
    js = json.loads((out / "state.json").read_text())
    G = js["gen"]
    cps = sorted({c for c in (checkpoints or CHECKPOINTS) if c <= G} | {G})
    items = []
    for L in js["lineages"]:
        for c in cps:
            e = state_at(L, c)
            items.append((f"L{L['id']:02d} g{c} ({e['gen']})", out / "images" / e["image"]))
    _contact(items, out / "lineages.png", f"lineages: columns = generations {cps} (in brackets: when that creature was accepted)",
             cols=len(cps), px=180)
    return out / "lineages.png"


def gif(out: Path, shape=(128, 128), sim_time: float = 30.0) -> Path:
    """Founders (top of each pair) and finals (below), animated at 2x."""
    from . import show
    out = Path(out)
    js = json.loads((out / "state.json").read_text())
    pats = np.load(out / "state_patterns.npz")
    clips = []
    for i, L in enumerate(js["lineages"]):
        f = Rule.from_dict(json.loads(L["events"][0]["rule"]))
        fp = _founder_pattern(out, i)
        clips.append((f"L{i:02d} founder", rescaled(f), ("cells", upsample(fp))))
        r = Rule.from_dict(json.loads(L["rule"]))
        clips.append((f"L{i:02d} gen {L['events'][-1]['gen']}", rescaled(r), ("cells", upsample(pats[str(i)]))))
    sims = show.simulate(clips, tuple(s * 2 for s in shape), time=sim_time, every=3)
    order = [k for i in range(0, len(sims), 8) for k in list(range(i, min(i + 8, len(sims)), 2)) + list(range(i + 1, min(i + 8, len(sims)), 2))]
    return show.gif([sims[k] for k in order], out / "lineages.gif", px=150, cols=4, fps=15)


def _founder_pattern(out: Path, i: int) -> np.ndarray:
    return np.load(out / "founders.npz")[str(i)]


# ---------- check ----------

def check(out: Path, judges: dict, prompt: str, catalogue_images: list, n_cat: int = 4, seed: int = 0, log=print) -> dict:
    """judges: {name: Judge}. catalogue_images: [(id, path)] rendered the same way (vlmtest v2)."""
    from .judge import Query
    out = Path(out)
    js = json.loads((out / "state.json").read_text())
    G = js["gen"]
    cps = [c for c in CHECKPOINTS if 0 < c <= G] + ([G] if G not in CHECKPOINTS else [])
    load = lambda p: np.asarray(Image.open(p).convert("RGB"))
    rng = np.random.default_rng(seed)
    pairs = []      # (kind, lineage, label_a, path_a, label_b, path_b)
    for L in js["lineages"]:
        f = out / "images" / L["events"][0]["image"]
        fin = out / "images" / state_at(L, G)["image"]
        for c in cps:
            e = state_at(L, c)
            if e["gen"] > 0:
                pairs.append(("checkpoint", L["id"], f"g{c}", out / "images" / e["image"], "founder", f))
        for k in rng.choice(len(catalogue_images), min(n_cat, len(catalogue_images)), replace=False):
            cid, cp = catalogue_images[k]
            pairs.append(("cat_final", L["id"], "final", fin, cid, Path(cp)))
            pairs.append(("cat_founder", L["id"], "founder", f, cid, Path(cp)))
    res = {"generations": G, "checkpoints": cps, "per_judge": {}}
    rows = []
    for name, j in judges.items():
        qs = []
        for (_, _, _, pa, _, pb) in pairs:
            A, B = load(pa), load(pb)
            qs += [Query(prompt, A, B), Query(prompt, B, A)]
        vs = j.judge_many(qs, desc=f"check {name}")
        P = []
        for k, pr in enumerate(pairs):
            v1, v2 = vs[2 * k], vs[2 * k + 1]
            p = (v1.p_first + 1 - v2.p_first) / 2 if v1.ok and v2.ok else np.nan
            P.append(p)
            rows.append({"judge": name, "kind": pr[0], "lineage": pr[1], "a": pr[2], "b": pr[4], "p_a": p})
        P = np.array(P)
        kinds = np.array([p[0] for p in pairs])
        labels = np.array([p[2] for p in pairs])
        r = {"checkpoint_vs_founder": {c: round(float(np.nanmean(P[(kinds == "checkpoint") & (labels == f"g{c}")])), 3)
                                       for c in cps if ((kinds == "checkpoint") & (labels == f"g{c}")).any()},
             "final_beats_founder_lineages": int(np.nansum(P[(kinds == "checkpoint") & (labels == f"g{cps[-1]}")] > 0.5)) if cps else 0,
             "lineages_changed": int(((kinds == "checkpoint") & (labels == f"g{cps[-1]}")).sum()) if cps else 0,
             "founder_beats_catalogue": round(float(np.nanmean(P[kinds == "cat_founder"])), 3),
             "final_beats_catalogue": round(float(np.nanmean(P[kinds == "cat_final"])), 3)}
        res["per_judge"][name] = r
    with open(out / "check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # drift of creature statistics, founder -> final (from the 2x picture runs)
    drift = {}
    for k, lab in (("rg_max_w", "size (rel. kernel)"), ("speed_w", "speed"), ("change_w", "liveliness"), ("mass_end", "mass")):
        a = np.array([L["events"][0][k] for L in js["lineages"]])
        b = np.array([state_at(L, G)[k] for L in js["lineages"]])
        drift[lab] = {"founder_median": round(float(np.nanmedian(a)), 4), "final_median": round(float(np.nanmedian(b)), 4),
                      "lineages_up": int(np.sum(b > a)), "lineages": len(a)}
    res["drift"] = drift
    res["accepted_per_lineage"] = [L["accepted"] for L in js["lineages"]]
    # pairs for you to rate: founder vs final per changed lineage, and final vs one catalogue species
    hp = []
    imgdir = out / "images"
    for L in js["lineages"]:
        fin = state_at(L, G)
        if fin["gen"] == 0:
            continue
        a, b = L["events"][0]["image"][:-4], fin["image"][:-4]
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        hp.append({"a": a, "b": b, "left": left, "right": right, "kind": "founder_vs_final"})
        cid, cp = catalogue_images[int(rng.integers(len(catalogue_images)))]
        dst = imgdir / f"{cid}.png"
        if not dst.exists():
            dst.write_bytes(Path(cp).read_bytes())
        left, right = (b, cid) if rng.random() < 0.5 else (cid, b)
        hp.append({"a": b, "b": cid, "left": left, "right": right, "kind": "final_vs_catalogue"})
    order = rng.permutation(len(hp))
    (out / "human_pairs.json").write_text(json.dumps([hp[k] for k in order], indent=1))
    res["human"] = _human(out)
    (out / "check.json").write_text(json.dumps(res, indent=2))
    _check_md(out, res, js)
    log((out / "check.md").read_text())
    return res


def _human(out: Path) -> dict:
    p = out / "human.csv"
    hp = {(h["a"], h["b"]): h["kind"] for h in json.loads((out / "human_pairs.json").read_text())} if (out / "human_pairs.json").exists() else {}
    if not p.exists():
        return {}
    rows = [r for r in csv.DictReader(open(p)) if r["choice"] in ("left", "right")]
    fv = [r["winner"] == r["b"] for r in rows if hp.get((r["a"], r["b"])) == "founder_vs_final"]
    fc = [r["winner"] == r["a"] for r in rows if hp.get((r["a"], r["b"])) == "final_vs_catalogue"]
    return {"final_beats_founder": (round(float(np.mean(fv)), 3), len(fv)) if fv else None,
            "final_beats_catalogue": (round(float(np.mean(fc)), 3), len(fc)) if fc else None}


def _check_md(out: Path, r: dict, js: dict) -> None:
    cfg = js["config"]
    L = [f"# Guided search: held-out check", "",
         f"Driver: {cfg['judge']} ({cfg['model']}), prompt `{cfg['prompt']}`, threshold {cfg['threshold']}; "
         f"{r['generations']} generations, {len(js['lineages'])} lineages; accepted steps per lineage: {r['accepted_per_lineage']}.", "",
         "P(row beats column), averaged over both orders and lineages.", "",
         "| judge | " + " | ".join(f"g{c} vs founder" for c in r["checkpoints"]) + " | lineages improved | founder vs catalogue | final vs catalogue |",
         "|---|" + "---|" * (len(r["checkpoints"]) + 3)]
    for name, x in r["per_judge"].items():
        L.append(f"| {name}{' (driver)' if name == cfg['judge'] else ''} | "
                 + " | ".join(f"{x['checkpoint_vs_founder'].get(c, float('nan')):.2f}" for c in r["checkpoints"])
                 + f" | {x['final_beats_founder_lineages']}/{x['lineages_changed']} | {x['founder_beats_catalogue']:.2f} | {x['final_beats_catalogue']:.2f} |")
    L += ["", "Drift founder -> final (2x runs):", ""]
    L += [f"- {k}: median {v['founder_median']} -> {v['final_median']} (up in {v['lineages_up']}/{v['lineages']} lineages)"
          for k, v in r["drift"].items()]
    h = r.get("human") or {}
    if h:
        L += ["", f"Your ratings: final beats founder {h.get('final_beats_founder')}, final beats catalogue {h.get('final_beats_catalogue')} "
              "(rate, n)."]
    L += ["", "Pictures: lineages.png (columns = checkpoints), lineages.gif (founder above final, animated)."]
    (out / "check.md").write_text("\n".join(L) + "\n")
