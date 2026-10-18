"""Phase 4: search against a reward model, under the alive constraint, with the teacher looking on.

run     (Colab, GPU) n lineages, founded by creatures from the labelling pool spread across the
        student's score range (the same founders for every reward, so runs are comparable). Every
        generation each lineage tries `children` mutants of its current rule, started from its current
        creature. A mutant must pass the frozen alive check (alive v2), cheapest first:
          1. alive after 100 time units
          2. coherent at 2x resolution: alive there, the same creature (descriptor distance) and keeping
             its fine structure (texture loss), with the thresholds of the pool's calibrated survey
        and is then rendered as a two-view image (as in the pool) and scored by the reward:
          student   the distilled reward model (frozen image model + head; mean over the ensemble)
          features  the Bradley-Terry reward on the five hand-made statistics, fitted on the same labels
        The best-scoring viable mutant replaces its parent if it scores higher and is still alive after
        200 time units (greedy hill climbing; no API calls). Every accepted creature is saved with its
        image, rule, pattern, scores under both rewards, the student's ensemble spread and its statistics.
        Resumable: state is saved after every generation.
check   (your Mac, API keys) the teacher (and a second judge) on founder vs the lineage's creature at each
        checkpoint generation, both orders: does the teacher agree that the reward went up? That is the
        RQ1 curve: student-measured gain against teacher-measured gain as search pressure grows. Also the
        drift of size, mass, speed, change and texture, and pairs for you to rate (founder vs final).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .alive import ALIVE, Thresholds
from .describe import Space
from .explore import evaluate, mutate, rescale_check, texture, texture_loss
from .rollout import Horizon
from .sim import Rule

CHECKPOINTS = [0, 5, 10, 20, 40, 60, 80, 100]
STATS = ("speed", "change", "size_rg", "mass", "texture")


def _camera(pool_dir: Path) -> str:
    """The pool's image format, so new creatures are drawn exactly as the student was trained on."""
    p = Path(pool_dir) / "build.json"
    return json.loads(p.read_text()).get("camera", "fixed") if p.exists() else "fixed"


def _render(rules: list, patterns: list, shape, chunk: int = 32, camera: str = "dual") -> list:
    """Images (in the pool's format; two-view for pool2) and statistics: [(PIL image, stats row, final frame)]."""
    from .pool import _simulate
    from .vlmtest import dual_image, grid_image
    draw = {"dual": lambda fr, R: dual_image(fr, R), "dual_body": lambda fr, R: dual_image(fr, R, scale="body"),
            "fixed": lambda fr, R: grid_image(fr[-4:], R, False)}[camera]
    items = [("x", "x", r, np.asarray(p, np.float32) if np.asarray(p).ndim == 3 else np.asarray(p, np.float32)[None])
             for r, p in zip(rules, patterns)]
    sims = _simulate(items, shape, 30.0, chunk, lambda *_: None, 2.0, 9)
    out = []
    for s in sims:
        st = s["stats"]
        row = {"class": st["class"], "speed": st.get("speed_w"), "change": st.get("change_w"), "size_rg": st.get("rg_max_w"),
               "mass": st.get("mass_end"), "texture": texture(s["frames"][-1])}
        out.append((Image.fromarray(draw(s["frames"], s["rule"].R)), row, s["frames"][-1]))
    return out


class _Scorer:
    def __init__(self, student_dir: Path, reward: str):
        from .distill import Student
        self.student, self.reward = Student(student_dir), reward

    def __call__(self, images: list, rows: list) -> dict:
        try:
            s_mean, s_spread = self.student.score_images(images) if images else (np.zeros(0), np.zeros(0))
        except ImportError:                 # no PyTorch here: only the five-statistic reward can drive the search
            if self.reward == "student":
                raise
            s_mean = s_spread = np.full(len(images), np.nan)
        f = self.student.score_features(rows) if rows else np.zeros(0)
        main = s_mean if self.reward == "student" else f
        return {"main": main, "student": s_mean, "spread": s_spread, "features": f}


def pick_founders(pool_dir: Path, student_dir: Path, n: int, seed: int = 0) -> list:
    """n pool creatures spread evenly across the student's score range (lower and middle ones leave room to
    climb; a few high ones show what happens at the top)."""
    from .pool import _item_of
    pool_dir = Path(pool_dir)
    rows = {r["id"]: r for r in csv.DictReader(open(pool_dir / "creatures.csv"))}
    z = np.load(Path(student_dir) / "student.npz")
    ids, sc = [str(i) for i in z["ids"]], z["score"]
    order = np.argsort(sc)
    pick = [ids[order[int(round(q))]] for q in np.linspace(0, len(ids) - 1, n + 2)[1:-1]]
    return [rows[i] for i in pick]


def viable_mutants(rules: list, starts: list, shape, th: Thresholds, batch: int, space: Space, max_dist: float,
                   max_loss, hz1: Horizon, camera: str):
    """The frozen alive check (alive v2), cheapest first, then the render. Returns (r1, ok, viable, rend):
    r1 = evaluate() results for every mutant, ok = indices alive after hz1, viable = indices that also pass
    the 2x coherence and texture checks and are still alive in the render, rend = their renders."""
    r1 = evaluate(rules, starts, shape, hz1, th, None, batch)
    ok = [k for k, (_, pat, _) in enumerate(r1) if pat is not None]
    r2 = rescale_check([rules[k] for k in ok], [r1[k][1] for k in ok], shape, th, None, batch, 2, 50.0, with_pattern=True)
    viable = []
    for k, (cl, d2, p2) in zip(ok, r2):
        if cl not in ALIVE or d2 is None:
            continue
        if float(space.dist(r1[k][2], np.asarray(d2)[None])[0]) > max_dist:
            continue
        if max_loss is not None and texture_loss(r1[k][1], p2) > max_loss:
            continue
        viable.append(k)
    rend = _render([rules[k] for k in viable], [r1[k][1] for k in viable], shape, camera=camera)
    keep = [j for j, (_, st, _) in enumerate(rend) if st["class"] in ALIVE]
    return r1, ok, [viable[j] for j in keep], [rend[j] for j in keep]


def _save(out: Path, state: dict, patterns: dict) -> None:
    tmp = out / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    np.savez_compressed(out / "patterns.tmp.npz", **patterns)
    os.replace(out / "patterns.tmp.npz", out / "patterns.npz")
    os.replace(tmp, out / "state.json")


def run(out: Path, pool_dir: Path, survey: Path, archive: Path, student_dir: Path, reward: str = "student",
        n_lineages: int = 16, generations: int = 60, children: int = 3, mut_scale: float = 1.0, seed: int = 0,
        shape=(256, 256), th: Thresholds = None, batch: int = 64, log=print) -> dict:
    from .found import calibration
    from .pool import _item_of
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    th = th or Thresholds()
    cal = calibration(survey)
    z = np.load(Path(survey) / "alive.npz")
    space = Space(z["desc"])
    max_dist, max_loss = cal["rescale_max_dist"], cal.get("texture_max_loss")
    scorer = _Scorer(student_dir, reward)
    camera = _camera(pool_dir)
    hz1 = Horizon(time=100.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    hz2 = Horizon(time=200.0, record_every=0.5, frames=6, film_span=5.0, thumb_every=0)
    t0 = time.time()
    patterns = {}
    if (out / "state.json").exists():
        state = json.loads((out / "state.json").read_text())
        pz = np.load(out / "patterns.npz")
        patterns = {k: pz[k] for k in pz.files}
        log(f"resuming at generation {state['gen']}")
    else:
        founders = pick_founders(pool_dir, student_dir, n_lineages, seed)
        cache = {}
        items = [_item_of(r, survey, archive, cache) for r in founders]
        rend = _render([it[2] for it in items], [it[3] for it in items], shape, camera=camera)
        sc = scorer([r[0] for r in rend], [r[1] for r in rend])
        lin = []
        for k, (row, it, (img, st, final)) in enumerate(zip(founders, items, rend)):
            name = f"L{k:02d}_g000.png"
            img.save(out / "images" / name)
            patterns[f"L{k:02d}"] = np.asarray(it[3], np.float32)
            patterns[f"L{k:02d}_g000"] = np.asarray(it[3], np.float32)
            lin.append({"id": k, "founder": row["id"], "source": row["source"], "rule": it[2].to_json(), "score": float(sc["main"][k]),
                        "accepted": 0, "events": [{"gen": 0, "image": name, "rule": it[2].to_json(), "pattern": f"L{k:02d}_g000",
                                                   "main": float(sc["main"][k]), "student": float(sc["student"][k]),
                                                   "spread": float(sc["spread"][k]), "features": float(sc["features"][k]),
                                                   **{s: (None if st[s] is None else float(st[s])) for s in STATS}}]})
        state = {"gen": 0, "lineages": lin, "history": [],
                 "config": {"reward": reward, "student": str(student_dir), "pool": str(pool_dir), "camera": camera, "n_lineages": n_lineages,
                            "generations": generations, "children": children, "mut_scale": mut_scale, "seed": seed,
                            "rescale_max_dist": max_dist, "texture_max_loss": max_loss}}
        _save(out, state, patterns)
        log(f"{len(lin)} lineages founded (reward: {reward})")
    lin = state["lineages"]
    for gen in range(state["gen"] + 1, generations + 1):
        rng = np.random.default_rng([seed, gen])
        parents = [i for i in range(len(lin)) for _ in range(children)]
        rules = [mutate(Rule.from_dict(json.loads(lin[i]["rule"])), rng, mut_scale) for i in parents]
        starts = [patterns[f"L{i:02d}"] for i in parents]
        r1, ok, viable, rend = viable_mutants(rules, starts, shape, th, batch, space, max_dist, max_loss, hz1, camera)
        sc = scorer([r[0] for r in rend], [r[1] for r in rend]) if rend else None
        best = {}
        for j, k in enumerate(viable):
            i = parents[k]
            if sc["main"][j] > lin[i]["score"] and (i not in best or sc["main"][j] > sc["main"][best[i]]):
                best[i] = j
        cand = list(best.items())
        conf = evaluate([rules[viable[j]] for _, j in cand], [r1[viable[j]][1] for _, j in cand], shape, hz2, th, None, batch) if cand else []
        accepted = []
        for (i, j), (st2, pat2, _) in zip(cand, conf):
            if pat2 is None:
                continue
            k = viable[j]
            L = lin[i]
            name = f"L{i:02d}_g{gen:03d}.png"
            rend[j][0].save(out / "images" / name)
            pkey = f"L{i:02d}_g{gen:03d}"
            patterns[f"L{i:02d}"] = np.asarray(pat2, np.float32)
            patterns[pkey] = np.asarray(r1[k][1], np.float32)
            row = rend[j][1]
            L.update(rule=rules[k].to_json(), score=float(sc["main"][j]), accepted=L["accepted"] + 1)
            L["events"].append({"gen": gen, "image": name, "rule": rules[k].to_json(), "pattern": pkey, "main": float(sc["main"][j]),
                                "student": float(sc["student"][j]), "spread": float(sc["spread"][j]),
                                "features": float(sc["features"][j]),
                                **{s: (None if row[s] is None else float(row[s])) for s in STATS}})
            accepted.append(i)
        state["history"].append({"gen": gen, "mutants": len(rules), "alive": len(ok), "viable": len(viable),
                                 "improved": len(cand), "accepted": len(accepted),
                                 "mean_score": round(float(np.mean([L["score"] for L in lin])), 4),
                                 "seconds": round(time.time() - t0, 1)})
        state["gen"] = gen
        _save(out, state, patterns)
        h = state["history"][-1]
        log(f"gen {gen}: {h['mutants']} mutants, alive {h['alive']}, viable {h['viable']}, improved {h['improved']}, "
            f"accepted {h['accepted']}; mean {reward} score {h['mean_score']:+.3f}; {h['seconds']:.0f}s")
    draw(out)
    return state


def state_at(L: dict, gen: int) -> dict:
    return [e for e in L["events"] if e["gen"] <= gen][-1]


def draw(out: Path) -> Path:
    from .vlmtest import _contact
    out = Path(out)
    st = json.loads((out / "state.json").read_text())
    G = st["gen"]
    cps = sorted({c for c in CHECKPOINTS if c <= G} | {G})
    items = []
    for L in st["lineages"]:
        for c in cps:
            e = state_at(L, c)
            items.append((f"L{L['id']:02d} g{c} ({e['gen']}) {e['main']:+.2f}", out / "images" / e["image"]))
    _contact(items, out / "lineages.png", f"{st['config']['reward']} search: columns = generations {cps} (when accepted; score)",
             cols=len(cps), px=200)
    return out / "lineages.png"


# ---------- check ----------

def check(out: Path, judges: dict, prompt: str = "curate_dual", log=print) -> dict:
    """judges: {name: Judge}. Founder vs each checkpoint creature, both orders, per lineage."""
    from .judge import Query
    out = Path(out)
    st = json.loads((out / "state.json").read_text())
    G = st["gen"]
    cps = [c for c in CHECKPOINTS if 0 < c <= G] + ([G] if G not in CHECKPOINTS else [])
    load = lambda n: np.asarray(Image.open(out / "images" / n).convert("RGB"))
    pairs = []
    for L in st["lineages"]:
        f = L["events"][0]
        for c in cps:                      # repeated creatures across checkpoints hit the judge cache
            e = state_at(L, c)
            if e["gen"] > 0:
                pairs.append((L["id"], c, f, e))
    res = {"generations": G, "checkpoints": cps, "reward": st["config"]["reward"], "per_judge": {}, "curve": []}
    rows = []
    P = {}
    for name, j in judges.items():
        qs = []
        for _, _, f, e in pairs:
            A, B = load(e["image"]), load(f["image"])
            qs += [Query(prompt, A, B), Query(prompt, B, A)]
        vs = j.judge_many(qs, desc=f"check {name}") if qs else []
        for k, (lid, c, f, e) in enumerate(pairs):
            v1, v2 = vs[2 * k], vs[2 * k + 1]
            p = (v1.p_first + 1 - v2.p_first) / 2 if v1.ok and v2.ok else np.nan
            P[(name, lid, c)] = p
            rows.append({"judge": name, "lineage": lid, "checkpoint": c, "accepted_gen": e["gen"], "p_new_beats_founder": p,
                         "student_gain": e["student"] - f["student"], "features_gain": e["features"] - f["features"],
                         "spread": e["spread"], **{f"{s}_founder": f[s] for s in STATS}, **{s: e[s] for s in STATS}})
    # the curve: per checkpoint, mean gain under the rewards and mean teacher win rate against the founder
    for c in cps:
        es = [(L, state_at(L, c)) for L in st["lineages"]]
        row = {"checkpoint": c, "lineages_changed": sum(e["gen"] > 0 for _, e in es),
               "student_gain": round(float(np.mean([e["student"] - L["events"][0]["student"] for L, e in es])), 4),
               "features_gain": round(float(np.mean([e["features"] - L["events"][0]["features"] for L, e in es])), 4)}
        for name in judges:
            ps = [P.get((name, L["id"], c), 0.5 if e["gen"] == 0 else np.nan) for L, e in es]
            row[f"{name}_p_new_beats_founder"] = round(float(np.nanmean(ps)), 3)
            ch = [P.get((name, L["id"], c)) for L, e in es if e["gen"] > 0]
            row[f"{name}_changed_lineages_won"] = f"{int(np.nansum(np.array(ch, float) > 0.5))}/{len(ch)}"
        for s in ("size_rg", "mass", "speed"):
            row[f"{s}_median"] = round(float(np.median([e[s] for _, e in es if e[s] is not None])), 4)
        res["curve"].append(row)
    if rows:
        with open(out / "check.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # pairs for you: founder vs final, per changed lineage
    rng = np.random.default_rng(0)
    hp = []
    for L in st["lineages"]:
        fin = state_at(L, G)
        if fin["gen"] == 0:
            continue
        a, b = L["events"][0]["image"][:-4], fin["image"][:-4]
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        hp.append({"a": a, "b": b, "left": left, "right": right, "kind": "founder_vs_final"})
    (out / "human_pairs.json").write_text(json.dumps(hp, indent=1))
    hum = [r for r in csv.DictReader(open(out / "human.csv"))] if (out / "human.csv").exists() else []
    won = [r["winner"] == r["b"] for r in hum if r["choice"] in ("left", "right")]
    res["you_final_beats_founder"] = (round(float(np.mean(won)), 3), len(won)) if won else None
    (out / "check.json").write_text(json.dumps(res, indent=2))
    _md(out, res, st, list(judges))
    log((out / "check.md").read_text())
    return res


def _md(out: Path, r: dict, st: dict, judges: list) -> None:
    cfg = st["config"]
    L = [f"# Reward-guided search ({cfg['reward']}): teacher check", "",
         f"{len(st['lineages'])} lineages, {r['generations']} generations, {cfg['children']} mutants per lineage per generation. "
         "Gains are relative to each lineage's founder (lineages that never changed count as 0 gain and p = 0.5). "
         "p = P(new creature beats its founder), both orders averaged.", "",
         "| gen | changed | student gain | features gain | " + " | ".join(f"{j} p (won)" for j in judges)
         + " | size med. | mass med. | speed med. |", "|---" * (7 + len(judges)) + "|"]
    for c in r["curve"]:
        L.append(f"| {c['checkpoint']} | {c['lineages_changed']} | {c['student_gain']:+.3f} | {c['features_gain']:+.3f} | "
                 + " | ".join(f"{c[f'{j}_p_new_beats_founder']:.2f} ({c[f'{j}_changed_lineages_won']})" for j in judges)
                 + f" | {c['size_rg_median']:.2f} | {c['mass_median']:.2f} | {c['speed_median']:.3f} |")
    if r.get("you_final_beats_founder"):
        L += ["", f"Your ratings: final beats founder {r['you_final_beats_founder'][0]} ({r['you_final_beats_founder'][1]} lineages)."]
    L += ["", "Pictures: lineages.png (columns = checkpoint generations)."]
    (out / "check.md").write_text("\n".join(L) + "\n")
