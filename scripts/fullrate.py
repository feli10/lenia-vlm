"""Blind rating session for the full run, and the judge test (11-06; project doc "Full run (seed 1) - diagnosis").

  python scripts/fullrate.py build --out RUNS/full-s1 --backend auto      # Colab: grows every world's film (GPU), no calls
  PYTHONPATH=src:scripts python3 scripts/fullrate.py rate --out RUNS/full-s1       # the rating page (Mac)
  PYTHONPATH=src:scripts python3 scripts/fullrate.py judges --out RUNS/full-s1     # AFTER rating: Jev and Gemini, same pairs
  PYTHONPATH=src:scripts python3 scripts/fullrate.py report --out RUNS/full-s1     # no calls
  PYTHONPATH=src:scripts python3 scripts/fullrate.py build --out ~/fr-dry --size 96 --warmup 200 --render 120 --gap 40
      # offline dry run on a dry full run; then `judges --judge fake --vlm fake`

Felix's three questions (11-06), and the pairs that answer them. "a" is the world counted as a win.
1. Is goal steering working? (goal question; the goal is shown)
   - goal:end-vs-ref: an expedition's end vs the world its final goal is judged against (the start, or after a switch
     the new stepping stone), with the final goal. --exp expeditions, about evenly over the rounds.
   - goal:control-vs-start: a random judged, unpicked step-1 child of the same expedition vs its start, with the
     original goal: what one unsteered mutation step scores. Steering works if ends beat their reference clearly more
     often than controls beat the start (Fisher, one-sided).
2. If steering doesn't refine, where do refinements come from? (interest question; no goal shown)
   - interest:end-vs-start: the same expeditions' ends vs their starts. Does steering make worlds more interesting?
   - interest:end-vs-mid: the end vs the parent after step 5 (continue / refine expeditions only, so both are on one
     line). Do the later steps refine?
   - interest:map-replacement: a map challenger vs the elite it replaced (accepted entry duels after iteration 20).
     Does the map's own competition refine, by Felix's eye?
3. Is Jev a good visual judge; would Gemini be better? Every pair above, plus
   - calib:cross-cell: two random current elites from different islands and cells (no "a" meaning; calibration only);
   - interest:gallery-vs-random: a gallery pick vs a random elite outside the shortlist (does selection add anything?).
   After rating, `judges` asks the same pairs, both orders, of: Jev with the rating's interest question (INTEREST),
   Jev with the map's aim duel (what drove the search), Jev's attain prompt for goal pairs (what drove the expeditions),
   and Gemini (PAIR_JUDGE) on the four stills Jev sees and on an eight-frame strip of the film Felix sees.
   `report` gives each judge's agreement with Felix on the pairs he decided.

Worlds are regrown from their rules (films over the snapshot window); a world whose regrown stills differ from its
stored picture by more than stage1.REPLAY_TOL is shown regrown, so film and stills always match. Sides, order and file
names are random; group names never reach the page. From papers: none beyond the sign and Fisher tests. Ours: the design.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import fullrun as FR
import mace_qd as QD
import stage1 as S1
from lenia import backend
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, load_env
from lenia.prompts import AIM_ALIFE, INTEREST, PAIR_JUDGE, PAIR_LAYOUT, attain_prompt, map_prompt
from lenia.sim import Rule

Q_INTEREST = ("Which world is more interesting as artificial life: richer structure or behaviour, more like something "
              "alive, more worth exploring further?")
Q_GOAL = "Which world shows this change more clearly? Judge only the change asked for, not which world is nicer."


def sdir(a) -> Path:
    return Path(a.dir) if getattr(a, "dir", None) else Path(a.out) / "rate"


def build(a) -> None:
    F, R = Path(a.out), sdir(a)
    if (R / "tasks.json").exists() and not a.force:
        sys.exit(f"{R / 'tasks.json'} exists (--force rebuilds it, only before any rating)")
    if (R / "ratings.csv").exists():
        sys.exit(f"{R / 'ratings.csv'} exists: not rebuilding")
    MAP = F / "map"
    backend.configure(a.backend)
    ea = S1.eval_settings(MAP, a)
    seed = json.loads((F / "state.json").read_text())["args"]["seed"]
    rng = np.random.default_rng([seed, 606])
    W, tasks = {}, []

    def world(name, rule, still):
        W.setdefault(name, {"rule": rule, "still": still})
        return name

    def add(group, q, x, y, g=None, **meta):
        if W[x]["rule"] == W[y]["rule"]:
            return
        tasks.append({"group": group, "question": q, "a": x, "b": y, "sees": g["sees"] if g else None,
                      "push": g["push"] if g else None, **meta})

    # ---- expeditions: --exp of them, spread over the rounds
    exps = []
    for Rr in sorted((F / "rounds").glob("r*"), key=lambda p: int(p.name[1:])):
        f = Rr / "exp" / "state.json"
        if not f.exists():
            continue
        est = json.loads(f.read_text())
        if est["step"] < est["end"]:
            continue
        tl = FR.read_jsonl(Rr / "exp" / "trace.jsonl")
        jd = {}
        for x in FR.read_jsonl(Rr / "exp" / "judged.jsonl"):
            jd[(x["trigger"], x["step"], x["child"])] = x
        for tr in est["triggers"]:
            if tr["history"]:
                exps.append((int(Rr.name[1:]), tr, tl, jd, est))
    by_round = {}
    for e in exps:
        by_round.setdefault(e[0], []).append(e)
    pick = []
    while len(pick) < min(a.exp, len(exps)):
        for r in sorted(by_round):
            pool = [e for e in by_round[r] if e not in pick]
            if pool and len(pick) < a.exp:
                pick.append(pool[int(rng.integers(len(pool)))])
    n_mid = 0
    for r, tr, tl, jd, est in pick:
        tid = tr["id"]
        dec = (tr.get("review") or {}).get("applied", "continue")
        start = world(f"start:{tid}", tr["start_rule"], tr["start"])
        end = world(f"end:{tid}", tr["parent_rule"], tr["parent"])
        ref = world(f"ref:{tid}", tr["ref_rule"], tr["ref"]) if tr.get("ref") else start
        g0 = (tr.get("goals_before") or [tr["goal"]])[0]
        meta = {"trigger": tid, "round": r, "decision": dec, "route": tr["route"]}
        add("goal:end-vs-ref", "goal", end, ref, tr["goal"], **meta)
        add("interest:end-vs-start", "interest", end, start, **meta)
        kids = sorted((x for x in jd.values() if x["trigger"] == tid and x["step"] == 1 and not x["picked"]),
                      key=lambda x: x["child"])
        if kids:
            k = kids[int(rng.integers(len(kids)))]
            add("goal:control-vs-start", "goal", world(f"ctrl:{tid}", k["rule"], None), start, g0, **meta)
        if dec != "switch" and n_mid < a.mid:
            h = [x for x in tl if x["trigger"] == tid and x["step"] <= est["review_at"]]
            if h:
                add("interest:end-vs-mid", "interest", end, world(f"mid:{tid}", h[-1]["rule"], h[-1]["image"]), **meta)
                n_mid += 1

    # ---- map: replacements, cross-cell calibration, gallery vs random
    rows = FR.ledger_rows(MAP)
    info = {x["id"]: x for x in rows if x["kind"] != "surprise"}
    has = lambda i: (MAP / "img" / f"{i}.png").exists()
    rep = [x for x in rows if x["kind"] == "duel" and x.get("placed") and x["it"] >= a.rep_after
           and x.get("opponent") in info and has(x["id"]) and has(x["opponent"])]
    for j in rng.permutation(len(rep))[:a.rep]:
        x = rep[int(j)]
        o = info[x["opponent"]]
        add("interest:map-replacement", "interest", world(f"map:{x['id']}", x["rule"], f"map/img/{x['id']}.png"),
            world(f"map:{o['id']}", o["rule"], f"map/img/{o['id']}.png"), it=x["it"], p=x["p"])
    sm = json.loads((MAP / "state.json").read_text())
    el = [(k, ck, e) for k, m in enumerate(QD.islands_of(sm)) for ck, e in sorted(m.items())]
    elw = lambda e: world(f"map:{Path(e['image']).stem}", e["rule"], f"map/img/{e['image']}")
    n = 0
    for j in rng.permutation(len(el)):
        if n >= a.cross:
            break
        k1, c1, e1 = el[int(j)]
        others = [x for x in el if x[0] != k1 and x[1] != c1]
        k2, c2, e2 = others[int(rng.integers(len(others)))]
        add("calib:cross-cell", "interest", elw(e1), elw(e2))
        n += 1
    gj = F / "gallery" / "gallery.json"
    if gj.exists():
        g = json.loads(gj.read_text())
        short = {x["id"] for x in g["shortlist"]}
        rest = [e for _, _, e in el if Path(e["image"]).stem not in short]
        byid = {Path(e["image"]).stem: e for _, _, e in el}
        for p in g["picks"]:
            if p["id"] in byid:
                add("interest:gallery-vs-random", "interest", elw(byid[p["id"]]), elw(rest[int(rng.integers(len(rest)))]))

    # ---- films, files, tasks
    names = sorted(W)
    print(f"{len(tasks)} pairs over {len(names)} worlds ({dict(Counter(t['group'] for t in tasks))}); growing films",
          file=sys.stderr)
    res = S1.v4_films([Rule.from_dict(W[n_]["rule"]) for n_ in names], ea, a.every)
    (R / "images").mkdir(parents=True, exist_ok=True)
    salt = int(rng.integers(1 << 30))
    replay = {}
    for n_, (frames, pic, snaps) in zip(names, res):
        h = hashlib.sha1(f"{n_}|{salt}".encode()).hexdigest()[:12]
        W[n_]["file"] = h
        st = W[n_]["still"]
        d = None
        if st and (F / st).exists():
            ref = np.asarray(Image.open(F / st).convert("RGB"), np.int16)
            d = float(np.abs(ref - pic.astype(np.int16)).mean()) if ref.shape == pic.shape else float("inf")
            replay[n_] = round(d, 2)
        if d is not None and d <= S1.REPLAY_TOL:
            Image.open(F / st).convert("RGB").save(R / "images" / f"{h}.png")
        else:
            Image.fromarray(pic).save(R / "images" / f"{h}.png")
        S1._gif(frames, snaps, R / "images" / f"{h}.gif", a.px, a.fps)
    out = []
    for k in rng.permutation(len(tasks)):
        t = tasks[int(k)]
        l, r = (t["a"], t["b"]) if rng.random() < 0.5 else (t["b"], t["a"])
        out.append({"id": len(out), **t, "left": W[l]["file"], "right": W[r]["file"], "left_world": l, "right_world": r})
    (R / "tasks.json").write_text(json.dumps(out, indent=1))
    bad = {k: v for k, v in replay.items() if v > S1.REPLAY_TOL}
    (R / "replay.json").write_text(json.dumps({"tol": S1.REPLAY_TOL, "diff": replay, "diverged": bad}, indent=1))
    (R / "design.md").write_text(DESIGN.format(counts=json.dumps(dict(Counter(t["group"] for t in out)), indent=1)))
    print(f"wrote {R}: {len(out)} pairs; replay: {len(replay)} stored pictures compared, {len(bad)} diverged (shown "
          "regrown)", file=sys.stderr)


DESIGN = """# Full run rating session: design (fixed before rating, 11-06)

Pairs per group:
```
{counts}
```
Tests, stated before rating (one-sided sign tests on decided pairs unless noted):
1. Goal steering works if goal:end-vs-ref wins clearly more often than goal:control-vs-start (Fisher, one-sided, p < 0.05).
2. Refinement: interest:end-vs-start > 0.5 (steering refines interest); interest:end-vs-mid > 0.5 (later steps refine);
   interest:map-replacement > 0.5 (the map's competition refines).
3. Selection: interest:gallery-vs-random > 0.5.
4. Judges: agreement with Felix on decided pairs, per judge and question; a judge is useful if clearly above 0.5
   (sign test) on interest pairs, which drive the map and the gallery.
"""


def rate(a) -> None:
    from lenia.rating import rate_pairs
    rate_pairs(sdir(a), port=a.port)


def strip(gif: Path, n: int = 8, px: int = 200) -> np.ndarray:
    im = Image.open(gif)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(im.convert("RGB").resize((px, px), Image.BILINEAR))
    pick = [frames[int(round(j))] for j in np.linspace(0, len(frames) - 1, n)]
    out = Image.new("RGB", (4 * px + 12, 2 * px + 4), (40, 40, 40))
    for j, f in enumerate(pick):
        out.paste(f, ((j % 4) * (px + 4), (j // 4) * (px + 4)))
    return np.asarray(out)


def judges(a) -> None:
    from lenia.describe import Ask, Describer
    R = sdir(a)
    tasks = json.loads((R / "tasks.json").read_text())
    if not (R / "ratings.csv").exists() and not a.before_rating:
        sys.exit("no ratings yet: rate first, so the judges can't influence anything (or --before-rating for a dry run)")
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    jev = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                R / "judge_cache.sqlite", yes=True)
    files = {}
    for t in tasks:
        files[t["left_world"]], files[t["right_world"]] = t["left"], t["right"]
    pics = {w: np.asarray(Image.open(R / "images" / f"{h}.png").convert("RGB")) for w, h in files.items()}
    res = {t["id"]: {} for t in tasks}
    aim = map_prompt(AIM_ALIFE)
    for name, sel, pid in (("jev:interest", "interest", lambda t: INTEREST), ("jev:aim", "interest", lambda t: aim),
                           ("jev:attain", "goal", lambda t: attain_prompt({"id": S1.goal_id(t["sees"], t["push"]),
                                                                            "sees": t["sees"], "push": t["push"]}))):
        ts = [t for t in tasks if t["question"] == sel]
        groups = {}
        for t in ts:
            groups.setdefault(pid(t), []).append(t)
        for p_id, tt in groups.items():
            ps = S1.duels(jev, p_id, [(t["a"], t["b"]) for t in tt], pics, name)
            for t, p in zip(tt, ps):
                res[t["id"]][name] = None if p is None else round(p, 4)
    d = Describer(a.vlm, cfg["vlms"][a.vlm], R / "describe_cache.sqlite", max_usd=a.max_vlm_usd)
    strips = {w: strip(R / "images" / f"{h}.gif") for w, h in files.items()} if "strip" in a.inputs else {}
    for inp in a.inputs.split(","):
        P = pics if inp == "stills" else strips
        asks, meta = [], []
        for t in tasks:
            fill = {"layout": PAIR_LAYOUT[inp],
                    "context": "" if t["question"] == "interest" else
                    f"Both descend from a world that showed this: {t['sees']}\nThis change was wanted: {t['push']}\n\n",
                    "question": Q_INTEREST if t["question"] == "interest" else Q_GOAL}
            for first, second in ((t["a"], t["b"]), (t["b"], t["a"])):
                asks.append(Ask(PAIR_JUDGE, [P[first], P[second]], fill))
            meta.append(t)
        ans = d.ask_many(asks, desc=f"gemini {inp}")
        for i, t in enumerate(meta):
            x, y = ans[2 * i], ans[2 * i + 1]
            c = lambda r: None if not r.ok or str(r.data["choice"]).strip() not in ("1", "2") else str(r.data["choice"]).strip()
            cx, cy = c(x), c(y)
            if cx is None or cy is None:
                res[t["id"]][f"gemini:{inp}"] = None
                continue
            res[t["id"]][f"gemini:{inp}"] = 0.5 * ((cx == "1") + (cy == "2"))     # P(a), from both orders
            res[t["id"]][f"gemini:{inp}:why"] = [x.data.get("why"), y.data.get("why")]
    (R / "judges.json").write_text(json.dumps({"judge": jev.model, "vlm": f"{a.vlm}:{d.model}", "results": res}, indent=1))
    print(f"judges: Jev ${jev.cache.total_cost():.3f}, {a.vlm} ${d.cache.total_cost():.3f}; {R / 'judges.json'}", file=sys.stderr)
    report(a)


def report(a) -> None:
    from lenia.stats import fisher_greater
    R = sdir(a)
    tasks = {t["id"]: t for t in json.loads((R / "tasks.json").read_text())}
    if not (R / "ratings.csv").exists():
        sys.exit("no ratings yet")
    ans = {}
    for r in csv.DictReader(open(R / "ratings.csv")):
        ans[int(r["id"])] = r["answer"]
    felix = {}                                   # id -> True (a chosen) / False (b) / None (neither)
    for i, x in ans.items():
        t = tasks[i]
        felix[i] = None if x == "neither" else ((t["left_world"] if x == "left" else t["right_world"]) == t["a"])
    by = {}
    for i, v in felix.items():
        by.setdefault(tasks[i]["group"], []).append(v)
    L = [f"# Full run rating session ({len(felix)} of {len(tasks)} pairs rated)\n",
         "Win = the first-named world (end, control, later elite, gallery pick) chosen. calib:cross-cell has no "
         "meaningful winner (calibration only).\n",
         "| group | wins | losses | neither | win rate (decided) | one-sided p |", "|---|---|---|---|---|---|"]
    rows = {}
    for g in sorted(by):
        v = by[g]
        w, l_ = sum(x is True for x in v), sum(x is False for x in v)
        rows[g] = (w, l_)
        L.append(f"| {g} | {w} | {l_} | {sum(x is None for x in v)} | {w / max(1, w + l_):.2f} | "
                 f"{S1._binom_greater(w, w + l_):.3f} |")
    e, c = rows.get("goal:end-vs-ref"), rows.get("goal:control-vs-start")
    if e and c:
        L.append(f"\n**Q1, steering vs an unsteered step:** ends beat their reference {e[0]}/{sum(e)}, controls beat the "
                 f"start {c[0]}/{sum(c)}; Fisher one-sided p = {fisher_greater(e[0], e[1], c[0], c[1]):.3f}.")
    for key in ("decision", "route", "round"):
        sub = {}
        for i, v in felix.items():
            t = tasks[i]
            if t["group"] in ("goal:end-vs-ref", "interest:end-vs-start") and key in t:
                sub.setdefault((t["group"], t[key]), []).append(v)
        if sub:
            L += [f"\nBy {key}:\n", "| group | " + key + " | wins | losses | neither |", "|---|---|---|---|---|"]
            for (g, k), v in sorted(sub.items(), key=lambda x: (x[0][0], str(x[0][1]))):
                L.append(f"| {g} | {k} | {sum(x is True for x in v)} | {sum(x is False for x in v)} | {sum(x is None for x in v)} |")
    jf = R / "judges.json"
    if jf.exists():
        J = json.loads(jf.read_text())["results"]
        names = sorted({k for r in J.values() for k in r if not k.endswith(":why")})
        L += ["\n## Judges vs Felix (pairs he decided)\n",
              "Agree = the judge's both-orders P(first-named) is on Felix's side of 0.5; ties (P = 0.5: the two orders "
              "disagreed) count as half. Decisive = share of pairs where the judge isn't at 0.5 (Gemini) or is "
              "outside 0.4–0.6 (Jev).\n",
              "| judge | question | n | agree | agreement | one-sided p | decisive |", "|---|---|---|---|---|---|---|"]
        for nm in names:
            for q in ("interest", "goal"):
                pts = [(J[str(i)][nm], v) for i, v in felix.items() if v is not None and tasks[i]["question"] == q
                       and J.get(str(i), {}).get(nm) is not None]
                if not pts:
                    continue
                ag = sum(1.0 if (p > 0.5) == v and p != 0.5 else 0.5 if p == 0.5 else 0.0 for p, v in pts)
                dec = np.mean([abs(p - 0.5) > (0.1 if nm.startswith("jev") else 0.0) for p, _ in pts])
                k = int(round(ag))
                L.append(f"| {nm} | {q} | {len(pts)} | {ag:.1f} | {ag / len(pts):.2f} | "
                         f"{S1._binom_greater(k, len(pts)):.3f} | {dec:.2f} |")
    sides = list(ans.values())
    L.append(f"\nSides: {sides.count('left')} left, {sides.count('right')} right, {sides.count('neither')} neither.")
    (R / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "rate", "judges", "report"):
        p = sub.add_parser(c)
        p.add_argument("--out", required=True, help="the full run's folder")
        p.add_argument("--dir", default=None, help="the session folder (default <out>/rate)")
    b = sub.choices["build"]
    b.add_argument("--backend", default="numpy")
    b.add_argument("--exp", type=int, default=16, help="expeditions (goal and interest pairs, plus controls)")
    b.add_argument("--mid", type=int, default=6, help="of those, end-vs-mid pairs (continue / refine only)")
    b.add_argument("--rep", type=int, default=10, help="map replacement pairs")
    b.add_argument("--rep-after", type=int, default=20, help="replacements from this iteration on")
    b.add_argument("--cross", type=int, default=10, help="cross-cell calibration pairs")
    b.add_argument("--every", type=int, default=30, help="simulation steps per film frame")
    b.add_argument("--px", type=int, default=280)
    b.add_argument("--fps", type=int, default=15)
    b.add_argument("--force", action="store_true")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        b.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    sub.choices["rate"].add_argument("--port", type=int, default=8774)
    j = sub.choices["judges"]
    j.add_argument("--judge", default="jev")
    j.add_argument("--vlm", default="gemini-pro")
    j.add_argument("--inputs", default="stills,strip", help="Gemini's inputs: stills (as Jev), strip (8 film frames)")
    j.add_argument("--config", default=None)
    j.add_argument("--max-usd", type=float, default=0.2)
    j.add_argument("--max-vlm-usd", type=float, default=5.0)
    j.add_argument("--before-rating", action="store_true", help="dry runs only")
    a = ap.parse_args()
    {"build": build, "rate": rate, "judges": judges, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
