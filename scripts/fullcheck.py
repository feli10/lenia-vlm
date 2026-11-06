"""Check a full run (scripts/fullrun.py) from what it has stored, finished or not (11-05; fix log "feedback ends crowd").

  python scripts/fullcheck.py --out RUNS/full-s1 --backend auto --evals /content/evals      # Colab: GPU + Jev (~$0.12)
  PYTHONPATH=src:scripts python3 scripts/fullcheck.py --out ~/fr-dry --judge fake          # offline dry run

Uses every round whose expeditions have finished (an interrupted last round is skipped and named). No Gemini calls.
Writes <out>/check/: report.md, aim.csv, swiss.csv, map.png, pairs.png and GIFs. Answers are cached in
<out>/check/judge_cache.sqlite, so a re-run costs nothing.

1. Per round, no calls: trigger sources, exact repeats (a start explored in an earlier round) and repeated lineages,
   routes, review decisions, Jev's end-vs-start goal and complexity scores (logged), feedback outcomes.
2. Aim duel, end vs its own start (the map's entry prompt, the user's aim; both orders, 644 px). The question the
   feedback duel never asked fairly: the map elite an end met had been selected by that very duel for up to 80
   iterations, the end for 10 steps on a different objective. Control: a random judged, unpicked step-1 child vs the
   same start (regrown), i.e. what one unsteered mutation step scores. The lift over the control is the effect.
3. Ends on neutral ground: the gallery's Bradley-Terry ratings (every aim entry duel in the ledger), then a 4-round Swiss
   tournament of aim duels among the map's cell champions plus every finished expedition end, refitted. Where the
   ends rank tells whether they lost their feedback duels because they are weak or because the pipe is narrow.
4. Pictures: the map (mace_qd's map.png), a sheet of start | end stills for the clearest aim wins and losses, and
   GIFs: start | end side by side for the 3 clearest wins and 3 clearest losses (ranked by the aim duel, chosen before
   looking at them), and the 4 top-rated worlds of the neutral-ground tournament.
From papers: Bradley-Terry (Hunter 2004), Swiss pairing. Ours: the checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import fullrun as FR
import mace_qd as QD
import stage1 as S1
from lenia import backend, render
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, load_env
from lenia.sim import Rule


def _binom_greater(k: int, n: int) -> float:
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def finished_rounds(F: Path) -> tuple:
    done, partial = [], []
    for R in sorted((F / "rounds").glob("r*"), key=lambda p: int(p.name[1:])):
        f = R / "exp" / "state.json"
        if f.exists():
            est = json.loads(f.read_text())
            (done if est["step"] >= est["end"] else partial).append((int(R.name[1:]), R, est))
    return done, partial


def round_table(F: Path, done: list) -> list:
    seen_starts, seen_roots, L = set(), set(), []
    L += ["| round | triggers: map / archive / feedback | exact repeats | repeated lineages | routes clip / floor | "
          "review: continue / refine / switch / none | end goal, mean | end complexity, mean | moved | feedback |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r, R, est in done:
        T = est["triggers"]
        src = Counter(t["src"] for t in T)
        rep = sum(t["start_id"] in seen_starts for t in T)
        lin = sum(t["root"] in seen_roots for t in T)
        rv = Counter(t.get("review", {}).get("applied", "none") if t.get("review", {}).get("decision") != "none" else "none"
                     for t in T)
        fb = Counter(x["outcome"] for x in json.loads((R / "feedback.json").read_text())) if (R / "feedback.json").exists() else {}
        L.append(f"| {r} | {src.get('map', 0)} / {src.get('archive', 0)} / {src.get('feedback', 0)} | {rep} | {lin} | "
                 f"{sum(t['route'] == 'clip' for t in T)} / {sum(t['route'] == 'floor' for t in T)} | "
                 f"{rv.get('continue', 0)} / {rv.get('refine', 0)} / {rv.get('switch', 0)} / {rv.get('none', 0)} | "
                 f"{np.mean([t['g'] for t in T]):.2f} | {np.mean([t['k'] for t in T]):.2f} | "
                 f"{sum(bool(t['history']) for t in T)}/{len(T)} | {dict(fb)} |")
        seen_starts |= {t["start_id"] for t in T}
        seen_roots |= {t["root"] for t in T}
    return L


def run(a) -> None:
    F = Path(a.out)
    C = F / "check"
    (C / "gif").mkdir(parents=True, exist_ok=True)
    MAP = F / "map"
    st = json.loads((F / "state.json").read_text())
    sm = json.loads((MAP / "state.json").read_text())
    am = QD.settings(sm)
    pid = QD.entry_prompt(am)
    backend.configure(a.backend)
    done, partial = finished_rounds(F)
    if not done:
        sys.exit("no round's expeditions have finished yet")
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                  C / "judge_cache.sqlite", yes=True)
    ea = S1.eval_settings(MAP, argparse.Namespace(sim_batch=a.sim_batch))
    evals = S1.Evals(Path(a.evals) if a.evals else F / "evals", ea, keep=F / "evals")
    seed = st["args"]["seed"]

    # ---- 1. per-round table
    L = [f"# Check of {F.name} (seed {seed})\n",
         f"Rounds with finished expeditions: {', '.join(str(r) for r, _, _ in done)}"
         + (("; skipped (unfinished): " + ", ".join(f"{r} (step {e['step']} of {e['end']})" for r, _, e in partial)) if partial else "")
         + f". Map at iteration {sm['it'] - 1}. Aim: \"{am.aim}\".\n", "## 1. Rounds (no calls)\n"]
    L += round_table(F, done)

    # ---- 2. aim duel: end vs start, control vs start
    exps = [(r, R, tr) for r, R, est in done for tr in est["triggers"]]
    ctrl_rules = []
    for r, R, tr in exps:
        js = {}
        for x in FR.read_jsonl(R / "exp" / "judged.jsonl"):
            js[(x["trigger"], x["step"], x["child"])] = x
        pool = sorted((x for x in js.values() if x["trigger"] == tr["id"] and x["step"] == 1 and not x["picked"]),
                      key=lambda x: x["child"])
        if not pool:
            ctrl_rules.append(None)
            continue
        ctrl_rules.append(pool[int(np.random.default_rng([seed, r, tr["rank"], 99]).integers(len(pool)))]["rule"])
    todo = [Rule.from_dict(x) for x in ctrl_rules if x is not None]
    got = iter(evals.get_many(todo) if todo else [])
    ctrl_ev = [next(got) if x is not None else None for x in ctrl_rules]
    pics, pairs, meta = {}, [], []
    for (r, R, tr), ce in zip(exps, ctrl_ev):
        s = f"start:{tr['id']}"
        pics[s] = FR.pic_of(F / tr["start"])
        if tr["history"]:
            pics[f"end:{tr['id']}"] = FR.pic_of(F / tr["parent"])
            pairs.append((f"end:{tr['id']}", s))
            meta.append(("end", r, tr))
        if ce is not None:
            pics[f"ctrl:{tr['id']}"] = ce["pic"]
            pairs.append((f"ctrl:{tr['id']}", s))
            meta.append(("control", r, tr))
    ps = S1.duels(judge, pid, pairs, pics, "aim: end/control vs start")
    if any(p is None for p in ps):
        sys.exit("some aim duels failed (not cached); re-run to retry them")
    rows = [{"kind": k, "round": r, "trigger": tr["id"], "route": tr["route"], "src": tr["src"],
             "review": (tr.get("review") or {}).get("applied", "-"), "goal": tr["g"], "complexity": tr["k"],
             "aim_p": round(p, 4)} for (k, r, tr), p in zip(meta, ps)]
    with open(C / "aim.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def line(name, xs):
        if not xs:
            return f"| {name} | 0 | - | - | - | - |"
        n, k = len(xs), sum(x > 0.5 for x in xs)
        return (f"| {name} | {n} | {np.mean(xs):.3f} | {k}/{n} | {sum(x >= 0.6 for x in xs)}/{n} | "
                f"{_binom_greater(k, n):.3f} |")
    E = [x["aim_p"] for x in rows if x["kind"] == "end"]
    K = [x["aim_p"] for x in rows if x["kind"] == "control"]
    L += ["", "## 2. Aim duel against the expedition's own start\n",
          f"P = Jev's probability that the world is chosen over its start, on the map's entry prompt (the aim), both "
          f"orders. Control = a random unpicked judged step-1 child.\n",
          "| group | n | mean P | P > 0.5 | P ≥ 0.6 (the map's bar) | sign test p (one-sided) |", "|---|---|---|---|---|---|",
          line("ends", E), line("controls", K)]
    for key, vals in (("route", ("clip", "floor")), ("review", ("continue", "refine", "switch", "-")),
                      ("round", sorted({x["round"] for x in rows}))):
        for v in vals:
            L.append(line(f"ends, {key} {v}", [x["aim_p"] for x in rows if x["kind"] == "end" and x[key] == v]))
    gk = [(x["goal"], x["aim_p"]) for x in rows if x["kind"] == "end"]
    if len(gk) > 2:
        rho = S1._spearman([g for g, _ in gk], [p for _, p in gk])
        L.append(f"\nRank correlation between Jev's goal score (end vs start) and the aim P: "
                 f"{'-' if rho is None else f'{rho:.2f}'}.")

    # ---- 3. ends on neutral ground
    lrows = FR.ledger_rows(MAP)
    info = {x["id"]: x for x in lrows if x["kind"] != "surprise"}
    edges = [(x["id"], x["opponent"], float(x["p"])) for x in lrows
             if x["kind"] in ("duel", "duel_new") and x.get("p") is not None and x.get("opponent")]
    has = lambda i: (MAP / "img" / f"{i}.png").exists()
    ids = sorted({i for e in edges for i in e[:2]} | {i for i in info if has(i)})
    r0 = FR.bt_fit(edges, ids)
    champ = {}
    for i in ids:
        x = info.get(i)
        if x is None or not has(i) or x.get("cell") in (None, ""):
            continue
        c = (x["island"], x["cell"])
        if c not in champ or r0[i] > r0[champ[c]]:
            champ[c] = i
    fin = {f"fb_r{r}_{tr['rank']:02d}" for r, _, est in done for tr in est["triggers"] if tr["history"]}
    ends = sorted(i for i in fin if has(i))
    S = sorted(set(champ.values()) | set(ends))
    spics = {i: FR.pic_of(MAP / "img" / f"{i}.png") for i in S}
    play = lambda prs, t: S1.duels(judge, pid, prs, spics, f"neutral-ground swiss round {t}")
    games = FR.swiss(S, r0, a.swiss, np.random.default_rng([seed, 37]), play)
    r1 = FR.bt_fit(edges + [(x, y, p) for x, y, p, _ in games], ids)
    order = sorted(S, key=lambda i: -r1[i])
    rank = {i: n + 1 for n, i in enumerate(order)}
    with open(C / "swiss.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "rating", "is_end", "champion", "source", "root", "cell"])
        for i in order:
            w.writerow([rank[i], i, round(r1[i], 4), int(i in fin), int(i in champ.values()), info[i].get("source"),
                        info[i].get("root"), info[i].get("cell")])
    er = [rank[i] for i in ends]
    n = len(S)
    top = lambda q: sum(x <= q * n for x in er)
    exp_top = lambda q: len(er) * q
    L += ["", "## 3. Ends on neutral ground\n",
          f"{len(S)} worlds ({len(set(champ.values()))} cell champions + {len(ends)} expedition ends, "
          f"{len(set(champ.values()) & set(ends))} ends are champions too), {len(games)} Swiss games of aim duels "
          f"({a.swiss} rounds), Bradley–Terry refitted on these and the ledger's {len(edges)} entry duels.\n",
          f"- Ends' median rank: {int(np.median(er))} of {n} (a random draw would sit at {n // 2}).",
          f"- Ends in the top 10%: {top(0.1)} (chance: {exp_top(0.1):.1f}); top 25%: {top(0.25)} (chance: {exp_top(0.25):.1f}); "
          f"bottom 25%: {sum(x > 0.75 * n for x in er)} (chance: {exp_top(0.25):.1f}).",
          f"- Top 10 overall: " + ", ".join(f"{i} ({'end' if i in fin else info[i].get('source')}, L{info[i].get('root')})"
                                         for i in order[:10]) + "."]

    # ---- 4. pictures and GIFs
    if (MAP / "map.png").exists():
        shutil.copy(MAP / "map.png", C / "map.png")
    er_rows = sorted([x for x in rows if x["kind"] == "end"], key=lambda x: -x["aim_p"])
    wins = er_rows[:3]
    losses = er_rows[max(3, len(er_rows) - 3):][::-1]           # never the same expedition twice
    byid = {tr["id"]: (r, tr) for r, _, tr in exps}
    picks = [("win", x) for x in wins] + [("loss", x) for x in losses]
    font = S1._font(15)
    px = 300
    sheet = Image.new("RGB", (2 * px + 24, len(picks) * (px + 46) + 8), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    for n_, (kind, x) in enumerate(picks):
        r, tr = byid[x["trigger"]]
        y = 8 + n_ * (px + 46)
        d.text((8, y), f"{kind}: {tr['id']}  aim P {x['aim_p']:.2f}  goal {tr['g']:.2f}  [{x['review']}]", fill="black", font=font)
        d.text((8, y + 18), tr["goal"]["push"][:80], fill=(90, 90, 90), font=S1._font(12))
        for c, rel in enumerate((tr["start"], tr["parent"])):
            sheet.paste(Image.open(F / rel).convert("RGB").resize((px, px)), (8 + c * (px + 8), y + 40))
    sheet.save(C / "pairs.png")
    films_for = []
    for kind, x in picks:
        r, tr = byid[x["trigger"]]
        films_for += [Rule.from_dict(tr["start_rule"]), Rule.from_dict(tr["parent_rule"])]
    tops = order[:4]
    films_for += [Rule.from_dict(info[i]["rule"]) for i in tops]
    films = S1.v4_films(films_for, ea, a.every)

    def frames_of(f, snaps):
        scale = max(float(np.percentile(snaps, 99.5)), 1e-6)
        return [Image.fromarray(render.colorize(np.clip(z / scale, 0, 1))).resize((a.px, a.px), Image.BILINEAR) for z in f]

    def save_gif(ims, path):
        ims = [im.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for im in ims]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0, optimize=True)

    gifs = []
    for n_, (kind, x) in enumerate(picks):
        (fa, _, sa), (fb, _, sb) = films[2 * n_], films[2 * n_ + 1]
        A, B = frames_of(fa, sa), frames_of(fb, sb)
        out = []
        for ia, ib in zip(A, B):
            im = Image.new("RGB", (2 * a.px + 8, a.px + 22), (20, 20, 24))
            im.paste(ia, (0, 22))
            im.paste(ib, (a.px + 8, 22))
            ImageDraw.Draw(im).text((4, 4), f"{x['trigger']}  start | end   aim P {x['aim_p']:.2f}", fill=(230, 230, 230))
            out.append(im)
        p = C / "gif" / f"{n_ + 1}_{kind}_{x['trigger']}.gif"
        save_gif(out, p)
        gifs.append(p.name)
    for n_, i in enumerate(tops):
        f, _, sn = films[2 * len(picks) + n_]
        p = C / "gif" / f"top{n_ + 1}_{i}.gif"
        save_gif(frames_of(f, sn), p)
        gifs.append(p.name)
    L += ["", "## 4. Pictures\n", "`map.png` (the map at its last iteration), `pairs.png` (start | end, the 3 clearest aim wins "
          "and losses), GIFs in `gif/`: " + ", ".join(gifs) + ".",
          "", f"Spend: Jev ${judge.cache.total_cost():.3f} in this check's cache."]
    (C / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the full run's folder")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--evals", default=None)
    ap.add_argument("--sim-batch", type=int, default=0)
    ap.add_argument("--judge", default="jev")
    ap.add_argument("--config", default=None)
    ap.add_argument("--swiss", type=int, default=4)
    ap.add_argument("--every", type=int, default=30, help="simulation steps per GIF frame")
    ap.add_argument("--px", type=int, default=240)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-usd", type=float, default=0.5)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
