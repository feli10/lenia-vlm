"""Map-prompt test (11-04; project doc "Handoff - Stage 1 aim-conditioned goals", road map step 2): does the base
search's entry duel do better with the user's aim as its goal than with "complex, life-like" (COMPLEXITY)? Felix 11-03:
"complex" may reward texture. Two runs that differ only in that sentence: seed 0's base run, and an aim run made with
`mace_qd.py run --like <base> --aim "..."` (same seed, settings and form-axis fit; lenia.prompts.map_prompt).

  PYTHONPATH=src:scripts python3 scripts/maptest.py stats --runs runs/mace-qd5-sarch-s0,runs/mace-qd5-sarch-s0-aim
  python scripts/maptest.py build --base RUNS/mace-qd5-sarch-s0 --aim RUNS/mace-qd5-sarch-s0-aim --backend auto  # Colab
  PYTHONPATH=src:scripts python3 scripts/maptest.py rate --aim runs/mace-qd5-sarch-s0-aim                        # Mac
  PYTHONPATH=src:scripts python3 scripts/maptest.py report --aim runs/mace-qd5-sarch-s0-aim

stats (no calls, no GPU): per run, the grid-noise rate among candidates (texture > --max-texture, rejected before any
duel), the texture of accepted map entries early and late, the final elites (texture median and share above 0.15 and
0.25, motion, bodies, cells, lineages), the surprise archive, and the judge spend per prompt.
build (GPU, no calls): --pairs blind interest pairs, each the two runs' final elites in the same island and cell (the
cells mean the same in both: --like copies the form fit), as films above stills (stage1.v4_films; replay check as in
v4rate). Writes <aim run>/maptest/. report: win rates (stage1.v4rate_report), plus how often the more textured world won.
Matched cells compare quality within a niche; coverage and texture over the whole map are in stats.
Ours: the test design. From papers: MAP-Elites (Mouret & Clune 2015), the novelty archive (Lehman & Stanley 2011).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import mace_qd as QD
import stage1 as S1
from lenia import backend
from lenia.sim import Rule

TEX_HI = (0.15, 0.25)


def _prompt_name(st: dict) -> str:
    aim = st["args"].get("aim")
    return f"aim: {aim}" if aim else "complex, life-like (COMPLEXITY)"


def run_stats(run: Path) -> dict:
    st = json.loads((run / "state.json").read_text())
    a = QD.settings(st)
    rows = list(csv.DictReader(open(run / "log.csv")))
    cand = [r for r in rows if int(r["it"]) >= 1]                  # iteration 0 = the seeds, the same in both runs
    acc = [r for r in cand if r["accepted"] == "1" or r["kind"] == "new"]
    last = max(int(r["it"]) for r in rows)
    half = lambda rs, lo, hi: [float(r["texture"]) for r in rs if lo <= int(r["it"]) <= hi and r["texture"] != ""]
    el = [e for m in QD.islands_of(st) for e in m.values()]
    tex = np.array([e["texture"] for e in el])
    spend = {}
    if (run / "judge_cache.sqlite").exists():
        c = sqlite3.connect(f"file:{run / 'judge_cache.sqlite'}?mode=ro", uri=True)
        spend = {p: (n, round(u or 0, 3)) for p, n, u in c.execute("SELECT prompt_id, count(*), sum(cost) FROM verdicts GROUP BY 1")}
    med = lambda x: float(np.median(x)) if len(x) else float("nan")
    return {"run": run.name, "prompt": _prompt_name(st), "iterations": st["it"] - 1,
            "grid_noise_rate": round(sum(r["kind"] == "texture" for r in cand) / max(1, len(cand)), 4),
            "max_texture": a.max_texture, "entries": len(acc),
            "entry_texture_early": round(med(half(acc, 1, last // 2)), 4),
            "entry_texture_late": round(med(half(acc, last // 2 + 1, last)), 4),
            "elites": len(el), "lineages": QD.lineages(st),
            "elite_texture_median": round(med(tex), 4),
            **{f"elite_texture_over_{t}": round(float((tex > t).mean()), 3) for t in TEX_HI},
            "elite_motion_median": round(med([e["motion"] for e in el]), 4),
            "elite_bodies_median": med([e["bodies"] for e in el]),
            "sarch": len(st.get("sarch", [])),
            "sarch_texture_median": round(med([e["texture"] for e in st.get("sarch", [])]), 4),
            "spend": spend}


def stats(a) -> None:
    R = [run_stats(Path(r)) for r in a.runs.split(",")]
    keys = [k for k in R[0] if k not in ("run", "spend")]
    L = ["# Map-prompt test: map statistics (no calls)\n", "| | " + " | ".join(r["run"] for r in R) + " |",
         "|---|" + "---|" * len(R)]
    L += [f"| {k} | " + " | ".join(str(r[k]) for r in R) + " |" for k in keys]
    L.append("| judge spend (prompt: calls, $) | " + " | ".join("; ".join(f"{p}: {n}, ${u}" for p, (n, u) in r["spend"].items())
                                                              for r in R) + " |")
    L.append("\ngrid_noise_rate: candidates (iterations 1+) rejected for texture before any duel. entry_texture: median "
             "texture of accepted map entries, first vs second half of the run. Lower texture is not better in itself; "
             "the question is whether one prompt drifts toward texture.")
    print("\n".join(L))
    if a.dest:
        Path(a.dest).mkdir(parents=True, exist_ok=True)
        (Path(a.dest) / "stats.md").write_text("\n".join(L) + "\n")


def build(a) -> None:
    base, aim = Path(a.base), Path(a.aim)
    rdir = aim / "maptest"
    if (rdir / "tasks.json").exists() and not a.force:
        sys.exit(f"{rdir / 'tasks.json'} exists (--force rebuilds it, but only before any rating)")
    if a.force and (rdir / "ratings.csv").exists():
        sys.exit(f"{rdir / 'ratings.csv'} exists: ratings were made against this build; not rebuilding")
    sb, sa = (json.loads((r / "state.json").read_text()) for r in (base, aim))
    if not sa["args"].get("aim") or sa["args"].get("like") is None:
        sys.exit(f"{aim} is not an aim run made with --like")
    if sa.get("form") != sb.get("form"):
        sys.exit("the two runs' form-axis fits differ: cells don't mean the same thing")
    MB, MA = QD.islands_of(sb), QD.islands_of(sa)
    both = [(k, ck) for k in range(len(MA)) for ck in sorted(set(MA[k]) & set(MB[k]))]
    cells = [(k, ck) for k, ck in both if MA[k][ck]["rule"] != MB[k][ck]["rule"]]   # same elite in both: nothing to rate
    rng = np.random.default_rng([a.seed, 505])
    pick = [cells[int(j)] for j in rng.choice(len(cells), min(a.pairs, len(cells)), replace=False)]
    print(f"{len(both)} cells filled in both maps, {len(both) - len(cells)} with the same elite in both; "
          f"{len(pick)} pairs from the other {len(cells)}", file=sys.stderr)
    W = {}
    for k, ck in pick:
        for tag, run, M in (("aim", aim, MA), ("complex", base, MB)):
            e = M[k][ck]
            W[f"{tag}:I{k}:{ck}"] = {"rule": e["rule"], "still": run / "img" / e["image"], "texture": e["texture"]}
    backend.configure(a.backend)
    ea = S1.eval_settings(base, a)
    names = sorted(W)
    res = S1.v4_films([Rule.from_dict(W[n]["rule"]) for n in names], ea, a.every)
    (rdir / "images").mkdir(parents=True, exist_ok=True)
    salt = rng.integers(1 << 30)
    replay = {}
    for n, (frames, pic, snaps) in zip(names, res):
        h = hashlib.sha1(f"{n}|{salt}".encode()).hexdigest()[:12]
        W[n]["file"] = h
        ref = np.asarray(Image.open(W[n]["still"]).convert("RGB"), np.int16)
        d = float(np.abs(ref - pic.astype(np.int16)).mean()) if ref.shape == pic.shape else float("inf")
        replay[n] = round(d, 2)
        (Image.open(W[n]["still"]).convert("RGB") if d <= S1.REPLAY_TOL else Image.fromarray(pic)).save(rdir / "images" / f"{h}.png")
        S1._gif(frames, snaps, rdir / "images" / f"{h}.gif", a.px, a.fps)
    out = []
    for k, ck in [pick[int(j)] for j in rng.permutation(len(pick))]:
        x, y = f"aim:I{k}:{ck}", f"complex:I{k}:{ck}"
        l, r = (x, y) if rng.random() < 0.5 else (y, x)
        out.append({"id": len(out), "group": "interest:aim-vs-complex", "question": "interest", "trigger": f"I{k}:{ck}",
                    "a": x, "b": y, "sees": None, "push": None, "left": W[l]["file"], "right": W[r]["file"],
                    "left_world": l, "right_world": r, "texture": {"a": W[x]["texture"], "b": W[y]["texture"]}})
    (rdir / "tasks.json").write_text(json.dumps(out, indent=1))
    bad = {n: d for n, d in replay.items() if d > S1.REPLAY_TOL}
    (rdir / "replay.json").write_text(json.dumps({"tol": S1.REPLAY_TOL, "diff": replay, "diverged": bad}, indent=1))
    print(f"wrote {rdir}: {len(out)} pairs; replay median diff {np.median(list(replay.values())):.2f}, {len(bad)} diverged "
          "(shown regrown)", file=sys.stderr)


def report(a) -> None:
    rdir = Path(a.aim) / "maptest"
    S1.v4rate_report(rdir, title="Map-prompt test (aim vs complex, life-like)", arms=(), controls=())
    tasks = {t["id"]: t for t in json.loads((rdir / "tasks.json").read_text())}
    R = {int(r["id"]): r["answer"] for r in csv.DictReader(open(rdir / "ratings.csv"))}
    tex_won = tex_n = aim_tex = aim_tex_won = smooth_won = smooth_n = 0
    for i, ans in R.items():
        t = tasks[i]
        if ans == "neither" or t["texture"]["a"] == t["texture"]["b"]:
            continue
        won = t["left_world"] if ans == "left" else t["right_world"]
        more = t["a"] if t["texture"]["a"] > t["texture"]["b"] else t["b"]
        tex_n += 1
        tex_won += won == more
        if more == t["a"]:
            aim_tex += 1
            aim_tex_won += won == t["a"]
        else:
            smooth_n += 1
            smooth_won += won == t["a"]
    L = [f"\nTexture (decided pairs): the more textured world won {tex_won} of {tex_n}.",
         f"- aim elite more textured: aim won {aim_tex_won} of {aim_tex}; aim elite less textured: aim won {smooth_won} of {smooth_n}."]
    with open(rdir / "report.md", "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--runs", required=True, help="comma-separated run folders")
    s.add_argument("--dest", default=None, help="also write stats.md here")
    b = sub.add_parser("build")
    b.add_argument("--base", required=True)
    b.add_argument("--aim", required=True)
    b.add_argument("--pairs", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--backend", default="numpy")
    b.add_argument("--every", type=int, default=30)
    b.add_argument("--px", type=int, default=280)
    b.add_argument("--fps", type=int, default=15)
    b.add_argument("--force", action="store_true")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        b.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    r = sub.add_parser("rate")
    r.add_argument("--aim", required=True)
    r.add_argument("--port", type=int, default=8774)
    p = sub.add_parser("report")
    p.add_argument("--aim", required=True)
    a = ap.parse_args()
    if a.cmd == "rate":
        from lenia.rating import rate_pairs
        return rate_pairs(Path(a.aim) / "maptest", port=a.port)
    {"stats": stats, "build": build, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
