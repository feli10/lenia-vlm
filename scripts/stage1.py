"""Stage 1 of the escalation loop (project doc "VLM in the loop - escalation plan (core first)").

Step 1, the trigger list:

  PYTHONPATH=src:scripts python3 scripts/stage1.py triggers --run runs/mace-qd5-sarch-s0 --out runs/stage1-s0 --max-usd 0.25
  PYTHONPATH=src:scripts python3 scripts/stage1.py triggers --run runs/mace-qd5-sarch-s0 --out /tmp/s1-dry --judge fake
  PYTHONPATH=src:scripts python3 scripts/stage1.py sheet --out runs/stage1-s0          # redraw triggers.png only

Triggers: every surprise-archive member of the base run is re-scored against the run's final map, and each
lineage (seed root) contributes its best member. A member's stored score was measured against the map and archive
of its own iteration (early members against near-random founders); re-scoring puts every member against one
reference, what the run eventually produced.
- Opponents: one panel of --opponents final-map elites per lineage, drawn uniformly from all islands with a fixed
  seed. Members of one lineage face the same panel, so the pick within a lineage is a paired comparison; panels of
  different lineages are independent uniform draws. Archive members are not opponents (a lineage would meet its
  own siblings).
- Score: the surprise duel (lenia.prompts.SURPRISE), both orders; P = mean over the panel of 0.5 (x + 1 - y).
- Pick: the highest score per lineage. No score floor (fixed before seeing the scores, Felix 10-29); picks below
  0.5 are marked on the sheet, and Felix's glance drops loophole worlds.
- Complexity (lenia.prompts.COMPLEXITY, the map's entry prompt) against the same panel, for the picks only: logged,
  never used to select.
Outputs in --out: judge_cache.sqlite (the seed's shared Stage 1 cache: size later --max-usd for all arms),
rescore.csv (every member), triggers.json (the start of the Tier 1 record), triggers.png (contact sheet).

From papers: the candidates come from a novelty-search archive with an adaptive threshold (Lehman & Stanley 2011).
Ours: the pairwise surprise duel as the novelty score, re-scoring against one final pool, one trigger per lineage,
the per-lineage opponent panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mace_qd as QD
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, Query, load_env
from lenia.prompts import COMPLEXITY, SURPRISE


def make_judge(a, out: Path) -> Judge:
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    return Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
                 out / "judge_cache.sqlite", yes=True)


def duels(judge: Judge, prompt: str, pairs: list, pics: dict, desc: str) -> list:
    """pairs: [(world, opponent)] image names -> [P(world chosen)], both orders averaged; None if a call failed."""
    qs = [q for w, o in pairs for q in (Query(prompt, pics[w], pics[o]), Query(prompt, pics[o], pics[w]))]
    vs = judge.judge_many(qs, desc=desc)
    return [0.5 * (x.p_first + 1 - y.p_first) if x.ok and y.ok else None for x, y in zip(vs[::2], vs[1::2])]


def triggers(a) -> None:
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads((run / "state.json").read_text())
    sar = st.get("sarch", [])
    ref = [e["image"] for m in QD.islands_of(st) for _, e in sorted(m.items())]
    if not sar or len(ref) < a.opponents:
        sys.exit(f"{run}: {len(sar)} archive members, {len(ref)} map elites; nothing to re-score")
    if len(sar) >= QD.settings(st).sarch_max:
        print(f"WARNING: the archive is at its cap ({len(sar)}); its oldest members were dropped", file=sys.stderr)
    names = sorted({e["image"] for e in sar} | set(ref))
    missing = [n for n in names if not (run / "img" / n).exists()]
    if missing:
        sys.exit(f"{len(missing)} pictures missing, e.g. {missing[0]} (rsync the run's img folder)")
    pics = {n: np.asarray(Image.open(run / "img" / n).convert("RGB")) for n in names}
    roots = sorted({e["root"] for e in sar})
    rng = np.random.default_rng([a.seed, 1])
    panel = {r: [ref[int(j)] for j in rng.choice(len(ref), a.opponents, replace=False)] for r in roots}

    judge = make_judge(a, out)
    pairs = [(e["image"], o) for e in sar for o in panel[e["root"]]]
    ps = duels(judge, SURPRISE, pairs, pics, "re-score surprise")
    if any(p is None for p in ps):
        sys.exit(f"{sum(p is None for p in ps)} duels failed (not cached); re-run to retry them")
    k = a.opponents
    rows = []
    for i, e in enumerate(sar):
        v = ps[i * k:(i + 1) * k]
        rows.append({"image": e["image"], "root": e["root"], "it": e["it"], "island": e["island"],
                     "stored": e["surprise"], "surprise": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4),
                     "bodies": e["bodies"], "motion": e["motion"], "coverage": e["coverage"]})
    size = {r: sum(x["root"] == r for x in rows) for r in roots}
    best = {}
    for x in rows:
        if x["root"] not in best or x["surprise"] > best[x["root"]]["surprise"]:
            best[x["root"]] = x
    for x in rows:
        x["rank_in_lineage"] = 1 + sum(y["root"] == x["root"] and y["surprise"] > x["surprise"] for y in rows)
    picks = sorted(best.values(), key=lambda x: -x["surprise"])

    cs = duels(judge, COMPLEXITY, [(x["image"], o) for x in picks for o in panel[x["root"]]], pics, "trigger complexity")
    by_img = {e["image"]: e for e in sar}
    trig = []
    for n, x in enumerate(picks):
        v = [c for c in cs[n * k:(n + 1) * k] if c is not None]
        e = by_img[x["image"]]
        trig.append({**x, "id": f"s0-L{x['root']}", "lineage_members": size[x["root"]],
                     "complexity": round(float(np.mean(v)), 4) if v else None, "opponents": panel[x["root"]],
                     "parent": e.get("parent"), "rule": e["rule"]})

    with open(out / "rescore.csv", "w", newline="") as fh:
        cols = ["image", "root", "it", "island", "stored", "surprise", "sd", "rank_in_lineage", "bodies", "motion", "coverage"]
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["root"], x["rank_in_lineage"])))
    (out / "triggers.json").write_text(json.dumps(
        {"run": str(run), "run_iteration": st["it"] - 1, "prompt": SURPRISE, "complexity_prompt": COMPLEXITY,
         "opponents": a.opponents, "seed": a.seed, "judge": judge.model, "snapshots": judge.cache.snapshots(),
         "triggers": trig}, indent=1))
    (out / "img").mkdir(exist_ok=True)                 # the sheet and later steps read trigger pictures from --out
    for t in trig:
        if not (out / "img" / t["image"]).exists():
            Image.fromarray(pics[t["image"]]).save(out / "img" / t["image"])
    sheet(a)
    rho = np.corrcoef([x["stored"] for x in rows], [x["surprise"] for x in rows])[0, 1]
    print(f"\n{len(rows)} members re-scored; corr(stored, re-scored) = {rho:.2f}; judge spent ${judge.spent:.4f} "
          f"(cache total ${judge.cache.total_cost():.4f}); snapshots {judge.cache.snapshots()}")
    print("trigger      it  isl  members  surprise (sd)  stored  complexity  bodies  coverage")
    for t in trig:
        c = "-" if t["complexity"] is None else f"{t['complexity']:.2f}"
        print(f"{t['id']:<11} {t['it']:>3}  I{t['island']}  {t['lineage_members']:>7}  {t['surprise']:.2f} ({t['sd']:.2f})"
              f"    {t['stored']:.2f}   {c:>9}  {t['bodies']:>6}  {t['coverage']:.2f}")
    print(out / "triggers.png")


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def sheet(a, cols: int = 4, px: int = 420) -> Path:
    """triggers.png: one 2x2 picture per trigger, highest re-scored surprise first."""
    out = Path(a.out)
    T = json.loads((out / "triggers.json").read_text())["triggers"]
    head = 46
    rows = (len(T) + cols - 1) // cols
    im = Image.new("RGB", (cols * (px + 8) + 8, rows * (px + head + 8) + 8), (250, 250, 250))
    d = ImageDraw.Draw(im)
    f1, f2 = _font(17), _font(14)
    for i, t in enumerate(T):
        x, y = 8 + (i % cols) * (px + 8), 8 + (i // cols) * (px + head + 8)
        low = t["surprise"] < 0.5
        d.text((x, y), f"{t['id']}  it{t['it']} I{t['island']}  ({t['lineage_members']} in lineage)", fill="black", font=f1)
        c = "-" if t["complexity"] is None else f"{t['complexity']:.2f}"
        d.text((x, y + 22), f"surprise {t['surprise']:.2f} (stored {t['stored']:.2f})  complexity {c}  "
               f"bodies {t['bodies']}  cov {t['coverage']:.2f}", fill=(190, 30, 30) if low else (60, 60, 60), font=f2)
        im.paste(Image.open(out / "img" / t["image"]).convert("RGB").resize((px, px), Image.BILINEAR), (x, y + head))
    p = out / "triggers.png"
    im.save(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triggers")
    t.add_argument("--run", required=True, help="the base run (mace_qd.py --sarch)")
    t.add_argument("--out", required=True, help="this seed's Stage 1 folder (shared judge cache)")
    t.add_argument("--judge", default="jev")
    t.add_argument("--config", default=None)
    t.add_argument("--opponents", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-usd", type=float, default=0.25, help="hard cap on the --out cache's total spend, across resumes")
    s = sub.add_parser("sheet")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"triggers": triggers, "sheet": sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
