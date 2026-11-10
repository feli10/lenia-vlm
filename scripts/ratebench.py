"""Rating benchmark: can Jev make Felix's calls from stills? (11-10; project doc "Jev motion benchmark - spec", Stage B)

Jev must make the calls Felix makes from films while seeing only stills. We regrow the worlds Felix rated blind in the
seed-1 session, show Jev each pair he decided as five different still pictures with two different questions, and score
agreement with his choices against R0 (seed 1's picture and question) and a free change measure. A configuration that
clearly beats R0 is confirmed later on a fresh set of his ratings, since choosing on these same pairs flatters it.

  python scripts/ratebench.py build --out RUNS/rate-bench --source RUNS/full-s1 --backend auto   # Colab GPU (cell 10)
  PYTHONPATH=src:scripts python3 scripts/ratebench.py judge --out RUNS/rate-bench                         # ~$0.15
  PYTHONPATH=src:scripts python3 scripts/ratebench.py report --out RUNS/rate-bench                        # no calls
  Offline dry run: build ... --limit 6, then judge --judge fake (device shell: --out in the VM's $HOME).

The two dimensions (every combination, both orders):
- picture: R0 (the pipeline's 2x2 of four snapshots, byte-identical; build checks every world against the stills Felix
  was shown), line (4 left to right), ghosts (7 overlaid, earlier fainter), labels (line + time labels + coordinate
  grid), grid9 (9 snapshots 150 steps apart, 3x3); each with its own layout description (lenia.prompts.BENCH_LAYOUT).
- question: the pipeline's own (INTEREST for interest pairs, the run's attain prompt for goal pairs), or the same with
  one sentence asking Jev to judge each world as a film (lenia.prompts.rate_prompt, BENCH_MOTION).
R0 with the pipeline's question is exactly what drove seed 1, so it should reproduce fullrate's judges result.

Worlds: fullrate.build's pair selection is deterministic, so it is replayed (no growing) to recover each rated world's
rule; build stops unless the replay reproduces the saved tasks.json exactly. Truth: Felix's left / right answers;
"neither" pairs are left out. Build on Colab's GPU, where the session's worlds were
grown: these worlds are chaotic, and a CPU regrowth differs visibly after 2,700 steps (mean pixel difference 8-11 on 11-10). The free baseline picks the world that changes more between its four snapshots.
From papers: none. Ours: the design.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import motionbench as MB
import stage1 as S1
from lenia import backend, render
from lenia import mace as ME
from lenia import prompts as PR
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, Query, load_env
from lenia.sim import Rule

FORMATS = ["R0", "line", "ghosts", "labels", "grid9"]
QUESTIONS = ["pipeline", "film"]


def recover_rules(F: Path, tmp: Path) -> dict:
    """Replay fullrate.build's selection (films stubbed out) and return {world name: rule}; exit unless it reproduces
    the saved session's tasks exactly."""
    import fullrate as FR
    ex = pathlib.Path.exists

    def safe_exists(self):                                 # a Drive file not downloaded here: skip the replay check
        try:
            return ex(self)
        except OSError:
            return False
    cap = {}

    def stub(rules, ea, every):
        cap["rules"] = [r.to_dict() for r in rules]
        return [([], np.zeros((2 * ME.TILE + 4, 2 * ME.TILE + 4, 3), np.uint8), None) for _ in rules]
    keep = (pathlib.Path.exists, S1.v4_films, S1._gif)
    pathlib.Path.exists, S1.v4_films, S1._gif = safe_exists, stub, (lambda *x, **k: None)
    try:
        FR.build(argparse.Namespace(out=str(F), dir=str(tmp), force=True, backend="numpy", exp=16, mid=6, rep=10,
                                    rep_after=20, cross=10, every=30, px=280, fps=15, size=None, warmup=None,
                                    render=None, gap=None, sim_batch=None))
    finally:
        pathlib.Path.exists, S1.v4_films, S1._gif = keep
    new = json.loads((tmp / "tasks.json").read_text())
    old = json.loads((F / "rate" / "tasks.json").read_text())
    key = lambda t: (t["id"], t["group"], t["a"], t["b"], t["left"], t["right"])
    if [key(t) for t in new] != [key(t) for t in old]:
        sys.exit("the replayed selection differs from the saved session (run files changed?): not guessing rules")
    names = sorted({w for t in old for w in (t["a"], t["b"])})
    if len(names) != len(cap["rules"]):
        sys.exit(f"{len(names)} worlds but {len(cap['rules'])} rules captured")
    return dict(zip(names, cap["rules"]))


def build(a) -> None:
    F, out = Path(a.source), Path(a.out)
    if (out / "items.json").exists() and not a.force:
        sys.exit(f"{out / 'items.json'} exists (--force rebuilds)")
    out.mkdir(parents=True, exist_ok=True)
    rules = recover_rules(F, out / "replay")
    tasks = json.loads((F / "rate" / "tasks.json").read_text())
    ans = {int(r["id"]): r["answer"] for r in csv.DictReader(open(F / "rate" / "ratings.csv"))}
    files = {}
    for t in tasks:
        files[t["left_world"]], files[t["right_world"]] = t["left"], t["right"]
    names = sorted(rules)
    if a.limit:                                            # dry runs: the worlds of the first decided pairs
        pick = []
        for t in tasks:
            if ans.get(t["id"]) not in (None, "neither") and len(pick) < a.limit:
                pick += [w for w in (t["a"], t["b"]) if w not in pick]
        names = sorted(pick)
    backend.configure(a.backend)
    ea = S1.eval_settings(F / "map", a)
    steps, gap, t0 = MB.window(ea)
    times = [t0 + k * gap // 8 for k in range(MB.NFILM)]
    (out / "frames").mkdir(exist_ok=True)
    change, mismatch = {}, []
    for w, film in zip(names, MB.grow([Rule.from_dict(rules[w]) for w in names], ea, times, steps)):
        snaps = film[[0, 8, 16, 24]]
        scale = max(float(np.percentile(snaps, 99.5)), 1e-6)
        col = np.stack([render.colorize(np.clip(f / scale, 0, 1)) for f in film])       # all 25 frames
        pic = MB.fmt_grid(col[::2])
        try:
            shown = np.asarray(Image.open(F / "rate" / "images" / f"{files[w]}.png").convert("RGB"))
            if not np.array_equal(shown, pic):
                mismatch.append((w, float(np.abs(shown.astype(int) - pic.astype(int)).mean())))
        except OSError:
            mismatch.append((w, None))
        np.savez_compressed(out / "frames" / f"{MB.fid(w)}.npz", f=col)
        change[w] = MB.change(snaps)
    if mismatch and not a.limit:
        sys.exit(f"{len(mismatch)} worlds don't match the stills Felix was shown, e.g. {mismatch[:3]}: not writing items")
    items = []
    for t in tasks:
        x = ans.get(t["id"])
        if x in (None, "neither") or t["a"] not in change or t["b"] not in change:
            continue
        win = t["left_world"] if x == "left" else t["right_world"]
        items.append({"id": t["id"], "group": t["group"], "question": t["question"], "a": t["a"], "b": t["b"],
                      "felix": win, "sees": t.get("sees"), "push": t.get("push"),
                      "change": {t["a"]: round(change[t["a"]], 5), t["b"]: round(change[t["b"]], 5)}})
    meta = {"source": str(F), "settings": {k: getattr(ea, k) for k in ("size", "R", "warmup", "render", "gap", "start_seed")},
            "worlds": len(names), "mismatch": mismatch}
    (out / "items.json").write_text(json.dumps({"meta": meta, "items": items}, indent=1))
    n = defaultdict(int)
    for t in items:
        n[t["question"]] += 1
    print(f"items.json: {len(items)} decided pairs ({dict(n)}); {len(names)} worlds regrown; "
          f"{len(mismatch)} differ from the stills Felix saw", file=sys.stderr)


def grid9(col: np.ndarray) -> np.ndarray:
    px = (2 * ME.TILE + 4 - 8) // 3                          # 212: the same 644-px image as the 2x2
    o = np.full((3 * px + 8, 3 * px + 8, 3), MB.GAPC, np.uint8)
    for j, k in enumerate(range(0, MB.NFILM, 3)):
        y, x = (j // 3) * (px + 4), (j % 3) * (px + 4)
        o[y:y + px, x:x + px] = MB._tile(col[k], px)
    return o


def picture(col: np.ndarray, fmt: str, s: dict) -> np.ndarray:
    c13 = col[::2]
    if fmt == "R0":
        return MB.fmt_grid(c13)
    if fmt == "labels":
        return MB.fmt_labels(c13, min(s["gap"], (s["warmup"] + s["render"]) // 3), s["size"])
    if fmt == "grid9":
        return grid9(col)
    return {"line": MB.fmt_line, "ghosts": MB.fmt_ghosts}[fmt](c13)


def base_prompt(t: dict) -> str:
    if t["question"] == "interest":
        return PR.INTEREST
    return PR.attain_prompt({"id": S1.goal_id(t["sees"], t["push"]), "sees": t["sees"], "push": t["push"]})


def judge(a) -> None:
    out = Path(a.out)
    D = json.loads((out / "items.json").read_text())
    items, s = D["items"], D["meta"]["settings"]
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    J = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
              out / f"judge_cache_{a.judge}.sqlite", yes=True)
    ws = sorted({w for t in items for w in (t["a"], t["b"])})
    cols = {w: np.load(out / "frames" / f"{MB.fid(w)}.npz")["f"] for w in ws}
    res = {}
    for fmt in FORMATS:
        P = {w: picture(cols[w], fmt, s) for w in ws}
        for qn in QUESTIONS:
            qs = []
            for t in items:
                pid = PR.rate_prompt(fmt, base_prompt(t), qn == "film")
                qs += [Query(pid, P[t["a"]], P[t["b"]]), Query(pid, P[t["b"]], P[t["a"]])]
            vs = J.judge_many(qs, desc=f"{fmt}/{qn}")
            res[f"{fmt}/{qn}"] = {t["id"]: {"p1": x.p_first, "p2": y.p_first}
                                  for t, x, y in zip(items, vs[::2], vs[1::2]) if x.ok and y.ok}
    (out / "results.json").write_text(json.dumps({"judge": J.model, "results": res}, indent=1))
    print(f"spent ${J.cache.total_cost():.3f} in total (this benchmark's {a.judge} cache)", file=sys.stderr)
    report(a)


def report(a) -> None:
    out = Path(a.out)
    items = {t["id"]: t for t in json.loads((out / "items.json").read_text())["items"]}
    R = json.loads((out / "results.json").read_text())["results"]
    by_q = {q: [i for i, t in items.items() if t["question"] == q] for q in ("interest", "goal")}
    by_q["all"] = list(items)

    def agree(r, t):                                       # P(Felix's world), both orders averaged; ties half
        p = 0.5 * (r["p1"] + 1 - r["p2"]) if t["felix"] == t["a"] else 0.5 * (r["p2"] + 1 - r["p1"])
        return 1.0 if p > 0.5 else 0.5 if p == 0.5 else 0.0

    def acc(name, q):
        r = R[name]
        v = [agree(r[str(i)], items[i]) for i in by_q[q] if str(i) in r]
        return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)

    def wilson(p, n):
        z = 1.96
        c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return f"{c - h:.2f}–{c + h:.2f}"
    n = {q: len(v) for q, v in by_q.items()}
    L = [f"# Rating benchmark: Jev vs Felix on his seed-1 pairs ({n['all']} decided: interest {n['interest']}, "
         f"goal {n['goal']})\n",
         "Agreement = Jev's both-orders P(Felix's world) > 0.5, ties half; chance 0.50. R0 / pipeline = seed 1 exactly.\n",
         "| picture | question | interest | goal | all | all, 95% CI | orders agree |", "|---|---|---|---|---|---|---|"]
    for fmt in FORMATS:
        for qn in QUESTIONS:
            k = f"{fmt}/{qn}"
            if k not in R:
                continue
            ai, ag, aa = acc(k, "interest"), acc(k, "goal"), acc(k, "all")
            rr = list(R[k].values())
            oa = np.mean([(x["p1"] > 0.5) == (x["p2"] < 0.5) for x in rr]) if rr else float("nan")
            L.append(f"| {fmt} | {qn} | {ai[0]:.2f} | {ag[0]:.2f} | {aa[0]:.2f} | {wilson(aa[0], aa[1])} | {oa:.2f} |")
    fc = {q: np.mean([1.0 if t["change"][t["felix"]] > max(v for w, v in t["change"].items() if w != t["felix"]) else 0.0
                      for i, t in items.items() if i in set(by_q[q])]) for q in ("interest", "goal", "all")}
    L.append(f"| *free change measure* | – | {fc['interest']:.2f} | {fc['goal']:.2f} | {fc['all']:.2f} | | |")
    L += ["\n## Each dimension on its own (mean over the other)\n", "| | interest | goal | all |", "|---|---|---|---|"]
    for lab, keys in [(f"picture {f}", [f"{f}/{q}" for q in QUESTIONS]) for f in FORMATS] + \
                     [(f"question {q}", [f"{f}/{q}" for f in FORMATS]) for q in QUESTIONS]:
        keys = [k for k in keys if k in R]
        if keys:
            L.append(f"| {lab} | " + " | ".join(f"{np.mean([acc(k, q)[0] for k in keys]):.2f}"
                                                for q in ("interest", "goal", "all")) + " |")
    if "R0/pipeline" in R:
        L += ["\n## Paired against R0 / pipeline (all pairs)\n", "| config | right where R0 wrong | wrong where R0 right | p |",
              "|---|---|---|---|"]
        r0 = R["R0/pipeline"]
        for k in R:
            if k == "R0/pipeline":
                continue
            both = [i for i in items if str(i) in R[k] and str(i) in r0]
            b = sum(agree(R[k][str(i)], items[i]) == 1 and agree(r0[str(i)], items[i]) == 0 for i in both)
            c = sum(agree(R[k][str(i)], items[i]) == 0 and agree(r0[str(i)], items[i]) == 1 for i in both)
            p = min(1.0, 2 * sum(math.comb(b + c, j) for j in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
            L.append(f"| {k} | {b} | {c} | {p:.3f} |")
    L.append("\nThese pairs pick the best configuration, so its number here is optimistic: confirm it on fresh ratings.")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "judge", "report"):
        sub.add_parser(c).add_argument("--out", required=True, help="the benchmark folder")
    b = sub.choices["build"]
    b.add_argument("--source", required=True, help="the full run with Felix's rating session (RUNS/full-s1)")
    b.add_argument("--backend", default="numpy")
    b.add_argument("--force", action="store_true")
    b.add_argument("--limit", type=int, default=0, help="dry runs: grow only this many worlds")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        b.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (dry runs only)")
    j = sub.choices["judge"]
    j.add_argument("--judge", default="jev")
    j.add_argument("--config", default=None)
    j.add_argument("--max-usd", type=float, default=0.4, help="hard cap on this benchmark's judge cache in total")
    a = ap.parse_args()
    {"build": build, "judge": judge, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
