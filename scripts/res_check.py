"""Gate 0f: does Jev judge as well on smaller pictures? (Jev bills ~1 token per 18x18 px; the 644-px MaCE
picture is 2x the world's resolution per side, so a smaller picture could cut every Jev call by 40-75%.)

Re-asks old, already-rated duels with the same pictures shrunk (box filter) to --sizes, and compares with the
644-px answers already in the caches:
  val   surprise duels from runs/surprise-val (Felix's 20 ratings), every rated world + random others
  cmp   surprise duels from runs/qd5-compare-s0 (Felix's 36 ratings; archive vs map), rated + random others
  comp  complexity duels from runs/mace-qd2/check (final vs mid-run elite, 89 pairs)
  retest  a subset of cmp re-asked at 644 px with a fresh sample: the run-to-run ceiling (also catches a
          change of the model behind `jev-latest` since the old runs)
Both orders throughout. Nothing is changed in the old runs; answers go to --out/judge_cache.sqlite.

  python scripts/res_check.py run --dry-run      # counts and estimated cost, no calls
  python scripts/res_check.py run                # ~$0.35, stops at --max-usd
  python scripts/res_check.py report             # (also printed after run) -> runs/res-check/report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lenia.config import REPO_ROOT, load
from lenia.judge import JEV_USD_PER_TOKEN, Judge, Query, image_hash, load_env
from lenia.prompts import COMPLEXITY as COMP, SURPRISE as SUR, get_prompt
from lenia.rating import label_answers
THR = 0.66          # the surprise archive's threshold in mace-qd5-sarch-s0 (median)


def tok(px: int) -> float:                      # measured: ~pixels / 324 per image + ~77 text
    return 2 * px * px / 324 + 77


def shrink(img: np.ndarray, px: int) -> np.ndarray:
    if px >= img.shape[0]:
        return img
    return np.asarray(Image.fromarray(img).resize((px, px), Image.BOX))


def load_img(p) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def sets(a) -> dict:
    """{name: {"pairs": [(focal_id, opp_id)], "img": {id: path}, "label": {id: 1/0}, "group": {id: str},
    "old": {id: 644-px score}, "prompt": pid}}"""
    rng = np.random.default_rng(a.seed)
    out = {}
    # val
    S = json.loads(Path("runs/surprise-val/scores.json").read_text())
    tasks = json.loads(Path("runs/surprise-val/tasks.json").read_text())
    ans = label_answers(Path("runs/surprise-val"))
    lab = {t["name"]: int(ans[t["id"]]["answer"] == "new") for t in tasks
           if t["id"] in ans and ans[t["id"]]["answer"] in ("new", "not_new")}
    rated = [t["name"] for t in tasks]
    rest = [p for p in S["pool"] if p not in rated]
    pick = rated + [rest[i] for i in rng.choice(len(rest), a.n_val - len(rated), replace=False)]
    ref = "runs/mace-qd2/check/images/"
    out["val"] = {"prompt": SUR, "pairs": [(p, ref + o) for p in pick for o in S["opponents"][p]],
                  "label": lab, "old": {p: S["score"][p] for p in pick}, "group": {}}
    # cmp
    C = json.loads(Path("runs/qd5-compare-s0/compare.json").read_text())
    pool, opp = C["pool"], C["opponents"]
    tasks = json.loads(Path("runs/qd5-compare-s0/tasks.json").read_text())
    ans = label_answers(Path("runs/qd5-compare-s0"))
    lab = {pool[t["pool_index"]]["image"]: int(ans[t["id"]]["answer"] == "new") for t in tasks
           if t["id"] in ans and ans[t["id"]]["answer"] in ("new", "not_new")}
    rated = [t["pool_index"] for t in tasks]
    rest = [i for i in range(len(pool)) if i not in rated]
    pick = rated + [rest[i] for i in rng.choice(len(rest), a.n_cmp - len(rated), replace=False)]
    old = C["scores"]["jev/surprise"]
    out["cmp"] = {"prompt": SUR, "pairs": [(pool[i]["image"], pool[j]["image"]) for i in pick for j in opp[i]],
                  "label": lab, "old": {pool[i]["image"]: old[str(i)] for i in pick if str(i) in old},
                  "group": {pool[i]["image"]: pool[i]["run"] for i in pick}}
    # comp
    ck = json.loads(Path("runs/mace-qd2/check/check.json").read_text())
    d = "runs/mace-qd2/check/images/"
    pairs = [(d + x + ".png", d + y + ".png") for x, y in ck["pairs"]]
    out["comp"] = {"prompt": COMP, "pairs": pairs, "label": {}, "group": {},
                   "old": {f"{x}|{y}": p for (x, y), p in zip(pairs, ck["p_first_of_pair"]["jev"])}}
    # retest: first n_retest picked cmp worlds at full size, fresh sample
    keep = set(list(out["cmp"]["old"])[:a.n_retest])
    out["retest"] = {**out["cmp"], "pairs": [pr for pr in out["cmp"]["pairs"] if pr[0] in keep],
                     "old": {k: v for k, v in out["cmp"]["old"].items() if k in keep}}
    return out


def plan(a, S):
    jobs = []                                  # (set, size, sample)
    for px in a.sizes:
        jobs += [(s, px, 0) for s in ("val", "cmp", "comp")]
    if a.n_retest:
        jobs.append(("retest", 644, 1))
    rows, est = [], 0.0
    for s, px, smp in jobs:
        n = 2 * len(S[s]["pairs"])
        c = n * tok(px) * JEV_USD_PER_TOKEN
        est += c
        rows.append((s, px, smp, n, c))
    return rows, est


def run(a) -> None:
    S = sets(a)
    rows, est = plan(a, S)
    for s, px, smp, n, c in rows:
        print(f"  {s:6s} {px:4d} px  sample {smp}: {n:5d} calls  ~${c:.3f}", file=sys.stderr)
    print(f"  total ~${est:.2f} (cap --max-usd {a.max_usd})", file=sys.stderr)
    if a.dry_run:
        return
    if est > a.max_usd:
        sys.exit("estimate exceeds --max-usd; nothing called")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    jc = {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "est_cost_per_call_usd": tok(644) * JEV_USD_PER_TOKEN,
          "max_run_usd": a.max_usd}
    judge = Judge(a.judge, jc, out / "judge_cache.sqlite", yes=True)
    res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    imgs = {}

    def im(p, px):
        if (p, px) not in imgs:
            imgs[(p, px)] = shrink(load_img(p), px)
        return imgs[(p, px)]

    for s, px, smp, n, c in rows:
        if judge.spent + c > a.max_usd:
            print(f"stopping before {s}/{px}: would pass --max-usd", file=sys.stderr)
            break
        prs = S[s]["pairs"]
        qs = [q for x, y in prs for q in (Query(S[s]["prompt"], im(x, px), im(y, px), smp),
                                          Query(S[s]["prompt"], im(y, px), im(x, px), smp))]
        vs = judge.judge_many(qs, desc=f"{s} {px}px")
        res[f"{s}/{px}"] = [{"a": x, "b": y, "x": vs[2 * k].p_first, "y": vs[2 * k + 1].p_first,
                             "cost": vs[2 * k].cost + vs[2 * k + 1].cost, "cached": vs[2 * k].cached}
                            for k, (x, y) in enumerate(prs)]
        (out / "results.json").write_text(json.dumps(res))
    print(f"spent ${judge.spent:.4f}", file=sys.stderr)
    report(a, S)


def spearman(u, v):
    r = lambda z: np.argsort(np.argsort(z)).astype(float)
    return float(np.corrcoef(r(np.asarray(u)), r(np.asarray(v)))[0, 1])


def auc(s, lab):
    s, lab = np.asarray(s, float), np.asarray(lab)
    p, n = s[lab == 1], s[lab == 0]
    if not len(p) or not len(n):
        return float("nan")
    return float(np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :]))


def report(a, S=None) -> None:
    S = S or sets(a)
    out = Path(a.out)
    res = json.loads((out / "results.json").read_text())
    L = ["# Gate 0f: Jev on smaller pictures\n",
         "Old = the 644-px answers from the original runs. Score = mean P over opponents, both orders. "
         "Accept a size if: rank correlation with old >= 0.9 (and close to the retest ceiling), AUC vs Felix "
         "not lower, position bias unchanged.\n",
         "| set | px | calls | tokens/call | P(first) | rank corr vs old | AUC vs Felix (old -> new) | "
         f"same side of {THR} | archive vs map AUC (old -> new) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, rows in res.items():
        s, px = key.split("/")
        d = S[s]
        ok = [r for r in rows if r["x"] is not None and r["y"] is not None]
        if not ok:
            continue
        allp = [r["x"] for r in ok] + [r["y"] for r in ok]
        newc = [r["cost"] / 2 for r in ok if not r["cached"] and r["cost"]]
        tk = f"{np.mean(newc) / JEV_USD_PER_TOKEN:.0f}" if newc else "-"
        if s == "comp":
            new = {f"{r['a']}|{r['b']}": 0.5 * (r["x"] + 1 - r["y"]) for r in ok}
            ks = [k for k in new if k in d["old"]]
            o, n = [d["old"][k] for k in ks], [new[k] for k in ks]
            agree = np.mean([(x >= 0.5) == (y >= 0.5) for x, y in zip(o, n)])
            L.append(f"| comp | {px} | {2 * len(ok)} | {tk} | {np.mean(allp):.2f} | {spearman(o, n):.2f} | - | "
                     f"winner agrees {agree:.2f}; mean P(final) {np.mean(o):.2f} -> {np.mean(n):.2f} | - |")
            continue
        sc = {}
        for r in ok:
            sc.setdefault(r["a"], []).append(0.5 * (r["x"] + 1 - r["y"]))
        new = {k: float(np.mean(v)) for k, v in sc.items()}
        ks = [k for k in new if k in d["old"]]
        o, n = [d["old"][k] for k in ks], [new[k] for k in ks]
        lk = [k for k in ks if k in d["label"]]
        au = (f"{auc([d['old'][k] for k in lk], [d['label'][k] for k in lk]):.2f} -> "
              f"{auc([new[k] for k in lk], [d['label'][k] for k in lk]):.2f} (n={len(lk)})") if lk else "-"
        side = np.mean([(x >= THR) == (y >= THR) for x, y in zip(o, n)])
        am = "-"
        if d["group"]:
            g = [d["group"][k] for k in ks]
            arch = np.array([x.endswith("(archive)") for x in g])
            base = np.array([x == "mace-qd2" for x in g])
            m = arch | base
            am = (f"{auc(np.array(o)[m], arch[m].astype(int)):.2f} -> {auc(np.array(n)[m], arch[m].astype(int)):.2f}")
        L.append(f"| {s} | {px} | {2 * len(ok)} | {tk} | {np.mean(allp):.2f} | {spearman(o, n):.2f} | {au} | "
                 f"{side:.2f} | {am} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", default="runs/res-check")
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[484, 324])
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-cmp", type=int, default=120)
    ap.add_argument("--n-retest", type=int, default=40)
    ap.add_argument("--max-usd", type=float, default=0.6)
    ap.add_argument("--judge", default="jev")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
