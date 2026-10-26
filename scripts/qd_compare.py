"""Compare two MAP-Elites runs' final archives: does surprise-weighted parent selection (mace-qd3-surprise) end
somewhere different from uniform parent selection (mace-qd2, same settings and seeds)?

Both final archives are pooled. Every elite meets --opponents random elites of the pool (both archives, never
itself), both orders, under
  - the surprise prompt (surprise_check.py; validated against Felix's ratings) with Luna (held out: Luna never
    drove either run) and with Jev (the driver of qd3's parent selection, so partly circular for qd3), and
  - the complexity prompt (mace_evo; the entry duel of both runs) with Luna, as a quality check.
A score is the mean win probability, so both archives are measured against the same reference. Reported: mean
score per archive, AUC (probability a random qd3 elite outscores a random qd2 elite; 0.5 = no difference) with a
permutation p-value, and the share of each archive among the pool's top 20. Elites of one lineage are not
independent, so the p-values are optimistic.

Felix's blind check: 12 random elites from each archive (not chosen by any judge), shuffled; same question as the
surprise check ("something unusual here, compared with the typical MaCE world?"), rated with surprise_check.py.

  uv run python scripts/qd_compare.py score                            # ~12,500 judge calls, about $1-2 (Mac)
  uv run python scripts/surprise_check.py rate --out runs/qd-compare   # your 24 ratings
  uv run python scripts/qd_compare.py report
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import mace_qd as QD
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, Query, load_env
from mace_evo import PROMPT_ID as COMPLEXITY
from surprise_check import SURPRISE, _answers, auc, fisher_greater

N_RATE, TOP = 12, 20


def elites(run: Path) -> list:
    st = json.loads((run / "state.json").read_text())
    out = []
    for k, m in enumerate(QD.islands_of(st)):
        for ck, e in sorted(m.items()):
            out.append({"run": run.name, "island": k, "cell": ck, "image": str(run / "img" / e["image"]),
                        "root": e.get("root"), "motion": e.get("motion"), "bodies": e.get("bodies")})
    return out


def score(a) -> None:
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    A, B = elites(Path(a.a)), elites(Path(a.b))
    pool = A + B
    missing = [e["image"] for e in pool if not Path(e["image"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing, e.g. {missing[0]} (rsync the runs' img folders)")
    print(f"{Path(a.a).name}: {len(A)} elites; {Path(a.b).name}: {len(B)} elites", file=sys.stderr)
    rng = np.random.default_rng(a.seed)
    opp = [[int(j) for j in rng.choice([x for x in range(len(pool)) if x != i], a.opponents, replace=False)]
           for i in range(len(pool))]
    pics = {}

    def pic(i):
        if i not in pics:
            pics[i] = np.asarray(Image.open(pool[i]["image"]).convert("RGB"))
        return pics[i]

    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    scores = {}
    for jn, pname, pid in (("luna", "surprise", SURPRISE), ("jev", "surprise", SURPRISE), ("luna", "complexity", COMPLEXITY)):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, out / "judge_cache.sqlite", yes=True)
        qs, meta = [], []
        for i in range(len(pool)):
            for j in opp[i]:
                qs += [Query(pid, pic(i), pic(j)), Query(pid, pic(j), pic(i))]
                meta.append(i)
        vs = judge.judge_many(qs, desc=f"{jn} {pname}")
        acc = {}
        for n, i in enumerate(meta):
            x, y = vs[2 * n], vs[2 * n + 1]
            if x.ok and y.ok:
                acc.setdefault(i, []).append(0.5 * (x.p_first + 1 - y.p_first))
        scores[f"{jn}/{pname}"] = {str(i): float(np.mean(v)) for i, v in acc.items()}
        print(f"{jn} {pname}: spent ${judge.spent:.4f}", file=sys.stderr)
    # Felix's blind sample: random elites from each archive
    rr = np.random.default_rng(a.seed + 1)
    pick = [(i, "A") for i in rr.choice(len(A), min(N_RATE, len(A)), replace=False)] + \
           [(len(A) + i, "B") for i in rr.choice(len(B), min(N_RATE, len(B)), replace=False)]
    pick = [pick[j] for j in rr.permutation(len(pick))]
    tasks = []
    for k, (i, side) in enumerate(pick):
        shutil.copy(pool[int(i)]["image"], out / "images" / f"w{k:02d}.png")
        tasks.append({"id": k, "image": f"w{k:02d}.png", "name": pool[int(i)]["image"], "selection": pool[int(i)]["run"],
                      "pool_index": int(i)})
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
    (out / "compare.json").write_text(json.dumps({"a": str(a.a), "b": str(a.b), "pool": pool, "opponents": opp,
                                                  "scores": scores}, indent=1))
    report(a)


def report(a) -> None:
    out = Path(a.out)
    C = json.loads((out / "compare.json").read_text())
    pool, scores = C["pool"], C["scores"]
    na, nb = Path(C["a"]).name, Path(C["b"]).name
    L = [f"# {nb} vs {na}: final archives\n",
         f"{sum(e['run'] == na for e in pool)} + {sum(e['run'] == nb for e in pool)} elites, each against "
         f"{len(C['opponents'][0])} random elites of the pooled archives, both orders. AUC = P(a {nb} elite outscores "
         f"a {na} elite); 0.5 = no difference. Lineage-correlated elites make p optimistic.\n",
         f"| judge / prompt | mean {na} | mean {nb} | AUC | p | {nb} share of top {TOP} |", "|---|---|---|---|---|---|"]
    for k, sc in scores.items():
        ia = [i for i, e in enumerate(pool) if e["run"] == na and str(i) in sc]
        ib = [i for i, e in enumerate(pool) if e["run"] == nb and str(i) in sc]
        va, vb = [sc[str(i)] for i in ia], [sc[str(i)] for i in ib]
        A_, p = auc(vb + va, [1] * len(vb) + [0] * len(va), n_perm=5000)
        top = sorted(sc, key=lambda i: -sc[i])[:TOP]
        share = np.mean([pool[int(i)]["run"] == nb for i in top])
        L.append(f"| {k} | {np.mean(va):.3f} | {np.mean(vb):.3f} | {A_:.2f} | {p:.3f} | {share:.2f} |")
    for nm in (na, nb):
        es = [e for e in pool if e["run"] == nm]
        L.append(f"\n{nm}: {len(es)} elites, {len({e['root'] for e in es})} seed lineages, mean motion "
                 f"{np.mean([e['motion'] for e in es]):.2f}")
    tasks = json.loads((out / "tasks.json").read_text())
    ans = _answers(out)
    if ans:
        L += ["", "**Felix's blind check** (12 random elites per archive, shuffled): \"something unusual?\"", "",
              "| archive | unusual | typical | unsure |", "|---|---|---|---|"]
        cnt = {}
        for nm in (na, nb):
            r = [ans[t["id"]]["answer"] for t in tasks if t["selection"] == nm and t["id"] in ans]
            cnt[nm] = (r.count("new"), r.count("not_new"))
            L.append(f"| {nm} | {r.count('new')} | {r.count('not_new')} | {r.count('unsure')} |")
        if all(sum(v) for v in cnt.values()):
            L.append(f"\nOne-sided Fisher ({nb} more often unusual): p = {fisher_greater(*cnt[nb], *cnt[na]):.3f}")
        dec = [t for t in tasks if t["id"] in ans and ans[t["id"]]["answer"] in ("new", "not_new")]
        if len({ans[t["id"]]["answer"] for t in dec}) == 2 and "luna/surprise" in scores:
            A_, p = auc([scores["luna/surprise"][str(t["pool_index"])] for t in dec],
                        [int(ans[t["id"]]["answer"] == "new") for t in dec], n_perm=5000)
            L.append(f"Luna surprise score vs your answers on these {len(dec)} worlds: AUC {A_:.2f} (p = {p:.3f}).")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "report"])
    ap.add_argument("--a", default="runs/mace-qd2", help="control run")
    ap.add_argument("--b", default="runs/mace-qd3-surprise", help="treatment run")
    ap.add_argument("--out", default="runs/qd-compare")
    ap.add_argument("--opponents", type=int, default=6)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
