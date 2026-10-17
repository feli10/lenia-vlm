"""Calibrate the growth-trend thresholds (alive.max_growth, alive.max_rg_growth).

Positives (should pass): Chan's catalogue species run from their own pattern for 100 time units.
Negatives (should fail): creatures from the first (30-unit) explore runs that look alive at 100 time
units but are no longer alive at 200 ("late failures"); compared with those still alive at 200
("durable"). Growth checks are switched off while measuring, so every rollout gets its trend values.

    uv run python scripts/calibrate_growth.py runs/explore-classic-v1 runs/explore-multikernel-v1
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from lenia import catalogue  # noqa: E402
from lenia.alive import ALIVE, Thresholds  # noqa: E402
from lenia.config import REPO_ROOT, load, resolve  # noqa: E402
from lenia.explore import load_archive  # noqa: E402
from lenia.rollout import Horizon, run_batch  # noqa: E402
from lenia.sim import place  # noqa: E402

OFF = dict(max_growth=1e9, max_rg_growth=1e9)
SHAPE = (128, 128)


def run(job):
    rules, inits, time = job
    th = Thresholds(**OFF)
    hz = Horizon(time=time, frames=2, film_span=1, stop_every=10, thumb_every=0)
    out = []
    for k in range(0, len(rules), 16):
        groups = {}
        for i in range(k, min(k + 16, len(rules))):
            groups.setdefault((rules[i].structure(), rules[i].T), []).append(i)
        for ids in groups.values():
            res = run_batch([rules[i] for i in ids], np.stack([inits[i] for i in ids]), SHAPE, hz, th, [0] * len(ids))
            out += [(i, r.stats) for i, r in zip(ids, res)]
    return sorted(out, key=lambda x: x[0])


def main():
    cfg = load()
    runs = sys.argv[1:]
    rng = np.random.default_rng(0)
    # positives
    sp = catalogue.load(resolve(cfg, "catalogue"))
    pos_rules = [s["rule"] for s in sp]
    pos_inits = [place(s["cells"], SHAPE) for s in sp]
    # negatives / durables: archive children of earlier runs
    neg_rules, neg_inits, src = [], [], []
    for r in runs:
        kids = [a for a in load_archive(Path(r)) if a["parent"] >= 0]
        for i in rng.choice(len(kids), min(120, len(kids)), replace=False):
            a = kids[i]
            neg_rules.append(a["rule"])
            neg_inits.append(place(np.asarray(a["init"], np.float32), SHAPE, a["rule"].channels))
            src.append(Path(r).name)
    half = len(neg_rules) // 2
    jobs = [(pos_rules, pos_inits, 100.0),
            (neg_rules[:half], neg_inits[:half], 100.0), (neg_rules[half:], neg_inits[half:], 100.0),
            (neg_rules[:half], neg_inits[:half], 200.0), (neg_rules[half:], neg_inits[half:], 200.0)]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run, jobs))
    pos = [s for _, s in res[0]]
    n100 = [s for _, s in res[1]] + [s for _, s in res[2]]
    n200 = [s for _, s in res[3]] + [s for _, s in res[4]]
    alive = lambda s: s["class"] in ALIVE
    real = [s for s in pos if alive(s)]
    late = [a for a, b in zip(n100, n200) if alive(a) and not alive(b)]
    durable = [a for a, b in zip(n100, n200) if alive(a) and alive(b)]
    q = lambda xs, k: [round(float(np.percentile([x[k] for x in xs], p)), 3) for p in (50, 90, 95, 99)] if xs else None
    report = {"catalogue_alive_at_100": len(real), "catalogue": len(pos),
              "late_failures": len(late), "durable": len(durable), "of": len(n100),
              "growth_w p50/90/95/99": {"catalogue": q(real, "growth_w"), "durable": q(durable, "growth_w"), "late": q(late, "growth_w")},
              "rg_growth_w p50/90/95/99": {"catalogue": q(real, "rg_growth_w"), "durable": q(durable, "rg_growth_w"), "late": q(late, "rg_growth_w")}}
    table = []
    for g in (0.1, 0.15, 0.2, 0.3, 0.5):
        f = lambda xs: round(float(np.mean([x["growth_w"] > g or x["rg_growth_w"] > g for x in xs])), 3) if xs else None
        table.append({"threshold": g, "catalogue_rejected": f(real), "durable_rejected": f(durable), "late_caught": f(late)})
    report["thresholds"] = table
    print(json.dumps(report, indent=1))
    (REPO_ROOT / "runs" / "lenia").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "runs" / "lenia" / "calibrate_growth.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
