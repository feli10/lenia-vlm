"""GIFs of MAP-Elites cells: each cell's first occupant and final elite under both rewards, side by side.

usage: uv run python scripts/qd_gifs.py <runs folder> pairs        # the representative cells below
       uv run python scripts/qd_gifs.py <runs folder> 4,7          # one cell's whole history, per reward
Camera follows the creature and zooms to it (sizes are not comparable across panels). Labels: role, the
run's own score, the class over this 50-unit replay; "~" = start pattern not saved (runs before the fix),
replayed from the cell's final creature instead, so treat those panels as approximate.
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from lenia.show import simulate, gif
from lenia.sim import Rule
from lenia.pool import _item_of
R = Path(sys.argv[1]); OUT = R / "qd-gifs"
SHAPE = (256, 256); TIME, START, DT = 50.0, 10.0, 0.5
pool = {r["id"]: r for r in csv.DictReader(open(R / "pool2/creatures.csv"))}
cache = {}
runs = {k: (json.load(open(R / f"qd-{k}/state.json")), np.load(R / f"qd-{k}/patterns.npz")) for k in ("student", "features")}

def start_of(run, cell, e):
    st, pz = runs[run]
    if e["pattern"] in pz.files:
        return np.asarray(pz[e["pattern"]], np.float32), ""
    if e["kind"] == "seed":
        return _item_of(pool[e["pool_id"]], R / "found-survey-v2", R / "expA2-explore", cache)[3], ""
    last = st["events"][st["cells"][cell][-1]]          # pattern lost (replaced elite): start from the cell's final creature
    return np.asarray(pz[last["pattern"]], np.float32), "~"

def sim(label, rule, cells):
    (lab, rl, frames, stt), = simulate([(label, rule, ("cells", cells))], SHAPE, time=TIME, every=max(1, int(round(DT * rule.T))))
    ts = np.array([t for t, _ in frames])
    grid = np.arange(0, TIME + 1e-9, DT)
    frames = [(g, frames[int(np.argmin(np.abs(ts - g)))][1]) for g in grid]
    return (lab, rl, frames, stt)

def clip(run, cell, e, tag):
    cells, mark = start_of(run, cell, e)
    return sim(f"{tag}{mark} {e['main']:+.2f}", Rule.from_dict(json.loads(e["rule"])), cells)

which = sys.argv[2]
if which == "pairs":
    for cell, name in [("7,3", "large-slow-spinner"), ("8,8", "large-fast-cage"), ("6,5", "large-mid-vortex"), ("4,7", "mid-fast-crescent"),
                       ("5,4", "mid-slow-jelly"), ("2,3", "small-slow"), ("1,8", "small-fast"), ("9,5", "huge-chain")]:
        sims = []
        for run in ("student", "features"):
            st, _ = runs[run]
            v = st["cells"].get(cell)
            if not v: continue
            f, e = st["events"][v[0]], st["events"][v[-1]]
            sims.append(clip(run, cell, f, f"{run}: first ({f['kind']})"))
            if e is not f: sims.append(clip(run, cell, e, f"{run}: final it{e['it']}"))
        p = gif(sims, OUT / f"cell_{cell.replace(',', '-')}_{name}.gif", px=200, cols=4, start=START, fps=12)
        print(p, [s[3]["class"] for s in sims], flush=True)
else:   # one cell's whole history under each reward
    cell = which
    for run in ("student", "features"):
        st, _ = runs[run]
        sims = [clip(run, cell, st["events"][k], f"{run} it{st['events'][k]['it']}") for k in st["cells"][cell]]
        p = gif(sims, OUT / f"history_{cell.replace(',', '-')}_{run}.gif", px=180, cols=min(len(sims), 5), start=START, fps=12)
        print(p, [s[3]["class"] for s in sims], flush=True)
