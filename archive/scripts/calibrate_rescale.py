"""Calibrate the coherence check on Chan's catalogue: do real species survive at double resolution, and
how far does their descriptor move?

Each species is grown from its own pattern for 100 units at normal resolution; those alive are rescaled
(R -> 2R, final pattern upsampled 2x, world 256x256) and run for 50 units. Reported: how many are alive
there, and quantiles of the descriptor distance between each species and its 2x self (in the catalogue's
own descriptor scale). explore.RESCALE_MAX_DIST is set from the p95. Writes runs/calibrate_rescale.json.
Usage: uv run python scripts/calibrate_rescale.py [out.json]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lenia import catalogue
from lenia.alive import ALIVE, Thresholds
from lenia.config import REPO_ROOT, load, resolve
from lenia.describe import Space
from lenia.explore import evaluate, rescale_check
from lenia.rollout import Horizon

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "runs/calibrate_rescale.json"
    sp = catalogue.load(resolve(load(), "catalogue"))
    th, shape = Thresholds(), (128, 128)
    ex = ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    r1 = evaluate([s["rule"] for s in sp], [s["cells"][None] for s in sp], shape,
                  Horizon(time=100, frames=6, film_span=5, thumb_every=0), th, ex)
    alive = [(s, pat, d) for s, (st, pat, d) in zip(sp, r1) if pat is not None]
    space = Space(np.array([d for _, _, d in alive]))
    r2 = rescale_check([s["rule"] for s, _, _ in alive], [p for _, p, _ in alive], shape, th, ex, time=50)
    ex.shutdown()
    cls = [c for c, _ in r2]
    dist = [float(space.dist(d1, np.asarray(d2)[None])[0]) if c in ALIVE else None for (_, _, d1), (c, d2) in zip(alive, r2)]
    ok = np.array([x for x in dist if x is not None])
    res = {"species": len(sp), "alive_1x_100": len(alive), "alive_2x_50": int(sum(c in ALIVE for c in cls)),
           "classes_2x": dict(Counter(cls)),
           "dist_quantiles": {f"p{q}": round(float(np.percentile(ok, q)), 3) for q in (50, 75, 90, 95, 99)},
           "seconds": round(time.time() - t0),
           "per_species": [{"code": s["code"], "class_1x": st, "class_2x": c, "dist": d}
                           for (s, _, _), st, c, d in zip(alive, [x[0]["class"] for x in r1 if x[1] is not None], cls, dist)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_species"}))
