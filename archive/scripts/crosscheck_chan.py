"""Run Chan's own Automaton (LeniaND.py, GUI and GPU stubbed out) and ours from the same start; compare.

Usage (from the repo root; needs scipy, which Chan's code imports, and his repo checked out):
    git clone https://github.com/Chakazul/Lenia ../Lenia
    uv run --with scipy python scripts/crosscheck_chan.py ../Lenia/Python
"""
import sys, types, json, numpy as np
CHAN_PY = sys.argv[1] if len(sys.argv) > 1 else "../Lenia/Python"
for name in ["reikna", "reikna.fft", "reikna.cluda", "PIL.ImageTk", "tkinter", "tkinter.messagebox", "tkinter.filedialog"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["reikna"].fft = sys.modules["reikna.fft"]; sys.modules["reikna"].cluda = sys.modules["reikna.cluda"]
def _no_gpu(*a, **k): raise RuntimeError("no gpu (stub)")
sys.modules["reikna.cluda"].any_api = _no_gpu
sys.argv = ["LeniaND", "-d", "2", "-s", "7"]
sys.path.insert(0, CHAN_PY)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import LeniaND as C
sys.path.insert(0, "src")
from lenia.sim import Rule, Batch, place
from lenia.rle import rle2arr
from lenia.alive import Thresholds
from lenia.rollout import run_batch, Horizon

shape = (128, 128)
an = [x for x in json.load(open("data/animals.json")) if "cells" in x and isinstance(x.get("params"), dict)]
rng = np.random.default_rng(3)
fit = [x for x in an if x["params"]["R"] <= 20 and float(x["params"].get("T", 10)) == 10 and max(rle2arr(x["cells"]).shape) <= 96]
pick = [fit[i] for i in rng.choice(len(fit), 30, replace=False)]
worst, rows = 0, []
for x in pick:
    p = dict(x["params"]); p["b"] = C.Board.st2fracs(p["b"])
    A0 = place(rle2arr(x["cells"]), shape)[0].astype(np.float64)
    board = C.Board.from_values(A0.copy(), p, [x["code"], x["name"], ""])
    with contextlib.redirect_stdout(io.StringIO()):
        auto = C.Automaton(board)
    rule = Rule.from_chan(x["params"])
    b = Batch([rule], shape)
    A = A0[None, None].astype(np.float32)
    diffs = []
    for k in range(300):
        auto.calc_once()
        A = b.step(A)
        if k in (0, 9, 99, 299):
            diffs.append(float(np.abs(board.cells - A[0, 0]).max()))
    rows.append((x["code"], diffs, float(board.cells.sum()), float(A.sum())))
    worst = max(worst, diffs[-1])
for c, d, m1, m2 in rows:
    print(f"{c:10s} max|diff| after 1/10/100/300 steps: " + " ".join(f"{v:.1e}" for v in d) + f"   mass chan {m1:.1f} ours {m2:.1f}")
print("worst after 300 steps:", worst)
