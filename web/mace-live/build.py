"""Build the static MaCE-Lenia Live page (11-12): index.html = page.html + gpu.js + sim.js + the worlds live.py shows
+ the shared start.

  cd ~/Projects/lenia-vlm && PYTHONPATH=src:scripts python3 web/mace-live/build.py

The page runs the simulation in the browser, so it can be hosted anywhere static (GitHub Pages). Two engines, both
checked against numpy's lenia.sim.Batch.step on the same start (mass conserved in both):
- gpu.js, WebGL2 (direct convolution + MaCE in three shader passes, float32): max difference ~5e-6 after 1 step,
  ~2e-5 after 50. Opt-in with ?gpu (a link on the page), and only if a 10-step benchmark beats 120 steps/s.
- sim.js, JavaScript FFT in a Web Worker (float64): ~3e-6 after 1 step, ~1e-4 after 50. The default (Felix, 11-12:
  the float32 GPU runs drift away from it over long runs, as any two precisions do on these chaotic worlds).
URL options: ?gpu opts into the GPU engine; ?speed=n starts at n steps per frame.
Worlds and the shared smoothed-noise start come from runs/mace-qd2 exactly as live.py uses them (live.HIDE applies),
plus the ASAL baseline's 256 worlds when runs/asal-illum-s1 has them (live.asal_worlds; same size, R and start).
"""
import base64
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import live                       # noqa: E402
import mace_qd as QD              # noqa: E402
from lenia import seeds           # noqa: E402

ea = QD.settings(json.loads((live.RUN / "state.json").read_text()))
data = {"size": ea.size, "worlds": live.worlds()}
a0 = seeds.smooth_noise(ea.R, (ea.size, ea.size), 3, seed=ea.start_seed, size_r=3.0).astype("float32")
sim, gpu = (HERE / "sim.js").read_text(), (HERE / "gpu.js").read_text()
assert "</script" not in sim and "</script" not in gpu
page = (HERE / "page.html").read_text()
page = page.replace("/*SIM*/", sim).replace("/*GPU*/", gpu).replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
page = page.replace("/*A0*/", base64.b64encode(a0.tobytes()).decode())
(HERE / "index.html").write_text(page)
n_asal = sum(w["group"].startswith("ASAL") for w in data["worlds"])
print(f"index.html: {len(data['worlds'])} worlds ({n_asal} from ASAL), {len(page) // 1024} KB")
