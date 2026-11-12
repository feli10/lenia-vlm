"""Live MaCE-Lenia viewer (11-12): runs one world continuously, like the Lenia web demo, in a local browser page.

  cd ~/Projects/lenia-vlm && PYTHONPATH=src:scripts python3 scripts/live.py            # then open http://localhost:8790

Worlds: the qd2 catalog (12) and its "more creatures" (8), grown exactly as the map grows them (same settings, same
fixed noise start) on the numpy backend, so a world here behaves like the long runs in lenia-runs/longrun. The page:
pick a world, pause, speed, reset, and click the world to drop a blob of mass (it is no longer mass-conserving then).
No judge calls, nothing written to disk.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

import mace_qd as QD
from lenia import backend, render, seeds
from lenia.config import REPO_ROOT
from lenia.sim import Rule

RUN = REPO_ROOT / "runs" / "mace-qd2"
HIDE = {"Labyrinth cell", "Four-lobed cluster", "Clover", "Lumpy amoeba", "Rimmed starfish"}   # Felix, 11-12


def worlds() -> list:
    out = [{"name": c["name"], "rule": c["rule"]} for c in json.loads((RUN / "catalog" / "catalog.json").read_text())]
    more = RUN / "catalog" / "more" / "more.json"
    if more.exists():
        out += [{"name": c["name"], "rule": c["rule"]} for c in json.loads(more.read_text())]
    return [w for w in out if w["name"] not in HIDE]


class Sim:
    def __init__(self, a):
        self.ea = QD.settings(json.loads((RUN / "state.json").read_text()))
        self.W, self.lock = worlds(), threading.Lock()
        self.speed, self.paused, self.px = a.speed, False, a.px
        self.load(next((i for i, w in enumerate(self.W) if w["name"].lower() == a.world.lower()), 0))

    def load(self, i: int) -> None:
        with self.lock:
            self.i = i % len(self.W)
            shape = (self.ea.size, self.ea.size)
            self.b = backend.make_batch([Rule.from_dict(self.W[self.i]["rule"])], shape)
            A0 = seeds.smooth_noise(self.ea.R, shape, 3, seed=self.ea.start_seed, size_r=3.0)
            self.X, self.t, self.scale, self.act = backend.to_device(A0[None]), 0, None, 0.0
            self.prev, self.prev_t = A0.astype(np.float32), 0

    def _blob(self, fx: float, fy: float, width: float) -> np.ndarray:   # a smooth bump at a fraction of the world
        n = self.ea.size
        yy, xx = np.mgrid[0:n, 0:n]
        dx = (xx - fx * n + n / 2) % n - n / 2                      # the world wraps around at its edges
        dy = (yy - fy * n + n / 2) % n - n / 2
        return np.exp(-(dx ** 2 + dy ** 2) / (2 * (self.ea.R * width) ** 2)).astype(np.float32)

    def drop(self, fx: float, fy: float) -> None:          # add a blob of mass
        with self.lock:
            A = np.asarray(backend.to_numpy(self.X)).copy()
            A[0] += 0.6 * self._blob(fx, fy, 0.8)[None] * (0.6 + 0.4 * np.random.rand(3, 1, 1)).astype(np.float32)
            self.X = backend.to_device(A)

    def erase(self, fx: float, fy: float) -> None:         # remove the mass under a soft round eraser
        with self.lock:
            A = np.asarray(backend.to_numpy(self.X)).copy()
            A[0] *= (1 - 0.97 * self._blob(fx, fy, 1.0))[None]
            self.X = backend.to_device(A)

    def loop(self) -> None:
        while True:
            if self.paused:
                time.sleep(0.05)
                continue
            with self.lock:
                for _ in range(self.speed):
                    self.X = self.b.step(self.X)
                    self.t += 1
                    if self.t - self.prev_t >= 100:                 # change per 100 steps, as in the long runs
                        A = np.asarray(backend.to_numpy(self.X))[0].astype(np.float32)
                        self.act = float(np.abs(A - self.prev).sum() / max(self.prev.sum(), 1e-9))
                        self.prev, self.prev_t = A, self.t
            time.sleep(0.001)

    def frame(self) -> tuple[bytes, dict]:
        with self.lock:
            A = np.asarray(backend.to_numpy(self.X))[0].astype(np.float32)
            t, name = self.t, self.W[self.i]["name"]
        s = max(float(np.percentile(A, 99.5)), 1e-6)
        self.scale = s if self.scale is None else 0.97 * self.scale + 0.03 * s     # smooth brightness, no flicker
        im = Image.fromarray(render.colorize(np.clip(A / self.scale, 0, 1))).resize((self.px, self.px), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return buf.getvalue(), {"t": t, "name": name, "i": self.i, "speed": self.speed, "paused": self.paused,
                                "activity": round(self.act, 3), "mass": round(float(A.sum()), 1)}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>MaCE-Lenia live</title><style>
body{background:#111;color:#ddd;font:14px -apple-system,Segoe UI,sans-serif;margin:0;display:flex;gap:20px;padding:20px}
#w{cursor:crosshair;border:1px solid #333;image-rendering:auto} select,button{font:inherit;padding:5px 9px;margin:3px 0;background:#222;color:#ddd;border:1px solid #444;border-radius:6px}
.p{width:260px} .stat{font-variant-numeric:tabular-nums;color:#9c9;margin-top:10px;line-height:1.7} .hint{color:#888;margin-top:14px}
</style></head><body><img id="w" width="PX" height="PX"><div class="p">
<div><select id="sel"></select></div>
<div><button id="pp">Pause</button> <button id="rs">Reset</button></div>
<div>Speed <button data-s="0.5">slower</button> <button data-s="2">faster</button></div>
<div>Click to <button id="mAdd">add mass</button> <button id="mErase">erase mass</button></div>
<div class="stat" id="st"></div>
<div class="hint">Click the world to add or erase mass (shift-click or right-click always erases; drag to keep erasing).<br>Activity = change per 100 steps (the long-run measure: closing ring ~0.5, worm frozen ~0.01).<br>Step 2,700 = where every search judged its worlds.</div></div>
<script>
const W=document.getElementById('w'),st=document.getElementById('st'),sel=document.getElementById('sel');
const post=(p,b)=>fetch(p,{method:'POST',body:JSON.stringify(b||{})});
fetch('/worlds').then(r=>r.json()).then(ns=>{ns.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n;sel.appendChild(o)})});
sel.onchange=()=>post('/cmd',{world:+sel.value});
document.getElementById('pp').onclick=()=>post('/cmd',{pause:'toggle'});
document.getElementById('rs').onclick=()=>post('/cmd',{world:+sel.value});
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>post('/cmd',{speed:+b.dataset.s}));
let mode='drop',down=false,last=0;
const mb={drop:document.getElementById('mAdd'),erase:document.getElementById('mErase')};
const setMode=m=>{mode=m;for(const k in mb)mb[k].style.borderColor=k===m?'#9c9':'#444'};setMode('drop');
mb.drop.onclick=()=>setMode('drop');mb.erase.onclick=()=>setMode('erase');
const pos=e=>{const r=W.getBoundingClientRect();return[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]};
const act=(e,m)=>post('/cmd',{[m]:pos(e)});
W.addEventListener('contextmenu',e=>e.preventDefault());
W.onmousedown=e=>{e.preventDefault();const m=(e.button===2||e.shiftKey)?'erase':mode;down=m;act(e,m)};
W.onmousemove=e=>{if(down==='erase'&&Date.now()-last>60){last=Date.now();act(e,'erase')}};
window.onmouseup=()=>{down=false};W.ondragstart=()=>false;
async function tick(){try{const r=await fetch('/frame?'+Date.now());const s=JSON.parse(r.headers.get('X-State'));
 const b=await r.blob();const u=URL.createObjectURL(b);W.onload=()=>URL.revokeObjectURL(u);W.src=u;
 sel.value=s.i;document.getElementById('pp').textContent=s.paused?'Play':'Pause';
 st.innerHTML=`${s.name}<br>step ${s.t.toLocaleString()} (${(s.t/2700).toFixed(1)}x normal)<br>activity ${s.activity.toFixed(2)}<br>${s.speed} steps per frame`;
}catch(e){} setTimeout(tick,50)} tick();
</script></body></html>"""


def serve(a) -> None:
    backend.configure(a.backend)
    S = Sim(a)
    threading.Thread(target=S.loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/frame"):
                jpg, state = S.frame()
                return self._send(200, jpg, "image/jpeg", {"X-State": json.dumps(state)})
            if self.path == "/worlds":
                return self._send(200, json.dumps([w["name"] for w in S.W]).encode(), "application/json")
            if self.path == "/":
                return self._send(200, PAGE.replace("PX", str(S.px)).encode(), "text/html; charset=utf-8")
            self._send(404, b"", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if "world" in body:
                S.load(int(body["world"]))
            if body.get("pause") == "toggle":
                S.paused = not S.paused
            if "speed" in body:
                S.speed = int(min(200, max(1, round(S.speed * float(body["speed"])))))
            if "drop" in body:
                S.drop(*map(float, body["drop"]))
            if "erase" in body:
                S.erase(*map(float, body["erase"]))
            self._send(200, b"{}", "application/json")

    print(f"MaCE-Lenia live: open http://localhost:{a.port}  ({len(S.W)} worlds; Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="Closing ring", help="a world's name from the catalog")
    ap.add_argument("--speed", type=int, default=10, help="simulation steps per displayed frame (changeable on the page)")
    ap.add_argument("--px", type=int, default=560, help="display size in pixels")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--backend", default="numpy")
    serve(ap.parse_args())


if __name__ == "__main__":
    main()
