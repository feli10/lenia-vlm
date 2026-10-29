"""A local page for rating pairs of pictures yourself (left / right / skip, with undo). Every click is appended
to <folder>/human.csv straight away; pairs come from <folder>/<source> (a JSON list of [a, b] image ids, images in
<folder>/images/<id>.png), a fixed random subset with random sides kept in human_pairs.json.
Moved here verbatim on 2026-10-28 from src/lenia/vlmtest.py (now in archive/).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HUMAN_QUESTION = "Which creature do you find more visually appealing?"


RATE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rate creatures</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f2;color:#222}
 header{padding:14px 20px;display:flex;justify-content:space-between;align-items:baseline}
 h1{font-size:18px;margin:0} #prog{color:#666}
 .q{padding:0 20px 8px;color:#444}
 .pair{display:flex;gap:16px;justify-content:center;padding:8px 20px}
 .pair figure{margin:0;background:#fff;border:3px solid transparent;border-radius:8px;padding:6px;cursor:pointer;flex:1;max-width:46vw}
 .pair figure:hover{border-color:#999}
 .pair img{display:block;margin:auto;max-width:100%;max-height:calc(100vh - 210px);width:auto;height:auto}
 .pair figcaption{text-align:center;color:#666;font-size:13px;padding-top:4px}
 .bar{display:flex;gap:10px;justify-content:center;padding:12px}
 button{font-size:15px;padding:8px 16px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
 #done{display:none;padding:40px;text-align:center;font-size:18px}
</style></head><body>
<header><h1>QUESTION</h1><span id="prog"></span></header>
<div class="q">DESCRIPTION Go with your first impression.</div>
<div id="ui"><div class="pair">
 <figure id="L"><img id="li"><figcaption>left &larr;</figcaption></figure>
 <figure id="R"><img id="ri"><figcaption>right &rarr;</figcaption></figure></div>
 <div class="bar"><button id="bl">&larr; Left</button><button id="bs">Can't decide (space)</button>
 <button id="br">Right &rarr;</button><button id="bu">Undo (backspace)</button></div></div>
<div id="done">All done. Your ratings are in runs/vlmtest/human.csv. You can close this tab and stop the server (Ctrl-C).</div>
<script>
let todo=[], total=0, busy=false;
async function load(){const r=await (await fetch('/pairs')).json(); todo=r.todo; total=r.total; show();}
function show(){
 document.getElementById('prog').textContent=(total-todo.length)+' / '+total+' rated';
 if(!todo.length){document.getElementById('ui').style.display='none';document.getElementById('done').style.display='block';return;}
 document.getElementById('ui').style.display='block';document.getElementById('done').style.display='none';
 const p=todo[0]; document.getElementById('li').src='/img/'+p.left+'.png'; document.getElementById('ri').src='/img/'+p.right+'.png';}
async function vote(choice){ if(busy||!todo.length) return; busy=true; const p=todo[0];
 await fetch('/vote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...p,choice})});
 todo.shift(); busy=false; show();}
async function undo(){ if(busy) return; busy=true; await fetch('/undo',{method:'POST'}); busy=false; load();}
document.getElementById('L').onclick=()=>vote('left'); document.getElementById('R').onclick=()=>vote('right');
document.getElementById('bl').onclick=()=>vote('left'); document.getElementById('br').onclick=()=>vote('right');
document.getElementById('bs').onclick=()=>vote('skip'); document.getElementById('bu').onclick=undo;
document.addEventListener('keydown',e=>{ if(e.key==='ArrowLeft')vote('left'); else if(e.key==='ArrowRight')vote('right');
 else if(e.key===' '){e.preventDefault();vote('skip');} else if(e.key==='Backspace'){e.preventDefault();undo();}});
load();
</script></body></html>"""


def _human_rows(out: Path) -> list:
    p = Path(out) / "human.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def human_pairs(out: Path, n: int = 60, seed: int = 1, source: str = "pairs.json") -> list:
    """The pairs you rate: a fixed random subset of `source` with random left/right sides (human_pairs.json)."""
    out = Path(out)
    p = out / "human_pairs.json"
    if p.exists():
        return json.loads(p.read_text())
    pairs = json.loads((out / source).read_text())
    rng = np.random.default_rng(seed)
    pick = [pairs[i] for i in sorted(rng.choice(len(pairs), min(n, len(pairs)), replace=False))]
    order = rng.permutation(len(pick))
    hp = []
    for k in order:
        a, b = pick[k]
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        hp.append({"a": a, "b": b, "left": left, "right": right})
    p.write_text(json.dumps(hp, indent=1))
    return hp


HUMAN_DESCRIPTION = "Each image shows one creature at four moments a few time units apart, in reading order."


def make_server(out: Path, n: int = 60, port: int = 8765, source: str = "pairs.json", question: str = HUMAN_QUESTION,
                description: str = HUMAN_DESCRIPTION):
    """The rating server (localhost only); every click is appended to human.csv straight away."""
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    out = Path(out)
    hp = human_pairs(out, n, source=source)
    path = out / "human.csv"
    lock = threading.Lock()
    fields = ["time", "a", "b", "left", "right", "choice", "winner"]

    def rated() -> set:
        return {(r["a"], r["b"]) for r in _human_rows(out)}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, RATE_PAGE.replace("QUESTION", question).replace("DESCRIPTION", description)
                                  .replace("runs/vlmtest/human.csv", str(path)).encode(),
                                  "text/html; charset=utf-8")
            if self.path == "/pairs":
                done = rated()
                return self._send(200, json.dumps({"todo": [p for p in hp if (p["a"], p["b"]) not in done], "total": len(hp)}).encode(),
                                  "application/json")
            if self.path.startswith("/img/") and self.path.endswith(".png"):
                name = Path(self.path).name
                f = out / "images" / name
                if f.parent == out / "images" and f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/vote":
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if d.get("choice") not in ("left", "right", "skip"):
                    return self._send(400, b"bad choice", "text/plain")
                winner = d["left"] if d["choice"] == "left" else d["right"] if d["choice"] == "right" else ""
                with lock:
                    new = not path.exists()
                    with open(path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow({"time": round(_time.time(), 1), "a": d["a"], "b": d["b"], "left": d["left"],
                                    "right": d["right"], "choice": d["choice"], "winner": winner})
                return self._send(200, b"ok", "text/plain")
            if self.path == "/undo":
                with lock:
                    rows = _human_rows(out)
                    if rows:
                        with open(path, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fields)
                            w.writeheader()
                            w.writerows(rows[:-1])
                return self._send(200, b"ok", "text/plain")
            self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.pairs_left = len(hp) - len(rated() & {(p["a"], p["b"]) for p in hp})
    srv.pairs_total = len(hp)
    return srv


def rate(out: Path, n: int = 60, port: int = 8765, open_browser: bool = True, source: str = "pairs.json",
         question: str = HUMAN_QUESTION, description: str = HUMAN_DESCRIPTION, log=print) -> None:
    import webbrowser
    srv = make_server(out, n, port, source, question, description)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    log(f"rating page: {url}  ({srv.pairs_left} of {srv.pairs_total} pairs left; answers go to "
        f"{Path(out) / 'human.csv'}; Ctrl-C to stop, progress is kept)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
