"""A local page for rating pairs of pictures yourself (left / right / skip, with undo). Every click is appended
to <folder>/human.csv straight away; pairs come from <folder>/<source> (a JSON list of [a, b] image ids, images in
<folder>/images/<id>.png), a fixed random subset with random sides kept in human_pairs.json.
Moved here verbatim on 2026-10-28 from src/lenia/vlmtest.py (now in archive/).

Also a one-world-at-a-time label page (unusual / typical / unsure, `rate_labels`), moved from
scripts/surprise_check.py (now in archive/); answers go to <folder>/ratings.csv, read with `label_answers`.
"""
from __future__ import annotations

import csv
import json
import threading
import time
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


# ---------- one world at a time: unusual / typical / unsure ----------

LABEL_PAGE = """<!doctype html><meta charset=utf-8><title>something new?</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:16px}img{width:620px;border:1px solid #333}
button{padding:8px 22px;font:15px system-ui;margin:8px 8px 0 0}#n{color:#999}input{width:600px;padding:6px;
font:14px system-ui;background:#222;color:#eee;border:1px solid #444}</style>
<div id=n></div><p>Does this world show <b>something unusual</b>: a feature or behaviour that stands out from the typical
MaCE worlds we get (not just bigger, smaller, more of the same, or another colour)? Whether you have seen this
particular world before does not matter. (four snapshots, reading order)</p><img id=im><br>
<input id=note placeholder="optional: what stands out?"><br>
<button data-a=new>unusual</button><button data-a=not_new>typical</button><button data-a=unsure>unsure</button>
<script>
let todo=[];async function load(){todo=await (await fetch('/todo')).json();show()}
function show(){if(!todo.length){document.body.innerHTML='<h2>Done, thank you. Close this tab and press Ctrl-C.</h2>';return}
document.getElementById('n').textContent=todo.length+' left';document.getElementById('im').src='/img/'+todo[0].image;
document.getElementById('note').value=''}
document.querySelectorAll('button').forEach(b=>b.onclick=async()=>{const t=todo[0];
await fetch('/answer',{method:'POST',body:JSON.stringify({id:t.id,answer:b.dataset.a,note:document.getElementById('note').value})});
todo.shift();show()});
load()</script>"""


def label_answers(out: Path) -> dict:
    p = out / "ratings.csv"
    return {int(r["id"]): r for r in csv.DictReader(open(p))} if p.exists() else {}


def rate_labels(a) -> None:
    """One world at a time: unusual / typical / unsure plus a note -> <out>/ratings.csv. a.out holds tasks.json
    ([{id, image}]) and images/; a.port is the local port."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    out = Path(a.out)
    tasks = [{"id": t["id"], "image": t["image"]} for t in json.loads((out / "tasks.json").read_text())]
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, LABEL_PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/todo":
                done = label_answers(out)
                return self._send(200, json.dumps([t for t in tasks if t["id"] not in done]).encode(), "application/json")
            if self.path.startswith("/img/"):
                f = out / "images" / Path(self.path).name
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            x = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                p = out / "ratings.csv"
                new = not p.exists()
                with open(p, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["time", "id", "answer", "note"])
                    if new:
                        w.writeheader()
                    w.writerow({"time": round(time.time(), 1), "id": x["id"], "answer": x["answer"], "note": x.get("note", "")})
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"rating page: {url} ({len(tasks)} worlds; answers go to {out / 'ratings.csv'}; Ctrl-C to stop, progress is kept)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


NAMES_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rate names</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f2;color:#222}
 header{padding:12px 20px;display:flex;justify-content:space-between;align-items:baseline}
 h1{font-size:17px;margin:0} #prog{color:#666}
 main{display:flex;gap:22px;padding:4px 20px 20px;align-items:flex-start}
 img{width:min(56vw,640px);border-radius:6px;background:#000}
 .side{max-width:520px} .txt{background:#fff;border-radius:8px;padding:12px 14px;margin-bottom:14px;line-height:1.4}
 .lab{font-size:12px;color:#777;text-transform:uppercase;letter-spacing:.04em}
 .q{margin:14px 0 6px;font-weight:600} button{font-size:15px;padding:7px 14px;margin:0 6px 6px 0;border-radius:6px;
 border:1px solid #bbb;background:#fff;cursor:pointer} button.on{background:#333;color:#fff;border-color:#333}
 textarea{width:100%;height:46px;margin-top:8px} #next{margin-top:12px;background:#2d6a4f;color:#fff;border-color:#2d6a4f}
 .hint{color:#777;font-size:13px}
</style></head><body>
<header><h1>Is the description right, and is it what is notable?</h1><span id="prog"></span></header>
<main><img id="im"><div class="side">
 <div class="txt"><div class="lab">sees</div><div id="sees"></div></div>
 <div class="txt"><div class="lab">push</div><div id="push"></div></div>
 <div class="hint">One whole world, four snapshots in reading order. Judge the "sees" text; the push is for context.</div>
 <div class="q">Is "sees" right about this world?</div>
 <div data-k="right"><button>yes</button><button>partly</button><button>no</button></div>
 <div class="q">Is it what is notable about this world (not a minor or generic detail)?</div>
 <div data-k="notable"><button>yes</button><button>no</button></div>
 <textarea id="note" placeholder="note (optional)"></textarea>
 <button id="next">next</button></div></main>
<script>
let todo=[],cur=null,ans={};
document.querySelectorAll('[data-k]').forEach(g=>g.querySelectorAll('button').forEach(b=>b.onclick=()=>{
  g.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');ans[g.dataset.k]=b.textContent;}));
async function load(){todo=await (await fetch('/todo')).json();show();}
function show(){document.getElementById('prog').textContent=todo.length+' left';
  if(!todo.length){document.querySelector('main').innerHTML='<p style="padding:20px">All done. You can close this tab.</p>';return;}
  cur=todo[0];ans={};document.querySelectorAll('button.on').forEach(x=>x.classList.remove('on'));
  document.getElementById('im').src='/img/'+cur.image;document.getElementById('sees').textContent=cur.sees;
  document.getElementById('push').textContent=cur.push;document.getElementById('note').value='';}
document.getElementById('next').onclick=async()=>{if(!ans.right||!ans.notable){alert('answer both questions');return;}
  await fetch('/answer',{method:'POST',body:JSON.stringify({id:cur.id,right:ans.right,notable:ans.notable,
  note:document.getElementById('note').value})});todo.shift();show();};
load();
</script></body></html>"""


def name_answers(out) -> dict:
    p = Path(out) / "ratings.csv"
    if not p.exists():
        return {}
    return {int(r["id"]): r for r in csv.DictReader(open(p))}


def rate_names(out, port: int = 8771, open_browser: bool = True) -> None:
    """One (world, description) at a time: is `sees` right (yes / partly / no), is it notable (yes / no), note ->
    <out>/ratings.csv. <out>/tasks.json: [{id, image, sees, push, ...}] (other fields, e.g. the model, are never sent
    to the page); images in <out>/images/. Progress is kept: answered ids are skipped."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    out = Path(out)
    tasks = [{k: t[k] for k in ("id", "image", "sees", "push")} for t in json.loads((out / "tasks.json").read_text())]
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, NAMES_PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/todo":
                done = name_answers(out)
                return self._send(200, json.dumps([t for t in tasks if t["id"] not in done]).encode(), "application/json")
            if self.path.startswith("/img/"):
                f = out / "images" / Path(self.path).name
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            x = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                p = out / "ratings.csv"
                new = not p.exists()
                with open(p, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["time", "id", "right", "notable", "note"])
                    if new:
                        w.writeheader()
                    w.writerow({"time": round(time.time(), 1), "id": x["id"], "right": x["right"], "notable": x["notable"],
                                "note": x.get("note", "")})
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"rating page: {url} ({len(tasks) - len(name_answers(out))} of {len(tasks)} left; answers go to "
          f"{out / 'ratings.csv'}; Ctrl-C to stop, progress is kept)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
