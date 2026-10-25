"""Can Jev recognise what a MaCE world is doing? Checks for the goal-library idea (agent sessions, step 1).

The goal library (data/goal_library.json, ~100 entries) pairs an observation ("small bodies are attached to
the edge of the main body") with a direction ("make them bud off and drift away"). In an agent session Jev
would be shown a world and a random menu of ~8 observations and asked which best matches; the chosen entry's
direction then steers the session. Before building sessions on that, three checks:

  A  image dependence: the same 5 menus on 40 worlds. If Jev ignores the image, each menu gets the same pick
     on every world (high modal share, low entropy). Also: how often it picks the control entries (things a
     MaCE world cannot show: letters, cars, people ...), which should be ~never.
  B  order stability: menu 0 on the same 40 worlds, shown in 2 more random orders. How often is the same
     entry picked (chance ~1/8)?
  C  agreement with you: 20 worlds, menu 0 in a fresh order; you pick in the browser (`rate`), and `report`
     compares your picks with Jev's.

  uv run python scripts/goal_match.py run                     # A + B (about 280 Jev calls, a few cents)
  uv run python scripts/goal_match.py rate                    # C: your 20 picks, in the browser
  uv run python scripts/goal_match.py report                  # report.md + sheet.png (worlds with Jev's pick)
  ... run --judges jev,luna                                   # also ask Luna (Decisions API), for comparison
  ... run --judges fake                                       # offline dry run
  ... --library data/goal_library_v2.json --out runs/goal-match-v2   # library v2 (built from mace-qd2 elites
                                                              # that are not among the 40 test worlds)

Worlds: the `*_final.png` images of runs/mace-qd2/check/images (evolved elites, 4 snapshots each).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lenia.config import REPO_ROOT, load
from lenia.judge import ChoiceQuery, Judge, load_env

INSTRUCTIONS = ("You are shown one world from a mass-conserving variant of Lenia, a continuous cellular automaton used "
                "in artificial-life research. The image {W} shows the whole world as four snapshots taken some time "
                "apart, in reading order (top-left, top-right, bottom-left, bottom-right); the colours are three "
                "interacting chemical channels. Which description best matches what you see in this world?")
N_MENUS, MENU_SIZE, N_WORLDS, N_HUMAN, N_PERM = 5, 8, 40, 20, 2


def setup(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "design.json").exists():
        return json.loads((out / "design.json").read_text())
    lib = json.loads((REPO_ROOT / a.library).read_text())
    rng = np.random.default_rng(a.seed)
    worlds = sorted(p.name for p in Path(a.images).glob("*_final.png"))
    worlds = [worlds[i] for i in sorted(rng.choice(len(worlds), min(N_WORLDS, len(worlds)), replace=False))]
    first = {}                                   # one entry per distinct observation (several directions may share one)
    for g in lib:
        first.setdefault(g["sees"], g)
    real = [g["id"] for g in first.values() if g["kind"] != "control"]
    ctrl = [g["id"] for g in lib if g["kind"] == "control"]
    menus = []
    for _ in range(N_MENUS):
        m = list(rng.choice(real, MENU_SIZE - 1, replace=False)) + [str(rng.choice(ctrl))]
        menus.append([m[i] for i in rng.permutation(MENU_SIZE)])
    perms = [[menus[0][i] for i in rng.permutation(MENU_SIZE)] for _ in range(N_PERM)]
    human = [[menus[0][i] for i in rng.permutation(MENU_SIZE)] for _ in range(N_HUMAN)]
    d = {"images": str(Path(a.images).resolve()), "worlds": worlds, "menus": menus, "perms": perms, "human_orders": human,
         "library": {g["id"]: g for g in lib}}
    (out / "design.json").write_text(json.dumps(d, indent=1))
    return d


def queries(d: dict) -> list:
    """(test, world, menu index, order) for every Jev call."""
    q = [("A", w, m, d["menus"][m]) for m in range(N_MENUS) for w in d["worlds"]]
    q += [("B", w, k, d["perms"][k]) for k in range(N_PERM) for w in d["worlds"]]
    return q


def ask(judge, d: dict, items: list, desc: str) -> list:
    lib = d["library"]
    img = {w: np.asarray(Image.open(Path(d["images"]) / w).convert("RGB")) for w in {i[1] for i in items}}
    qs = [ChoiceQuery(img[w], INSTRUCTIONS, {gid: lib[gid]["sees"] + "." for gid in order}) for _, w, _, order in items]
    return judge.choose_many(qs, desc=desc)


def run(a) -> None:
    d = setup(a)
    out = Path(a.out)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    items = queries(d)
    res = {}
    for jn in a.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, out / "judge_cache.sqlite", yes=True)
        res[jn] = ask(judge, d, items, f"match {jn}")
        print(f"{jn}: spent ${judge.spent:.4f}", file=sys.stderr)
    json.dump({"items": items, "probs": res}, open(out / "results.json", "w"))
    report(a)


def _human(out: Path) -> dict:
    p = out / "human.csv"
    return {r["world"]: r["pick"] for r in csv.DictReader(open(p))} if p.exists() else {}


def report(a) -> None:
    out = Path(a.out)
    d = json.loads((out / "design.json").read_text())
    lib = d["library"]
    r = json.loads((out / "results.json").read_text())
    items = [tuple(x) for x in r["items"]]
    lines = [f"# Goal matching: can the judge recognise what a world is doing?\n",
             f"{len(d['worlds'])} evolved worlds (mace-qd2), {N_MENUS} menus of {MENU_SIZE} observations (7 real + 1 control "
             "each), chance = 1/8.\n"]
    human = _human(out)
    for jn, probs in r["probs"].items():
        pick = {}
        for (t, w, m, order), p in zip(items, probs):
            if p is not None:
                pick[(t, w, m)] = max(p, key=p.get)
        lines += [f"## {jn}\n", "**A. Image dependence** (same menu, different worlds). If the image is ignored, the modal "
                  "share is 1.0 and the distinct count is 1.\n",
                  "| menu | worlds | distinct picks | modal share | norm. entropy | control picks |", "|---|---|---|---|---|---|"]
        for m in range(N_MENUS):
            ps = [pick[("A", w, m)] for w in d["worlds"] if ("A", w, m) in pick]
            if not ps:
                continue
            vals, cnt = np.unique(ps, return_counts=True)
            f = cnt / cnt.sum()
            ent = float(-(f * np.log(f)).sum() / np.log(MENU_SIZE))
            nctrl = sum(lib[x]["kind"] == "control" for x in ps)
            lines.append(f"| {m} | {len(ps)} | {len(vals)} | {cnt.max() / len(ps):.2f} | {ent:.2f} | {nctrl} |")
        agree = []
        for w in d["worlds"]:
            base = pick.get(("A", w, 0))
            for k in range(N_PERM):
                if base and ("B", w, k) in pick:
                    agree.append(pick[("B", w, k)] == base)
        lines += ["", f"**B. Order stability:** same world and menu, reshuffled: same entry picked in "
                  f"{np.mean(agree):.2f} of {len(agree)} comparisons (chance ~0.13).\n" if agree else ""]
        pos = []
        for (t, w, m, order), p in zip(items, probs):
            if p is not None:
                pos.append(order.index(max(p, key=p.get)))
        if pos:
            lines.append(f"Position of the pick in the menu (0 = first shown): {np.bincount(pos, minlength=MENU_SIZE).tolist()}\n")
        if human:
            hits = [pick.get(("A", w, 0)) == human[w] for w in human if ("A", w, 0) in pick]
            lines.append(f"**C. Agreement with Felix** (menu 0): {sum(hits)}/{len(hits)} = {np.mean(hits):.2f} (chance ~0.13).\n")
    if human:
        lines += ["Felix's picks:", ""] + [f"- {w}: {lib[g]['sees']}" for w, g in sorted(human.items())] + [""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _sheet(out, d, items, r)


def _sheet(out: Path, d: dict, items: list, r: dict) -> None:
    from lenia import render
    jn = next(iter(r["probs"]))
    lib = d["library"]
    pick = {}
    for (t, w, m, order), p in zip(items, r["probs"][jn]):
        if t == "A" and m == 0 and p is not None:
            g = max(p, key=p.get)
            pick[w] = (lib[g]["sees"], p[g])
    rows = []
    for w in d["worlds"][:24]:
        if w in pick:
            im = Image.open(Path(d["images"]) / w).convert("RGB").resize((240, 240))
            rows.append((f"{pick[w][1]:.2f} {pick[w][0][:44]}", im))
    render.sheet(rows, cols=4, title=f"{jn}'s pick from menu 0 (probability, description)").save(out / "sheet.png")


PAGE = """<!doctype html><meta charset=utf-8><title>goal matching</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:20px}img{width:520px;border:1px solid #333}
button{display:block;width:520px;text-align:left;margin:4px 0;padding:8px;font:15px system-ui;background:#222;color:#eee;
border:1px solid #444;cursor:pointer}button:hover{background:#2c3e50}#n{color:#999}</style>
<div id=n></div><p>Which description best matches what you see in this world? (four snapshots, reading order)</p>
<div style="display:flex;gap:20px"><img id=im><div id=opts></div></div>
<script>
let todo=[];async function load(){const r=await fetch('/todo');todo=await r.json();show()}
function show(){if(!todo.length){document.body.innerHTML='<h2>Done, thank you. Close this tab and press Ctrl-C.</h2>';return}
const t=todo[0];document.getElementById('n').textContent=todo.length+' left';document.getElementById('im').src='/img/'+t.world;
const o=document.getElementById('opts');o.innerHTML='';for(const [id,txt] of t.options){const b=document.createElement('button');
b.textContent=txt;b.onclick=async()=>{await fetch('/pick',{method:'POST',body:JSON.stringify({world:t.world,pick:id})});
todo.shift();show()};o.appendChild(b)}}
load()</script>"""


def rate(a) -> None:
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    out = Path(a.out)
    d = setup(a)
    lib = d["library"]
    lock = threading.Lock()
    path = out / "human.csv"
    tasks = [{"world": w, "options": [[g, lib[g]["sees"]] for g in order]}
             for w, order in zip(d["worlds"][:N_HUMAN], d["human_orders"])]

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
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/todo":
                done = _human(out)
                return self._send(200, json.dumps([t for t in tasks if t["world"] not in done]).encode(), "application/json")
            if self.path.startswith("/img/"):
                f = Path(d["images"]) / Path(self.path).name
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            x = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                new = not path.exists()
                with open(path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["time", "world", "pick"])
                    if new:
                        w.writeheader()
                    w.writerow({"time": round(time.time(), 1), "world": x["world"], "pick": x["pick"]})
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"rating page: {url} (answers go to {path}; Ctrl-C to stop, progress is kept)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "rate", "report"])
    ap.add_argument("--out", default="runs/goal-match")
    ap.add_argument("--library", default="data/goal_library.json", help="relative to the repo root")
    ap.add_argument("--images", default="runs/mace-qd2/check/images")
    ap.add_argument("--judges", default="jev")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    {"run": run, "rate": rate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
