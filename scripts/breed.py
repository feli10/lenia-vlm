"""Breed MaCE-Lenia worlds by hand, Picbreeder style (11-06; project doc "Full run (seed 1) - audit").

  PYTHONPATH=src:scripts python3 scripts/breed.py serve --run RUNS/full-s1 --backend auto       # Mac (Apple GPU) or Colab
  PYTHONPATH=src:scripts python3 scripts/breed.py serve --res 0.5 --out RUNS/breed-half --backend auto
      # half resolution (80 cells, R 6.5), random founders only, its own folder (~4-5x faster)
  PYTHONPATH=src:scripts python3 scripts/breed.py serve --out ~/breed-dry --size 48 --warmup 20 --render 12 --gap 4
      # offline dry run (numpy, tiny worlds)

Why: the seed-1 audit couldn't tell whether refinement fails because the judge is wrong or because this substrate and
mutation don't allow small inherited improvements. Breeding by eye answers the second question without any judge, and
every pick is a label ("this child over its siblings") for testing judges or fitting a preference model later.

The page shows the parent and --kids mutants of it as films. Click a child to make it the parent; click the parent to
re-roll its children; pick the mutation strength for the next children; save worlds you like with the star (shift-click adds a note); go back.
"Random founders" starts from new random rules; a run's map elites (and gallery picks) can be used as starting points.
No scores are shown.

Worlds are grown exactly as the map grows them (the run's map settings: size, R, warmup, render, gap, start seed; the
same fixed noise start) and filmed as in the rating sessions (stage1._gif, frame every --every steps after warmup), so a
world here and in the run is the same world. Mutation is the expedition operator (stage1.mutants_of: lenia.mace.mutate
at the chosen strength, beta jitter scaled with it); ring edits only at strength >= 1 (as the map; v3's 0.3 had none).

Files (<out>, default <run>/breed or runs/breed):
- w/<id>.gif (film) and w/<id>.png (the four stills the judges see, full size): kept only while needed (the current
  generation, the last --keep generations you can go back to, starred worlds; the stills also for every parent).
  Everything else is deleted as you go, so a session stays small (about 0.3 MB per generation plus a window of a few
  tens of MB). Growing is deterministic per rule, so a deleted film is regrown on demand (e.g. going back further).
- w/<id>.jpg: a small copy of the stills for every world shown (the record; about 30 KB)
- saved/<id>.gif, .png, .json: worlds you saved with the star (film, stills, rule, note, lineage back to its
  founder, settings). Never pruned; unsaving removes them
- settings.json: the simulation settings; a folder holds one substrate (another --res needs another --out)
- worlds.jsonl: one row per world: id, rule, parent, strength, source, descriptors (motion, bodies, texture, coverage)
- log.jsonl: every event: show (parent, children, strength), pick (chosen, the siblings it beat, seconds looked),
  reroll, back, star, start
- session.json: the current state (a restarted server resumes it)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

import mace_qd as QD
import stage1 as S1
from lenia import backend, seeds
from lenia import mace as ME
from lenia.sample import sample
from lenia.sim import Rule, with_flow

DEFAULTS = {"size": 160, "R": 13, "warmup": 1500, "render": 1200, "gap": 400, "start_seed": 0, "sim_batch": 64,
            "beta": 2.0}                                   # seed 1's map settings (full-s1), used without --run
STRENGTHS = (0.3, 1.0, 2.0, 3.0)


# ---------- worlds ----------

def sim_settings(a) -> argparse.Namespace:
    if a.run:
        if a.res != 1:
            sys.exit("--res changes the substrate, so a run's map worlds don't carry over: use --res without --run")
        ea = S1.eval_settings(Path(a.run) / "map", a)      # the run's own settings (+ dry-run overrides)
    else:
        ea = argparse.Namespace(**DEFAULTS)
        if a.res != 1:                                     # grid, kernel radius and noise start scale together
            ea.size, ea.R = int(round(ea.size * a.res)), ea.R * a.res
        for k in ("size", "warmup", "render", "gap", "sim_batch"):
            if getattr(a, k, None):
                setattr(ea, k, getattr(a, k))
    ea.beta = float(getattr(ea, "beta", DEFAULTS["beta"]))
    return ea


def films(rules: list, ea, every: int, progress=None) -> list:
    """stage1.v4_films with a progress callback: per rule (film frames, the four-still picture, the stills)."""
    shape = (ea.size, ea.size)
    A0 = seeds.smooth_noise(ea.R, shape, 3, seed=ea.start_seed, size_r=3.0)
    steps = ea.warmup + ea.render
    gap = min(ea.gap, steps // (ME.FRAMES - 1))
    keep = {steps - i * gap for i in range(ME.FRAMES)}
    out = []
    for k in range(0, len(rules), ea.sim_batch):
        rr = rules[k:k + ea.sim_batch]
        b = backend.make_batch(rr, shape)
        X = backend.to_device(np.stack([A0] * len(rr)))
        fr, sn = [[] for _ in rr], [[] for _ in rr]
        for t in range(1, steps + 1):
            X = b.step(X)
            film_t = t >= ea.warmup and (t - ea.warmup) % every == 0
            if film_t or t in keep:
                x = backend.to_numpy(X).copy()
                for j in range(len(rr)):
                    if film_t:
                        fr[j].append(x[j])
                    if t in keep:
                        sn[j].append(x[j])
            if progress and t % 50 == 0:
                progress((k + len(rr) * t / steps) / len(rules))
        for f, s in zip(fr, sn):
            out.append((f, ME.picture(np.stack(s)), np.stack(s)))
    return out


class Store:
    """Grown worlds on disk, keyed by rule and simulation settings (growing is deterministic per rule)."""

    def __init__(self, out: Path, ea, a):
        self.out, self.ea, self.a = out, ea, a
        (out / "w").mkdir(parents=True, exist_ok=True)
        self.tag = json.dumps({k: getattr(ea, k) for k in ("size", "R", "warmup", "render", "gap", "start_seed")})
        self.rows = {}
        p = out / "worlds.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                self.rows[r["id"]] = r

    def key(self, rule: Rule) -> str:
        return hashlib.sha1((json.dumps(rule.to_dict(), sort_keys=True) + self.tag).encode()).hexdigest()[:12]

    def rule(self, wid: str) -> Rule:
        return Rule.from_dict(self.rows[wid]["rule"])

    def has_files(self, wid: str) -> bool:
        return (self.out / "w" / f"{wid}.gif").exists() and (self.out / "w" / f"{wid}.png").exists()

    def grow(self, items: list, progress=None) -> list:
        """items: (rule, meta) -> ids; grows the worlds that are new or whose films were deleted."""
        ids, todo = [], []
        for rule, meta in items:
            wid = self.key(rule)
            ids.append(wid)
            if not (wid in self.rows and self.has_files(wid)) and wid not in [w for w, _, _ in todo]:
                todo.append((wid, rule, meta))
        if not todo:
            return ids
        t0 = time.time()
        res = films([r for _, r, _ in todo], self.ea, self.a.every, progress)
        t1 = time.time()

        def write(job):
            (wid, rule, meta), (frames, pic, snaps) = job
            W = self.out / "w"
            S1._gif(frames, snaps, W / f"{wid}.gif", self.a.px, self.a.fps)
            Image.fromarray(pic).save(W / f"{wid}.png")
            Image.fromarray(pic).resize((322, 322), Image.BILINEAR).save(W / f"{wid}.jpg", quality=85)
            first, last = snaps[0], snaps[-1]
            mass = max(float(last.sum()), 1e-9)
            return {"id": wid, "rule": rule.to_dict(), **meta,
                    "motion": round(float(np.abs(last - first).sum() / mass), 4),
                    "bodies": QD.n_bodies(last), "texture": round(float(ME.texture(last)), 4),
                    "coverage": round(float(QD.coverage(last)), 4)}

        with ThreadPoolExecutor(4) as ex:                  # film encoding in parallel (PIL releases the GIL)
            rows = list(ex.map(write, zip(todo, res)))
        with open(self.out / "worlds.jsonl", "a") as fh:
            for row in rows:
                if row["id"] not in self.rows:             # a regrown film keeps its first row
                    self.rows[row["id"]] = row
                    fh.write(json.dumps(row) + "\n")
        self.timing = {"worlds": len(todo), "grow_s": round(t1 - t0, 1), "films_s": round(time.time() - t1, 1)}
        return ids

    def regrow(self, ids: list, progress=None) -> None:
        """Bring back deleted films (same rule, same settings -> the same world on the same machine)."""
        self.grow([(self.rule(w), {}) for w in ids if w and not self.has_files(w)], progress)

    def prune(self, keep_gif: set, keep_png: set) -> None:
        for f in (self.out / "w").glob("*.gif"):
            if f.stem not in keep_gif:
                f.unlink(missing_ok=True)
        for f in (self.out / "w").glob("*.png"):
            if f.stem not in keep_png:
                f.unlink(missing_ok=True)


# ---------- session ----------

class Session:
    def __init__(self, a):
        backend.configure(a.backend)
        self.a = a
        self.out = Path(a.out) if a.out else (Path(a.run) / "breed" if a.run else
                                              Path("runs/breed" if a.res == 1 else f"runs/breed-res{a.res:g}"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.ea = sim_settings(a)
        meta = self.out / "settings.json"                 # one substrate per folder
        mine = {k: getattr(self.ea, k) for k in ("size", "R", "warmup", "render", "gap", "start_seed", "beta")}
        if meta.exists() and json.loads(meta.read_text()) != mine:
            sys.exit(f"{self.out} holds a session grown with {meta.read_text().strip()}; these settings are {mine}. "
                     "Use another --out (or the same settings).")
        meta.write_text(json.dumps(mine))
        self.store = Store(self.out, self.ea, a)
        self.lock = threading.Lock()
        self.busy, self.progress, self.error = False, 0.0, None
        sp = self.out / "session.json"
        self.s = json.loads(sp.read_text()) if sp.exists() else {
            "seed": int(np.random.default_rng().integers(1 << 31)), "draws": 0, "gen": 0, "parent": None, "kids": [],
            "strength": a.strength, "stack": [], "stars": {}, "shown_at": None}
        self.s.setdefault("parents", [])
        self.s.setdefault("timing", None)
        self.starts = self._starts()
        self.prune()
        print(f"breeder: {backend._resolve()['label']}; worlds {self.ea.size}px, {self.ea.warmup}+{self.ea.render} steps; "
              f"{len(self.store.rows)} worlds on disk; {len(self.starts)} starting points; session in {self.out}",
              file=sys.stderr)

    # -- persistence and log
    def save(self):
        tmp = self.out / "session.tmp.json"
        tmp.write_text(json.dumps(self.s))
        tmp.replace(self.out / "session.json")

    def prune(self):
        """Delete films and full stills nobody can see any more (see the module docstring)."""
        live = set(self.s["kids"]) | {self.s["parent"]}
        for e in self.s["stack"][-self.a.keep:]:
            live |= set(e["kids"]) | {e["parent"]}
        stars = set(self.s["stars"])
        self.store.prune(live | stars, live | stars | set(self.s["parents"]))

    def log(self, event: str, **kw):
        with open(self.out / "log.jsonl", "a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 2), "event": event, "gen": self.s["gen"], **kw}) + "\n")

    # -- starting points from a run
    def _starts(self) -> list:
        if not self.a.run:
            return []
        M = Path(self.a.run) / "map"
        st = json.loads((M / "state.json").read_text())
        picks = set()
        g = Path(self.a.run) / "gallery" / "gallery.json"
        if g.exists():
            picks = {p["id"] for p in json.loads(g.read_text()).get("picks", [])}
        out = []
        for k, isl in enumerate(QD.islands_of(st)):
            for cell, e in sorted(isl.items()):
                name = Path(e["image"]).stem
                out.append({"name": name, "island": k, "cell": cell, "root": e.get("root"), "gallery": name in picks,
                            "rule": e["rule"]})
        out.sort(key=lambda x: (not x["gallery"], x["island"], x["cell"]))
        return out

    def thumb(self, name: str) -> Path | None:
        p = self.out / "thumbs" / f"{name}.jpg"
        if not p.exists():
            src = Path(self.a.run) / "map" / "img" / f"{name}.png"
            if not src.exists():
                return None
            p.parent.mkdir(exist_ok=True)
            Image.open(src).convert("RGB").resize((180, 180), Image.BILINEAR).save(p, quality=85)
        return p

    # -- background growing
    def _run(self, fn):
        def work():
            try:
                fn()
            except Exception as e:                     # shown on the page; the session stays usable
                self.error = f"{type(e).__name__}: {e}"
                print(f"breeder error: {self.error}", file=sys.stderr)
            finally:
                with self.lock:
                    self.busy, self.progress = False, 1.0
                    self.save()
        with self.lock:
            if self.busy:
                return False
            self.busy, self.progress, self.error = True, 0.0, None
        threading.Thread(target=work, daemon=True).start()
        return True

    def _prog(self, x):
        self.progress = x

    def _rng(self):
        self.s["draws"] += 1
        return np.random.default_rng([self.s["seed"], self.s["draws"]])

    def _show(self, parent, kids, why, strength="now"):
        self.s["gen"] += 1
        self.s["parent"], self.s["kids"] = parent, kids
        self.s["kids_strength"] = (None if parent is None else self.s["strength"]) if strength == "now" else strength
        self.s["shown_at"] = time.time()
        if parent and parent not in self.s["parents"]:
            self.s["parents"].append(parent)
        t = getattr(self.store, "timing", None)
        self.s["timing"], self.store.timing = t, None
        self.log("show", parent=parent, kids=kids, strength=self.s["kids_strength"], why=why, timing=t)
        self.prune()

    def _children(self, parent: str, why: str):
        sc = float(self.s["strength"])
        rules = S1.mutants_of(self.store.rule(parent), self._rng(), self.a.kids, self.ea, sc, 1.0 if sc >= 1 else 0.0)
        ids = self.store.grow([(r, {"parent": parent, "strength": sc, "source": "mutant"}) for r in rules], self._prog)
        self._show(parent, ids, why)

    def _founders(self):
        rng = self._rng()
        rules = [with_flow(r, "mace", self.ea.beta) for r in
                 sample("multichannel", self.a.kids + 1, seed=int(rng.integers(1 << 31)), R=self.ea.R, channels=3,
                        self_k=1, cross_k=1)]
        ids = self.store.grow([(r, {"parent": None, "strength": None, "source": "founder"}) for r in rules], self._prog)
        self._show(None, ids, "founders")

    # -- actions (called from the page)
    def start(self, body) -> bool:
        if body.get("random"):
            self.log("start", source="random")
            self.s["stack"] = []
            return self._run(self._founders)
        if body.get("world"):                          # a world grown here (a starred one)
            wid = body["world"]
            if wid not in self.store.rows:
                return False
            self.log("start", source="world", world=wid)
            self.s["stack"] = []

            def go_w():
                self.store.regrow([wid], self._prog)
                self._children(wid, "start")
            return self._run(go_w)
        e = next((x for x in self.starts if x["name"] == body.get("elite")), None)
        if e is None:
            return False

        def go():
            rule = Rule.from_dict(e["rule"])
            wid = self.store.grow([(rule, {"parent": None, "strength": None, "source": f"elite:{e['name']}"})],
                                  self._prog)[0]
            self._children(wid, "start")
        self.log("start", source="elite", elite=e["name"])
        self.s["stack"] = []
        return self._run(go)

    def pick(self, wid: str) -> bool:
        if wid not in self.s["kids"]:
            return False
        looked = round(time.time() - (self.s["shown_at"] or time.time()), 1)
        self.log("pick", chosen=wid, parent=self.s["parent"], others=[k for k in self.s["kids"] if k != wid],
                 strength=self.s.get("kids_strength"), seconds=looked)
        self.s["stack"].append({"parent": self.s["parent"], "kids": self.s["kids"],
                                "strength": self.s.get("kids_strength")})
        return self._run(lambda: self._children(wid, "pick"))

    def reroll(self) -> bool:
        self.log("reroll", parent=self.s["parent"], rejected=self.s["kids"], strength=self.s.get("kids_strength"),
                 seconds=round(time.time() - (self.s["shown_at"] or time.time()), 1))
        if self.s["parent"] is None:
            return self._run(self._founders)
        return self._run(lambda: self._children(self.s["parent"], "reroll"))

    def back(self) -> bool:
        if self.busy or not self.s["stack"]:
            return False
        prev = self.s["stack"].pop()
        self.log("back", left=self.s["parent"], to=prev["parent"])
        ids = [prev["parent"]] + prev["kids"]
        if all(w is None or self.store.has_files(w) for w in ids):
            self._show(prev["parent"], prev["kids"], "back", prev.get("strength"))
            self.save()
            return True

        def go():                                      # further back than the kept films: regrow them first
            self.store.regrow(ids, self._prog)
            self._show(prev["parent"], prev["kids"], "back", prev.get("strength"))
        return self._run(go)

    def lineage(self, wid: str) -> list:
        out = []
        while wid and wid in self.store.rows and len(out) < 10000:
            out.append(wid)
            wid = self.store.rows[wid].get("parent")
        return out

    def star(self, wid: str, note: str) -> bool:
        """Save a world: <out>/saved/<id>.gif, .png (stills) and .json (rule, note, lineage, settings)."""
        if wid not in self.store.rows:
            return False
        sv = self.out / "saved"
        if wid in self.s["stars"] and not note:
            del self.s["stars"][wid]
            for ext in (".gif", ".png", ".json"):
                (sv / f"{wid}{ext}").unlink(missing_ok=True)
            self.log("unstar", world=wid)
        else:
            self.s["stars"][wid] = {"note": note, "t": round(time.time())}
            sv.mkdir(exist_ok=True)
            for ext in (".gif", ".png"):
                f = self.out / "w" / f"{wid}{ext}"
                if f.exists():
                    (sv / f"{wid}{ext}").write_bytes(f.read_bytes())
            row = self.store.rows[wid]
            (sv / f"{wid}.json").write_text(json.dumps(
                {"id": wid, "note": note, "saved": time.strftime("%Y-%m-%d %H:%M"), "rule": row["rule"],
                 "lineage": self.lineage(wid), "settings": json.loads((self.out / "settings.json").read_text()),
                 **{k: row.get(k) for k in ("motion", "bodies", "texture", "coverage")}}, indent=1))
            self.log("star", world=wid, note=note)
        self.save()
        return True

    def home(self) -> bool:
        if self.busy:
            return False
        self.log("home", left=self.s["parent"])
        self.s["parent"], self.s["kids"], self.s["stack"] = None, [], []
        self.save()
        return True

    def view(self) -> dict:
        return {"busy": self.busy, "progress": round(self.progress, 3), "error": self.error, "gen": self.s["gen"],
                "parent": self.s["parent"], "kids": self.s["kids"], "strength": self.s["strength"],
                "kids_strength": self.s.get("kids_strength"), "timing": self.s.get("timing"),
                "strengths": STRENGTHS, "depth": len(self.s["stack"]), "stars": self.s["stars"],
                "starts": [{k: v for k, v in x.items() if k != "rule"} for x in self.starts],
                "backend": backend._resolve()["label"]}


# ---------- the page ----------

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MaCE-Lenia breeder</title>
<style>
:root{--bg:#111214;--panel:#1b1d21;--line:#2c2f35;--text:#e6e6e6;--dim:#9aa0a6;--accent:#f2c14e;--pick:#6fb1ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
header b{font-size:15px}.sp{flex:1}
button{background:#2a2d33;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer}
button:hover{border-color:#555}button.on{background:var(--accent);color:#111;border-color:var(--accent)}button:disabled{opacity:.4;cursor:default}
.seg{display:inline-flex;gap:0}.seg button{border-radius:0}.seg button:first-child{border-radius:6px 0 0 6px}.seg button:last-child{border-radius:0 6px 6px 0}
#status{color:var(--dim);min-width:150px}
main{padding:16px;max-width:1200px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.tile{position:relative;background:#000;border:2px solid transparent;border-radius:8px;overflow:hidden;cursor:pointer}
.tile:hover{border-color:var(--pick)}.tile.parent{border-color:#555}.tile.parent:hover{border-color:var(--accent)}
.tile img{display:block;width:100%;aspect-ratio:1/1;image-rendering:auto}
.tag{position:absolute;left:6px;top:6px;background:rgba(0,0,0,.6);padding:1px 6px;border-radius:4px;font-size:12px;color:var(--dim)}
.star{position:absolute;right:6px;top:4px;font-size:20px;color:#888;background:none;border:none;padding:2px 4px}
.star.on{color:var(--accent)}
#veil{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:9;font-size:16px}
#veil>div{background:var(--panel);padding:16px 22px;border-radius:8px;border:1px solid var(--line);min-width:220px}
.bar{height:6px;background:#333;border-radius:3px;margin-top:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--pick);width:0}
h2{font-size:15px;margin:22px 0 8px;color:var(--dim);font-weight:600}
.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px}
.thumbs div{position:relative;cursor:pointer;border:2px solid transparent;border-radius:6px;overflow:hidden}.thumbs div:hover{border-color:var(--pick)}
.thumbs img{display:block;width:100%}
.thumbs span{position:absolute;left:4px;bottom:3px;font-size:11px;background:rgba(0,0,0,.6);padding:0 4px;border-radius:3px;color:var(--dim)}
.help{color:var(--dim);font-size:13px;margin:10px 0 0}
#err{color:#ff8a80}
@media (min-width:701px){.grid{max-width:calc(100vh - 100px);margin:0 auto}}
@media (max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><b>MaCE-Lenia breeder</b>
<span class="seg" id="str"></span>
<button id="back" title="b">Back</button><button id="reroll" title="r">New children</button>
<button id="view" title="v">Stills</button><button id="home">Start over</button>
<span class="sp"></span><span id="status"></span></header>
<main><div id="err"></div><div id="app"></div></main>
<div id="veil"><div><span id="vtxt">Growing…</span><div class="bar"><i id="vbar"></i></div></div></div>
<script>
let S=null, stills=false, poll=null;
const $=id=>document.getElementById(id);
async function post(u,b){const r=await fetch(u,{method:'POST',body:JSON.stringify(b||{})});if(!r.ok&&r.status!==409)$('err').textContent='request failed: '+u;await refresh();}
function src(id){return 'w/'+id+(stills?'.png':'.gif');}
function tile(id,isParent,i){
  const st=S.stars[id]?'on':'';
  return `<div class="tile ${isParent?'parent':''}" data-id="${id}" data-p="${isParent?1:0}">
  <img src="${src(id)}" loading="lazy"><span class="tag">${isParent?'parent · click for new children':(i)}</span>
  <button class="star ${st}" data-s="${id}" title="save this world (shift-click to add a note); click again to unsave">★</button></div>`;}
function render(){
  $('str').innerHTML=S.strengths.map(x=>`<button class="${x==S.strength?'on':''}" data-x="${x}" title="mutation strength">${x}</button>`).join('');
  $('back').disabled=!S.depth||S.busy; $('reroll').disabled=S.busy||!S.kids.length; $('view').textContent=stills?'Films':'Stills';
  const tm=S.timing?` · grew ${S.timing.worlds} in ${S.timing.grow_s} s + films ${S.timing.films_s} s`:'';
  $('status').textContent=(S.kids.length?`generation ${S.gen} · depth ${S.depth}`+(S.kids_strength!=null?` · these children at ${S.kids_strength}`:' · founders')+tm+' · ':'')+S.backend;
  $('err').textContent=S.error||'';
  $('veil').style.display=S.busy?'flex':'none'; $('vbar').style.width=Math.round(100*S.progress)+'%';
  if(!S.kids.length){
    const st=Object.keys(S.stars);
    $('app').innerHTML=`<p class="help">Click a starting world. Then click the child you like to make it the parent; click the parent for new children; the strength buttons set how far the next children move (0.3 small, 3 = the map's). Keys: 1–9 pick, r new children, b back, v films/stills.</p>
    <button id="rand">Random founders</button>
    ${st.length?`<h2>Saved</h2><div class="thumbs">${st.map(w=>`<div data-w="${w}"><img src="w/${w}.png"><span>${(S.stars[w].note||'').slice(0,18)}</span></div>`).join('')}</div>`:''}
    ${S.starts.length?`<h2>From the run's map (gallery picks first)</h2><div class="thumbs">${S.starts.map(e=>`<div data-e="${e.name}"><img src="thumb/${e.name}.jpg" loading="lazy"><span>${e.gallery?'★ ':''}I${e.island} ${e.cell}</span></div>`).join('')}</div>`:''}`;
    if($('rand'))$('rand').onclick=()=>post('start',{random:1});
    document.querySelectorAll('[data-e]').forEach(d=>d.onclick=()=>post('start',{elite:d.dataset.e}));
    document.querySelectorAll('[data-w]').forEach(d=>d.onclick=()=>post('start',{world:d.dataset.w}));
  } else {
    const t=[]; if(S.parent)t.push(tile(S.parent,true,0)); S.kids.forEach((k,i)=>t.push(tile(k,false,i+1)));
    $('app').innerHTML=`<div class="grid">${t.join('')}</div>`;
    document.querySelectorAll('.tile').forEach(d=>d.onclick=e=>{if(e.target.dataset.s)return; if(S.busy)return;
      d.dataset.p==='1'?post('reroll'):post('pick',{id:d.dataset.id});});
  }
  document.querySelectorAll('[data-s]').forEach(b=>b.onclick=e=>{e.stopPropagation();
    const note=e.shiftKey?(prompt('Note for this world:')||''):''; post('star',{id:b.dataset.s,note});});
  document.querySelectorAll('[data-x]').forEach(b=>b.onclick=()=>post('strength',{x:+b.dataset.x}));
}
async function refresh(){S=await (await fetch('state')).json();render();
  clearTimeout(poll); if(S.busy)poll=setTimeout(refresh,800);}
$('back').onclick=()=>post('back');$('reroll').onclick=()=>post('reroll');$('home').onclick=()=>post('home');
$('view').onclick=()=>{stills=!stills;render();};
document.addEventListener('keydown',e=>{if(!S||S.busy||e.target.tagName==='INPUT')return;
  const n=parseInt(e.key); if(n>=1&&n<=S.kids.length)post('pick',{id:S.kids[n-1]});
  else if(e.key==='r'&&S.kids.length)post('reroll');else if(e.key==='b')post('back');else if(e.key==='v'){stills=!stills;render();}});
refresh();
</script></body></html>"""


def serve(a) -> None:
    S = Session(a)
    types = {".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(("application", "text")) else "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0].lstrip("/")
            if p in ("", "index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "state":
                return self._send(200, json.dumps(S.view()).encode(), "application/json")
            if p.startswith("w/") and Path(p).suffix in (".gif", ".png"):
                f = S.out / "w" / Path(p).name
            elif p.startswith("thumb/") and S.a.run:
                f = S.thumb(Path(p).stem)
            else:
                f = None
            if f is None or not f.exists():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, f.read_bytes(), types[f.suffix])

        def do_POST(self):
            p = self.path.split("?")[0].lstrip("/")
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if p == "strength":
                x = float(body.get("x", 1.0))
                ok = x in STRENGTHS
                if ok:
                    S.s["strength"] = x
                    S.save()
            elif S.busy and p in ("start", "pick", "reroll", "back", "home"):
                return self._send(409, b"busy", "text/plain")
            elif p == "start":
                ok = S.start(body)
            elif p == "pick":
                ok = S.pick(str(body.get("id")))
            elif p == "reroll":
                ok = bool(S.s["kids"]) and S.reroll()
            elif p == "back":
                ok = S.back()
            elif p == "home":
                ok = S.home()
            elif p == "star":
                ok = S.star(str(body.get("id")), str(body.get("note") or ""))
            else:
                return self._send(404, b"not found", "text/plain")
            return self._send(200 if ok else 400, json.dumps({"ok": bool(ok)}).encode(), "application/json")

    srv = ThreadingHTTPServer((a.host, a.port), H)
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/"
    print(f"breeder at {url} (Ctrl-C to stop; the session resumes on restart)", file=sys.stderr)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--run", default=None, help="a run folder: its map settings and elites (else seed 1's settings)")
    s.add_argument("--out", default=None, help="session folder (default <run>/breed, else runs/breed)")
    s.add_argument("--backend", default="auto")
    s.add_argument("--kids", type=int, default=8, help="children per generation")
    s.add_argument("--strength", type=float, default=1.0, choices=STRENGTHS, help="starting mutation strength")
    s.add_argument("--every", type=int, default=60,
                   help="simulation steps per film frame (the rating sessions used 30 at 15 fps; 60 at 8 fps keeps "
                        "the loop length and halves film size and encoding time)")
    s.add_argument("--px", type=int, default=280)
    s.add_argument("--fps", type=int, default=8)
    s.add_argument("--keep", type=int, default=3, help="generations back whose films are kept (older ones regrow)")
    s.add_argument("--res", type=float, default=1.0,
                   help="resolution factor: 0.5 = an 80-cell grid with R 6.5 and a matching noise start (~4-5x faster). "
                        "A related but different substrate (MaCE's flow works per cell), so not with --run")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8776)
    s.add_argument("--no-browser", action="store_true")
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        s.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (offline dry runs only)")
    a = ap.parse_args()
    serve(a)


if __name__ == "__main__":
    main()
