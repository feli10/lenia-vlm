"""Motion benchmark, Stage A: can Jev see motion? (11-07; project doc "Jev motion benchmark - spec")

In seed 1, Jev ignored the change between snapshots that predicts Felix's choices; Stage A asks whether that's because
of how motion is shown. On pairs where the simulation says which world changes more, including a world against its own
frozen copy, Jev is scored on seed 1's picture (R0) and on four other pictures of the same frames, each with a
description of the time order. The best picture on dev is confirmed once on test.

  python scripts/motionbench.py build --source RUNS/full-s1 --out RUNS/motion-bench --backend auto   # Colab, no calls
  PYTHONPATH=src:scripts python3 scripts/motionbench.py check --out RUNS/motion-bench    # Felix: 10 pairs, films
  PYTHONPATH=src:scripts python3 scripts/motionbench.py judge --out RUNS/motion-bench --split dev    # ~$0.15
  PYTHONPATH=src:scripts python3 scripts/motionbench.py report --out RUNS/motion-bench --split dev
  Then once: judge --split test --configs R0,<best>; report --split test.
  Only if no picture works: gemini --split test (Gemini on an 8-frame film strip: input limit or model limit?).
  Offline dry run (device shell: SQLite needs the VM's $HOME):
  PYTHONPATH=src:scripts python3 scripts/motionbench.py build --source RUNS/full-s1 --out ~/mb-dry --max-worlds 60 \\
      --size 96 --warmup 200 --render 120 --gap 40;  then judge --judge fake, gemini --vlm fake.

Details:
- Worlds: seed 1's final map elites, map entry duels (challenger and elite) and expedition siblings, regrown exactly as
  the map grew them (the stored frames reproduce lenia.mace.picture byte for byte; build checks).
- Truth: change = mean |snapshot(t+1) - snapshot(t)| / mass over the four snapshots every picture shows (the map's
  motion descriptor uses first vs last; this uses each step). Felix's 10-pair check confirms it matches what he sees.
- Change pairs: easy (ratio >= --easy) from random elites, close (--close-lo..--close-hi) from comparisons the
  pipeline made (duels, siblings); random pairs fill shortfalls. Frozen pairs: a world that visibly changes (change >=
  the pool median) vs itself with its last snapshot repeated. World-disjoint dev / test; at most 2 items per world.
- Configurations: R0 (the pipeline's grid and preamble) and grid / line / ghosts / labels with the time order
  described. One question for all (lenia.prompts.bench_prompt), first / second, both orders.
From papers: none (frozen copies are a standard ablation). Ours: the design.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import threading
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mace_qd as QD
import stage1 as S1
from lenia import backend, render, seeds
from lenia import mace as ME
from lenia import prompts as PR
from lenia.config import REPO_ROOT, load
from lenia.judge import Judge, Query, load_env
from lenia.sim import Rule

NFILM = 25                          # film frames over the snapshot window, gap/8 apart
KEEP = list(range(0, NFILM, 2))     # the 13 frames stored per world, gap/4 apart
SNAP = [0, 4, 8, 12]                # the four snapshots, as indices into KEEP
GHOST = list(range(0, 13, 2))       # the ghosts picture: 7 of the 13, gap/2 apart
CONFIGS = ["R0"] + list(PR.BENCH_FORMATS)
GAPC = 40                           # separator colour, as lenia.mace.picture


def read_jsonl(p: Path) -> list:
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def fid(wid: str) -> str:
    return hashlib.sha1(wid.encode()).hexdigest()[:16]


# ---------- build: pool, growing, the change measure ----------

def source_pool(F: Path, a, rng) -> tuple[dict, list]:
    """Worlds (id -> rule, source, stored still) and the pipeline's own pairs [(x, y, source)]."""
    MAP = F / "map"
    led = {r["id"]: r for r in read_jsonl(MAP / "ledger.jsonl")}
    st = json.loads((MAP / "state.json").read_text())
    elites = [(Path(e["image"]).stem, e["rule"]) for m in QD.islands_of(st) for _, e in sorted(m.items())]
    duels = [(r["id"], r["opponent"]) for r in led.values()
             if r["kind"] == "duel" and r.get("opponent") in led and led[r["opponent"]].get("rule")]
    groups = defaultdict(list)
    for Rr in sorted((F / "rounds").glob("r*")):
        for x in read_jsonl(Rr / "exp" / "judged.jsonl"):
            groups[(x["round"], x["trigger"], x["step"])].append(x)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    n_el, n_du, n_sib = len(elites), a.duels, a.siblings
    if a.max_worlds:
        n_el, n_du, n_sib = a.max_worlds // 3, a.max_worlds // 6, a.max_worlds // 6
    W, by_rule, pairs = {}, {}, []

    def add(wid, rule, src, still=None):
        k = hashlib.sha1(json.dumps(rule, sort_keys=True).encode()).hexdigest()
        if k not in by_rule:
            by_rule[k] = wid
            W[wid] = {"rule": rule, "src": src, "still": still}
        return by_rule[k]

    for wid, rule in [elites[int(j)] for j in rng.permutation(len(elites))[:n_el]]:
        add(f"m:{wid}", rule, "elite", f"map/img/{wid}.png")
    for j in rng.permutation(len(duels))[:n_du]:
        x, o = duels[int(j)]
        u, v = add(f"m:{x}", led[x]["rule"], "duel", f"map/img/{x}.png"), add(f"m:{o}", led[o]["rule"], "duel", f"map/img/{o}.png")
        if u != v:
            pairs.append((u, v, "duel"))
    keys = sorted(groups)
    for j in rng.permutation(len(keys))[:n_sib]:
        g = groups[keys[int(j)]]
        c1, c2 = (g[int(i)] for i in rng.choice(len(g), 2, replace=False))
        u, v = (add(f"x:r{c['round']}:{c['trigger']}:s{c['step']}:c{c['child']}", c["rule"], "sibling") for c in (c1, c2))
        if u != v:
            pairs.append((u, v, "sibling"))
    print(f"pool: {len(W)} worlds ({dict(Counter(w['src'] for w in W.values()))}); pipeline pairs "
          f"{dict(Counter(p[2] for p in pairs))}", file=sys.stderr)
    return W, pairs


def window(ea) -> tuple[int, int, int]:
    steps = ea.warmup + ea.render
    gap = min(ea.gap, steps // (ME.FRAMES - 1))              # as lenia.mace.simulate
    if gap % 8:
        sys.exit(f"gap {gap} must be a multiple of 8")
    return steps, gap, steps - 3 * gap


def grow(rules: list, ea, times: list, steps: int):
    """Yield each rule's frames at `times` (float32, (len(times), C, H, W)), grown from the map's fixed start."""
    shape = (ea.size, ea.size)
    A0 = seeds.smooth_noise(ea.R, shape, 3, seed=ea.start_seed, size_r=3.0)
    want = set(times)
    for k in range(0, len(rules), ea.sim_batch):
        rr = rules[k:k + ea.sim_batch]
        b = backend.make_batch(rr, shape)
        X = backend.to_device(np.stack([A0] * len(rr)))
        films = [[A0.astype(np.float32).copy()] if 0 in want else [] for _ in rr]
        for t in range(1, steps + 1):
            X = b.step(X)
            if t in want:
                x = backend.to_numpy(X)
                for j in range(len(rr)):
                    films[j].append(np.asarray(x[j], np.float32).copy())
        for f in films:
            yield np.stack(f)
        print(f"grown {min(k + ea.sim_batch, len(rules))}/{len(rules)}", file=sys.stderr, flush=True)


def change(snaps: np.ndarray) -> float:
    """Mean |snapshot(t+1) - snapshot(t)| over the four snapshots, / mass."""
    mass = max(float(snaps.sum((1, 2, 3)).mean()), 1e-9)
    return float(np.mean([np.abs(snaps[i + 1] - snaps[i]).sum() for i in range(len(snaps) - 1)]) / mass)


def build(a) -> None:
    F, out = Path(a.source), Path(a.out)
    if (out / "pool.json").exists() and not a.force:
        sys.exit(f"{out / 'pool.json'} exists (--force regrows; `select` re-picks items without growing)")
    backend.configure(a.backend)
    ea = S1.eval_settings(F / "map", a)
    steps, gap, t0 = window(ea)
    times = [t0 + k * gap // 8 for k in range(NFILM)]
    W, pairs = source_pool(F, a, np.random.default_rng([a.seed, 711]))
    (out / "frames").mkdir(parents=True, exist_ok=True)
    ids = sorted(W)
    bad, rep = 0, []
    for wid, film in zip(ids, grow([Rule.from_dict(W[w]["rule"]) for w in ids], ea, times, steps)):
        snaps = film[[KEEP[i] for i in SNAP]]
        pic = ME.picture(snaps)
        scale = max(float(np.percentile(snaps, 99.5)), 1e-6)       # picture()'s brightness, for every stored frame
        col = np.stack([render.colorize(np.clip(film[k] / scale, 0, 1)) for k in KEEP])
        bad += not np.array_equal(fmt_grid(col), pic)
        np.savez_compressed(out / "frames" / f"{fid(wid)}.npz", f=col)
        W[wid]["change"] = round(change(snaps), 5)
        st = W[wid]["still"]
        try:
            ref = np.asarray(Image.open(F / st).convert("RGB"), np.int16) if st and (F / st).exists() else None
        except OSError:                                    # e.g. a Drive file not downloaded on this machine
            ref = None
        if ref is not None and ref.shape == pic.shape:
            rep.append(float(np.abs(ref - pic.astype(np.int16)).mean()))
    if bad:
        sys.exit(f"{bad} worlds: stored frames don't reproduce lenia.mace.picture; not writing pool.json")
    meta = {"settings": {k: getattr(ea, k) for k in ("size", "R", "warmup", "render", "gap", "start_seed")},
            "source": str(F), "replay": {"n": len(rep), "diverged": sum(r > S1.REPLAY_TOL for r in rep)}}
    (out / "pool.json").write_text(json.dumps({"meta": meta, "worlds": W, "pairs": pairs}))
    print(f"pool.json: {len(W)} worlds; replay vs stored stills {meta['replay']}", file=sys.stderr)
    select(a)
    films(a)


# ---------- select: the items ----------

def select(a) -> None:
    out = Path(a.out)
    P = json.loads((out / "pool.json").read_text())
    W = P["worlds"]
    if any((out / "results").rglob("*.json")) and not a.force:
        sys.exit("results exist: re-picking items now would mix designs (--force if you mean it)")
    rng = np.random.default_rng([a.seed, 712])
    ids = sorted(W)
    split = {w: ("test" if rng.random() < a.test_share else "dev") for w in ids}
    ch = {w: W[w]["change"] for w in ids}
    med = float(np.median(list(ch.values())))
    items, use, short = [], Counter(), []

    def bin_of(x, y):
        hi, lo = max(ch[x], ch[y]), min(ch[x], ch[y])
        r = (hi + 1e-3) / (lo + 1e-3)
        return ("easy" if r >= a.easy else "close" if a.close_lo <= r < a.close_hi else None), r

    for sp in ("dev", "test"):
        n_easy, n_close, n_frozen = (a.n_dev if sp == "dev" else a.n_test)
        S = [w for w in ids if split[w] == sp]
        shuf = lambda c: [c[int(j)] for j in rng.permutation(len(c))]
        # frozen first: a world that visibly changes vs itself frozen
        fz = shuf([w for w in S if ch[w] >= med])[:n_frozen]
        for w in fz:
            items.append({"split": sp, "kind": "frozen", "source": W[w]["src"], "a": w, "b": f"frozen:{w}",
                          "ratio": None, "ca": ch[w], "cb": 0.0})
            use[w] += 1
        if len(fz) < n_frozen:
            short.append(f"{sp}/frozen {len(fz)} of {n_frozen}")
        inside = lambda c: [p for p in c if split[p[0]] == sp and split[p[1]] == sp]
        rand = [(S[int(i)], S[int(j)], "random") for i, j in rng.integers(0, len(S), (a.n_random, 2)) if i != j]
        el = [w for w in S if W[w]["src"] == "elite"]
        for kind, n, first in (("close", n_close, inside([tuple(p) for p in P["pairs"]])),
                               ("easy", n_easy, list(combinations(el, 2)))):
            got = 0
            for p in shuf(first) + shuf(rand):
                if got >= n:
                    break
                x, y = p[0], p[1]
                src = p[2] if len(p) > 2 else "elite"
                b, r = bin_of(x, y)
                if b != kind or use[x] >= 2 or use[y] >= 2:
                    continue
                w, l = (x, y) if ch[x] > ch[y] else (y, x)
                items.append({"split": sp, "kind": kind, "source": src, "a": w, "b": l, "ratio": round(r, 3),
                              "ca": ch[w], "cb": ch[l]})
                use[x] += 1
                use[y] += 1
                got += 1
            if got < n:
                short.append(f"{sp}/{kind} {got} of {n}")
    for i, t in enumerate(items):
        t["id"] = i
    (out / "items.json").write_text(json.dumps(items, indent=1))
    q = np.quantile(list(ch.values()), [0.1, 0.25, 0.5, 0.75, 0.9])
    L = [f"# Motion benchmark, Stage A: items ({time.strftime('%Y-%m-%d')})\n",
         f"Source {P['meta']['source']}; {len(W)} worlds; settings {P['meta']['settings']}; replay {P['meta']['replay']}.",
         f"Change over the pool (10/25/50/75/90%): {' / '.join(f'{x:.3f}' for x in q)}.\n",
         "| split | kind | n | sources |", "|---|---|---|---|"]
    g = defaultdict(list)
    for t in items:
        g[(t["split"], t["kind"])].append(t)
    L += [f"| {s} | {k} | {len(v)} | {dict(Counter(t['source'] for t in v))} |" for (s, k), v in sorted(g.items())]
    L.append("\n**Shortfalls:** " + ("; ".join(short) if short else "none"))
    (out / "pool.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- pictures ----------

def _tile(f: np.ndarray, px: int = ME.TILE) -> np.ndarray:
    return np.asarray(Image.fromarray(f).resize((px, px), Image.BILINEAR))


def fmt_grid(col: np.ndarray) -> np.ndarray:
    """The pipeline's 2x2 picture from the stored frames (byte-identical to lenia.mace.picture; build checks)."""
    t = [_tile(col[i]) for i in SNAP]
    g = np.full((ME.TILE, 4, 3), GAPC, np.uint8)
    rows = [np.concatenate([t[0], g, t[1]], 1), np.concatenate([t[2], g, t[3]], 1)]
    return np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), GAPC, np.uint8), rows[1]], 0)


def fmt_line(col: np.ndarray) -> np.ndarray:
    t = [_tile(col[i]) for i in SNAP]
    g = np.full((ME.TILE, 4, 3), GAPC, np.uint8)
    return np.concatenate([t[0], g, t[1], g, t[2], g, t[3]], 1)


def fmt_ghosts(col: np.ndarray) -> np.ndarray:
    px = 2 * ME.TILE + 4
    acc = np.zeros((px, px, 3), np.float32)
    for k, w in zip(GHOST, np.linspace(0.25, 1.0, len(GHOST))):
        acc = np.maximum(acc, w * _tile(col[k], px).astype(np.float32))
    return acc.round().astype(np.uint8)


def fmt_labels(col: np.ndarray, gap_steps: int, size: int) -> np.ndarray:
    line = fmt_line(col).copy()
    step = 40 * ME.TILE / size                              # a grid line every 40 cells
    for j in range(4):
        x0 = j * (ME.TILE + 4)
        for v in np.arange(step, ME.TILE - 1, step):
            v = int(round(v))
            line[:, x0 + v - 1:x0 + v + 1] = 110
            line[v - 1:v + 1, x0:x0 + ME.TILE] = 110
    bar = Image.new("RGB", (line.shape[1], 30), (GAPC,) * 3)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:                                       # Pillow < 10.1
        font = ImageFont.load_default()
    d = ImageDraw.Draw(bar)
    for j in range(4):
        d.text((j * (ME.TILE + 4) + 8, 4), f"t = {j * gap_steps}", fill=(235, 235, 235), font=font)
    return np.concatenate([np.asarray(bar), line], 0)


def fmt_strip(col: np.ndarray, px: int = 200) -> np.ndarray:
    """8 frames, two rows of four (Gemini's film strip, as fullrate.strip)."""
    o = np.full((2 * px + 4, 4 * px + 12, 3), GAPC, np.uint8)
    for j, k in enumerate(int(round(x)) for x in np.linspace(0, len(col) - 1, 8)):
        o[(j // 4) * (px + 4):(j // 4) * (px + 4) + px, (j % 4) * (px + 4):(j % 4) * (px + 4) + px] = _tile(col[k], px)
    return o


class Pictures:
    def __init__(self, out: Path, meta: dict):
        self.out, self.s, self._f = out, meta["settings"], {}

    def get(self, wid: str, fmt: str) -> np.ndarray:
        base = wid.split("frozen:")[-1]
        if base not in self._f:
            self._f[base] = np.load(self.out / "frames" / f"{fid(base)}.npz")["f"]
        f = self._f[base]
        if wid.startswith("frozen:"):
            f = np.repeat(f[-1:], len(f), 0)
        if fmt in ("R0", "grid"):
            return fmt_grid(f)
        if fmt == "labels":
            return fmt_labels(f, min(self.s["gap"], (self.s["warmup"] + self.s["render"]) // 3), self.s["size"])
        return {"line": fmt_line, "ghosts": fmt_ghosts, "strip": fmt_strip}[fmt](f)


# ---------- the truth check: does the change measure match what Felix sees? ----------

def films(a) -> None:
    """Films for 10 dev change pairs (5 close, 5 easy), regrown every gap/20 steps; sides random."""
    out = Path(a.out)
    P = json.loads((out / "pool.json").read_text())
    items = json.loads((out / "items.json").read_text())
    rng = np.random.default_rng([a.seed, 713])
    pick = []
    for kind in ("close", "easy"):
        c = [t for t in items if t["split"] == "dev" and t["kind"] == kind]
        pick += [c[int(j)] for j in rng.permutation(len(c))[:5]]
    ws = sorted({w for t in pick for w in (t["a"], t["b"])})
    if not ws:
        return
    backend.configure(getattr(a, "backend", "numpy"))
    ea = argparse.Namespace(**P["meta"]["settings"], sim_batch=getattr(a, "sim_batch", None) or 64)
    steps, gap, t0 = window(ea)
    times = list(range(t0, steps + 1, max(1, gap // 20)))
    C = out / "check"
    (C / "images").mkdir(parents=True, exist_ok=True)
    names = {}
    for w, film in zip(ws, grow([Rule.from_dict(P["worlds"][w]["rule"]) for w in ws], ea, times, steps)):
        names[w] = hashlib.sha1(f"{w}|{a.seed}".encode()).hexdigest()[:12]
        n = len(times)
        S1._gif(list(film), film[[0, n // 3, 2 * n // 3, n - 1]], C / "images" / f"{names[w]}.gif", 280, 15)
    tasks = []
    for k in rng.permutation(len(pick)):
        t = pick[int(k)]
        l, r = (t["a"], t["b"]) if rng.random() < 0.5 else (t["b"], t["a"])
        tasks.append({"id": len(tasks), "item": t["id"], "kind": t["kind"], "left": names[l], "right": names[r],
                      "left_world": l, "right_world": r, "a": t["a"]})
    (C / "tasks.json").write_text(json.dumps(tasks, indent=1))
    print(f"check: {len(tasks)} pairs in {C}", file=sys.stderr)


CHECK_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Truth check</title>
<style>body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f2;color:#222}
header{padding:10px 20px;display:flex;justify-content:space-between} h1{font-size:18px;margin:0} #prog{color:#666}
.pair{display:flex;gap:16px;justify-content:center;padding:0 20px} img{width:min(45vw,470px);border-radius:6px;background:#000}
.btns{text-align:center;padding:12px} button{font-size:16px;padding:9px 18px;margin:0 6px;border-radius:6px;border:1px solid #bbb;
background:#fff;cursor:pointer} button:hover{background:#333;color:#fff} .hint{text-align:center;color:#777;font-size:13px}</style>
</head><body><header><h1>Which world changes more over time?</h1><span id="prog"></span></header>
<div class="pair"><img id="L"><img id="R"></div>
<div class="btns"><button data-a="left">&larr; left</button><button data-a="neither">can't tell (&darr;)</button>
<button data-a="right">right &rarr;</button></div>
<div class="hint">Films loop over the snapshot window. Only how much each changes, not which is nicer.</div>
<script>let todo=[],cur=null,busy=false;
async function load(){todo=await (await fetch('/todo')).json();show();}
function show(){document.getElementById('prog').textContent=todo.length+' left';
 if(!todo.length){document.body.innerHTML='<p style="padding:20px">All done. You can close this tab.</p>';return;}
 cur=todo[0];document.getElementById('L').src='/img/'+cur.left+'.gif';document.getElementById('R').src='/img/'+cur.right+'.gif';}
async function vote(x){if(busy||!cur)return;busy=true;await fetch('/answer',{method:'POST',body:JSON.stringify({id:cur.id,answer:x})});
 todo.shift();busy=false;show();}
document.querySelectorAll('button').forEach(b=>b.onclick=()=>vote(b.dataset.a));
document.addEventListener('keydown',e=>{const k={ArrowLeft:'left',ArrowRight:'right',ArrowDown:'neither'}[e.key];if(k){e.preventDefault();vote(k);}});
load();</script></body></html>"""


def check(a) -> None:
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    C = Path(a.out) / "check"
    tasks = [{k: t[k] for k in ("id", "left", "right")} for t in json.loads((C / "tasks.json").read_text())]
    lock = threading.Lock()
    done = lambda: {int(r["id"]) for r in csv.DictReader(open(C / "ratings.csv"))} if (C / "ratings.csv").exists() else set()

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
                return self._send(200, CHECK_PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/todo":
                d = done()
                return self._send(200, json.dumps([t for t in tasks if t["id"] not in d]).encode(), "application/json")
            f = C / "images" / Path(self.path).name
            if self.path.startswith("/img/") and f.suffix == ".gif" and f.exists():
                return self._send(200, f.read_bytes(), "image/gif")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            x = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                if int(x["id"]) not in done():
                    new = not (C / "ratings.csv").exists()
                    with open(C / "ratings.csv", "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["time", "id", "answer"])
                        if new:
                            w.writeheader()
                        w.writerow({"time": round(time.time(), 1), "id": x["id"], "answer": x["answer"]})
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"truth check: {url} ({len(tasks) - len(done())} of {len(tasks)} left); then `check-report`. Ctrl-C stops.")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def check_report(a) -> None:
    C = Path(a.out) / "check"
    tasks = {t["id"]: t for t in json.loads((C / "tasks.json").read_text())}
    by = defaultdict(Counter)
    for r in csv.DictReader(open(C / "ratings.csv")):
        t = tasks[int(r["id"])]
        x = r["answer"]
        by[t["kind"]]["can't tell" if x == "neither" else
                      "agree" if (t["left_world"] if x == "left" else t["right_world"]) == t["a"] else "disagree"] += 1
    ct = "can't tell"
    L = ["# Truth check: Felix vs the change measure (blind, films)\n", "| pairs | agree | disagree | can't tell |",
         "|---|---|---|---|"] + [f"| {k} | {v['agree']} | {v['disagree']} | {v[ct]} |" for k, v in sorted(by.items())]
    L.append("\nThe measure stands if Felix agrees on most decided pairs; more than one disagreement on easy pairs means "
             "the measure doesn't capture what he sees, and the benchmark needs another truth before judging.")
    (C / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- judging and the report ----------

def _pics(out, items):
    P = json.loads((out / "pool.json").read_text())
    return Pictures(out, P["meta"]), sorted({w for t in items for w in (t["a"], t["b"])})


def judge(a) -> None:
    out = Path(a.out)
    items = [t for t in json.loads((out / "items.json").read_text()) if t["split"] == a.split]
    names = CONFIGS if a.configs == "all" else a.configs.split(",")
    if set(names) - set(CONFIGS):
        sys.exit(f"configs: choose from {CONFIGS}")
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    J = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge], "max_run_usd": a.max_usd},
              out / f"judge_cache_{a.judge}.sqlite", yes=True)
    pics, ws = _pics(out, items)
    R = out / "results" / a.split
    R.mkdir(parents=True, exist_ok=True)
    for name in names:
        P = {w: pics.get(w, name) for w in ws}
        pid = PR.bench_prompt(name)
        vs = J.judge_many([q for t in items for q in (Query(pid, P[t["a"]], P[t["b"]]), Query(pid, P[t["b"]], P[t["a"]]))],
                          desc=name)
        res = {t["id"]: {"p1": x.p_first, "p2": y.p_first} for t, x, y in zip(items, vs[::2], vs[1::2]) if x.ok and y.ok}
        (R / f"{name}.json").write_text(json.dumps({"judge": J.model, "results": res}, indent=1))
    print(f"spent ${J.cache.total_cost():.3f} in total (this bench's {a.judge} cache)", file=sys.stderr)
    report(a)


def gemini(a) -> None:
    """Only if no picture works: Gemini on an 8-frame film strip, same pairs and question. Input limit or model limit?"""
    from lenia.describe import Ask, Describer
    out = Path(a.out)
    items = [t for t in json.loads((out / "items.json").read_text()) if t["split"] == a.split]
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    d = Describer(a.vlm, cfg["vlms"][a.vlm], out / "describe_cache.sqlite", max_usd=a.max_vlm_usd)
    pics, ws = _pics(out, items)
    P = {w: pics.get(w, "strip") for w in ws}
    fill = {"layout": PR.PAIR_LAYOUT["strip"], "context": "", "question": PR.BENCH_QUESTION}
    ans = d.ask_many([Ask(PR.PAIR_JUDGE, [P[x], P[y]], fill) for t in items for x, y in ((t["a"], t["b"]), (t["b"], t["a"]))],
                     desc="gemini strip")

    def p_first(r):
        if not (r.ok and r.data):
            return None
        x = str(r.data.get("choice")).strip()
        if d.backend == "fake":                             # dry runs: the fake VLM answers placeholders
            x = "1" if int(hashlib.sha1(x.encode()).hexdigest(), 16) % 2 else "2"
        return {"1": 1.0, "2": 0.0}.get(x)

    res = {}
    for t, x, y in zip(items, ans[::2], ans[1::2]):
        p1, p2 = p_first(x), p_first(y)
        if p1 is not None and p2 is not None:
            res[t["id"]] = {"p1": p1, "p2": p2, "why": [x.data.get("why"), y.data.get("why")]}
    R = out / "results" / a.split
    R.mkdir(parents=True, exist_ok=True)
    (R / "gemini-strip.json").write_text(json.dumps({"judge": f"{a.vlm}:{d.model}", "results": res}, indent=1))
    print(f"gemini-strip: {len(res)}/{len(items)}; ${d.cache.total_cost():.3f}", file=sys.stderr)
    report(a)


def wilson(k: float, n: int) -> tuple[float, float]:
    p, z = k / n, 1.96
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def report(a) -> None:
    out = Path(a.out)
    items = {t["id"]: t for t in json.loads((out / "items.json").read_text()) if t["split"] == a.split}
    R = out / "results" / a.split
    res = {f.stem: json.loads(f.read_text()) for f in sorted(R.glob("*.json"))} if R.exists() else {}
    if not res:
        sys.exit(f"no results in {R}")
    order = [n for n in CONFIGS if n in res] + sorted(n for n in res if n not in CONFIGS)
    # P(right world) from both orders: order 1 shows the right world first, order 2 second
    pa = lambda r: 0.5 * (r["p1"] + 1 - r["p2"])
    right = lambda r: 1.0 if pa(r) > 0.5 else 0.5 if pa(r) == 0.5 else 0.0
    kinds = ("all", "easy", "close", "frozen")
    sel = lambda k: [i for i, t in items.items() if k == "all" or t["kind"] == k]
    L = [f"# Motion benchmark, Stage A: {a.split} ({len(items)} items: "
         + ", ".join(f"{k} {len(sel(k))}" for k in kinds[1:]) + ")\n",
         "Accuracy = the both-orders P(the world that changes more) > 0.5, ties half; chance 0.50. Orders agree = both "
         "orders pick the same world. P(first) = the mean probability given to whichever world is shown first.\n",
         "| config | " + " | ".join(kinds) + " | all, 95% CI | orders agree | P(first) |", "|---|---|---|---|---|---|---|---|"]
    for n in order:
        r = res[n]["results"]
        acc = [np.mean([right(r[str(i)]) for i in sel(k) if str(i) in r] or [np.nan]) for k in kinds]
        v = [right(r[str(i)]) for i in sel("all") if str(i) in r]
        lo, hi = wilson(sum(v), len(v)) if v else (np.nan, np.nan)
        rr = [r[str(i)] for i in sel("all") if str(i) in r]
        agree = np.mean([(x["p1"] > 0.5) == (x["p2"] < 0.5) for x in rr]) if rr else np.nan
        pf = np.mean([p for x in rr for p in (x["p1"], x["p2"])]) if rr else np.nan
        L.append(f"| {n} | " + " | ".join(f"{x:.2f}" for x in acc) + f" | {lo:.2f}–{hi:.2f} | {agree:.2f} | {pf:.2f} |")
    if "R0" in res:
        L += ["\nPaired against R0 (same items): right where R0 was wrong / wrong where R0 was right, two-sided sign test.\n",
              "| config | better | worse | p |", "|---|---|---|---|"]
        r0 = res["R0"]["results"]
        for n in order[1:]:
            r = res[n]["results"]
            both = [str(i) for i in sel("all") if str(i) in r and str(i) in r0]
            b = sum(right(r[i]) == 1 and right(r0[i]) == 0 for i in both)
            c = sum(right(r[i]) == 0 and right(r0[i]) == 1 for i in both)
            p = min(1.0, 2 * sum(math.comb(b + c, j) for j in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
            L.append(f"| {n} | {b} | {c} | {p:.3f} |")
    (out / f"report-{a.split}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"build": build, "select": select, "films": films, "check": check, "check-report": check_report,
            "judge": judge, "gemini": gemini, "report": report}
    for c in cmds:
        sub.add_parser(c).add_argument("--out", required=True, help="the benchmark folder")
    three = lambda s: [int(x) for x in s.split(",")]
    for c in ("build", "select", "films"):
        p = sub.choices[c]
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--force", action="store_true")
    b = sub.choices["build"]
    b.add_argument("--source", required=True, help="the full run's folder (RUNS/full-s1)")
    b.add_argument("--duels", type=int, default=150, help="map entry duels sampled")
    b.add_argument("--siblings", type=int, default=100, help="expedition sibling pairs sampled")
    b.add_argument("--max-worlds", type=int, default=0, help="dry runs only")
    for c in ("build", "films"):
        p = sub.choices[c]
        p.add_argument("--backend", default="numpy")
        for k in ("size", "warmup", "render", "gap", "sim_batch"):
            p.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (dry runs only)")
    for c in ("build", "select"):
        p = sub.choices[c]
        p.add_argument("--test-share", type=float, default=0.4)
        p.add_argument("--easy", type=float, default=2.0, help="change ratio for easy pairs")
        p.add_argument("--close-lo", type=float, default=1.2)
        p.add_argument("--close-hi", type=float, default=1.5)
        p.add_argument("--n-dev", type=three, default=[40, 40, 25], help="easy,close,frozen")
        p.add_argument("--n-test", type=three, default=[25, 25, 15], help="easy,close,frozen")
        p.add_argument("--n-random", type=int, default=8000, help="random pairs per split (fills shortfalls)")
    sub.choices["check"].add_argument("--port", type=int, default=8778)
    for c in ("judge", "gemini", "report"):
        sub.choices[c].add_argument("--split", default="dev", choices=["dev", "test"])
    for c in ("judge", "gemini"):
        sub.choices[c].add_argument("--config", default=None)
    j = sub.choices["judge"]
    j.add_argument("--configs", default="all", help=f"comma list from {','.join(CONFIGS)}, or all")
    j.add_argument("--judge", default="jev")
    j.add_argument("--max-usd", type=float, default=0.4, help="hard cap on this bench's judge cache in total")
    g = sub.choices["gemini"]
    g.add_argument("--vlm", default="gemini-pro")
    g.add_argument("--max-vlm-usd", type=float, default=3.0)
    a = ap.parse_args()
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
