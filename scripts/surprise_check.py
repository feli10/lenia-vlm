"""Can we detect "something new" cheaply? First test of the cheap-perception / rare-imagination loop.

The plan (project doc "Surprise detector check"): Jev does the frequent work; a full VLM is called only when a
world shows something the current goal library cannot describe, to name it and propose directions. That needs
a detector for such moments. This script tests two cheap signals and their combination, before anything is
built on them:

- jev:  Jev's "None of these fits." probability, per library group (interior, shape, edge, surroundings, bodies,
        change; 2 orders each), z-scored against the same groups on the reference worlds and averaged over groups.
        (Per-group calibration matters: on the goal-matching test worlds Jev said "none" for bodies 29/40 times.)
- dino: mean cosine distance from the world's DINOv2 embedding (mean over its four snapshots) to its 3 nearest
        reference worlds, z-scored against the reference worlds' own leave-one-out distances.
- combined: mean of the two z-scores.

Reference ("known") worlds: the 138 mace-qd2 check images the goal library was written from (all of
runs/mace-qd2/check/images except the 40 goal-matching test worlds). Pool ("candidates"): those 40 test worlds
plus every image of the goal-session run (elites and accepted branch children), de-duplicated by content.

Felix's blind check: 30 pool worlds in random order, without scores: the 10 most surprising by the combined
signal, the 5 most surprising by Jev alone and 5 by DINOv2 alone (not already chosen), and 10 at random. For each:
"something new here?" (new / not new / unsure) and an optional note on what is new. `report` gives the share
judged new per selection; the detector is worth building on only if its picks clearly beat the random ones.

  python scripts/surprise_check.py score --runs RUNS --out RUNS/surprise    # Colab (Jev + DINOv2)
  uv run python scripts/surprise_check.py rate   --out runs/surprise                       # Mac, ~10 min
  uv run python scripts/surprise_check.py report --out runs/surprise
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

import goal_groups as GG
from lenia.config import REPO_ROOT, load
from lenia.judge import ChoiceQuery, Judge, load_env

GROUPS = ["interior", "shape", "edge", "surroundings", "bodies", "change"]
NONE = GG.NONE
N_TOP, N_SINGLE, N_RANDOM, K_NN = 10, 5, 10, 3


def _hash(p: Path) -> str:
    return hashlib.sha256(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest()


def collect(runs: Path, seed: int) -> tuple[list, list]:
    """Reference and pool image paths."""
    qd = runs / "mace-qd2" / "check" / "images"
    finals = sorted(p.name for p in qd.glob("*_final.png"))
    rng = np.random.default_rng(seed)                                    # the goal-matching test worlds (seed 0)
    test = {finals[i] for i in rng.choice(len(finals), min(GG.N_WORLDS, len(finals)), replace=False)}
    ref = [p for p in sorted(qd.glob("*.png")) if p.name not in test]
    seen = {_hash(p) for p in ref}
    pool = []
    for p in [qd / n for n in sorted(test)] + sorted((runs / "goal-session" / "img").glob("*.png")):
        h = _hash(p)
        if h not in seen:
            seen.add(h)
            pool.append(p)
    return ref, pool


def jev_none(judge, lib: dict, groups: dict, ctrl: list, paths: list, rng) -> np.ndarray:
    """(n_images, n_groups) mean P("none") over 2 orders."""
    qs, meta = [], []
    for i, p in enumerate(paths):
        img = np.asarray(Image.open(p).convert("RGB"))
        for g, k in enumerate(GROUPS):
            for _ in range(2):
                m = groups[k] + [NONE, str(rng.choice(ctrl))]
                m = [m[j] for j in rng.permutation(len(m))]
                opts = {x: GG.NONE_TEXT if x == NONE else lib[x]["sees"][0].upper() + lib[x]["sees"][1:] + "." for x in m}
                qs.append(ChoiceQuery(img, GG.INSTRUCTIONS.replace("{ABOUT}", GG.GROUPS[k]), opts))
                meta.append((i, g))
    out = np.zeros((len(paths), len(GROUPS)))
    cnt = np.zeros_like(out)
    for (i, g), pr in zip(meta, judge.choose_many(qs, desc="none")):
        if pr:
            out[i, g] += pr.get(NONE, 0.0)
            cnt[i, g] += 1
    return out / np.maximum(cnt, 1)


def embed(paths: list, model: str) -> np.ndarray:
    """Mean of the four snapshot embeddings, unit length."""
    import mace_qd as QD
    emb = QD.embedder(model)
    tiles = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        tw, th = (w - 4) // 2, (h - 4) // 2                       # mace_evo.picture: 2x2 tiles with a 4 px gap
        tiles += [im.crop((x, y, x + tw, y + th)) for y in (0, h - th) for x in (0, w - tw)]
    E = np.asarray(emb(tiles), np.float64).reshape(len(paths), 4, -1).mean(1)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def knn_dist(A: np.ndarray, B: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    D = 1 - A @ B.T
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k].mean(1)


def z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (x - ref.mean(0)) / (ref.std(0) + 1e-9)


def score(a) -> None:
    runs, out = Path(a.runs), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ref, pool = collect(runs, a.seed)
    print(f"{len(ref)} reference worlds, {len(pool)} candidate worlds", file=sys.stderr)
    lib_list = json.loads((REPO_ROOT / a.library).read_text())
    first = {}
    for g in lib_list:
        first.setdefault(g["sees"], g)
    lib = {g["id"]: g for g in lib_list}
    groups = {k: [g["id"] for g in first.values() if g["group"] == k] for k in GROUPS}
    ctrl = [g["id"] for g in first.values() if g["group"] == "control"]
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge]}, out / "judge_cache.sqlite", yes=True)
    rng = np.random.default_rng(a.seed + 1)
    n_ref = jev_none(judge, lib, groups, ctrl, ref, rng)
    n_pool = jev_none(judge, lib, groups, ctrl, pool, rng)
    print(f"jev spent ${judge.spent:.4f}", file=sys.stderr)
    jz = z(n_pool, n_ref).mean(1)
    jz_ref = z(n_ref, n_ref).mean(1)
    E_ref, E_pool = embed(ref, a.embed_model), embed(pool, a.embed_model)
    d_ref = knn_dist(E_ref, E_ref, K_NN, exclude_self=True)
    d_pool = knn_dist(E_pool, E_ref, K_NN)
    dz = (d_pool - d_ref.mean()) / (d_ref.std() + 1e-9)
    comb = (jz + dz) / 2
    rows = [{"name": p.name, "src": str(p), "jev_z": round(float(j), 3), "dino_z": round(float(d), 3),
             "combined": round(float(c), 3), "p_none": {k: round(float(v), 3) for k, v in zip(GROUPS, n)}}
            for p, j, d, c, n in zip(pool, jz, dz, comb, n_pool)]
    # selections for the blind check
    order = lambda key: [r["name"] for r in sorted(rows, key=lambda r: -r[key])]
    chosen = {}
    for nm in order("combined")[:N_TOP]:
        chosen[nm] = "top_combined"
    for key, lab in (("jev_z", "top_jev"), ("dino_z", "top_dino")):
        for nm in [n for n in order(key) if n not in chosen][:N_SINGLE]:
            chosen[nm] = lab
    rest = [r["name"] for r in rows if r["name"] not in chosen]
    for nm in np.random.default_rng(a.seed + 2).choice(rest, min(N_RANDOM, len(rest)), replace=False):
        chosen[str(nm)] = "random"
    tasks = list(chosen.items())
    tasks = [tasks[i] for i in np.random.default_rng(a.seed + 3).permutation(len(tasks))]
    by = {r["name"]: r for r in rows}
    for k, (nm, sel) in enumerate(tasks):
        shutil.copy(by[nm]["src"], out / "images" / f"w{k:02d}.png")
    (out / "tasks.json").write_text(json.dumps([{"id": k, "image": f"w{k:02d}.png", "name": nm, "selection": sel}
                                                for k, (nm, sel) in enumerate(tasks)], indent=1))
    (out / "surprise.json").write_text(json.dumps({"reference": [p.name for p in ref], "pool": rows,
                                                   "ref_jev_z": jz_ref.round(3).tolist(), "ref_dino": d_ref.round(4).tolist()},
                                                  indent=1))
    rs = np.corrcoef(jz, dz)[0, 1]
    print(f"wrote {out / 'tasks.json'} ({len(tasks)} worlds to rate); correlation of the two signals over the pool: {rs:.2f}")


PAGE = """<!doctype html><meta charset=utf-8><title>something new?</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:16px}img{width:620px;border:1px solid #333}
button{padding:8px 22px;font:15px system-ui;margin:8px 8px 0 0}#n{color:#999}input{width:600px;padding:6px;
font:14px system-ui;background:#222;color:#eee;border:1px solid #444}</style>
<div id=n></div><p>Does this world show <b>something new</b>: a feature or behaviour you have not seen in our MaCE worlds
before? (four snapshots, reading order)</p><img id=im><br>
<input id=note placeholder="optional: what is new?"><br>
<button data-a=new>new</button><button data-a=not_new>not new</button><button data-a=unsure>unsure</button>
<script>
let todo=[];async function load(){todo=await (await fetch('/todo')).json();show()}
function show(){if(!todo.length){document.body.innerHTML='<h2>Done, thank you. Close this tab and press Ctrl-C.</h2>';return}
document.getElementById('n').textContent=todo.length+' left';document.getElementById('im').src='/img/'+todo[0].image;
document.getElementById('note').value=''}
document.querySelectorAll('button').forEach(b=>b.onclick=async()=>{const t=todo[0];
await fetch('/answer',{method:'POST',body:JSON.stringify({id:t.id,answer:b.dataset.a,note:document.getElementById('note').value})});
todo.shift();show()});
load()</script>"""


def _answers(out: Path) -> dict:
    p = out / "ratings.csv"
    return {int(r["id"]): r for r in csv.DictReader(open(p))} if p.exists() else {}


def rate(a) -> None:
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
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/todo":
                done = _answers(out)
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


def report(a) -> None:
    out = Path(a.out)
    tasks = json.loads((out / "tasks.json").read_text())
    S = json.loads((out / "surprise.json").read_text())
    by = {r["name"]: r for r in S["pool"]}
    ans = _answers(out)
    L = ["# Surprise detector check\n",
         f"{len(S['reference'])} reference worlds (the ones the goal library was written from), {len(S['pool'])} candidates "
         "(goal-matching test worlds + goal-session images). Felix rated 30 candidates blind: \"something new here?\"\n",
         "| selection | rated | new | not new | unsure | share new (of new + not new) |", "|---|---|---|---|---|---|"]
    for sel in ("top_combined", "top_jev", "top_dino", "random"):
        r = [ans[t["id"]]["answer"] for t in tasks if t["selection"] == sel and t["id"] in ans]
        n, nn, u = r.count("new"), r.count("not_new"), r.count("unsure")
        L.append(f"| {sel} | {len(r)} | {n} | {nn} | {u} | {n / (n + nn):.2f} |" if n + nn else f"| {sel} | {len(r)} | {n} | {nn} | {u} | - |")
    rated = [(by[t["name"]], ans[t["id"]]["answer"]) for t in tasks if t["id"] in ans and ans[t["id"]]["answer"] != "unsure"]
    if len(rated) > 3:
        y = np.array([a_ == "new" for _, a_ in rated], float)
        L += ["", "Correlation of each signal with your 'new' answers over all rated worlds (point-biserial): " + ", ".join(
            f"{k} {np.corrcoef([r[k] for r, _ in rated], y)[0, 1]:.2f}" for k in ("jev_z", "dino_z", "combined")) + "."]
    notes = [(t, ans[t["id"]]) for t in tasks if t["id"] in ans and ans[t["id"]]["note"].strip()]
    if notes:
        L += ["", "Your notes:", ""] + [f"- {t['image']} ({t['selection']}, {r['answer']}): {r['note']}" for t, r in notes]
    L += ["", "Worlds (selection, combined / jev / dino z):", ""] + [
        f"- {t['image']}: {t['selection']}, {by[t['name']]['combined']:.2f} / {by[t['name']]['jev_z']:.2f} / "
        f"{by[t['name']]['dino_z']:.2f}, {t['name']}" + (f" -> {ans[t['id']]['answer']}" if t["id"] in ans else "") for t in tasks]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "rate", "report"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/surprise")
    ap.add_argument("--library", default="data/goal_library_v3.json")
    ap.add_argument("--judge", default="jev")
    ap.add_argument("--config", default=None)
    ap.add_argument("--embed-model", default="dinov2_vits14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8769)
    a = ap.parse_args()
    {"score": score, "rate": rate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
