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
"something unusual here, compared with the typical MaCE world?" (unusual / typical / unsure; stored as
new / not_new / unsure) and an optional note. The candidates come from runs Felix has seen, so the question is
about rarity, not about whether he remembers the world. `report` gives the share
judged new per selection; the detector is worth building on only if its picks clearly beat the random ones.

  python scripts/surprise_check.py score --runs RUNS --out RUNS/surprise    # Colab (Jev + DINOv2)
  uv run python scripts/surprise_check.py rate   --out runs/surprise                       # Mac, ~10 min
  uv run python scripts/surprise_check.py report --out runs/surprise
  uv run python scripts/surprise_check.py duel   --out runs/surprise --runs runs   # v2: direct pairwise question (Mac)
  uv run python scripts/surprise_check.py validate --out runs/surprise-val          # v3: held-out check of v2 (Mac)
  uv run python scripts/surprise_check.py rate     --out runs/surprise-val
  uv run python scripts/surprise_check.py val-report --out runs/surprise-val

v2 (`duel`), after the library-misfit / DINOv2 signals failed (they flagged worlds far from home, not
interesting ones): ask the judges directly, in the pairwise form that has worked best. Each of the 30 rated worlds
meets the same 6 random reference worlds, both orders, under two prompts: "surprise" (which world shows something
more unusual *and* interesting, a distinct feature or behaviour rather than more or less of the same?) and, as a
baseline, the generic complexity prompt of mace_evo. A world's score is its mean win probability. Measured
against Felix's ratings as AUC (unusual vs typical; 0.5 = chance) with a one-sided permutation p-value.

v3 (`validate`): the v2 prompt was written after v1 failed, on the same 30 worlds, so it is checked on fresh ones.
Pool: worlds from --pool-dirs (default: every elite picture of mace-qd2 and every lineage picture of the
mace-evo-jev-v3 hill climber), minus the 138 reference worlds and the 30 rated ones (by content), de-duplicated,
then --n-pool sampled at random, equally from each folder. Each pool world meets 6 random reference worlds
(both orders) under the v2 surprise prompt with Jev. Felix rates, blind and shuffled, the 10 top-scoring worlds
and 10 random others with the same question. Pass: the top 10 are judged unusual clearly more often.
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
from lenia.judge import ChoiceQuery, Judge, Query, load_env
from lenia.prompts import PROMPTS

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
         "(goal-matching test worlds + goal-session images). Felix rated 30 candidates blind: \"something unusual here?\" (new = unusual)\n",
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


# ---------- v2: direct pairwise surprise duels ----------

SURPRISE = "surprise_unusual_interesting"
PROMPTS.setdefault(SURPRISE, {
    "instructions": ("You are shown two worlds from a mass-conserving variant of Lenia, a continuous cellular automaton "
                     "used in artificial-life research. Each image shows one whole world as four snapshots taken some time "
                     "apart, in reading order (top-left, top-right, bottom-left, bottom-right); the colours are three "
                     "interacting chemical channels. The two images are {A} and {B}. Which world shows something more "
                     "unusual and interesting: a distinct feature or behaviour that stands out, rather than just more or "
                     "less of the same, a different size, or a different colour?"),
    "first": "The world in {A} shows something more unusual and interesting.",
    "second": "The world in {B} shows something more unusual and interesting."})
N_OPP = 6


def auc(scores: list, labels: list, n_perm: int = 20000, seed: int = 0) -> tuple[float, float]:
    """AUC of scores for label 1 vs 0 (ties count half) and a one-sided permutation p-value."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    def a_(yy):
        pos, neg = s[yy == 1], s[yy == 0]
        return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())
    obs = a_(y)
    rng = np.random.default_rng(seed)
    perm = np.array([a_(rng.permutation(y)) for _ in range(n_perm)])
    return obs, float((perm >= obs - 1e-12).mean())


def duel(a) -> None:
    out = Path(a.out)
    tasks = json.loads((out / "tasks.json").read_text())
    S = json.loads((out / "surprise.json").read_text())
    ans = _answers(out)
    ref_dir = Path(a.runs) / "mace-qd2" / "check" / "images"
    refs = [ref_dir / n for n in S["reference"]]
    missing = [r for r in refs if not r.exists()]
    if missing:
        sys.exit(f"{len(missing)} reference images missing under {ref_dir}")
    rng = np.random.default_rng(a.seed + 5)
    opp = {t["id"]: [refs[i] for i in rng.choice(len(refs), N_OPP, replace=False)] for t in tasks}
    img = lambda p: np.asarray(Image.open(p).convert("RGB"))
    cand = {t["id"]: img(out / "images" / t["image"]) for t in tasks}
    cache = {}
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    from mace_evo import PROMPT_ID as COMPLEXITY
    res = {}
    for jn in a.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, out / "duel_cache.sqlite", yes=True)
        for pname, pid in (("surprise", SURPRISE), ("complexity", COMPLEXITY)):
            qs, meta = [], []
            for t in tasks:
                for r in opp[t["id"]]:
                    if r not in cache:
                        cache[r] = img(r)
                    qs += [Query(pid, cand[t["id"]], cache[r]), Query(pid, cache[r], cand[t["id"]])]
                    meta.append(t["id"])
            vs = judge.judge_many(qs, desc=f"{jn} {pname}")
            sc = {}
            for n, tid in enumerate(meta):
                x, y = vs[2 * n], vs[2 * n + 1]
                if x.ok and y.ok:
                    sc.setdefault(tid, []).append(0.5 * (x.p_first + 1 - y.p_first))
            res[f"{jn}/{pname}"] = {str(k): float(np.mean(v)) for k, v in sc.items()}
        print(f"{jn}: spent ${judge.spent:.4f}", file=sys.stderr)
    (out / "duel.json").write_text(json.dumps({"opponents": {str(k): [p.name for p in v] for k, v in opp.items()},
                                               "scores": res}, indent=1))
    by = {r["name"]: r for r in S["pool"]}
    dec = [t for t in tasks if t["id"] in ans and ans[t["id"]]["answer"] in ("new", "not_new")]
    y = [int(ans[t["id"]]["answer"] == "new") for t in dec]
    L = ["# Surprise detector v2: direct pairwise duels\n",
         f"{len(dec)} decisive ratings by Felix ({sum(y)} unusual, {len(y) - sum(y)} typical). Each world met the same "
         f"{N_OPP} random reference worlds, both orders. AUC = probability an unusual world scores above a typical one "
         "(0.5 chance); p = one-sided permutation test.\n",
         "| signal | AUC | p | mean score, unusual | mean score, typical |", "|---|---|---|---|---|"]
    rows = [(k, {t["id"]: v[str(t["id"])] for t in dec if str(t["id"]) in v}) for k, v in res.items()]
    rows += [(f"v1 {k}", {t["id"]: by[t["name"]][k] for t in dec}) for k in ("combined", "jev_z", "dino_z")]
    for k, v in rows:
        ids = [t["id"] for t in dec if t["id"] in v]
        yy = [y[i] for i, t in enumerate(dec) if t["id"] in v]
        sc = [v[i] for i in ids]
        A, p = auc(sc, yy)
        mu = np.mean([s_ for s_, l in zip(sc, yy) if l]), np.mean([s_ for s_, l in zip(sc, yy) if not l])
        L.append(f"| {k} | {A:.2f} | {p:.3f} | {mu[0]:.2f} | {mu[1]:.2f} |")
    L += ["", "Per world (Felix's answer; surprise score per judge):", ""]
    for t in tasks:
        a_ = ans.get(t["id"], {}).get("answer", "-")
        L.append(f"- {t['image']} ({t['selection']}, {a_}): " + ", ".join(
            f"{k} {v.get(str(t['id']), float('nan')):.2f}" for k, v in res.items()))
    (out / "duel.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- v3: held-out validation of the v2 surprise duel ----------

def validate(a) -> None:
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    src = Path(a.out_v1)
    S = json.loads((src / "surprise.json").read_text())
    ref_dir = Path(a.runs) / "mace-qd2" / "check" / "images"
    refs = [ref_dir / n for n in S["reference"]]
    excl = {_hash(p) for p in refs} | {_hash(p) for p in sorted((src / "images").glob("*.png"))}
    rng = np.random.default_rng(a.seed + 11)
    dirs = [Path(d) for d in a.pool_dirs.split(",") if Path(d).is_dir()]
    if not dirs:
        sys.exit(f"none of {a.pool_dirs} exists")
    per_dir = {}
    for d in dirs:
        seen, keep = set(), []
        for p in sorted(d.glob("*.png")):
            h = _hash(p)
            if h not in excl and h not in seen:
                seen.add(h)
                keep.append(p)
        excl |= seen
        per_dir[d] = keep
        print(f"{d}: {len(keep)} fresh worlds", file=sys.stderr)
    n_each = a.n_pool // len(dirs)
    pool = []
    for d, keep in per_dir.items():
        pool += [keep[i] for i in sorted(rng.choice(len(keep), min(n_each, len(keep)), replace=False))]
    opp = [[refs[i] for i in rng.choice(len(refs), N_OPP, replace=False)] for _ in pool]
    img = lambda p: np.asarray(Image.open(p).convert("RGB"))
    cache = {}
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge]}, out / "judge_cache.sqlite", yes=True)
    qs, meta = [], []
    for i, p in enumerate(pool):
        c = img(p)
        for r in opp[i]:
            if r not in cache:
                cache[r] = img(r)
            qs += [Query(SURPRISE, c, cache[r]), Query(SURPRISE, cache[r], c)]
            meta.append(i)
    vs = judge.judge_many(qs, desc="surprise")
    print(f"{a.judge}: spent ${judge.spent:.4f}", file=sys.stderr)
    sc = {}
    for n, i in enumerate(meta):
        x, y = vs[2 * n], vs[2 * n + 1]
        if x.ok and y.ok:
            sc.setdefault(i, []).append(0.5 * (x.p_first + 1 - y.p_first))
    score_ = {i: float(np.mean(v)) for i, v in sc.items()}
    ranked = sorted(score_, key=lambda i: -score_[i])
    top = ranked[:N_TOP]
    rest = [i for i in score_ if i not in top]
    rnd = [int(i) for i in np.random.default_rng(a.seed + 12).choice(rest, min(N_RANDOM, len(rest)), replace=False)]
    chosen = [(i, "top_jev_surprise") for i in top] + [(i, "random") for i in rnd]
    chosen = [chosen[j] for j in np.random.default_rng(a.seed + 13).permutation(len(chosen))]
    tasks = []
    for k, (i, sel) in enumerate(chosen):
        shutil.copy(pool[i], out / "images" / f"w{k:02d}.png")
        tasks.append({"id": k, "image": f"w{k:02d}.png", "name": str(pool[i]), "selection": sel, "score": round(score_[i], 4)})
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
    (out / "scores.json").write_text(json.dumps({"pool": [str(p) for p in pool], "score": {str(pool[i]): v for i, v in score_.items()},
                                                 "opponents": {str(pool[i]): [r.name for r in opp[i]] for i in range(len(pool))}}, indent=1))
    print(f"{len(score_)} worlds scored; top-10 scores {[round(score_[i], 2) for i in top]}; "
          f"pool median {np.median(list(score_.values())):.2f}. Rate with: surprise_check.py rate --out {out}")


def fisher_greater(a1: int, b1: int, a2: int, b2: int) -> float:
    """One-sided Fisher exact test: P(top row has >= a1 successes) given the margins (no scipy needed)."""
    from math import comb
    n1, n2, k = a1 + b1, a2 + b2, a1 + a2
    tot = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a1, min(n1, k) + 1) if 0 <= k - x <= n2) / tot


def val_report(a) -> None:
    out = Path(a.out)
    tasks = json.loads((out / "tasks.json").read_text())
    ans = _answers(out)
    L = ["# Surprise detector v3: held-out validation of the Jev surprise duel\n",
         "Fresh worlds (not the 30 rated in v1/v2, not the 138 reference worlds). Felix rated, blind and shuffled, the 10 "
         "top-scoring worlds and 10 random others: \"something unusual here, compared with the typical MaCE world?\"\n",
         "| selection | rated | unusual | typical | unsure | share unusual (of decisive) |", "|---|---|---|---|---|---|"]
    cnt = {}
    for sel in ("top_jev_surprise", "random"):
        r = [ans[t["id"]]["answer"] for t in tasks if t["selection"] == sel and t["id"] in ans]
        n, nn, u = r.count("new"), r.count("not_new"), r.count("unsure")
        cnt[sel] = (n, nn)
        L.append(f"| {sel} | {len(r)} | {n} | {nn} | {u} | " + (f"{n / (n + nn):.2f} |" if n + nn else "- |"))
    if all(sum(v) for v in cnt.values()):
        p = fisher_greater(*cnt["top_jev_surprise"], *cnt["random"])
        L.append(f"\nOne-sided Fisher exact test, top vs random: p = {p:.3f}.")
    dec = [t for t in tasks if t["id"] in ans and ans[t["id"]]["answer"] in ("new", "not_new")]
    if len({ans[t["id"]]["answer"] for t in dec}) == 2:
        A, p = auc([t["score"] for t in dec], [int(ans[t["id"]]["answer"] == "new") for t in dec])
        L.append(f"AUC of the Jev surprise score over the {len(dec)} decisive ratings: {A:.2f} (permutation p = {p:.3f}).")
    L += ["", "Per world:", ""] + [f"- {t['image']} ({t['selection']}, score {t['score']:.2f}): "
                                    f"{ans.get(t['id'], {}).get('answer', '-')} {ans.get(t['id'], {}).get('note', '')} [{Path(t['name']).parent.parent.name}/{Path(t['name']).name}]"
                                    for t in tasks]
    (out / "val_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "rate", "report", "duel", "validate", "val-report"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None, help="default: runs/surprise-val for validate / val-report, else runs/surprise")
    ap.add_argument("--library", default="data/goal_library_v3.json")
    ap.add_argument("--judge", default="jev")
    ap.add_argument("--judges", default="jev,luna", help="duel: judges to ask")
    ap.add_argument("--out-v1", default="runs/surprise", help="validate: the v1/v2 folder (reference list, rated worlds)")
    ap.add_argument("--pool-dirs", default="runs/mace-qd2/img,runs/mace-evo-jev-v3/img")
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--config", default=None)
    ap.add_argument("--embed-model", default="dinov2_vits14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8769)
    a = ap.parse_args()
    if a.out is None:
        a.out = "runs/surprise-val" if a.cmd in ("validate", "val-report") else "runs/surprise"
    if a.cmd == "validate" and Path(a.out).resolve() == Path(a.out_v1).resolve():
        sys.exit("--out must differ from --out-v1 (validate would overwrite the v1 tasks and images)")
    {"score": score, "rate": rate, "report": report, "duel": duel, "validate": validate, "val-report": val_report}[a.cmd](a)


if __name__ == "__main__":
    main()
