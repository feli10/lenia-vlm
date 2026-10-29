"""Goal-steered branches vs. generic-prompt branches from the same MAP-Elites elites (agent sessions, pilot).

Step 1 (archive/scripts/goal_groups.py, project doc "Goal matching results") showed Jev can say what a world shows: Felix said yes
to 82% of its picks vs 46% of decoys. This pilot asks whether steering by a matched goal does anything.

Setup
- Elites: --elites distinct seed lineages ("roots") of a mace_qd.py run (default mace-qd2), round-robin over its
  islands; per root the elite that won most duels. Every world is grown as in mace_qd.py: the same smoothed-noise
  start, --warmup steps, then --render steps filmed as four snapshots (no world is inherited, so a child and its
  parent are always the same age).
- Matching (Jev): the world is shown with each steering group of library v3 as a menu (interior, shape,
  surroundings, bodies, change; + "none of these fits" + 1 control; 2 orders, probabilities averaged). Edge
  (Jev over-picks "cog") and leaps (not verifiable) are not used for steering.
- Per elite, --goal-branches branches each get a goal: an observation sampled with probability proportional to
  Jev's match probability (no two branches of an elite share an observation), with one of its directions.
  --free-branches more branches use the generic complexity prompt (lenia.mace).
- Each generation every branch makes --mutants mutants (lenia.mace.mutate), the grid-noise rule discards some,
  and each survivor duels the branch's current parent (Jev, both orders) under the branch's prompt; the best with
  P >= --threshold becomes the parent. Goal prompt: the observation, the direction, "which world would you choose
  as the parent, to move toward this goal?". Every --phase generations a goal branch is re-matched on its current
  parent and takes a new goal (not one it has had).

  python scripts/goal_session.py run   --out RUNS/goal-session --elites-from RUNS/mace-qd2 --backend auto   # Colab
  python scripts/goal_session.py check --out RUNS/goal-session              # held-out Luna (+ Jev), DINOv2 spread
  uv run python scripts/goal_session.py rate   --out runs/goal-session       # your blind ratings (Mac, ~8 min)
  uv run python scripts/goal_session.py report --out runs/goal-session       # report.md + report.html
  python scripts/goal_session.py run --out /tmp/gs --elites-from runs/mace-qd2 --judge fake --elites 2 \\
      --generations 2 --phase 1 --size 64 --warmup 60 --render 60 --gap 20   # offline dry run

check compares, with a held-out judge (Luna) in both orders:
- specificity (end of phase 1, gen --phase): goal branch X vs the elite's other goal branches and its free
  branches, asked which world shows X's change more clearly. Chance 0.5; steering works if X wins its own goal.
- progress: X at gen --phase vs the elite, on X's goal.
- phase 2 (the re-matched goal): X at the end vs X at the end of phase 1, and vs the free branches at the end.
- interest: every branch at the end vs the elite, and goal vs free branches at the end ("more interesting as
  artificial life").
- spread: mean pairwise DINOv2 distance between an elite's goal-branch finals vs between its free-branch finals.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lenia import mace as ME
import mace_qd as QD
from lenia import backend
from lenia.config import REPO_ROOT, load
from lenia.mace import mutate
from lenia.judge import ChoiceQuery, Judge, Query, load_env
from lenia.prompts import PROMPTS
from lenia.prompts import INTEREST, NONE, STEER, WORLD, attain_prompt, steer_prompt   # texts in prompts.py
from lenia.prompts import SESSION_MATCH as MATCH
from lenia.sim import Rule

FREE = ME.PROMPT_ID


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# ---------- setup ----------

def pick_elites(path: Path, n: int) -> tuple[list, argparse.Namespace]:
    st = json.loads((path / "state.json").read_text())
    qa = QD.settings(st)
    best = {}
    for k, m in enumerate(QD.islands_of(st)):
        for ck, e in m.items():
            if qa.max_texture > 0 and e.get("texture", 0) > qa.max_texture:
                continue
            score = (e.get("replaced", 0), e.get("p") or 0)
            if e["root"] not in best or score > best[e["root"]][0]:
                best[e["root"]] = (score, k, ck, e)
    by_island = {}
    for root, (score, k, ck, e) in sorted(best.items(), key=lambda x: (-x[1][0][0], -x[1][0][1], x[0])):
        by_island.setdefault(k, []).append({"root": root, "island": k, "cell": ck, "rule": e["rule"]})
    out, i = [], 0
    while len(out) < n and any(by_island.values()):
        k = sorted(by_island)[i % len(by_island)]
        if by_island[k]:
            out.append(by_island[k].pop(0))
        i += 1
    return out, qa


def eval_args(a, qa) -> argparse.Namespace:
    """mace_qd evaluation settings (size, R, start, warmup, render, gap); no form axis."""
    v = argparse.Namespace(**vars(qa))
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        if getattr(a, k, None):
            setattr(v, k, getattr(a, k))
    v.axes = []
    return v


def library(a) -> dict:
    lib = json.loads((REPO_ROOT / a.library).read_text())
    first = {}
    for g in lib:
        first.setdefault(g["sees"], g["id"])
    obs = {}                                     # observation id (first entry) -> all entries (directions)
    for g in lib:
        obs.setdefault(first[g["sees"]], []).append(g)
    groups = {k: [o for o, gs in obs.items() if gs[0]["group"] == k] for k in STEER}
    ctrl = [o for o, gs in obs.items() if gs[0]["group"] == "control"]
    return {"lib": {g["id"]: g for g in lib}, "obs": obs, "groups": groups, "ctrl": ctrl}


def match(judge, L: dict, pics: list, rng, desc: str) -> list:
    """Per picture: {observation id or "none:<group>": P} over the steering groups (each group's menu sums to 1;
    2 orders averaged)."""
    qs, meta = [], []
    for i, pic in enumerate(pics):
        for k, ids in L["groups"].items():
            for o in range(2):
                m = ids + [NONE, str(rng.choice(L["ctrl"]))]
                m = [m[j] for j in rng.permutation(len(m))]
                opts = {x: ("None of these fits." if x == NONE else _cap(L["obs"][x][0]["sees"]) + ".") for x in m}
                qs.append(ChoiceQuery(pic, MATCH.replace("{ABOUT}", STEER[k]), opts))
                meta.append((i, k))
    res = [dict() for _ in pics]
    for (i, k), p in zip(meta, judge.choose_many(qs, desc=desc)):
        for x, v in (p or {}).items():
            x = f"{NONE}:{k}" if x == NONE else x
            res[i][x] = res[i].get(x, 0) + v / 2
    return res


def draw_goals(L: dict, probs: dict, n: int, rng, exclude: set) -> list:
    """n distinct observations, each the top pick of its group (what goal_groups.py verified; now in archive/), sampled
    proportional to Jev's match probability; if fewer than n groups have a usable top pick, the next best
    observations fill up. One random direction each."""
    top = []
    for k, ids in L["groups"].items():
        ok = [o for o in ids if o not in exclude]
        if ok:
            o = max(ok, key=lambda x: probs.get(x, 0))
            if probs.get(o, 0) > probs.get(f"{NONE}:{k}", 0) and probs.get(o, 0) > 0:
                top.append(o)
    rest = sorted((o for ids in L["groups"].values() for o in ids if o not in exclude and o not in top),
                  key=lambda x: -probs.get(x, 0))
    cand = top if len(top) >= n else top + rest[:n - len(top)]
    w = np.array([probs.get(o, 0) + 1e-3 for o in cand])
    pick = rng.choice(len(cand), size=min(n, len(cand)), replace=False, p=w / w.sum())
    out = []
    for j in pick:
        o = cand[j]
        g = L["obs"][o][int(rng.integers(len(L["obs"][o])))]
        out.append({"obs": o, "id": g["id"], "group": g["group"], "sees": g["sees"], "push": g["push"],
                    "p_match": round(float(probs.get(o, 0)), 4)})
    return out


# ---------- run ----------

def save_img(out: Path, name: str, pic: np.ndarray) -> str:
    Image.fromarray(pic).save(out / "img" / name)
    return name


def run(a) -> None:
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    judge = Judge(a.judge, {**cfg["judges"]["defaults"], **cfg["judges"][a.judge]}, out / "judge_cache.sqlite", yes=True)
    L = library(a)
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        ea = argparse.Namespace(**st["eval"])
        print(f"resuming at generation {st['gen']}", file=sys.stderr)
    else:
        elites, qa = pick_elites(Path(a.elites_from), a.elites)
        ea = eval_args(a, qa)
        t0 = time.time()
        ev = QD.evaluate([Rule.from_dict(e["rule"]) for e in elites], ea)
        print(f"{len(elites)} elites grown in {time.time() - t0:.0f}s", file=sys.stderr)
        rng = np.random.default_rng([a.seed, 0])
        probs = match(judge, L, [e["pic"] for e in ev], rng, "match elites")
        st = {"gen": 0, "args": vars(a), "eval": {k: v for k, v in vars(ea).items() if k != "axes"} | {"axes": []},
              "elites": [], "branches": []}
        for i, (e, v, p) in enumerate(zip(elites, ev, probs)):
            img = save_img(out, f"E{i:02d}_start.png", v["pic"])
            st["elites"].append({**e, "image": img, "match": p, "bodies": v["bodies"], "motion": round(v["motion"], 4)})
            goals = draw_goals(L, p, a.goal_branches, rng, set())
            for b in range(a.goal_branches + a.free_branches):
                goal = goals[b] if b < len(goals) else None
                st["branches"].append({"elite": i, "name": f"E{i:02d}{'G' if goal else 'F'}{b}", "kind": "goal" if goal else "free",
                                       "rule": e["rule"], "image": img, "goals": [goal] if goal else [],
                                       "history": [{"gen": 0, "image": img, "goal": goal and goal["id"],
                                                    "bodies": v["bodies"], "motion": round(v["motion"], 4)}]})
        QD.save_state(out, st)
    pics = {}

    def pic(name):
        if name not in pics:
            pics[name] = np.asarray(Image.open(out / "img" / name).convert("RGB"))
        return pics[name]

    log_path = out / "duels.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["gen", "branch", "kind", "goal", "mutant", "texture", "p", "accepted"])
        while st["gen"] < a.generations:
            g = st["gen"] + 1
            t0 = time.time()
            rng = np.random.default_rng([a.seed, g])
            B = st["branches"]
            rules, owner = [], []
            for bi, br in enumerate(B):
                parent = Rule.from_dict(br["rule"])
                for _ in range(a.mutants):
                    k = mutate(parent, rng, a.mut_scale)
                    k.flow = "mace"
                    k.beta = float(np.clip(parent.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
                    rules.append(k)
                    owner.append(bi)
            ev = QD.evaluate(rules, ea)
            qs, meta = [], []
            for j, (bi, v) in enumerate(zip(owner, ev)):
                if ea.max_texture > 0 and v["texture"] > ea.max_texture:
                    continue
                br = B[bi]
                pid = steer_prompt(br["goals"][-1]) if br["kind"] == "goal" else FREE
                par = pic(br["image"])
                qs += [Query(pid, v["pic"], par), Query(pid, par, v["pic"])]
                meta.append((bi, j))
            vs = judge.judge_many(qs, desc=f"gen {g}")
            best = {}
            for n, (bi, j) in enumerate(meta):
                x, y = vs[2 * n], vs[2 * n + 1]
                if not (x.ok and y.ok):
                    continue
                p = 0.5 * (x.p_first + 1 - y.p_first)
                acc = p >= a.threshold
                if acc and p > best.get(bi, (0, None))[0]:
                    best[bi] = (p, j)
                br = B[bi]
                w.writerow([g, br["name"], br["kind"], br["goals"][-1]["id"] if br["goals"] else "", j % a.mutants,
                            round(ev[j]["texture"], 4), round(p, 4), int(acc)])
            for bi, (p, j) in best.items():
                br = B[bi]
                name = save_img(out, f"{br['name']}_g{g:03d}.png", ev[j]["pic"])
                pics[name] = ev[j]["pic"]
                br.update(rule=rules[j].to_dict(), image=name)
                br["history"].append({"gen": g, "image": name, "p": round(p, 4), "rule": rules[j].to_dict(),
                                      "goal": br["goals"][-1]["id"] if br["goals"] else None,
                                      "bodies": ev[j]["bodies"], "motion": round(ev[j]["motion"], 4)})
            fh.flush()
            if g % a.phase == 0 and g < a.generations:          # re-match goal branches, new goal each
                gb = [br for br in B if br["kind"] == "goal"]
                probs = match(judge, L, [pic(br["image"]) for br in gb], rng, f"re-match gen {g}")
                for br, p in zip(gb, probs):
                    new = draw_goals(L, p, 1, rng, {x["obs"] for x in br["goals"]})
                    br["goals"].append({**new[0], "from_gen": g})
                    br.setdefault("matches", []).append({"gen": g, "match": p})
            st["gen"] = g
            QD.save_state(out, st)
            print(f"gen {g}: {len(best)}/{len(B)} branches took a mutant, {len(qs)} judge calls, {time.time() - t0:.0f}s, "
                  f"judge spent ${judge.spent:.4f}", file=sys.stderr)


# ---------- check ----------

def at(br: dict, g: int) -> str:
    return [h for h in br["history"] if h["gen"] <= g][-1]["image"]


def goal_at(br: dict, g: int) -> dict | None:
    """The goal a branch was steered by during the phase that ends at generation g."""
    gs = [x for x in br["goals"] if x.get("from_gen", 0) < g]
    return gs[-1] if gs else None


def comparisons(st: dict) -> list:
    """(test, elite, x image, y image, prompt kind, goal) for every held-out comparison."""
    P, G = st["args"]["phase"], st["gen"]
    rows = []
    for i, e in enumerate(st["elites"]):
        bs = [b for b in st["branches"] if b["elite"] == i]
        gb, fb = [b for b in bs if b["kind"] == "goal"], [b for b in bs if b["kind"] == "free"]
        for x in gb:
            g1 = goal_at(x, P)
            for y in gb:
                if y is not x:
                    rows.append(("spec_vs_goal", i, at(x, P), at(y, P), "attain", g1))
            for y in fb:
                rows.append(("spec_vs_free", i, at(x, P), at(y, P), "attain", g1))
            rows.append(("progress", i, at(x, P), e["image"], "attain", g1))
            if G > P:
                g2 = goal_at(x, G)
                rows.append(("phase2_vs_own_past", i, at(x, G), at(x, P), "attain", g2))
                for y in fb:
                    rows.append(("phase2_vs_free", i, at(x, G), at(y, G), "attain", g2))
            for y in fb:
                rows.append(("interest_goal_vs_free", i, at(x, G), at(y, G), "interest", None))
        for x in bs:
            rows.append((f"interest_{x['kind']}_vs_start", i, at(x, G), e["image"], "interest", None))
    return rows


def check(a) -> None:
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    rows = comparisons(st)
    cache = {}

    def pic(name):
        if name not in cache:
            cache[name] = np.asarray(Image.open(out / "img" / name).convert("RGB"))
        return cache[name]

    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    res = {}
    for jn in a.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, out / "check_cache.sqlite", yes=True)
        qs = []
        for t, i, x, y, kind, g in rows:
            pid = attain_prompt(g) if kind == "attain" else INTEREST
            qs += [Query(pid, pic(x), pic(y)), Query(pid, pic(y), pic(x))]
        vs = judge.judge_many(qs, desc=f"check {jn}")
        res[jn] = [None if not (vs[2 * k].ok and vs[2 * k + 1].ok) else round(0.5 * (vs[2 * k].p_first + 1 - vs[2 * k + 1].p_first), 4)
                   for k in range(len(rows))]
        print(f"{jn}: spent ${judge.spent:.4f}", file=sys.stderr)
    spread = None
    try:
        emb = QD.embedder(a.embed_model)
        names = sorted({b["image"] for b in st["branches"]} | {e["image"] for e in st["elites"]})
        tiles = [Image.fromarray(pic(n)[-ME.TILE:, -ME.TILE:]) for n in names]
        E = dict(zip(names, np.asarray(emb(tiles), np.float64)))
        spread = {n: v.tolist() for n, v in E.items()}
    except Exception as ex:                                 # no torch on this machine: skip the spread metric
        print(f"spread skipped ({type(ex).__name__}: {ex})", file=sys.stderr)
    (out / "check.json").write_text(json.dumps({"rows": rows, "p_first": res, "emb": spread}))
    report(a)


# ---------- your ratings ----------

PAGE = """<!doctype html><meta charset=utf-8><title>goal session</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:16px}img{width:430px;border:1px solid #333}
.ref img{width:220px}button{padding:8px 22px;font:15px system-ui;margin:8px 8px 0 0}#n{color:#999}.q{font-size:17px;margin:8px 0 12px}</style>
<div id=n></div><div class=q id=q></div>
<div style="display:flex;gap:16px;align-items:flex-start"><div class=ref id=ref></div>
<div><img id=l><br><button data-p=left>left</button></div><div><img id=r><br><button data-p=right>right</button></div></div>
<button data-p=neither>neither / can't tell</button>
<script>
let todo=[];async function load(){todo=await (await fetch('/todo')).json();show()}
function show(){if(!todo.length){document.body.innerHTML='<h2>Done, thank you. Close this tab and press Ctrl-C.</h2>';return}
const t=todo[0];document.getElementById('n').textContent=todo.length+' left';document.getElementById('q').innerHTML=t.q;
document.getElementById('l').src='/img/'+t.left;document.getElementById('r').src='/img/'+t.right;
document.getElementById('ref').innerHTML=t.ref?'<div>start</div><img src="/img/'+t.ref+'">':''}
document.querySelectorAll('button').forEach(b=>b.onclick=async()=>{const t=todo[0];
await fetch('/pick',{method:'POST',body:JSON.stringify({id:t.id,pick:b.dataset.p})});todo.shift();show()});
load()</script>"""


def human_tasks(st: dict, seed: int) -> list:
    """Per goal branch: it vs a random free branch at the end of phase 1, on its goal (start shown for reference);
    per elite: two goal-vs-free pairs at the end, on interest. Left/right at random."""
    rng = np.random.default_rng(seed)
    P, G = st["args"]["phase"], st["gen"]
    tasks = []
    for i, e in enumerate(st["elites"]):
        bs = [b for b in st["branches"] if b["elite"] == i]
        gb, fb = [b for b in bs if b["kind"] == "goal"], [b for b in bs if b["kind"] == "free"]
        if not fb:
            continue
        for x in gb:
            y = fb[int(rng.integers(len(fb)))]
            g = goal_at(x, P)
            tasks.append({"test": "goal", "elite": i, "a": at(x, P), "b": at(y, P), "goal": g["id"], "ref": e["image"],
                          "q": f"The start (small, far left) showed: <i>{html.escape(g['sees'])}</i>. We wanted this change: "
                               f"<b>{html.escape(g['push'])}</b>. Which world shows it more clearly?"})
        for x in [gb[int(j)] for j in rng.choice(len(gb), min(2, len(gb)), replace=False)]:
            y = fb[int(rng.integers(len(fb)))]
            tasks.append({"test": "interest", "elite": i, "a": at(x, G), "b": at(y, G), "goal": None, "ref": None,
                          "q": "Which world is more interesting as artificial life: richer structure or behaviour, more "
                               "like something alive, more worth exploring further?"})
    order = rng.permutation(len(tasks))
    tasks = [tasks[j] for j in order]
    for k, t in enumerate(tasks):
        t["id"] = k
        t["left"], t["right"] = (t["a"], t["b"]) if rng.random() < 0.5 else (t["b"], t["a"])
    return tasks


def _human(out: Path) -> dict:
    p = out / "human.csv"
    return {int(r["id"]): r["pick"] for r in csv.DictReader(open(p))} if p.exists() else {}


def rate(a) -> None:
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    tp = out / "human_tasks.json"
    if not tp.exists():
        tp.write_text(json.dumps(human_tasks(st, a.seed), indent=1))
    tasks = json.loads(tp.read_text())
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
                done = _human(out)
                return self._send(200, json.dumps([t for t in tasks if t["id"] not in done]).encode(), "application/json")
            if self.path.startswith("/img/"):
                f = out / "img" / Path(self.path).name
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            x = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                p = out / "human.csv"
                new = not p.exists()
                with open(p, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["time", "id", "pick"])
                    if new:
                        w.writeheader()
                    w.writerow({"time": round(time.time(), 1), "id": x["id"], "pick": x["pick"]})
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"rating page: {url} ({len(tasks)} pairs; answers go to {out / 'human.csv'}; Ctrl-C to stop, progress is kept)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ---------- report ----------

def report(a) -> None:
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    P, G = st["args"]["phase"], st["gen"]
    L = [f"# Goal session pilot: {out.name}\n",
         f"{len(st['elites'])} elites x ({st['args']['goal_branches']} goal + {st['args']['free_branches']} free branches), "
         f"{G} generations, re-match every {P}. P = probability the first-named world wins, both orders; chance 0.5.\n"]
    rows_d = list(csv.DictReader(open(out / "duels.csv"))) if (out / "duels.csv").exists() else []
    if rows_d:
        L += ["**Driver duels**", "", "| branches | duels | accepted |", "|---|---|---|"]
        for k in ("goal", "free"):
            r = [x for x in rows_d if x["kind"] == k]
            if r:
                L.append(f"| {k} | {len(r)} | {np.mean([x['accepted'] == '1' for x in r]):.2f} |")
        L.append("")
    if (out / "check.json").exists():
        c = json.loads((out / "check.json").read_text())
        rows = [tuple(r) for r in c["rows"]]
        tests = [("spec_vs_goal", f"own goal: X vs the elite's other goal branches (gen {P})"),
                 ("spec_vs_free", f"own goal: X vs the elite's free branches (gen {P})"),
                 ("progress", f"own goal: X at gen {P} vs the elite it started from"),
                 ("phase2_vs_own_past", f"re-matched goal: X at gen {G} vs X at gen {P}"),
                 ("phase2_vs_free", f"re-matched goal: X vs free branches (gen {G})"),
                 ("interest_goal_vs_free", f"interest: goal vs free branches (gen {G})"),
                 ("interest_goal_vs_start", f"interest: goal branch (gen {G}) vs its elite"),
                 ("interest_free_vs_start", f"interest: free branch (gen {G}) vs its elite")]
        L += ["**Held-out check**", "", "| comparison (first-named = X) | " + " | ".join(c["p_first"]) + " |",
              "|---|" + "---|" * len(c["p_first"])]
        for t, lab in tests:
            cells = []
            for jn, ps in c["p_first"].items():
                v = [p for r, p in zip(rows, ps) if r[0] == t and p is not None]
                cells.append(f"{np.mean(v):.2f} (wins {sum(p > 0.5 for p in v)}/{len(v)})" if v else "-")
            if any(x != "-" for x in cells):
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")
        if c.get("emb"):
            E = {k: np.asarray(v) for k, v in c["emb"].items()}
            cos = lambda u, v: 1 - float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            def spread(names):
                return [cos(E[x], E[y]) for k, x in enumerate(names) for y in names[k + 1:]]
            sg, sf, dg, df = [], [], [], []
            for i, e in enumerate(st["elites"]):
                bs = [b for b in st["branches"] if b["elite"] == i]
                gn = [b["image"] for b in bs if b["kind"] == "goal"]
                fn = [b["image"] for b in bs if b["kind"] == "free"]
                sg += spread(gn); sf += spread(fn)
                dg += [cos(E[x], E[e["image"]]) for x in gn]; df += [cos(E[x], E[e["image"]]) for x in fn]
            L += ["**Spread** (DINOv2 cosine distance of the final snapshot, gen %d)" % G, "",
                  "| | goal branches | free branches |", "|---|---|---|",
                  f"| between branches of one elite | {np.mean(sg):.3f} | {np.mean(sf):.3f} |" if sg and sf else "",
                  f"| from the elite | {np.mean(dg):.3f} | {np.mean(df):.3f} |", ""]
    human = _human(out)
    if human and (out / "human_tasks.json").exists():
        tasks = {t["id"]: t for t in json.loads((out / "human_tasks.json").read_text())}
        L += ["**Your blind ratings** (X = the goal branch)", "", "| test | X chosen | other chosen | neither |", "|---|---|---|---|"]
        for test in ("goal", "interest"):
            r = [(tasks[k], v) for k, v in human.items() if k in tasks and tasks[k]["test"] == test]
            if r:
                xs = sum(v != "neither" and t[v] == t["a"] for t, v in r)
                ns = sum(v == "neither" for t, v in r)
                L.append(f"| {test} | {xs} | {len(r) - xs - ns} | {ns} |")
        L.append("")
    L += ["**Goals**", ""]
    for b in st["branches"]:
        if b["kind"] == "goal":
            acc = sum(1 for h in b["history"] if h["gen"] > 0)
            L.append(f"- {b['name']} ({acc} accepted): " + " | then: ".join(
                f"{g['sees']} -> {g['push']} (match {g['p_match']:.2f})" for g in b["goals"]))
    (out / "report.md").write_text("\n".join(x for x in L if x is not None) + "\n")
    print("\n".join(L))
    _html(out, st)
    print(f"wrote {out / 'report.md'} and {out / 'report.html'}")


def _html(out: Path, st: dict) -> None:
    P, G = st["args"]["phase"], st["gen"]
    cards = []
    for i, e in enumerate(st["elites"]):
        cells = [f"<div class=b><div class=t>elite (root {e['root']})</div><img src='img/{e['image']}'></div>"]
        for b in [b for b in st["branches"] if b["elite"] == i]:
            gl = "".join(f"<div class=g>{'phase %d: ' % (k + 1)}{html.escape(g['push'])}</div>" for k, g in enumerate(b["goals"]))
            imgs = "".join(f"<div><div class=t>gen {g}</div><img src='img/{at(b, g)}'></div>" for g in sorted({P, G}))
            cells.append(f"<div class=b><div class=t>{b['name']} ({b['kind']})</div>{gl or '<div class=g>complexity prompt</div>'}"
                         f"<div class=row>{imgs}</div></div>")
        cards.append(f"<h3>E{i:02d}</h3><div class=row>{''.join(cells)}</div>")
    page = ("<!doctype html><meta charset=utf-8><title>goal session</title><style>body{font:13px system-ui;background:#111;"
            "color:#eee;margin:16px}img{width:220px;border:1px solid #333}.row{display:flex;gap:8px;flex-wrap:wrap}"
            ".b{margin-bottom:8px;max-width:460px}.t{color:#999}.g{color:#9cf}</style><h2>Goal session pilot</h2>" + "".join(cards))
    (out / "report.html").write_text(page)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "check", "rate", "report"])
    ap.add_argument("--out", default="runs/goal-session")
    ap.add_argument("--elites-from", default="runs/mace-qd2", help="a mace_qd.py run folder")
    ap.add_argument("--library", default="data/goal_library_v3.json")
    ap.add_argument("--judge", default="jev", help="driver (run)")
    ap.add_argument("--judges", default="luna,jev", help="held-out first (check)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default="numpy", help="numpy | auto | mps | cuda")
    ap.add_argument("--elites", type=int, default=10)
    ap.add_argument("--goal-branches", type=int, default=3)
    ap.add_argument("--free-branches", type=int, default=3)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--phase", type=int, default=5, help="generations per goal before re-matching")
    ap.add_argument("--mutants", type=int, default=4)
    ap.add_argument("--mut-scale", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--size", type=int, default=None, help="default: the MAP-Elites run's")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--render", type=int, default=None)
    ap.add_argument("--gap", type=int, default=None)
    ap.add_argument("--sim-batch", type=int, default=None)
    ap.add_argument("--embed-model", default="dinov2_vits14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8768)
    a = ap.parse_args()
    backend.configure(a.backend)
    {"run": run, "check": check, "rate": rate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
