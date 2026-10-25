"""Goal matching, grouped version: can Jev say what a MaCE world shows when the right answer is on the menu?

goal_match.py drew random 8-of-76 menus, so the entry that fits was usually missing and both Jev and the human
were forced into bad picks. Here the library (data/goal_library_v3.json, field "group"; no colour-specific
entries) is split into 7 groups (interior, shape, edge, surroundings, bodies, change, leaps). For each world Jev
answers one question per group with the whole group on the menu, plus "none of these fits" and one control entry
(something a MaCE world cannot show), each in 2 random orders.

Your check (`verify`, after `run`): for each of 20 worlds you see 7 statements, one per group, and answer yes / no /
unsure. Each statement is, at random, either Jev's pick or a decoy (another entry of the same group, or "none
fits"); you are not told which. If Jev recognises worlds, you say yes to its picks clearly more often than to the
decoys. About 20 s per world.

The longer alternative (`rate`): tick every entry that fits; `report` then scores Jev's pick as a hit if you ticked
it, against a random pick and against always giving your most common answer for the group.

  uv run python scripts/goal_groups.py run                   # Jev: 40 worlds x 7 groups x 2 orders = 560 calls
  uv run python scripts/goal_groups.py verify                # your blind yes/no check, in the browser (~7 min)
  uv run python scripts/goal_groups.py report                # report.md + report.html (worlds, your answers, Jev's picks)
  uv run python scripts/goal_groups.py rate                  # optional: tick-all ratings (~1 min per world)
  ... run --judges jev,luna                                  # also Luna, for comparison
  ... run --judges fake                                      # offline dry run

Worlds: the same 40 test worlds as goal_match.py (seed 0), which the library was not built from; you check the first 20.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lenia.config import REPO_ROOT, load
from lenia.judge import ChoiceQuery, Judge, load_env

GROUPS = {   # group -> what the question is about
    "interior": "the inside of the main body",
    "shape": "the overall shape and size of the main body",
    "edge": "the edge or outline of the main body",
    "surroundings": "what is around the main body",
    "bodies": "how many bodies there are and how they relate to each other",
    "change": "how the world changes between the snapshots",
    "leaps": "what the world looks a bit like",
}
NONE = "none"
NONE_TEXT = "None of these fits."
INSTRUCTIONS = ("You are shown one world from a mass-conserving variant of Lenia, a continuous cellular automaton used "
                "in artificial-life research. The image {W} shows the whole world as four snapshots taken some time "
                "apart, in reading order (top-left, top-right, bottom-left, bottom-right); the colours are three "
                "interacting chemical channels. Looking only at {ABOUT}, which description best matches this world? "
                "If none of them fits, choose \"None of these fits.\"")
N_WORLDS, N_HUMAN, N_ORDERS = 40, 20, 2


def setup(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "design.json").exists():
        return json.loads((out / "design.json").read_text())
    lib = json.loads((REPO_ROOT / a.library).read_text())
    rng = np.random.default_rng(a.seed)
    worlds = sorted(p.name for p in Path(a.images).glob("*_final.png"))
    if not worlds:
        sys.exit(f"no *_final.png images in {a.images}")
    worlds = [worlds[i] for i in sorted(rng.choice(len(worlds), min(N_WORLDS, len(worlds)), replace=False))]  # = goal_match.py
    first = {}                                   # one entry per distinct observation
    for g in lib:
        first.setdefault(g["sees"], g)
    groups = {k: [g["id"] for g in first.values() if g["group"] == k] for k in GROUPS}
    ctrl = [g["id"] for g in first.values() if g["group"] == "control"]
    menus = {}                                   # "world|group|order" -> option ids in display order
    for w in worlds:
        for k, ids in groups.items():
            for o in range(N_ORDERS):
                m = ids + [NONE, str(rng.choice(ctrl))]
                menus[f"{w}|{k}|{o}"] = [m[i] for i in rng.permutation(len(m))]
    d = {"images": str(Path(a.images).resolve()), "worlds": worlds, "human_worlds": worlds[:N_HUMAN],
         "groups": groups, "menus": menus, "library": {g["id"]: g for g in lib}}
    (out / "design.json").write_text(json.dumps(d, indent=1))
    return d


def _text(d: dict, gid: str) -> str:
    return NONE_TEXT if gid == NONE else d["library"][gid]["sees"][0].upper() + d["library"][gid]["sees"][1:] + "."


def run(a) -> None:
    d = setup(a)
    out = Path(a.out)
    cfg = load(a.config)
    load_env(REPO_ROOT / ".env")
    keys = list(d["menus"])
    img = {w: np.asarray(Image.open(Path(d["images"]) / w).convert("RGB")) for w in d["worlds"]}
    qs = []
    for key in keys:
        w, k, _ = key.split("|")
        qs.append(ChoiceQuery(img[w], INSTRUCTIONS.replace("{ABOUT}", GROUPS[k]), {g: _text(d, g) for g in d["menus"][key]}))
    res = {}
    for jn in a.judges.split(","):
        judge = Judge(jn, {**cfg["judges"]["defaults"], **cfg["judges"][jn]}, out / "judge_cache.sqlite", yes=True)
        res[jn] = dict(zip(keys, judge.choose_many(qs, desc=f"groups {jn}")))
        print(f"{jn}: spent ${judge.spent:.4f}", file=sys.stderr)
    (out / "results.json").write_text(json.dumps(res))
    report(a)


def _ticks(out: Path) -> dict:
    """world -> {"ticks": set of ids, "note": str}; the last answer per world wins."""
    p = out / "human_ticks.jsonl"
    res = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                res[x["world"]] = {"ticks": set(x["ticks"]), "note": x.get("note", "")}
    return res


def report(a) -> None:
    out = Path(a.out)
    d = setup(a)
    lib = d["library"]
    res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    human = _ticks(out)
    L = [f"# Goal matching, grouped menus\n",
         f"{len(d['worlds'])} test worlds (mace-qd2 elites not used to build the library) x {len(GROUPS)} groups x "
         f"{N_ORDERS} orders. Each menu = the whole group + \"none of these fits\" + 1 control.\n"]
    if human:
        L += _human_summary(d, human)
    ver = _verdicts(out)
    if ver:
        L += _verify_summary(d, ver)
    for jn, probs in res.items():
        pick = {key: max(p, key=p.get) for key, p in probs.items() if p}
        L += [f"## {jn}\n", f"{len(pick)}/{len(probs)} calls answered.\n",
              "**Image dependence and stability** (order 0; modal share 1.0 = same pick for every world).\n",
              "| group | options | distinct picks | modal share | modal pick | 'none' | control | same pick when reshuffled |",
              "|---|---|---|---|---|---|---|---|"]
        for k in GROUPS:
            ps = [pick[f"{w}|{k}|0"] for w in d["worlds"] if f"{w}|{k}|0" in pick]
            ps = ["control" if x != NONE and lib[x]["group"] == "control" else x for x in ps]   # controls count as one option
            if not ps:
                continue
            vals, cnt = np.unique(ps, return_counts=True)
            modal = vals[cnt.argmax()]
            same = [pick[f"{w}|{k}|0"] == pick[f"{w}|{k}|{o}"] for w in d["worlds"] for o in range(1, N_ORDERS)
                    if f"{w}|{k}|0" in pick and f"{w}|{k}|{o}" in pick]
            nctrl = ps.count("control")
            L.append(f"| {k} | {len(d['groups'][k]) + 2} | {len(vals)} | {cnt.max() / len(ps):.2f} | "
                     f"{'(control)' if modal == 'control' else _text(d, modal)[:50]} | {ps.count(NONE)} | {nctrl} | {np.mean(same):.2f} |")
        L.append("")
        if human:
            L += _agreement(d, human, probs, pick)
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    _html(out, d, human, res, ver)
    print(f"\nwrote {out / 'report.md'} and {out / 'report.html'}")


def _human_summary(d: dict, human: dict) -> list:
    lib = d["library"]
    L = [f"## Felix's ticks ({len(human)} worlds)\n", "| group | ticks per world (mean) | worlds with no tick | most ticked |",
         "|---|---|---|---|"]
    for k, ids in d["groups"].items():
        n = [len(h["ticks"] & set(ids)) for h in human.values()]
        cnt = {g: sum(g in h["ticks"] for h in human.values()) for g in ids}
        top = sorted(cnt, key=cnt.get, reverse=True)[:2]
        L.append(f"| {k} | {np.mean(n):.1f} | {sum(x == 0 for x in n)} | " +
                 "; ".join(f"{lib[g]['sees'][:40]} ({cnt[g]})" for g in top if cnt[g]) + " |")
    never = [g for ids in d["groups"].values() for g in ids if not any(g in h["ticks"] for h in human.values())]
    L += ["", f"Never ticked ({len(never)}): " + "; ".join(lib[g]["sees"] for g in never) + "\n"]
    notes = [(w, h["note"]) for w, h in sorted(human.items()) if h["note"].strip()]
    if notes:
        L += ["Things you saw that no entry describes:", ""] + [f"- {w}: {n}" for w, n in notes] + [""]
    return L


def _agreement(d: dict, human: dict, probs: dict, pick: dict) -> list:
    """Per group: is the judge's pick among Felix's ticks? 'none' counts as right if he ticked nothing in the group."""
    L = ["**Agreement with Felix** (order 0 and 1 pooled; a pick is right if Felix ticked it, \"none\" is right if he "
         "ticked nothing in that group). Random = uniform pick from the same menu; prior = always Felix's most "
         "common answer for the group on the other worlds.\n",
         "| group | calls | judge top pick | judge prob. on right answers | random | prior |", "|---|---|---|---|---|---|"]
    tot = {"n": 0, "hit": 0, "soft": 0, "rand": 0, "prior": 0}
    for k, ids in d["groups"].items():
        n = hit = soft = rand = prior = 0
        for w, h in human.items():
            right = (h["ticks"] & set(ids)) or {NONE}
            others = [human[v]["ticks"] & set(ids) or {NONE} for v in human if v != w]
            freq = {}
            for s in others:
                for g in s:
                    freq[g] = freq.get(g, 0) + 1
            best = max(freq, key=freq.get) if freq else NONE
            for o in range(N_ORDERS):
                key = f"{w}|{k}|{o}"
                if key not in pick:
                    continue
                n += 1
                hit += pick[key] in right
                soft += sum(p for g, p in probs[key].items() if g in right)
                rand += len(right & set(d["menus"][key])) / len(d["menus"][key])
                prior += best in right
        if n:
            L.append(f"| {k} | {n} | {hit / n:.2f} | {soft / n:.2f} | {rand / n:.2f} | {prior / n:.2f} |")
            for x, v in zip(("n", "hit", "soft", "rand", "prior"), (n, hit, soft, rand, prior)):
                tot[x] += v
    if tot["n"]:
        L.append(f"| **all** | {tot['n']} | **{tot['hit'] / tot['n']:.2f}** | {tot['soft'] / tot['n']:.2f} | "
                 f"{tot['rand'] / tot['n']:.2f} | {tot['prior'] / tot['n']:.2f} |")
    return L + [""]


def _html(out: Path, d: dict, human: dict, res: dict, ver: list) -> None:
    """One card per world: the image, then per group Felix's ticks and each judge's pick (green = he ticked it)."""
    seen = set(human) | {v["world"] for v in ver}
    worlds = [w for w in d["worlds"] if w in seen] + [w for w in d["worlds"] if w not in seen]
    jns = list(res)
    rows = []
    for w in worlds:
        src = os.path.relpath(Path(d["images"]) / w, out)
        t = ["<tr><th>group</th><th>Felix ticked</th>" + "".join(f"<th>{j} (order 0)</th>" for j in jns) + "</tr>"]
        for k, ids in d["groups"].items():
            ticks = human[w]["ticks"] & set(ids) if w in human else None
            felix = "-" if ticks is None else ("<i>nothing</i>" if not ticks else "<br>".join(html.escape(_text(d, g)) for g in ticks))
            cells = []
            for j in jns:
                p = res[j].get(f"{w}|{k}|0")
                if not p:
                    cells.append("<td>failed</td>")
                    continue
                g = max(p, key=p.get)
                ok = None if ticks is None else (g in ticks or (g == NONE and not ticks))
                col = "" if ok is None else (" style='color:#7f7'" if ok else " style='color:#f88'")
                cells.append(f"<td{col}>{html.escape(_text(d, g))} ({p[g]:.2f})</td>")
            t.append(f"<tr><td>{k}</td><td>{felix}</td>{''.join(cells)}</tr>")
        vw = [v for v in ver if v["world"] == w]
        if vw:
            t.append("<tr><th colspan=9>your blind check</th></tr>" + "".join(
                f"<tr><td>{v['group']}</td><td colspan=8>{html.escape(_text(d, v['gid']))} <i>({v['source']})</i>: "
                f"<b>{v['answer']}</b></td></tr>" for v in vw))
        note = human.get(w, {}).get("note", "")
        rows.append(f"<div class=c><h3>{html.escape(w)}</h3><div class=f><img src='{html.escape(src)}'><div><table>"
                    f"{''.join(t)}</table>{'<p>Note: ' + html.escape(note) + '</p>' if note else ''}</div></div></div>")
    page = ("<!doctype html><meta charset=utf-8><title>goal groups</title><style>body{font:14px system-ui;background:#111;"
            "color:#eee;margin:20px}img{width:420px;border:1px solid #333}.f{display:flex;gap:16px}.c{margin-bottom:30px}"
            "td,th{border-bottom:1px solid #333;padding:3px 8px;text-align:left;vertical-align:top}</style>"
            "<h2>Goal matching, grouped menus</h2>" + "".join(rows))
    (out / "report.html").write_text(page)


def _verdicts(out: Path) -> list:
    """Blind-check answers; the last answer per (world, group) wins."""
    p = out / "verify.jsonl"
    res = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                res[(x["world"], x["group"])] = x
    return list(res.values())


def _verify_items(a, d: dict) -> list:
    """Per (human world, group): Jev's order-0 pick or, with prob. 1/2, a decoy from the same group (or "none")."""
    out = Path(a.out)
    p = out / "verify_design.json"
    if p.exists():
        return json.loads(p.read_text())
    if not (out / "results.json").exists():
        sys.exit("run Jev first: goal_groups.py run")
    res = json.loads((out / "results.json").read_text())
    jn = a.judges.split(",")[0]
    if jn not in res:
        sys.exit(f"no results for judge {jn}")
    rng = np.random.default_rng(a.seed + 1)
    items = []
    for w in d["human_worlds"]:
        for k, ids in d["groups"].items():
            pr = res[jn].get(f"{w}|{k}|0")
            if not pr:
                continue
            jp = max(pr, key=pr.get)
            if rng.random() < 0.5:
                pool = [g for g in ids + [NONE] if g != jp]
                items.append({"world": w, "group": k, "gid": str(rng.choice(pool)), "source": "decoy"})
            else:
                items.append({"world": w, "group": k, "gid": jp, "source": jn})
    p.write_text(json.dumps(items, indent=1))
    return items


def _verify_summary(d: dict, ver: list) -> list:
    L = [f"## Blind check ({len({v['world'] for v in ver})} worlds)\n",
         "Share of statements you said yes to, for the judge's picks vs. decoys (\"unsure\" left out). "
         "The gap is the signal.\n", "| group | judge: yes / answered | decoy: yes / answered |", "|---|---|---|"]
    def rate(vs):
        a = [v["answer"] == "yes" for v in vs if v["answer"] != "unsure"]
        return f"{np.mean(a):.2f} ({sum(a)}/{len(a)})" if a else "-"
    for k in d["groups"]:
        vk = [v for v in ver if v["group"] == k]
        L.append(f"| {k} | {rate([v for v in vk if v['source'] != 'decoy'])} | {rate([v for v in vk if v['source'] == 'decoy'])} |")
    L.append(f"| **all** | **{rate([v for v in ver if v['source'] != 'decoy'])}** | {rate([v for v in ver if v['source'] == 'decoy'])} |")
    nn = [v for v in ver if v["source"] != "decoy" and v["gid"] == NONE]
    L += ["", f"The judge said \"none fits\" in {len(nn)} of the checked (world, group) pairs.\n"]
    return L


VPAGE = """<!doctype html><meta charset=utf-8><title>blind check</title>
<style>body{font:15px system-ui;background:#111;color:#eee;margin:16px}#im{width:600px;border:1px solid #333}
.r{border-bottom:1px solid #333;padding:8px 0}.k{color:#9cf;font-size:13px}.b button{margin:4px 6px 0 0;padding:4px 14px;
font:14px system-ui;background:#222;color:#eee;border:1px solid #555;cursor:pointer}.b button.on{background:#2c5e8a}
details{font-size:12px;color:#aaa}#go{padding:8px 24px;font:15px system-ui;margin-top:10px}#n{color:#999}</style>
<div id=n></div><p>Is each statement true of this world? (four snapshots, reading order)</p>
<div style="display:flex;gap:20px;align-items:flex-start"><img id=im><div style="flex:1"><div id=rows></div>
<button id=go disabled>Next</button></div></div>
<script>
let todo=[];async function load(){const r=await fetch('/todo');todo=await r.json();show()}
function show(){if(!todo.length){document.body.innerHTML='<h2>Done, thank you. Close this tab and press Ctrl-C.</h2>';return}
const t=todo[0];document.getElementById('n').textContent=todo.length+' worlds left';document.getElementById('im').src='/img/'+t.world;
const R=document.getElementById('rows');R.innerHTML='';const go=document.getElementById('go');go.disabled=true;
t.items.forEach(it=>{it.answer=null;const d=document.createElement('div');d.className='r';
d.innerHTML='<div class=k>'+it.about+'</div><div>'+it.text+'</div>'+(it.list?'<details><summary>the list</summary>'+it.list+'</details>':'');
const b=document.createElement('div');b.className='b';for(const a of ['yes','no','unsure']){const x=document.createElement('button');
x.textContent=a;x.onclick=()=>{it.answer=a;b.querySelectorAll('button').forEach(y=>y.classList.toggle('on',y===x));
go.disabled=!t.items.every(i=>i.answer)};b.appendChild(x)}d.appendChild(b);R.appendChild(d)})}
document.getElementById('go').onclick=async()=>{const t=todo[0];
await fetch('/answer',{method:'POST',body:JSON.stringify(t.items.map(i=>({world:t.world,group:i.group,gid:i.gid,source:i.source,answer:i.answer})))});
todo.shift();show()};
load()</script>"""


def verify(a) -> None:
    out = Path(a.out)
    d = setup(a)
    lib = d["library"]
    items = _verify_items(a, d)
    path = out / "verify.jsonl"

    def todo():
        done = {(v["world"], v["group"]) for v in _verdicts(out)}
        res = []
        for w in d["human_worlds"]:
            its = [i for i in items if i["world"] == w]
            if its and any((w, i["group"]) not in done for i in its):
                res.append({"world": w, "items": [{**i, "about": f"{i['group']}: {GROUPS[i['group']]}",
                            "text": (f"None of the {i['group']} descriptions in the library fits." if i["gid"] == NONE
                                     else html.escape(_text(d, i["gid"]))),
                            "list": ("<br>".join(html.escape(lib[g]["sees"]) for g in d["groups"][i["group"]])
                                     if i["gid"] == NONE else "")} for i in its]})
        return res

    def save(x):
        with open(path, "a") as f:
            for v in x:
                f.write(json.dumps({"time": round(time.time(), 1), **v}) + "\n")

    _serve(a, d, out, VPAGE, todo, "/answer", save)


def _serve(a, d: dict, out: Path, page: str, todo, post_path: str, save) -> None:
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if self.path == "/todo":
                return self._send(200, json.dumps(todo()).encode(), "application/json")
            if self.path.startswith("/img/"):
                f = Path(d["images"]) / Path(self.path).name
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != post_path:
                return self._send(404, b"not found", "text/plain")
            x = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                save(x)
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"page: {url} (answers go to {out}; Ctrl-C to stop, progress is kept)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = """<!doctype html><meta charset=utf-8><title>tick what you see</title>
<style>body{font:14px system-ui;background:#111;color:#eee;margin:16px}#im{width:560px;border:1px solid #333;position:sticky;top:10px}
.g{margin:0 0 10px;break-inside:avoid}.g h4{margin:6px 0 2px;color:#9cf}label{display:block;padding:2px 0;cursor:pointer}label:hover{background:#223}
textarea{width:100%;height:50px;background:#222;color:#eee;border:1px solid #444;font:14px system-ui}
button{padding:8px 24px;font:15px system-ui;margin:8px 8px 0 0}#n{color:#999}.cols{columns:2;column-gap:24px}</style>
<div id=n></div><p>Tick <b>every</b> description that fits this world (four snapshots, reading order). Leave a group
empty if nothing in it fits. "Looks a bit like" entries: tick them only if the resemblance actually strikes you.</p>
<div style="display:flex;gap:20px;align-items:flex-start"><img id=im><div style="flex:1"><div id=opts class=cols></div>
<p>Anything you see that no entry describes? (optional)</p><textarea id=note></textarea>
<button id=go>Next</button><button id=back>Back</button></div></div>
<script>
let todo=[],done=[],groups={};
async function load(){const r=await fetch('/todo');const x=await r.json();todo=x.todo;groups=x.groups;show()}
function show(){if(!todo.length){document.body.innerHTML='<h2>Done, thank you. Close this tab and press Ctrl-C.</h2>';return}
const t=todo[0];document.getElementById('n').textContent=todo.length+' left';document.getElementById('im').src='/img/'+t.world;
const o=document.getElementById('opts');o.innerHTML='';
for(const [g,opts] of groups){const d=document.createElement('div');d.className='g';d.innerHTML='<h4>'+g+'</h4>';
for(const [id,txt] of opts){const l=document.createElement('label');const c=document.createElement('input');c.type='checkbox';
c.value=id;c.checked=(t.ticks||[]).includes(id);l.appendChild(c);l.appendChild(document.createTextNode(' '+txt));d.appendChild(l)}
o.appendChild(d)}document.getElementById('note').value=t.note||'';window.scrollTo(0,0)}
document.getElementById('go').onclick=async()=>{const t=todo[0];
const ticks=[...document.querySelectorAll('#opts input:checked')].map(c=>c.value);const note=document.getElementById('note').value;
await fetch('/tick',{method:'POST',body:JSON.stringify({world:t.world,ticks,note})});t.ticks=ticks;t.note=note;done.push(todo.shift());show()};
document.getElementById('back').onclick=()=>{if(done.length){todo.unshift(done.pop());show()}};
load()</script>"""


def rate(a) -> None:
    out = Path(a.out)
    d = setup(a)
    lib = d["library"]
    path = out / "human_ticks.jsonl"
    groups = [[f"{k}: {about}", [[g, lib[g]["sees"]] for g in d["groups"][k]]] for k, about in GROUPS.items()]

    def todo():
        done = _ticks(out)
        return {"todo": [{"world": w} for w in d["human_worlds"] if w not in done], "groups": groups}

    def save(x):
        with open(path, "a") as f:
            f.write(json.dumps({"time": round(time.time(), 1), "world": x["world"], "ticks": x["ticks"],
                                "note": x.get("note", "")}) + "\n")

    _serve(a, d, out, PAGE, todo, "/tick", save)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "verify", "rate", "report"])
    ap.add_argument("--out", default="runs/goal-groups-v3")
    ap.add_argument("--library", default="data/goal_library_v3.json", help="relative to the repo root")
    ap.add_argument("--images", default="runs/mace-qd2/check/images")
    ap.add_argument("--judges", default="jev")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args()
    {"run": run, "verify": verify, "rate": rate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
