"""Disagreement review (11-11; project doc "Handoff - improving Jev-Felix agreement"): the rating-benchmark pairs where
(almost) every Jev configuration disagreed with Felix, as the films he rated, so he can say what he was looking at.

Before his old pick is shown he re-picks each pair blind (sides swapped from the session): if he flips, the miss is
his own noise rather than Jev's blind spot. One self-contained page (films embedded), no judge calls.

  python3 scripts/disagree_review.py --bench RUNS/rate-bench --source RUNS/full-s1 --max-right 1
  open RUNS/rate-bench/review/review.html
From papers: none. Ours: the design.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
from pathlib import Path

Q_INTEREST = ("Which world is more interesting as artificial life: richer structure or behaviour, more like something "
              "alive, more worth exploring further?")
Q_GOAL = "Which world shows this change more clearly? Judge only the change asked for, not which world is nicer."
GLOSS = {"goal:control-vs-start": "one unsteered child vs the expedition's start, with the goal",
         "goal:end-vs-ref": "an expedition's end vs the world its goal is judged against",
         "interest:end-vs-start": "an expedition's end vs its start",
         "interest:end-vs-mid": "an expedition's end vs its world after step 5",
         "interest:map-replacement": "a map challenger vs the elite it replaced",
         "interest:gallery-vs-random": "a gallery pick vs a random elite",
         "calib:cross-cell": "two random elites from different cells"}


def agree(r, t):
    p = 0.5 * (r["p1"] + 1 - r["p2"]) if t["felix"] == t["a"] else 0.5 * (r["p2"] + 1 - r["p1"])
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--max-right", type=int, default=1, help="include pairs where at most this many configs agreed")
    a = ap.parse_args()
    B, F = Path(a.bench), Path(a.source)
    items = json.loads((B / "items.json").read_text())["items"]
    R = json.loads((B / "results.json").read_text())["results"]
    tasks = {t["id"]: t for t in json.loads((F / "rate" / "tasks.json").read_text())}
    pairs, counts = [], {}
    for t in items:
        ps = {k: agree(R[k][str(t["id"])], t) for k in R if str(t["id"]) in R[k]}
        right = sum(p > 0.5 for p in ps.values())
        counts[right] = counts.get(right, 0) + 1
        if right > a.max_right:
            continue
        s = tasks[t["id"]]
        other = t["b"] if t["felix"] == t["a"] else t["a"]
        # sides swapped from the session: the session's right world is shown on the left
        left_w, right_w = s["right_world"], s["left_world"]
        file_of = {s["left_world"]: s["left"], s["right_world"]: s["right"]}
        gif = lambda w: base64.b64encode((F / "rate" / "images" / f"{file_of[w]}.gif").read_bytes()).decode()
        cf, co = t["change"][t["felix"]], t["change"][other]
        pairs.append({"id": t["id"], "group": t["group"], "gloss": GLOSS.get(t["group"], ""),
                      "question": Q_GOAL if t["question"] == "goal" else Q_INTEREST,
                      "sees": t.get("sees"), "push": t.get("push"),
                      "felix_side": "left" if t["felix"] == left_w else "right",
                      "right": right, "n": len(ps),
                      "cfg": {k: round(p, 2) for k, p in ps.items()},
                      "change_side": ("left" if (cf > co) == (t["felix"] == left_w) else "right"),
                      "change_ratio": round(max(cf, co) / max(min(cf, co), 1e-9), 2),
                      "gl": gif(left_w), "gr": gif(right_w)})
    random.Random(11).shuffle(pairs)
    out = B / "review"
    out.mkdir(exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(pairs)).replace("__COUNTS__", json.dumps(counts))
    (out / "review.html").write_text(html)
    print(f"review.html: {len(pairs)} pairs (at most {a.max_right} of 10 configs agreed); "
          f"configs right per pair: {dict(sorted(counts.items()))}")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Disagreement review</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#dedcd6;--you:#1f7a4d;--jev:#b5452b;--acc:#2f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#211f1d;--ink:#ecebe7;--mute:#9a988f;--line:#38352f;--you:#4fbf86;--jev:#e0775c;--acc:#7aa2ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 80px}h1{font-size:20px;margin:0 0 4px}.mute{color:var(--mute)}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}
button{font:inherit;padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.on{border-color:var(--acc);outline:2px solid var(--acc)}button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:12px}
.films{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.film{position:relative;border:3px solid transparent;border-radius:10px;overflow:hidden}
.film img{width:100%;display:block;image-rendering:pixelated;background:#000}
.film .tag{position:absolute;top:8px;left:8px;padding:2px 8px;border-radius:6px;font-size:13px;font-weight:600;color:#fff}
.film.you{border-color:var(--you)}.tag.you{background:var(--you)}.film .side{position:absolute;bottom:6px;right:8px;color:#fff;font-size:13px;text-shadow:0 0 3px #000}
.goal{background:var(--bg);border-radius:8px;padding:10px 12px;margin:8px 0;font-size:14px}
.q{font-weight:600}.reveal{display:none}.reveal.show{display:block}
.cfgs{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}.cfg{font-size:12px;padding:2px 6px;border-radius:5px;border:1px solid var(--line)}
.cfg.w{color:var(--jev);border-color:var(--jev)}.cfg.r{color:var(--you);border-color:var(--you)}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.tags label{border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:13px;cursor:pointer}
.tags input{margin-right:5px}textarea{width:100%;min-height:70px;font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.flip{color:var(--jev);font-weight:600}.same{color:var(--you);font-weight:600}
@media (max-width:640px){.films{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Disagreement review</h1>
<div class="mute">Pairs where at most 1 of Jev's 10 picture × question combinations agreed with your seed-1 call. For each: re-pick blind first (sides are swapped from the session), then see your old pick and Jev, then note what you were looking at. Notes save in this browser as you go.</div>
<div class="bar"><button id="prev">← Prev</button><span id="pos" class="mute"></span><button id="next">Next →</button>
<span style="flex:1"></span><button id="copy" class="primary">Copy all notes</button><button id="dl">Download notes</button></div>
<div id="card" class="card"></div>
<div class="mute" style="margin-top:14px;font-size:13px">How many of the 10 combinations agreed, across all 59 pairs: <span id="counts"></span></div>
</main>
<script>
const P=__DATA__, COUNTS=__COUNTS__, KEY="disagree-review-v1";
const TAGS=["motion / behaviour","shape / structure","texture / fine detail","which body (several bodies)","the goal's specific change","overall look / colour","close call / gut"];
let S={}; try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
let i=0;
document.getElementById("counts").textContent=Object.keys(COUNTS).sort((a,b)=>a-b).map(k=>k+" of 10: "+COUNTS[k]+" pairs").join(" · ");
function st(p){return S[p.id]||(S[p.id]={repick:null,tags:[],note:""})}
function render(){
  const p=P[i], s=st(p), shown=s.repick!==null;
  document.getElementById("pos").textContent=`Pair ${i+1} of ${P.length}`;
  const film=(side,b64)=>{const you=shown&&p.felix_side===side;
    return `<div class="film ${you?'you':''}"><img src="data:image/gif;base64,${b64}">${you?'<span class="tag you">your seed-1 pick</span>':''}<span class="side">${side}</span></div>`};
  const goal=p.sees?`<div class="goal"><div><b>Started from:</b> ${p.sees}</div><div><b>Wanted change:</b> ${p.push}</div></div>`:"";
  const cfgs=Object.entries(p.cfg).map(([k,v])=>`<span class="cfg ${v>0.5?'r':'w'}">${k} ${v.toFixed(2)}</span>`).join("");
  const flip=shown&&s.repick!=="neither"&&s.repick!==p.felix_side;
  const verdict=!shown?"":s.repick==="neither"?`<span class="mute">Today: can't tell.</span>`:
     flip?`<span class="flip">Today you picked the other world: you flipped.</span>`:`<span class="same">Same pick as in seed 1.</span>`;
  document.getElementById("card").innerHTML=`
   <div class="mute" style="font-size:13px">#${p.id} · ${p.group} — ${p.gloss}</div>
   ${goal}<div class="q">${p.question}</div>
   <div class="films">${film("left",p.gl)}${film("right",p.gr)}</div>
   <div class="bar" style="margin:4px 0"><span>Your pick today:</span>
     ${["left","right","neither"].map(x=>`<button data-r="${x}" class="${s.repick===x?'on':''}">${x==="neither"?"can't tell":x}</button>`).join("")}
     <span>${verdict}</span></div>
   <div class="reveal ${shown?'show':''}">
     <div style="margin-top:8px"><b style="color:var(--jev)">Jev:</b> ${p.right} of ${p.n} combinations agreed with your seed-1 pick. P(your world) per combination:</div>
     <div class="cfgs">${cfgs}</div>
     <div class="mute" style="font-size:14px">Free change measure: the <b>${p.change_side}</b> world changes more (×${p.change_ratio.toFixed(2)}).</div>
     <div style="margin-top:12px"><b>What were you looking at?</b> Tick what decided it, then a sentence or two.</div>
     <div class="tags">${TAGS.map(t=>`<label><input type="checkbox" data-t="${t}" ${s.tags.includes(t)?'checked':''}>${t}</label>`).join("")}</div>
     <textarea id="note" placeholder="e.g. the left one's small body orbits the big one; the right just shimmers">${s.note}</textarea>
   </div>`;
  document.querySelectorAll("[data-r]").forEach(b=>b.onclick=()=>{s.repick=b.dataset.r;save();render()});
  document.querySelectorAll("[data-t]").forEach(c=>c.onchange=()=>{s.tags=[...document.querySelectorAll("[data-t]:checked")].map(x=>x.dataset.t);save()});
  const n=document.getElementById("note"); if(n)n.oninput=()=>{s.note=n.value;save()};
}
function dump(){return JSON.stringify(P.map(p=>{const s=st(p);return{id:p.id,group:p.group,seed1_pick:p.felix_side,repick:s.repick,
  flipped:s.repick&&s.repick!=="neither"?s.repick!==p.felix_side:null,tags:s.tags,note:s.note}}),null,1)}
document.getElementById("prev").onclick=()=>{i=(i+P.length-1)%P.length;render();scrollTo(0,0)};
document.getElementById("next").onclick=()=>{i=(i+1)%P.length;render();scrollTo(0,0)};
document.getElementById("copy").onclick=async()=>{const b=document.getElementById("copy");try{await navigator.clipboard.writeText(dump());b.textContent="Copied ✓"}catch(e){b.textContent="Copy failed: use Download"}setTimeout(()=>b.textContent="Copy all notes",2000)};
document.getElementById("dl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([dump()],{type:"application/json"}));a.download="review-notes.json";a.click()};
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
