"""Learnable novelty (Zhang & Levin 2026) on MaCE: does the estimator work here, before we use it for anything?

Step 0 (their code, unmodified): clone github.com/Zhangyanbo/learnable-novelty and run `python src/eca.py`;
rule 110 should come out on top, about 63 bits (their Fig. 2). Colab section 15 does this.

  python scripts/ln_check.py equiv  --upstream /content/learnable-novelty     # our copy + 2D reservoir = theirs
  python scripts/ln_check.py sanity --out RUNS/ln-sanity2 --elites-from RUNS/mace-qd2 --backend auto

equiv: (1) the vendored rc_epiplexity is byte-identical to the upstream commit; (2) RCEpiplexity2D, with each
1D kernel copied into the middle row of a 2D kernel, gives the same score as their RCEpiplexity1D on their
ECA data (rules 110, 54, 30, 3, 1, 2, their ECA settings).

sanity: worlds whose answer we know, scored under every combination of our choices (the paper has no 2D
multichannel continuous substrate, so these are ours):
  dt (MaCE steps between the stacked target states): 1, 10, 50
  down (average-pool factor, i.e. how coarsely the reservoir sees): 1, 2, 4
  depth (reservoir layers; receptive-field radius depth-1 cells after pooling): 3, 4
  y_scale: raw (their fixed sigma_y = 1, on mass density) | std (each world's targets scaled to unit std)
  rows: all sites (as the paper) | occupied (4,000 rows from sites with mass; ours, see below)
v2 (after v1, runs/ln-sanity, ranked i.i.d. noise and grid confetti above every creature in all 36 settings) adds
two controls of ours: (1) a shuffled null, the same rows with the targets permuted across rows, subtracted from S
(the readout's chance fit grows with the target's scale; i.i.d. noise scores ~18 bits even in their own ECA
setup); (2) occupied-site rows, so a creature on a mostly empty grid is not diluted by background rows.
Fixed from the paper: 256 channels, kernel 3, ELU, pre-activation norm, ridge 0.3, eta 1, tau 8, 3 reservoir
draws per score. Worlds (4 rules each, 16 smoothed-noise starts each, 1500 warmup steps):
  noise     i.i.d. uniform X and Y (the "noisy TV"; no simulation)          expected ~0
  diffusion the mace_evo founders at beta 0 (pure diffusion, ends uniform)  expected ~0 (dark room)
  confetti  founders at beta 8 (grid noise, texture > 0.3)                  expected low
  frozen    the 4 mace-qd2 elites with the lowest motion                    expected low
  founders  founders at beta 2 (the substrate's default)                    in between
  active    the 4 mace-qd2 elites with the highest motion                   expected high
  copy      an active world's X with Y = X repeated (static but intricate)  unknown: the paper's zero cases
                                                                            are uniform states, not frozen detail
"""
from __future__ import annotations

import argparse
import filecmp
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import mace_qd as QD
from lenia import backend, epiplexity as EP, render, seeds
from lenia.explore import texture
from lenia.sample import sample
from lenia.sim import Rule, with_flow

GRID = {"dt": [1, 10, 50], "down": [1, 2, 4], "depth": [3, 4], "y_scale": ["raw", "std"], "rows": ["all", "occupied"]}
ORDER = ["noise", "diffusion", "confetti", "frozen", "founders", "active", "copy"]


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------- equiv ----------

def equiv(a) -> None:
    import torch
    up = Path(a.upstream)
    mine = EP._TP / "rc_epiplexity"
    for f in ("__init__.py", "core.py", "reservoirs.py", "online.py"):
        same = filecmp.cmp(mine / f, up / "src" / "rc_epiplexity" / f, shallow=False)
        print(f"vendored {f}: {'identical' if same else 'DIFFERENT'}")
        if not same:
            sys.exit("vendored copy differs from upstream")
    sys.path.insert(0, str(up / "src"))
    from systems.eca import eca_stacked_pair
    from rc_epiplexity import RCEpiplexity1D
    dev = device()
    R2 = EP.RCEpiplexity2D
    worst = 0.0
    for rule in (110, 54, 30, 3, 1, 2):
        torch.manual_seed(0)
        r1 = RCEpiplexity1D(depth=3, channels=256, kernel_size=3, ridge_lambda=0.03, eps=1e-8, device=dev)
        r2 = R2(depth=3, channels=256, kernel_size=3, ridge_lambda=0.03, eps=1e-8, device=dev)
        with torch.no_grad():
            for c1, c2 in zip([m for m in r1.phi if isinstance(m, torch.nn.Conv1d)],
                              [m for m in r2.phi if isinstance(m, torch.nn.Conv2d)]):
                c2.weight.zero_()
                k = c1.weight.shape[-1]
                c2.weight[:, :, k // 2, :] = c1.weight
                c2.bias.copy_(c1.bias)
            x, y = eca_stacked_pair(rule=rule, num_samples=512, width=64, burnin=1000, tau_max=32, seed=rule, device=dev)
            s1 = float(r1.epiplexity(x, y).epiplexity)
            s2 = float(r2.epiplexity(x.unsqueeze(2), y.unsqueeze(2)).epiplexity)
        worst = max(worst, abs(s1 - s2) / max(abs(s1), 1e-6))
        print(f"rule {rule:>3}: 1D {s1:8.3f}  2D {s2:8.3f}")
    print(f"largest relative difference {worst:.2e}: {'OK' if worst < 1e-3 else 'MISMATCH'}")


# ---------- sanity ----------

def worlds(a) -> tuple[dict, np.ndarray, dict]:
    """class -> list of rules (simulated classes) and the start states."""
    shape = (a.size, a.size)
    st = json.loads((Path(a.elites_from) / "state.json").read_text())
    el = [e for m in QD.islands_of(st) for e in m.values() if e.get("texture", 0) <= 0.3]
    el.sort(key=lambda e: e["motion"])
    base = sample("multichannel", 4, seed=3, R=13, channels=3, self_k=1, cross_k=1)
    cls = {"diffusion": [with_flow(r, "mace", 0.0) for r in base],
           "confetti": [with_flow(r, "mace", 8.0) for r in base],
           "frozen": [Rule.from_dict(e["rule"]) for e in el[:4]],
           "founders": [with_flow(r, "mace", 2.0) for r in base],
           "active": [Rule.from_dict(e["rule"]) for e in el[-4:]]}
    starts = np.stack([seeds.smooth_noise(13, shape, 3, seed=s, size_r=3.0) for s in range(a.starts)])
    meta = {"frozen_motion": [round(e["motion"], 3) for e in el[:4]], "active_motion": [round(e["motion"], 3) for e in el[-4:]]}
    return cls, starts, meta


def sanity(a, grid: dict | None = None) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    settings = [EP.Setting(dt=d, down=w, depth=p, y_scale=y, rows=r)
                for d, w, p, y, r in itertools.product(*(grid or GRID).values())]
    times = EP.times_for(settings)
    cls, starts, meta = worlds(a)
    res = json.loads((out / "sanity.json").read_text()) if (out / "sanity.json").exists() else {"scores": {}, "meta": meta}
    shape = (a.size, a.size)
    sheet = []
    active_frames = None
    t0 = time.time()
    for c in ORDER:
        if c in ("noise", "copy"):
            continue
        for i, rule in enumerate(cls[c]):
            name = f"{c}_{i}"
            fr = EP.film([rule] * a.starts, starts, a.warmup, times, shape)
            tex = float(np.mean([texture(f) for f in fr[:, 0]]))
            last = fr[0, 0]
            sheet.append((f"{name} tex {tex:.2f}", Image.fromarray(render.colorize(np.clip(last / max(np.percentile(last, 99.5), 1e-6), 0, 1))).resize((200, 200))))
            if c == "active" and active_frames is None:
                active_frames = fr
            if name in res["scores"] and len(res["scores"][name]["S"]) == len(settings):
                continue
            res["scores"][name] = {"class": c, "texture": round(tex, 4), "S": {}}
            for s in settings:
                res["scores"][name]["S"][s.key()] = EP.score(fr, times, s, runs=a.runs, device=dev)
            (out / "sanity.json").write_text(json.dumps(res, indent=1))
            print(f"{name}: done ({time.time() - t0:.0f}s)", file=sys.stderr)
    rng = np.random.default_rng(0)
    synth = {"noise": [rng.random(active_frames.shape, dtype=np.float32) * float(active_frames.max()) for _ in range(2)],
             "copy": [np.repeat(active_frames[:, :1], len(times), 1)]}
    for c, frs in synth.items():
        for i, fr in enumerate(frs):
            name = f"{c}_{i}"
            res["scores"][name] = {"class": c, "texture": None, "S": {}}
            for s in settings:
                res["scores"][name]["S"][s.key()] = EP.score(fr, times, s, runs=a.runs, device=dev)
    res["settings"] = [s.as_dict() for s in settings]
    (out / "sanity.json").write_text(json.dumps(res, indent=1))
    render.sheet(sheet, cols=4, title="ln-sanity worlds (burned-in state, first start)").save(out / "worlds.png")
    report(out, res, settings)


def report(out: Path, res: dict, settings: list) -> None:
    by = {}
    for name, r in res["scores"].items():
        by.setdefault(r["class"], []).append(r)
    cl = [c for c in ORDER if c in by]
    L = ["# Learnable novelty on MaCE: sanity check (v2: shuffled null, occupied sites)\n",
         "Per class of worlds, mean **net** S = S - null (bits): S is their estimator; the null is the same rows "
         "with the targets shuffled across rows (what the readout fits by chance). `_occ` settings score "
         f"{EP.OCC_ROWS} rows from occupied sites only. Expected: noise, diffusion ~ 0; confetti, frozen low; "
         "active high. `ok` = active above noise, diffusion, confetti and frozen.\n",
         "| setting | " + " | ".join(cl) + " | ok |", "|---|" + "---|" * (len(cl) + 1)]
    oks, okocc, nocc = 0, 0, 0
    L[1] = L[1].replace("`_occ` settings score", "`_occ` settings score at most").replace(
        "occupied sites only.", "occupied sites only; `_occfull` up to 25,600 occupied rows pooled over all starts.")
    for st in settings:
        k = st.key()
        means = {c: float(np.mean([r["S"][k]["net"][0] for r in by[c]])) for c in cl}
        ok = all(means["active"] > means[c] for c in ("noise", "diffusion", "confetti", "frozen") if c in means)
        oks += ok
        if st.rows != "all":
            nocc += 1
            okocc += ok
        L.append(f"| {k} | " + " | ".join(f"{means[c]:.1f}" for c in cl) + f" | {'yes' if ok else 'no'} |")
    L += ["", f"ok in {oks}/{len(settings)} settings ({okocc}/{nocc} of the occupied-site ones).", "",
          "Raw S and null for a few settings (class means):", "",
          "| setting | " + " | ".join(f"{c} S / null" for c in cl) + " |", "|---|" + "---|" * len(cl)]
    for st in settings:
        if st.dt == 10 and st.depth == 4 and st.down == 2 or (len(settings) <= 16 and st.y_scale == "raw"):
            k = st.key()
            L.append(f"| {k} | " + " | ".join(
                f"{np.mean([r['S'][k]['S'][0] for r in by[c]]):.1f} / {np.mean([r['S'][k]['null'][0] for r in by[c]]):.1f}"
                f" ({np.mean([r['S'][k]['rows'][0] for r in by[c]]) / 1000:.1f}k rows)"
                for c in cl) + " |")
    L += ["", f"Frozen elites' motion {res['meta']['frozen_motion']}, active {res['meta']['active_motion']}."]
    (out / "sanity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- equal rows, and the mace-qd2 check pairs ----------

EQUAL_GRID = {"dt": [10, 50], "down": [1, 2], "depth": [4], "y_scale": ["raw", "std"], "rows": ["all", "occ_full"]}


def equalrows(a) -> None:
    """The sanity worlds again, comparing all sites with occupied sites at the same row count (25,600)."""
    sanity(a, EQUAL_GRID)


ELITE_SETTINGS = [EP.Setting(dt=d, down=2, depth=4, y_scale="raw", rows=r) for d in (10, 50) for r in ("all", "occ_full")]


def elites(a) -> None:
    """Score the worlds of the mace-qd2 check pairs (final elite vs the cell's elite at the half-way snapshot) and
    compare learnable novelty with (1) the cheap descriptors mace_qd already computes and (2) Luna's and Jev's
    pairwise preferences on those pairs."""
    import re
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qd = Path(a.elites_from)
    st = json.loads((qd / "state.json").read_text())
    chk = json.loads((qd / "check" / "check.json").read_text())
    at = chk["at"]
    old = QD.islands_of(json.loads((qd / "snapshots" / f"archive_it{at:04d}.json").read_text()))
    new = QD.islands_of(st)
    entry = {}
    for x, y in chk["pairs"]:
        for nm in (x, y):
            k, ck, which = re.fullmatch(r"I(\d+)c(.+)_(final|it\d+)", nm).groups()
            entry[nm] = (new if which == "final" else old)[int(k)][ck]
    names = sorted(entry)
    starts = np.stack([seeds.smooth_noise(13, (a.size, a.size), 3, seed=s, size_r=3.0) for s in range(a.elite_starts)])
    times = EP.times_for(ELITE_SETTINGS)
    dev = device()
    path = out / "elites.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    todo = [n for n in names if n not in res]
    per = max(1, 64 // a.elite_starts)
    t0 = time.time()
    for b in range(0, len(todo), per):
        group = todo[b:b + per]
        rules = [Rule.from_dict(entry[n]["rule"]) for n in group for _ in range(a.elite_starts)]
        fr = EP.film(rules, np.concatenate([starts] * len(group)), a.warmup, times, (a.size, a.size))
        for j, n in enumerate(group):
            f = fr[j * a.elite_starts:(j + 1) * a.elite_starts]
            e = entry[n]
            res[n] = {"desc": {k: e.get(k) for k in ("motion", "coverage", "bodies", "texture", "form")},
                      "S": {s.key(): EP.score(f, times, s, runs=2, device=dev) for s in ELITE_SETTINGS}}
        path.write_text(json.dumps(res, indent=1))
        print(f"{min(b + per, len(todo))}/{len(todo)} worlds ({time.time() - t0:.0f}s)", file=sys.stderr)
    elites_report(out, res, chk)


def elites_report(out: Path, res: dict, chk: dict) -> None:
    from scipy.stats import spearmanr
    names = sorted(res)
    L = ["# Learnable novelty vs. descriptors and judges (mace-qd2 check pairs)\n",
         f"{len(names)} worlds: the {len(chk['pairs'])} cells whose elite changed after iteration {chk['at']}, final and old "
         "elite. Net S = S - shuffled null, down 2, depth 4, raw targets.\n",
         "**Spearman correlation of net S with the mace_qd descriptors** (if S mostly tracks these, it adds little):\n",
         "| setting | motion | coverage | bodies | texture |", "|---|---|---|---|---|"]
    keys = list(next(iter(res.values()))["S"])
    for k in keys:
        v = [res[n]["S"][k]["net"][0] for n in names]
        row = []
        for d in ("motion", "coverage", "bodies", "texture"):
            w = [res[n]["desc"][d] for n in names]
            ok = [i for i, x in enumerate(w) if x is not None]
            row.append(f"{spearmanr([v[i] for i in ok], [w[i] for i in ok])[0]:.2f}" if len(ok) > 3 else "-")
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L += ["", "**Agreement with the judges on the pairs** (share of pairs where the world with higher S is the one the "
          "judge preferred, P > 0.5; chance 0.5). Same for a world's motion and coverage, as baselines.\n",
          "| judge | " + " | ".join(keys) + " | motion | coverage |", "|---|" + "---|" * (len(keys) + 2)]
    for jn, ps in chk["p_first_of_pair"].items():
        cells = []
        for k in keys + ["motion", "coverage"]:
            agree = []
            for (x, y), p in zip(chk["pairs"], ps):
                if p is None or x not in res or y not in res or abs(p - 0.5) < 1e-9:
                    continue
                if k in ("motion", "coverage"):
                    dx, dy = res[x]["desc"][k], res[y]["desc"][k]
                else:
                    dx, dy = res[x]["S"][k]["net"][0], res[y]["S"][k]["net"][0]
                if dx is None or dy is None or dx == dy:
                    continue
                agree.append((dx > dy) == (p > 0.5))
            cells.append(f"{np.mean(agree):.2f} ({sum(agree)}/{len(agree)})" if agree else "-")
        L.append(f"| {jn} | " + " | ".join(cells) + " |")
    L += ["", "Rows per score (mean): " + ", ".join(
        f"{k} {np.mean([res[n]['S'][k]['rows'][0] for n in names]) / 1000:.1f}k" for k in keys)]
    (out / "elites.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["equiv", "sanity", "report", "equalrows", "elites"])
    ap.add_argument("--upstream", default="/content/learnable-novelty")
    ap.add_argument("--out", default="runs/ln-sanity2")
    ap.add_argument("--elites-from", default="runs/mace-qd2")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--elite-starts", type=int, default=8, help="elites: smoothed-noise starts per world")
    a = ap.parse_args()
    backend.configure(a.backend)
    if a.cmd == "report":
        out = Path(a.out)
        res = json.loads((out / "sanity.json").read_text())
        return report(out, res, [EP.Setting(**s) for s in res["settings"]])
    {"equiv": equiv, "sanity": sanity, "equalrows": equalrows, "elites": elites}[a.cmd](a)


if __name__ == "__main__":
    main()
