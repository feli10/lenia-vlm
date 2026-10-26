"""Learnable novelty (Zhang & Levin 2026) on MaCE: does the estimator work here, before we use it for anything?

Step 0 (their code, unmodified): clone github.com/Zhangyanbo/learnable-novelty and run `python src/eca.py`;
rule 110 should come out on top, about 63 bits (their Fig. 2). Colab section 15 does this.

  python scripts/ln_check.py equiv  --upstream /content/learnable-novelty     # our copy + 2D reservoir = theirs
  python scripts/ln_check.py sanity --out RUNS/ln-sanity --elites-from RUNS/mace-qd2 --backend auto

equiv: (1) the vendored rc_epiplexity is byte-identical to the upstream commit; (2) RCEpiplexity2D, with each
1D kernel copied into the middle row of a 2D kernel, gives the same score as their RCEpiplexity1D on their
ECA data (rules 110, 54, 30, 3, 1, 2, their ECA settings).

sanity: worlds whose answer we know, scored under every combination of our choices (the paper has no 2D
multichannel continuous substrate, so these are ours):
  dt (MaCE steps between the stacked target states): 1, 10, 50
  down (average-pool factor, i.e. how coarsely the reservoir sees): 1, 2, 4
  depth (reservoir layers; receptive-field radius depth-1 cells after pooling): 3, 4
  y_scale: raw (their fixed sigma_y = 1, on mass density) | std (each world's targets scaled to unit std)
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

GRID = {"dt": [1, 10, 50], "down": [1, 2, 4], "depth": [3, 4], "y_scale": ["raw", "std"]}
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


def sanity(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    settings = [EP.Setting(dt=d, down=w, depth=p, y_scale=y)
                for d, w, p, y in itertools.product(GRID["dt"], GRID["down"], GRID["depth"], GRID["y_scale"])]
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
                m, sd = EP.score(fr, times, s, runs=a.runs, device=dev)
                res["scores"][name]["S"][s.key()] = [round(m, 3), round(sd, 3)]
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
                m, sd = EP.score(fr, times, s, runs=a.runs, device=dev)
                res["scores"][name]["S"][s.key()] = [round(m, 3), round(sd, 3)]
    res["settings"] = [s.as_dict() for s in settings]
    (out / "sanity.json").write_text(json.dumps(res, indent=1))
    render.sheet(sheet, cols=4, title="ln-sanity worlds (burned-in state, first start)").save(out / "worlds.png")
    report(out, res, settings)


def report(out: Path, res: dict, settings: list) -> None:
    by = {}
    for name, r in res["scores"].items():
        by.setdefault(r["class"], []).append(r)
    L = ["# Learnable novelty on MaCE: sanity check\n",
         "Mean S (bits) per class of worlds, under every setting. Expected order: noise, diffusion ~ 0; confetti, "
         "frozen low; active high. `ok` = active above every other class.\n",
         "| setting | " + " | ".join(c for c in ORDER if c in by) + " | ok |", "|---|" + "---|" * (len([c for c in ORDER if c in by]) + 1)]
    oks = 0
    for s in settings:
        k = s.key()
        means = {c: float(np.mean([r["S"][k][0] for r in by[c]])) for c in ORDER if c in by}
        ok = all(means["active"] > v for c, v in means.items() if c not in ("active", "copy"))
        oks += ok
        L.append(f"| {k} | " + " | ".join(f"{means[c]:.1f}" for c in ORDER if c in by) + f" | {'yes' if ok else 'no'} |")
    L += ["", f"Active above all other simulated/noise classes in {oks}/{len(settings)} settings.", "",
          f"Frozen elites' motion {res['meta']['frozen_motion']}, active {res['meta']['active_motion']}."]
    (out / "sanity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["equiv", "sanity", "report"])
    ap.add_argument("--upstream", default="/content/learnable-novelty")
    ap.add_argument("--out", default="runs/ln-sanity")
    ap.add_argument("--elites-from", default="runs/mace-qd2")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    backend.configure(a.backend)
    if a.cmd == "report":
        out = Path(a.out)
        res = json.loads((out / "sanity.json").read_text())
        return report(out, res, [EP.Setting(**s) for s in res["settings"]])
    {"equiv": equiv, "sanity": sanity}[a.cmd](a)


if __name__ == "__main__":
    main()
