"""ASAL's illumination search on MaCE-Lenia: the baseline for "is our work an advance, a rehash or a step back?"
(11-17, agreed with Sam; project doc "ASAL illumination baseline - spec").

In three sentences: ASAL's illumination search keeps a population of worlds whose final frames are far apart in CLIP
space, dropping the least novel one at a time. Here it runs on our substrate with everything but the search held at the
seed-1 full run's settings (seeds, start, rollout, mutation, grid-noise rule) and the same number of simulations. Its
output and ours then go through Felix's blind interest ratings and numbers neither method optimised (DINOv2 diversity,
rehash tests, distance from the seeds, behaviour).

  python scripts/asal_illum.py run    --out RUNS/asal-illum-s1 --like RUNS/full-s1/map --backend auto   # Colab, ~8 h T4
  python scripts/asal_illum.py atlas  --out RUNS/asal-illum-s1                                          # anywhere
  python scripts/asal_illum.py stats  --out RUNS/asal-illum-s1 --full RUNS/full-s1 --backend auto       # Colab, ~20 min
  python scripts/asal_illum.py build  --out RUNS/asal-illum-s1 --full RUNS/full-s1 --backend auto       # Colab, films
  PYTHONPATH=src:scripts python3 scripts/asal_illum.py rate   --out RUNS/asal-illum-s1                  # Mac
  PYTHONPATH=src:scripts python3 scripts/asal_illum.py report --out RUNS/asal-illum-s1                  # anywhere
  Offline dry run: run --out ~/ai-dry --like RUNS/full-s1/map --clip fake --size 64 --R 5 --warmup 100 --render 60
                   --gap 20 --init 4 --pop 24 --children 8 --budget 64 --snapshots 32,64

The search (ASAL, Kumar et al. 2024, Eq. 4 and appendix B.3; checked against github.com/SakanaAI/asal main_illuminate.py
and asal_metrics.py on 11-17): a steady-state genetic algorithm. The population starts as --pop identical copies (theirs:
256 copies of the zero parameter vector). Each iteration mutates --children uniformly drawn members, adds the children,
then removes the least novel member (smallest mean distance to its 2 nearest neighbours; distance = minus the dot product
of CLIP ViT-B/32 embeddings of the final frame rendered alone at 224 px) --children times. Their removal masks each
removed row and column of one distance matrix, which is the same as recomputing on the remaining members, as done here.
Population 256, 32 children per iteration and 2 neighbours are the repo's defaults (the paper's appendix reports 8,192
and 100,000 iterations for its own runs; the repo's default is 1,000 iterations, 32,000 simulations).
Changed for this port, and why: our genome and mutation (lenia.mace.mutate at the map's scale, beta jitter, as
mace_qd.py) instead of LeniaBreeder's Gaussian sigma 0.1, so a difference comes from the search, not the space; copies
of our 64 seeds (12 founders + 52 draws, as every map run; copies cycled to fill --pop, not re-simulated) instead of
copies of one Chan solution; our 1,500 + 1,200-step rollout from the map's fixed start; the grid-noise rule (a child with
texture > max_texture is discarded; ASAL has no validity check); embeddings are unit-normalised (cosine), which their
illumination curves (0.92-1.00, paper figure 11) imply.

Budget: the seed-1 full run simulated 64 seeds + 100 map iterations x 64 (= 6,464) + 50 expeditions x 10 steps x 48
children (= 24,000): 30,464, i.e. the seeds + 950 iterations of 32. Snapshots at 1,344 (the map-only baseline base-s1,
which stopped at iteration 20), 6,464 (the full run's map alone) and 30,464 (the whole full run), plus "unselected": the
first --pop distinct worlds simulated, kept whether or not they survived (seeds and their unselected mutants, ASAL's
"random" inset). Resumes from <out>/state.json.
From papers: the search and its objective (ASAL). Ours: the port, the comparison sets, the measures and the session.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mace_qd as QD
import stage1 as S1
from lenia import backend, render
from lenia import mace as ME
from lenia.mace import mutate
from lenia.sample import sample
from lenia.sim import Rule, with_flow

CLIP_MODEL = "openai/clip-vit-base-patch32"          # ASAL's CLIP
TPX = 224                                            # final-frame tile, CLIP's and DINOv2's input size
FULL_SIMS = 30464


# ---------- shared pieces ----------

def final_img(frame: np.ndarray) -> np.ndarray:
    """The final state rendered on its own at TPX px (ASAL: time_sampling='final', img_size=224); its own brightness
    scale (99.5th percentile), our colour map."""
    scale = max(float(np.percentile(frame, 99.5)), 1e-6)
    return np.asarray(Image.fromarray(render.colorize(np.clip(frame / scale, 0, 1))).resize((TPX, TPX), Image.BILINEAR))


def evaluate(rules: list, ea, pics: bool = False) -> list:
    """Grow every rule from the map's fixed start (as mace_qd.evaluate): the final frame rendered alone, the map's
    descriptors, and with pics the 2x2 picture (replay checks against stored pictures)."""
    from lenia import seeds
    shape = (ea.size, ea.size)
    A0 = seeds.smooth_noise(ea.R, shape, 3, seed=ea.start_seed, size_r=3.0)
    out = []
    for k in range(0, len(rules), ea.sim_batch):
        rr = rules[k:k + ea.sim_batch]
        _, films = ME.simulate(rr, np.stack([A0] * len(rr)), ea.warmup + ea.render, shape, ea.gap)
        for f in films:
            first, last = f[0], f[-1]
            mass = max(float(last.sum()), 1e-9)
            out.append({"img": final_img(last), "bodies": QD.n_bodies(last), "motion": float(np.abs(last - first).sum() / mass),
                        "texture": float(ME.texture(last)), "coverage": QD.coverage(last),
                        **({"pic": ME.picture(f)} if pics else {})})
    return out


def eval_settings(like: Path, a) -> argparse.Namespace:
    """The base map run's settings (seed-1 full run's map), no axes; dry-run overrides."""
    ea = S1.eval_settings(like, a)
    for k in ("R", "init"):
        if getattr(a, k, None):
            setattr(ea, k, getattr(a, k))
    return ea


def ea_of(st: dict) -> argparse.Namespace:
    return argparse.Namespace(**st["settings"], axes=[])


def seed_rules(ea) -> list:
    """The 64 seeds every map run starts from (mace_qd.py, iteration 0)."""
    found = ME.founders(argparse.Namespace(lineages=12, rule_seed=3, R=ea.R, beta=ea.beta, size=ea.size),
                        (ea.size, ea.size))[0]
    extra = [with_flow(r, "mace", ea.beta) for r in
             sample("multichannel", ea.init, seed=ea.init_seed, R=ea.R, channels=3, self_k=1, cross_k=1)]
    return found + extra


def child_of(par: Rule, rng, scale: float) -> Rule:
    """The map's variation (mace_qd.py): kernel mutation at its scale, beta x exp(N(0, 0.1)) in [0.5, 4]."""
    m = mutate(par, rng, scale)
    m.flow = "mace"
    m.beta = float(np.clip(par.beta * np.exp(rng.normal(0, 0.1)), 0.5, 4.0))
    return m


def nn_novelty(Z: np.ndarray, k: int = 2) -> np.ndarray:
    """ASAL's removal score: mean cosine distance to the k nearest other members (unit rows)."""
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    return 1 - np.sort(S, axis=1)[:, -k:].mean(1)


def asal_loss(Z: np.ndarray) -> float:
    """Eq. 4's population score: mean over members of the highest similarity to any other (lower = more diverse)."""
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    return float(S.max(1).mean())


def cull(Z: np.ndarray, n: int) -> list:
    keep = list(range(len(Z)))
    while len(keep) > n:
        keep.pop(int(np.argmin(nn_novelty(Z[keep]))))
    return keep


def read_members(out: Path) -> dict:
    m = {}
    for line in open(out / "members.jsonl"):
        x = json.loads(line)
        m[x["id"]] = x                                   # a repeated iteration rewrites the same rows: keep the last
    return m


def save_npz(path: Path, **kw) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **kw)
    tmp.replace(path)


def load_set(out: Path, name: str) -> dict:
    d = np.load(out / "snapshots" / f"pop_{name}.npz")
    ids = [str(x) for x in d["ids"]]
    first = sorted({i: n for n, i in reversed(list(enumerate(ids)))}.values())   # early snapshots can hold seed copies
    return {"ids": [ids[n] for n in first], "Z": d["Z"][first], "tiles": d["tiles"][first]}


def snap_names(out: Path) -> list:
    return sorted((p.stem[4:] for p in (out / "snapshots").glob("pop_*.npz")),
                  key=lambda s: -1 if s == "unselected" else int(s))


# ---------- run ----------

def run(a) -> None:
    out = Path(a.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    from lenia.embed import ClipScorer
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        ea = ea_of(st)
        for k in ("budget", "snapshots"):                # a resume may extend the run (e.g. --budget 50000)
            if getattr(a, k) is not None:
                st["args"][k] = getattr(a, k)
        P = np.load(out / "pop.npz")
        ids, Z, T = [str(x) for x in P["ids"]], P["Z"], P["tiles"]
        print(f"resuming {out} at iteration {st['it']} ({st['sims']:,} simulations)", file=sys.stderr)
    else:
        if not a.like:
            sys.exit("--like (the full run's map folder) is needed for a new run")
        ea = eval_settings(Path(a.like), a)
        keep_ = ("size", "R", "beta", "warmup", "render", "gap", "start_seed", "sim_batch", "max_texture", "mut_scale",
                 "init", "init_seed", "seed")
        st = {"args": {**{k: v for k, v in vars(a).items() if k != "cmd"},
                       "budget": a.budget or FULL_SIMS, "snapshots": a.snapshots or "1344,6464,30464"},
              "settings": {k: getattr(ea, k) for k in keep_}, "like": str(a.like), "it": 0, "sims": 0,
              "history": [], "snaps": []}
        ids, Z, T = [], np.zeros((0, 0), np.float32), np.zeros((0, TPX, TPX, 3), np.uint8)
    if getattr(a, "sim_batch", None):
        ea.sim_batch = a.sim_batch
    clip = ClipScorer("fake" if st["args"]["clip"] == "fake" else CLIP_MODEL)
    budget, N, C = st["args"]["budget"], st["args"]["pop"], st["args"]["children"]
    snaps = sorted(int(x) for x in str(st["args"]["snapshots"]).split(","))
    members = read_members(out) if (out / "members.jsonl").exists() else {}
    mf = open(out / "members.jsonl", "a")
    lf = open(out / "log.csv", "a", newline="")
    log = csv.writer(lf)
    if lf.tell() == 0:
        log.writerow(["it", "id", "parent", "root", "texture", "motion", "bodies", "coverage", "grid_noise", "kept"])
    while st["sims"] < budget:
        it, t0 = st["it"], time.time()
        rng = np.random.default_rng([ea.seed, 717, it])
        if it == 0:
            rules = seed_rules(ea)
            cands = [(f"s{i:03d}", r, None, i) for i, r in enumerate(rules)]
        else:
            pars = [ids[int(rng.integers(len(ids)))] for _ in range(C)]
            cands = [(f"i{it:04d}_{n:02d}", child_of(Rule.from_dict(members[p]["rule"]), rng, ea.mut_scale), p,
                      members[p]["root"]) for n, p in enumerate(pars)]
        evs = evaluate([c[1] for c in cands], ea)
        ok = [n for n, e in enumerate(evs) if not (ea.max_texture > 0 and e["texture"] > ea.max_texture)]
        imgs = np.stack([evs[n]["img"] for n in ok]) if ok else np.zeros((0, TPX, TPX, 3), np.uint8)
        z = clip.images(list(imgs)).astype(np.float32) if ok else np.zeros((0, max(Z.shape[1], 1)), np.float32)
        new_ids = [cands[n][0] for n in ok]
        if not len(Z):                                    # the embedding's width comes from the first batch
            Z = np.zeros((0, z.shape[1]), np.float32)
        unsel = []
        if "unselected" not in st["snaps"]:              # the first N distinct worlds, kept whether they survive or not
            ub = out / "snapshots" / "unselected_buffer.npz"
            B = np.load(ub) if ub.exists() else {"ids": np.array([], str), "Z": z[:0], "tiles": imgs[:0]}
            take = list(range(min(len(ok), N - len(B["ids"]))))
            unsel = [new_ids[i] for i in take]
            bi, bz, bt = list(B["ids"]) + unsel, np.concatenate([B["Z"], z[take]]), np.concatenate([B["tiles"], imgs[take]])
            if len(bi) >= N:
                save_npz(out / "snapshots" / "pop_unselected.npz", ids=np.array(bi), Z=bz, tiles=bt)
                st["snaps"].append("unselected")
            else:
                save_npz(ub, ids=np.array(bi), Z=bz, tiles=bt)
        if it == 0:                                      # ASAL: the population starts as N identical copies; here
            cyc = [j % len(ok) for j in range(N)] if ok else []   # copies of the valid seeds, cycled (not re-simulated)
            ids, Z, T = [new_ids[j] for j in cyc], z[cyc], imgs[cyc]
        else:
            all_ids, allZ, allT = ids + new_ids, np.concatenate([Z, z]), np.concatenate([T, imgs])
            keep = cull(allZ, N) if len(all_ids) > N else list(range(len(all_ids)))
            ids, Z, T = [all_ids[i] for i in keep], allZ[keep], allT[keep]
        kept, unsel = set(ids), set(unsel)
        for n, (cid, rule, par, root) in enumerate(cands):
            e = evs[n]
            row = {"id": cid, "it": it, "parent": par, "root": root, "texture": round(e["texture"], 4),
                   "motion": round(e["motion"], 4), "bodies": e["bodies"], "coverage": round(e["coverage"], 4)}
            log.writerow([it, cid, par or "", root, row["texture"], row["motion"], row["bodies"], row["coverage"],
                          int(n not in ok), int(cid in kept)])
            if cid in kept or cid in unsel:              # rules of every world a snapshot can contain
                members[cid] = {**row, "rule": rule.to_dict()}
                mf.write(json.dumps(members[cid]) + "\n")
        mf.flush()
        lf.flush()
        st["sims"] += len(cands)
        for s in snaps:
            if st["sims"] >= s and str(s) not in st["snaps"]:
                save_npz(out / "snapshots" / f"pop_{s}.npz", ids=np.array(ids), Z=Z, tiles=T)
                st["snaps"].append(str(s))
        h = {"it": it, "sims": st["sims"], "pop": len(ids), "children": len(cands), "grid_noise": len(cands) - len(ok),
             "kept": sum(c[0] in kept for c in cands), "asal_loss": round(asal_loss(Z), 5) if len(ids) > 1 else None,
             "lineages": len({members[i]["root"] for i in ids}), "sec": round(time.time() - t0, 1)}
        st["history"].append(h)
        st["it"] = it + 1
        save_npz(out / "pop.npz", ids=np.array(ids), Z=Z, tiles=T)
        tmp = out / "state.tmp.json"
        tmp.write_text(json.dumps(st))
        tmp.replace(out / "state.json")
        print(f"it {it}: {st['sims']:,}/{budget:,} simulations, population {len(ids)}, {h['kept']}/{len(cands)} kept, "
              f"{h['grid_noise']} grid-noise, ASAL diversity loss {h['asal_loss']}, {h['lineages']} seed lineages, "
              f"{h['sec']:.0f}s", file=sys.stderr, flush=True)
    mf.close()
    lf.close()
    progress(a)


def progress(a) -> None:
    out = Path(a.out)
    st = json.loads((out / "state.json").read_text())
    H = st["history"]
    L = [f"# ASAL illumination on MaCE-Lenia: {out.name}\n",
         f"Settings from {st['like']}; population {st['args']['pop']}, {st['args']['children']} children per "
         f"iteration; {st['sims']:,} of {st['args']['budget']:,} simulations; snapshots: {', '.join(st['snaps'])}.\n",
         "ASAL diversity loss = Eq. 4: mean highest CLIP similarity to another member (lower = more diverse).\n",
         "| iteration | simulations | population | children kept | grid-noise | ASAL diversity loss | seed lineages |",
         "|---|---|---|---|---|---|---|"]
    step = max(1, len(H) // 25)
    for h in H[::step] + ([H[-1]] if (len(H) - 1) % step else []):
        L.append(f"| {h['it']} | {h['sims']:,} | {h['pop']} | {h['kept']}/{h['children']} | {h['grid_noise']} | "
                 f"{h['asal_loss']} | {h['lineages']} |")
    (out / "progress.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- atlas (ASAL's figure 6) ----------

def atlas(a) -> None:
    out = Path(a.out)
    names = snap_names(out)
    name = a.snap or names[-1]
    S = load_set(out, name)
    Z, T, ids = S["Z"], S["tiles"], S["ids"]
    try:
        from sklearn.manifold import TSNE
        xy = TSNE(2, perplexity=min(30, max(2, len(Z) // 4)), init="pca", random_state=0).fit_transform(Z)
        how = "t-SNE of CLIP"
    except ImportError:
        xy = (Z - Z.mean(0)) @ np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][:2].T
        how = "PCA of CLIP"
    xy = (xy - xy.min(0)) / np.maximum(np.ptp(xy, 0), 1e-9)
    G, px = a.grid, 120
    im = Image.new("RGB", (G * px, G * px + 24), (12, 13, 20))
    d = ImageDraw.Draw(im)
    shown = 0
    for gy in range(G):
        for gx in range(G):
            c = np.array([(gx + 0.5) / G, (gy + 0.5) / G])
            inside = np.where((np.floor(xy[:, 0] * G * 0.9999) == gx) & (np.floor(xy[:, 1] * G * 0.9999) == gy))[0]
            if len(inside):
                j = inside[np.argmin(np.linalg.norm(xy[inside] - c, axis=1))]
                im.paste(Image.fromarray(T[j]).resize((px - 2, px - 2)), (gx * px + 1, gy * px + 25))
                shown += 1
    d.text((6, 6), f"{out.name}, population at {name} simulations: {len(ids)} worlds ({how}; nearest world per tile, "
                   f"{shown} tiles filled)", fill=(225, 228, 235))
    p1 = out / f"atlas_{name}.png"
    im.save(p1)
    cols, px = 16, 112                                    # every world, numbered, for picking
    rows = (len(ids) + cols - 1) // cols
    sh = Image.new("RGB", (cols * px, rows * (px + 14)), (12, 13, 20))
    d = ImageDraw.Draw(sh)
    order = np.lexsort((xy[:, 0], np.floor(xy[:, 1] * rows)))
    for k, j in enumerate(order):
        x, y = (k % cols) * px, (k // cols) * (px + 14)
        sh.paste(Image.fromarray(T[j]).resize((px - 2, px - 2)), (x + 1, y + 14))
        d.text((x + 3, y + 1), str(k), fill=(200, 205, 215))
    p2 = out / f"sheet_{name}.png"
    sh.save(p2)
    (out / f"sheet_{name}.json").write_text(json.dumps({str(k): ids[int(j)] for k, j in enumerate(order)}, indent=0))
    print(f"{p1}\n{p2} (numbers -> ids in {p2.with_suffix('.json').name})")


# ---------- our side: the seed-1 full run's worlds ----------

def ours(full: Path) -> dict:
    """The full run's final pool: map elites (final), surprise archive, finished expedition ends. id -> world."""
    st = json.loads((full / "map" / "state.json").read_text())
    W = {}
    for k, m in enumerate(QD.islands_of(st)):
        for ck, e in m.items():
            W[f"map:{Path(e['image']).stem}"] = {"set": "map", "rule": e["rule"], "root": e.get("root"),
                                                  "still": f"map/img/{e['image']}"}
    for e in st.get("sarch", []):
        W.setdefault(f"map:{Path(e['image']).stem}", {"set": "sarch", "rule": e["rule"], "root": e.get("root"),
                                                     "still": f"map/img/{e['image']}"})
    for Rr in sorted((full / "rounds").glob("r*")):
        f = Rr / "exp" / "state.json"
        if not f.exists():
            continue
        est = json.loads(f.read_text())
        if est["step"] < est["end"]:
            continue
        for tr in est["triggers"]:
            if tr.get("history"):
                W[f"end:{tr['id']}"] = {"set": "end", "rule": tr["parent_rule"], "root": tr.get("root"),
                                        "still": tr.get("parent")}
    seen, out = set(), {}
    for k, w in W.items():                                 # an end fed back into the map is the same world: once
        key = json.dumps(w["rule"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out[k] = w
    return out


def our_spend(full: Path) -> str:
    rep = full / "report.md"
    if rep.exists():
        for line in rep.read_text().splitlines():
            if line.startswith("Spend:"):
                return line[len("Spend:"):].strip().rstrip(".")
    return "unknown"


def regrow(rules: list, ea) -> list:
    return evaluate([Rule.from_dict(r) for r in rules], ea, pics=True)


def replay_diff(full: Path, still: str | None, pic: np.ndarray) -> float | None:
    if not still or not (full / still).exists():
        return None
    ref = np.asarray(Image.open(full / still).convert("RGB"), np.int16)
    return float(np.abs(ref - pic.astype(np.int16)).mean()) if ref.shape == pic.shape else float("inf")


# ---------- stats: numbers neither method optimised ----------

def vendi(X: np.ndarray) -> float:
    """Vendi score (Friedman & Dieng 2023) with the cosine kernel: exp(entropy of the eigenvalues of K/n)."""
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    w = np.linalg.eigvalsh(X @ X.T / len(X))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def cosd(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return 1 - X @ Y.T


def nn_within(X: np.ndarray) -> np.ndarray:
    D = cosd(X, X)
    np.fill_diagonal(D, np.inf)
    return D.min(1)


def sub_stat(f, X: np.ndarray, n: int, rng, draws: int) -> tuple:
    """f on n-member subsamples (without replacement): (value at n from the first draw's mean, 2.5%, 97.5%)."""
    v = [f(X[rng.choice(len(X), n, replace=False)]) for _ in range(draws)]
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def coverage_of(X: np.ndarray, Y: np.ndarray) -> float:
    """Share of Y's worlds whose nearest X world is closer than X's own median nearest-neighbour distance."""
    return float((cosd(Y, X).min(1) <= np.median(nn_within(X))).mean())


def classifier(X: np.ndarray, Y: np.ndarray, rng, perms: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n = min(len(X), len(Y))
    F = np.concatenate([X[rng.choice(len(X), n, replace=False)], Y[rng.choice(len(Y), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = float(cross_val_score(clf, F, y, cv=cv).mean())
    null = [float(cross_val_score(clf, F, rng.permutation(y), cv=cv).mean()) for _ in range(perms)]
    return {"n_each": n, "accuracy": round(acc, 3), "p": round((1 + sum(x >= acc for x in null)) / (1 + perms), 3),
            "null_mean": round(float(np.mean(null)), 3) if null else None}


def oe_late(rules: list, ea, clip, steps: int, every: int, batch: int) -> list:
    """ASAL's open-endedness score over the late half of a long run (oecheck.py's scoring). Novelty = 1 - loss."""
    import oecheck as OC
    s = {"size": ea.size, "R": ea.R, "start_seed": ea.start_seed}
    times = list(range(0, steps + 1, every))
    tt = np.array(times)
    out = []
    for film in OC.grow([Rule.from_dict(r) for r in rules], s, times, steps, batch):
        scale = max(float(np.percentile(film[tt >= min(1500, steps // 2)], 99.5)), 1e-6)
        m = tt >= steps // 2
        z = clip.images([render.colorize(np.clip(f / scale, 0, 1)) for f in film[m]])
        out.append(round(1 - OC.asal_oe(z), 5))
    return out


def stats(a) -> None:
    out, full = Path(a.out), Path(a.full)
    st = json.loads((out / "state.json").read_text())
    ea = ea_of(st)
    for k in ("size", "warmup", "render", "gap", "sim_batch"):        # dry runs only
        if getattr(a, k, None):
            setattr(ea, k, getattr(a, k))
    backend.configure(a.backend)
    rng = np.random.default_rng([ea.seed, 1117])
    members = read_members(out)
    names = snap_names(out)
    fin = str(a.snap or max(int(x) for x in names if x != "unselected"))
    mid = str(a.mid or min((int(x) for x in names if x != "unselected"), key=lambda s: abs(s - 6464)))
    sets = {"A": load_set(out, fin), "A@map": load_set(out, mid), "R": load_set(out, "unselected")}
    for S in sets.values():
        S["desc"] = [members[i] for i in S["ids"]]
        S["roots"] = [members[i]["root"] for i in S["ids"]]
    print(f"A = ASAL at {fin} simulations, A@map = at {mid}, R = unselected; regrowing our worlds and the seeds",
          file=sys.stderr)
    W = ours(full)
    seeds_ = [r.to_dict() for r in seed_rules(ea)]
    keys = list(W)
    evs = regrow([W[k]["rule"] for k in keys] + seeds_, ea)
    ev_w, ev_s = evs[:len(keys)], evs[len(keys):]
    replay = {k: replay_diff(full, W[k]["still"], e["pic"]) for k, e in zip(keys, ev_w)}
    rd = [v for v in replay.values() if v is not None]
    pick = lambda which: [n for n, k in enumerate(keys) if W[k]["set"] in which]
    for nm, which in (("B", ("map", "sarch", "end")), ("B@map", ("map",))):
        ns = pick(which)
        sets[nm] = {"ids": [keys[n] for n in ns], "tiles": np.stack([ev_w[n]["img"] for n in ns]),
                    "desc": [ev_w[n] for n in ns], "roots": [W[keys[n]]["root"] for n in ns]}
    seedT = np.stack([e["img"] for e in ev_s if not (ea.max_texture > 0 and e["texture"] > ea.max_texture)])
    from lenia.embed import ClipScorer, Embedder
    dino = Embedder(a.embed)
    clip = ClipScorer("fake" if a.clip == "fake" else CLIP_MODEL)
    for S in sets.values():
        ims = [Image.fromarray(t) for t in S["tiles"]]
        S["D"] = dino(ims)
        S["C"] = clip.images(list(S["tiles"]))
    seedD = dino([Image.fromarray(t) for t in seedT])
    seed_nn = nn_within(seedD)
    res = {"sets": {k: len(v["ids"]) for k, v in sets.items()}, "snapshots": {"A": fin, "A@map": mid},
           "replay": {"compared": len(rd), "median": round(float(np.median(rd)), 2) if rd else None,
                      "above_tol": sum(v > S1.REPLAY_TOL for v in rd), "tol": S1.REPLAY_TOL},
           "spend_ours": our_spend(full), "per_set": {}, "pairs": {}}
    for nm, S in sets.items():
        D, d = S["D"], S["desc"]
        mot = np.array([x["motion"] for x in d])
        cov = np.array([x["coverage"] for x in d])
        res["per_set"][nm] = {
            "n": len(D), "lineages": len({r for r in S["roots"] if r is not None}),
            "dist_to_seeds_median": round(float(np.median(cosd(D, seedD).min(1))), 4),
            "beyond_seed_p90": round(float((cosd(D, seedD).min(1) > np.percentile(seed_nn, 90)).mean()), 3),
            "motion_median": round(float(np.median(mot)), 3), "static_share": round(float((mot < 0.5).mean()), 3),
            "bodies_median": float(np.median([x["bodies"] for x in d])),
            "near_empty_share": round(float(np.mean([x["bodies"] == 0 for x in d])), 3),
            "coverage_median": round(float(np.median(cov)), 3), "world_filling_share": round(float((cov > 0.6).mean()), 3),
            "texture_median": round(float(np.median([x["texture"] for x in d])), 4)}
    for x, y in (("A", "B"), ("A@map", "B@map"), ("A", "R"), ("B", "R")):
        X, Y = sets[x], sets[y]
        n = min(len(X["D"]), len(Y["D"]))
        m = max(2, int(0.8 * n))
        r = {"n": n}
        for emb in ("D", "C"):
            tag = "dino" if emb == "D" else "clip"
            for nm_, S in ((x, X), (y, Y)):
                r[f"{nm_}:vendi_{tag}"] = [round(v, 2) for v in sub_stat(vendi, S[emb], m, rng, a.draws)]
                r[f"{nm_}:nn_{tag}"] = [round(v, 4) for v in sub_stat(lambda Q: float(np.median(nn_within(Q))), S[emb], m, rng,
                                                                        a.draws)]
        cxy, cyx = [], []
        for _ in range(a.draws):
            Xs, Ys = X["D"][rng.choice(len(X["D"]), m, replace=False)], Y["D"][rng.choice(len(Y["D"]), m, replace=False)]
            cxy.append(coverage_of(Xs, Ys))
            cyx.append(coverage_of(Ys, Xs))
        r[f"{y} covered by {x}"] = [round(float(np.mean(cxy)), 3), round(float(np.percentile(cxy, 2.5)), 3),
                                    round(float(np.percentile(cxy, 97.5)), 3)]
        r[f"{x} covered by {y}"] = [round(float(np.mean(cyx)), 3), round(float(np.percentile(cyx, 2.5)), 3),
                                    round(float(np.percentile(cyx, 97.5)), 3)]
        try:
            r["classifier"] = classifier(X["D"], Y["D"], rng, a.perms)
        except ImportError:
            r["classifier"] = None
        res["pairs"][f"{x} vs {y}"] = r
    try:                                                     # shared clusters over A, B, R together
        from sklearn.cluster import KMeans
        U = np.concatenate([sets[k]["D"] for k in ("A", "B", "R")])
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        lab = KMeans(a.k, n_init=10, random_state=0).fit_predict(U)
        pos, cl = 0, {}
        for k in ("A", "B", "R"):
            n = len(sets[k]["D"])
            c = Counter(lab[pos:pos + n])
            p = np.array(list(c.values())) / n
            cl[k] = {"clusters_reached": len(c), "effective_clusters": round(float(np.exp(-(p * np.log(p)).sum())), 2)}
            pos += n
        res["clusters"] = {"k": a.k, **cl}
    except ImportError:
        res["clusters"] = None
    res["efficiency"] = {}                                    # ASAL's Vendi (DINOv2) along the run, our two stopping points
    nmin = min(len(load_set(out, s)["ids"]) for s in names)
    nb = min(nmin, len(sets["B@map"]["D"]))
    for s in names:
        T = load_set(out, s)["tiles"]
        res["efficiency"][f"A@{s}"] = round(sub_stat(vendi, dino([Image.fromarray(t) for t in T]), nb, rng, a.draws)[0], 2)
    for nm in ("B@map", "B"):
        res["efficiency"][nm] = round(sub_stat(vendi, sets[nm]["D"], nb, rng, a.draws)[0], 2)
    res["efficiency"]["n"] = nb
    L = {g["id"]: g for g in csv.DictReader(open(out / "log.csv"))}          # a repeated iteration logs its rows twice
    res["grid_noise_rate"] = {"A": round(float(np.mean([int(x["grid_noise"]) for x in L.values()])), 4)}
    pr = json.loads((full / "map" / "state.json").read_text())["progress"]
    res["grid_noise_rate"]["B@map"] = round(sum(x.get("texture_fail", 0) for x in pr) / (64 + 64 * (len(pr) - 1)), 4)
    if a.oe_sample:
        print(f"open-endedness: growing {a.oe_sample} worlds per set to {a.oe_steps:,} steps", file=sys.stderr)
        res["oe_late"] = {}
        for nm, rules in (("A", [members[i]["rule"] for i in sets["A"]["ids"]]),
                          ("B", [W[k]["rule"] for k in sets["B"]["ids"]]),
                          ("R", [members[i]["rule"] for i in sets["R"]["ids"]])):
            sel = [rules[int(j)] for j in rng.choice(len(rules), min(a.oe_sample, len(rules)), replace=False)]
            v = oe_late(sel, ea, clip, a.oe_steps, a.oe_every, 36)          # 36 long films fit in Colab's memory
            res["oe_late"][nm] = {"n": len(v), "median": round(float(np.median(v)), 4),
                                  "q25": round(float(np.percentile(v, 25)), 4), "q75": round(float(np.percentile(v, 75)), 4)}
    (out / "stats.json").write_text(json.dumps(res, indent=1))
    stats_md(out, res)


def stats_md(out: Path, r: dict) -> None:
    f = lambda v: f"{v[0]} ({v[1]}–{v[2]})"
    L = [f"# ASAL illumination vs the seed-1 full run: numbers\n",
         f"Sets: {', '.join(f'{k} {v}' for k, v in r['sets'].items())}. A = ASAL's population at {r['snapshots']['A']} "
         f"simulations (the full run's budget), A@map = at {r['snapshots']['A@map']} (the map's budget alone), R = the "
         "first worlds before any selection (seeds and unselected mutants). B = the full run's final pool (map elites, "
         "surprise archive, expedition ends), B@map = its map elites only.\n",
         "Rule: no method is scored by what it optimised. CLIP numbers favour A by construction and are for reference; "
         "DINOv2 is held out. Set-level numbers are on equal-size subsamples (80% of the smaller set), "
         "mean (95% range over draws): spread within one run, not between runs.\n",
         f"Our worlds were regrown for these numbers: {r['replay']['compared']} compared with their stored pictures, "
         f"median difference {r['replay']['median']}, {r['replay']['above_tol']} above the replay tolerance "
         f"({r['replay']['tol']}). Our spend: {r['spend_ours']}. ASAL's: none.\n",
         "## Rehash, diversity\n"]
    for pk, p in r["pairs"].items():
        x, y = pk.split(" vs ")
        L += [f"### {pk} (n = {p['n']})\n", "| | " + x + " | " + y + " |", "|---|---|---|"]
        for tag, lab in (("dino", "Vendi, DINOv2 (effective distinct worlds)"),
                         ("clip", "Vendi, CLIP (reference)")):
            L.append(f"| {lab} | {f(p[f'{x}:vendi_{tag}'])} | {f(p[f'{y}:vendi_{tag}'])} |")
        L.append(f"| median nearest-neighbour distance, DINOv2 | {f(p[f'{x}:nn_dino'])} | {f(p[f'{y}:nn_dino'])} |")
        L.append(f"| median nearest-neighbour distance, CLIP (reference) | {f(p[f'{x}:nn_clip'])} | {f(p[f'{y}:nn_clip'])} |")
        L.append(f"| share covered by the other set | {f(p[f'{x} covered by {y}'])} | {f(p[f'{y} covered by {x}'])} |")
        c = p.get("classifier")
        if c:
            L.append(f"\nCan a linear classifier on DINOv2 tell them apart? accuracy {c['accuracy']} (chance 0.5; "
                     f"shuffled labels {c['null_mean']}), permutation p = {c['p']}, {c['n_each']} each.\n")
    L.append("Covered = the world's nearest neighbour in the other set is closer than that set's own median "
             "nearest-neighbour distance. Our set mostly covered by A, but not the reverse, means ours is a subset of "
             "what ASAL finds.\n")
    if r.get("clusters"):
        c = r["clusters"]
        L += [f"**Shared clusters** (k-means, k = {c['k']}, on A, B and R together):"] + \
             [f"- {k}: reaches {c[k]['clusters_reached']}, effective {c[k]['effective_clusters']}" for k in ("A", "B", "R")]
    L += ["\n## Per set\n", "| | " + " | ".join(r["per_set"]) + " |", "|---|" + "---|" * len(r["per_set"])]
    rows = [("seed lineages", "lineages"), ("median DINOv2 distance to the nearest seed", "dist_to_seeds_median"),
            ("share beyond the seeds' own p90 spacing", "beyond_seed_p90"), ("median motion", "motion_median"),
            ("static (motion < 0.5)", "static_share"), ("median bodies", "bodies_median"),
            ("near-empty (no body)", "near_empty_share"), ("median coverage", "coverage_median"),
            ("world-filling (coverage > 0.6)", "world_filling_share"), ("median texture", "texture_median")]
    for lab, k in rows:
        L.append(f"| {lab} | " + " | ".join(str(v[k]) for v in r["per_set"].values()) + " |")
    L.append("\nMotion = |last - first snapshot| / mass over the 1,200 filmed steps (the map's axis); it agreed with "
             "Felix's interest choices 72% of the time in the seed-1 audit. A proxy for his taste, not ground truth.")
    L.append(f"\nGrid-noise rejections during search: " + ", ".join(f"{k} {v:.1%}" for k, v in r["grid_noise_rate"].items()))
    e = r["efficiency"]
    L += [f"\n## Diversity against simulations spent (Vendi, DINOv2, n = {e['n']})\n",
          ", ".join(f"{k} {v}" for k, v in e.items() if k != "n")]
    if r.get("oe_late"):
        L += ["\n## ASAL's open-endedness score, late half of a long run (higher = keeps looking new)\n",
              "| set | n | median | 25%–75% |", "|---|---|---|---|"]
        for k, v in r["oe_late"].items():
            L.append(f"| {k} | {v['n']} | {v['median']} | {v['q25']}–{v['q75']} |")
    (out / "stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- the blind session ----------

DESIGN = """# ASAL baseline rating session: design (fixed before rating, 11-17)

Pairs: {counts}. Each pair is one random world from each set, sampled without replacement; the question is the
seed-1 rating session's interest question. a = the ASAL world.
- asal-vs-ours: A (ASAL at {fin} simulations) vs B (the full run's final pool: map elites, surprise archive,
  expedition ends). Decision rule: a clear win either way if the sign test on decided pairs gives two-sided p < 0.05
  (21 of 30 decided); anything closer is no difference.
- asal-vs-unselected: A vs R (the first worlds before any selection). Does ASAL's selection do anything on MaCE?
All worlds are regrown as films; replay.json lists how far each regrown final frame is from the one stored.
"""


def build(a) -> None:
    out, full = Path(a.out), Path(a.full)
    R_ = Path(a.dir) if a.dir else out / "rate"
    if (R_ / "ratings.csv").exists():
        sys.exit(f"{R_ / 'ratings.csv'} exists: not rebuilding")
    if (R_ / "tasks.json").exists() and not a.force:
        sys.exit(f"{R_ / 'tasks.json'} exists (--force rebuilds it, only before any rating)")
    st = json.loads((out / "state.json").read_text())
    ea = ea_of(st)
    for k in ("size", "warmup", "render", "gap", "sim_batch"):
        if getattr(a, k, None):
            setattr(ea, k, getattr(a, k))
    backend.configure(a.backend)
    rng = np.random.default_rng([ea.seed, 1170])
    members = read_members(out)
    names = snap_names(out)
    fin = str(a.snap or max(int(x) for x in names if x != "unselected"))
    A = load_set(out, fin)
    U = load_set(out, "unselected")
    W = ours(full)
    na, nr = a.ours + a.unselected, a.unselected
    ai = [A["ids"][int(j)] for j in rng.choice(len(A["ids"]), min(na, len(A["ids"])), replace=False)]
    rule_key = lambda r: json.dumps(r, sort_keys=True)
    a_keys = {rule_key(members[i]["rule"]) for i in ai}
    r_pool = [i for i in U["ids"] if rule_key(members[i]["rule"]) not in a_keys]
    ri = [r_pool[int(j)] for j in rng.choice(len(r_pool), min(nr, len(r_pool)), replace=False)]
    bk = list(W)
    bi = [bk[int(j)] for j in rng.choice(len(bk), min(a.ours, len(bk)), replace=False)]
    Wd = {}
    for i in ai:
        Wd[f"asal:{i}"] = {"rule": members[i]["rule"], "tile": A["tiles"][A["ids"].index(i)]}
    for i in ri:
        Wd[f"unsel:{i}"] = {"rule": members[i]["rule"], "tile": U["tiles"][U["ids"].index(i)]}
    for k in bi:
        Wd[f"ours:{k}"] = {"rule": W[k]["rule"], "still": W[k]["still"]}
    tasks = [{"group": "interest:asal-vs-ours", "question": "interest", "a": f"asal:{x}", "b": f"ours:{y}"}
             for x, y in zip(ai[:len(bi)], bi)]
    tasks += [{"group": "interest:asal-vs-unselected", "question": "interest", "a": f"asal:{x}", "b": f"unsel:{y}"}
              for x, y in zip(ai[len(bi):], ri)]
    names_ = sorted(Wd)
    print(f"{len(tasks)} pairs over {len(names_)} worlds; growing films", file=sys.stderr)
    res = S1.v4_films([Rule.from_dict(Wd[n]["rule"]) for n in names_], ea, a.every)
    (R_ / "images").mkdir(parents=True, exist_ok=True)
    salt = int(rng.integers(1 << 30))
    replay = {}
    for n, (frames, pic, snaps) in zip(names_, res):
        h = hashlib.sha1(f"{n}|{salt}".encode()).hexdigest()[:12]
        Wd[n]["file"] = h
        if "tile" in Wd[n]:
            replay[n] = round(float(np.abs(final_img(snaps[-1]).astype(np.int16) - Wd[n]["tile"].astype(np.int16)).mean()), 2)
        else:
            d = replay_diff(full, Wd[n]["still"], pic)
            if d is not None:
                replay[n] = round(d, 2)
        Image.fromarray(pic).save(R_ / "images" / f"{h}.png")
        S1._gif(frames, snaps, R_ / "images" / f"{h}.gif", a.px, a.fps)
    out_t = []
    for k in rng.permutation(len(tasks)):
        t = tasks[int(k)]
        l, r = (t["a"], t["b"]) if rng.random() < 0.5 else (t["b"], t["a"])
        out_t.append({"id": len(out_t), **t, "sees": None, "push": None, "left": Wd[l]["file"], "right": Wd[r]["file"],
                      "left_world": l, "right_world": r})
    (R_ / "tasks.json").write_text(json.dumps(out_t, indent=1))
    bad = {k: v for k, v in replay.items() if v > S1.REPLAY_TOL}
    (R_ / "replay.json").write_text(json.dumps({"tol": S1.REPLAY_TOL, "diff": replay, "diverged": bad}, indent=1))
    (R_ / "design.md").write_text(DESIGN.format(counts=dict(Counter(t["group"] for t in out_t)), fin=fin))
    print(f"wrote {R_}: {len(out_t)} pairs; replay: {len(replay)} compared, {len(bad)} above tolerance (films are "
          "always the regrown world)", file=sys.stderr)


def rate(a) -> None:
    from lenia.rating import rate_pairs
    rate_pairs(Path(a.dir) if a.dir else Path(a.out) / "rate", port=a.port)


def two_sided(k: int, n: int) -> float:
    return min(1.0, 2 * min(S1._binom_greater(k, n), S1._binom_greater(n - k, n))) if n else float("nan")


def report(a) -> None:
    R_ = Path(a.dir) if a.dir else Path(a.out) / "rate"
    tasks = {t["id"]: t for t in json.loads((R_ / "tasks.json").read_text())}
    if not (R_ / "ratings.csv").exists():
        sys.exit("no ratings yet")
    ans = {int(r["id"]): r["answer"] for r in csv.DictReader(open(R_ / "ratings.csv"))}
    by = {}
    for i, x in ans.items():
        t = tasks[i]
        v = None if x == "neither" else ((t["left_world"] if x == "left" else t["right_world"]) == t["a"])
        by.setdefault(t["group"], []).append(v)
    L = [f"# ASAL baseline rating session ({len(ans)} of {len(tasks)} pairs rated)\n",
         "Win = the ASAL world chosen. Decision rule (fixed before rating): two-sided sign test on decided pairs, "
         "p < 0.05.\n", "| group | ASAL wins | other wins | neither | ASAL share (decided) | two-sided p | verdict |",
         "|---|---|---|---|---|---|---|"]
    for g in sorted(by):
        v = by[g]
        w, l_ = sum(x is True for x in v), sum(x is False for x in v)
        p = two_sided(w, w + l_)
        verdict = "no difference" if not p < 0.05 else ("ASAL better" if w > l_ else "ASAL worse")
        L.append(f"| {g} | {w} | {l_} | {sum(x is None for x in v)} | {w / max(1, w + l_):.2f} | {p:.3f} | {verdict} |")
    sides = list(ans.values())
    L.append(f"\nSides: {sides.count('left')} left, {sides.count('right')} right, {sides.count('neither')} neither.")
    (R_ / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "progress", "atlas", "stats", "build", "rate", "report"):
        sub.add_parser(c).add_argument("--out", required=True, help="the ASAL run's folder")
    r = sub.choices["run"]
    r.add_argument("--like", default=None, help="the map run whose settings and seeds to use (the full run's map/)")
    r.add_argument("--backend", default="numpy")
    r.add_argument("--clip", default="openai", help="openai (ASAL's ViT-B/32) or fake (dry runs)")
    r.add_argument("--pop", type=int, default=256, help="population (ASAL's repo default)")
    r.add_argument("--children", type=int, default=32, help="children per iteration (ASAL's repo default)")
    r.add_argument("--budget", type=int, default=None, help=f"simulations, seeds included (default {FULL_SIMS:,}; "
                   "raise it on a resume to continue the run)")
    r.add_argument("--snapshots", default=None, help="simulation counts at which the population is saved "
                   "(default 1344,6464,30464)")
    for k in ("size", "warmup", "render", "gap", "init"):
        r.add_argument(f"--{k}", type=int, default=None, help="override (offline dry runs only)")
    r.add_argument("--R", type=float, default=None, help="override (offline dry runs only)")
    r.add_argument("--sim-batch", type=int, default=None, help="worlds simulated together (e.g. 128 on an L4 or A100)")
    t = sub.choices["atlas"]
    t.add_argument("--snap", default=None, help="snapshot name (default: the last)")
    t.add_argument("--grid", type=int, default=12)
    s = sub.choices["stats"]
    s.add_argument("--full", required=True, help="the seed-1 full run's folder")
    s.add_argument("--backend", default="numpy")
    s.add_argument("--snap", default=None, help="A's snapshot (default: the largest)")
    s.add_argument("--mid", default=None, help="A@map's snapshot (default: the one nearest 6,464)")
    s.add_argument("--embed", default="dinov2_vits14", help="held-out embedder (fake for dry runs)")
    s.add_argument("--clip", default="openai")
    s.add_argument("--draws", type=int, default=200)
    s.add_argument("--perms", type=int, default=200)
    s.add_argument("--k", type=int, default=20, help="shared clusters")
    s.add_argument("--oe-sample", type=int, default=24, help="worlds per set for the open-endedness score (0 = skip)")
    s.add_argument("--oe-steps", type=int, default=27000)
    s.add_argument("--oe-every", type=int, default=250)
    b = sub.choices["build"]
    b.add_argument("--full", required=True)
    b.add_argument("--backend", default="numpy")
    b.add_argument("--snap", default=None)
    b.add_argument("--ours", type=int, default=30, help="ASAL vs ours pairs")
    b.add_argument("--unselected", type=int, default=10, help="ASAL vs unselected pairs")
    b.add_argument("--every", type=int, default=30, help="simulation steps per film frame")
    b.add_argument("--px", type=int, default=280)
    b.add_argument("--fps", type=int, default=15)
    b.add_argument("--force", action="store_true")
    for c in ("stats", "build"):
        for k in ("size", "warmup", "render", "gap", "sim_batch"):
            sub.choices[c].add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override (dry runs only)")
    for c in ("build", "rate", "report"):
        sub.choices[c].add_argument("--dir", default=None, help="the session folder (default <out>/rate)")
    sub.choices["rate"].add_argument("--port", type=int, default=8775)
    a = ap.parse_args()
    {"run": run, "progress": progress, "atlas": atlas, "stats": stats, "build": build, "rate": rate,
     "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
