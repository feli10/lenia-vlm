"""ASAL's supervised target search on MaCE-Lenia, with frames of a real GIF as the targets (11-24): the baseline for
the GIF-mimicry project (project doc "Target candidates - real biological GIFs"; plan agreed 11-21).

In three sentences: ASAL's target search (Kumar et al. 2024, Eq. 2) asks Sep-CMA-ES for genomes that make a world
whose K evenly spaced frames match K targets in CLIP space. Here the targets are K keyframes of a GIF (ASAL used text
prompts), the world is MaCE-Lenia, and everything else is copied from github.com/SakanaAI/asal (main_opt.py,
asal_metrics.py, rollout.py, substrates/lenia.py; commit 677ba0e, read 11-24) and from evosax 0.1.6, the version ASAL
pins. A random-search control (ASAL's own random-genome sampler) runs alongside on the same budget.

  python scripts/asal_target.py run  --out RUNS/asal-target-urchin --target "urchin=RUNS/targets/urchin_2to4.gif" \\
         --center RUNS/asal-illum-s1:i0004_09 --backend auto                     # Colab, ~6-8 h on a T4 (resumes)
  python scripts/asal_target.py eval --out RUNS/asal-target-urchin --backend auto  # Colab, ~10 min
  python scripts/asal_target.py escheck                                          # Colab: our Sep-CMA-ES vs evosax 0.1.6
  Offline dry run (no PyTorch): run --out ~/at-dry --target "urchin=RUNS/targets/urchin_2to4.gif"
      --center RUNS/asal-illum-s1:i0004_09 --fm fake --size 64 --steps 60 --pop 4 --iters 3 --random 4 --ckpt 2
      then eval --out ~/at-dry --fm fake

COPIED FROM ASAL (main_opt.py defaults unless said otherwise):
  objective   asal_metrics.calc_supervised_target_score: loss = -mean_k <z_target_k, z_world_k>, unit CLIP vectors,
              with image targets in place of text prompts (z_txt -> CLIP image embeddings of the keyframes).
  frames      rollout.py time_sampling=(K, True): K states at the ends of K equal chunks (t = T/K, 2T/K, ..., T),
              rendered at 224 px. coef_softmax = coef_oe = 0 (their defaults), bs = 1 (MaCE has no randomness).
  CLIP        foundation_models/clip.py: openai/clip-vit-base-patch32, image squashed to 224x224 (bilinear, no
              centre crop), normalised with CLIP's mean and std, image features unit-normalised. Theirs is the Flax
              model; this is the PyTorch model with the same weights.
  genome      substrates/lenia.py: the dynamics (growth centre m, width s, weight h of every kernel) and the starting
              pattern, each as an offset in inverse-sigmoid space from a known world: value = sigmoid(logit(base) +
              offset), base clipped to [1e-6, 1-1e-6]; dynamics offsets clipped to [-1, 1] (clip1 = 1), pattern
              offsets unclipped (clip2 = inf). Kernel shapes (rings b, radius r), R, T are not in the genome. As
              in ASAL, cells at exactly 0 sit at logit -13.8 and stay empty at sigma 0.1, so the pattern can
              reshape but not spread.
  optimiser   evosax 0.1.6 Sep_CMA_ES(popsize=16, sigma_init=0.1), mean starting at offset 0 (the known world),
              1,000 iterations (16,000 simulations); the result is es_state.best_member, the best genome ever
              evaluated. Ported to numpy line for line (class SepCMAES; `escheck` compares it with evosax). Port note:
              in 0.1.6, tell() never recomputes D from C (diag_eigen_decomp returns the old D), so sampling stays
              isotropic with only the global step size adapting. That is what ASAL ran, so it is kept.
  random      substrate.default_params: offsets ~ N(0, 0.1^2) per dimension, i.e. ASAL's random genome.

CHANGED FOR MACE, AND WHY:
  substrate   MaCE-Lenia (9 kernels, 3 channels, 160 x 160, R 13) instead of LeniaBreeder Lenia: the domain of the
              project. Genome: 27 dynamics numbers (m, s, h x 9) + the 39 x 39 x 3 starting patch (4,590 numbers;
              ASAL's Lenia has 45 + 32 x 32 x 3 = 3,117). MaCE's beta, like ASAL's T, is held fixed.
  known world ASAL starts from Chan's 5N7KKM creature. Ours: --center, a world from the illumination run; default
              i0004_09, the closest CLIP match to the urchin clip among 512 (doc "Embedding check on first targets",
              11-24), with the fixed start every MaCE world grows from (smooth noise, seed 0, 3R wide).
  rollout     --steps 2,700 (our standard 1,500 + 1,200) instead of their Lenia's 256: MaCE moves about one cell per
              step, and divisions in our runs take hundreds of steps. Substrate default, as in ASAL.
  render      ASAL shows a 64 x 64 window of the 128 world centred on the centre of mass, channels as RGB, nearest-
              upscaled to 224. Same here with an 80 x 80 window (half our 160 world), centred by circular mean on
              the torus; channels as RGB (lenia.render.colorize), scaled by the window's 99.5th percentile because
              MaCE values exceed 1 (Chan Lenia is clipped to [0, 1]).
  targets     --keyframes K (default 6) GIF frames, evenly spaced over the clip (first and last included).
OURS (not in ASAL): the random control run in the same batches, the eval command (held-out DINOv2 and pixel similarity,
an order check, persistence past the clip, films).

Outputs in --out: state.json (settings, progress), es.npz (optimiser state), log.csv (one row per iteration),
best.npz / random_best.npz (genome, loss, keyframe tiles), strip.png (targets / start / best / random best), and from
`eval`: eval.json, report.md, persist.png, films/*.gif.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLIP_MODEL = "openai/clip-vit-base-patch32"
TPX = 224


# ---------- Sep-CMA-ES, evosax 0.1.6, ported to numpy ----------

class SepCMAES:
    """evosax 0.1.6 Sep_CMA_ES with default params, elite_ratio 0.5, mean_decay 0, no fitness shaping, minimising.
    Every formula below is theirs (strategies/sep_cma_es.py, strategy.py, utils/helpers.py, utils/eigen_decomp.py)."""

    KEYS = ("p_sigma", "p_c", "C", "D", "mean", "sigma", "sigma_scale", "best_member", "best_fitness", "gen_counter")

    def __init__(self, popsize: int, num_dims: int, sigma_init: float, elite_ratio: float = 0.5):
        self.popsize, self.n = popsize, num_dims
        self.elite = max(1, int(popsize * elite_ratio))
        wp = np.array([np.log(self.elite + 1) - np.log(i + 1) for i in range(self.elite)])
        self.weights = wp / wp.sum()
        self.weights_truncated = np.zeros(popsize)
        self.weights_truncated[:self.elite] = self.weights
        mds = min(num_dims, 40000)                                   # their max_dims_sq (not squared)
        mu_eff = 1 / np.sum(self.weights ** 2)
        c_1 = 2 / ((mds + 1.3) ** 2 + mu_eff)
        c_mu_full = 2 / mu_eff / ((mds + np.sqrt(2)) ** 2) + (1 - 1 / mu_eff) * min(1, (2 * mu_eff - 1) / ((mds + 2) ** 2 + mu_eff))
        self.p = dict(mu_eff=mu_eff, c_1=c_1, c_mu=(num_dims + 2) / 3 * c_mu_full,
                      c_sigma=(mu_eff + 2) / (num_dims + mu_eff + 3), c_c=4 / (num_dims + 4), c_m=1.0)
        self.p["d_sigma"] = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (num_dims + 1)) - 1) + self.p["c_sigma"]
        self.p["chi_n"] = np.sqrt(num_dims) * (1.0 - 1.0 / (4.0 * num_dims) + 1.0 / (21.0 * mds ** 2))
        self.s = dict(p_sigma=np.zeros(num_dims), p_c=np.zeros(num_dims), C=np.ones(num_dims), D=np.ones(num_dims),
                      mean=np.zeros(num_dims), sigma=np.ones(num_dims) * sigma_init, sigma_scale=float(sigma_init),
                      best_member=np.zeros(num_dims), best_fitness=float(np.finfo(np.float32).max), gen_counter=0)

    def ask(self, rng) -> np.ndarray:
        z = rng.standard_normal((self.n, self.popsize))
        return self.s["mean"] + self.s["sigma_scale"] * (self.s["D"][:, None] * z).T

    def tell(self, x: np.ndarray, fitness: np.ndarray) -> None:
        s, p = self.s, self.p
        order = np.argsort(fitness, kind="stable")
        x_k = x[order]
        y_k = (x_k - s["mean"]) / s["sigma"]
        y_w = (y_k.T * self.weights_truncated).sum(1)
        mean = s["mean"] + p["c_m"] * s["sigma"] * y_w
        D = s["D"]                                                    # diag_eigen_decomp(C, D) returns D as is
        p_sigma = (1 - p["c_sigma"]) * s["p_sigma"] + np.sqrt(p["c_sigma"] * (2 - p["c_sigma"]) * p["mu_eff"]) * (y_w / D)
        gen = s["gen_counter"] + 1
        norm_ps = np.linalg.norm(p_sigma)
        h_sigma = 1.0 * (norm_ps / np.sqrt(1 - (1 - p["c_sigma"]) ** (2 * gen)) < (1.4 + 2 / (self.n + 1)) * p["chi_n"])
        p_c = (1 - p["c_c"]) * s["p_c"] + h_sigma * np.sqrt(p["c_c"] * (2 - p["c_c"]) * p["mu_eff"]) * y_w
        delta_h = (1 - h_sigma) * p["c_c"] * (2 - p["c_c"])
        rank_mu = sum(w * y ** 2 for w, y in zip(self.weights, y_k))
        C = ((1 + p["c_1"] * delta_h - p["c_1"] - p["c_mu"] * self.weights.sum()) * s["C"]
             + p["c_1"] * p_c ** 2 + p["c_mu"] * rank_mu)
        sigma_scale = s["sigma_scale"] * np.exp((p["c_sigma"] / p["d_sigma"]) * (norm_ps / p["chi_n"] - 1))
        i = int(np.argmin(fitness))
        if fitness[i] < s["best_fitness"]:
            s["best_fitness"], s["best_member"] = float(fitness[i]), x[i].copy()
        s.update(mean=mean, p_sigma=p_sigma, C=C, D=D, p_c=p_c, sigma_scale=float(sigma_scale),
                 sigma=sigma_scale * D, gen_counter=gen)

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **{k: np.asarray(self.s[k]) for k in self.KEYS})
        tmp.replace(path)

    def load(self, path: Path) -> None:
        d = np.load(path)
        for k in self.KEYS:
            v = d[k]
            self.s[k] = v if v.ndim else (int(v) if k == "gen_counter" else float(v))


# ---------- the genome (ASAL substrates/lenia.py, on MaCE) ----------

def logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-6, 1 - 1e-6)
    return np.log(x) - np.log1p(-x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Genome:
    """offsets (n,) -> (rule, starting world). Base: the --center world's m, s, h per kernel and the fixed start patch."""

    def __init__(self, rule, size: int, R: float, start_seed: int, clip1: float = 1.0):
        from lenia import seeds
        self.rule, self.size, self.clip1 = rule, size, clip1
        n = max(4, int(round(3.0 * R)))                              # seeds.smooth_noise's patch width at size_r=3
        self.patch = seeds.smooth_noise(R, (n, n), 3, seed=start_seed, size_r=3.0)   # (3, n, n), the fixed start
        k = rule.kernels
        self.base_dyn = logit(np.array([[x.m for x in k], [x.s for x in k], [x.h for x in k]]).ravel())
        self.base_init = logit(self.patch.ravel())
        self.n_dyn = self.base_dyn.size
        self.n = self.n_dyn + self.base_init.size

    def express(self, g: np.ndarray):
        import copy
        from lenia.sim import place
        dyn = sigmoid(self.base_dyn + np.clip(g[:self.n_dyn], -self.clip1, self.clip1)).reshape(3, -1)
        cells = sigmoid(self.base_init + g[self.n_dyn:]).reshape(self.patch.shape)
        r = copy.deepcopy(self.rule)
        for i, kk in enumerate(r.kernels):
            kk.m, kk.s, kk.h = float(dyn[0, i]), float(dyn[1, i]), float(dyn[2, i])
        return r, place(cells.astype(np.float32), (self.size, self.size), 3)


# ---------- rollout, render, embed ----------

def view(A: np.ndarray, crop: int) -> np.ndarray:
    """(C, H, W) -> (224, 224, 3) uint8: a crop x crop window centred on the centre of mass (circular mean on the
    torus), channels as RGB scaled by the window's 99.5th percentile, nearest-upscaled (ASAL's render_state)."""
    from lenia import render
    m = A.sum(0)
    H, W = m.shape
    tot = max(float(m.sum()), 1e-12)

    def centre(w, n):
        th = 2 * np.pi * np.arange(n) / n
        return int(round((np.angle((w * np.exp(1j * th)).sum() / tot) % (2 * np.pi)) * n / (2 * np.pi))) % n

    cy, cx = centre(m.sum(1), H), centre(m.sum(0), W)
    A = np.roll(A, (H // 2 - cy, W // 2 - cx), axis=(-2, -1))
    y0, x0 = H // 2 - crop // 2, W // 2 - crop // 2
    win = A[:, y0:y0 + crop, x0:x0 + crop]
    scale = max(float(np.percentile(win, 99.5)), 1e-6)
    return np.asarray(Image.fromarray(render.colorize(np.clip(win / scale, 0, 1))).resize((TPX, TPX), Image.NEAREST))


def rollout(rules: list, worlds: np.ndarray, steps: int, K: int, size: int, upto: int | None = None) -> np.ndarray:
    """Run one batch; return the states at chunk ends t = c, 2c, ... (c = steps // K) up to `upto` (default steps):
    (B, n_samples, C, H, W). rollout.py's time_sampling=(K, True)."""
    from lenia import backend
    c = steps // K
    keep = set(range(c, (upto or steps) + 1, c))
    b = backend.make_batch(rules, (size, size))
    A = backend.to_device(worlds)
    out = []
    for t in range(1, max(keep) + 1):
        A = b.step(A)
        if t in keep:
            out.append(backend.to_numpy(A).copy())
    return np.stack(out, 1)


class AsalClip:
    """foundation_models/clip.py's embed_img in PyTorch; fm='fake': a fixed random projection of 8x8 colour thumbnails."""

    def __init__(self, fm: str):
        self.fake = fm == "fake"
        if self.fake:
            self.P = np.random.default_rng(0).normal(size=(192, 64))
            return
        import torch
        from transformers import CLIPImageProcessor, CLIPModel
        from lenia.embed import _device
        self.torch, self.dev = torch, _device()
        self.m = CLIPModel.from_pretrained(CLIP_MODEL).to(self.dev).eval()
        pr = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
        self.mean = torch.tensor(pr.image_mean).view(1, 3, 1, 1).to(self.dev)
        self.std = torch.tensor(pr.image_std).view(1, 3, 1, 1).to(self.dev)

    def __call__(self, imgs: list, batch: int = 64) -> np.ndarray:
        sq = [np.asarray(Image.fromarray(np.asarray(im, np.uint8)).resize((TPX, TPX), Image.BILINEAR)) if
              np.asarray(im).shape[:2] != (TPX, TPX) else np.asarray(im, np.uint8) for im in imgs]
        if self.fake:
            v = np.stack([np.asarray(Image.fromarray(x).resize((8, 8), Image.BILINEAR), np.float32).ravel() / 255
                          for x in sq]) @ self.P
        else:
            out = []
            for k in range(0, len(sq), batch):
                x = self.torch.from_numpy(np.stack(sq[k:k + batch]).astype(np.float32) / 255).permute(0, 3, 1, 2).to(self.dev)
                with self.torch.no_grad():                 # towers + projection (get_image_features changed in transformers 5)
                    po = self.m.vision_model(pixel_values=(x - self.mean) / self.std).pooler_output
                    out.append(self.m.visual_projection(po).float().cpu().numpy())
            v = np.concatenate(out)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def target_loss(z: np.ndarray, zt: np.ndarray) -> float:
    """asal_metrics.calc_supervised_target_score with image targets: z (T, D) world frames, zt (T2, D) targets."""
    T, T2 = len(z), len(zt)
    assert T % T2 == 0
    zt = np.tile(zt, (T // T2, 1))
    return float(-np.diag(zt @ z.T).mean())


# ---------- setup shared by run and eval ----------

def load_center(spec: str, a) -> tuple:
    """'DIR:ID' (a run folder with members.jsonl, and state.json for size, R and start seed) or a rule .json."""
    from lenia.sim import Rule
    p = Path(spec)
    if p.suffix == ".json" and p.exists():
        return Rule.from_dict(json.loads(p.read_text())), {"size": a.size or 160, "R": 13, "start_seed": 0}, p.stem
    d, _, wid = spec.rpartition(":")
    d = Path(d)
    rule = None
    for line in open(d / "members.jsonl"):
        x = json.loads(line)
        if x["id"] == wid:
            rule = x["rule"]                                     # a repeated id rewrites the same rule: keep the last
    if rule is None:
        sys.exit(f"{wid} not in {d / 'members.jsonl'}")
    st = json.loads((d / "state.json").read_text())["settings"]
    return Rule.from_dict(rule), {k: st[k] for k in ("size", "R", "start_seed")}, wid


def keyframes(target: str, K: int) -> tuple[list, list]:
    import embedcheck as EC
    frames = EC.load_frames(EC.parse_gif(target))
    idx = [int(round(x)) for x in np.linspace(0, len(frames) - 1, K)]
    return [np.asarray(frames[i]) for i in idx], idx


def setup(st: dict, fm: str):
    from lenia.sim import Rule
    s = st["settings"]
    geno = Genome(Rule.from_dict(st["center_rule"]), s["size"], s["R"], s["start_seed"], s["clip1"])
    clip = AsalClip(fm)
    kf, _ = keyframes(st["args"]["target"], s["K"])
    return geno, clip, kf, clip(kf)


def evaluate(G: np.ndarray, geno: Genome, clip: AsalClip, zt: np.ndarray, s: dict, batch: int) -> tuple:
    """Genomes (B, n) -> losses (B,), keyframe tiles (B, K, 224, 224, 3)."""
    losses, tiles = [], []
    for k in range(0, len(G), batch):
        pairs = [geno.express(g) for g in G[k:k + batch]]
        F = rollout([r for r, _ in pairs], np.stack([w for _, w in pairs]), s["steps"], s["K"], s["size"])
        for f in F:
            t = [view(x, s["crop"]) for x in f]
            losses.append(target_loss(clip(t), zt))
            tiles.append(np.stack(t))
    return np.array(losses), np.stack(tiles)


def strip(rows: list, path: Path, px: int = 112) -> None:
    """rows: [(label, [images])] -> one PNG, a labelled row per entry."""
    lab = 150
    n = max(len(r[1]) for r in rows)
    W, H = lab + n * (px + 4), len(rows) * (px + 4)
    out = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(out)
    for i, (name, ims) in enumerate(rows):
        d.text((6, i * (px + 4) + px // 2 - 12), name, fill=(230, 230, 230))
        for j, im in enumerate(ims):
            out.paste(Image.fromarray(np.asarray(im, np.uint8)).resize((px, px), Image.BILINEAR),
                      (lab + j * (px + 4), i * (px + 4)))
    out.save(path)


def save_npz(path: Path, **kw) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **kw)
    tmp.replace(path)


# ---------- run ----------

def run(a) -> None:
    from lenia import backend
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backend.configure(a.backend)
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        if a.iters:
            st["args"]["iters"] = a.iters                          # a resume may extend the run
        print(f"resuming {out} at iteration {st['it']} ({st['sims']:,} simulations)", file=sys.stderr)
    else:
        if not (a.target and a.center):
            sys.exit("--target and --center are needed for a new run")
        rule, cs, wid = load_center(a.center, a)
        s = {"size": a.size or cs["size"], "R": cs["R"], "start_seed": cs["start_seed"], "steps": a.steps,
             "K": a.keyframes, "crop": a.crop if a.crop else (a.size or cs["size"]) // 2, "pop": a.pop,
             "sigma": a.sigma, "clip1": 1.0, "random": a.random, "seed": a.seed, "fm": a.fm}
        st = {"args": {"target": a.target, "center": a.center, "iters": a.iters or 1000}, "settings": s,
              "center_id": wid, "center_rule": rule.to_dict(), "it": 0, "sims": 0}
    s, iters = st["settings"], st["args"]["iters"]
    if a.fm != s["fm"]:
        sys.exit(f"this run was started with --fm {s['fm']}")
    geno, clip, kf, zt = setup(st, s["fm"])
    _, kidx = keyframes(st["args"]["target"], s["K"])
    es = SepCMAES(s["pop"], geno.n, s["sigma"])
    rng = np.random.default_rng(s["seed"])
    if (out / "es.npz").exists():
        es.load(out / "es.npz")
        rng.bit_generator.state = st["rng"]
        best, rbest = dict(np.load(out / "best.npz")), dict(np.load(out / "random_best.npz"))
    else:
        l0, t0 = evaluate(np.zeros((1, geno.n)), geno, clip, zt, s, 1)
        save_npz(out / "start.npz", genome=np.zeros(geno.n, np.float32), loss=l0[0], tiles=t0[0])
        best = {"genome": np.zeros(geno.n, np.float32), "loss": np.float64(np.inf), "tiles": t0[0]}
        rbest = dict(best)
        st["start_loss"] = float(l0[0])
        st["keyframe_index"] = kidx
        print(f"genome: {geno.n_dyn} dynamics + {geno.n - geno.n_dyn} pattern numbers; start world loss "
              f"{l0[0]:.4f}; {iters} iterations x ({s['pop']} + {s['random']} random) simulations", file=sys.stderr)
    newlog = not (out / "log.csv").exists()
    logf = open(out / "log.csv", "a", newline="")
    log = csv.writer(logf)
    if newlog:
        log.writerow(["it", "sims", "es_min", "es_mean", "es_best", "random_min", "random_best", "sigma_scale", "sec"])
    start0 = np.load(out / "start.npz")

    def checkpoint():
        es.save(out / "es.npz")
        save_npz(out / "best.npz", **best)
        save_npz(out / "random_best.npz", **rbest)
        st["rng"] = rng.bit_generator.state
        st["best_loss"], st["random_best_loss"] = float(best["loss"]), float(rbest["loss"])
        tmp = out / "state.tmp.json"
        tmp.write_text(json.dumps(st, indent=1, default=float))
        tmp.replace(out / "state.json")
        strip([("target (GIF)", kf), (f"start {st['center_id']}", start0["tiles"]),
               (f"ES best {float(best['loss']):.3f}", best["tiles"]),
               (f"random best {float(rbest['loss']):.3f}", rbest["tiles"])], out / "strip.png")

    while st["it"] < iters:
        t0 = time.time()
        X = es.ask(rng)
        Xr = rng.normal(0, 0.1, (s["random"], geno.n)) if s["random"] else np.zeros((0, geno.n))
        L, T = evaluate(np.concatenate([X, Xr]), geno, clip, zt, s, a.batch or (s["pop"] + s["random"]))
        le, lr = L[:s["pop"]], L[s["pop"]:]
        es.tell(X, le)
        i = int(np.argmin(le))
        if le[i] < best["loss"]:
            best = {"genome": X[i].astype(np.float32), "loss": np.float64(le[i]), "tiles": T[i]}
        if len(lr) and lr.min() < rbest["loss"]:
            j = int(np.argmin(lr))
            rbest = {"genome": Xr[j].astype(np.float32), "loss": np.float64(lr[j]), "tiles": T[s["pop"] + j]}
        st["it"] += 1
        st["sims"] += len(L)
        log.writerow([st["it"], st["sims"], f"{le.min():.5f}", f"{le.mean():.5f}", f"{es.s['best_fitness']:.5f}",
                      f"{lr.min():.5f}" if len(lr) else "", f"{float(rbest['loss']):.5f}" if len(lr) else "",
                      f"{es.s['sigma_scale']:.5f}", f"{time.time() - t0:.1f}"])
        logf.flush()
        if st["it"] % a.ckpt == 0 or st["it"] == iters:
            checkpoint()
            print(f"it {st['it']}/{iters}: ES best {es.s['best_fitness']:.4f}, random best {float(rbest['loss']):.4f}, "
                  f"start {st['start_loss']:.4f}, sigma {es.s['sigma_scale']:.4f} ({time.time() - t0:.1f} s/it)",
                  file=sys.stderr)
    logf.close()


# ---------- eval (ours) ----------

def eval_(a) -> None:
    import embedcheck as EC
    from lenia import backend
    from lenia.embed import Embedder
    import mace_qd as QD
    out = Path(a.out)
    backend.configure(a.backend)
    st = json.loads((out / "state.json").read_text())
    s = st["settings"]
    fm = a.fm or s["fm"]
    geno, clip, kf, zt = setup(st, fm)
    dino = Embedder("fake" if fm == "fake" else "dinov2_vitb14")
    frames = [np.asarray(f) for f in EC.load_frames(EC.parse_gif(st["args"]["target"]))]
    kidx = st["keyframe_index"]

    def emb(model, ims):
        if model == "clip":
            return clip(ims)
        if model == "dino":
            v = dino([Image.fromarray(np.asarray(im, np.uint8)).resize((TPX, TPX)) for im in ims])
        else:
            v = np.stack([np.asarray(Image.fromarray(np.asarray(im, np.uint8)).convert("L").resize((32, 32)),
                                     np.float32).ravel() for im in ims])
            v = v - v.mean(1, keepdims=True)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    models = ("clip", "dino", "pixels")
    tgt = {m: emb(m, kf) for m in models}
    # ceiling: each keyframe against the GIF frame next to it (how alike two real frames a moment apart are)
    nb = [frames[i + 1] if i + 1 < len(frames) else frames[i - 1] for i in kidx]
    ceil = {m: float(np.mean((emb(m, nb) * tgt[m]).sum(1))) for m in models}
    worlds = {"start": np.zeros(geno.n), "ES best": np.load(out / "best.npz")["genome"].astype(np.float64)}
    if s["random"]:
        worlds["random best"] = np.load(out / "random_best.npz")["genome"].astype(np.float64)
    pairs = [geno.express(g) for g in worlds.values()]
    upto = a.persist * s["steps"]
    F = rollout([r for r, _ in pairs], np.stack([w for _, w in pairs]), s["steps"], s["K"], s["size"], upto)
    res, rows = {}, [("target (GIF)", kf)]
    films = out / "films"
    films.mkdir(exist_ok=True)
    for (name, g), f in zip(worlds.items(), F):
        tiles = [view(x, s["crop"]) for x in f]
        r = {"bodies": [QD.n_bodies(x) for x in f], "sample_steps": [(i + 1) * (s["steps"] // s["K"]) for i in range(len(f))]}
        for m in models:
            S = tgt[m] @ emb(m, tiles[:s["K"]]).T                  # (target k, world frame j)
            diag = float(np.diag(S).mean())
            r[m] = {"match": round(diag, 4), "order_gain": round(diag - float(S[~np.eye(len(S), dtype=bool)].mean()), 4),
                    "per_frame": [round(float(x), 4) for x in np.diag(S)]}
        r["clip_loss_replayed"] = round(target_loss(clip(tiles[:s["K"]]), zt), 5)
        res[name] = r
        rows.append((f"{name} ({r['clip']['match']:.3f})", tiles[:s["K"]]))
        rows.append((f"  ...after the clip", tiles[s["K"]:]))
    stored = {"ES best": float(np.load(out / "best.npz")["loss"]), "start": st.get("start_loss")}
    strip(rows, out / "persist.png")
    for name, (r, w) in zip(worlds, pairs):                        # films: every steps/90 steps, to upto
        G = rollout([r], w[None], upto, min(90 * a.persist, upto), s["size"])[0]
        ims = [Image.fromarray(view(x, s["crop"])).resize((160, 160), Image.NEAREST) for x in G]
        ims[0].save(films / f"{name.replace(' ', '_')}.gif", save_all=True, append_images=ims[1:], duration=60, loop=0)
    ev = {"worlds": res, "ceiling_neighbour_frame": ceil, "stored_loss": stored, "keyframe_index": kidx,
          "iterations": st["it"], "simulations": st["sims"], "fm": fm}
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    L = [f"# ASAL target search: {st['args']['target']} (eval)", "",
         f"{st['it']} iterations, {st['sims']:,} simulations, start world {st['center_id']}, {s['K']} keyframes "
         f"(GIF frames {kidx}), {s['steps']} steps. fm = {fm}" + (" (FAKE: placeholders only)" if fm == "fake" else ""), "",
         "Match = mean similarity of world frame k to keyframe k (CLIP is the objective; DINOv2 and pixels are held out).",
         "Order gain = match minus the mean similarity to the other keyframes (> 0: frame k looks most like keyframe k).",
         f"Ceiling = a keyframe vs the GIF frame next to it: CLIP {ceil['clip']:.3f}, DINOv2 {ceil['dino']:.3f}, "
         f"pixels {ceil['pixels']:.3f}.", "",
         "| world | CLIP match | CLIP order gain | DINOv2 match | DINOv2 order gain | pixel match | bodies at samples |",
         "|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        L.append(f"| {n} | {r['clip']['match']:.3f} | {r['clip']['order_gain']:+.3f} | {r['dino']['match']:.3f} | "
                 f"{r['dino']['order_gain']:+.3f} | {r['pixels']['match']:.3f} | {' '.join(map(str, r['bodies']))} |")
    L += ["", f"Bodies are counted at steps {res['start']['sample_steps']} (the first {s['K']} are the matched frames; the "
          f"rest run on past the clip: does it keep dividing?).",
          f"Replay check: ES best loss stored {stored['ES best']:.5f}, replayed {res['ES best']['clip_loss_replayed']:.5f}.",
          "", "Pictures: strip.png (from the run), persist.png, films/*.gif."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------- escheck (ours): the numpy Sep-CMA-ES against evosax 0.1.6 ----------

def escheck(a) -> None:
    """Feed the same candidates and fitnesses to evosax 0.1.6 and to SepCMAES for 30 generations; compare states.
    Colab: pip install evosax==0.1.6 first."""
    try:
        import jax
        import evosax
    except ImportError as e:
        sys.exit(f"needs jax and evosax 0.1.6 ({e}); pip install evosax==0.1.6")
    print(f"evosax {getattr(evosax, '__version__', '?')}, jax {jax.__version__}")
    n, pop = 50, 16
    ref = evosax.Sep_CMA_ES(popsize=pop, num_dims=n, sigma_init=0.1)
    ep = ref.default_params
    rs = ref.initialize(jax.random.PRNGKey(0), ep)
    mine = SepCMAES(pop, n, 0.1)
    rng = np.random.default_rng(0)
    worst = 0.0
    for g in range(30):
        x, rs = ref.ask(jax.random.PRNGKey(g + 1), rs, ep)
        x = np.asarray(x, np.float64)
        f = (x ** 2).sum(1) + rng.normal(0, 1e-3, pop)
        rs = ref.tell(x, f, rs, ep)
        mine.tell(x, f)
        for k in ("mean", "p_sigma", "p_c", "C", "sigma"):
            d = float(np.max(np.abs(np.asarray(getattr(rs, k)) - mine.s[k])) / (np.max(np.abs(mine.s[k])) + 1e-12))
            worst = max(worst, d)
        worst = max(worst, abs(float(rs.sigma_scale) - mine.s["sigma_scale"]) / mine.s["sigma_scale"])
    print(f"largest relative difference over 30 generations: {worst:.2e} "
          f"({'same' if worst < 1e-4 else 'DIFFERENT'}; float32 vs float64 gives ~1e-6)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--target", help="GIF spec name=path[:crop_top=N] (as embedcheck.py)")
    r.add_argument("--center", help="known world: RUN_DIR:ID (members.jsonl) or a rule .json")
    r.add_argument("--keyframes", type=int, default=6)
    r.add_argument("--steps", type=int, default=2700)
    r.add_argument("--crop", type=int, default=0, help="render window (default: half the world)")
    r.add_argument("--pop", type=int, default=16)
    r.add_argument("--sigma", type=float, default=0.1)
    r.add_argument("--iters", type=int, default=0, help="default 1000; on resume, extends the run")
    r.add_argument("--random", type=int, default=16, help="random genomes per iteration (0 = no control)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--size", type=int, default=0, help="dry runs only: a smaller world")
    r.add_argument("--batch", type=int, default=0, help="simulations per GPU batch (default: one iteration)")
    r.add_argument("--ckpt", type=int, default=10)
    r.add_argument("--fm", default="real", choices=("real", "fake"))
    r.add_argument("--backend", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--out", required=True)
    e.add_argument("--persist", type=int, default=2, help="run to this many times the clip's length")
    e.add_argument("--fm", default=None, choices=("real", "fake"))
    e.add_argument("--backend", default=None)
    sub.add_parser("escheck")
    a = ap.parse_args()
    {"run": run, "eval": eval_, "escheck": escheck}[a.cmd](a)


if __name__ == "__main__":
    main()
