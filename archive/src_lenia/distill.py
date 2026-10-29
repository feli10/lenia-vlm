"""Phase 3: distil the teacher's pairwise preferences into a small reward model (the student).

embed  (Colab or your Mac; needs PyTorch and an internet connection the first time) run a frozen,
       pretrained image model over every creature image of a pool. Two-view images (twice as wide
       as tall) are split into their two views, each embedded on its own, and the two vectors joined,
       so the student sees motion and shape separately. Writes embed_<model>.npz in the pool folder.
       Models: dinov2_vits14 (default, 384 per view), dinov2_vitb14 (768), clip_vitb32 (open_clip).
train  (anywhere; numpy only) fit a Bradley-Terry reward on the teacher's training pairs:
       P(a beats b) = sigmoid(r(a) - r(b)), r = a linear function (or a small MLP) of the standardised
       embedding, soft targets = the teacher's order-averaged p_a, L2 penalty chosen on a validation
       split of the training creatures. An ensemble of bootstrap fits gives a mean reward and its
       spread (for uncertainty-aware search later).
       Baselines fitted the same way: "larger wins", "heavier wins", and a reward on the five
       hand-made statistics (speed, change, size, mass, texture).
       Evaluated on held-out pairs only (creatures the student never saw): agreement with the
       teacher (all pairs and decisive ones), with the second judge, with your ratings, and on
       size-matched pairs (where the teacher is not just picking the larger creature).
       Writes distill_<model>_<prompt>/: report.md, report.json, scores.csv, student.npz,
       top/bottom contact sheets.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

MODELS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "clip_vitb32": 512}


# ---------- embed ----------

def _views(img) -> list:
    """A two-view image (width about 2x height) -> [motion view, close-up block]; otherwise [img]."""
    w, h = img.size
    if w > 1.6 * h:
        return [img.crop((0, 0, h, h)), img.crop((w - h, 0, w, h))]
    return [img]


def _load_model(name: str, device: str):
    import torch
    if name.startswith("dinov2"):
        m = torch.hub.load("facebookresearch/dinov2", name).to(device).eval()
        mean, std, size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
        return (lambda x: m(x)), mean, std, size
    if name == "clip_vitb32":
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        m = m.to(device).eval()
        mean, std, size = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758), 224
        return (lambda x: m.encode_image(x)), mean, std, size
    raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """A frozen image model, loaded once: embed(list of PIL images) -> (n, dims) array, two-view images
    split into their two views and the vectors joined (as `embed` does for a pool)."""

    def __init__(self, model: str = "dinov2_vits14", device: str | None = None):
        self.model = model
        if model == "fake":                 # tests: no PyTorch; 8x8 grey thumbnails of each view
            self.device = "cpu"
            return
        import torch
        self.device = device or _device()
        self.fwd, mean, std, self.size = _load_model(model, self.device)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

    def __call__(self, images: list, batch: int = 64) -> np.ndarray:
        from PIL import Image
        if self.model == "fake":
            return np.array([np.concatenate([np.asarray(v.convert("L").resize((8, 8)), np.float32).ravel() / 255
                                             for v in _views(im)]) for im in images])
        import torch
        out = []
        for k in range(0, len(images), batch):
            views = [_views(im) for im in images[k:k + batch]]
            nv = len(views[0])
            x = np.stack([np.asarray(v.resize((self.size, self.size), Image.BICUBIC), np.float32) / 255 for vs in views for v in vs])
            xt = (torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device) - self.mean) / self.std
            with torch.no_grad():
                e = self.fwd(xt).float().cpu().numpy()
            out.append(e.reshape(len(views), nv * e.shape[-1]))
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def embed(pool_dir: Path, model: str = "dinov2_vits14", batch: int = 64, log=print) -> Path:
    from PIL import Image
    pool_dir = Path(pool_dir)
    emb = Embedder(model)
    ids = [r["id"] for r in csv.DictReader(open(pool_dir / "creatures.csv"))]
    out = []
    for k in range(0, len(ids), batch):
        part = ids[k:k + batch]
        out.append(emb([Image.open(pool_dir / "images" / f"{i}.png").convert("RGB") for i in part], batch))
        log(f"  embedded {min(k + batch, len(ids))}/{len(ids)}")
    E = np.concatenate(out)
    path = pool_dir / f"embed_{model}.npz"
    np.savez_compressed(path, ids=np.array(ids), emb=E.astype(np.float32), model=model)
    log(f"{path.name}: {E.shape[0]} creatures x {E.shape[1]} dims ({model} on {emb.device})")
    return path


# ---------- train ----------

def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class Reward:
    """r(x) = w.x (linear) or v.tanh(W x + b) (mlp), fitted by Adam on the soft Bradley-Terry loss."""

    def __init__(self, dim: int, kind: str = "linear", hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.kind = kind
        if kind == "linear":
            self.p = {"w": np.zeros(dim)}
        else:
            self.p = {"W": rng.normal(0, 1 / np.sqrt(dim), (hidden, dim)), "b": np.zeros(hidden),
                      "v": rng.normal(0, 1 / np.sqrt(hidden), hidden)}

    def __call__(self, X):
        if self.kind == "linear":
            return X @ self.p["w"]
        return np.tanh(X @ self.p["W"].T + self.p["b"]) @ self.p["v"]

    def _grads(self, X, ia, ib, y, lam):
        r = self(X)
        d = _sig(r[ia] - r[ib]) - y                    # dLoss/d(r_a - r_b), mean soft BCE
        g_r = np.zeros(len(X))
        np.add.at(g_r, ia, d / len(y))
        np.add.at(g_r, ib, -d / len(y))
        if self.kind == "linear":
            return {"w": X.T @ g_r + lam * self.p["w"]}
        h = np.tanh(X @ self.p["W"].T + self.p["b"])
        gh = np.outer(g_r, self.p["v"]) * (1 - h ** 2)
        return {"v": h.T @ g_r + lam * self.p["v"], "W": gh.T @ X + lam * self.p["W"], "b": gh.sum(0)}

    def fit(self, X, ia, ib, y, lam: float = 1e-3, steps: int = 600, lr: float = 0.02):
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        s = {k: np.zeros_like(v) for k, v in self.p.items()}
        for t in range(1, steps + 1):
            g = self._grads(X, ia, ib, y, lam)
            for k in self.p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
        return self


def _labels(pool_dir: Path, judge: str, prompt: str) -> dict:
    p = Path(pool_dir) / f"labels_{judge}_{prompt}.csv"
    if not p.exists():
        return {}
    fl = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    return {(r["a"], r["b"]): (fl(r["p_a"]), r["split"]) for r in csv.DictReader(open(p)) if np.isfinite(fl(r["p_a"]))}


def _pairs_arrays(pairs: list, idx: dict):
    return np.array([idx[a] for a, _ in pairs]), np.array([idx[b] for _, b in pairs])


def _bce(r, ia, ib, y):
    q = np.clip(_sig(r[ia] - r[ib]), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _fit_select(X, idx, pairs, y, kind, rng, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Pick the L2 penalty on a creature-level validation split of the training pairs, then refit on all."""
    creatures = sorted({c for p in pairs for c in p})
    val_c = set(rng.choice(creatures, max(2, len(creatures) // 8), replace=False).tolist())
    tr = [k for k, (a, b) in enumerate(pairs) if a not in val_c and b not in val_c]
    va = [k for k, (a, b) in enumerate(pairs) if a in val_c and b in val_c]
    ia, ib = _pairs_arrays(pairs, idx)
    best = (np.inf, lams[0])
    if len(va) >= 10:
        for lam in lams:
            m = Reward(X.shape[1], kind).fit(X, ia[tr], ib[tr], y[tr], lam)
            loss = _bce(m(X), ia[va], ib[va], y[va])
            best = min(best, (loss, lam))
    return best[1]


def train(pool_dir: Path, model: str = "dinov2_vits14", judge: str = "luna", prompt: str = "curate_dual",
          kind: str = "linear", ensemble: int = 5, seed: int = 0, out: Path | None = None, log=print) -> dict:
    from .pool import FEATURES
    pool_dir = Path(pool_dir)
    rng = np.random.default_rng(seed)
    z = np.load(pool_dir / f"embed_{model}.npz")
    ids = [str(i) for i in z["ids"]]
    idx = {i: k for k, i in enumerate(ids)}
    cre = {r["id"]: r for r in csv.DictReader(open(pool_dir / "creatures.csv"))}
    L = _labels(pool_dir, judge, prompt)
    train_pairs = [k for k, (p, s) in L.items() if s == "train"]
    held_pairs = [k for k, (p, s) in L.items() if s == "heldout"]
    if not train_pairs:
        raise SystemExit(f"no training labels in labels_{judge}_{prompt}.csv")
    train_ids = sorted({c for p in train_pairs for c in p})
    E = z["emb"].astype(np.float64)
    mu, sd = E[[idx[i] for i in train_ids]].mean(0), E[[idx[i] for i in train_ids]].std(0) + 1e-6
    X = (E - mu) / sd
    # hand-made statistics, standardised the same way (log for speed, change, mass)
    fl = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    F = np.array([[fl(cre[i][f]) for f in FEATURES] for i in ids])
    for j, f in enumerate(FEATURES):
        if f in ("speed", "change", "mass"):
            F[:, j] = np.log(np.maximum(F[:, j], 1e-4))
    f_mu, f_sd = np.nanmean(F[[idx[i] for i in train_ids]], 0), np.nanstd(F[[idx[i] for i in train_ids]], 0) + 1e-6
    F = np.nan_to_num((F - f_mu) / f_sd)
    y = np.array([L[k][0] for k in train_pairs])
    lam = _fit_select(X, idx, train_pairs, y, kind, rng)
    ia, ib = _pairs_arrays(train_pairs, idx)
    members = []
    for e in range(ensemble):
        boot = rng.integers(len(train_pairs), size=len(train_pairs)) if ensemble > 1 else np.arange(len(train_pairs))
        members.append(Reward(X.shape[1], kind, seed=seed + e).fit(X, ia[boot], ib[boot], y[boot], lam))
    R = np.stack([m(X) for m in members])
    score, spread = R.mean(0), R.std(0)
    lam_f = _fit_select(F, idx, train_pairs, y, "linear", rng)
    feat = Reward(F.shape[1], "linear").fit(F, ia, ib, y, lam_f)
    scorers = {"student": score, "features (5 statistics)": feat(F),
               "larger wins": np.array([fl(cre[i]["size_rg"]) for i in ids]),
               "heavier wins": np.array([fl(cre[i]["mass"]) for i in ids])}
    res = {"model": model, "judge": judge, "prompt": prompt, "kind": kind, "lambda": lam, "ensemble": ensemble,
           "train_pairs": len(train_pairs), "heldout_pairs": len(held_pairs), "train_creatures": len(train_ids),
           "feature_weights": {f: round(float(w), 2) for f, w in zip(FEATURES, feat.p["w"])}, "eval": {}}
    res["eval"] = evaluate(pool_dir, scorers, ids, cre, judge, prompt, held_pairs, L)
    out = Path(out or pool_dir / f"distill_{model}_{prompt}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "student.npz", ids=np.array(ids), score=score, spread=spread, mu=mu, sd=sd,
                        feat_w=feat.p["w"], feat_mu=f_mu, feat_sd=f_sd, feat_names=np.array(FEATURES), n_members=len(members),
                        lam=lam, kind=kind, model=model, **{f"m{k}_{n}": v for k, m in enumerate(members) for n, v in m.p.items()})
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "split", "source", "student", "spread", "features", "size_rg", "mass", "speed"])
        for k, i in enumerate(ids):
            w.writerow([i, cre[i]["group"], cre[i]["split"], cre[i]["source"], round(float(score[k]), 4), round(float(spread[k]), 4),
                        round(float(scorers["features (5 statistics)"][k]), 4), cre[i]["size_rg"], cre[i]["mass"], cre[i]["speed"]])
    _sheets(pool_dir, out, ids, cre, score)
    (out / "report.json").write_text(json.dumps(res, indent=2))
    _markdown(out, res)
    log((out / "report.md").read_text())
    return res


class Student:
    """A trained student (student.npz) as a scorer of new creatures: from images (embedding + ensemble
    head; mean and spread over members) or, for the hand-made baseline, from the five statistics."""

    def __init__(self, path: Path):
        z = np.load(Path(path) / "student.npz" if Path(path).is_dir() else path, allow_pickle=False)
        self.model, self.kind = str(z["model"]), str(z["kind"])
        self.mu, self.sd = z["mu"], z["sd"]
        n = int(z["n_members"]) if "n_members" in z else len({k.split("_")[0] for k in z.files if k.startswith("m") and k[1].isdigit()})
        self.members = []
        for k in range(n):
            names = [f for f in z.files if f.startswith(f"m{k}_")]
            m = Reward(len(self.mu), self.kind)
            m.p = {f.split("_", 1)[1]: z[f] for f in names}
            self.members.append(m)
        self.feat_w, self.feat_mu, self.feat_sd = z["feat_w"], z["feat_mu"], z["feat_sd"]
        self.feat_names = [str(x) for x in z["feat_names"]]
        self._embedder = None

    def score_embeddings(self, E: np.ndarray) -> tuple:
        X = (np.asarray(E, np.float64) - self.mu) / self.sd
        R = np.stack([m(X) for m in self.members])
        return R.mean(0), R.std(0)

    def score_images(self, images: list) -> tuple:
        if self._embedder is None:
            self._embedder = Embedder(self.model)
        return self.score_embeddings(self._embedder(images))

    def score_features(self, rows: list) -> np.ndarray:
        """rows: dicts with the five statistics (speed, change, size_rg, mass, texture)."""
        F = np.array([[float(r[f]) for f in self.feat_names] for r in rows], np.float64)
        for j, f in enumerate(self.feat_names):
            if f in ("speed", "change", "mass"):
                F[:, j] = np.log(np.maximum(F[:, j], 1e-4))
        return np.nan_to_num((F - self.feat_mu) / self.feat_sd) @ self.feat_w


def evaluate(pool_dir, scorers: dict, ids: list, cre: dict, judge: str, prompt: str, held_pairs: list, L: dict) -> dict:
    """Held-out agreement of each scorer with the teacher, the other judges and your ratings."""
    from .vlmtest import _human_rows
    idx = {i: k for k, i in enumerate(ids)}
    fl = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    others = {}
    for p in sorted(Path(pool_dir).glob(f"labels_*_{prompt}.csv")):
        j = p.stem[len("labels_"):-len(prompt) - 1]
        if j != judge:
            others[j] = _labels(pool_dir, j, prompt)
    hum = {(r["a"], r["b"]): float(r["winner"] == r["a"]) for r in _human_rows(Path(pool_dir)) if r["choice"] in ("left", "right")}
    size = lambda i: fl(cre[i]["size_rg"])
    matched = [k for k in held_pairs if max(size(k[0]), size(k[1])) / max(min(size(k[0]), size(k[1])), 1e-9) < 1.1]
    decisive = [k for k in held_pairs if abs(L[k][0] - 0.5) > 0.25]
    res = {}
    for name, sc in scorers.items():
        agree = lambda pairs, target: (round(float(np.mean([(sc[idx[a]] > sc[idx[b]]) == (target[(a, b)] > 0.5)
                                                             for a, b in pairs])), 3), len(pairs)) if pairs else None
        t = {k: L[k][0] for k in held_pairs}
        r = {"teacher": agree(held_pairs, t), "teacher_decisive": agree(decisive, t),
             "teacher_size_matched": agree(matched, t)}
        for j, Lj in others.items():
            hp = [k for k in held_pairs if k in Lj]
            r[j] = agree(hp, {k: Lj[k][0] for k in hp})
        hp = [k for k in hum if k[0] in idx and k[1] in idx]
        r["you"] = agree(hp, hum)
        res[name] = r
    return res


def _sheets(pool_dir: Path, out: Path, ids: list, cre: dict, score) -> None:
    from .vlmtest import _contact
    held = [k for k, i in enumerate(ids) if cre[i]["split"] == "heldout"]
    order = sorted(held, key=lambda k: -score[k])
    for tag, sel in (("top", order[:12]), ("bottom", order[-12:])):
        _contact([(f"{ids[k]} {cre[ids[k]]['group']} {score[k]:+.2f}", Path(pool_dir) / "images" / f"{ids[k]}.png") for k in sel],
                 out / f"{tag}_heldout.png", f"student: {tag} 12 held-out creatures")


def _markdown(out: Path, r: dict) -> None:
    cols = ["teacher", "teacher_decisive", "teacher_size_matched"] + sorted(
        {k for v in r["eval"].values() for k in v} - {"teacher", "teacher_decisive", "teacher_size_matched", "you"}) + ["you"]
    L = [f"# Student: {r['model']} -> {r['judge']}/{r['prompt']} ({r['kind']}, ensemble {r['ensemble']}, L2 {r['lambda']})", "",
         f"Trained on {r['train_pairs']} pairs over {r['train_creatures']} creatures; evaluated on {r['heldout_pairs']} held-out pairs "
         "between creatures it never saw. Cells: share of pairs where the scorer picks the same winner (number of pairs). "
         "teacher_size_matched = pairs whose sizes differ by less than 10%, where 'larger wins' cannot help.", "",
         "| scorer | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, v in r["eval"].items():
        L.append(f"| {name} | " + " | ".join(f"{v[c][0]:.3f} ({v[c][1]})" if v.get(c) else "-" for c in cols) + " |")
    L += ["", "Feature reward weights (per standard deviation): " + ", ".join(f"{k} {w:+.2f}" for k, w in r["feature_weights"].items())]
    (out / "report.md").write_text("\n".join(L) + "\n")
