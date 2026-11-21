"""Embedding check for GIF targets (11-21): can CLIP / DINOv2 (and raw pixels) see a biological event over time,
and how far are our MaCE renders from real footage? No search, no judge calls; nothing billed.

  cd ~/Projects/lenia-vlm && PYTHONPATH=src:scripts python3 scripts/embedcheck.py \\
      --gif "stem=RUNS/targets/stem_mitosis.gif:event=30-37" \\
      --gif "bud=RUNS/targets/budding_hub.gif:crop_top=32" \\
      --out RUNS/targets/embedcheck                      # Colab: downloads CLIP and DINOv2 once
  Offline dry run: add --fm fake (no PyTorch; fake embeddings, so the plots and report are placeholders).

GIF spec: name=path[:crop_top=N][:event=A-B]. crop_top removes a title bar (our catalog GIFs have 32 px);
event marks frames A..B on every plot (a visual read, not ground truth).

For every GIF, style and model:
  styles: raw; grey (luminance); soft (grey + Gaussian blur, removes speckle: a crude common look)
  models: clip (OpenAI ViT-B/32, ASAL's model), dino (DINOv2 ViT-B/14), pixels (32x32 grey, z-scored)
Outputs in --out:
  selfsim_<model>.png  frames x frames similarity, one panel per GIF and style
  curves_<model>.png   similarity of each frame to the previous one and to frame 0, and a change-point score
  cross_<model>.png    frames(GIF a) x frames(GIF b) for every pair, per style
  pca_<model>.png      all frames of all GIFs on the same two principal axes, per style
  text_probes.png      CLIP similarity to a few prompts over time (ASAL's text-target view)
  report.md            numbers: where the biggest frame-to-frame jump is and how big, cross-GIF similarity
  emb.npz              every embedding
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageSequence

STYLES = ("raw", "grey", "soft")
PROMPTS = ("a single cell", "a cell dividing into two", "two cells")


def parse_gif(spec: str) -> dict:
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    g = {"name": name, "path": parts[0], "crop_top": 0, "event": None}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if k == "crop_top":
            g["crop_top"] = int(v)
        elif k == "event":
            a, _, b = v.partition("-")
            g["event"] = (int(a), int(b))
        else:
            raise SystemExit(f"unknown option {k!r} in {spec!r}")
    return g


def load_frames(g: dict) -> list:
    im = Image.open(g["path"])
    out = []
    for f in ImageSequence.Iterator(im):
        f = f.convert("RGB")
        if g["crop_top"]:
            f = f.crop((0, g["crop_top"], f.width, f.height))
        out.append(f.copy())
    return out


def style(frames: list, s: str) -> list:
    if s == "raw":
        return frames
    grey = [f.convert("L").convert("RGB") for f in frames]
    if s == "grey":
        return grey
    r = max(1.0, min(frames[0].size) / 150)               # ~2 px blur at 300 px
    return [f.filter(ImageFilter.GaussianBlur(r)) for f in grey]


class Models:
    def __init__(self, fm: str):
        from lenia.embed import ClipScorer, Embedder
        fake = fm == "fake"
        self.clip = ClipScorer("fake" if fake else "openai/clip-vit-base-patch32")
        self.dino = Embedder("fake" if fake else "dinov2_vitb14")

    def embed(self, model: str, frames: list) -> np.ndarray:
        if model == "clip":
            return self.clip.images([np.asarray(f) for f in frames])
        if model == "dino":
            v = self.dino([f.resize((224, 224)) for f in frames])
        else:                                                 # pixels
            v = np.stack([np.asarray(f.convert("L").resize((32, 32)), np.float32).ravel() for f in frames])
            v = v - v.mean(1, keepdims=True)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def novelty(E: np.ndarray, k: int) -> np.ndarray:
    """Foote's change-point score on the self-similarity matrix S: for a boundary at frame t, the mean similarity
    within the k frames before and within the k frames after, minus the similarity across the boundary.
    High where one phase ends and another begins; ~0 inside a phase. NaN where the window doesn't fit."""
    S = E @ E.T
    n = len(E)
    out = np.full(n, np.nan)
    for t in range(k, n - k + 1):
        a, b = slice(t - k, t), slice(t, t + k)
        out[t] = S[a, a].mean() + S[b, b].mean() - 2 * S[a, b].mean()
    return out


def jump_stats(E: np.ndarray, k: int) -> dict:
    """Where the change-point score peaks and how far above its median it is; plus the noisiest-frame view."""
    nv = novelty(E, k)
    ok = ~np.isnan(nv)
    i = int(np.nanargmax(nv))
    med = float(np.median(nv[ok]))
    mad = float(np.median(np.abs(nv[ok] - med))) or 1e-9
    d = 1 - (E[1:] * E[:-1]).sum(1)
    return {"peak_frame": i, "peak": round(float(nv[i]), 4), "median": round(med, 4),
            "z": round((float(nv[i]) - med) / (1.4826 * mad), 1),
            "frame_to_frame_median": round(float(np.median(d)), 4), "sim0_min": round(float((E @ E[0]).min()), 3)}


def shade(ax, ev, axis="x"):
    if ev:
        (ax.axvspan if axis == "x" else ax.axhspan)(ev[0], ev[1], color="orange", alpha=0.25, lw=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", action="append", required=True, help="name=path[:crop_top=N][:event=A-B]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fm", default="real", choices=["real", "fake"], help="fake: no PyTorch (dry run)")
    ap.add_argument("--models", default="clip,dino,pixels")
    ap.add_argument("--k", type=int, default=5, help="change-point window in frames (default 5)")
    ap.add_argument("--prompt", action="append", help=f"CLIP text probe (default: {list(PROMPTS)})")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gifs = [parse_gif(s) for s in a.gif]
    frames = {g["name"]: load_frames(g) for g in gifs}
    models = a.models.split(",")
    M = Models(a.fm)
    E = {(m, g["name"], s): M.embed(m, style(frames[g["name"]], s)) for m in models for g in gifs for s in STYLES}
    np.savez_compressed(out / "emb.npz", **{f"{m}|{n}|{s}": v for (m, n, s), v in E.items()})
    names = [g["name"] for g in gifs]
    ev = {g["name"]: g["event"] for g in gifs}

    rep = [f"# Embedding check ({'FAKE embeddings: dry run' if a.fm == 'fake' else 'CLIP ViT-B/32, DINOv2 ViT-B/14, pixels'})",
           "", "GIFs: " + "; ".join(f"{g['name']} = {g['path']} ({len(frames[g['name']])} frames, "
                                       f"{frames[g['name']][0].size[0]}x{frames[g['name']][0].size[1]}"
                                       + (f", event {g['event'][0]}-{g['event'][1]}" if g["event"] else "") + ")"
                                       for g in gifs), "",
           "## 1. Does the embedding see the event?",
           f"Change-point score (Foote novelty, window {a.k} frames): the frame where it peaks, the peak, its median, and "
           "how many robust standard deviations above the median the peak is (z). Frame-to-frame median: typical "
           "1 - similarity between consecutive frames (flicker). sim0 min: lowest similarity to frame 0.", "",
           "| model | GIF | style | peak frame | peak | median | z | frame-to-frame median | sim0 min |",
           "|---|---|---|---|---|---|---|---|---|"]
    rep2, rep3 = [], []
    stats = {}
    for m in models:
        for n in names:
            for s in STYLES:
                j = jump_stats(E[(m, n, s)], a.k)
                stats[f"{m}|{n}|{s}"] = j
                rep.append(f"| {m} | {n} | {s} | {j['peak_frame']} | {j['peak']} | {j['median']} | {j['z']} | "
                           f"{j['frame_to_frame_median']} | {j['sim0_min']} |")

        fig, axs = plt.subplots(len(names), len(STYLES), figsize=(4 * len(STYLES), 3.6 * len(names)), squeeze=False)
        for r, n in enumerate(names):
            for c, s in enumerate(STYLES):
                X = E[(m, n, s)]
                ax = axs[r][c]
                im = ax.imshow(X @ X.T, cmap="viridis")
                ax.set_title(f"{n} · {s}", fontsize=9)
                if ev[n]:
                    for v in ev[n]:
                        ax.axvline(v, color="orange", lw=0.8)
                        ax.axhline(v, color="orange", lw=0.8)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{m}: frame x frame similarity (orange: marked event)")
        fig.tight_layout()
        fig.savefig(out / f"selfsim_{m}.png", dpi=110)
        plt.close(fig)

        fig, axs = plt.subplots(len(names), len(STYLES), figsize=(4 * len(STYLES), 2.8 * len(names)), squeeze=False)
        for r, n in enumerate(names):
            for c, s in enumerate(STYLES):
                X = E[(m, n, s)]
                ax = axs[r][c]
                ax.plot(range(1, len(X)), (X[1:] * X[:-1]).sum(1), label="vs previous frame")
                ax.plot(range(len(X)), X @ X[0], label="vs frame 0")
                ax2 = ax.twinx()
                ax2.plot(range(len(X)), novelty(X, a.k), color="green", lw=1, label="change-point score")
                ax2.tick_params(axis="y", colors="green", labelsize=7)
                shade(ax, ev[n])
                ax.set_title(f"{n} · {s}", fontsize=9)
                ax.set_xlabel("frame")
        axs[0][0].legend(fontsize=8)
        fig.suptitle(f"{m}: similarity over time (green, right axis: change-point score)")
        fig.tight_layout()
        fig.savefig(out / f"curves_{m}.png", dpi=110)
        plt.close(fig)

        pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
        if pairs:
            rep2 += ["", f"## 2. Cross-GIF similarity ({m})",
                    "Max and median of frames(a) x frames(b); compare with each GIF's own median off-diagonal similarity.", "",
                    "| a | b | style | max | median | a's own median | b's own median |", "|---|---|---|---|---|---|---|"]
            fig, axs = plt.subplots(len(pairs), len(STYLES), figsize=(4 * len(STYLES), 3.6 * len(pairs)), squeeze=False)
            for r, (x, y) in enumerate(pairs):
                for c, s in enumerate(STYLES):
                    C = E[(m, x, s)] @ E[(m, y, s)].T
                    own = lambda n: float(np.median((E[(m, n, s)] @ E[(m, n, s)].T)[~np.eye(len(E[(m, n, s)]), dtype=bool)]))
                    rep2.append(f"| {x} | {y} | {s} | {C.max():.3f} | {np.median(C):.3f} | {own(x):.3f} | {own(y):.3f} |")
                    ax = axs[r][c]
                    im = ax.imshow(C, cmap="viridis", aspect="auto")
                    ax.set_ylabel(f"{x} frame")
                    ax.set_xlabel(f"{y} frame")
                    ax.set_title(f"{x} x {y} · {s}", fontsize=9)
                    fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"{m}: cross-similarity")
            fig.tight_layout()
            fig.savefig(out / f"cross_{m}.png", dpi=110)
            plt.close(fig)

        fig, axs = plt.subplots(1, len(STYLES), figsize=(4.5 * len(STYLES), 4.2), squeeze=False)
        for c, s in enumerate(STYLES):
            Xs = [E[(m, n, s)] for n in names]
            allX = np.concatenate(Xs)
            mu = allX.mean(0)
            _, _, Vt = np.linalg.svd(allX - mu, full_matrices=False)
            ax = axs[0][c]
            for n, X in zip(names, Xs):
                P = (X - mu) @ Vt[:2].T
                ax.plot(P[:, 0], P[:, 1], "-", lw=0.8, alpha=0.6)
                sc = ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="plasma", s=12)
                ax.annotate(f"{n} start", P[0], fontsize=8)
                if ev[n]:
                    a_, b_ = ev[n]
                    ax.plot(P[a_:b_ + 1, 0], P[a_:b_ + 1, 1], color="orange", lw=2.5, alpha=0.8)
            ax.set_title(f"{s}", fontsize=9)
        fig.colorbar(sc, ax=axs[0][-1], label="frame")
        fig.suptitle(f"{m}: frames on the first two principal axes (orange: marked event)")
        fig.tight_layout()
        fig.savefig(out / f"pca_{m}.png", dpi=110)
        plt.close(fig)

    if "clip" in models:
        prompts = a.prompt or list(PROMPTS)
        fig, axs = plt.subplots(len(names), len(STYLES), figsize=(4 * len(STYLES), 2.8 * len(names)), squeeze=False)
        rep3 += ["", "## 3. CLIP text probes", "Frame where each prompt scores highest, and its range over the clip.", "",
                "| GIF | style | prompt | best frame | min | max |", "|---|---|---|---|---|---|"]
        for r, n in enumerate(names):
            for c, s in enumerate(STYLES):
                X = E[("clip", n, s)]
                ax = axs[r][c]
                for p in prompts:
                    t = X @ M.clip.text(p)
                    ax.plot(t, label=p)
                    rep3.append(f"| {n} | {s} | {p} | {int(np.argmax(t))} | {t.min():.3f} | {t.max():.3f} |")
                shade(ax, ev[n])
                ax.set_title(f"{n} · {s}", fontsize=9)
        axs[0][0].legend(fontsize=7)
        fig.suptitle("CLIP similarity to text prompts over time")
        fig.tight_layout()
        fig.savefig(out / "text_probes.png", dpi=110)
        plt.close(fig)

    (out / "report.md").write_text("\n".join(rep + rep2 + rep3) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out} ({len(names)} GIFs x {len(STYLES)} styles x {len(models)} models)")


if __name__ == "__main__":
    main()
