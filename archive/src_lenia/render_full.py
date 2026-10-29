"""Pictures of rollouts: fixed colour maps, filmstrips, contact sheets.

The colour map is fixed per run on purpose: a VLM's 'aesthetic' preference could otherwise just be a
colour preference. One channel goes through a colour map; two or three channels map straight to RGB.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .alive import centroid, recentre

# (position, RGB) stops; 'chan' imitates the look of Chan's web demo (white background).
CMAPS = {
    "chan": [(0.0, (255, 255, 255)), (0.12, (175, 235, 255)), (0.35, (70, 130, 255)),
             (0.6, (200, 60, 255)), (0.85, (175, 20, 95)), (1.0, (90, 0, 40))],
    "gray": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "magma": [(0.0, (0, 0, 4)), (0.25, (80, 18, 123)), (0.5, (183, 55, 121)), (0.75, (251, 136, 97)),
              (1.0, (252, 253, 191))],
}


def colorize(A: np.ndarray, cmap: str = "chan") -> np.ndarray:
    """(C, H, W) or (H, W) in [0, 1] -> (H, W, 3) uint8."""
    A = np.asarray(A, np.float32)
    if A.ndim == 2:
        A = A[None]
    if A.shape[0] == 1:
        stops = CMAPS[cmap]
        xs = np.array([s[0] for s in stops])
        v = np.clip(A[0], 0, 1)
        rgb = np.stack([np.interp(v, xs, [s[1][c] for s in stops]) for c in range(3)], -1)
        return rgb.astype(np.uint8)
    rgb = np.zeros(A.shape[1:] + (3,), np.float32)
    rgb[..., : min(3, A.shape[0])] = np.moveaxis(A[:3], 0, -1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def extent(A: np.ndarray, occupied: float = 0.05) -> float:
    """Distance from the centroid to the farthest occupied cell (torus-aware), in cells."""
    F = np.asarray(A, np.float32)
    F = F.sum(0) if F.ndim == 3 else F
    if not (F > occupied).any():
        return 0.0
    H, W = F.shape
    cy, cx = centroid(F)
    dy = (np.arange(H) - cy + H / 2) % H - H / 2
    dx = (np.arange(W) - cx + W / 2) % W - W / 2
    d = np.hypot(dy[:, None], dx[None, :])
    return float(d[F > occupied].max())


def crop_size(film_or_frame, R: float, crop_r: float, H: int) -> int:
    """Window size for a strip: at least crop_r kernel radii, enough to hold the largest frame."""
    frames = film_or_frame if isinstance(film_or_frame, list) else [(0, film_or_frame)]
    ext = max(extent(A) for _, A in frames)
    return int(min(H, max(8, round(crop_r * R), round(2 * ext + 0.4 * R))))


def crop(A: np.ndarray, R: float, crop_r: float, n: int | None = None) -> np.ndarray:
    """Square window n cells wide (default crop_r kernel radii), centred on the pattern (torus-aware)."""
    A = np.asarray(A, np.float32)
    if A.ndim == 2:
        A = A[None]
    H, W = A.shape[-2:]
    n = int(min(H, W, n or max(8, round(crop_r * R))))
    Ac = recentre(A, centroid(A.sum(0)))
    y0, x0 = (H - n) // 2, (W - n) // 2
    return Ac[:, y0:y0 + n, x0:x0 + n]


def frame_image(A, R: float, cmap: str, px: int, crop_r: float | None, n: int | None = None) -> Image.Image:
    a = crop(A, R, crop_r, n) if crop_r else np.asarray(A, np.float32)
    return Image.fromarray(colorize(a, cmap)).resize((px, px), Image.NEAREST if a.shape[-1] <= px else Image.BILINEAR)


def filmstrip(film: list, R: float, cmap: str = "chan", px: int = 96, crop_r: float | None = 3.0,
              gap: int = 2, bg=(230, 230, 230)) -> Image.Image:
    """film: [(t, (C, H, W))]. One tile per frame, left to right; all frames share one crop size,
    large enough for the biggest frame (so sizes are comparable along the strip, not across strips)."""
    n = crop_size(film, R, crop_r, min(np.asarray(film[0][1]).shape[-2:])) if crop_r else None
    tiles = [frame_image(A, R, cmap, px, crop_r, n) for _, A in film]
    out = Image.new("RGB", (len(tiles) * px + (len(tiles) - 1) * gap, px), bg)
    for i, t in enumerate(tiles):
        out.paste(t, (i * (px + gap), 0))
    return out


def sheet(items: list, cols: int = 1, title: str | None = None, pad: int = 6, label_h: int = 14,
          bg=(245, 245, 245)) -> Image.Image:
    """items: [(label, PIL image)] laid out in a grid with a one-line label above each."""
    if not items:
        return Image.new("RGB", (200, 40), bg)
    w = max(im.width for _, im in items)
    h = max(im.height for _, im in items)
    rows = (len(items) + cols - 1) // cols
    top = 20 if title else 0
    out = Image.new("RGB", (cols * (w + pad) + pad, top + rows * (h + label_h + pad) + pad), bg)
    d = ImageDraw.Draw(out)
    if title:
        d.text((pad, 4), title, fill=(0, 0, 0))
    for i, (lab, im) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = pad + c * (w + pad), top + pad + r * (h + label_h + pad)
        d.text((x, y), lab, fill=(40, 40, 40))
        out.paste(im, (x, y + label_h))
    return out
