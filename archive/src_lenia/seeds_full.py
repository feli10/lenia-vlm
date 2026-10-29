"""The fixed bank of starting patterns every rule is tried from.

Lenia forgets most of its starting pattern, but whether anything survives depends strongly on it:
pixel noise almost never grows a creature, even under rules that have well-known creatures; smooth
noise about 1.5 kernel radii wide sometimes does, and the shapes of known creatures (rescaled to the
rule's kernel) do best (docs/lenia.md, calibration on Chan's catalogue). A rule is judged by its
outcomes over the whole bank, so its score does not hinge on one lucky or unlucky start.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rle import rle2arr
from .sim import place


def _blur(z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with FFTs (periodic), so we need nothing beyond numpy."""
    if sigma <= 0:
        return z
    H, W = z.shape
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(z) * g))


def smooth_noise(R: float, shape, C: int, seed: int, size_r: float = 1.5, sigma_r: float = 0.25) -> np.ndarray:
    """A square of smoothed uniform noise, size_r kernel radii wide, rescaled to [0, 1], per channel."""
    rng = np.random.default_rng(seed)
    n = max(4, int(round(size_r * R)))
    pad = int(np.ceil(3 * sigma_r * R))
    out = []
    for _ in range(C):
        z = np.zeros((n + 2 * pad, n + 2 * pad))
        z[pad:pad + n, pad:pad + n] = rng.random((n, n))
        z = _blur(z, sigma_r * R)[pad:pad + n, pad:pad + n]
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        out.append(z)
    return place(np.stack(out), shape, C)


def blob(R: float, shape, C: int, seed: int, radius_r: float = 0.8, peak: float = 1.0) -> np.ndarray:
    """A soft round bump; the seed jitters its size and, with several channels, its offset per channel."""
    rng = np.random.default_rng(seed)
    H, W = shape
    y, x = np.mgrid[:H, :W]
    out = np.zeros((C, H, W), np.float32)
    for c in range(C):
        rad = radius_r * R * (1 + 0.15 * rng.standard_normal())
        oy, ox = (rng.standard_normal(2) * 0.3 * R) if C > 1 else (0, 0)
        d = np.hypot(y - H / 2 - oy, x - W / 2 - ox) / max(rad, 1)
        out[c] = peak * np.exp(-2 * d ** 2)
    return out


_DONORS = None


def donors() -> dict:
    """Five creatures from Chan's catalogue (MIT licence), stored in donors.json next to this file."""
    global _DONORS
    if _DONORS is None:
        _DONORS = json.loads((Path(__file__).parent / "donors.json").read_text())
    return _DONORS


def resize(cells: np.ndarray, factor: float) -> np.ndarray:
    h, w = cells.shape
    im = Image.fromarray(np.clip(cells * 255, 0, 255).astype(np.uint8))
    im = im.resize((max(2, round(w * factor)), max(2, round(h * factor))), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255


def donor(R: float, shape, C: int, seed: int, code: str = "O2u") -> np.ndarray:
    """A known creature's shape, rescaled from its own kernel radius to R. With several channels each
    channel gets the shape at a random strength (0.5-1), so the channels start out different."""
    d = donors()[code]
    cells = resize(rle2arr(d["cells"]), R / d["R"])
    if C == 1:
        return place(cells, shape, 1)
    rng = np.random.default_rng(seed)
    return place(np.stack([cells * rng.uniform(0.5, 1.0) for _ in range(C)]), shape, C)


# Calibrated on Chan's catalogue (docs/lenia.md, 2026-10-14): from these eight starts, 45% of 80
# catalogue rules grow a living creature (84% do from their own catalogued pattern); pixel noise
# finds almost none. S1s and PG1a do most of the work.
DEFAULT_BANK = [
    {"kind": "donor", "seed": 0, "code": "O2u"},
    {"kind": "donor", "seed": 1, "code": "S1s"},
    {"kind": "donor", "seed": 2, "code": "K4s"},
    {"kind": "donor", "seed": 3, "code": "PG1a"},
    {"kind": "donor", "seed": 4, "code": "H3s"},
    {"kind": "smooth", "seed": 5, "size_r": 1.5},
    {"kind": "smooth", "seed": 6, "size_r": 1.5},
    {"kind": "smooth", "seed": 7, "size_r": 1.5},
]


def make(spec: dict, R: float, shape, C: int) -> np.ndarray:
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "smooth":
        return smooth_noise(R, shape, C, **spec)
    if kind == "blob":
        return blob(R, shape, C, **spec)
    if kind == "donor":
        return donor(R, shape, C, **spec)
    raise ValueError(f"unknown seed kind {kind}")


def label(spec: dict) -> str:
    extra = ",".join(f"{k}={v}" for k, v in spec.items() if k not in ("kind", "seed"))
    return f"{spec['kind']}#{spec['seed']}({extra})"
