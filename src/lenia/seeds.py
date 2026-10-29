"""Starting patterns: a square of smoothed noise (every MaCE world grows from one, lenia.mace.founders and
mace_qd's fixed start). Trimmed on 2026-10-28; the classic-Lenia seed bank (blobs, Chan's donor creatures) is in
archive/src_lenia/seeds_full.py.
"""
from __future__ import annotations

import numpy as np

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
