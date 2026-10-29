"""Colour for MaCE pictures: fixed colour maps; two or three channels map straight to RGB.

Trimmed on 2026-10-28 to what the live pipeline uses (colorize, called by lenia.mace.picture); the filmstrips,
crops and contact sheets of the classic-Lenia phase are in archive/src_lenia/render_full.py.
"""
from __future__ import annotations

import numpy as np

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
