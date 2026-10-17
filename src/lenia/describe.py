"""Cheap, hand-made descriptors of a creature (no learned image model), for novelty and coverage.

From the rollout statistics: speed, shape change, mass and spread (all in kernel-radius units).
From the final frame, re-centred on the creature and sampled on a polar grid out to 3 R:
  - angular spectrum: magnitudes of the Fourier orders 1..8 around the centre, relative to order 0
    (rotation-invariant; picks up symmetry: a 5-armed star peaks at order 5, a glider at order 1)
  - radial profile: mean density in 8 rings from the centre out (holes, shells, solid cores)
20 numbers in all. Distances use a fixed per-dimension scale, set once from a reference set.
"""
from __future__ import annotations

import numpy as np

from .alive import centroid, recentre

N_ORDERS, N_RINGS = 8, 8
NAMES = (["log_speed", "log_change", "log_mass", "rg"] + [f"ang{k}" for k in range(1, N_ORDERS + 1)]
         + [f"rad{k}" for k in range(N_RINGS)])


def polar(F: np.ndarray, R: float, n_r: int = 24, n_th: int = 64, r_max_r: float = 3.0) -> np.ndarray:
    """(n_r, n_th) samples of F around its centroid (nearest cell, torus-aware)."""
    H, W = F.shape
    Fc = recentre(F, centroid(F))
    cy, cx = H / 2, W / 2
    r = (np.arange(n_r) + 0.5) / n_r * r_max_r * R
    th = np.arange(n_th) / n_th * 2 * np.pi
    y = (np.round(cy + r[:, None] * np.sin(th)[None]).astype(int)) % H
    x = (np.round(cx + r[:, None] * np.cos(th)[None]).astype(int)) % W
    return Fc[y, x]


def describe(frame: np.ndarray, stats: dict, R: float) -> np.ndarray:
    F = np.asarray(frame, np.float32)
    F = F.sum(0) if F.ndim == 3 else F
    P = polar(F, R)
    spec = np.abs(np.fft.rfft(P, axis=1))                 # (n_r, n_th/2+1), per ring
    w = P.mean(1, keepdims=True)                          # weight rings by their density
    tot = (spec[:, :1] * w).sum() + 1e-9
    ang = (spec[:, 1:N_ORDERS + 1] * w).sum(0) / tot      # orders 1..8 relative to order 0
    rad = P.mean(1).reshape(N_RINGS, -1).mean(1)
    rad = rad / (rad.max() + 1e-9)
    nz = lambda v: 0.0 if v is None or v != v else float(v)
    return np.concatenate([[np.log10(nz(stats["speed_w"]) + 1e-3), np.log10(nz(stats["change_w"]) + 1e-3),
                            np.log10(nz(stats["mass_end"]) + 1e-3), nz(stats["rg_end"])], ang, rad]).astype(np.float32)


class Space:
    """Per-dimension scales so no single descriptor dominates: the robust spread of a reference set,
    but never below half its standard deviation or 0.05. (Without the floors, a dimension that is
    almost always zero in the reference, like the outermost radial ring, gets a near-zero scale, and
    any creature reaching that far looks enormously novel; the first multikernel run exploited this.)"""

    def __init__(self, ref: np.ndarray):
        ref = np.asarray(ref, np.float32)
        q = np.percentile(ref, [25, 75], axis=0)
        self.scale = np.maximum.reduce([(q[1] - q[0]) / 1.35, 0.5 * ref.std(0), np.full(ref.shape[1], 0.05)])

    def dist(self, a: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Distances from a (d,) to each row of B (n, d)."""
        return np.sqrt((((np.asarray(B) - a) / self.scale) ** 2).sum(-1))

    def knn_mean(self, a: np.ndarray, B: np.ndarray, k: int = 5) -> float:
        if len(B) == 0:
            return float("inf")
        d = np.sort(self.dist(a, B))
        return float(d[:k].mean())

    def nearest(self, a: np.ndarray, B: np.ndarray) -> float:
        return float(self.dist(a, B).min()) if len(B) else float("inf")


def bins(desc: np.ndarray) -> tuple:
    """Coarse cell for coverage counts: moving?, size (4 bins), dominant symmetry order (0 = none, 1..8)."""
    speed = 10 ** desc[0]
    moving = int(speed >= 0.05)
    size = int(np.digitize(desc[3], [0.4, 0.6, 0.9]))
    ang = desc[4:4 + N_ORDERS]
    order = int(np.argmax(ang) + 1) if ang.max() > 0.08 else 0
    return moving, size, order
