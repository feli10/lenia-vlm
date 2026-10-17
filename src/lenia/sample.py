"""Random rules, following the ranges Chan's own random search uses (LeniaNDKC.py, random_params).

  classic       one channel, one kernel: m, s and 1-3 rings (b in twelfths, one of them 1)
  grid          classic with a single ring on an even m x s grid: a map of the (m, s) plane
  multikernel   one channel, several kernels, each with its own rings, m, s, h and relative radius r
  multichannel  C channels; `self_k` kernels from each channel to itself and `cross_k` from each
                channel to each other one (Chan's "expanded universe")

R and T are fixed per run (Lenia is roughly scale-free, so varying R mostly changes the cost).
"""
from __future__ import annotations

import numpy as np

from .sim import Kernel, Rule


def _rings(rng) -> tuple:
    B = int(rng.integers(1, 4))
    b = [int(rng.integers(0, 13)) / 12 for _ in range(B)]
    b[int(rng.integers(B))] = 1.0
    return tuple(b)


def _kernel(rng, c0=0, c1=0, multi=False) -> Kernel:
    m = float(rng.uniform(0.1, 0.5))
    s = float(m / 10 * rng.uniform(0.5, 3.0))
    if not multi:
        return Kernel(m=round(m, 4), s=round(s, 5), b=_rings(rng))
    return Kernel(m=round(m, 4), s=round(s, 5), b=_rings(rng), r=round(float(rng.uniform(0.5, 1.0)), 3),
                  h=round(float(rng.uniform(0.1, 1.0)), 3), c0=c0, c1=c1)


def sample(mode: str, n: int, seed: int, R: float = 13, T: float = 10, kernels: int = 3, channels: int = 3,
           self_k: int = 2, cross_k: int = 1, grid_m=(0.1, 0.5), grid_s=(0.005, 0.08)) -> list:
    rng = np.random.default_rng(seed)
    out = []
    if mode == "classic":
        for _ in range(n):
            out.append(Rule([_kernel(rng)], R=R, T=T))
    elif mode == "grid":
        k = int(round(np.sqrt(n)))
        ms = np.linspace(*grid_m, k)
        ss = np.geomspace(*grid_s, k)
        for s in ss:
            for m in ms:
                out.append(Rule([Kernel(m=round(float(m), 4), s=round(float(s), 5))], R=R, T=T))
    elif mode == "multikernel":
        for _ in range(n):
            out.append(Rule([_kernel(rng, multi=True) for _ in range(kernels)], R=R, T=T))
    elif mode == "multichannel":
        for _ in range(n):
            ks = []
            for c in range(channels):
                ks += [_kernel(rng, c, c, multi=True) for _ in range(self_k)]
            for c0 in range(channels):
                for c1 in range(channels):
                    if c0 != c1:
                        ks += [_kernel(rng, c0, c1, multi=True) for _ in range(cross_k)]
            out.append(Rule(ks, R=R, T=T, channels=channels))
    else:
        raise ValueError(f"unknown mode {mode}")
    return out
