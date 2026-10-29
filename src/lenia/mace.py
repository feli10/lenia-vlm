"""MaCE-Lenia worlds as the search sees them: simulate a batch, draw the 2x2 picture the judges see, the founder
rules, rule mutation, the grid-noise (texture) measure, and the complexity prompt.

Moved here verbatim on 2026-10-28 from scripts/mace_evo.py (the retired hill climber: simulate, picture,
founders, the complexity prompt) and src/lenia/explore.py (mutate, texture) and src/lenia/alive.py (bodies), all now in archive/. Do not edit
picture(), the prompt text or mutate() casually: judge caches are keyed on the picture bytes and the prompt
text, and saved runs depend on mutate(). tests/test_golden.py guards all three.
"""
from __future__ import annotations

import copy

import numpy as np
from PIL import Image

from . import backend, render, seeds
from .prompts import PROMPTS
from .prompts import COMPLEXITY as PROMPT_ID   # the complexity prompt (text in prompts.py)
from .sample import sample
from .sim import Kernel, Rule, with_flow


FRAMES, TILE = 4, 320                        # 4 snapshots, each drawn 320 px (image 640x640)


# ---------- simulation and pictures ----------

def simulate(rules: list, worlds: np.ndarray, steps: int, shape, gap: int = 400) -> tuple[np.ndarray, list]:
    """Run one batch; return final worlds and, per world, the last FRAMES snapshots `gap` steps apart."""
    gap = min(gap, steps // (FRAMES - 1))
    b = backend.make_batch(rules, shape)
    A = backend.to_device(worlds)
    keep = {steps - i * gap for i in range(FRAMES)}
    film = [np.asarray(worlds, np.float32).copy()] if 0 in keep else []
    for t in range(1, steps + 1):
        A = b.step(A)
        if t in keep:
            film.append(backend.to_numpy(A).copy())
    film = np.stack(film, 1)                                  # (B, FRAMES, C, H, W)
    return film[:, -1], [f for f in film]


def picture(frames: np.ndarray) -> np.ndarray:
    """(FRAMES, C, H, W) -> 2x2 composite (uint8); one brightness scale shared by the four snapshots."""
    scale = max(float(np.percentile(frames, 99.5)), 1e-6)
    tiles = [np.asarray(Image.fromarray(render.colorize(np.clip(f / scale, 0, 1))).resize((TILE, TILE), Image.BILINEAR))
             for f in frames]
    gap = np.full((TILE, 4, 3), 40, np.uint8)
    rows = [np.concatenate([tiles[0], gap, tiles[1]], 1), np.concatenate([tiles[2], gap, tiles[3]], 1)]
    return np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), 40, np.uint8), rows[1]], 0)


def founders(args, shape):
    rules = sample("multichannel", args.lineages, seed=args.rule_seed, R=args.R, channels=3, self_k=1, cross_k=1)
    rules = [with_flow(r, "mace", args.beta) for r in rules]
    A0 = np.stack([seeds.smooth_noise(args.R, shape, 3, seed=i, size_r=3.0) for i in range(len(rules))])
    return rules, A0


# ---------- variation and the grid-noise rule ----------

def _twelfths(x: float) -> float:
    return round(min(1.0, max(0.0, x)) * 12) / 12


def mutate_kernel(k: Kernel, rng, multi: bool, scale: float = 1.0, struct: float = 1.0) -> Kernel:
    k = copy.deepcopy(k)
    k.m = float(np.clip(k.m + rng.normal(0, 0.01 * scale), 0.05, 0.6))
    k.s = float(np.clip(k.s * np.exp(rng.normal(0, 0.08 * scale)), 0.003, 0.2))
    b = list(k.b)
    u = rng.random()
    # struct scales the chance of these discrete ring edits (1 = as always; 0 = continuous parameters only)
    if u < 0.15 * struct:                                   # nudge one ring by a twelfth
        i = int(rng.integers(len(b)))
        b[i] = _twelfths(b[i] + rng.choice([-1, 1]) / 12)
    elif u < 0.20 * struct and len(b) < 3:                  # add a ring
        b.insert(int(rng.integers(len(b) + 1)), _twelfths(rng.random()))
    elif u < 0.25 * struct and len(b) > 1:                  # drop a ring
        b.pop(int(rng.integers(len(b))))
    if max(b) < 1:
        b[int(np.argmax(b))] = 1.0
    k.b = tuple(b)
    if multi:
        k.r = float(np.clip(k.r + rng.normal(0, 0.03 * scale), 0.3, 1.0))
        k.h = float(np.clip(k.h + rng.normal(0, 0.05 * scale), 0.05, 1.0))
    return k


def mutate(rule: Rule, rng, scale: float = 1.0, struct: float = 1.0) -> Rule:
    multi = len(rule.kernels) > 1
    ks = [mutate_kernel(k, rng, multi, scale, struct) if (not multi or rng.random() < 0.5) else copy.deepcopy(k)
          for k in rule.kernels]
    return Rule(ks, R=rule.R, T=rule.T, channels=rule.channels, kn=rule.kn, gn=rule.gn,
                flow=rule.flow, beta=rule.beta)


TEXTURE_CUTOFF = 0.25      # cycles per cell: wavelengths shorter than 4 cells count as grid-scale


def texture(P: np.ndarray, cutoff: float = TEXTURE_CUTOFF) -> float:
    """Share of a creature's spatial power at wavelengths below 1/cutoff cells (mean removed per channel;
    channels weighted by their mass). Experiment A, first run: archive median 0.03; a striped family at
    R = 12 0.20-0.32 (its stripes turned out to be real: they widen with R)."""
    P = np.asarray(P, np.float32)
    P = P[None] if P.ndim == 2 else P
    fy = np.fft.fftfreq(P.shape[-2])[:, None]
    fx = np.fft.fftfreq(P.shape[-1])[None]
    hi = np.hypot(fy, fx) > cutoff
    num = den = 0.0
    for c in P:
        m = float(c.sum())
        if m <= 0:
            continue
        F = np.abs(np.fft.fft2(c - c.mean())) ** 2
        num += m * float(F[hi].sum()) / max(float(F.sum()), 1e-12)
        den += m
    return num / den if den else 0.0


# ---------- bodies (connected components on the torus; the map's bodies axis) ----------

def bodies(mask: np.ndarray, max_iter: int = 4096) -> np.ndarray:
    """Connected components of a boolean mask on the torus (4-neighbour), by min-label propagation."""
    H, W = mask.shape
    big = H * W
    lab = np.where(mask, np.arange(big).reshape(H, W), big)
    for _ in range(max_iter):
        n = lab
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            n = np.minimum(n, np.roll(lab, sh, axis=ax))
        n = np.where(mask, n, big)
        if np.array_equal(n, lab):
            break
        lab = n
    return lab
