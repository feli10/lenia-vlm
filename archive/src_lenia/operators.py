"""Variation operators for the mutation-operator pilot (pilot.py), beyond `explore.mutate`.

`explore.mutate` nudges m, s, r, h and the rings of about half the kernels and starts the child from the
parent's pattern. It never changes the number of kernels, the channel wiring or the starting pattern, and
never combines two creatures, which is why the MAP-Elites runs stayed near the pool. The operators here do:

  pattern     same rule, new start: `reseed` (a seed-bank pattern at the rule's R), `cut` (the parent with a
              half-plane through its centre wiped), `merge` (the parent overlaid with another creature)
  structural  `add` (a weak new kernel with a random source channel), `remove` (drop a kernel), `duplicate`
              (split a kernel's weight with a mutated copy, so the zero-step copy is the parent's rule),
              `rewire` (move a kernel's source or target channel); each followed by a short re-tune of m, s
              (`nudge_ms`), done by the pilot
  crossover   `crossover`: two parents with the same layout (`Rule.structure()` and T); for each target
              channel, its kernels come from one parent or the other (the update of channel c depends only
              on the kernels that write to c, so blocks keep each channel's own dynamics intact)

`copy` (the parent unchanged, through the same checks) is the pilot's baseline: the noise floor of the
distances and a check that elites come back viable.

Structural edits never leave a channel with no incoming kernel (it would freeze: its update is the
h-weighted mean of incoming growths). An operator that cannot apply to a rule returns None.
All functions take an explicit numpy Generator and leave their inputs unchanged.
"""
from __future__ import annotations

import copy

import numpy as np

from . import seeds as seedbank
from .explore import mutate, mutate_kernel
from .sim import Kernel, Rule


def _rule(rule: Rule, kernels: list) -> Rule:
    return Rule([copy.deepcopy(k) for k in kernels], R=rule.R, T=rule.T, channels=rule.channels, kn=rule.kn, gn=rule.gn,
                flow=rule.flow, beta=rule.beta)


def incoming(rule: Rule) -> np.ndarray:
    """Number of kernels writing to each channel."""
    n = np.zeros(rule.channels, int)
    for k in rule.kernels:
        n[k.c1] += 1
    return n


def valid(rule: Rule) -> bool:
    """Every channel is written to by at least one kernel with positive weight, channels in range."""
    if not rule.kernels:
        return False
    if any(not (0 <= k.c0 < rule.channels and 0 <= k.c1 < rule.channels) for k in rule.kernels):
        return False
    w = np.zeros(rule.channels)
    for k in rule.kernels:
        w[k.c1] += k.h
    return bool((w > 0).all())


def layout(rule: Rule) -> tuple:
    """Rules with the same layout can be crossed (and share a simulation batch)."""
    return rule.structure() + (rule.T,)


# ---------- parameter mutations ----------

def scaled(rule: Rule, rng, scale: float) -> Rule:
    """The current mutation (explore.mutate) at `scale` times its step sizes."""
    return mutate(rule, rng, scale)


def nudge_ms(rule: Rule, rng, scale: float = 0.5) -> Rule:
    """A small step in m and s of every kernel (the re-tune after a structural edit); rings, r, h, wiring kept."""
    ks = []
    for k in rule.kernels:
        k = copy.deepcopy(k)
        k.m = float(np.clip(k.m + rng.normal(0, 0.01 * scale), 0.05, 0.6))
        k.s = float(np.clip(k.s * np.exp(rng.normal(0, 0.08 * scale)), 0.003, 0.2))
        ks.append(k)
    return _rule(rule, ks)


# ---------- structural edits ----------

def add_kernel(rule: Rule, rng, weight: float = 0.25, scale: float = 2.0) -> Rule:
    """A new kernel: a mutated copy of a random kernel with a random source channel, at `weight` times the
    template's weight (a new full-weight kernel almost always kills the creature)."""
    t = rule.kernels[int(rng.integers(len(rule.kernels)))]
    k = mutate_kernel(t, rng, True, scale)
    k.c0 = int(rng.integers(rule.channels))
    k.c1 = t.c1
    k.h = float(np.clip(t.h * weight, 0.02, 1.0))
    return _rule(rule, list(rule.kernels) + [k])


def remove_kernel(rule: Rule, rng) -> Rule | None:
    """Drop a random kernel whose target channel keeps another incoming kernel."""
    n = incoming(rule)
    ok = [i for i, k in enumerate(rule.kernels) if n[k.c1] > 1]
    if not ok:
        return None
    i = ok[int(rng.integers(len(ok)))]
    return _rule(rule, [k for j, k in enumerate(rule.kernels) if j != i])


def duplicate_kernel(rule: Rule, rng, scale: float = 2.0) -> Rule:
    """Split a random kernel into two with half the weight each (the same rule so far) and mutate one copy
    (`scale` times the usual step), so the copy can drift into a new role."""
    i = int(rng.integers(len(rule.kernels)))
    a = copy.deepcopy(rule.kernels[i])
    a.h = a.h / 2
    b = mutate_kernel(a, rng, True, scale)
    b.h = a.h
    ks = list(rule.kernels)
    ks[i] = a
    return _rule(rule, ks + [b])


def rewire(rule: Rule, rng) -> Rule | None:
    """Move one kernel's source channel (c0) or, if its old target keeps another incoming kernel, its target
    channel (c1) to another channel."""
    C = rule.channels
    if C < 2:
        return None
    n = incoming(rule)
    moves = [(i, "c0") for i in range(len(rule.kernels))] + [(i, "c1") for i, k in enumerate(rule.kernels) if n[k.c1] > 1]
    i, which = moves[int(rng.integers(len(moves)))]
    ks = [copy.deepcopy(k) for k in rule.kernels]
    old = getattr(ks[i], which)
    setattr(ks[i], which, int(rng.choice([c for c in range(C) if c != old])))
    return _rule(rule, ks)


STRUCTURAL = {"add": add_kernel, "remove": remove_kernel, "duplicate": duplicate_kernel, "rewire": rewire}


# ---------- crossover ----------

def crossover(a: Rule, b: Rule, rng) -> Rule | None:
    """For each target channel, its kernels from a or from b (at least one channel from each). Needs the same
    layout; with one channel, kernels are swapped one by one instead (at least one from each)."""
    if layout(a) != layout(b) or len(a.kernels) < 2:
        return None
    C = a.channels
    if C > 1:
        while True:
            take_b = rng.random(C) < 0.5
            if 0 < take_b.sum() < C:
                break
        ks = [kb if take_b[ka.c1] else ka for ka, kb in zip(a.kernels, b.kernels)]
    else:
        while True:
            take_b = rng.random(len(a.kernels)) < 0.5
            if 0 < take_b.sum() < len(a.kernels):
                break
        ks = [kb if t else ka for ka, kb, t in zip(a.kernels, b.kernels, take_b)]
    return _rule(a, ks)


# ---------- pattern operators (same rule, new start) ----------

def _centred(P: np.ndarray, n: int) -> np.ndarray:
    """(C, h, w) -> (C, n, n), centred, cropped or zero-padded."""
    C, h, w = P.shape
    out = np.zeros((C, n, n), np.float32)
    sy, sx = max(0, (h - n) // 2), max(0, (w - n) // 2)
    P = P[:, sy:sy + n, sx:sx + n]
    h, w = P.shape[-2:]
    y0, x0 = (n - h) // 2, (n - w) // 2
    out[:, y0:y0 + h, x0:x0 + w] = P
    return out


def reseed(rule: Rule, rng, size: int) -> np.ndarray:
    """A start from the seed bank (a donor creature's shape or smoothed noise) at the rule's R, as a (C, n, n)
    pattern with n = `size` (pass the parent pattern's size, so the start fits the world)."""
    spec = dict(seedbank.DEFAULT_BANK[int(rng.integers(len(seedbank.DEFAULT_BANK)))])
    spec["seed"] = int(rng.integers(1 << 30))           # donors: random channel strengths; noise: a new draw
    A = seedbank.make(spec, rule.R, (size, size), rule.channels)
    return np.asarray(A, np.float32)


def cut(P: np.ndarray, rng, soft: float = 2.0) -> np.ndarray:
    """The pattern with a half-plane through its centre of mass wiped (random angle, edge `soft` cells wide)."""
    P = np.asarray(P, np.float32)
    m = P.sum(0)
    H, W = m.shape
    y, x = np.mgrid[:H, :W]
    tot = max(float(m.sum()), 1e-9)
    cy, cx = float((m * y).sum() / tot), float((m * x).sum() / tot)
    th = rng.uniform(0, 2 * np.pi)
    d = (y - cy) * np.sin(th) + (x - cx) * np.cos(th)
    keep = 1 / (1 + np.exp(-d / max(soft, 1e-3)))
    return (P * keep[None]).astype(np.float32)


def merge(P: np.ndarray, Q: np.ndarray, rng) -> np.ndarray:
    """Two creatures overlaid: the partner turned by a random multiple of 90 degrees (and mirrored half the
    time), both centred, combined by the cell-wise maximum. Channel counts must match."""
    P, Q = np.asarray(P, np.float32), np.asarray(Q, np.float32)
    if P.shape[0] != Q.shape[0]:
        raise ValueError("merge: channel counts differ")
    Q = np.rot90(Q, int(rng.integers(4)), axes=(1, 2))
    if rng.random() < 0.5:
        Q = Q[:, :, ::-1]
    n = max(P.shape[1], P.shape[2], Q.shape[1], Q.shape[2])
    return np.maximum(_centred(P, n), _centred(np.ascontiguousarray(Q), n))


PATTERN = ("reseed", "cut", "merge")
OPERATORS = ("copy", "mut1", "mut2", "mut4", "reseed", "cut", "merge", "add", "remove", "duplicate", "rewire", "crossover")
DESCRIPTION = {
    "copy": "the parent unchanged (baseline: noise floor)",
    "mut1": "current mutation, 1x", "mut2": "current mutation, 2x", "mut4": "current mutation, 4x",
    "reseed": "same rule, seed-bank start", "cut": "same rule, half the parent wiped",
    "merge": "same rule, parent + another creature overlaid",
    "add": "add a weak kernel (+ re-tune)", "remove": "remove a kernel (+ re-tune)",
    "duplicate": "duplicate a kernel, mutate the copy (+ re-tune)", "rewire": "rewire one kernel's channel (+ re-tune)",
    "crossover": "target-channel blocks from two parents, same layout",
}
