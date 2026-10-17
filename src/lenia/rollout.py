"""Run batches of rules from a set of seeds, track statistics, keep a few frames for pictures.

Rollouts that have clearly failed are dropped from the batch at checkpoints (every `stop_every`
units of time), which saves most of the cost of a long horizon: nothing left (fizzled), more than
explode_occupancy of the world filled (exploded), or, from `spread_after` on, more than
spread_occupancy filled at two checkpoints in a row (spread). Their class is final; their
statistics cover the time they ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alive import Thresholds, Tracker
from .backend import make_batch, to_device, to_numpy
from .sim import Rule


@dataclass
class Horizon:
    time: float = 100.0         # units of time per rollout (1,000 steps at T = 10)
    record_every: float = 0.5   # statistics every this many units of time
    frames: int = 6             # frames kept for the filmstrip, evenly over the last `film_span`
    film_span: float = 10.0     # units of time covered by the filmstrip (ends at the last step)
    stop_every: float = 10.0    # early-stop checkpoints (0 = never stop early)
    spread_after: float = 20.0  # spread counts for early stopping only from this time on
    thumb_every: float = 10.0   # a 32x32 thumbnail of the world every this many units (0 = none)
    thumb_px: int = 32

    @staticmethod
    def from_dict(d: dict | None) -> "Horizon":
        return Horizon(**(d or {}))


@dataclass
class Result:
    rule: Rule
    seed: int
    stats: dict
    film: list = field(default_factory=list)     # [(t, (C, H, W) float16)]
    trace: np.ndarray = None                     # (n, 6): t, mass, occupancy, spread, speed, change
    thumbs: list = field(default_factory=list)   # [(t, (px, px) uint8)] of the summed field


def thumbnail(A: np.ndarray, px: int) -> np.ndarray:
    F = np.asarray(A, np.float32)
    F = F.sum(0) if F.ndim == 3 else F
    H, W = F.shape
    f = max(1, H // px)
    F = F[: px * f, : px * f].reshape(px, f, px, f).mean((1, 3))
    return (np.clip(F, 0, 1) * 255).astype(np.uint8)


def run_batch(rules: list, inits: np.ndarray, shape, hz: Horizon, th: Thresholds, seeds: list) -> list:
    """rules: B rules of one structure; inits: (B, C, H, W). All rules must share T."""
    T = rules[0].T
    assert all(r.T == T for r in rules), "a batch needs one T"
    steps = int(round(hz.time * T))
    rec = max(1, int(round(hz.record_every * T)))
    stop = int(round(hz.stop_every * T)) if hz.stop_every else 0
    thumb = int(round(hz.thumb_every * T)) if hz.thumb_every else 0
    film_steps = set(np.linspace(steps - hz.film_span * T, steps, hz.frames).round().astype(int).tolist())
    count_at = set(range(steps - 4 * rec, steps + 1, 2 * rec))     # count bodies near the end only
    full = make_batch(rules, shape)
    b = full
    A = to_device(inits.astype(np.float32).copy())
    An = None                                     # numpy copy of A for the current step (recording only)
    active = list(range(len(rules)))             # positions in `rules` still running
    trackers = [Tracker(r.R, shape, th) for r in rules]
    films = [[] for _ in rules]
    thumbs = [[] for _ in rules]
    spread_hits = [0] * len(rules)
    for k in range(steps + 1):
        t = k / T
        if k % rec == 0 or k in film_steps or (thumb and k % thumb == 0):
            An = to_numpy(A)
            for pos, i in enumerate(active):
                if k % rec == 0:
                    trackers[i].add(t, An[pos], count_components=(k in count_at))
                if k in film_steps:
                    films[i].append((t, An[pos].astype(np.float16)))
                if thumb and k % thumb == 0:
                    thumbs[i].append((t, thumbnail(An[pos], hz.thumb_px)))
        if stop and k % stop == 0 and 0 < k < steps:
            keep = []
            for pos, i in enumerate(active):
                tr = trackers[i]
                m, occ = tr.mass[-1], tr.occ[-1]
                spread_hits[i] = spread_hits[i] + 1 if (t >= hz.spread_after and occ > th.spread_occupancy) else 0
                why = ("fizzled" if m < th.min_mass_r2 else "exploded" if occ > th.explode_occupancy
                       else "spread" if spread_hits[i] >= 2 else None)
                if why:
                    tr.forced = why
                    films[i] = [(t, to_numpy(A[pos:pos + 1])[0].astype(np.float16))]
                else:
                    keep.append(pos)
            if len(keep) < len(active):
                if not keep:
                    break
                A = A[keep]
                b = b.subset(keep)
                active = [active[p] for p in keep]
        if k < steps:
            A = b.step(A)
    out = []
    for i, r in enumerate(rules):
        out.append(Result(rule=r, seed=seeds[i], stats=trackers[i].summary(), film=films[i],
                          trace=trackers[i].trace(), thumbs=thumbs[i]))
    return out
