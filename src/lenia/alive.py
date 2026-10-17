"""Is a rollout a living creature? Per-frame statistics and a rule-based classifier.

Everything is measured on the total field (channels summed) and in units of the kernel radius R, so
the same thresholds work across rules and world sizes. The verdict looks only at the second half of
the rollout (the "window"), after the start-up transients.

Classes, checked in this order:
  fizzled    mass falls below min_mass_r2 (nothing left)
  exploded   the pattern covers more than explode_occupancy of the world
  spread     it covers more than spread_occupancy, or its spread (radius of gyration) exceeds
             max_rg_r: space-filling stripes, labyrinths, smears
  dying      mass at the end is below dying_ratio x the window's peak
  growing    still growing at the end of the window: mass at the end above growing_ratio x the
             window's start, or the fitted trend of mass (or spread) across the window adds more than
             max_growth (max_rg_growth) of its mean value: a slow explosion that has not hit the size
             limits yet
  multi      two or more separate bodies (fission, or a cloud of debris)
  frozen     neither moves nor changes shape: a static blob or string
  mover      alive, travels at least min_speed_r kernel radii per unit of time
  pulser     alive, stays in place but keeps changing shape (rotating, breathing)
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

CLASSES = ["fizzled", "exploded", "spread", "dying", "growing", "multi", "frozen", "mover", "pulser"]
ALIVE = {"mover", "pulser"}


@dataclass
class Thresholds:
    occupied: float = 0.1            # a cell is occupied if its total value exceeds this
    min_mass_r2: float = 0.02        # mass / R^2 below this = fizzled (Orbium is about 0.43)
    explode_occupancy: float = 0.5   # fraction of the world
    spread_occupancy: float = 0.15
    max_rg_r: float = 2.5            # radius of gyration, in kernel radii (Orbium is about 0.6)
    dying_ratio: float = 0.5
    growing_ratio: float = 2.0
    min_body_r2: float = 0.05        # a body counts if it has this much area (in R^2) ...
    min_body_mass: float = 0.05      # ... and this share of the mass
    min_speed_r: float = 0.02        # kernel radii per unit of time
    max_growth: float = 0.15         # relative mass growth across the window (linear fit), calibrated: docs/lenia.md
    max_rg_growth: float = 0.15      # relative growth of the radius of gyration across the window
    min_change: float = 0.05         # shape change per unit of time (mass-normalised L1, in a co-moving frame);
                                     # below about 0.05 a "pulser" looks static: re-centring jitter, not life

    @staticmethod
    def from_dict(d: dict | None) -> "Thresholds":
        return Thresholds(**(d or {}))


def centroid(F: np.ndarray) -> tuple[float, float]:
    """Mass centre on the torus (circular mean along each axis)."""
    H, W = F.shape
    tot = F.sum()
    if tot <= 0:
        return H / 2, W / 2
    py, px = F.sum(1) / tot, F.sum(0) / tot
    cy = np.angle((py * np.exp(2j * np.pi * np.arange(H) / H)).sum()) * H / (2 * np.pi) % H
    cx = np.angle((px * np.exp(2j * np.pi * np.arange(W) / W)).sum()) * W / (2 * np.pi) % W
    return float(cy), float(cx)


def torus_delta(a: float, b: float, n: int) -> float:
    d = (b - a) % n
    return d - n if d > n / 2 else d


def recentre(F: np.ndarray, c: tuple[float, float]) -> np.ndarray:
    """Roll so the centroid sits in the middle of the world. F is (H, W) or (C, H, W)."""
    H, W = F.shape[-2:]
    return np.roll(F, (int(round(H / 2 - c[0])), int(round(W / 2 - c[1]))), axis=(-2, -1))


def radius_of_gyration(F: np.ndarray, c: tuple[float, float]) -> float:
    H, W = F.shape
    tot = F.sum()
    if tot <= 0:
        return 0.0
    dy = (np.arange(H) - c[0] + H / 2) % H - H / 2
    dx = (np.arange(W) - c[1] + W / 2) % W - W / 2
    d2 = dy[:, None] ** 2 + dx[None, :] ** 2
    return float(np.sqrt((F * d2).sum() / tot))


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


def count_bodies(F: np.ndarray, R: float, th: Thresholds) -> int:
    mask = F > th.occupied
    if not mask.any():
        return 0
    lab = bodies(mask)
    ids, area = np.unique(lab[mask], return_counts=True)
    mass = np.array([F[lab == i].sum() for i in ids])
    keep = (area >= th.min_body_r2 * R * R) & (mass >= th.min_body_mass * F.sum())
    return int(keep.sum())


class Tracker:
    """Collects statistics frame by frame for one world, then classifies the rollout."""

    def __init__(self, R: float, shape, th: Thresholds):
        self.R, self.shape, self.th = R, tuple(shape), th
        self.t, self.mass, self.occ, self.rg, self.speed, self.change = [], [], [], [], [], []
        self._prev_c = self._prev_F = self._prev_t = None
        self.bodies_last = None
        self.forced = None          # set when a rollout is stopped early (fizzled / exploded / spread)

    def add(self, t: float, A: np.ndarray, count_components: bool = False):
        F = A.sum(0) if A.ndim == 3 else A
        H, W = F.shape
        m = float(F.sum())
        c = centroid(F)
        Fc = recentre(F, c)
        self.t.append(t)
        self.mass.append(m / self.R ** 2)
        self.occ.append(float((F > self.th.occupied).mean()))
        self.rg.append(radius_of_gyration(F, c) / self.R)
        if self._prev_c is not None and m > 0:
            dt = t - self._prev_t
            dist = np.hypot(torus_delta(self._prev_c[0], c[0], H), torus_delta(self._prev_c[1], c[1], W))
            self.speed.append(dist / self.R / dt)
            self.change.append(float(np.abs(Fc - self._prev_F).sum() / max(m, 1e-9)) / dt)
        else:
            self.speed.append(np.nan)
            self.change.append(np.nan)
        self._prev_c, self._prev_F, self._prev_t = c, Fc, t
        if count_components:
            self.bodies_last = count_bodies(F, self.R, self.th)

    def trace(self) -> np.ndarray:
        """(n, 6) float32: t, mass/R^2, occupancy, spread/R, speed (R/time), shape change (/time)."""
        return np.array([self.t, self.mass, self.occ, self.rg, self.speed, self.change], np.float32).T

    def summary(self) -> dict:
        warnings.filterwarnings("ignore", message="All-NaN slice")
        th = self.th
        n = len(self.t)
        w = slice(n // 2, n)
        mass, occ, rg = np.array(self.mass), np.array(self.occ), np.array(self.rg)
        sp, ch = np.array(self.speed), np.array(self.change)
        tw = np.array(self.t)[w]

        def rel_trend(x):     # fitted change across the window, relative to the window mean
            if len(tw) < 3 or np.mean(x) <= 1e-9:
                return 0.0
            return float(np.polyfit(tw, x, 1)[0] * (tw[-1] - tw[0]) / np.mean(x))

        s = {
            "mass_end": float(mass[-1]), "mass_min_w": float(mass[w].min()), "mass_max_w": float(mass[w].max()),
            "mass_start_w": float(mass[w][0]), "occ_max_w": float(occ[w].max()), "rg_max_w": float(rg[w].max()),
            "rg_end": float(rg[-1]), "speed_w": float(np.nanmedian(sp[w])) if n > 2 else 0.0,
            "change_w": float(np.nanmedian(ch[w])) if n > 2 else 0.0, "bodies": self.bodies_last,
            "growth_w": rel_trend(mass[w]), "rg_growth_w": rel_trend(rg[w]), "t_end": float(self.t[-1]),
        }
        s["class"] = self.forced or classify(s, th)
        s["alive"] = s["class"] in ALIVE
        return s


def classify(s: dict, th: Thresholds) -> str:
    """The verdict from a rollout's summary statistics (so runs can be re-judged without re-simulating)."""
    nan = lambda v: v is None or v != v
    speed = 0.0 if nan(s.get("speed_w")) else float(s["speed_w"])
    change = 0.0 if nan(s.get("change_w")) else float(s["change_w"])
    nb = 0 if nan(s.get("bodies")) or s.get("bodies") in ("", None) else int(float(s["bodies"]))
    if s["mass_end"] < th.min_mass_r2 or s["mass_min_w"] < th.min_mass_r2:
        return "fizzled"
    if s["occ_max_w"] > th.explode_occupancy:
        return "exploded"
    if s["occ_max_w"] > th.spread_occupancy or s["rg_max_w"] > th.max_rg_r:
        return "spread"
    if s["mass_end"] < th.dying_ratio * s["mass_max_w"]:
        return "dying"
    g = lambda k: 0.0 if nan(s.get(k)) or s.get(k) in ("", None) else float(s[k])
    if (s["mass_end"] > th.growing_ratio * s["mass_start_w"] or g("growth_w") > th.max_growth
            or g("rg_growth_w") > th.max_rg_growth):
        return "growing"
    if nb >= 2:
        return "multi"
    if speed < th.min_speed_r and change < th.min_change:
        return "frozen"
    return "mover" if speed >= th.min_speed_r else "pulser"
