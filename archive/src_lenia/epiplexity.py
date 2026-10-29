"""Learnable novelty (reservoir epiplexity, Zhang & Levin 2026, arXiv:2607.18433) for MaCE worlds.

The estimator itself is the authors' code, vendored unmodified in third_party/learnable_novelty/rc_epiplexity
(ridge readout on a frozen random reservoir, score S = 1/2 sum_i log2(1 + eta s_i(W)^2) in bits). This module
adds only what the paper does not cover, kept as close to their 1D reservoir as possible:

- RCEpiplexity2D: their RCEpiplexity1D with Conv2d instead of Conv1d (same depth / channels / kernel layout,
  circular padding, their PreActNorm + ELU before every non-final layer, 1x1 final layer, one readout shared
  by all sites, every (sample, site) one row of the ridge problem). `ln_check.py equiv` shows it reproduces
  RCEpiplexity1D exactly on 1D data.
- MaCE data: X = a burned-in world, Y = the next `tau` states `dt` simulation steps apart, stacked along the
  channel axis (3 * tau target channels), as their ECA / NCA data (X = burned-in state, Y = stacked window).

Choices that are ours (the paper has no 2D, multichannel, continuous-valued substrate), all exposed as
settings so ln_check.py can sweep them: `dt` (MaCE steps between stacked states), `down` (average-pool factor
before scoring, i.e. the reservoir's view in cells), reservoir `depth`, and the target scale (`y_scale`: "raw"
keeps their fixed sigma_y = 1 on mass density; "std" divides each world's targets by their own std).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_TP = Path(__file__).resolve().parents[2] / "third_party" / "learnable_novelty"
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))

TAU = 8                      # their inverse-NCA window (the continuous-field CA closest to MaCE)
CHANNELS = 256               # their reservoir width for CA data
KERNEL = 3
RIDGE = 0.3                  # their inverse-NCA ridge (ECA ranking: 0.03)
ROWS = 25_600                # ridge rows per score: one 160x160 world (ECA: 512 x 64 = 32,768)
OCC_ROWS = 4_000             # rows per score when only occupied sites are used (~ one creature's area at 160x160)
OCC_FRAC = 0.05              # a site is occupied if its mass exceeds this share of the world's peak (as mace_qd coverage)


@dataclass(frozen=True)
class Setting:
    dt: int = 10
    down: int = 1
    depth: int = 4
    y_scale: str = "raw"
    rows: str = "all"            # "all" sites (as the paper) | "occupied": OCC_ROWS rows from sites with mass |
                                 # "occ_full": up to ROWS occupied rows pooled over every start (ours)
    tau: int = TAU
    ridge: float = RIDGE

    def key(self) -> str:
        return f"dt{self.dt}_down{self.down}_depth{self.depth}_{self.y_scale}" + {"all": "", "occupied": "_occ", "occ_full": "_occfull"}[self.rows]

    def as_dict(self) -> dict:
        return asdict(self)


def reservoir_2d(depth: int, input_channels: int, seed: int, device, ridge: float = RIDGE,
                 channels: int = CHANNELS, kernel: int = KERNEL):
    """A frozen 2D reservoir drawn from `seed`."""
    import torch
    torch.manual_seed(seed)
    r = __getattr__("RCEpiplexity2D")(depth=depth, channels=channels, kernel_size=kernel, input_channels=input_channels,
                       ridge_lambda=ridge, eps=1e-8, device=device)
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def _make_2d_class():
    import torch.nn as nn
    from rc_epiplexity.core import RCEpiplexity
    from rc_epiplexity.reservoirs import PreActNorm

    class _RCEpiplexity2D(RCEpiplexity):
        """RCEpiplexity1D (their code) with 2D convolutions; everything else identical."""

        def __init__(self, depth, channels, kernel_size, input_channels=1, ridge_lambda=0.1, sigma_y=1.0,
                     lambda_code=1.0, eps=1e-8, device="cpu"):
            super().__init__(channels, ridge_lambda, sigma_y, lambda_code, eps, device)
            self.channels = channels
            layers, cin = [], input_channels
            for i in range(depth):
                last = i == depth - 1
                k = 1 if last else kernel_size
                pad = k // 2
                layers.append(nn.Conv2d(cin, channels, k, padding=pad, padding_mode="circular" if pad > 0 else "zeros",
                                        bias=True, device=self.device))
                if not last:
                    layers += [PreActNorm(), nn.ELU()]
                cin = channels
            self.phi = nn.Sequential(*layers)

        def epiplexity(self, x, y):
            # x: (batch, input_channels, H, W); y: (batch, target_channels, H, W)
            self.eval()
            h = self.phi(x)                                            # (batch, channels, H, W)
            features = h.movedim(1, -1).reshape(-1, self.channels)    # one row per (sample, site)
            target = y.movedim(1, -1).reshape(-1, y.shape[1])
            return self.multioutput_epiplexity(features, target)

    return _RCEpiplexity2D


class _Lazy:
    cls = None


def __getattr__(name):                     # RCEpiplexity2D is built on first use, so importing needs no torch
    if name == "RCEpiplexity2D":
        if _Lazy.cls is None:
            _Lazy.cls = _make_2d_class()
        return _Lazy.cls
    raise AttributeError(name)


# ---------- MaCE trajectories ----------

def film(rules: list, starts: np.ndarray, warmup: int, times: list, shape) -> np.ndarray:
    """Run every rule from its start for `warmup` steps, then keep the states at warmup + t for t in `times`
    (0 = the burned-in state). Returns (B, len(times), C, H, W) float32."""
    from . import backend
    keep = sorted(set(times))
    b = backend.make_batch(rules, shape)
    A = backend.to_device(starts)
    out = {}
    for _ in range(warmup):
        A = b.step(A)
    if 0 in keep:
        out[0] = backend.to_numpy(A).copy()
    last = max(keep)
    for t in range(1, last + 1):
        A = b.step(A)
        if t in keep:
            out[t] = backend.to_numpy(A).copy()
    return np.stack([out[t] for t in times], 1).astype(np.float32)


def times_for(settings: list) -> list:
    return sorted({0} | {s.dt * k for s in settings for k in range(1, s.tau + 1)})


def xy(frames: np.ndarray, times: list, s: Setting):
    """(B, T, C, H, W) frames -> X (B, C, h, w), Y (B, C * tau, h, w) for setting s (average-pooled by s.down),
    and the occupied sites (B, h, w): mass above OCC_FRAC of the world's peak in X or in any target frame."""
    idx = [times.index(s.dt * k) for k in range(1, s.tau + 1)]
    X = frames[:, times.index(0)]
    Y = frames[:, idx]                                            # (B, tau, C, H, W)
    if s.down > 1:
        B, T, C, H, W = Y.shape
        d = s.down
        X = X.reshape(B, C, H // d, d, W // d, d).mean((3, 5))
        Y = Y.reshape(B, T, C, H // d, d, W // d, d).mean((4, 6))
    B, T, C, h, w = Y.shape
    mass = np.maximum(X.sum(1), Y.sum(2).max(1))                 # (B, h, w): mass now or at any target time
    occ = mass > OCC_FRAC * mass.reshape(B, -1).max(1)[:, None, None]
    Y = Y.reshape(B, T * C, h, w)
    if s.y_scale == "std":
        Y = (Y - Y.mean()) / (Y.std() + 1e-8)
    return X, Y, occ


def _rows(res, xb, yb, ob, s: "Setting", rng):
    """Feature and target rows as their RCEpiplexity2D.epiplexity builds them (one row per sample and site);
    with s.rows == "occupied", only OCC_ROWS rows drawn from occupied sites (the reservoir still sees the whole
    grid around each site)."""
    import torch
    h = res.phi(xb)
    F = h.movedim(1, -1).reshape(-1, res.channels)
    T = yb.movedim(1, -1).reshape(-1, yb.shape[1])
    if s.rows in ("occupied", "occ_full"):
        n = OCC_ROWS if s.rows == "occupied" else ROWS
        idx = torch.nonzero(ob.reshape(-1)).squeeze(1)
        if idx.numel() > n:
            idx = idx[torch.from_numpy(rng.choice(idx.numel(), n, replace=False)).to(idx.device)]
        F, T = F[idx], T[idx]
    return F, T


def score(frames: np.ndarray, times: list, s: Setting, runs: int = 3, device="cpu", seed: int = 0) -> dict:
    """S (their estimator), the shuffled-target null and their difference, each as [mean, std] over `runs`
    reservoir draws. Null (ours): the same rows with the target rows permuted, which breaks the X -> Y relation
    and leaves only what the readout fits by chance at this target scale and row count. Each run uses its own
    block of starts when there are enough (rows per score ~ ROWS), otherwise all starts (data shared); "occ_full"
    always pools all starts."""
    import torch
    X, Y, O = xy(frames, times, s)
    per = max(1, int(round(ROWS / (X.shape[-1] * X.shape[-2]))))
    if s.rows == "occ_full":                       # pool every start; each run draws its own ROWS occupied rows
        per = X.shape[0]
    out = {"S": [], "null": [], "net": [], "rows": []}
    for r in range(runs):
        rng = np.random.default_rng(seed + 7 * r + 1)
        lo = (r * per) % X.shape[0] if X.shape[0] >= per * runs else 0
        xb = torch.from_numpy(np.ascontiguousarray(X[lo:lo + per])).to(device)
        yb = torch.from_numpy(np.ascontiguousarray(Y[lo:lo + per])).to(device)
        ob = torch.from_numpy(np.ascontiguousarray(O[lo:lo + per])).to(device)
        res = reservoir_2d(s.depth, X.shape[1], seed + 1000 * r, device, ridge=s.ridge)
        with torch.no_grad():
            F, T = _rows(res, xb, yb, ob, s, rng)
            v = float(res.multioutput_epiplexity(F, T).epiplexity)
            perm = torch.from_numpy(rng.permutation(T.shape[0])).to(T.device)
            n = float(res.multioutput_epiplexity(F, T[perm]).epiplexity)
        out["S"].append(v); out["null"].append(n); out["net"].append(v - n); out["rows"].append(int(F.shape[0]))
    return {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in out.items()}
