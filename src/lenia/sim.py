"""Lenia (Chan 2019; multi-kernel / multi-channel as in Chan 2020), batched with numpy FFTs.

A rule is a set of kernels. Kernel k reads channel c0, convolves it with a ring-shaped kernel of
radius R*r, maps the result through a growth function (centre m, width s) and adds h * growth to
channel c1. Each channel then moves by dt = 1/T times the h-weighted mean of its incoming growths
and is clipped to [0, 1]. With one channel and one kernel (h = 1) this is classic Lenia, and the
parameters match Chan's animals.json (R, T, b, m, s, kn, gn).

The world is a torus. Kernels are built around (0, 0) on the torus, so no fftshift is needed.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np

# Chan's kernel cores (kn) and growth functions (gn), 1-based as in his files.
KERNEL_CORE = {
    1: lambda r: (4 * r * (1 - r)) ** 4,                                              # polynomial (quad4)
    2: lambda r: np.exp(4 - 1 / np.maximum(r * (1 - r), 1e-12)),                      # exponential bump (bump4)
    3: lambda r: ((r >= 0.25) & (r <= 0.75)).astype(float),                          # step
    4: lambda r: ((r >= 0.25) & (r <= 0.75)) + (r < 0.25) * 0.5,                     # staircase (life)
}
GROWTH = {
    1: lambda u, m, s: np.maximum(0, 1 - (u - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1,  # polynomial (quad4)
    2: lambda u, m, s: np.exp(-((u - m) ** 2) / (2 * s ** 2)) * 2 - 1,                # gaussian
    3: lambda u, m, s: (np.abs(u - m) <= s) * 2.0 - 1,                               # step
}


def _fracs(b) -> tuple[float, ...]:
    """Chan writes ring heights as '1,1/3' strings; accept that, lists of numbers, or Fractions."""
    if isinstance(b, str):
        b = b.strip("[]").split(",")
    return tuple(float(Fraction(str(x).strip())) for x in b)


@dataclass
class Kernel:
    m: float                       # growth centre
    s: float                       # growth width
    b: tuple = (1.0,)              # ring heights, inner to outer
    r: float = 1.0                 # radius relative to the rule's R
    h: float = 1.0                 # weight of this kernel in its target channel
    c0: int = 0                    # source channel
    c1: int = 0                    # target channel


@dataclass
class Rule:
    kernels: list
    R: float = 13.0                # kernel radius in cells
    T: float = 10.0                # steps per unit of time (dt = 1/T)
    channels: int = 1
    kn: int = 1
    gn: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Rule":
        d = dict(d)
        d["kernels"] = [Kernel(**{**k, "b": tuple(k["b"])}) for k in d["kernels"]]
        return Rule(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_chan(params) -> "Rule":
        """Chan's params: a dict (classic) or a list of dicts (multi-kernel; channels from 'c')."""
        plist = params if isinstance(params, list) else [params]
        p0 = plist[0]
        ks = []
        for p in plist:
            c0, c1 = p.get("c", [0, 0])
            ks.append(Kernel(m=float(p["m"]), s=float(p["s"]), b=_fracs(p.get("b", "1")),
                             r=float(p.get("r", 1)), h=float(p.get("h", 1)), c0=int(c0), c1=int(c1)))
        C = 1 + max(max(k.c0, k.c1) for k in ks)
        return Rule(kernels=ks, R=float(p0["R"]), T=float(p0.get("T", 10)), channels=C,
                    kn=int(p0.get("kn", 1)), gn=int(p0.get("gn", 1)))

    def structure(self) -> tuple:
        """Rules with the same structure can share a batch."""
        return (self.channels, len(self.kernels), tuple((k.c0, k.c1) for k in self.kernels), self.kn, self.gn)


def torus_dist(shape) -> np.ndarray:
    """Distance of every cell from (0, 0) on the torus."""
    H, W = shape
    y = np.minimum(np.arange(H), H - np.arange(H))[:, None]
    x = np.minimum(np.arange(W), W - np.arange(W))[None, :]
    return np.sqrt(y ** 2 + x ** 2)


def kernel_fields(rule: Rule, shape) -> np.ndarray:
    """(K, H, W) normalised kernels, centred on (0, 0)."""
    D = torus_dist(shape) / rule.R
    core = KERNEL_CORE[rule.kn]
    out = np.zeros((len(rule.kernels),) + tuple(shape))
    for i, k in enumerate(rule.kernels):
        bs = np.asarray(k.b, float)
        B = len(bs)
        Br = B * D / k.r
        ring = bs[np.minimum(np.floor(Br).astype(int), B - 1)]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            kk = (D < k.r) * core(np.minimum(Br % 1, 1)) * ring
        kk = np.nan_to_num(kk)
        tot = kk.sum()
        out[i] = kk / tot if tot > 0 else kk
    return out


class Batch:
    """B rules with the same structure, stepped together on (B, C, H, W) worlds."""

    def __init__(self, rules: list, shape=(128, 128)):
        assert len({r.structure() for r in rules}) == 1, "a batch needs rules with the same structure"
        self.rules, self.shape = rules, tuple(shape)
        r0 = rules[0]
        self.C, self.K = r0.channels, len(r0.kernels)
        self.c0 = np.array([k.c0 for k in r0.kernels])
        self.c1 = np.array([k.c1 for k in r0.kernels])
        self.growth = GROWTH[r0.gn]
        self.kf = np.stack([np.fft.rfft2(kernel_fields(r, shape)) for r in rules]).astype(np.complex64)
        par = lambda a: np.array([[getattr(k, a) for k in r.kernels] for r in rules], np.float32)[:, :, None, None]
        self.m, self.s, self.h = par("m"), par("s"), par("h")
        self.dt = np.array([1.0 / r.T for r in rules], np.float32)[:, None, None, None]
        # h-weighted mean per target channel
        hsum = np.zeros((len(rules), self.C), np.float32)
        for i in range(self.K):
            hsum[:, self.c1[i]] += self.h[:, i, 0, 0]
        self.hsum = np.maximum(hsum, 1e-12)[:, :, None, None]

    def subset(self, idx) -> "Batch":
        """The same batch restricted to worlds idx (used to drop finished rollouts)."""
        b = copy.copy(self)
        idx = np.asarray(idx)
        b.rules = [self.rules[i] for i in idx]
        for k in ("kf", "m", "s", "h", "dt", "hsum"):
            setattr(b, k, getattr(self, k)[idx])
        return b

    def step(self, A: np.ndarray) -> np.ndarray:
        Af = np.fft.rfft2(A)                                   # (B, C, H, Wr)
        U = np.fft.irfft2(self.kf * Af[:, self.c0], s=self.shape)  # (B, K, H, W)
        G = self.h * self.growth(U, self.m, self.s)
        D = np.zeros_like(A)
        for i in range(self.K):
            D[:, self.c1[i]] += G[:, i]
        return np.clip(A + self.dt * D / self.hsum, 0.0, 1.0).astype(np.float32)


GROWTH_TORCH = {
    1: lambda t, u, m, s: t.clamp(1 - (u - m) ** 2 / (9 * s ** 2), min=0) ** 4 * 2 - 1,
    2: lambda t, u, m, s: t.exp(-((u - m) ** 2) / (2 * s ** 2)) * 2 - 1,
    3: lambda t, u, m, s: ((u - m).abs() <= s).to(u.dtype) * 2.0 - 1,
}


class TorchBatch:
    """Batch on PyTorch (GPU): the same update, line for line. Worlds are (B, C, H, W) float32 tensors
    on `device`; use backend.to_device / to_numpy to move them."""

    def __init__(self, rules: list, shape=(128, 128), device: str = "cuda"):
        import torch
        assert len({r.structure() for r in rules}) == 1, "a batch needs rules with the same structure"
        self.t, self.device = torch, device
        self.rules, self.shape = rules, tuple(shape)
        r0 = rules[0]
        self.C, self.K = r0.channels, len(r0.kernels)
        self.c0 = torch.tensor([k.c0 for k in r0.kernels], dtype=torch.long, device=device)
        self.c1 = torch.tensor([k.c1 for k in r0.kernels], dtype=torch.long, device=device)
        self.gn = r0.gn
        kf = np.stack([np.fft.rfft2(kernel_fields(r, shape)) for r in rules]).astype(np.complex64)
        self.kf = torch.from_numpy(kf).to(device)
        par = lambda a: torch.tensor([[getattr(k, a) for k in r.kernels] for r in rules], dtype=torch.float32,
                                     device=device)[:, :, None, None]
        self.m, self.s, self.h = par("m"), par("s"), par("h")
        self.dt = torch.tensor([1.0 / r.T for r in rules], dtype=torch.float32, device=device)[:, None, None, None]
        hsum = torch.zeros(len(rules), self.C, dtype=torch.float32, device=device)
        hsum.index_add_(1, self.c1, self.h[:, :, 0, 0])
        self.hsum = hsum.clamp(min=1e-12)[:, :, None, None]

    def subset(self, idx) -> "TorchBatch":
        b = copy.copy(self)
        idx_t = self.t.as_tensor(np.asarray(idx), dtype=self.t.long, device=self.device)
        b.rules = [self.rules[i] for i in np.asarray(idx)]
        for k in ("kf", "m", "s", "h", "dt", "hsum"):
            setattr(b, k, getattr(self, k).index_select(0, idx_t))
        return b

    def step(self, A):
        t = self.t
        Af = t.fft.rfft2(A)                                          # (B, C, H, Wr)
        U = t.fft.irfft2(self.kf * Af[:, self.c0], s=self.shape)     # (B, K, H, W)
        G = self.h * GROWTH_TORCH[self.gn](t, U, self.m, self.s)
        D = t.zeros_like(A).index_add_(1, self.c1, G)
        return (A + self.dt * D / self.hsum).clamp_(0.0, 1.0)


def place(cells: np.ndarray, shape, C: int = 1) -> np.ndarray:
    """Put a (C, h, w) or (h, w) pattern in the middle of an empty (C, H, W) world."""
    cells = np.asarray(cells, np.float32)
    if cells.ndim == 2:
        cells = cells[None]
    A = np.zeros((C,) + tuple(shape), np.float32)
    H, W = shape
    h, w = cells.shape[-2:]
    if h > H or w > W:                              # too big: keep the middle
        cy, cx = (h - min(h, H)) // 2, (w - min(w, W)) // 2
        cells = cells[:, cy:cy + min(h, H), cx:cx + min(w, W)]
        h, w = cells.shape[-2:]
    y0, x0 = (H - h) // 2, (W - w) // 2
    A[: cells.shape[0], y0:y0 + h, x0:x0 + w] = cells
    return A


def noise_seed(R: float, shape, C: int, seed: int, size_r: float = 2.0, density: float = 1.0) -> np.ndarray:
    """A square of uniform noise, `size_r` kernel radii wide, in the middle of the world (one per channel)."""
    rng = np.random.default_rng(seed)
    n = max(4, int(round(size_r * R)))
    return place(rng.random((C, n, n)) * density, shape, C)


def mutate_copy(rule: Rule) -> Rule:
    return copy.deepcopy(rule)
