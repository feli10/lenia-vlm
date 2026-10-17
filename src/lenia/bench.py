"""`lenia bench`: is the GPU simulator correct, and how fast is it?

1. Correctness: the PyTorch step against the numpy reference, from identical starts. One step must
   agree to float32 rounding (max cell difference < 1e-4); Orbium run for 30 time units must still
   be the same creature (max difference < 1e-2, mass within 0.5%). Chaotic patterns drift apart
   slowly in any two float32 implementations, so long runs are compared only on stable creatures.
2. Speed: milliseconds per creature per step for the three setups we use, numpy vs PyTorch.
"""
from __future__ import annotations

import time

import numpy as np

from . import backend
from .sample import sample
from .sim import Batch, Rule, place

ORBIUM = {"R": 13, "T": 10, "b": "1", "m": 0.15, "s": 0.015, "kn": 1, "gn": 1}


def _setups():
    import dataclasses
    from . import seeds
    from .rle import rle2arr
    orb = Rule.from_chan(ORBIUM)
    cells = rle2arr(seeds.donors()["O2u"]["cells"])
    mk = [dataclasses.replace(r, R=26) for r in sample("multikernel", 16, seed=3, kernels=3)]
    mc = [dataclasses.replace(r, R=26) for r in sample("multichannel", 16, seed=3, channels=3, self_k=2, cross_k=1)]
    return [("classic 128x128, R=13 (Orbium)", [orb] * 16, (128, 128), cells),
            ("multikernel 256x256, R=26", mk, (256, 256), None),
            ("multichannel 256x256, R=26", mc, (256, 256), None)]


def _sync():
    s = backend._resolve()
    if s["device"] == "cuda":
        s["torch"].cuda.synchronize()
    elif s["device"] == "mps":
        s["torch"].mps.synchronize()


def verify(log=print) -> bool:
    from .sim import TorchBatch
    s = backend._resolve()
    ok = True
    rng = np.random.default_rng(0)
    for name, rules, shape, cells in _setups():
        C = rules[0].channels
        A = (np.stack([place(cells, shape)] * len(rules)) if cells is not None
             else rng.random((len(rules), C) + shape).astype(np.float32) * 0.6)
        a1 = Batch(rules, shape).step(A)
        t1 = backend.to_numpy(TorchBatch(rules, shape, s["device"]).step(backend.to_device(A)))
        d1 = float(np.abs(a1 - t1).max())
        line = f"  {name}: one step max diff {d1:.1e}"
        good = d1 < 1e-4
        if cells is not None:                  # stable creature: run 30 time units on both
            bn, bt = Batch(rules[:1], shape), TorchBatch(rules[:1], shape, s["device"])
            An, At = A[:1].copy(), backend.to_device(A[:1])
            for _ in range(300):
                An, At = bn.step(An), bt.step(At)
            At = backend.to_numpy(At)
            dl = float(np.abs(An - At).max())
            dm = abs(float(An.sum()) - float(At.sum())) / max(float(An.sum()), 1e-9)
            line += f"; after 30 units max diff {dl:.1e}, mass diff {dm:.1e}"
            good = good and dl < 1e-2 and dm < 5e-3
        ok = ok and good
        log(line + ("  OK" if good else "  MISMATCH"))
    return ok


def timing(log=print, steps: int = 40) -> list:
    from .sim import TorchBatch
    rows = []
    rng = np.random.default_rng(1)
    for name, rules, shape, cells in _setups():
        C = rules[0].channels
        A = rng.random((len(rules), C) + shape).astype(np.float32) * 0.6
        b = Batch(rules, shape)
        n = max(5, steps // 4)
        b.step(A)
        t0 = time.perf_counter()
        X = A
        for _ in range(n):
            X = b.step(X)
        ms_np = (time.perf_counter() - t0) / n / len(rules) * 1000
        row = {"setup": name, "numpy_ms": round(ms_np, 3)}
        if backend.is_torch():
            s = backend._resolve()
            for B in (16, 64):
                rr = (rules * ((B + len(rules) - 1) // len(rules)))[:B]
                AA = backend.to_device(np.concatenate([A] * ((B + len(A) - 1) // len(A)))[:B])
                tb = TorchBatch(rr, shape, s["device"])
                Y = tb.step(AA)
                _sync()
                t0 = time.perf_counter()
                for _ in range(steps):
                    Y = tb.step(Y)
                _sync()
                row[f"torch_B{B}_ms"] = round((time.perf_counter() - t0) / steps / B * 1000, 4)
            best = min(row["torch_B16_ms"], row["torch_B64_ms"])
            row["speedup"] = round(ms_np / best, 1)
        rows.append(row)
        log("  " + ", ".join(f"{k} {v}" for k, v in row.items()))
    return rows


def run(log=print) -> bool:
    log(f"backend: {backend.info()}")
    ok = True
    if backend.is_torch():
        log("correctness (PyTorch vs numpy reference):")
        ok = verify(log)
    else:
        log("(numpy backend: nothing to verify; run with --backend auto on a GPU machine)")
    log("speed (ms per creature per step; numpy is one CPU process):")
    timing(log)
    log("ALL CHECKS PASSED" if ok else "CHECKS FAILED: do not use this backend for runs")
    return ok
