"""Where the simulation runs: numpy on the CPU (default) or PyTorch on a GPU.

Only the update step (FFT convolution + growth) moves to the GPU. Everything that looks at a world
(the alive classifier, descriptors, pictures) still runs in numpy; worlds are copied back only when
they are recorded (every 0.5 time units, i.e. every 5th step), so the GPU does almost all the work.

Choose with `lenia --backend ...` or the LENIA_BACKEND environment variable (inherited by worker
processes):
  numpy  the reference implementation (CPU, one process per worker)
  auto   CUDA if available, else Apple's GPU (MPS) if its FFT works, else numpy
  cuda | mps | cpu   PyTorch on that device (cpu = PyTorch on the CPU, for testing)
On a GPU the CLI runs one process with large batches instead of many CPU workers.

LENIA_FLOW switches every batch to another update scheme without touching the rules on disk:
  LENIA_FLOW=mace        mass-conserving MaCE update, beta from the rule (default 8)
  LENIA_FLOW=mace:4      ... with beta = 4 for every rule
Unset (or "growth") keeps each rule's own scheme.
"""
from __future__ import annotations

import os

import numpy as np

_state: dict = {}


def configure(name: str | None) -> None:
    if name:
        os.environ["LENIA_BACKEND"] = name
    _state.clear()


def _resolve() -> dict:
    if _state:
        return _state
    name = os.environ.get("LENIA_BACKEND", "numpy").lower()
    _state.update(kind="numpy", device=None, torch=None, label="numpy (CPU)")
    if name == "numpy":
        return _state
    try:
        import torch
    except ImportError:
        if name != "auto":
            raise RuntimeError(f"backend {name!r} needs PyTorch (pip install torch)")
        return _state
    dev = None
    if name in ("auto", "cuda") and torch.cuda.is_available():
        dev = "cuda"
    elif name in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:                                   # MPS FFT support varies by PyTorch version
            x = torch.rand(1, 1, 8, 8, device="mps")
            torch.fft.irfft2(torch.fft.rfft2(x), s=(8, 8)).sum().item()
            dev = "mps"
        except Exception:
            dev = None
    elif name == "cpu":
        dev = "cpu"
    if dev is None:
        if name != "auto":
            raise RuntimeError(f"backend {name!r} is not available here")
        return _state
    label = f"torch {torch.__version__} on {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
    _state.update(kind="torch", device=dev, torch=torch, label=label)
    return _state


def to_device(A: np.ndarray):
    s = _resolve()
    A = np.ascontiguousarray(A, dtype=np.float32)
    return s["torch"].from_numpy(A).to(s["device"]) if s["kind"] == "torch" else A


def to_numpy(A) -> np.ndarray:
    if isinstance(A, np.ndarray):
        return A
    return A.detach().to("cpu").numpy()


def flow_override() -> tuple[str, float | None] | None:
    """(flow, beta) from LENIA_FLOW, or None."""
    v = os.environ.get("LENIA_FLOW", "").strip().lower()
    if not v:
        return None
    flow, _, beta = v.partition(":")
    return flow, (float(beta) if beta else None)


def make_batch(rules: list, shape):
    from .sim import Batch, TorchBatch, with_flow
    fo = flow_override()
    if fo:
        rules = [with_flow(r, *fo) for r in rules]
    s = _resolve()
    return TorchBatch(rules, shape, s["device"]) if s["kind"] == "torch" else Batch(rules, shape)
