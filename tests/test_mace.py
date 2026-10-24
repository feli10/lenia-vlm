"""MaCE (mass-conserving) update: exact mass conservation, numpy/torch parity, old rules unchanged."""
from __future__ import annotations

import json

import numpy as np
import pytest

from lenia import backend
from lenia.sim import Batch, Kernel, Rule, with_flow

SHAPE = (64, 64)


def _rule3():
    ks = [Kernel(m=0.15 + 0.03 * i, s=0.015 + 0.005 * i, b=(1.0, 0.5), c0=i % 3, c1=(i + 1) % 3) for i in range(6)]
    return Rule(kernels=ks, R=10, T=10, channels=3)


def _world(B=2, C=3, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((B, C) + SHAPE, np.float32)
    A[:, :, 20:44, 20:44] = rng.random((B, C, 24, 24))
    return A


def test_default_flow_is_growth_and_old_dicts_load():
    r = _rule3()
    assert r.flow == "growth"
    d = json.loads(r.to_json()); d.pop("flow"); d.pop("beta")      # a rule saved before MaCE existed
    assert Rule.from_dict(d).flow == "growth"


def test_mace_conserves_mass_per_channel():
    r = with_flow(_rule3(), "mace", 8.0)
    b, A = Batch([r, with_flow(r, "mace", 3.0)], SHAPE), _world()
    m0 = A.sum(axis=(-2, -1))
    for _ in range(300):
        A = b.step(A)
    assert A.min() >= 0
    assert np.allclose(A.sum(axis=(-2, -1)), m0, rtol=1e-4)


def test_flow_is_part_of_structure():
    r = _rule3()
    assert r.structure() != with_flow(r, "mace").structure()
    with pytest.raises(AssertionError):
        Batch([r, with_flow(r, "mace")], SHAPE)


def test_env_override(monkeypatch):
    monkeypatch.setenv("LENIA_FLOW", "mace:5")
    backend.configure("numpy")
    b = backend.make_batch([_rule3()], SHAPE)
    assert b.flow == "mace" and float(b.beta.ravel()[0]) == 5.0
    monkeypatch.delenv("LENIA_FLOW")


def test_mace_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from lenia.sim import TorchBatch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = with_flow(_rule3(), "mace")
    A = _world(B=1)
    n, t = A, torch.from_numpy(A).to(dev)
    nb, tb = Batch([r], SHAPE), TorchBatch([r], SHAPE, dev)
    for _ in range(50):                        # a few steps, so mass has started to pile up
        n, t = nb.step(n), tb.step(t)
    t = t.cpu().numpy()
    assert np.abs(n - t).max() < 1e-3 * max(1.0, float(n.max()))
    assert abs(float(t.sum()) - float(A.sum())) < 1e-3 * float(A.sum())
