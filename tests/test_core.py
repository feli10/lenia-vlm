"""Live core: simulator, seeds, sampler, body counting, judge request shapes, the rating page (offline, a few s).
The simulator and judge tests are copied verbatim from the old tests/test_lenia.py (now in archive/tests/).
"""
from __future__ import annotations

import json

import numpy as np

from lenia import seeds
from lenia.alive import Thresholds, bodies, count_bodies
from lenia.rle import rle2arr
from lenia.sample import sample
from lenia.sim import Batch, Kernel, Rule, kernel_fields

SHAPE = (96, 96)
ORBIUM = Rule.from_chan({"R": 13, "T": 10, "b": "1", "m": 0.15, "s": 0.015, "kn": 1, "gn": 1})


def orbium_cells():
    return rle2arr(seeds.donors()["O2u"]["cells"])

def test_rle_orbium():
    c = orbium_cells()
    assert c.shape == (20, 20)
    assert abs(c.sum() - 76.86) < 0.01

def test_chan_params():
    r = Rule.from_chan([{"R": 20, "T": 10, "b": "1/2,1", "m": 0.2, "s": 0.02, "h": 0.5, "r": 0.8, "c": [0, 1]},
                        {"R": 20, "T": 10, "b": "1", "m": 0.3, "s": 0.03, "c": [1, 1]}])
    assert r.channels == 2 and r.kernels[0].b == (0.5, 1.0) and r.kernels[0].c1 == 1
    assert Rule.from_dict(json.loads(r.to_json())) == r

def test_kernels_normalised_and_symmetric():
    K = kernel_fields(Rule([Kernel(m=0.2, s=0.02, b=(1 / 3, 1.0))], R=13), SHAPE)
    assert abs(K.sum() - 1) < 1e-6
    assert np.allclose(K[0], np.roll(K[0][::-1, ::-1], (1, 1), axis=(0, 1)))   # point-symmetric on the torus

def test_bodies_on_torus():
    m = np.zeros((40, 40), bool)
    m[0:3, 10:13] = m[38:40, 10:13] = True        # one body across the top/bottom edge
    m[20:24, 20:24] = True                         # a second one
    lab = bodies(m)
    assert len(np.unique(lab[m])) == 2
    F = m.astype(float)
    assert count_bodies(F, R=4, th=Thresholds()) == 2

def test_seed_bank():
    for spec in seeds.DEFAULT_BANK:
        a = seeds.make(spec, 13, SHAPE, 1)
        b = seeds.make(spec, 13, SHAPE, 1)
        assert a.shape == (1,) + SHAPE and np.array_equal(a, b) and 0 <= a.min() and a.max() <= 1 and a.sum() > 1
    assert seeds.make(seeds.DEFAULT_BANK[0], 13, SHAPE, 3).shape == (3,) + SHAPE

def test_sampler_structures():
    mc = sample("multichannel", 3, seed=0, channels=3, self_k=2, cross_k=1)
    assert all(len(r.kernels) == 12 and r.channels == 3 for r in mc)
    assert len({r.structure() for r in mc}) == 1
    g = sample("grid", 16, seed=0)
    assert len(g) == 16 and len({(r.kernels[0].m, r.kernels[0].s) for r in g}) == 16
    Batch(mc, SHAPE).step(np.zeros((3, 3) + SHAPE, np.float32))                 # one batched step runs

def test_request_shapes(monkeypatch, tmp_path):
    """What we send to OpenRouter Decisions (Luna) and TypeSafe (Jev), and how answers are read."""
    from lenia.judge import Judge, Query, parse_probs
    img = np.full((300, 300, 3), 200, np.uint8)
    img2 = img.copy(); img2[100:200, 100:200] = 20
    seen = {}

    def fake_post(self, url, body, headers):
        seen[self.backend] = (url, body, headers)
        return {"answers": {"preference": {"choice": "second", "probabilities": {"first": 0.2, "second": 0.8}}},
                "usage": {"input_tokens": 1000, "cost": 0.0001}}
    monkeypatch.setattr(Judge, "_post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("TYPESAFE_API_KEY", "y")
    for b in ("decisions", "jev"):
        j = Judge(b, {"backend": b, "model": "m"}, tmp_path / "cache.sqlite", log=lambda m: None)
        v = j._call(Query("appeal", img, img2))
        assert abs(v.p_first - 0.2) < 1e-9 and v.choice == "second"
    url, body, headers = seen["jev"]
    assert url.endswith("/v1/systemone") and headers["TypeSafe-Danger-Zone-Images"] == "true"
    refs = list(body["assets"])
    assert len(refs) == 2 and body["state"] == {"first": refs[0], "second": refs[1]}
    assert "`first`" in body["questions"]["preference"]["instructions"] and set(body["questions"]["preference"]["criteria"]) == {"first", "second"}
    assert all(a["format"] == "png" and not a["data_base64"].startswith("data:") for a in body["assets"].values())
    url, body, headers = seen["decisions"]
    assert "openrouter.ai" in url and body["state"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parse_probs({"choice": "first"}) == (1.0, "first")


def test_torch_backend_matches_numpy():
    """PyTorch step (on CPU here, GPU on Colab) against the numpy reference; skipped without PyTorch."""
    try:
        import torch  # noqa: F401
    except ImportError:
        try:
            import pytest
            pytest.skip("PyTorch not installed")
        except ImportError:
            return
    from lenia.sim import TorchBatch
    from lenia import backend
    mc = sample("multichannel", 3, seed=0, channels=3, self_k=2, cross_k=1)
    A = np.random.default_rng(0).random((3, 3) + SHAPE).astype(np.float32) * 0.6
    backend.configure("cpu")
    try:
        t = backend.to_numpy(TorchBatch(mc, SHAPE, "cpu").step(backend.to_device(A)))
        assert np.abs(Batch(mc, SHAPE).step(A) - t).max() < 1e-4
    finally:
        backend.configure("numpy")


def test_rating_page(tmp_path):
    """lenia.rating: serves the pairs, records votes and undo in human.csv, refuses path traversal."""
    import threading
    import urllib.error
    import urllib.request
    from PIL import Image
    from lenia import rating
    (tmp_path / "images").mkdir()
    for k in range(6):
        Image.fromarray(np.full((32, 32, 3), 40 * k, np.uint8)).save(tmp_path / "images" / f"w{k}.png")
    (tmp_path / "pairs.json").write_text(json.dumps([[f"w{k}", f"w{k + 1}"] for k in range(5)]))
    srv = rating.make_server(tmp_path, n=5, port=0, question="Which world is more unusual?")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    post = lambda path, d=None: urllib.request.urlopen(urllib.request.Request(
        base + path, data=json.dumps(d or {}).encode(), method="POST", headers={"Content-Type": "application/json"}))
    try:
        assert b"Which world is more unusual?" in urllib.request.urlopen(base + "/").read()
        todo = json.loads(urllib.request.urlopen(base + "/pairs").read())["todo"]
        assert len(todo) == 5 and urllib.request.urlopen(base + f"/img/{todo[0]['left']}.png").status == 200
        for k, p in enumerate(todo):
            post("/vote", {**p, "choice": ["left", "right", "skip"][k % 3]})
        post("/undo")
        assert len(json.loads(urllib.request.urlopen(base + "/pairs").read())["todo"]) == 1
        assert len(rating._human_rows(tmp_path)) == 4
        try:
            urllib.request.urlopen(base + "/img/..%2Fpairs.json")
            assert False, "path traversal must 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()
        srv.server_close()
