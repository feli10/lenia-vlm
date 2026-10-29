"""Live core: simulator, sampler, body counting, judge request shapes, the rating pages (offline, a few s).
The simulator and judge tests are copied verbatim from the old tests/test_lenia.py (now in archive/tests/).
"""
from __future__ import annotations

import json

import numpy as np

from lenia.mace import bodies
from lenia.sample import sample
from lenia.sim import Batch, Kernel, Rule, kernel_fields

SHAPE = (96, 96)


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
        v = j._call(Query("evolve_complexity_world", img, img2))
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


def test_label_page_and_stats(tmp_path):
    """lenia.rating.rate_labels' page records one answer per world; lenia.stats gives sane AUC / Fisher values."""
    import threading
    import urllib.request
    from argparse import Namespace
    from http.server import ThreadingHTTPServer
    from PIL import Image
    from lenia import rating
    from lenia.stats import auc, fisher_greater
    (tmp_path / "images").mkdir()
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(tmp_path / "images" / "w00.png")
    (tmp_path / "tasks.json").write_text(json.dumps([{"id": 0, "image": "w00.png"}, {"id": 1, "image": "w00.png"}]))
    started = {}
    real_init = ThreadingHTTPServer.__init__

    def init(self, addr, handler):                      # bind to a free port and serve in a thread, no browser
        real_init(self, ("127.0.0.1", 0), handler)
        started["srv"] = self
    import webbrowser
    ThreadingHTTPServer.__init__, webbrowser.open = init, lambda url: None
    try:
        th = threading.Thread(target=rating.rate_labels, args=(Namespace(out=str(tmp_path), port=0),), daemon=True)
        th.start()
        while "srv" not in started:
            pass
        base = f"http://127.0.0.1:{started['srv'].server_address[1]}"
        assert len(json.loads(urllib.request.urlopen(base + "/todo").read())) == 2
        urllib.request.urlopen(urllib.request.Request(base + "/answer", method="POST",
                               data=json.dumps({"id": 0, "answer": "new", "note": "x"}).encode()))
        assert [t["id"] for t in json.loads(urllib.request.urlopen(base + "/todo").read())] == [1]
        assert rating.label_answers(tmp_path)[0]["answer"] == "new"
    finally:
        ThreadingHTTPServer.__init__ = real_init
        if "srv" in started:
            started["srv"].shutdown()
    a, p = auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], n_perm=500)
    assert a == 1.0 and p < 0.5
    assert abs(fisher_greater(6, 0, 1, 6) - 0.0047) < 0.001


def test_spend_cap_snapshots_and_all_failed(tmp_path, monkeypatch):
    """The run-level cap refuses a batch even with yes=True; model versions are recorded; a batch where every
    call fails stops the run instead of reading as rejections."""
    from lenia.judge import BudgetError, FatalAPIError, Judge, Query, Verdict
    img = np.full((40, 40, 3), 200, np.uint8)
    imgs = [img.copy() for _ in range(4)]
    for k, im in enumerate(imgs):
        im[k:k + 5] = 10
    qs = [Query("evolve_complexity_world", imgs[i], imgs[(i + 1) % 4]) for i in range(4)]
    j = Judge("fake", {"backend": "fake", "est_cost_per_call_usd": 1.0, "max_run_usd": 2.5}, tmp_path / "a.sqlite",
              yes=True, log=lambda m: None)
    try:
        j.judge_many(qs)
        assert False, "cap must refuse a $4 batch under a $2.50 cap"
    except BudgetError:
        pass
    assert j.cache.total_cost() == 0                     # nothing was called
    v = Verdict(0.7, "first", "{}", 0.001, snapshot="jev-1.14-20261014")
    j.cache.put("k", v, "jev-1.14.0", "p")
    assert j.cache.snapshots() == {"jev-1.14-20261014": 1} and abs(j.cache.total_cost() - 0.001) < 1e-12
    j2 = Judge("fake", {"backend": "fake"}, tmp_path / "b.sqlite", yes=True, log=lambda m: None)
    monkeypatch.setattr(Judge, "_call", lambda self, q: Verdict(None, None, "HTTP 400: unknown model"))
    try:
        j2.judge_many(qs)
        assert False, "an all-failed batch must stop the run"
    except FatalAPIError as e:
        assert "unknown model" in str(e)
