"""lenia: simulator, classifier, seeds, sampler and a tiny end-to-end survey (offline, ~30 s)."""
from __future__ import annotations

import json

import numpy as np

from lenia import seeds
from lenia.alive import Thresholds, bodies, count_bodies
from lenia.rle import rle2arr
from lenia.rollout import Horizon, run_batch
from lenia.sample import sample
from lenia.sim import Batch, Kernel, Rule, kernel_fields, place
from lenia.survey import survey

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


def test_orbium_glides_and_empty_fizzles():
    hz, th = Horizon(time=20, stop_every=5), Thresholds()
    inits = np.stack([place(orbium_cells(), SHAPE), np.zeros((1,) + SHAPE, np.float32)])
    res = run_batch([ORBIUM, ORBIUM], inits, SHAPE, hz, th, [0, 0])
    assert res[0].stats["class"] == "mover" and res[0].stats["speed_w"] > 0.3
    assert abs(res[0].stats["mass_end"] - 0.43) < 0.03                          # mass stays put
    assert res[1].stats["class"] == "fizzled"
    assert len(res[0].film) == hz.frames


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


def test_survey_end_to_end(tmp_path):
    rules = [ORBIUM] + sample("classic", 2, seed=1)
    bank = [{"kind": "donor", "seed": 0, "code": "O2u"}, {"kind": "smooth", "seed": 1, "size_r": 1.5}]
    s = survey(rules, tmp_path, bank=bank, shape=SHAPE, hz=Horizon(time=8, film_span=4, frames=3),
               workers=1, batch=4, log=lambda *_: None)
    assert s["rollouts"] == 6 and s["alive_rules"] >= 1                         # Orbium from its own shape
    assert (tmp_path / "rules.csv").exists() and (tmp_path / "sheet_alive.png").exists()


def test_show_gif(tmp_path):
    from PIL import Image
    from lenia import show
    clips = [("orbium", ORBIUM, ("cells", orbium_cells())), show.clip_from_json(ORBIUM.to_json(), seed=1)]
    sims = show.simulate(clips, SHAPE, time=3, every=5)
    assert sims[0][3]["mass_end"] > 0.3
    p = show.gif(sims, tmp_path / "x.gif", px=64)
    im = Image.open(p)
    assert im.n_frames == len(sims[0][2]) and im.size[0] > 128


def test_explore_small(tmp_path):
    from lenia.explore import explore, load_archive, mutate
    import numpy.random as npr
    r = mutate(ORBIUM, npr.default_rng(0))
    assert r.structure() == ORBIUM.structure() and r.kernels[0].m != ORBIUM.kernels[0].m
    roots = [("orbium", ORBIUM, 0)]          # seed 0 of the bank is Orbium's own shape
    s = explore(roots, tmp_path, None, SHAPE, evals=8, children_per_gen=8, child_time=4, confirm_time=6, rescale_time=4,
                workers=1, log=lambda *_: None)
    assert s["roots"] == 1 and s["evals"] >= 8 and (tmp_path / "archive.csv").exists()
    assert len(load_archive(tmp_path)) == s["archive"]


def test_early_stop_and_growth():
    from lenia.alive import Tracker
    hz, th = Horizon(time=20, stop_every=5, thumb_every=5), Thresholds()
    inits = np.stack([place(orbium_cells(), SHAPE), np.zeros((1,) + SHAPE, np.float32)])
    res = run_batch([ORBIUM, ORBIUM], inits, SHAPE, hz, th, [0, 0])
    assert res[0].stats["class"] == "mover" and res[0].stats["t_end"] == 20
    assert res[1].stats["class"] == "fizzled" and res[1].stats["t_end"] == 5        # dropped at the first checkpoint
    assert res[0].trace.shape[1] == 6 and len(res[0].thumbs) == 5 and res[0].thumbs[0][1].shape == (32, 32)
    assert abs(res[0].stats["growth_w"]) < 0.05                                     # Orbium keeps its mass
    tr = Tracker(13, SHAPE, th)                                                     # a pattern that keeps growing
    for k in range(40):
        A = np.zeros(SHAPE, np.float32)
        r = 6 + k // 4
        A[48 - r:48 + r, 48 - r:48 + r] = 0.5
        tr.add(k * 0.5, A)
    assert tr.summary()["class"] == "growing"


def test_trace_plot(tmp_path):
    from lenia.trace import plot
    rules = [ORBIUM]
    bank = [{"kind": "donor", "seed": 0, "code": "O2u"}]
    survey(rules, tmp_path, bank=bank, shape=SHAPE, hz=Horizon(time=10, film_span=4, frames=3, stop_every=5, thumb_every=5),
           workers=1, batch=4, log=lambda *_: None)
    assert (tmp_path / "traces.npz").exists() and (tmp_path / "thumbs.npz").exists()
    assert plot(tmp_path, [0], tmp_path / "t.png").exists()


def test_rescale_check():
    from lenia.explore import rescale_check, rescaled, upsample
    P = np.zeros((1, 4, 4), np.float32)
    P[0, 1:3, 1:3] = 1
    U = upsample(P, 2)
    assert U.shape == (1, 8, 8) and abs(U.sum() / 4 - P.sum()) < 0.5 and 0 <= U.min() and U.max() <= 1
    assert rescaled(ORBIUM).R == 26 and rescaled(ORBIUM).kernels == ORBIUM.kernels
    speck = np.random.default_rng(0).random((1, 20, 20)).astype(np.float32) * (np.indices((20, 20)).sum(0) % 2)
    res = rescale_check([ORBIUM, ORBIUM], [orbium_cells()[None], speck], SHAPE, Thresholds(), None, time=20)
    assert res[0][0] == "mover" and res[0][1] is not None and res[1][0] not in ("mover", "pulser")   # Orbium is resolution-free; a checkerboard is not


def _fake_test_folder(tmp_path, n=8):
    import csv as _csv
    from PIL import Image as _Image
    from lenia.vlmtest import grid_image
    (tmp_path / "images").mkdir()
    rows = []
    groups = ["catalogue", "coherent", "speckle", "failure"]
    for k in range(n):
        A = np.zeros((1, 64, 64), np.float32)
        r = 4 + k
        A[0, 32 - r:32 + r, 32 - r:32 + r] = 0.8
        if groups[k % 4] == "speckle":                     # pixel noise instead of a solid body
            A[0] *= np.random.default_rng(k).random((64, 64)) > 0.5
        _Image.fromarray(grid_image([A] * 4, 6, whole_world=False)).save(tmp_path / "images" / f"c{k:03d}.png")
        rows.append({"id": f"c{k:03d}", "group": groups[k % 4], "source": "t", "class": "failure" if k % 4 == 3 else "pulser",
                     "R": 6, "scale": 1, "mass": r, "size_rg": r, "speed": 0, "change": 0.1, "roughness": 1.0, "rule": "{}"})
    with open(tmp_path / "creatures.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pairs = [(f"c{a:03d}", f"c{b:03d}") for a in range(n) for b in range(a + 1, n)]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    return pairs


def test_vlmtest_fake_end_to_end(tmp_path):
    from lenia.judge import Judge
    from lenia import vlmtest
    pairs = _fake_test_folder(tmp_path)
    j = Judge("fake", {"backend": "fake", "fake_position_bias": 0.3, "fake_noise": 0.1}, tmp_path / "cache.sqlite", log=lambda m: None)
    vlmtest.run(tmp_path, {"fake": j}, ["appeal", "lifelike"], log=lambda *_: None)
    s = vlmtest.report(tmp_path, log=lambda *_: None)
    r = s["per_judge_prompt"]["fake/appeal"]
    assert r["pairs"] == len(pairs) and 0 <= r["order_agreement"] <= 1
    assert r["group_mean_score"]["speckle"] < r["group_mean_score"]["catalogue"]   # the fake judge dislikes rough images
    assert (tmp_path / "report.md").exists() and (tmp_path / "top_fake_appeal.png").exists()
    j2 = Judge("fake", {"backend": "fake", "fake_position_bias": 0.3, "fake_noise": 0.1}, tmp_path / "cache.sqlite", log=lambda m: None)
    from lenia.judge import Query
    from PIL import Image as _Image
    img = np.asarray(_Image.open(tmp_path / "images" / "c000.png"))
    assert j2.judge_many([Query("appeal", img, img[::-1])])[0].cached is False   # new pair
    assert j2.judge_many([Query("appeal", img, img[::-1])])[0].cached is True    # now cached


def test_bradley_terry_orders():
    from lenia.vlmtest import bradley_terry
    games = [("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", 0.95)] * 3
    s = bradley_terry(["a", "b", "c"], games)
    assert s["a"] > s["b"] > s["c"]


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


def test_rating_server_and_human_agreement(tmp_path):
    import threading
    import urllib.request
    from lenia.judge import Judge
    from lenia import vlmtest
    _fake_test_folder(tmp_path)
    j = Judge("fake", {"backend": "fake", "fake_noise": 0.1}, tmp_path / "cache.sqlite", log=lambda m: None)
    vlmtest.run(tmp_path, {"fake": j}, ["appeal"], log=lambda *_: None)
    srv = vlmtest.make_server(tmp_path, n=5, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Which creature" in urllib.request.urlopen(base + "/").read()
        todo = json.loads(urllib.request.urlopen(base + "/pairs").read())["todo"]
        assert len(todo) == 5 and urllib.request.urlopen(base + f"/img/{todo[0]['left']}.png").status == 200
        post = lambda path, d=None: urllib.request.urlopen(urllib.request.Request(
            base + path, data=json.dumps(d or {}).encode(), method="POST", headers={"Content-Type": "application/json"}))
        for k, p in enumerate(todo):
            post("/vote", {**p, "choice": ["left", "right", "skip"][k % 3]})
        post("/undo")                                                    # removes the last vote
        assert len(json.loads(urllib.request.urlopen(base + "/pairs").read())["todo"]) == 1
        try:
            urllib.request.urlopen(base + "/img/..%2Fcreatures.csv")
            assert False, "path traversal must 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()
        srv.server_close()
    s = vlmtest.report(tmp_path, log=lambda *_: None)
    assert s["human"]["rated"] + s["human"]["skipped"] == 4
    assert s["per_judge_prompt"]["fake/appeal"]["human_agreement"][1] == s["human"]["rated"]
    assert "Prompt comparison" in (tmp_path / "report.md").read_text()


def test_guided_small(tmp_path):
    """Two Orbium lineages, two generations, fake judge: founding, the constraint funnel, resume, check."""
    from lenia import guided
    from lenia.explore import evaluate
    from lenia.judge import Judge
    from lenia.rollout import Horizon
    (st, pat, d), = evaluate([ORBIUM], [orbium_cells()[None]], SHAPE, Horizon(time=20, frames=6, film_span=5, thumb_every=0),
                             Thresholds(), None)
    assert pat is not None
    founders = [{"label": f"orb{k}", "rule": ORBIUM, "pattern": np.asarray(pat, np.float32), "desc": d} for k in range(2)]
    cat_desc = np.stack([d + 0.05 * k for k in range(-3, 4)])
    j = Judge("fake", {"backend": "fake", "fake_noise": 0.1}, tmp_path / "cache.sqlite", log=lambda m: None)
    s = guided.run(tmp_path, founders, j, "curate", cat_desc, generations=1, shape=SHAPE, workers=1, n_lineages=2,
                   attempts=1, log=lambda *_: None)
    assert s["gen"] == 1 and len(s["lineages"]) == 2 and (tmp_path / "lineages.png").exists()
    s2 = guided.run(tmp_path, founders, j, "curate", cat_desc, generations=2, shape=SHAPE, workers=1, n_lineages=2,
                    attempts=1, log=lambda *_: None)                     # resumes from generation 1
    assert s2["gen"] == 2 and len(s2["history"]) == 2
    cat_imgs = [(f"cat{k}", tmp_path / "images" / "L00_g000.png") for k in range(3)]
    r = guided.check(tmp_path, {"fake": j}, "curate", cat_imgs, n_cat=2, log=lambda *_: None)
    assert "fake" in r["per_judge"] and (tmp_path / "check.md").exists() and (tmp_path / "human_pairs.json").exists()


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
        res = run_batch([ORBIUM], place(orbium_cells(), SHAPE)[None], SHAPE, Horizon(time=20, stop_every=5), Thresholds(), [0])
        assert res[0].stats["class"] == "mover"
    finally:
        backend.configure("numpy")
