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
