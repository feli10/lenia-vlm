"""Catalogue rankings: the report's pieces (offline, fast). End to end: test_lenia.test_pool_end_to_end."""
from __future__ import annotations

import numpy as np

from lenia import rank


def test_residual_removes_size_trend():
    size = np.geomspace(0.5, 2.0, 50)
    extra = np.random.default_rng(0).normal(size=50)
    r = rank.residual(3 * np.log(size) + extra, size)
    assert abs(np.corrcoef(r, np.log(size))[0, 1]) < 1e-6
    assert np.allclose(r, rank.residual(extra, size))                         # only the size trend goes
    assert np.isnan(rank.residual([1, np.nan, 2, 3], [1, 1, np.nan, 2])[1])


def test_pair_agreement_counts_decisive_pairs_only():
    sc = {"a": 2.0, "b": 1.0, "c": 0.0}
    games = [("a", "b", 0.9), ("b", "c", 0.1), ("a", "c", 0.6), ("a", "z", 0.9)]   # 0.6: not decisive; z: unscored
    assert rank.pair_agreement(games, sc) == (0.5, 2)
    assert rank.pair_agreement([("a", "c", 0.6)], sc) == (None, 0)


def test_split_half_tracks_judge_noise():
    rng = np.random.default_rng(1)
    ids = [f"x{i}" for i in range(60)]
    true = dict(zip(ids, rng.normal(size=60) * 2))
    pairs = [(ids[i], ids[j]) for i in range(60) for j in rng.choice(60, 8, replace=False) if i != j]
    sig = lambda x: 1 / (1 + np.exp(-x))
    clean = [(a, b, float(sig(true[a] - true[b]))) for a, b in pairs]
    noisy = [(a, b, float(rng.random())) for a, b in pairs]                    # a judge flipping coins
    assert rank.split_half(ids, clean) > 0.8
    assert abs(rank.split_half(ids, noisy)) < 0.3
    assert rank.split_half(ids, clean[:10]) is None
