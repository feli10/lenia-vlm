"""Small statistics without scipy: AUC with a permutation p-value, one-sided Fisher exact test.
Moved verbatim on 2026-10-28 from scripts/surprise_check.py (now in archive/).
"""
from __future__ import annotations

import numpy as np


def auc(scores: list, labels: list, n_perm: int = 20000, seed: int = 0) -> tuple[float, float]:
    """AUC of scores for label 1 vs 0 (ties count half) and a one-sided permutation p-value."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    def a_(yy):
        pos, neg = s[yy == 1], s[yy == 0]
        return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())
    obs = a_(y)
    rng = np.random.default_rng(seed)
    perm = np.array([a_(rng.permutation(y)) for _ in range(n_perm)])
    return obs, float((perm >= obs - 1e-12).mean())


def fisher_greater(a1: int, b1: int, a2: int, b2: int) -> float:
    """One-sided Fisher exact test: P(top row has >= a1 successes) given the margins (no scipy needed)."""
    from math import comb
    n1, n2, k = a1 + b1, a2 + b2, a1 + a2
    tot = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a1, min(n1, k) + 1) if 0 <= k - x <= n2) / tot
