"""Decode the run-length strings in Chan's Lenia catalogue (2-D only), following LeniaND.py.

Values: '.'/'b' = 0, 'o' = 255, 'A'..'X' = 1..24, two-letter 'pA'..'yO' = 25..255; '$' ends a row,
'!' ends the pattern, digits repeat the next value (or the next row break).
"""
from __future__ import annotations

import numpy as np


def _val(tok: str) -> int:
    if tok in (".", "b"):
        return 0
    if tok == "o":
        return 255
    if len(tok) == 1:
        return ord(tok) - ord("A") + 1
    return (ord(tok[0]) - ord("p")) * 24 + (ord(tok[1]) - ord("A") + 25)


def rle2arr(st: str) -> np.ndarray:
    rows, row = [], []
    count, prefix = "", ""
    for ch in st.rstrip("!") + "$":
        if ch.isdigit():
            count += ch
        elif ch in "pqrstuvwxy":
            prefix = ch
        elif ch == "$":
            rows.append(row)
            rows.extend([[] for _ in range(int(count) - 1)] if count else [])
            row, count, prefix = [], "", ""
        else:
            row.extend([_val(prefix + ch)] * (int(count) if count else 1))
            count, prefix = "", ""
    w = max((len(r) for r in rows), default=0)
    A = np.zeros((len(rows), w), np.float32)
    for i, r in enumerate(rows):
        A[i, : len(r)] = r
    return A / 255.0
