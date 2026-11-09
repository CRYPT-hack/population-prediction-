"""
Reusable linear-algebra core for the NCR population model.

Every function is pure (no printing, no globals) so it can be unit-tested and reused by
`project_code.py` (closed system), `open_system.py` (births, deaths, migration) and the tests.
Convention: column-stochastic matrices, x_{k+1} = P x_k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
def validate_stochastic(P: np.ndarray, tol: float = 1e-9) -> None:
    """Raise ValueError (with a readable reason) unless P is a square column-stochastic matrix."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square, got shape {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError("P contains NaN or infinite entries")
    if (P < -tol).any():
        i, j = np.argwhere(P < -tol)[0]
        raise ValueError(f"P[{i},{j}] = {P[i, j]} is negative")
    sums = P.sum(axis=0)
    bad = np.where(abs(sums - 1) > tol)[0]
    if bad.size:
        raise ValueError(f"column(s) {bad.tolist()} sum to {sums[bad].round(6).tolist()}, not 1")


def regularity_index(P: np.ndarray) -> int | None:
    """Smallest k with P^k > 0 (entrywise), or None if P is not regular.

    By Wielandt's theorem a primitive n x n matrix satisfies P^k > 0 for some k <= (n-1)^2 + 1,
    so checking up to that bound is a PROOF, not a guess (the original code looped to an arbitrary 50/200).
    Only the zero pattern matters, so we work with booleans (no round-off can create fake zeros).
    """
    n = P.shape[0]
    B = np.asarray(P) > 0
    Bk = B.copy()
    for k in range(1, (n - 1) ** 2 + 2):
        if Bk.all():
            return k
        Bk = (B.astype(int) @ Bk.astype(int)) > 0
    return None


