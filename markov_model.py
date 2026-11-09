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


# ----------------------------------------------------------------------------------------------
# Steady state - three independent methods
# ----------------------------------------------------------------------------------------------
def steady_state_eig(P: np.ndarray) -> np.ndarray:
    """Eigenvector for the eigenvalue closest to 1, normalised to sum 1."""
    w, V = np.linalg.eig(P)
    i = int(np.argmin(abs(w - 1)))
    if abs(w[i] - 1) > 1e-8:
        raise ValueError(f"no eigenvalue equal to 1 (closest is {w[i]})")
    v = V[:, i]
    v = v / v.sum()                       # complex-safe normalisation, removes arbitrary phase
    if np.max(abs(v.imag)) > 1e-10:
        raise ValueError("steady-state eigenvector is not real")
    return v.real


def gauss_solve(A: np.ndarray, b: np.ndarray, log: list | None = None, tol: float = 1e-14) -> np.ndarray:
    """Gaussian elimination with partial pivoting + back substitution (course style).

    BUG FIX: the original divided by the pivot without checking it, so a singular system
    (e.g. a non-regular P with several steady states) produced inf/nan silently.
    If `log` is a list, the row-echelon form is appended to it as text lines.
    """
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    m = len(b)
    for c in range(m):
        p = c + int(np.argmax(abs(A[c:, c])))
        if abs(A[p, c]) < tol:
            raise np.linalg.LinAlgError(f"matrix is singular (no pivot in column {c})")
        A[[c, p]] = A[[p, c]]
        b[[c, p]] = b[[p, c]]
        for r in range(c + 1, m):
            f = A[r, c] / A[c, c]
            A[r, c:] -= f * A[c, c:]
            b[r] -= f * b[c]
    if log is not None:
        log.append("Row-echelon form [A | b] after forward elimination:")
        for r in range(m):
            log.append("   " + "  ".join(f"{A[r, j]:9.5f}" for j in range(m)) + "  |  " + f"{b[r]:9.5f}")
    x = np.zeros(m)
    for r in range(m - 1, -1, -1):
        x[r] = (b[r] - A[r, r + 1:] @ x[r + 1:]) / A[r, r]
    return x


def steady_state_system(P: np.ndarray):
    """Solve (P - I) pi = 0 with the last equation replaced by sum(pi) = 1.

    Returns the augmented system (A, b); solve it with gauss_solve. Replacing one equation is valid because the
    rows of (P - I) are linearly dependent (they sum to the zero row)."""
    n = P.shape[0]
    A = P - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return A, b


def steady_state_power(P: np.ndarray, k: int = 200) -> np.ndarray:
    """Column 0 of P^k. (All columns are equal once P^k has converged.)"""
    return np.linalg.matrix_power(P, k)[:, 0]


