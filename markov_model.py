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


# ----------------------------------------------------------------------------------------------
# Spectrum
# ----------------------------------------------------------------------------------------------
@dataclass
class Eigen:
    values: np.ndarray        # sorted by |lambda| descending
    V: np.ndarray             # columns = eigenvectors (v1 sums to 1, others unit length)
    Vinv: np.ndarray


def eigendecomposition(P: np.ndarray) -> Eigen:
    """P = V D V^-1 with eigenvalues sorted by MODULUS.

    BUG FIX: the original sorted by the real value (np.argsort(-w)). For a matrix with a
    negative eigenvalue such as -0.95 this puts it last, so |lambda_2| (convergence speed)
    was read from the wrong eigenvalue. Sorting by modulus is what the theory needs.
    """
    w, V = np.linalg.eig(P)
    if np.allclose(w.imag, 0) and np.allclose(V.imag, 0):
        w, V = w.real, V.real
    order = np.lexsort((-np.real(w), -abs(w)))   # by |lambda|, ties broken by real part
    w, V = w[order], V[:, order].copy()
    for j in range(V.shape[1]):
        if j == 0 and abs(V[:, 0].sum()) > 1e-12:
            V[:, 0] = V[:, 0] / V[:, 0].sum()
        else:
            V[:, j] = V[:, j] / np.linalg.norm(V[:, j])
            first = int(np.argmax(abs(V[:, j]) > 1e-12))
            if np.real(V[first, j]) < 0:
                V[:, j] *= -1
    if np.linalg.cond(V) > 1e12:
        raise np.linalg.LinAlgError("P is (numerically) not diagonalisable")
    return Eigen(w, V, np.linalg.inv(V))


def second_eigenvalue_modulus(P: np.ndarray) -> float:
    ev = np.linalg.eigvals(P)
    ev = ev[np.argsort(-abs(ev))]
    return float(abs(ev[1]))


def half_life(lam2: float) -> float:
    """Years for the slowest transient mode to halve."""
    return math.log(0.5) / math.log(lam2)


# ----------------------------------------------------------------------------------------------
# Projection & convergence
# ----------------------------------------------------------------------------------------------
def project(P: np.ndarray, x0: np.ndarray, K: int) -> np.ndarray:
    """Rows are x_0 .. x_K."""
    X = np.zeros((K + 1, len(x0)))
    X[0] = x0
    for k in range(K):
        X[k + 1] = P @ X[k]
    return X


def distance_curve(P: np.ndarray, start: np.ndarray, pi: np.ndarray, kmax: int = 120) -> np.ndarray:
    """L1 distance d_k = sum_i |x_k,i - pi_i| between the share vector and the steady state."""
    x = np.asarray(start, dtype=float)
    d = []
    for _ in range(kmax + 1):
        d.append(np.abs(x - pi).sum())
        x = P @ x
    return np.array(d)


def first_below(dist: np.ndarray, thr: float) -> int | None:
    """First k with dist[k] < thr.

    BUG FIX: the original used int(np.argmax(dist < thr)), which returns 0 (='converged
    immediately') when the threshold is NEVER reached. Now we return None in that case."""
    hits = np.flatnonzero(dist < thr)
    return int(hits[0]) if hits.size else None


def convergence_constant_theory(eig: Eigen, start_share: np.ndarray) -> float:
    """C in d_k ~ C |lambda_2|^k, computed from the eigen-decomposition: C = ||c_2 v_2||_1.

    BUG FIX: the original 'theory' fitted C from the simulated curve itself (C = d_20/|l2|^20),
    so comparing 'theory vs actual' was partly circular. This value uses only V^-1 x_0."""
    c = eig.Vinv @ start_share
    return float(np.abs(c[1] * eig.V[:, 1]).sum())


def years_to_threshold(C: float, lam2: float, thr: float) -> int:
    return max(0, math.ceil(math.log(thr / C) / math.log(lam2)))


# ----------------------------------------------------------------------------------------------
# Sensitivity
# ----------------------------------------------------------------------------------------------
def perturb(P: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply each entry by U[1-rel, 1+rel] and renormalise columns."""
    Q = P * rng.uniform(1 - rel, 1 + rel, size=P.shape)
    return Q / Q.sum(axis=0, keepdims=True)


def sensitivity(P: np.ndarray, n_runs: int = 1000, rel: float = 0.02, seed: int = 42):
    rng = np.random.default_rng(seed)
    shares = np.zeros((n_runs, P.shape[0]))
    lam2 = np.zeros(n_runs)
    for t in range(n_runs):
        Q = perturb(P, rel, rng)
        shares[t] = steady_state_eig(Q)
        lam2[t] = second_eigenvalue_modulus(Q)
    return shares, lam2


def replace_column(P: np.ndarray, col: int, new_col) -> np.ndarray:
    """Return a copy of P with one column replaced, checking it is still a probability vector."""
    M = np.array(P, dtype=float)
    M[:, col] = new_col
    if abs(M[:, col].sum() - 1) > 1e-12 or (M[:, col] < 0).any():
        raise ValueError(f"new column {col} is not a probability vector (sum = {M[:, col].sum()})")
    return M


# ----------------------------------------------------------------------------------------------
# Open system:  x_{k+1} = A x_k + m,   A = G M
# ----------------------------------------------------------------------------------------------
def open_system_matrix(P: np.ndarray, emigration: np.ndarray, growth: np.ndarray) -> np.ndarray:
    """A = G M where
         M = P diag(1 - e)   (column j: a fraction e_j leaves NCR, the rest moves according to P)
         G = diag(1 + g)     (natural growth: births - deaths, applied after moving).
    A is non-negative but no longer stochastic: column sums of M are 1 - e_j < 1."""
    e = np.asarray(emigration, dtype=float)
    g = np.asarray(growth, dtype=float)
    if ((e < 0) | (e >= 1)).any():
        raise ValueError("emigration rates must lie in [0, 1)")
    if (g <= -1).any():
        raise ValueError("growth rates must exceed -100 %")
    return np.diag(1 + g) @ P @ np.diag(1 - e)


def homogeneous_matrix(A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Augmented (n+1)x(n+1) matrix B = [[A, m], [0, 1]] so that [x_{k+1}; 1] = B [x_k; 1].
    This turns the AFFINE update x -> A x + m into a purely LINEAR one (same trick as in graphics)."""
    n = A.shape[0]
    B = np.zeros((n + 1, n + 1))
    B[:n, :n] = A
    B[:n, n] = m
    B[n, n] = 1.0
    return B


def project_open(A: np.ndarray, m: np.ndarray, x0: np.ndarray, K: int) -> np.ndarray:
    X = np.zeros((K + 1, len(x0)))
    X[0] = x0
    for k in range(K):
        X[k + 1] = A @ X[k] + m
    return X


def geometric_sum(A: np.ndarray, K: int) -> np.ndarray:
    """S_K = I + A + ... + A^(K-1)."""
    n = A.shape[0]
    S, Ak = np.zeros((n, n)), np.eye(n)
    for _ in range(K):
        S += Ak
        Ak = Ak @ A
    return S


def calibrate_immigration(A: np.ndarray, x_start: np.ndarray, x_end: np.ndarray, years: int) -> np.ndarray:
    """Solve for the constant yearly immigration m that carries x_start to x_end in `years` steps:

        x_end = A^K x_start + S_K m     =>    S_K m = x_end - A^K x_start

    A 4x4 linear system, solved with our own Gaussian elimination."""
    rhs = x_end - np.linalg.matrix_power(A, years) @ x_start
    return gauss_solve(geometric_sum(A, years), rhs)


def perron(A: np.ndarray):
    """Dominant eigenvalue rho and its eigenvector (normalised to sum 1) of a non-negative matrix.
    Perron-Frobenius: for a positive matrix rho is real, simple, and its eigenvector is positive."""
    w, V = np.linalg.eig(A)
    i = int(np.argmax(abs(w)))
    v = V[:, i] / V[:, i].sum()
    return float(w[i].real), v.real


def affine_fixed_point(A: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """x* = (I - A)^-1 m if rho(A) < 1 (population settles); None if rho(A) >= 1 (it keeps growing)."""
    rho, _ = perron(A)
    if rho >= 1:
        return None
    return np.linalg.solve(np.eye(A.shape[0]) - A, m)


# ----------------------------------------------------------------------------------------------
# Calibration of mobility against the census
# ----------------------------------------------------------------------------------------------
def scale_mobility(P: np.ndarray, alpha: float) -> np.ndarray:
    """P_alpha = I + alpha (P - I).  alpha = 1 gives P, alpha = 0 means nobody moves.

    Columns still sum to 1 and P_alpha pi = pi for the SAME steady state pi, while every
    eigenvalue maps to 1 + alpha (lambda - 1): less mobility -> same destination, slower journey."""
    n = P.shape[0]
    return np.eye(n) + alpha * (np.asarray(P, dtype=float) - np.eye(n))


def calibrate_mobility(P, emigration, growth, x_start, x_end, years, tol=1e-6):
    """Largest alpha in [0, 1] for which the census change x_start -> x_end can be explained
    with NON-NEGATIVE immigration into every region (bisection; min(m) decreases with alpha).

    Returns (alpha, m, A). Raises ValueError if even alpha = 0 needs negative immigration."""
    def m_of(a):
        A = open_system_matrix(scale_mobility(P, a), emigration, growth)
        return calibrate_immigration(A, x_start, x_end, years), A

    m1, A1 = m_of(1.0)
    if m1.min() >= 0:
        return 1.0, m1, A1
    m0, _ = m_of(0.0)
    if m0.min() < 0:
        raise ValueError("census change needs negative immigration even with zero internal mobility")
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if m_of(mid)[0].min() >= 0:
            lo = mid
        else:
            hi = mid
    m, A = m_of(lo)
    return lo, m, A
