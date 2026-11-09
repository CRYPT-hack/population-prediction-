"""Unit tests for the linear-algebra core, including one regression test per bug fixed."""
import numpy as np
import pytest

import inputs
import markov_model as mm

P = inputs.P


def test_validate_accepts_project_matrix():
    mm.validate_stochastic(P)


@pytest.mark.parametrize("bad, msg", [
    (np.array([[0.5, 0.5], [0.4, 0.5]]), "sum"),
    (np.array([[1.2, 0.0], [-0.2, 1.0]]), "negative"),
    (np.ones((2, 3)) / 2, "square"),
])
def test_validate_rejects_bad_matrices(bad, msg):
    with pytest.raises(ValueError, match=msg):
        mm.validate_stochastic(bad)


def test_three_steady_state_methods_agree():
    a = mm.steady_state_eig(P)
    A, b = mm.steady_state_system(P)
    bb = mm.gauss_solve(A, b)
    c = mm.steady_state_power(P, 200)
    assert np.allclose(a, bb, atol=1e-12) and np.allclose(a, c, atol=1e-10)
    assert np.allclose(P @ a, a) and abs(a.sum() - 1) < 1e-12


def test_gauss_solve_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)); b = rng.normal(size=6)
    assert np.allclose(mm.gauss_solve(A, b), np.linalg.solve(A, b))


def test_gauss_solve_detects_singular_matrix():          # bug: silently returned inf/nan
    with pytest.raises(np.linalg.LinAlgError):
        mm.gauss_solve(np.array([[1., 2.], [2., 4.]]), np.array([1., 2.]))


def test_regularity_index():
    assert mm.regularity_index(P) == 1
    # zero pattern that needs k = 2 (some entries zero, but P^2 > 0)
    Q = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    assert mm.regularity_index(Q) == 2
    assert mm.regularity_index(np.array([[0., 1.], [1., 0.]])) is None        # periodic
    assert mm.regularity_index(np.eye(3)) is None                             # reducible


def test_eigenvalues_sorted_by_modulus():                 # bug: sorted by real value
    # eigenvalues 1, -0.9 and 0.2: |lambda_2| must be 0.9, not 0.2
    Q = np.array([[0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]])
    Q = Q / Q.sum(0)
    e = mm.eigendecomposition(Q)
    assert abs(e.values[0] - 1) < 1e-12
    assert abs(abs(e.values[1]) - mm.second_eigenvalue_modulus(Q)) < 1e-12
    assert abs(e.values[1]) > abs(e.values[2])


def test_diagonalisation_reconstructs_P():
    e = mm.eigendecomposition(P)
    assert np.allclose(e.V @ np.diag(e.values) @ e.Vinv, P, atol=1e-12)
    assert np.allclose(e.V[:, 0], mm.steady_state_eig(P))
    assert all(abs(e.V[:, j].sum()) < 1e-10 for j in range(1, 4))


def test_first_below_returns_none_if_never_reached():    # bug: returned 0
    d = np.array([1.0, 0.5, 0.2])
    assert mm.first_below(d, 0.01) is None
    assert mm.first_below(d, 0.6) == 1


def test_theoretical_convergence_constant_is_close_to_empirical():
    pi = mm.steady_state_eig(P); e = mm.eigendecomposition(P)
    s0 = np.array(list(inputs.INITIAL_POPULATION.values()), float); s0 /= s0.sum()
    d = mm.distance_curve(P, s0, pi, 200)
    C = mm.convergence_constant_theory(e, s0); lam2 = abs(e.values[1])
    assert abs(d[150] / (C * lam2 ** 150) - 1) < 0.05     # asymptotically d_k ~ C |l2|^k


def test_scale_mobility_keeps_steady_state_and_maps_eigenvalues():
    for a in (0.1, 0.5, 0.9):
        Pa = mm.scale_mobility(P, a)
        mm.validate_stochastic(Pa)
        assert np.allclose(mm.steady_state_eig(Pa), mm.steady_state_eig(P))
        assert np.allclose(np.sort(np.linalg.eigvals(Pa).real), np.sort(1 + a * (np.linalg.eigvals(P).real - 1)))


def test_replace_column_rejects_non_probability():
    with pytest.raises(ValueError):
        mm.replace_column(P, 0, [0.5, 0.5, 0.5, 0.0])


def test_sensitivity_is_reproducible():
    a, la = mm.sensitivity(P, 50, 0.02, seed=1)
    b, lb = mm.sensitivity(P, 50, 0.02, seed=1)
    assert np.array_equal(a, b) and np.array_equal(la, lb)
    assert np.allclose(a.sum(1), 1)
