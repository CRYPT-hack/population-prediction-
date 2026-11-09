import numpy as np
import pytest

import inputs
import markov_model as mm

REG = inputs.REGIONS
e = np.array([inputs.EMIGRATION[r] for r in REG])
g = np.array([inputs.NATURAL_GROWTH[r] for r in REG])
x01 = np.array([inputs.POPULATION_2001[r] for r in REG], float)
x11 = np.array([inputs.INITIAL_POPULATION[r] for r in REG], float)


def test_open_matrix_reduces_to_P_without_growth_or_emigration():
    assert np.allclose(mm.open_system_matrix(inputs.P, np.zeros(4), np.zeros(4)), inputs.P)


def test_open_matrix_column_sums():
    A = mm.open_system_matrix(inputs.P, e, np.zeros(4))
    assert np.allclose(A.sum(0), 1 - e)


def test_homogeneous_matrix_equals_affine_iteration():
    A = mm.open_system_matrix(inputs.P, e, g); m = np.array([1e5, 2e4, 3e4, 1e4])
    B = mm.homogeneous_matrix(A, m)
    X = mm.project_open(A, m, x11, 15)
    assert np.allclose(np.linalg.matrix_power(B, 15) @ np.append(x11, 1), np.append(X[-1], 1))


def test_calibration_reproduces_census_exactly():
    alpha, m, A = mm.calibrate_mobility(inputs.P, e, g, x01, x11, 10)
    assert 0 <= alpha <= 1 and (m >= -1e-6).all()
    assert np.allclose(mm.project_open(A, m, x01, 10)[-1], x11, rtol=1e-9)


def test_full_mobility_is_inconsistent_with_census():
    """Documents the modelling finding: simulated P needs negative immigration."""
    A = mm.open_system_matrix(inputs.P, e, g)
    assert mm.calibrate_immigration(A, x01, x11, 10).min() < 0


def test_affine_fixed_point_when_shrinking():
    A = mm.open_system_matrix(inputs.P, np.full(4, 0.05), np.zeros(4))   # 5 % leave, no births
    m = np.array([1e5, 1e4, 1e4, 1e4])
    xs = mm.affine_fixed_point(A, m)
    assert xs is not None and np.allclose(A @ xs + m, xs)
    assert np.allclose(mm.project_open(A, m, x11, 2000)[-1], xs, rtol=1e-6)


def test_perron_vector_positive():
    rho, v = mm.perron(mm.open_system_matrix(inputs.P, e, g))
    assert rho > 0 and (v > 0).all() and abs(v.sum() - 1) < 1e-12


def test_rejects_bad_rates():
    with pytest.raises(ValueError):
        mm.open_system_matrix(inputs.P, np.full(4, 1.2), g)
