import numpy as np

import census_data as cd
import census_model
import inputs
import markov_model as mm


def test_census_populations_match_inputs():
    pop = cd.load_population()
    for r in cd.REGIONS:
        assert pop[r] == inputs.INITIAL_POPULATION[r]


def test_census_matrix_is_stochastic_and_uses_direct_flows():
    P, m, notes = cd.census_transition_matrix()
    mm.validate_stochastic(P)
    assert (P > 0).all()
    # Delhi -> Gurugram is read straight from D-02: 10,906 people resident < 1 year
    assert abs(P[1, 0] - 10906 / 16_787_941) < 1e-12
    assert notes[("Gurugram", "Delhi")] == "direct"
    assert (m > 0).all()


def test_allocation_weight_scales_only_allocated_flows():
    F1, _, _ = cd.flow_matrix(1.0)
    F2, _, _ = cd.flow_matrix(2.0)
    assert F2[1, 0] == F1[1, 0]                       # direct flow unchanged
    assert np.isclose(F2[0, 1], 2 * F1[0, 1])         # allocated flow doubles


def test_srs_growth_period_average():
    g = cd.srs_growth(2001, 2010)
    assert abs(g["Delhi"] - (18.10 - 4.73) / 1000) < 1e-4
    assert g["Gurugram"] == g["Faridabad"]            # both Haryana


def test_newton_recovers_known_emigration():
    P, m, _ = cd.census_transition_matrix()
    g = np.array([0.013, 0.017, 0.021, 0.017])
    e_true = np.array([0.004, 0.012, 0.009, 0.006])
    x0 = np.array([1.4e7, 9e5, 1.2e6, 1.4e6])
    x10 = mm.project_open(mm.open_system_matrix(P, e_true, g), m, x0, 10)[-1]
    e, iters = mm.calibrate_emigration(P, g, m, x0, x10, 10)
    assert np.allclose(e, e_true, atol=1e-9) and iters < 10


def test_batched_sensitivity_matches_loop():
    rng = np.random.default_rng(7)
    loop_s, loop_l = [], []
    for _ in range(50):
        Q = mm.perturb(inputs.P, 0.02, rng)
        loop_s.append(mm.steady_state_eig(Q)); loop_l.append(mm.second_eigenvalue_modulus(Q))
    s, l = mm.sensitivity(inputs.P, 50, 0.02, seed=7)
    assert np.allclose(s, loop_s, atol=1e-12) and np.allclose(l, loop_l, atol=1e-12)


def test_census_model_pipeline(tmp_path):
    R = census_model.run(make_figures=False, outdir=tmp_path, quiet=True)
    assert R["design_warnings"] == []
    assert R["census_spectrum"]["years_to_1pct_census"] > 1000
    assert R["ratio_simulated_to_census"]["median"] > 30
    totals = R["forecast"]["total_2041_by_weight"].values()
    assert max(totals) / min(totals) < 1.02
