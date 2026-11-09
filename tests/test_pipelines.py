"""End-to-end: both scripts run, and the closed model reproduces the original submission."""
import json
from pathlib import Path

import numpy as np

import project_code

ORIG = json.loads((Path(__file__).parent / "original_results.json").read_text())


def test_closed_pipeline_matches_original_results(tmp_path):
    R = project_code.run(make_figures=False, outdir=tmp_path, quiet=True)
    # quantities that do NOT depend on the (corrected) census numbers must be identical
    assert np.allclose(R["steady_state"]["eigenvector"], ORIG["steady_state"]["eigenvector"], atol=1e-9)
    assert np.allclose(R["eigenvalues"], ORIG["eigenvalues"], atol=1e-8)
    assert np.allclose(R["lambda2_abs"], ORIG["lambda2_abs"])
    for nm in ORIG["scenarios"]:
        assert np.allclose(R["scenarios"][nm]["steady_state"], ORIG["scenarios"][nm]["steady_state"], atol=1e-6)
    assert abs(R["sensitivity"]["max_deviation_pct_points"] - ORIG["sensitivity"]["max_deviation_pct_points"]) < 1e-9
    assert R["design_warnings"] == []
    assert (tmp_path / "results" / "results.json").exists()


def test_closed_pipeline_uses_corrected_census():
    assert ORIG["initial_population_approx_census2011"]["Gurugram"] == 1_514_085     # old, wrong
    assert project_code.inputs.INITIAL_POPULATION["Gurugram"] == 1_514_432          # Census 2011


def test_closed_pipeline_works_with_custom_regions(tmp_path):
    """Original crashed on any region name not in its hard-coded colour/position dicts."""
    P = np.array([[0.8, 0.1, 0.2], [0.1, 0.8, 0.1], [0.1, 0.1, 0.7]])
    R = project_code.run(P=P, initial_population={"A": 100, "B": 50, "C": 25}, make_figures=True,
                         outdir=tmp_path, quiet=True)
    assert np.allclose(np.array(P) @ R["steady_state"]["eigenvector"], R["steady_state"]["eigenvector"], atol=1e-9)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 9
