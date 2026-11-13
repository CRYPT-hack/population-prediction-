"""Keep the written submission in sync with the code: key numbers in the report, slides and
viva notes must match what the scripts produce. Skipped if python-docx / python-pptx are missing."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
R = json.loads((ROOT / "results" / "results.json").read_text())
O = json.loads((ROOT / "results" / "open_system.json").read_text())


def report_text():
    docx = pytest.importorskip("docx")
    d = docx.Document(SUB / "Group14_Report.docx")
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def test_report_uses_census_2011_figures():
    text = report_text()
    for v in R["initial_population_census2011"].values():
        assert f"{v:,}" in text
    assert "1,514,085" not in text.replace("(1,514,085)", "")   # old value only mentioned as the corrected draft


def test_report_projection_and_hand_example_match_results():
    text = report_text()
    for k in ("10", "30"):
        for v in R["projection_counts"][k]:
            assert f"{v:,}" in text
    assert f"{R['hand_example']['x1_value']:,.1f}" in text


def test_report_convergence_and_open_model_numbers():
    text = report_text()
    c = R["convergence"]
    assert f"{c['C_theory']:.4f}" in text
    assert str(c["years_to_1pct_theory"]) in text
    assert f"{O['calibration']['alpha']:.3f}" in text
    total_2041 = sum(O["forecast"]["open"][-1]) / 1e6
    assert f"{total_2041:.2f}" in text


def test_slides_and_viva_mention_current_results():
    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation(SUB / "Group14_Slides.pptx")
    slides = "\n".join(sh.text_frame.text for s in deck.slides for sh in s.shapes if sh.has_text_frame)
    assert "theory 33" in slides and "α = 0.137" in slides and "approximate" not in slides
    viva = (SUB / "Group14_Viva_Prep.md").read_text(encoding="utf-8")
    assert "21,760,221" in viva and "1,787,338.7" in viva and "0.137" in viva


def test_documents_quote_census_data_results():
    C = json.loads((ROOT / "results" / "census_model.json").read_text())
    text = report_text()
    assert f"{C['flows'][1][0]:,.0f}" in text                                   # Delhi -> Gurugram flow
    assert f"{C['census_spectrum']['years_to_1pct_census']:,}" in text
    total_2041 = sum(C["forecast"]["census_model"][-1]) / 1e6
    assert f"{total_2041:.1f} million in 2041" in text
    viva = (SUB / "Group14_Viva_Prep.md").read_text(encoding="utf-8")
    assert f"{total_2041:.1f} M" in viva
