# Group 14 – Predicting Population Movement Using Transition Matrices
MTH2802 Linear Algebra & Differential Equations · BML Munjal University · Instructor: Dr. Luckshay Batra

This folder holds the written submission, **updated** to match the fixed and extended code in the repository root.
The version originally handed in is preserved in git: `git checkout v1.0-submission -- submission/`.

## Files
1. `Group14_Report.docx` / `.pdf` – the report (22 pages). Sections 1–6 closed model, Section 7 open-system extension, Appendix A core code, Appendix B hand calculations.
2. `Group14_Slides.pptx` – 10-slide presentation (slides 7–8 cover the census test and the open model).
3. `Group14_Viva_Prep.md` – one-page summary, 32 Q&A, numbers to remember, section talking points.
4. `project_code.py`, `results.json`, `results.txt`, `figures/` – the **original** script and its outputs, kept for comparison. Use `../project_code.py` and `../open_system.py` to reproduce the report.

## What changed since the original submission
* Census 2011 populations for Gurugram (1,514,432) and Noida (1,648,115) corrected; projections, hand calculation and coefficients updated.
* Convergence theory computed from the eigen-decomposition (33 / 52 years) instead of fitted from the simulation.
* Figures regenerated; Figure 9 now in the complex plane.
* New Section 7 / slides 7–8: open model with births, deaths and migration, calibrated to Census 2001 → 2011.
* See `../BUGFIXES.md` for the code-level fixes.

## Honesty note
Transition probabilities and emigration rates are **simulated**. Populations are Census 2011 (and 2001) district totals; natural growth is SRS 2020 at state level.
