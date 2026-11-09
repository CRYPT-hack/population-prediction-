# Group 14 – Predicting Population Movement Using Transition Matrices
MTH2802 Linear Algebra & Differential Equations · BML Munjal University · Instructor: Dr. Luckshay Batra

## How to run
```
pip install numpy pandas matplotlib seaborn      # networkx is not needed
python project_code.py
```
Python 3.9+ recommended. Running creates/overwrites `figures/` (9 PNGs, 200 dpi), `results.json` and `results.txt` next to the script. The random seed is fixed (42), so results are reproducible. If a check fails, the script stops with an `AssertionError`.

**Edit inputs** at the top of `project_code.py`: the dictionary `INITIAL_POPULATION` (approximate Census 2011 figures, to be verified) and the matrix `P` (SIMULATED probabilities).

## File guide
1. `project_code.py` – all computations, assertions and figures.
2. `results.json` / `results.txt` – every number used in the report (machine-readable / readable log with the assertion log).
3. `figures/` – fig1–fig9 (heat map, flow diagram, time series, steady state, convergence, starts, scenarios, sensitivity, eigenvalues).
4. `Group14_Report.docx` (+ `.pdf`) – the report; `Group14_Viva_Prep.md` – viva summary, 25 Q&A, numbers to remember, section talking points.
5. `Group14_Slides.pptx` – 8-slide presentation.

## Honesty note
Transition probabilities are **simulated**. Initial populations are **approximate** Census 2011 figures and must be verified by the group.
