# NCR Population Movement — Markov-chain model

Predicting how population moves between **Delhi, Gurugram, Noida and Faridabad** with transition
matrices. Started as Group 14's project for MTH2802 Linear Algebra & Differential Equations
(BML Munjal University), then extended with an open-system model, a transition matrix built from real
Census 2011 migration data, tests and an interactive dashboard.

```
x_{k+1} = P x_k                      closed system: people only move between the 4 regions
x_{k+1} = A x_k + m                  open system:   + births/deaths, emigration, immigration
A = diag(1+g) P diag(1-e)
```

## Quick start

```bash
pip install -r requirements.txt
python run_all.py             # everything below, in one process, with timings
python project_code.py        # closed model  -> figures/fig1..9, results/results.json|txt
python open_system.py         # open model    -> figures/fig10..12 (calibration, forecast, growth decomposition), results/open_system.json|txt
python census_model.py        # census data   -> figures/fig13..15, results/census_model.json|txt
python -m pytest -q           # 39 tests
```
Add `--no-figures` for a fast run. Open `docs/index.html` in a browser for the interactive dashboard
(edit P, switch closed/open, watch the steady state, eigenvalues and projection update).

## Files

| File | What it does |
|---|---|
| `inputs.py` | All inputs in one place: Census 2011/2001 populations, simulated P, SRS 2020 growth, emigration |
| `markov_model.py` | Pure linear-algebra functions: validation, regularity, 3 steady-state methods, diagonalisation, convergence, sensitivity, open-system matrix, homogeneous coordinates, calibration, Perron vector |
| `project_code.py` | Closed-system analysis (the original project, bugs fixed) |
| `open_system.py` | Open-system extension and calibration against the census |
| `census_data.py` | Builds a transition matrix from the Census 2011 migration table D-02 |
| `census_model.py` | Census-data model: real mobility, Newton fit of emigration, independent check against the Economic Survey of Delhi, forecast |
| `run_all.py` | Runs all three analyses in one process and prints timings |
| `data/` | Census, SRS and Economic Survey extracts with `SOURCES.md`; `extract_census.py` rebuilds them from the raw spreadsheets |
| `tests/` | Unit tests, a regression test against the original submission's numbers, and checks that the report, slides and viva notes quote the current results |
| `docs/index.html` | Interactive dashboard (no install; works offline except fonts) |
| `BUGFIXES.md` | Every bug fixed, and why |
| `submission/` | Report (.docx/.pdf, 26 pages), 12 slides and viva notes, **updated** to match this code; also the original script and figures for comparison. The version handed in is tag `v1.0-submission` |

## Key results

**Closed model (simulated P).** Steady state Delhi 55.48 %, Gurugram 17.47 %, Noida 16.73 %,
Faridabad 10.32 %, found three ways (eigenvector, Gaussian elimination, P^200). Eigenvalues
1, 0.8862, 0.8638, 0.8600; |lambda_2| = 0.8862 gives about 32 years to get within 1 % of the steady state.

**Open model.**
* With the simulated P, reproducing Census 2001 -> 2011 needs *negative* immigration into Gurugram and
  Noida, so P overstates mobility. The largest mobility consistent with the census is alpha = 0.137 of P.
* Calibrated immigration: about 346,000 people a year into the four districts (mostly Delhi).
* Long-run growth rho(A) - 1 = 0.63 % a year; total population 21.8 M (2011) -> about 36.8 M (2041).
* Long-run shares (Perron vector of A): Delhi 49.8 %, Gurugram 17.4 %, Noida 21.8 %, Faridabad 11.0 %.

**Census data (D-02).** People who had lived in their district for under a year give real yearly flows.
* Real movement between the four regions is 31-431 times smaller than the simulated P (median 111x):
  about 0.1 % of each region moves to another of the four each year, not 5-12 %.
* With that mobility the closed model needs about 3,400 years to settle (|lambda_2| = 0.99927), and the
  steady state depends on assumptions the data cannot pin down. The closed steady state is a teaching
  result, not a forecast.
* Open model on census data: measured arrivals (about 369,000 a year), SRS 2001-2010 natural growth, and
  emigration fitted by Newton's method to Census 2001 -> 2011. Independent check: modelled net migration
  into Delhi 0.93 lakh/yr vs 0.69 lakh/yr in the Economic Survey of Delhi; the CRS vs SRS birth/death
  data explain 0.20 of the 0.24 lakh gap. Forecast 33.7 M in 2041 (simulated-P model: 36.8 M).

## Performance

All three analyses compute in about 0.15 s; `python run_all.py --no-figures` finishes in about 1 s.
The 1000-run sensitivity analysis is vectorised (one batched eigen-decomposition, 55 ms -> 5 ms) and
plotting libraries are imported only when figures are drawn (`--no-figures`: 1.7 s -> 0.5 s per script).

## Linear algebra used

Column-stochastic matrices · regularity and Wielandt's bound · eigenvectors for lambda = 1 ·
Gaussian elimination with partial pivoting · diagonalisation P = V D V^-1 · spectral gap and convergence rate ·
Perron–Frobenius · homogeneous coordinates (affine -> linear) · closed form x_k = A^k (x_0 - x*) + x* ·
geometric matrix series · calibration as a 4×4 linear system · eigenvalue map under P -> I + alpha(P - I) ·
Newton's method with a finite-difference Jacobian.

## Data and honesty note

* Populations: **Census of India 2011 / 2001** district totals (Gurugram = Gurgaon district,
  Noida = Gautam Buddha Nagar). Faridabad 2001 is back-computed from the official comparable
  decadal growth (32.54 %) because Palwal was split off in 2008.
* Natural growth: **SRS Bulletin 2020** (birth rate - death rate) at **state** level, applied to districts.
* The census-data model (`census_model.py`) uses **measured** flows; where D-02 reports origins only by
  state, flows are allocated by population share (`ncr_weight` tests the effect).
* In the original model, transition probabilities and emigration rates are **simulated** — no public annual district-to-district
  flow matrix exists. The calibration is an exact fit (4 equations, 4 unknowns), not an out-of-sample test.
* This is a teaching model, not an official forecast.
