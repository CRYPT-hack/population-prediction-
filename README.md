# NCR Population Movement — Markov-chain model

Predicting how population moves between **Delhi, Gurugram, Noida and Faridabad** with transition
matrices. Started as Group 14's project for MTH2802 Linear Algebra & Differential Equations
(BML Munjal University), then extended with an open-system model, census calibration, tests and an
interactive dashboard.

```
x_{k+1} = P x_k                      closed system: people only move between the 4 regions
x_{k+1} = A x_k + m                  open system:   + births/deaths, emigration, immigration
A = diag(1+g) P diag(1-e)
```

## Quick start

```bash
pip install -r requirements.txt
python project_code.py        # closed model  -> figures/fig1..9, results/results.json|txt
python open_system.py         # open model    -> figures/fig10..12 (calibration, forecast, growth decomposition), results/open_system.json|txt
python -m pytest -q           # 27 tests
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
| `tests/` | Unit tests + regression test against the original submission's numbers |
| `docs/index.html` | Interactive dashboard (no install; works offline except fonts) |
| `BUGFIXES.md` | Every bug fixed, and why |
| `submission/` | The original Group 14 submission: report (.docx/.pdf), slides, viva notes, original script and figures. Its projection tables use the old census figures |

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

## Linear algebra used

Column-stochastic matrices · regularity and Wielandt's bound · eigenvectors for lambda = 1 ·
Gaussian elimination with partial pivoting · diagonalisation P = V D V^-1 · spectral gap and convergence rate ·
Perron–Frobenius · homogeneous coordinates (affine -> linear) · closed form x_k = A^k (x_0 - x*) + x* ·
geometric matrix series · calibration as a 4×4 linear system · eigenvalue map under P -> I + alpha(P - I).

## Data and honesty note

* Populations: **Census of India 2011 / 2001** district totals (Gurugram = Gurgaon district,
  Noida = Gautam Buddha Nagar). Faridabad 2001 is back-computed from the official comparable
  decadal growth (32.54 %) because Palwal was split off in 2008.
* Natural growth: **SRS Bulletin 2020** (birth rate - death rate) at **state** level, applied to districts.
* Transition probabilities and emigration rates are **simulated** — no public annual district-to-district
  flow matrix exists. The calibration is an exact fit (4 equations, 4 unknowns), not an out-of-sample test.
* This is a teaching model, not an official forecast.
