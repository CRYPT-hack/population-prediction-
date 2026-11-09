# Bugs fixed and changes from the original submission

The original `project_code.py` and the full Group 14 submission are kept unchanged in `submission/` for reference.

| # | Problem in the original | Effect | Fix |
|---|---|---|---|
| 1 | **Wrong Census 2011 figures**: Gurugram 1,514,085 and Noida 1,674,714 | Projections and hand example used wrong data | Gurgaon district 1,514,432; Gautam Buddha Nagar 1,648,115 (Census 2011). Steady state, eigenvalues, scenarios and sensitivity are unaffected (they do not depend on x0) |
| 2 | Eigenvalues sorted by **real value** (`np.argsort(-w)`) | For any P with a negative eigenvalue, \|lambda_2\| (convergence speed) was read from the wrong eigenvalue | Sort by modulus (`markov_model.eigendecomposition`) |
| 3 | `int(np.argmax(dist < thr))` for "years to converge" | Returns **0** ("converged immediately") when the threshold is never reached | `first_below` returns `None` |
| 4 | "Theoretical" convergence constant was fitted from the simulated curve (`C = d_20/|l2|^20`) | Theory-vs-actual comparison was partly circular | `C = ||c_2 v_2||_1` from the eigen-decomposition only. Result: theory 33 / 52 years vs actual 32 / 51 |
| 5 | Gaussian elimination had no zero-pivot check | A singular system (non-regular P) gave silent `inf`/`nan` | Raises `LinAlgError` |
| 6 | Regularity tested up to an arbitrary k = 50 / 200 | Not a proof | Wielandt bound (n-1)^2+1, boolean matrix powers |
| 7 | Design choices written as hard `assert`s (retention in [0.88, 0.96], every share > 5 %, distinct eigenvalues...) | README invited editing P, but almost any edit crashed the script | Split into `check()` (math facts, must hold) and `design()` (prints `[WARN]`) |
| 8 | `results.txt` contained the absolute path `/mnt/user-data/outputs/figures` | Leaked a machine path; README said `figures/` but figures sat in the root | Relative `figures/` and `results/` folders, `--outdir` option |
| 9 | Colours, flow-diagram positions and figure 6 layout hard-coded for 4 named regions | `KeyError` on any renamed / extra region | Generic palette, circular layout, adaptive subplot grid (tested) |
| 10 | Fixed axis limits (fig 4 `ylim 85`, fig 9 `xlim 0.84..1.03`) | Bars / eigenvalues could fall off the plot after editing P | Limits computed from data; fig 9 now shows the complex plane and unit circle with a zoom inset |
| 11 | `warnings.filterwarnings("ignore")`, unused `np.random.seed`, rounding P to 3 decimals in results.json | Hidden problems; lost precision | Removed; P stored with 6 decimals |
| 12 | Everything in one 400-line script | Nothing testable or reusable | `markov_model.py` (pure functions) + scripts + 27 pytest tests |

## Modelling finding (not a code bug, but important)

The simulated P sends 2 % of Delhi to Gurugram every year, which is about 335,000 people, i.e. +22 % of
Gurugram's population in one year. Census 2001 -> 2011 shows Gurugram grew about 74 % in **ten** years.
`open_system.py` shows that matching the census with non-negative migration requires scaling mobility down to
about 14 % of the simulated values (alpha = 0.137). The steady state is unchanged by this scaling, but
convergence is about 7.7 times slower (|lambda_2| goes from 0.886 to 0.984).
