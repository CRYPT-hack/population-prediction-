#!/usr/bin/env python3
"""
Group 14 - MTH2802 Linear Algebra & Differential Equations (BML Munjal University)
Project : Predicting Population Movement Using Transition Matrices  -  CLOSED system
Convention (Poole): P is COLUMN-stochastic, x_{k+1} = P x_k,
                    P[i][j] = Prob(person in region j this year lives in region i next year).
Run     : python project_code.py [--no-figures] [--outdir DIR]
Outputs : figures/fig1..fig9 PNGs, results/results.json, results/results.txt
Inputs  : edit inputs.py. Transition probabilities are SIMULATED; populations are Census 2011.

Two kinds of checks are run:
  check()  - MATHEMATICAL facts that must always hold (failure = bug)      -> AssertionError
  design() - properties of OUR chosen P (e.g. retention in a range); these may legitimately
             fail if you edit P, so they print a [WARN] instead of crashing the script.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch

import markov_model as mm
import inputs

try:
    import seaborn as sns
except ImportError:           # graceful fallback: heatmap drawn with matplotlib only
    sns = None

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
class Reporter:
    def __init__(self, quiet=False):
        self.log, self.passed, self.warnings, self.quiet = [], [], [], quiet

    def out(self, s=""):
        if not self.quiet:
            print(s)
        self.log.append(str(s))

    def check(self, cond, msg):
        assert cond, "FAILED: " + msg
        self.passed.append(msg)
        self.out(f"[PASS] {msg}")

    def design(self, cond, msg):
        if cond:
            self.passed.append(msg)
            self.out(f"[PASS] {msg}")
        else:
            self.warnings.append(msg)
            self.out(f"[WARN] design property not met: {msg}")

    def header(self, title):
        self.out("\n" + "=" * 78)
        self.out(title)
        self.out("=" * 78)


def L(a, d=6):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def region_colors(regions):
    """Okabe-Ito colours; works for any region names (original crashed on renamed regions)."""
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
    return {r: inputs.COLORS.get(r, palette[i % len(palette)]) for i, r in enumerate(regions)}


def run(P=None, initial_population=None, make_figures=True, outdir=HERE, quiet=False):
    """Run the full closed-system analysis. Returns the results dictionary."""
    P = np.array(inputs.P if P is None else P, dtype=float)
    pop = dict(inputs.INITIAL_POPULATION if initial_population is None else initial_population)
    REGIONS = list(pop)
    n = len(REGIONS)
    rep = Reporter(quiet)
    out, check, design = rep.out, rep.check, rep.design
    R = {}

    # ------------------------------------------------------------------------------
    # 1. BUILD P AND VERIFY BASIC PROPERTIES
    # ------------------------------------------------------------------------------
    rep.header("1. TRANSITION MATRIX P (SIMULATED, column-stochastic)")
    if P.shape != (n, n):
        raise ValueError(f"P is {P.shape} but there are {n} regions")
    mm.validate_stochastic(P)
    out(pd.DataFrame(P, index=["to " + r for r in REGIONS], columns=["from " + r for r in REGIONS]).to_string())
    check(np.allclose(P.sum(axis=0), 1, atol=1e-12), "Every column of P sums to 1")
    check((P >= 0).all(), "All entries of P are non-negative")
    design((P > 0).all(), "All entries of P strictly positive (so P^1 > 0)")
    design(all(0.88 <= P[i, i] <= 0.96 for i in range(n)), "Retention (diagonal) lies in [0.88, 0.96]")
    k_reg = mm.regularity_index(P)
    check(k_reg is not None, f"P is regular (smallest k with P^k > 0 is k = {k_reg}; Wielandt bound {(n - 1) ** 2 + 1})")
    R["P"] = L(P, 6); R["regions"] = REGIONS; R["retention"] = {r: float(P[i, i]) for i, r in enumerate(REGIONS)}
    R["column_sums"] = L(P.sum(axis=0), 12); R["smallest_k_regular"] = k_reg
    R["initial_population_census2011"] = pop

    x0 = np.array(list(pop.values()), dtype=float); N0 = x0.sum(); s0 = x0 / N0
    R["total_population"] = int(N0); R["initial_share"] = L(s0)
    out(f"\nTotal initial population (Census 2011) = {int(N0):,};  initial shares = {np.round(s0, 4)}")

    # ------------------------------------------------------------------------------
    # 2. HAND-CHECKABLE EXAMPLE (was labelled 9a and sat before section 2)
    # ------------------------------------------------------------------------------
    rep.header("2. HAND-CHECKABLE EXAMPLE: population of region 2 after 1 year")
    row = 1 if n > 1 else 0
    terms = [P[row, j] * x0[j] for j in range(n)]
    out(f"x_1[{REGIONS[row]}] = sum_j P[{REGIONS[row]}][j] * x_0[j]")
    for j in range(n):
        out(f"   from {REGIONS[j]:<10}: {P[row, j]:.3f} x {x0[j]:>12,.0f} = {terms[j]:>14,.1f}")
    out(f"   TOTAL = {sum(terms):,.1f}  (people in {REGIONS[row]} after 1 year)")
    R["hand_example"] = {"region": REGIONS[row], "row_of_P": L(P[row], 6), "x0": x0.tolist(),
                         "terms": L(terms, 1), "x1_value": round(float(sum(terms)), 1)}
    x1 = P @ x0
    check(abs(x1[row] - sum(terms)) < 1e-6, "Hand-calculation of x_1 matches P @ x_0")
    check(abs(x1.sum() - N0) < 1e-6 * N0, "Total population conserved after 1 step (closed system)")

    # ------------------------------------------------------------------------------
    # 3. PROJECTIONS x_0 ... x_30 AND MATRIX POWERS
    # ------------------------------------------------------------------------------
    rep.header("3. PROJECTIONS x_{k+1} = P x_k")
    K = 30
    X = mm.project(P, x0, K)
    S = X / X.sum(axis=1, keepdims=True)
    show = [0, 1, 2, 5, 10, 20, 30]
    tab = pd.DataFrame({"k (years)": show})
    for i, r in enumerate(REGIONS):
        tab[r + " (persons)"] = [round(X[k, i]) for k in show]
    out(tab.to_string(index=False))
    tab2 = pd.DataFrame({"k (years)": show})
    for i, r in enumerate(REGIONS):
        tab2[r + " (%)"] = [round(100 * S[k, i], 2) for k in show]
    out("\nShares in percent:"); out(tab2.to_string(index=False))
    R["projection_k"] = show
    R["projection_counts"] = {str(k): [round(float(v)) for v in X[k]] for k in show}
    R["projection_shares"] = {str(k): L(S[k], 6) for k in show}
    check(np.allclose(X.sum(axis=1), N0), "Total population constant for all k = 0..30")
    check(np.allclose(S.sum(axis=1), 1), "Shares sum to 1 for all k")

    out("\nMATRIX POWERS (columns converge to the same vector):")
    R["matrix_powers"] = {}; R["power_column_spread"] = {}
    for p_ in [2, 5, 10, 50]:
        Pp = np.linalg.matrix_power(P, p_)
        spread = float(np.max(Pp.max(axis=1) - Pp.min(axis=1)))
        out(f"\nP^{p_} =\n{np.round(Pp, 4)}\n   largest difference between columns in any row = {spread:.2e}")
        R["matrix_powers"][f"P^{p_}"] = L(Pp, 6); R["power_column_spread"][f"P^{p_}"] = spread
    design(R["power_column_spread"]["P^50"] < 1e-2, "Columns of P^50 are (almost) identical")

    # ------------------------------------------------------------------------------
    # 4. STEADY STATE - THREE INDEPENDENT METHODS
    # ------------------------------------------------------------------------------
    rep.header("4. STEADY STATE (three methods)")
    pi_a = mm.steady_state_eig(P)
    A_c, b_c = mm.steady_state_system(P)
    out("Method (b): replace the last equation of (P-I)pi = 0 by pi1+...+pin = 1.")
    out("Augmented matrix [A | b] BEFORE elimination:")
    for r in range(n):
        out("   " + "  ".join(f"{A_c[r, j]:9.4f}" for j in range(n)) + "  |  " + f"{b_c[r]:6.1f}")
    ge_log = []
    pi_b = mm.gauss_solve(A_c, b_c, log=ge_log)
    for line in ge_log:
        out(line)
    pi_c = mm.steady_state_power(P, 200)
    for nm, v in zip(["(a) eigenvector", "(b) linear system", "(c) P^200 column"], [pi_a, pi_b, pi_c]):
        out(f"  {nm:<20}: " + "  ".join(f"{REGIONS[i]} {v[i]:.8f}" for i in range(n)))
    check(np.max(abs(pi_a - pi_b)) < 1e-8, "Steady state: eigenvector method == linear-system method (<1e-8)")
    check(np.max(abs(pi_a - pi_c)) < 1e-8, "Steady state: eigenvector method == P^200 method (<1e-8)")
    check(np.max(abs(pi_b - pi_c)) < 1e-8, "Steady state: linear-system method == P^200 method (<1e-8)")
    pi = pi_a
    check(np.allclose(P @ pi, pi, atol=1e-12), "P pi = pi holds")
    check((pi > 0).all(), "Steady state is strictly positive (Perron-Frobenius for a regular matrix)")
    design(pi.min() > 0.05, f"No region collapses below 5% (smallest share = {100 * pi.min():.2f}%)")
    R["steady_state"] = {"eigenvector": L(pi_a, 10), "linear_system": L(pi_b, 10), "P200": L(pi_c, 10),
                         "max_abs_difference": float(max(abs(pi_a - pi_b).max(), abs(pi_a - pi_c).max()))}
    R["steady_state_counts"] = [round(float(v)) for v in pi * N0]

    # ------------------------------------------------------------------------------
    # 5. EIGEN-DECOMPOSITION
    # ------------------------------------------------------------------------------
    rep.header("5. EIGEN-DECOMPOSITION  P = V D V^-1  (eigenvalues sorted by |lambda|)")
    eig = mm.eigendecomposition(P)
    w, V, Vinv = eig.values, eig.V, eig.Vinv
    real = np.isrealobj(w)
    design(real, "All eigenvalues are real")
    design(len(set(np.round(w, 8))) == n, "All eigenvalues are distinct (sufficient for diagonalisability)")
    D = np.diag(w)
    rec_err = float(np.max(abs(V @ D @ Vinv - P)))
    out("Eigenvalues: " + ", ".join(f"{x:.6f}" for x in w))
    out("Eigenvectors (columns of V):\n" + str(np.round(V, 5)))
    out(f"Reconstruction error max|V D V^-1 - P| = {rec_err:.2e}")
    check(rec_err < 1e-10, "P = V D V^-1 reconstruction error < 1e-10")
    check(abs(w[0] - 1) < 1e-12, "Dominant eigenvalue is exactly 1 (lambda_1 = 1)")
    check(all(abs(w[j]) < 1 - 1e-12 for j in range(1, n)), "All other eigenvalues satisfy |lambda| < 1")
    check(abs(V[:, 0] - pi).max() < 1e-10, "Eigenvector for lambda = 1 equals the steady state")
    check(all(abs(V[:, j].sum()) < 1e-10 for j in range(1, n)), "Other eigenvectors have components summing to 0")
    P10_dir = np.linalg.matrix_power(P, 10); P10_diag = np.real(V @ np.diag(w ** 10) @ Vinv)
    err10 = float(np.max(abs(P10_dir - P10_diag)))
    out(f"P^10 via V D^10 V^-1 vs matrix_power: max error = {err10:.2e}")
    check(err10 < 1e-10, "P^10 = V D^10 V^-1 matches np.linalg.matrix_power(P,10)")
    c = Vinv @ s0
    out("\nDecomposition of the initial SHARE vector: x_0 = c1 v1 + c2 v2 + ... ")
    for j in range(n):
        out(f"   c{j + 1} = {np.real(c[j]): .6f}   (lambda_{j + 1} = {np.real(w[j]):.6f}, |lambda|^30 = {abs(w[j]) ** 30:.3e})")
    check(np.allclose(V @ c, s0), "x_0 = sum c_j v_j reconstruction")
    check(abs(c[0] - 1) < 1e-10, "c1 = 1 for a share vector")
    out("Why: x_k = c1 (1)^k v1 + c2 (l2)^k v2 + ...; since |l2|,|l3|,... < 1 those terms -> 0.")
    R["eigenvalues"] = L(np.real(w), 8); R["eigenvectors_V"] = L(np.real(V), 6); R["V_inverse"] = L(np.real(Vinv), 6)
    R["D"] = L(np.real(D), 8); R["reconstruction_error"] = rec_err; R["P10_error"] = err10
    R["x0_coefficients"] = L(np.real(c), 6); R["eigenvalue_powers_k30"] = [float(abs(v) ** 30) for v in w]
    R["P10_direct"] = L(P10_dir, 6)
    lam2 = float(abs(w[1])); R["lambda2_abs"] = lam2; R["slowest_mode_halflife_years"] = mm.half_life(lam2)
    out(f"Half-life of the slowest transient mode = ln(0.5)/ln|lambda_2| = {mm.half_life(lam2):.2f} years")

    # ------------------------------------------------------------------------------
    # 6. CONVERGENCE SPEED
    # ------------------------------------------------------------------------------
    rep.header("6. CONVERGENCE SPEED")
    dist = mm.distance_curve(P, s0, pi, 120)
    k1, k01 = mm.first_below(dist, 0.01), mm.first_below(dist, 0.001)
    C_fit = float(dist[20] / lam2 ** 20)                       # empirical (as in the original)
    C_th = mm.convergence_constant_theory(eig, s0)             # from V^-1 x_0 only (non-circular)
    out(f"|lambda_2| = {lam2:.6f};  L1 distance d_k = sum_i |x_k,i - pi_i| (shares)")
    out(f"d_0 = {dist[0]:.5f}; C from eigen-decomposition = ||c2 v2||_1 = {C_th:.5f};  empirical d_20/|l2|^20 = {C_fit:.5f}")
    t1, t01 = mm.years_to_threshold(C_th, lam2, 0.01), mm.years_to_threshold(C_th, lam2, 0.001)
    out(f"Years until d_k < 1%   : actual {k1},  theory ceil(ln(0.01/C)/ln|l2|) = {t1}")
    out(f"Years until d_k < 0.1% : actual {k01}, theory ceil(ln(0.001/C)/ln|l2|) = {t01}")
    check(k1 is not None and k01 is not None, "Distance falls below 1% and 0.1% within 120 years")
    design(abs(k1 - t1) <= 2 and abs(k01 - t01) <= 2, "Actual convergence times agree with |lambda_2|^k theory (within 2 years)")
    R["convergence"] = {"d0": float(dist[0]), "C_theory": C_th, "C_fit": C_fit,
                        "years_to_1pct_actual": k1, "years_to_0.1pct_actual": k01,
                        "years_to_1pct_theory": t1, "years_to_0.1pct_theory": t01,
                        "d_at_k": {str(k): float(dist[k]) for k in [0, 1, 2, 5, 10, 20, 30]}}

    # ------------------------------------------------------------------------------
    # 7. MULTIPLE INITIAL DISTRIBUTIONS
    # ------------------------------------------------------------------------------
    rep.header("7. DIFFERENT STARTING POINTS -> SAME LIMIT")
    e_first, e_last = np.eye(n)[0], np.eye(n)[-1]
    starts = {"Census 2011": s0, f"All in {REGIONS[0]}": e_first, f"Uniform ({100 / n:.0f}% each)": np.ones(n) / n,
              f"All in {REGIONS[-1]}": e_last}
    traj = {}; R["multi_start"] = {}
    for nm, s in starts.items():
        T = mm.project(P, s, 120); traj[nm] = T
        out(f"  {nm:<22} share after 120 yrs: {np.round(T[-1], 6)}")
        check(np.abs(T[-1] - pi).max() < 1e-4, f"Start '{nm}' converges to the steady state")
        R["multi_start"][nm] = {"start": L(s), "after_120": L(T[-1], 8)}

    # ------------------------------------------------------------------------------
    # 8. SCENARIOS (only meaningful for the 4 default regions)
    # ------------------------------------------------------------------------------
    rep.header("8. SCENARIO ANALYSIS (all SIMULATED what-if changes)")
    scen = {"Baseline": P}
    if REGIONS == inputs.REGIONS:
        PA = P.copy()  # A: Faridabad improves infrastructure -> retention 0.88 -> 0.92 and more inflow
        PA = mm.replace_column(PA, 0, [0.945, 0.020, 0.020, 0.015])
        PA = mm.replace_column(PA, 1, [0.050, 0.890, 0.030, 0.030])
        PA = mm.replace_column(PA, 2, [0.050, 0.030, 0.890, 0.030])
        PA = mm.replace_column(PA, 3, [0.045, 0.020, 0.015, 0.920])
        PB = P.copy()  # B: job boom in Gurugram and Noida
        PB = mm.replace_column(PB, 0, [0.920, 0.035, 0.035, 0.010])
        PB = mm.replace_column(PB, 1, [0.040, 0.920, 0.025, 0.015])
        PB = mm.replace_column(PB, 2, [0.040, 0.025, 0.920, 0.015])
        PB = mm.replace_column(PB, 3, [0.050, 0.040, 0.030, 0.880])
        scen.update({"A: Faridabad infrastructure": PA, "B: Gurugram/Noida job boom": PB})
    else:
        out("(Custom regions: scenarios A and B skipped - they are written for Delhi/Gurugram/Noida/Faridabad.)")
    R["scenarios"] = {}; rows = []
    for nm, M in scen.items():
        mm.validate_stochastic(M)
        check(mm.regularity_index(M) is not None, f"Scenario '{nm}': valid regular column-stochastic matrix")
        ps = mm.steady_state_eig(M); l2 = mm.second_eigenvalue_modulus(M)
        design(ps.min() > 0.05, f"Scenario '{nm}': smallest steady share {100 * ps.min():.2f}% > 5%")
        ev = np.linalg.eigvals(M)
        R["scenarios"][nm] = {"P": L(M, 6), "steady_state": L(ps, 6), "lambda2_abs": l2,
                              "eigenvalues": L(np.sort(ev.real)[::-1], 6),
                              "change_vs_baseline_pct_points": L(100 * (ps - pi), 4)}
        rows.append([nm] + [f"{100 * v:.2f}" for v in ps] + [f"{l2:.4f}"])
    out(pd.DataFrame(rows, columns=["Scenario"] + [r + " (%)" for r in REGIONS] + ["|lambda_2|"]).to_string(index=False))

    # ------------------------------------------------------------------------------
    # 9. SENSITIVITY
    # ------------------------------------------------------------------------------
    rep.header("9. SENSITIVITY ANALYSIS (1000 random +/-2% perturbations)")
    sens, sens_l2 = mm.sensitivity(P, 1000, 0.02, seed=42)
    sm = pd.DataFrame({"Region": REGIONS, "Baseline %": 100 * pi, "Mean %": 100 * sens.mean(0),
                       "Std %": 100 * sens.std(0, ddof=1), "Min %": 100 * sens.min(0), "Max %": 100 * sens.max(0)})
    out(sm.round(3).to_string(index=False))
    max_dev = float(np.max(100 * np.abs(sens - pi)))
    out(f"Largest deviation of any share from baseline over all 1000 runs = {max_dev:.3f} percentage points")
    out(f"|lambda_2| over runs: mean {sens_l2.mean():.4f}, min {sens_l2.min():.4f}, max {sens_l2.max():.4f}")
    design(max_dev < 3.0, "Steady-state shares stay within 3 percentage points of baseline (robust)")
    R["sensitivity"] = {"n_runs": 1000, "perturbation": "each entry multiplied by U[0.98,1.02], columns renormalised",
                        "per_region": {REGIONS[i]: {"baseline_pct": float(100 * pi[i]), "mean_pct": float(100 * sens[:, i].mean()),
                                                    "std_pct": float(100 * sens[:, i].std(ddof=1)), "min_pct": float(100 * sens[:, i].min()),
                                                    "max_pct": float(100 * sens[:, i].max())} for i in range(n)},
                        "max_deviation_pct_points": max_dev, "lambda2_mean": float(sens_l2.mean()),
                        "lambda2_min": float(sens_l2.min()), "lambda2_max": float(sens_l2.max())}

    # ------------------------------------------------------------------------------
    # 10. 2x2 TOY EXAMPLE and 11. NON-REGULAR CHAINS
    # ------------------------------------------------------------------------------
    rep.header("10. 2x2 TOY EXAMPLE  T = [[0.9, 0.2],[0.1, 0.8]]")
    T = np.array([[0.9, 0.2], [0.1, 0.8]]); tr = T.trace(); det = np.linalg.det(T)
    disc = tr ** 2 - 4 * det; l_1 = (tr + math.sqrt(disc)) / 2; l_2 = (tr - math.sqrt(disc)) / 2
    out(f"Characteristic equation: lambda^2 - {tr:.1f} lambda + {det:.2f} = 0  =>  lambda = {l_1:.1f}, {l_2:.1f}")
    out("lambda = 1: (T - I)v = 0 -> -0.1 a + 0.2 b = 0 -> v1 = (2,1) -> normalised (2/3, 1/3)")
    out("lambda = 0.7: (T - 0.7I)v = 0 -> 0.2 a + 0.2 b = 0 -> v2 = (1,-1)")
    pT = mm.steady_state_eig(T)
    check(np.allclose(sorted(np.linalg.eigvals(T)), [0.7, 1.0]), "Toy 2x2: eigenvalues are 1 and 0.7")
    check(np.allclose(pT, [2 / 3, 1 / 3]), "Toy 2x2: steady state is (2/3, 1/3)")
    toy_seq = mm.project(T, np.array([0.5, 0.5]), 3).tolist()
    out("Start (0.5,0.5): " + " -> ".join(str(np.round(v, 4)) for v in toy_seq))
    R["toy_2x2"] = {"T": T.tolist(), "eigenvalues": [l_1, l_2], "steady_state": pT.tolist(),
                    "sequence_from_(0.5,0.5)": toy_seq, "v1": [2, 1], "v2": [1, -1]}

    rep.header("11. WHAT IF P IS NOT REGULAR?")
    Pabs = np.array([[0.8, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.1, 1.0]])
    mm.validate_stochastic(Pabs)
    check(mm.regularity_index(Pabs) is None, "Absorbing example is NOT regular (proved with Wielandt bound)")
    xa = np.linalg.matrix_power(Pabs, 200) @ np.array([1., 0, 0])
    out(f"Absorbing chain, start in state 1, after 200 steps: {np.round(xa, 6)}  -> everybody ends in state 3")
    Pper = np.array([[0., 1.], [1., 0.]])
    check(mm.regularity_index(Pper) is None, "Periodic example is NOT regular")
    out("Periodic chain [[0,1],[1,0]]: eigenvalues " + str(np.linalg.eigvals(Pper).real) + "; start (1,0) flips forever")
    R["non_regular"] = {"absorbing_P": Pabs.tolist(), "absorbing_limit_from_state1": L(xa, 6),
                        "absorbing_eigenvalues": L(np.sort(np.linalg.eigvals(Pabs).real)[::-1], 6),
                        "periodic_P": Pper.tolist(), "periodic_eigenvalues": [1.0, -1.0]}

    # ------------------------------------------------------------------------------
    # FIGURES + OUTPUTS
    # ------------------------------------------------------------------------------
    outdir = Path(outdir)
    if make_figures:
        make_all_figures(outdir / "figures", P, REGIONS, s0, pi, S, K, dist, C_th, lam2, traj, scen, sens, w)
    R["assertions_passed"] = rep.passed; R["design_warnings"] = rep.warnings
    R["note"] = "Transition probabilities are SIMULATED; populations are Census 2011 district totals."
    R["scenario_table_rows"] = rows
    rep.header(f"CHECKS PASSED: {len(rep.passed)}   DESIGN WARNINGS: {len(rep.warnings)}")
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    (outdir / "results" / "results.json").write_text(json.dumps(R, indent=2))
    (outdir / "results" / "results.txt").write_text("\n".join(rep.log))
    # BUG FIX: the original printed an absolute machine path (/mnt/user-data/outputs/figures) into results.txt
    return R


def make_all_figures(FIG, P, REGIONS, s0, pi, S, K, dist, Cth, lam2, traj, scen, sens, w):
    FIG.mkdir(parents=True, exist_ok=True)
    n = len(REGIONS); COLORS = region_colors(REGIONS); REG_COL = [COLORS[r] for r in REGIONS]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
                         "axes.grid": True, "grid.alpha": 0.3})

    def savefig(name):
        plt.tight_layout(); plt.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close()

    # fig1 heatmap
    fig, ax = plt.subplots(figsize=(7, 5.8))
    lab_f = ["from " + r for r in REGIONS]; lab_t = ["to " + r for r in REGIONS]
    if sns is not None:
        sns.heatmap(P, annot=True, fmt=".3f", cmap="viridis", xticklabels=lab_f, yticklabels=lab_t, ax=ax,
                    cbar_kws={"label": "Probability"}, annot_kws={"size": 12})
    else:
        im = ax.imshow(P, cmap="viridis"); plt.colorbar(im, label="Probability")
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(lab_f); ax.set_yticklabels(lab_t)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center", color="w")
    ax.set_title("Fig 1: Transition matrix P (columns sum to 1)")
    ax.set_xlabel("Region this year (column j)"); ax.set_ylabel("Region next year (row i)")
    ax.grid(False); plt.xticks(rotation=30); savefig("fig1_transition_heatmap.png")

    # fig2 flow diagram - fixed layout for the 4 NCR regions, circle layout for anything else
    default_pos = {"Delhi": (0, 1), "Gurugram": (-1.4, -0.6), "Noida": (1.4, -0.6), "Faridabad": (0, -1.8)}
    if set(REGIONS) == set(default_pos):
        pos = default_pos
    else:
        pos = {r: (1.6 * math.cos(math.pi / 2 - 2 * math.pi * i / n), 1.6 * math.sin(math.pi / 2 - 2 * math.pi * i / n) - 0.4)
               for i, r in enumerate(REGIONS)}
    fig, ax = plt.subplots(figsize=(8.5, 7.5)); ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
    for i, ri in enumerate(REGIONS):
        for j, rj in enumerate(REGIONS):
            if i == j or P[i, j] <= 0:
                continue
            (xa_, ya_), (xb_, yb_) = pos[rj], pos[ri]
            ax.add_patch(FancyArrowPatch((xa_, ya_), (xb_, yb_), connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                                         mutation_scale=14 + 220 * P[i, j], lw=0.6 + 90 * P[i, j], color=COLORS[rj],
                                         alpha=0.75, shrinkA=34, shrinkB=34))
            dx, dy = xb_ - xa_, yb_ - ya_
            mx, my = (xa_ + xb_) / 2 + 0.09 * dy, (ya_ + yb_) / 2 - 0.09 * dx
            ax.text(mx, my, f"{P[i, j]:.3f}", fontsize=9, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    for k, r in enumerate(REGIONS):
        px, py = pos[r]; txt = "white" if COLORS[r] in ("#0072B2", "#009E73", "#000000", "#D55E00") else "black"
        ax.scatter([px], [py], s=5200, color=COLORS[r], edgecolor="k", zorder=5)
        ax.text(px, py + 0.07, r, ha="center", va="center", fontsize=11, fontweight="bold", color=txt, zorder=6)
        ax.text(px, py - 0.14, f"stay {P[k, k]:.3f}", ha="center", va="center", fontsize=9, color=txt, zorder=6)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.6, 1.8)
    ax.set_title("Fig 2: Yearly movement between regions (arrow thickness ~ probability; colour = origin)")
    savefig("fig2_flow_diagram.png")

    # fig3 shares over time
    fig, axs = plt.subplots(1, 2, figsize=(13, 5)); yrs = np.arange(K + 1)
    for i, r in enumerate(REGIONS):
        axs[0].plot(yrs, 100 * S[:, i], color=COLORS[r], lw=2.2, label=r)
        axs[0].axhline(100 * pi[i], color=COLORS[r], ls=":", lw=1)
    axs[0].set_title("Fig 3a: Population share over time (dotted = steady state)")
    axs[0].set_xlabel("Years k"); axs[0].set_ylabel("Share of total population (%)"); axs[0].legend()
    axs[1].stackplot(yrs, 100 * S.T, labels=REGIONS, colors=REG_COL, alpha=0.9)
    axs[1].set_title("Fig 3b: Stacked area of shares"); axs[1].set_xlabel("Years k"); axs[1].set_ylabel("Share (%)")
    axs[1].set_xlim(0, K); axs[1].set_ylim(0, 100); axs[1].legend(loc="center right")
    savefig("fig3_population_over_time.png")

    # fig4 initial vs steady (y-limit now adapts; original was fixed at 85 %)
    fig, ax = plt.subplots(figsize=(8.5, 5)); xx = np.arange(n); wd = 0.38
    b1 = ax.bar(xx - wd / 2, 100 * s0, wd, color=REG_COL, edgecolor="k", label="Initial (Census 2011)")
    b2 = ax.bar(xx + wd / 2, 100 * pi, wd, color=REG_COL, edgecolor="k", hatch="//", label="Steady state")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", fontsize=10)
    ax.set_xticks(xx); ax.set_xticklabels(REGIONS); ax.set_ylabel("Share of total population (%)")
    ax.set_title("Fig 4: Initial vs steady-state distribution"); ax.legend()
    ax.set_ylim(0, min(100, 100 * max(s0.max(), pi.max()) * 1.12 + 2)); savefig("fig4_initial_vs_steady.png")

    # fig5 convergence
    fig, ax = plt.subplots(figsize=(8.5, 5)); kk = np.arange(len(dist))
    ax.semilogy(kk, dist, "o-", ms=3, color="#0072B2", label="Actual distance $d_k$ (L1)")
    ax.semilogy(kk, Cth * lam2 ** kk, "--", color="#D55E00", lw=2,
                label=f"Theory $\\|c_2 v_2\\|_1|\\lambda_2|^k$, $|\\lambda_2|$={lam2:.4f}")
    ax.axhline(0.01, color="gray", ls=":", label="1% threshold"); ax.axhline(0.001, color="k", ls=":", label="0.1% threshold")
    ax.set_ylim(1e-8, 2); ax.set_xlim(0, len(dist) - 1); ax.set_xlabel("Years k"); ax.set_ylabel("Distance to steady state (log scale)")
    ax.set_title("Fig 5: Convergence to steady state (log scale)"); ax.legend(); savefig("fig5_convergence.png")

    # fig6 several starts
    rows_ = math.ceil(n / 2)
    fig, axs = plt.subplots(rows_, 2, figsize=(12, 4 * rows_), squeeze=False)
    mk = ["-", "--", "-.", ":"]; scol = ["#000000", "#D55E00", "#56B4E9", "#666666"]
    for i, r in enumerate(REGIONS):
        ax = axs.flat[i]
        for (nm, tr_), ls, cc in zip(traj.items(), mk, scol):
            ax.plot(100 * tr_[:81, i], ls=ls, color=cc, lw=2, label=nm)
        ax.axhline(100 * pi[i], color=COLORS[r], lw=3, alpha=0.5, label="Steady state")
        ax.set_title(r); ax.set_xlabel("Years k"); ax.set_ylabel("Share (%)")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=8)
    fig.suptitle("Fig 6: Different starting points converge to the same steady state", fontsize=14)
    savefig("fig6_multiple_starts.png")

    # fig7 scenarios
    fig, ax = plt.subplots(figsize=(10, 5.5)); m_ = len(scen); wd = 0.8 / m_; hat = ["", "//", "..", "xx"]
    for s_i, (nm, M) in enumerate(scen.items()):
        ps = mm.steady_state_eig(M)
        bars = ax.bar(xx + (s_i - (m_ - 1) / 2) * wd, 100 * ps, wd, color=REG_COL, edgecolor="k", hatch=hat[s_i % 4], label=nm)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(REGIONS); ax.set_ylabel("Steady-state share (%)")
    ax.set_title("Fig 7: Steady state under scenarios (hatch pattern = scenario)")
    ax.legend(handles=[Patch(fc="white", ec="k", hatch=hat[i % 4], label=nm) for i, nm in enumerate(scen)])
    savefig("fig7_scenarios.png")

    # fig8 sensitivity
    fig, axs = plt.subplots(1, n, figsize=(3.25 * n, 5), squeeze=False)
    for i, r in enumerate(REGIONS):
        ax = axs[0, i]
        bp = ax.boxplot([100 * sens[:, i]], patch_artist=True, widths=0.5, medianprops=dict(color="k", lw=2))
        bp["boxes"][0].set_facecolor(COLORS[r]); bp["boxes"][0].set_alpha(0.9)
        ax.scatter([1], [100 * pi[i]], marker="D", color="k", zorder=5, label="Baseline")
        ax.set_xticks([1]); ax.set_xticklabels([r]); ax.set_ylabel("Steady-state share (%)" if i == 0 else "")
        if i == 0:
            ax.legend()
    fig.suptitle("Fig 8: Sensitivity of steady-state shares to +/-2% perturbations (1000 runs)", fontsize=13)
    savefig("fig8_sensitivity_boxplot.png")

    # fig9 eigenvalues - now in the complex plane with the unit circle (handles complex/negative eigenvalues)
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="#56B4E9", lw=1.5, label="Unit circle |lambda| = 1")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    for j, lam in enumerate(np.atleast_1d(w)):
        lam = complex(lam)
        ax.scatter(lam.real, lam.imag, s=110, color="#D55E00" if j == 0 else "#0072B2", edgecolor="k", zorder=5)
        if np.ptp(np.real(np.atleast_1d(w))) >= 0.5 or not np.allclose(np.imag(w), 0):
            ax.annotate(f"$\\lambda_{j + 1}$={lam.real:.3f}{lam.imag:+.3f}i", (lam.real, lam.imag), xytext=(-30, 12 + 14 * j),
                        textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    wr = np.real(np.atleast_1d(w))
    if np.ptp(wr) < 0.5 and np.allclose(np.imag(w), 0):     # eigenvalues bunched together: add a zoomed real-line inset
        ins = ax.inset_axes([0.08, 0.62, 0.6, 0.25]); ins.set_facecolor("white"); ins.set_zorder(10)
        lo, hi = wr.min() - 0.02, max(1.0, wr.max()) + 0.02
        ins.axhline(0, color="k", lw=0.8); ins.set_xlim(lo, hi); ins.set_ylim(-1, 1.6); ins.set_yticks([])
        for j, lam in enumerate(wr):
            ins.scatter(lam, 0, s=60, color="#D55E00" if j == 0 else "#0072B2", edgecolor="k", zorder=5)
            ins.annotate(f"$\\lambda_{j + 1}$={lam:.4f}", (lam, 0), xytext=(0, 8 + 11 * (j % 3)), textcoords="offset points",
                         ha="center", fontsize=8)
        ins.set_title("zoom on the real axis", fontsize=9); ins.tick_params(labelsize=8)
    ax.set_xlabel("Re"); ax.set_ylabel("Im"); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Fig 9: Eigenvalues of P (only $\\lambda_1$ = 1 on the unit circle)")
    savefig("fig9_eigenvalues.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation (faster)")
    ap.add_argument("--outdir", default=str(HERE), help="where figures/ and results/ are written")
    a = ap.parse_args()
    run(make_figures=not a.no_figures, outdir=a.outdir)
