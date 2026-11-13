#!/usr/bin/env python3
"""
OPEN-SYSTEM extension: births, deaths and migration across the NCR boundary.

The closed model x_{k+1} = P x_k freezes the total population at 21.8 million forever.
Here every year
    1. a fraction e_j of region j's residents leaves the NCR          (EMIGRATION, simulated)
    2. the rest move between regions according to P                   (P, simulated)
    3. natural growth g_i = birth rate - death rate is applied        (SRS 2020, real)
    4. m_i people arrive from the rest of India                       (CALIBRATED to Census 2001 -> 2011)

    x_{k+1} = A x_k + m,     A = diag(1+g) P diag(1-e)

Linear-algebra ideas used
    * homogeneous coordinates: [x_{k+1};1] = [[A, m],[0, 1]] [x_k;1]  (affine -> linear)
    * closed-form solution     x_k = A^k (x_0 - x*) + x*,  x* = (I - A)^-1 m
    * Perron-Frobenius         long-run growth factor rho(A) and long-run shares = Perron vector
    * calibration              solve S_10 m = x_2011 - A^10 x_2001 (4x4 linear system)

Run: python open_system.py [--no-figures] [--outdir DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import inputs
import markov_model as mm
from project_code import Reporter, L, region_colors

HERE = Path(__file__).resolve().parent
BASE_YEAR = 2011


def run(make_figures=True, outdir=HERE, quiet=False, horizon=30):
    REG = inputs.REGIONS; n = len(REG)
    rep = Reporter(quiet); out, check, design = rep.out, rep.check, rep.design
    P = np.array(inputs.P, float)
    e = np.array([inputs.EMIGRATION[r] for r in REG])
    g = np.array([inputs.NATURAL_GROWTH[r] for r in REG])
    x01 = np.array([inputs.POPULATION_2001[r] for r in REG], float)
    x11 = np.array([inputs.INITIAL_POPULATION[r] for r in REG], float)
    R = {"regions": REG, "emigration": L(e), "natural_growth": L(g), "population_2001": L(x01, 0),
         "population_2011": L(x11, 0), "sources": inputs.SOURCES}

    # ------------------------------------------------------------------------------
    rep.header("1. CAN THE SIMULATED P EXPLAIN THE CENSUS?  (calibration 2001 -> 2011)")
    A_full = mm.open_system_matrix(P, e, g)
    m_full = mm.calibrate_immigration(A_full, x01, x11, 10)
    out("With the simulated P at full strength, the immigration needed to reproduce Census 2011 is:")
    out("   " + ", ".join(f"{r} {v:,.0f}/yr" for r, v in zip(REG, m_full)))
    if m_full.min() < 0:
        out("[FINDING] Negative immigration is impossible, so the simulated P moves too many people")
        out("          out of Delhi each year (2% to Gurugram = ~335,000 people/yr, i.e. +22% of Gurugram).")
    out("Fix: scale mobility, P_alpha = I + alpha (P - I) (same steady state, slower), and find the")
    out("largest alpha for which every m_i >= 0.")
    alpha, m, A = mm.calibrate_mobility(P, e, g, x01, x11, 10)
    out(f"\n   alpha* = {alpha:.4f}  ->  the data support about {100 * alpha:.0f}% of the simulated mobility")
    out("   calibrated immigration m = " + ", ".join(f"{r} {v:,.0f}/yr" for r, v in zip(REG, m)))
    out(f"   total net immigration into the 4 districts = {m.sum():,.0f} people / year")
    Palpha = mm.scale_mobility(P, alpha)
    check(np.allclose(Palpha.sum(0), 1), "P_alpha is still column-stochastic")
    check(np.allclose(mm.steady_state_eig(Palpha), mm.steady_state_eig(P)), "P_alpha has the same steady state as P")
    lam_P = np.sort(np.linalg.eigvals(P).real)[::-1]; lam_Pa = np.sort(np.linalg.eigvals(Palpha).real)[::-1]
    check(np.allclose(lam_Pa, 1 + alpha * (lam_P - 1)), "eigenvalues of P_alpha are 1 + alpha (lambda - 1)")
    check((m >= -1e-6).all(), "Calibrated immigration is non-negative")
    back = mm.project_open(A, m, x01, 10)[-1]
    check(np.allclose(back, x11, rtol=1e-9), "Back-cast 2001 -> 2011 reproduces Census 2011 exactly")
    R["calibration"] = {"m_needed_at_full_mobility": L(m_full, 0), "alpha": alpha, "immigration_per_year": L(m, 0),
                        "P_alpha": L(Palpha, 6), "A": L(A, 8),
                        "note": "exact fit: 4 equations, 4 unknowns; NOT an out-of-sample validation"}

    # ------------------------------------------------------------------------------
    rep.header("2. HOMOGENEOUS COORDINATES AND CLOSED-FORM SOLUTION")
    B = mm.homogeneous_matrix(A, m)
    with np.printoptions(suppress=True, precision=5, linewidth=120):
        out("B = [[A, m],[0, 1]] =\n" + str(B))
    X = mm.project_open(A, m, x11, horizon)
    XB = np.array([np.linalg.matrix_power(B, k) @ np.append(x11, 1) for k in range(horizon + 1)])[:, :n]
    check(np.allclose(X, XB, rtol=1e-10), "Linear 5x5 iteration with B equals affine iteration x -> A x + m")
    eigB = np.sort(abs(np.linalg.eigvals(B)))[::-1]; eigA = np.sort(abs(np.linalg.eigvals(A)))[::-1]
    check(np.allclose(np.sort(np.append(eigA, 1.0))[::-1], eigB), "eig(B) = eig(A) together with 1 (block-triangular)")
    xstar = np.linalg.solve(np.eye(n) - A, m)
    Xcf = np.array([np.linalg.matrix_power(A, k) @ (x11 - xstar) + xstar for k in range(horizon + 1)])
    check(np.allclose(X, Xcf, rtol=1e-8), "Closed form x_k = A^k (x_0 - x*) + x* matches iteration")
    R["homogeneous_B"] = L(B, 8); R["x_star"] = L(xstar, 0)

    # ------------------------------------------------------------------------------
    rep.header("3. LONG-RUN BEHAVIOUR (Perron-Frobenius)")
    rho, v = mm.perron(A)
    check((A > 0).all() and (v > 0).all(), "A > 0, so its dominant eigenvector is strictly positive")
    out(f"rho(A) = {rho:.6f}  ->  long-run growth {100 * (rho - 1):.3f}% per year")
    out("Long-run shares (Perron vector of A): " + ", ".join(f"{r} {100 * s:.2f}%" for r, s in zip(REG, v)))
    pi = mm.steady_state_eig(P)
    out("Closed-model steady state for comparison: " + ", ".join(f"{r} {100 * s:.2f}%" for r, s in zip(REG, pi)))
    if rho >= 1:
        out("rho >= 1: the population does NOT settle; x* is not an attractor, shares tend to the Perron vector.")
    else:
        out(f"rho < 1: the population settles at x* = (I - A)^-1 m, total {xstar.sum():,.0f}.")
    R["perron"] = {"rho": rho, "growth_pct_per_year": 100 * (rho - 1), "long_run_shares": L(v)}

    # ------------------------------------------------------------------------------
    rep.header(f"4. FORECAST {BASE_YEAR} -> {BASE_YEAR + horizon}")
    years = BASE_YEAR + np.arange(horizon + 1)
    show = [k for k in (0, 5, 10, 15, 20, 25, 30) if k <= horizon]
    tab = pd.DataFrame({"Year": years[show]})
    for i, r in enumerate(REG):
        tab[r] = [f"{X[k, i] / 1e6:.2f} M" for k in show]
    tab["Total"] = [f"{X[k].sum() / 1e6:.2f} M" for k in show]
    out(tab.to_string(index=False))
    from_x0 = np.linalg.matrix_power(A, horizon) @ x11
    from_m = mm.geometric_sum(A, horizon) @ m
    check(np.allclose(from_x0 + from_m, X[-1], rtol=1e-9), "x_K = A^K x_0 + S_K m (decomposition of growth)")
    out(f"\nOf the {X[-1].sum() / 1e6:.2f} M people in {years[-1]}: {from_x0.sum() / 1e6:.2f} M descend from the "
        f"{BASE_YEAR} residents, {from_m.sum() / 1e6:.2f} M from immigration after {BASE_YEAR}.")
    Xc = mm.project(P, x11, horizon)
    R["forecast"] = {"years": years.tolist(), "open": L(X, 0), "closed": L(Xc, 0),
                     "from_initial": L(from_x0, 0), "from_immigration": L(from_m, 0)}

    # ------------------------------------------------------------------------------
    rep.header("5. WHAT-IF: sensitivity of the 2041 forecast")
    whatif = {}
    for nm, (ee, gg, mmul) in {"Baseline": (e, g, 1.0), "Emigration x2": (2 * e, g, 1.0),
                               "Natural growth -30%": (e, 0.7 * g, 1.0), "Immigration +50%": (e, g, 1.5),
                               "Immigration stops": (e, g, 0.0)}.items():
        Aw = mm.open_system_matrix(Palpha, ee, gg)
        Xw = mm.project_open(Aw, mmul * m, x11, horizon)
        whatif[nm] = L(Xw[-1], 0)
        out(f"  {nm:<22} total {years[-1]}: {Xw[-1].sum() / 1e6:6.2f} M   shares " +
            " ".join(f"{100 * s:5.1f}%" for s in Xw[-1] / Xw[-1].sum()))
    R["what_if_final_year"] = whatif

    outdir = Path(outdir)
    if make_figures:
        figures(outdir / "figures", REG, years, X, Xc, from_x0, from_m, v, pi, alpha, m_full, m)
    R["assertions_passed"] = rep.passed; R["design_warnings"] = rep.warnings
    rep.header(f"CHECKS PASSED: {len(rep.passed)}   DESIGN WARNINGS: {len(rep.warnings)}")
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    (outdir / "results" / "open_system.json").write_text(json.dumps(R, indent=2))
    (outdir / "results" / "open_system.txt").write_text("\n".join(rep.log))
    return R


def figures(FIG, REG, years, X, Xc, from_x0, from_m, v, pi, alpha, m_full, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True); C = region_colors(REG); n = len(REG)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    def save(name):
        plt.tight_layout(); plt.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for i, r in enumerate(REG):
        axs[0].plot(years, X[:, i] / 1e6, color=C[r], lw=2.2, label=f"{r} (open)")
        axs[0].plot(years, Xc[:, i] / 1e6, color=C[r], lw=1.4, ls="--")
    axs[0].set_title("Fig 11a: Population, open (solid) vs closed (dashed)"); axs[0].set_ylabel("Millions"); axs[0].legend(fontsize=9)
    axs[1].plot(years, X.sum(1) / 1e6, color="k", lw=2.4, label="Open model")
    axs[1].plot(years, Xc.sum(1) / 1e6, color="gray", lw=2, ls="--", label="Closed model (constant)")
    axs[1].set_title("Fig 11b: Total population of the four districts"); axs[1].set_ylabel("Millions"); axs[1].legend()
    save("fig11_open_vs_closed.png")

    fig, ax = plt.subplots(figsize=(10, 5)); xx = np.arange(n)
    ax.bar(xx - 0.2, from_x0 / 1e6, 0.4, color=[C[r] for r in REG], edgecolor="k", label="Descendants of 2011 residents")
    ax.bar(xx - 0.2, from_m / 1e6, 0.4, bottom=from_x0 / 1e6, color="white", edgecolor="k", hatch="//", label="Immigrants after 2011 (+ descendants)")
    ax.bar(xx + 0.2, X[0] / 1e6, 0.4, color="lightgray", edgecolor="k", label="2011 Census")
    ax.set_xticks(xx); ax.set_xticklabels(REG); ax.set_ylabel("Millions")
    ax.set_title(f"Fig 12: Where the {years[-1]} population comes from  ($x_K = A^K x_0 + S_K m$)"); ax.legend()
    save("fig12_growth_decomposition.png")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].bar(xx - 0.2, m_full / 1e3, 0.4, color="#D55E00", edgecolor="k", label="Simulated P at full strength")
    axs[0].bar(xx + 0.2, m / 1e3, 0.4, color="#009E73", edgecolor="k", label=f"Calibrated, alpha = {alpha:.3f}")
    axs[0].axhline(0, color="k"); axs[0].set_xticks(xx); axs[0].set_xticklabels(REG)
    axs[0].set_ylabel("Thousand people / year"); axs[0].set_title("Fig 10a: Immigration needed to match Census 2011"); axs[0].legend(fontsize=9)
    axs[1].bar(xx - 0.2, 100 * pi, 0.4, color=[C[r] for r in REG], edgecolor="k", label="Closed steady state")
    axs[1].bar(xx + 0.2, 100 * v, 0.4, color=[C[r] for r in REG], edgecolor="k", hatch="//", label="Open: Perron vector of A")
    axs[1].set_xticks(xx); axs[1].set_xticklabels(REG); axs[1].set_ylabel("Long-run share (%)")
    axs[1].set_title("Fig 10b: Long-run shares, closed vs open"); axs[1].legend(fontsize=9)
    save("fig10_calibration_and_longrun.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--horizon", type=int, default=30, help="years to forecast after 2011")
    a = ap.parse_args()
    run(make_figures=not a.no_figures, outdir=a.outdir, horizon=a.horizon)
