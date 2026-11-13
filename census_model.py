#!/usr/bin/env python3
"""
REAL-DATA version of the model, built from the Census 2011 migration table D-02.

1. Annual flows between the four regions from migrants with < 1 year duration of residence,
   giving a census-based transition matrix P_c (compare with the simulated P).
2. Spectrum of P_c: how long the closed model would really take to settle.
3. Open model x_{k+1} = A x_k + m with
     m  = yearly arrivals from outside the four regions          (measured, D-02)
     g  = SRS 2001-2010 average natural growth                     (measured)
     e  = emigration out of the four regions                       (fitted by Newton's method to Census 2001 -> 2011)
4. Independent check: Delhi's yearly net migration from the model vs the Economic Survey of Delhi 2014-15.
5. Forecast 2011 -> 2041 and comparison with the simulated-P open model.

Run: python census_model.py [--no-figures] [--outdir DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import census_data as cd
import inputs
import markov_model as mm
from project_code import Reporter, L, region_colors

HERE = Path(__file__).resolve().parent
WEIGHTS = (1.0, 2.0, 4.0)          # proximity factor for allocating state-level flows (see census_data.py)


def years_to_settle(P, start, thr=0.01, kmax=20000):
    pi = mm.steady_state_eig(P)
    x = np.array(start, float)
    for k in range(kmax + 1):
        if np.abs(x - pi).sum() < thr:
            return k
        x = P @ x
    return None


def run(make_figures=True, outdir=HERE, quiet=False):
    REG = cd.REGIONS
    rep = Reporter(quiet); out, check, design = rep.out, rep.check, rep.design
    pop = cd.load_population()
    x11 = np.array([pop[r] for r in REG], float)
    check(np.array_equal(x11, np.array([inputs.INITIAL_POPULATION[r] for r in REG], float)),
          "Census 2011 populations in inputs.py match the Primary Census Abstract file")
    x01 = np.array([inputs.POPULATION_2001[r] for r in REG], float)
    R = {"regions": REG, "sources": "data/SOURCES.md"}

    # ------------------------------------------------------------------------------------------
    rep.header("1. CENSUS-BASED TRANSITION MATRIX (D-02, migrants resident < 1 year)")
    F, m, notes = cd.flow_matrix(1.0)
    Pc, _, _ = cd.census_transition_matrix(1.0)
    with np.printoptions(suppress=True, linewidth=120):
        out("Yearly flows F[i, j] (people moving from column j to row i):")
        out(pd.DataFrame(F.round(0), index=["to " + r for r in REG], columns=["from " + r for r in REG]).to_string())
        out("\nHow each flow was obtained:")
        for (d, o), how in notes.items():
            out(f"   {o:>9} -> {d:<9}: {how}")
        out("\nCensus-based P_c (ncr_weight = 1):\n" + pd.DataFrame(Pc, index=REG, columns=REG).round(6).to_string())
    mm.validate_stochastic(Pc)
    check(mm.regularity_index(Pc) == 1, "P_c is strictly positive, hence regular")
    ratio = inputs.P / Pc
    off = ~np.eye(4, dtype=bool)
    out(f"\nSimulated P / census P_c, off-diagonal: min {ratio[off].min():.0f}x, median {np.median(ratio[off]):.0f}x, max {ratio[off].max():.0f}x")
    out(f"Share of each region moving to the other three per year: " +
        ", ".join(f"{r} {100 * (1 - Pc[i, i]):.3f}%" for i, r in enumerate(REG)) + "  (simulated P: 5-12%)")
    out("Arrivals per year from outside the four regions, m = " + ", ".join(f"{r} {v:,.0f}" for r, v in zip(REG, m)))
    robust = {}
    for w in WEIGHTS:
        for col in ("moved_last_year", "moved_1_4_years"):
            P_, m_, _ = cd.census_transition_matrix(w, col)
            robust[f"w={w:g}, {col}"] = {"steady_state": L(mm.steady_state_eig(P_), 4), "lambda2": mm.second_eigenvalue_modulus(P_),
                                         "max_offdiag": float(P_[off].max()), "arrivals": L(m_, 0)}
    out("\nRobustness (allocation weight w, and 'moved last year' vs 'moved 1-4 years ago' / 4):")
    for k, v in robust.items():
        out(f"   {k:<28} steady state {[round(100 * s, 1) for s in v['steady_state']]}%  |lambda_2| = {v['lambda2']:.5f}")
    R["flows"] = L(F, 0); R["P_census"] = L(Pc, 8); R["arrivals_outside"] = L(m, 0)
    R["ratio_simulated_to_census"] = {"min": float(ratio[off].min()), "median": float(np.median(ratio[off])), "max": float(ratio[off].max())}
    R["robustness"] = robust; R["allocation_notes"] = {f"{o}->{d}": h for (d, o), h in notes.items()}

    # ------------------------------------------------------------------------------------------
    rep.header("2. HOW LONG WOULD THE CLOSED MODEL TAKE TO SETTLE WITH REAL MOBILITY?")
    eig = mm.eigendecomposition(Pc); lam2 = abs(eig.values[1])
    s0 = x11 / x11.sum()
    k_real = years_to_settle(Pc, s0); k_sim = years_to_settle(inputs.P, s0)
    out(f"Eigenvalues of P_c: " + ", ".join(f"{v:.6f}" for v in np.real(eig.values)))
    out(f"|lambda_2| = {lam2:.6f}  ->  half-life {mm.half_life(lam2):,.0f} years")
    out(f"Years to come within 1% of the steady state: census P_c {k_real:,}  vs simulated P {k_sim}")
    sp = [v["steady_state"] for v in robust.values()]
    out("The steady state itself is NOT pinned down by the data: Delhi's long-run share ranges "
        f"{100 * min(s[0] for s in sp):.0f}%-{100 * max(s[0] for s in sp):.0f}% across the allocation assumptions.")
    check(k_real > 1000, "With census mobility the closed model needs over a thousand years to settle")
    R["census_spectrum"] = {"eigenvalues": L(np.real(eig.values), 8), "lambda2": lam2, "half_life_years": mm.half_life(lam2),
                            "years_to_1pct_census": k_real, "years_to_1pct_simulated": k_sim}

    # ------------------------------------------------------------------------------------------
    rep.header("3. OPEN MODEL ON CENSUS DATA: fit emigration to Census 2001 -> 2011 (Newton)")
    g01 = np.array([cd.srs_growth(2001, 2010)[r] for r in REG])
    out("Natural growth g (SRS 2001-2010 average, state level): " + ", ".join(f"{r} {100 * v:.2f}%" for r, v in zip(REG, g01)))
    e, iters = mm.calibrate_emigration(Pc, g01, m, x01, x11, 10)
    A = mm.open_system_matrix(Pc, e, g01)
    X = mm.project_open(A, m, x01, 10)
    out(f"Newton converged in {iters} iterations: e = " + ", ".join(f"{r} {100 * v:.2f}%" for r, v in zip(REG, e)))
    check(np.allclose(X[-1], x11, rtol=1e-9), "Back-cast 2001 -> 2011 reproduces Census 2011")
    check((e >= 0).all() and (e < 0.05).all(), "Fitted emigration rates are between 0% and 5% a year")
    out("Noida and Gurugram need the highest emigration because state-wide (UP, Haryana) birth rates overstate")
    out("the natural growth of these urban districts; emigration absorbs that difference.")
    R["open_census"] = {"growth_2001_2010": L(g01), "emigration": L(e), "newton_iterations": iters, "A": L(A, 8)}

    # ------------------------------------------------------------------------------------------
    rep.header("4. INDEPENDENT CHECK: Delhi's net migration vs Economic Survey of Delhi 2014-15")
    es = pd.read_csv(cd.DATA / "delhi_migration_2001_2012.csv").dropna()
    model_nm = []
    for t in range(1, 11):
        moved = Pc @ ((1 - e) * X[t - 1])
        nat = g01[0] * moved[0]
        model_nm.append((X[t][0] - X[t - 1][0] - nat) / 1e5)
    years = list(range(2002, 2012))
    survey = es.set_index("year").loc[years, "net_migration_lakh"].to_numpy()
    tab = pd.DataFrame({"Year": years, "Model (lakh)": np.round(model_nm, 2), "Economic Survey (lakh)": survey})
    out(tab.to_string(index=False))
    out(f"Average: model {np.mean(model_nm):.2f} lakh/yr, survey {survey.mean():.2f} lakh/yr. They agree best in 2010-2011,")
    out("the years the D-02 flows were measured; earlier, the model assumes the 2011 arrival numbers held all decade.")
    crs_ni = 1000 * (es.births_lakh / es.mid_year_population_lakh - es.deaths_lakh / es.mid_year_population_lakh)[es.year <= 2010].mean()
    gap_ni = (crs_ni / 1000 - g01[0]) * X[:10, 0].mean() / 1e5
    out(f"The survey's natural increase comes from CRS registrations ({crs_ni:.1f} per 1000) while the model uses SRS")
    out(f"({1000 * g01[0]:.1f} per 1000). That difference alone is {gap_ni:.2f} lakh/yr of the {np.mean(model_nm) - survey.mean():.2f} lakh/yr gap.")
    R_gap = {"crs_natural_increase_per_1000": float(crs_ni), "srs_natural_increase_per_1000": float(1000 * g01[0]), "gap_lakh": float(gap_ni)}
    check(0.5 < np.mean(model_nm) / survey.mean() < 2.0, "Model and survey net migration agree within a factor of 2")
    R["delhi_net_migration"] = {"years": years, "model_lakh": L(model_nm, 3), "survey_lakh": survey.tolist(), **R_gap}

    # ------------------------------------------------------------------------------------------
    rep.header("5. FORECAST 2011 -> 2041: census-based vs simulated-P open model")
    g20 = np.array([inputs.NATURAL_GROWTH[r] for r in REG])
    Xc = mm.project_open(mm.open_system_matrix(Pc, e, g20), m, x11, 30)
    sim = json.loads((Path(outdir) / "results" / "open_system.json").read_text()) if (Path(outdir) / "results" / "open_system.json").exists() else None
    Xs = np.array(sim["forecast"]["open"]) if sim else None
    yrs = 2011 + np.arange(31)
    rows = []
    for k in (0, 10, 20, 30):
        rows.append([yrs[k]] + [f"{v / 1e6:.2f}" for v in Xc[k]] + [f"{Xc[k].sum() / 1e6:.2f}"] +
                    ([f"{Xs[k].sum() / 1e6:.2f}"] if Xs is not None else []))
    cols = ["Year"] + REG + ["Total (census P)"] + (["Total (simulated P)"] if Xs is not None else [])
    out(pd.DataFrame(rows, columns=cols).to_string(index=False))
    rob_f = {}
    for w in WEIGHTS:
        P_, m_, _ = cd.census_transition_matrix(w)
        e_, _ = mm.calibrate_emigration(P_, g01, m_, x01, x11, 10)
        rob_f[f"w={w:g}"] = float(mm.project_open(mm.open_system_matrix(P_, e_, g20), m_, x11, 30)[-1].sum())
    out("2041 total for allocation weights " + ", ".join(f"{k}: {v / 1e6:.2f} M" for k, v in rob_f.items()))
    check(max(rob_f.values()) / min(rob_f.values()) < 1.02, "2041 forecast changes by under 2% across allocation weights")
    R["forecast"] = {"years": yrs.tolist(), "census_model": L(Xc, 0), "total_2041_by_weight": rob_f}

    outdir = Path(outdir)
    if make_figures:
        figures(outdir / "figures", REG, Pc, m, years, model_nm, survey, yrs, Xc, Xs)
    R["assertions_passed"] = rep.passed; R["design_warnings"] = rep.warnings
    rep.header(f"CHECKS PASSED: {len(rep.passed)}   DESIGN WARNINGS: {len(rep.warnings)}")
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    (outdir / "results" / "census_model.json").write_text(json.dumps(R, indent=2))
    (outdir / "results" / "census_model.txt").write_text("\n".join(rep.log))
    return R


def figures(FIG, REG, Pc, m, years, model_nm, survey, yrs, Xc, Xs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    FIG.mkdir(parents=True, exist_ok=True); C = region_colors(REG)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    def save(name):
        plt.tight_layout(); plt.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.08})
    for ax, M, title in ((axs[0], inputs.P, "Simulated P"), (axs[1], Pc, "Census-based $P_c$ (D-02, 2011)")):
        Z = np.where(np.eye(4, dtype=bool), np.nan, 100 * M)
        im = ax.imshow(Z, cmap="viridis", norm=LogNorm(vmin=0.004, vmax=10))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "stay" if i == j else f"{100 * M[i, j]:.3f}%", ha="center", va="center",
                        color="k" if i == j else "w", fontsize=10)
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(["from " + r for r in REG], rotation=25)
        ax.set_yticklabels(["to " + r for r in REG] if ax is axs[0] else []); ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=axs, label="Moves per year (% of origin population, log scale)", shrink=0.85)
    fig.suptitle("Fig 13: Yearly movement between regions, simulated vs census data", fontsize=13)
    plt.savefig(FIG / "fig13_census_vs_simulated.png", dpi=200, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.6)); xx = np.arange(len(years))
    ax.bar(xx - 0.2, survey, 0.4, color="#999999", edgecolor="k", label="Economic Survey of Delhi 2014-15 (CRS-based)")
    ax.bar(xx + 0.2, model_nm, 0.4, color="#0072B2", edgecolor="k", label="Census-data model (D-02 + SRS)")
    ax.set_xticks(xx); ax.set_xticklabels(years); ax.set_ylabel("Net migration into Delhi (lakh / year)")
    ax.set_title("Fig 14: Independent check of the census-data model"); ax.legend(fontsize=9)
    save("fig14_delhi_net_migration_check.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(yrs, Xc.sum(1) / 1e6, color="#0072B2", lw=2.6, label="Census-data model")
    if Xs is not None:
        ax.plot(yrs, Xs.sum(1) / 1e6, color="#D55E00", lw=2, ls="--", label="Simulated P, alpha = 0.137")
    ax.axhline(Xc[0].sum() / 1e6, color="gray", ls=":", label="Closed model (constant total)")
    ax.set_ylabel("Total population of the four districts (millions)")
    ax.set_title("Fig 15: Forecast 2011-2041, census-data vs simulated model"); ax.legend()
    save("fig15_forecast_comparison.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--outdir", default=str(HERE))
    a = ap.parse_args()
    run(make_figures=not a.no_figures, outdir=a.outdir)
