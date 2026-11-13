"""
Turn the Census 2011 migration table (D-02) into a REAL transition matrix for the four regions.

D-02 counts migrants by place of last residence and by how long they have lived where they were
enumerated. People who have lived in a district for less than one year are (approximately) those who
moved there in the year before the census, so they give annual flows F[i, j] = people per year moving
from region j to region i. Dividing column j by the population of j gives transition probabilities.

What the table gives directly                        What has to be allocated (assumption)
- Delhi -> Gurugram, Faridabad, Noida                - Gurugram/Faridabad -> Delhi: D-02 only says "from Haryana"
                                                     - Noida -> Delhi: only "from Uttar Pradesh"
                                                     - Gurugram <-> Faridabad: only "other districts of Haryana"
                                                     - Gurugram/Faridabad <-> Noida: only "from Haryana" / "from UP"
State-level flows are split in proportion to population, times `ncr_weight` for the NCR districts
(ncr_weight = 1: no proximity effect; > 1: NCR districts send more than their population share).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
REGIONS = ["Delhi", "Gurugram", "Noida", "Faridabad"]
STATE_OF = {"Delhi": "Delhi", "Gurugram": "Haryana", "Noida": "Uttar Pradesh", "Faridabad": "Haryana"}
D02_STATE = {"Haryana": "Haryana", "Uttar Pradesh": "Uttar Pradesh", "Delhi": "NCT of Delhi"}


def load_flows() -> pd.DataFrame:
    return pd.read_csv(DATA / "census_d02_flows.csv")


def load_population() -> dict:
    p = pd.read_csv(DATA / "census_population_2011.csv")
    return dict(zip(p.region, p.population_2011))


def srs_growth(first: int, last: int) -> dict:
    """Average SRS natural growth (birth rate - death rate) per person per year over [first, last]."""
    s = pd.read_csv(DATA / "srs_2001_2011.csv")
    s = s[(s.year >= first) & (s.year <= last)]
    g = ((s.birth_rate_per_1000 - s.death_rate_per_1000) / 1000).groupby(s.state).mean()
    return {r: float(g[STATE_OF[r]]) for r in REGIONS}


def flow_matrix(ncr_weight: float = 1.0, column: str = "moved_last_year"):
    """column = "moved_last_year" (default) or "moved_1_4_years" (divided by 4: average yearly flow 2007-2010)."""
    return _flow_matrix(ncr_weight, column)


def _flow_matrix(ncr_weight, column):
    """Return (F, outside, notes).

    F[i, j]   people per year moving from region j to region i (i != j); diagonal 0
    outside   people per year arriving in region i from anywhere outside the four regions
    notes     how every entry was obtained ('direct' or the allocation share used)
    """
    fl, pop = load_flows(), load_population()
    st_pop = {"Haryana": pop["Haryana (state)"], "Uttar Pradesh": pop["Uttar Pradesh (state)"]}
    scale = 0.25 if column == "moved_1_4_years" else 1.0
    get = lambda dest, origin: scale * float(fl[(fl.destination == dest) & (fl.last_residence == origin)][column].sum())
    n = len(REGIONS); F = np.zeros((n, n)); notes = {}

    def share(origin_region, dest_region):
        """Fraction of a state-level flow attributed to `origin_region`."""
        s = STATE_OF[origin_region]
        base = st_pop[s] - (pop[dest_region] if STATE_OF[dest_region] == s else 0)
        return min(1.0, ncr_weight * pop[origin_region] / base)

    for i, dest in enumerate(REGIONS):
        for j, orig in enumerate(REGIONS):
            if i == j:
                continue
            if orig == "Delhi":                                   # Delhi is its own state: direct
                F[i, j] = get(dest, "NCT of Delhi"); notes[(dest, orig)] = "direct"
            elif STATE_OF[orig] == STATE_OF[dest]:                # same state, other district
                sh = share(orig, dest)
                F[i, j] = sh * get(dest, "In other districts of the state of enumeration")
                notes[(dest, orig)] = f"{sh:.4f} x other districts of {STATE_OF[dest]}"
            else:                                                 # another state
                sh = share(orig, dest)
                F[i, j] = sh * get(dest, D02_STATE[STATE_OF[orig]])
                notes[(dest, orig)] = f"{sh:.4f} x from {STATE_OF[orig]}"
    outside = np.array([get(d, "Total") - get(d, "Elsewhere in the district of enumeration") - F[i].sum()
                        for i, d in enumerate(REGIONS)])
    return F, outside, notes


def census_transition_matrix(ncr_weight: float = 1.0, column: str = "moved_last_year"):
    """P[i, j] = F[i, j] / x_j off the diagonal, P[j, j] = 1 - (share of j that moved to the other three)."""
    F, outside, notes = flow_matrix(ncr_weight, column)
    pop = load_population()
    x = np.array([pop[r] for r in REGIONS], float)
    P = F / x[None, :]
    P[np.diag_indices_from(P)] = 1 - P.sum(axis=0)
    return P, outside, notes
