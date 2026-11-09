"""
All editable model inputs live here (one place, so tests and scripts agree).

Convention (Poole): matrices are COLUMN-stochastic and x_{k+1} = P x_k.
P[i][j] = Prob(person living in region j this year lives in region i next year).

Data status
-----------
* INITIAL_POPULATION  : Census of India 2011, district totals (VERIFIED, see SOURCES).
* POPULATION_2001     : Census of India 2001 on 2011 district boundaries (used for calibration).
* P                   : SIMULATED internal movement probabilities (no public annual
                        district-to-district flow matrix exists).
* NATURAL_GROWTH      : SRS 2020 state-level (birth rate - death rate), applied to each district.
* EMIGRATION          : SIMULATED share of residents who leave NCR each year.
"""
import numpy as np

REGIONS = ["Delhi", "Gurugram", "Noida", "Faridabad"]

# Census 2011 district totals (Gurugram = Gurgaon district, Noida = Gautam Buddha Nagar district).
# BUG FIX: the original script had Gurugram 1,514,085 and Noida 1,674,714 (both wrong).
INITIAL_POPULATION = {
    "Delhi":     16_787_941,   # NCT of Delhi
    "Gurugram":   1_514_432,   # Gurgaon district
    "Noida":      1_648_115,   # Gautam Buddha Nagar district
    "Faridabad":  1_809_733,   # Faridabad district
}

# Census 2001.  Faridabad's 2001 district still contained Palwal (split off in 2008), so its
# value is back-computed from the official comparable decadal growth of 32.54 %.
POPULATION_2001 = {
    "Delhi":     13_850_507,
    "Gurugram":     870_539,
    "Noida":      1_202_030,
    "Faridabad":  round(1_809_733 / 1.3254),
}

# SIMULATED internal transition matrix (closed system). Columns sum to 1.
#                 from: Delhi  Gurugram Noida  Faridabad
P = np.array([[0.950, 0.060, 0.060, 0.070],    # to Delhi
              [0.020, 0.890, 0.030, 0.030],    # to Gurugram
              [0.020, 0.030, 0.890, 0.020],    # to Noida
              [0.010, 0.020, 0.020, 0.880]])   # to Faridabad

# SRS 2020 Bulletin (per 1000): Delhi CBR 14.2 / CDR 3.6, Haryana 19.9 / 6.1, Uttar Pradesh 25.1 / 6.5.
NATURAL_GROWTH = {
    "Delhi":     (14.2 - 3.6) / 1000,
    "Gurugram":  (19.9 - 6.1) / 1000,   # Haryana
    "Noida":     (25.1 - 6.5) / 1000,   # Uttar Pradesh
    "Faridabad": (19.9 - 6.1) / 1000,   # Haryana
}

# SIMULATED: fraction of each region's residents leaving the NCR per year.
EMIGRATION = {"Delhi": 0.006, "Gurugram": 0.008, "Noida": 0.008, "Faridabad": 0.007}

SOURCES = {
    "census_2011_2001": "Census of India 2011 district totals, via census2011.co.in (Gurgaon, Gautam Buddha Nagar, Haryana district list)",
    "srs_2020": "SRS Bulletin 2020 (Vol 55 No 1), Office of the Registrar General of India",
}

COLORS = {"Delhi": "#0072B2", "Gurugram": "#E69F00", "Noida": "#009E73", "Faridabad": "#CC79A7"}  # Okabe-Ito
