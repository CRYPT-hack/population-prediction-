"""
All editable model inputs live here (one place, so tests and scripts agree).

Convention (Poole): matrices are COLUMN-stochastic and x_{k+1} = P x_k.
P[i][j] = Prob(person living in region j this year lives in region i next year).
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

# SIMULATED internal transition matrix (closed system). Columns sum to 1.
#                 from: Delhi  Gurugram Noida  Faridabad
P = np.array([[0.950, 0.060, 0.060, 0.070],    # to Delhi
              [0.020, 0.890, 0.030, 0.030],    # to Gurugram
              [0.020, 0.030, 0.890, 0.020],    # to Noida
              [0.010, 0.020, 0.020, 0.880]])   # to Faridabad

SOURCES = {
    "census_2011_2001": "Census of India 2011 district totals, via census2011.co.in (Gurgaon, Gautam Buddha Nagar, Haryana district list)",
}

COLORS = {"Delhi": "#0072B2", "Gurugram": "#E69F00", "Noida": "#009E73", "Faridabad": "#CC79A7"}  # Okabe-Ito
