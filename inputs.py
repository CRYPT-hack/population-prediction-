"""
All editable model inputs live here (one place, so tests and scripts agree).

Convention (Poole): matrices are COLUMN-stochastic and x_{k+1} = P x_k.
P[i][j] = Prob(person living in region j this year lives in region i next year).
"""
import numpy as np

REGIONS = ["Delhi", "Gurugram", "Noida", "Faridabad"]

# Approximate Census 2011 district totals (to be verified).
INITIAL_POPULATION = {
    "Delhi":     16_787_941,
    "Gurugram":   1_514_085,
    "Noida":      1_674_714,
    "Faridabad":  1_809_733,
}

# SIMULATED internal transition matrix (closed system). Columns sum to 1.
#                 from: Delhi  Gurugram Noida  Faridabad
P = np.array([[0.950, 0.060, 0.060, 0.070],    # to Delhi
              [0.020, 0.890, 0.030, 0.030],    # to Gurugram
              [0.020, 0.030, 0.890, 0.020],    # to Noida
              [0.010, 0.020, 0.020, 0.880]])   # to Faridabad

COLORS = {"Delhi": "#0072B2", "Gurugram": "#E69F00", "Noida": "#009E73", "Faridabad": "#CC79A7"}  # Okabe-Ito
