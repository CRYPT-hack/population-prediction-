# Group 14 – Viva Preparation
**Project:** Predicting Population Movement Using Transition Matrices · MTH2802 · BML Munjal University · Instructor: Dr. Luckshay Batra

> The viva tests **individual** understanding. Every member must be able to explain every part. Remember: transition probabilities are **simulated**; initial populations are **Census 2011 district totals** (verified).

---
## Part 1 – One-page summary (say this aloud)

We studied how people move between four NCR regions: Delhi, Gurugram, Noida and Faridabad. We built a 4×4 **transition matrix P**. The entry P[i][j] is the probability that a person living in region j this year lives in region i next year, so each **column sums to 1** (column-stochastic, as in Poole). The population after k years is x_k = P^k x_0.

The starting populations are Census 2011 district totals. The movement probabilities are **simulated**, because public yearly city-to-city flow matrices do not exist at this level. We designed P so that people mostly stay (retention 0.88–0.95), every entry is positive (so P is **regular**), and all eigenvalues are real and distinct.

We computed projections for 30 years and found the **steady state** three ways: the eigenvector for λ = 1, solving (P − I)π = 0 by Gaussian elimination with sum = 1, and a high power P^200. All agree. The steady state is Delhi 55.5%, Gurugram 17.5%, Noida 16.7%, Faridabad 10.3%.

We diagonalised P = V D V⁻¹. The eigenvalues are 1.0000, 0.8862, 0.8638, 0.8600. The eigenvalue 1 gives the steady state; the others are less than 1 so their parts die out. The size of |λ₂| = 0.8862 controls the speed: 32 years to come within 1% and 51 years within 0.1%; the eigen-decomposition alone predicts 33 and 52.

Different starting points reach the same steady state. Two scenarios (Faridabad infrastructure; Gurugram/Noida job boom) change the steady state a lot, while ±2% random errors change it by at most 1.20 percentage points. Limits: closed system, constant P, memoryless, simulated data.

Finally we opened the system: births and deaths (SRS 2020), people leaving the NCR and people arriving, x_(k+1) = A x_k + m. Fitting the arrivals to Census 2001 and 2011 showed that our simulated P moves about seven times too many people (α = 0.137). The open model gives about 36.8 million people in the four districts by 2041.

---
## Part 2 – 32 likely viva questions with model answers

### A. Matrices and stochastic properties
**Q1. What is a transition matrix? How is yours arranged?**
A square matrix of probabilities. Ours is column-stochastic: P[i][j] = probability of moving from region j (column) to region i (row). Order: Delhi, Gurugram, Noida, Faridabad. We use x_(k+1) = P x_k.

**Q2. Why do the columns sum to 1?**
Column j describes everyone living in region j. Each person must be somewhere next year – the same region or another – so the probabilities of all outcomes add to 1. This also keeps the total population constant.

**Q3. What is the difference between row-stochastic and column-stochastic?**
Row-stochastic has rows summing to 1 and uses x_(k+1) = xᵀP with row vectors. Column-stochastic has columns summing to 1 and uses x_(k+1) = P x with column vectors (Poole's convention, which we follow). One is the transpose of the other.

**Q4. What does the diagonal of P mean?**
Retention: the probability of staying in the same region. Ours are Delhi 0.95, Gurugram 0.89, Noida 0.89, Faridabad 0.88.

**Q5. Explain the hand calculation of x_1 for one region.**
Take the row of P for the region and multiply with x_0 entry by entry, then add. For Gurugram: 0.020×Delhi + 0.890×Gurugram + 0.030×Noida + 0.030×Faridabad = 1,787,338.7 people (Appendix B).

### B. Eigenvalues and diagonalisation
**Q6. Why is λ = 1 always an eigenvalue of a stochastic matrix?**
Columns sum to 1, so 1ᵀP = 1ᵀ (1 = vector of ones), i.e. Pᵀ1 = 1. So 1 is an eigenvalue of Pᵀ. P and Pᵀ have the same characteristic polynomial, since det(P − λI) = det(Pᵀ − λI), so 1 is an eigenvalue of P too.

**Q7. Why do the other eigenvectors have entries summing to zero?**
If Pv = λv with λ ≠ 1, then 1ᵀPv = 1ᵀv, so λ(1ᵀv) = 1ᵀv, hence (1 − λ)(1ᵀv) = 0, so 1ᵀv = 0. They represent shifts of population from one region to another, not population itself.

**Q8. How did you compute P^n using diagonalisation?**
P = V D V⁻¹, so P^n = V D^n V⁻¹; D^n just raises each eigenvalue to the power n. We checked P^10 this way against np.linalg.matrix_power: the maximum difference is 2.0e-15.

**Q9. Why can P be diagonalised?**
P is 4×4 and has 4 distinct real eigenvalues, so it has 4 linearly independent eigenvectors. (Distinct eigenvalues are sufficient, not necessary.)

**Q10. What did the decomposition of x_0 show?**
x_0 = c₁v₁ + c₂v₂ + c₃v₃ + c₄v₄ with c = V⁻¹x_0. Here c₁ = 1.00, c₂ = 0.284, c₃ = -0.160, c₄ = -0.120. After k years x_k = c₁v₁ + c₂λ₂^k v₂ + …; only the λ = 1 term survives.

### C. Convergence and steady state
**Q11. What is a steady state?**
A distribution π with Pπ = π and entries summing to 1. Once reached, shares do not change: inflow equals outflow for every region.

**Q12. How did you find it, and why three ways?**
(a) eigenvector for λ = 1 normalised to sum 1; (b) solve (P − I)π = 0 plus Σπ = 1 by Gaussian elimination; (c) a high power P^200, whose columns all equal π. Three independent methods agreeing within 1e-8 is strong verification.

**Q13. Why does the population converge?**
P is regular, so λ = 1 is simple and all other eigenvalues have |λ| < 1. Their powers shrink to 0, leaving only the steady state.

**Q14. What does |λ₂| control?**
The speed of convergence: the distance behaves like C|λ₂|^k. Ours is 0.8862, so it takes 32 years to get within 1%. Theory with C = ‖c₂v₂‖₁ = 0.489 gives 33; it is one year high because the faster modes partly cancel the slowest one. Closer to 1 means slower.

**Q15. What is a regular matrix? Is P regular?**
Some power of P has all entries strictly positive. Our P itself has all entries positive, so P¹ > 0 and k = 1.

**Q16. What if P is not regular?**
The steady-state theorem may fail. Example 1: an absorbing state (retention 1): everyone ends there (limit ≈ 0.000, 0.000, 1.000). Example 2: a periodic matrix [[0,1],[1,0]] has eigenvalues 1 and −1 and flips forever with no limit.

### D. Modelling choices and limitations
**Q17. Why did you use simulated data?**
Public annual city-to-city migration matrices do not exist at this granularity. We designed plausible probabilities with justification per column and stated this openly. The method is what the project demonstrates, not a real forecast.

**Q18. What would real data change?**
Probabilities would likely have retention closer to 1, asymmetric flows and a time dependence; we could add outside regions, births and deaths. The numbers and steady state would change, not the method.

**Q19. What are the limitations?**
Closed system in Sections 4–6 (the open model of Section 7 relaxes it), constant P, memoryless, all people identical, simulated probabilities and emigration, old (2011) census, and an exact (not out-of-sample) census fit.

**Q20. Delhi's population falls in your model. Is that a prediction?**
No. It is a consequence of the closed-system assumption and simulated probabilities. In the open model (births, deaths, arrivals) Delhi grows from 16.8 M to about 26.1 M by 2041.

### E. Code and results
**Q21. What does each plot show?**
Fig 1 heat map of P. Fig 2 flow diagram (thickness ∝ probability). Fig 3 shares over time. Fig 4 initial vs steady shares. Fig 5 distance to steady state on a log scale (a straight line: geometric decay at rate |λ₂|). Fig 6 four starting points reaching the same limit. Fig 7 scenario steady states. Fig 8 sensitivity box plots. Fig 9 eigenvalues in the complex plane with the unit circle. Fig 10 census calibration and long-run shares. Fig 11 open vs closed forecast. Fig 12 where the 2041 population comes from.

**Q22. What did the scenarios change and why?**
A: Faridabad retention 0.88 → 0.92 and more inflow; its steady share goes 10.3% → 20.9%. B: more movers from Delhi to Gurugram and Noida; their shares go to 28.5% and 27.5%. |λ₂| rises in both (0.9015 and 0.8950 vs 0.8862), so convergence is slightly slower.

**Q23. What does the sensitivity analysis tell us?**
We multiplied entries by random factors in [0.98, 1.02] 1000 times and renormalised. Each share varied by a standard deviation of at most 0.39 percentage points: the result is robust to small errors, but not to the large structural changes of the scenarios.

**Q24. How did you verify the code?**
Two kinds of check: mathematical facts must hold (columns sum to 1, regularity up to Wielandt's bound, three steady-state methods agree to 1e-8, V D V⁻¹ rebuilds P with error 6e-16, P^10 matches, totals conserved); design choices (retention range, shares above 5%) only warn. 43 checks pass with 0 warnings, and 27 automated tests (pytest) run on every GitHub push.

**Q25. Explain the 2×2 example.**
T = [[0.9, 0.2],[0.1, 0.8]] has λ = 1 and 0.7 (trace 1.7, det 0.7). v₁ = (2,1) ⇒ steady state (2/3, 1/3). v₂ = (1,−1). T^n → [[2/3,2/3],[1/3,1/3]] since 0.7^n → 0.

### F. Open system and bug fixes (new)
**Q26. How does the open model differ from the closed one?**
x_(k+1) = A x_k + m. A = diag(1+g) P diag(1−e): first a fraction e_j leaves the NCR, then people move by P, then natural growth g_i (SRS 2020 birth minus death rate) applies; m is the yearly number of arrivals. Columns of A do not sum to 1, so the total can grow.

**Q27. What are homogeneous coordinates and why use them?**
Append a 1 to the state: (x_(k+1), 1) = B (x_k, 1) with B = [[A, m],[0, 1]]. The affine step becomes one matrix multiplication, so matrix powers still work. B is block-triangular, so its eigenvalues are those of A plus 1.

**Q28. How did you fit the model to the census?**
x_2011 = A^10 x_2001 + (I + A + … + A^9) m. Both censuses are known, so m solves a 4×4 linear system, solved with our Gaussian elimination.

**Q29. What did the census fit show?**
At full strength P needs negative arrivals into Gurugram (−210,414/yr) and Noida (−192,075/yr): impossible, so P moves too many people (2% of Delhi = 335,000 people a year into Gurugram). With P_α = I + α(P − I) the largest valid α is 0.137. Same steady state, but |λ₂| becomes 0.984, about 7.7 times slower.

**Q30. Why does P_α keep the same steady state?**
P_α π = π + α(Pπ − π) = π. Each eigenvalue maps to 1 + α(λ − 1), so λ = 1 stays and the others move towards 1.

**Q31. What happens in the long run in the open model?**
ρ(A) = 1.00626 > 1, so the total grows 0.63% a year and the shares approach the Perron vector of A: Delhi 49.8%, Gurugram 17.4%, Noida 21.8%, Faridabad 11.0%. Forecast: 21.76 M (2011) → 36.78 M (2041); 11.2 M of that from arrivals after 2011.

**Q32. Which bugs did you fix in the original code?**
Wrong census figures for Gurugram and Noida; eigenvalues sorted by real part instead of modulus (wrong |λ₂| for a negative eigenvalue); "years to converge" returned 0 when a threshold was never reached; the "theory" constant was fitted from the same curve it predicted; no zero-pivot check in Gaussian elimination; design choices written as hard assertions so any edit crashed the script.

---
## Part 3 – Numbers to remember

**Final P** (rows: to; columns: from Delhi, Gurugram, Noida, Faridabad)

| to \ from | Delhi | Gurugram | Noida | Faridabad |
|---|---|---|---|---|
| Delhi | 0.950 | 0.060 | 0.060 | 0.070 |
| Gurugram | 0.020 | 0.890 | 0.030 | 0.030 |
| Noida | 0.020 | 0.030 | 0.890 | 0.020 |
| Faridabad | 0.010 | 0.020 | 0.020 | 0.880 |

**Eigenvalues:** 1.000000, 0.886180, 0.863820, 0.860000  (all real, distinct)
**|λ₂|:** 0.886180  ·  half-life of slowest mode ≈ 5.7 years
**Steady state:** Delhi 55.48%, Gurugram 17.47%, Noida 16.73%, Faridabad 10.32%
**Initial shares (Census 2011):** Delhi 77.15%, Gurugram 6.96%, Noida 7.57%, Faridabad 8.32%  ·  total 21,760,221
**Convergence:** within 1% after 32 years; within 0.1% after 51 years (theory 33 / 52).
**Scenario steady states:** A (Delhi/Gur/Noi/Far) 46.6% / 16.6% / 15.9% / 20.9%; B 34.2% / 28.5% / 27.5% / 9.8%.
**Sensitivity:** max deviation 1.20 percentage points.
**Open model:** α = 0.137 · arrivals 346,381/yr · ρ(A) = 1.00626 (+0.63%/yr) · 36.78 M in 2041 · long-run shares 49.8 / 17.4 / 21.8 / 11.0%.

---
## Part 4 – "If you are asked to explain section X"
1. **Introduction:** We model yearly movement between four NCR regions. The brief asks for a transition matrix, matrix powers and long-term behaviour. We use column-stochastic P with x_(k+1) = P x_k.
2. **Mathematical background:** Columns sum to 1 so total population is conserved and 1 is always an eigenvalue. Regular means some power is positive, giving a unique steady state. P^n = V D^n V⁻¹ and |λ₂| controls speed.
3. **Data and assumptions:** Populations are Census 2011 district totals; the open model adds Census 2001, SRS 2020 growth and simulated emigration. P is simulated and each column is justified. Assumptions: closed, constant P, memoryless, identical people.
4. **Model construction:** P is shown with heat map and flow diagram. The program checks column sums, positivity and regularity (must hold) and warns if design choices such as retention in [0.88, 0.96] fail.
5. **Results:** Projection for 30 years, P^k columns converge, steady state by three methods, diagonalisation checks, and convergence speed matching |λ₂|^k.
6. **Scenario and sensitivity:** Scenario A raises Faridabad's retention and inflow; B raises flows to Gurugram/Noida. Both change steady states and |λ₂|. 1000 ±2% perturbations show robustness to small errors.
7. **Open-system extension:** x_(k+1) = A x_k + m with A = diag(1+g) P diag(1−e). Fitting arrivals to Census 2001→2011 needs negative arrivals at full P, so mobility is scaled to α = 0.137. Forecast 36.8 M by 2041; long-run growth ρ(A) − 1 = 0.63%/yr.
8. **Discussion:** Delhi's share falls to 55.5% because of simulated flows in a closed system, not a real forecast. Limitations: closed, constant, memoryless, simulated. Real data would add outside regions and time-varying P.
9. **Conclusion:** The model converges to a unique steady state independent of the start; the λ = 1 eigenvector survives and |λ₂| sets the speed; conclusions are about the model, not real migration.
