# replay_paper (exploratory)

Mean (sd) over 3 seed(s). best fraction = best measured yield found / best recipe in the pool. found = seeds that measured the best recipe in the pool; round found = first round it was measured, over those seeds (round 1 is random). AUC = mean normalised regret over rounds (0 = best recipe from the start, lower is faster). R2, Spearman and 80% coverage (share of held-out recipes inside the model's 80% interval; 0.8 is honest) are on the held-out 20% after the last fit. time = wall-clock seconds per campaign on the machine that ran it.

| dataset | model | found | best fraction | round found | AUC norm. regret | held-out R2 | held-out Spearman | 80% coverage | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| PURE | XGBoost (METIS) | 3/3 | 1.000 (0.000) | 5.667 (2.517) | 0.105 (0.058) | 0.359 (0.028) | 0.625 (0.031) | 0.295 (0.147) | 1156 |
| PURE | Linear | 3/3 | 1.000 (0.000) | 6.667 (1.528) | 0.113 (0.015) | -4.608 (6.136) | 0.601 (0.022) | 0.000 (0.000) | 3 |
| PURE | TabPFN-3.5 | 3/3 | 1.000 (0.000) | 3.667 (1.155) | 0.057 (0.034) | 0.386 (0.041) | 0.693 (0.031) | 0.930 (0.015) | 40 |
| PURE | Random | 1/3 | 0.940 (0.053) | 10.000 (0.000) | 0.198 (0.021) | - | - | - | 1 |
| Borkowski | XGBoost (METIS) | 3/3 | 1.000 (0.000) | 2.000 (0.000) | 0.025 (0.008) | 0.853 (0.031) | 0.909 (0.034) | 0.317 (0.052) | 945 |
| Borkowski | Linear | 3/3 | 1.000 (0.000) | 2.333 (0.577) | 0.030 (0.013) | 0.804 (0.002) | 0.872 (0.003) | 0.000 (0.000) | 4 |
| Borkowski | TabPFN-3.5 | 3/3 | 1.000 (0.000) | 4.333 (0.577) | 0.065 (0.028) | 0.927 (0.002) | 0.960 (0.001) | 0.884 (0.025) | 6 |
| Borkowski | Random | 2/3 | 0.998 (0.004) | 3.000 (0.000) | 0.075 (0.034) | - | - | - | 1 |
| CETCH yield | XGBoost (METIS) | 3/3 | 1.000 (0.000) | 4.333 (2.082) | 0.122 (0.060) | 0.569 (0.016) | 0.797 (0.019) | 0.411 (0.039) | 2266 |
| CETCH yield | Linear | 3/3 | 1.000 (0.000) | 6.000 (1.000) | 0.168 (0.029) | 0.373 (0.094) | 0.703 (0.015) | 0.000 (0.000) | 3 |
| CETCH yield | TabPFN-3.5 | 3/3 | 1.000 (0.000) | 4.000 (1.000) | 0.111 (0.044) | 0.681 (0.008) | 0.859 (0.013) | 0.915 (0.009) | 6 |
| CETCH yield | Random | 1/3 | 0.827 (0.150) | 7.000 (0.000) | 0.294 (0.086) | - | - | - | 1 |
| CETCH efficiency | XGBoost (METIS) | 3/3 | 1.000 (0.000) | 3.333 (0.577) | 0.082 (0.014) | 0.560 (0.012) | 0.801 (0.008) | 0.400 (0.013) | 2213 |
| CETCH efficiency | Linear | 3/3 | 1.000 (0.000) | 4.333 (2.082) | 0.107 (0.051) | 0.333 (0.085) | 0.729 (0.010) | 0.000 (0.000) | 20 |
| CETCH efficiency | TabPFN-3.5 | 3/3 | 1.000 (0.000) | 2.667 (1.155) | 0.054 (0.019) | 0.716 (0.022) | 0.877 (0.010) | 0.913 (0.019) | 8 |
| CETCH efficiency | Random | 0/3 | 0.846 (0.038) | - | 0.244 (0.014) | - | - | - | 1 |

## Paired by seed against the reference

Difference = model minus reference, same seed. Negative round found and AUC = better; positive R2 = better. A seed that never found the best recipe counts as round 11.

| model | reference | dataset | seeds | best found sooner / same / later | mean diff. round found | mean diff. AUC | mean diff. held-out R2 |
|---|---|---|---|---|---|---|---|
| Linear | XGBoost (METIS) | PURE | 3 | 2 / 0 / 1 | +1.00 | +0.008 | -4.967 |
| Linear | XGBoost (METIS) | Borkowski | 3 | 0 / 2 / 1 | +0.33 | +0.006 | -0.049 |
| Linear | XGBoost (METIS) | CETCH yield | 3 | 0 / 2 / 1 | +1.67 | +0.046 | -0.196 |
| Linear | XGBoost (METIS) | CETCH efficiency | 3 | 1 / 0 / 2 | +1.00 | +0.025 | -0.226 |
| Linear | XGBoost (METIS) | all | 12 | 3 / 4 / 5 | +1.00 | +0.021 | -1.360 |
| TabPFN-3.5 | XGBoost (METIS) | PURE | 3 | 2 / 1 / 0 | -2.00 | -0.048 | +0.028 |
| TabPFN-3.5 | XGBoost (METIS) | Borkowski | 3 | 0 / 0 / 3 | +2.33 | +0.041 | +0.074 |
| TabPFN-3.5 | XGBoost (METIS) | CETCH yield | 3 | 2 / 0 / 1 | -0.33 | -0.011 | +0.111 |
| TabPFN-3.5 | XGBoost (METIS) | CETCH efficiency | 3 | 2 / 1 / 0 | -0.67 | -0.028 | +0.156 |
| TabPFN-3.5 | XGBoost (METIS) | all | 12 | 6 / 2 / 4 | -0.17 | -0.012 | +0.092 |
