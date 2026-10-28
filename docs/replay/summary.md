# replay_paper (exploratory)

Mean (sd) over 3 seed(s). best fraction = best measured yield found / best recipe in the pool. found = seeds that measured the best recipe in the pool; round found = first round it was measured, over those seeds (round 1 is random). AUC = mean normalised regret over rounds (0 = best recipe from the start, lower is faster). R2 and Spearman are on the held-out 20% after the last fit.

| dataset | model | found | best fraction | round found | AUC norm. regret | held-out R2 | held-out Spearman | time (s) |
|---|---|---|---|---|---|---|---|---|
| PURE | incumbent | 3/3 | 1.000 (0.000) | 5.667 (2.517) | 0.105 (0.058) | 0.359 (0.028) | 0.625 (0.031) | 1156 |
| PURE | linear | 3/3 | 1.000 (0.000) | 6.667 (1.528) | 0.113 (0.015) | -4.608 (6.136) | 0.601 (0.022) | 3 |
| PURE | random | 1/3 | 0.940 (0.053) | 10.000 (0.000) | 0.198 (0.021) | - | - | 1 |
| Borkowski | incumbent | 3/3 | 1.000 (0.000) | 2.000 (0.000) | 0.025 (0.008) | 0.853 (0.031) | 0.909 (0.034) | 945 |
| Borkowski | linear | 3/3 | 1.000 (0.000) | 2.333 (0.577) | 0.030 (0.013) | 0.804 (0.002) | 0.872 (0.003) | 4 |
| Borkowski | random | 2/3 | 0.998 (0.004) | 3.000 (0.000) | 0.075 (0.034) | - | - | 1 |
| CETCH yield | incumbent | 3/3 | 1.000 (0.000) | 4.333 (2.082) | 0.122 (0.060) | 0.569 (0.016) | 0.797 (0.019) | 2266 |
| CETCH yield | linear | 3/3 | 1.000 (0.000) | 6.000 (1.000) | 0.168 (0.029) | 0.373 (0.094) | 0.703 (0.015) | 3 |
| CETCH yield | random | 1/3 | 0.827 (0.150) | 7.000 (0.000) | 0.294 (0.086) | - | - | 1 |
| CETCH efficiency | incumbent | 3/3 | 1.000 (0.000) | 3.333 (0.577) | 0.082 (0.014) | 0.560 (0.012) | 0.801 (0.008) | 2213 |
| CETCH efficiency | linear | 3/3 | 1.000 (0.000) | 4.333 (2.082) | 0.107 (0.051) | 0.333 (0.085) | 0.729 (0.010) | 20 |
| CETCH efficiency | random | 0/3 | 0.846 (0.038) | - | 0.244 (0.014) | - | - | 1 |
