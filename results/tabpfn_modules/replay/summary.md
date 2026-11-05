# tabpfn_modules (exploratory)

Mean (standard error) over 3 seeds. best fraction = best measured value found / best in the dataset; found = seeds that measured the best recipe; AUC = mean normalised regret over rounds (lower = good recipes sooner). Spread and gap: mean and smallest distance between recipes in a batch, in units of the dataset's typical spacing, averaged over rounds. Climb moves: median [quartiles] over all climbs, capped at 20.

| dataset | arm | found | best fraction | AUC norm. regret | batch spread | smallest gap | climb moves | climbs capped | climbed recipes picked |
|---|---|---|---|---|---|---|---|---|---|
| Borkowski | TabPFN, no module | 0/3 | 0.904 (0.044) | 0.297 (0.014) | 3.73 | 1.09 | - | - | - |
| Borkowski | Penalty, low | 3/3 | 1.000 (0.000) | 0.234 (0.021) | 4.83 | 2.62 | - | - | - |
| Borkowski | Penalty, high | 1/3 | 0.948 (0.048) | 0.236 (0.041) | 7.03 | 4.47 | - | - | - |
| Borkowski | Fantasy batch | 3/3 | 1.000 (0.000) | 0.200 (0.020) | 4.16 | 1.27 | - | - | - |
| Cell growth | TabPFN, no module | 0/3 | 0.760 (0.076) | 0.431 (0.108) | 2.05 | 1.00 | - | - | - |
| Cell growth | Penalty, low | 0/3 | 0.907 (0.004) | 0.257 (0.056) | 3.76 | 2.19 | - | - | - |
| Cell growth | Penalty, high | 0/3 | 0.857 (0.030) | 0.317 (0.084) | 4.02 | 2.89 | - | - | - |
| Cell growth | Fantasy batch | 1/3 | 0.882 (0.089) | 0.262 (0.116) | 2.54 | 1.08 | - | - | - |
| Cell growth | Hill-climbing | 0/3 | 0.819 (0.039) | 0.376 (0.071) | 1.81 | 1.00 | 1 [1, 1] | 0/270 | 55 |
| Synthetic (12 factors) | TabPFN, no module | 0/3 | 0.876 (0.045) | 0.306 (0.032) | 2.25 | 1.18 | - | - | - |
| Synthetic (12 factors) | Penalty, low | 0/3 | 0.849 (0.062) | 0.372 (0.055) | 3.02 | 2.16 | - | - | - |
| Synthetic (12 factors) | Penalty, high | 0/3 | 0.773 (0.034) | 0.417 (0.018) | 3.90 | 2.74 | - | - | - |
| Synthetic (12 factors) | Fantasy batch | 0/3 | 0.871 (0.048) | 0.344 (0.031) | 2.27 | 1.24 | - | - | - |
| Synthetic (12 factors) | Hill-climbing | 1/3 | 0.966 (0.031) | 0.245 (0.025) | 1.73 | 0.60 | 6 [4, 9] | 0/270 | 237 |

## Paired by seed against the reference

Difference = model minus reference, same seed. Negative round found and AUC = better; positive R2 = better. A seed that never found the best recipe counts as round 11.

| model | reference | dataset | seeds | best found sooner / same / later | mean diff. round found | mean diff. AUC | mean diff. held-out R2 |
|---|---|---|---|---|---|---|---|
| Penalty, low | TabPFN, no module | Borkowski | 3 | 3 / 0 / 0 | -3.33 | -0.063 | +0.015 |
| Penalty, low | TabPFN, no module | Cell growth | 3 | 0 / 3 / 0 | +0.00 | -0.173 | +0.237 |
| Penalty, low | TabPFN, no module | Synthetic (12 factors) | 3 | 0 / 3 / 0 | +0.00 | +0.066 | +0.006 |
| Penalty, low | TabPFN, no module | all | 9 | 3 / 6 / 0 | -1.11 | -0.057 | +0.086 |
| Penalty, high | TabPFN, no module | Borkowski | 3 | 1 / 2 / 0 | -1.67 | -0.061 | +0.072 |
| Penalty, high | TabPFN, no module | Cell growth | 3 | 0 / 3 / 0 | +0.00 | -0.114 | +0.265 |
| Penalty, high | TabPFN, no module | Synthetic (12 factors) | 3 | 0 / 3 / 0 | +0.00 | +0.111 | +0.050 |
| Penalty, high | TabPFN, no module | all | 9 | 1 / 8 / 0 | -0.56 | -0.021 | +0.129 |
| Fantasy batch | TabPFN, no module | Borkowski | 3 | 3 / 0 / 0 | -3.33 | -0.097 | +0.025 |
| Fantasy batch | TabPFN, no module | Cell growth | 3 | 1 / 2 / 0 | -2.33 | -0.168 | +0.121 |
| Fantasy batch | TabPFN, no module | Synthetic (12 factors) | 3 | 0 / 3 / 0 | +0.00 | +0.038 | +0.038 |
| Fantasy batch | TabPFN, no module | all | 9 | 4 / 5 / 0 | -1.89 | -0.076 | +0.062 |
| Hill-climbing | TabPFN, no module | Cell growth | 3 | 0 / 3 / 0 | +0.00 | -0.055 | -0.021 |
| Hill-climbing | TabPFN, no module | Synthetic (12 factors) | 3 | 1 / 2 / 0 | -0.67 | -0.062 | -0.084 |
| Hill-climbing | TabPFN, no module | all | 6 | 1 / 5 / 0 | -0.33 | -0.058 | -0.052 |

## Each module against the control (TabPFN, no module), seed by seed

AUC regret of the module minus that of TabPFN, no module in the same seed (negative = better), and the number of seeds in which the module was better.

| dataset | module | difference per seed | mean | better in |
|---|---|---|---|---|
| Borkowski | Penalty, low | -0.055, -0.031, -0.103 | -0.063 | 3/3 |
| Borkowski | Penalty, high | -0.123, +0.045, -0.104 | -0.061 | 2/3 |
| Borkowski | Fantasy batch | -0.127, -0.035, -0.129 | -0.097 | 3/3 |
| Cell growth | Penalty, low | -0.021, -0.371, -0.128 | -0.173 | 3/3 |
| Cell growth | Penalty, high | -0.001, -0.328, -0.012 | -0.114 | 3/3 |
| Cell growth | Fantasy batch | -0.121, -0.089, -0.295 | -0.168 | 3/3 |
| Cell growth | Hill-climbing | +0.020, -0.165, -0.019 | -0.055 | 2/3 |
| Synthetic (12 factors) | Penalty, low | +0.093, +0.091, +0.014 | +0.066 | 0/3 |
| Synthetic (12 factors) | Penalty, high | +0.111, +0.027, +0.196 | +0.111 | 0/3 |
| Synthetic (12 factors) | Fantasy batch | +0.062, +0.023, +0.028 | +0.038 | 0/3 |
| Synthetic (12 factors) | Hill-climbing | -0.072, -0.071, -0.041 | -0.062 | 3/3 |

## Held-out accuracy after the last round

R2 and Spearman rank correlation on the held-out recipes (synthetic: 1,000 random ones), and the share inside the model's 80% interval. Mean over seeds.

| dataset | arm | R2 | Spearman | 80% coverage |
|---|---|---|---|---|
| Borkowski | TabPFN, no module | 0.824 | 0.904 | 0.983 |
| Borkowski | Penalty, low | 0.839 | 0.915 | 0.977 |
| Borkowski | Penalty, high | 0.896 | 0.944 | 0.922 |
| Borkowski | Fantasy batch | 0.849 | 0.926 | 0.982 |
| Cell growth | TabPFN, no module | -0.026 | 0.290 | 0.885 |
| Cell growth | Penalty, low | 0.211 | 0.500 | 0.859 |
| Cell growth | Penalty, high | 0.239 | 0.488 | 0.837 |
| Cell growth | Fantasy batch | 0.095 | 0.341 | 0.893 |
| Cell growth | Hill-climbing | -0.046 | 0.341 | 0.871 |
| Synthetic (12 factors) | TabPFN, no module | 0.881 | 0.937 | 0.964 |
| Synthetic (12 factors) | Penalty, low | 0.886 | 0.949 | 0.949 |
| Synthetic (12 factors) | Penalty, high | 0.931 | 0.981 | 0.923 |
| Synthetic (12 factors) | Fantasy batch | 0.919 | 0.955 | 0.967 |
| Synthetic (12 factors) | Hill-climbing | 0.796 | 0.909 | 0.953 |

## Cell growth: what a batch covers

Distinct media and strains among the 10 recipes of a batch (mean over rounds 2-10 and seeds).

| arm | media | strains |
|---|---|---|
| TabPFN, no module | 3.1 | 9.8 |
| Penalty, low | 10.0 | 9.1 |
| Penalty, high | 10.0 | 9.9 |
| Fantasy batch | 3.4 | 9.3 |
| Hill-climbing | 2.9 | 9.4 |
