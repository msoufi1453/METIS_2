# tabpfn_schedule_tune: tuning (auc_norm_regret, lower is better)

Winner: **schedule_exploit_ramp_slope4.5** (4.5). Mean over 4 problem(s) of the per-problem mean over 5 seed(s). Paired difference to the winner: mean (standard error) over problem x seed pairs.

Note: the winner is at the edge of the candidate grid, so a value beyond it might do better.

| candidate | role | Borkowski | CETCH efficiency | CETCH yield | PURE | mean | diff. to winner | rank |
|---|---|---|---|---|---|---|---|---|
| 0.8 | candidate | 0.064 | 0.106 | 0.131 | 0.100 | 0.100 | +0.024 (0.006) | 10 |
| 0.9 | candidate | 0.060 | 0.119 | 0.131 | 0.096 | 0.101 | +0.026 (0.008) | 11 |
| 1 | candidate | 0.061 | 0.114 | 0.120 | 0.089 | 0.096 | +0.020 (0.008) | 7 |
| 1.125 | candidate | 0.059 | 0.129 | 0.107 | 0.100 | 0.099 | +0.023 (0.009) | 9 |
| 1.3 | candidate | 0.058 | 0.125 | 0.101 | 0.096 | 0.095 | +0.019 (0.008) | 6 |
| 1.5 | candidate | 0.059 | 0.117 | 0.106 | 0.103 | 0.096 | +0.020 (0.007) | 8 |
| 1.8 | candidate | 0.059 | 0.101 | 0.101 | 0.113 | 0.093 | +0.017 (0.007) | 5 |
| 2.25 | candidate | 0.051 | 0.084 | 0.109 | 0.086 | 0.083 | +0.007 (0.007) | 3 |
| 3 | candidate | 0.046 | 0.073 | 0.097 | 0.116 | 0.083 | +0.007 (0.003) | 4 |
| 4.5 (winner) | candidate | 0.039 | 0.071 | 0.093 | 0.101 | 0.076 | +0.000 (0.000) | 2 |
| tabpfn_v1_schedule | reference | 0.062 | 0.060 | 0.099 | 0.072 | 0.073 | -0.002 (0.006) | 1 |
