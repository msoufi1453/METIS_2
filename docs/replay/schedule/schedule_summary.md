# tabpfn_schedule_tune: the winning ramp against METIS's schedule

Mean over 5 seeds. AUC = mean normalised regret over rounds (lower = good recipes sooner). Difference = winner minus METIS's schedule, same seed (mean and standard error).

| dataset | AUC, winner (slope 4.5) | AUC, METIS's schedule | difference | best recipe found (winner / METIS) | round found (winner / METIS) |
|---|---|---|---|---|---|
| PURE | 0.101 | 0.072 | +0.029 (0.013) | 5 / 5 | 5.2 / 4.0 |
| Borkowski | 0.039 | 0.062 | -0.024 (0.008) | 5 / 5 | 3.0 / 4.2 |
| CETCH yield | 0.093 | 0.099 | -0.006 (0.006) | 5 / 5 | 3.4 / 3.6 |
| CETCH efficiency | 0.071 | 0.060 | +0.011 (0.007) | 5 / 5 | 3.4 / 2.8 |
| all four | 0.076 | 0.073 | +0.002 (0.006) | 20 / 20 | 3.8 / 3.6 |
