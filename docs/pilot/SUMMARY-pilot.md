# METIS benchmark: pilot results

**Pilot runs check the machinery.** They use a few problems and seeds and a cheaper incumbent (15 hyperparameter draws instead of 200), and the decision step ignores significance so Phase 3 has something to run. None of this is evidence for or against an intervention; see docs/PREREGISTRATION.md.

## fig1b_check

8 campaigns: 1 problems x 2 arms, 4 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm           |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:--------------|--------------------:|-------------:|------:|----------------:|----------:|
| metis_v1      |               0.028 |        0.274 | 0.067 |               0 |     1.23  |
| random_search |               0.113 |        0.191 | 0.206 |               0 |     0.017 |

Family descriptive (paired differences, negative = arm better):

| arm      | reference     |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:---------|:--------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| metis_v1 | random_search |            1 |         4 |      -0.085 |   -0.159 |    -0.037 |              1 |             0 |         1 |        1 | False         |

Final normalised regret by landscape:

| landscape   |   metis_v1 |   random_search |
|:------------|-----------:|----------------:|
| real        |      0.028 |           0.113 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm           |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:--------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| metis_v1      |     0.478 |           0.762 |        0.702 |        0.912 |       0.113 |       0.959 |            0.823 |
| random_search |   nan     |         nan     |      nan     |      nan     |     nan     |     nan     |          nan     |

![regret curves](../../results/fig1b_check-pilot/report/regret_curves.png)

## phase2_init

160 campaigns: 10 problems x 4 arms, 4 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm           |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:--------------|--------------------:|-------------:|------:|----------------:|----------:|
| incumbent     |               0.215 |        0.324 | 0.269 |           0.275 |     0.969 |
| init_lhs      |               0.215 |        0.304 | 0.26  |           0.275 |     0.975 |
| init_sobol    |               0.213 |        0.317 | 0.256 |           0.275 |     0.969 |
| random_search |               0.327 |        0.392 | 0.381 |           0.025 |     0.017 |

Family F1 (paired differences, negative = arm better):

| arm        | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:-----------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| init_sobol | incumbent   |           10 |        40 |      -0.002 |   -0.006 |         0 |            0.4 |           0.1 |     0.125 |     0.25 | False         |
| init_lhs   | incumbent   |           10 |        40 |      -0.001 |   -0.003 |         0 |            0.3 |           0.1 |     0.25  |     0.25 | False         |

Final normalised regret by landscape:

| landscape   |   incumbent |   init_lhs |   init_sobol |   random_search |
|:------------|------------:|-----------:|-------------:|----------------:|
| interaction |       0.011 |      0.01  |        0.01  |           0.141 |
| multipeak   |       0.476 |      0.469 |        0.468 |           0.559 |
| needle      |       1     |      1     |        1     |           1     |
| real        |       0.064 |      0.064 |        0.064 |           0.296 |
| smooth      |       0.132 |      0.132 |        0.132 |           0.275 |
| threshold   |       0.11  |      0.109 |        0.105 |           0.194 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm           |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:--------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| incumbent     |     0.398 |           0.79  |        0.27  |        0.485 |       0.35  |       2.026 |            0.572 |
| init_lhs      |     0.364 |           0.791 |        0.27  |        0.497 |       0.346 |       2.101 |            0.582 |
| init_sobol    |     0.296 |           0.771 |        0.237 |        0.47  |       0.366 |       2.11  |            0.562 |
| random_search |   nan     |         nan     |      nan     |      nan     |     nan     |     nan     |          nan     |

![regret curves](../../results/phase2_init-pilot/report/regret_curves.png)

## phase2_import

80 campaigns: 10 problems x 2 arms, 4 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm         |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:------------|--------------------:|-------------:|------:|----------------:|----------:|
| import_fill |               0.212 |        0.328 | 0.252 |             0.3 |     0.919 |
| import_v1   |               0.21  |        0.338 | 0.252 |             0.3 |     1.016 |

Family F1b (paired differences, negative = arm better):

| arm         | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:------------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| import_fill | import_v1   |           10 |        40 |       0.001 |        0 |     0.004 |              0 |           0.2 |       0.5 |      0.5 | False         |

Final normalised regret by landscape:

| landscape   |   import_fill |   import_v1 |
|:------------|--------------:|------------:|
| interaction |         0.007 |       0.007 |
| multipeak   |         0.484 |       0.484 |
| needle      |         1     |       1     |
| real        |         0.04  |       0.04  |
| smooth      |         0.129 |       0.128 |
| threshold   |         0.106 |       0.103 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm         |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| import_fill |     0.532 |           0.78  |        0.307 |        0.55  |       0.324 |       2.235 |            0.616 |
| import_v1   |     0.206 |           0.733 |        0.193 |        0.353 |       0.387 |       2.336 |            0.572 |

![regret curves](../../results/phase2_import-pilot/report/regret_curves.png)

## phase2_surrogates

60 campaigns: 10 problems x 3 arms, 2 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm         |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:------------|--------------------:|-------------:|------:|----------------:|----------:|
| incumbent   |               0.239 |        0.323 | 0.289 |            0.25 |     1.259 |
| surr_gp     |               0.235 |        0.336 | 0.281 |            0.25 |     0.971 |
| surr_tabpfn |               0.236 |        0.349 | 0.286 |            0.25 |     7.321 |

Family F2 (paired differences, negative = arm better):

| arm         | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:------------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| surr_gp     | incumbent   |           10 |        20 |      -0.004 |   -0.016 |     0     |            0.2 |             0 |       0.5 |        1 | False         |
| surr_tabpfn | incumbent   |           10 |        20 |      -0.003 |   -0.015 |     0.002 |            0.2 |             0 |       0.5 |        1 | False         |

Final normalised regret by landscape:

| landscape   |   incumbent |   surr_gp |   surr_tabpfn |
|:------------|------------:|----------:|--------------:|
| interaction |       0.015 |     0.015 |         0.015 |
| multipeak   |       0.551 |     0.512 |         0.521 |
| needle      |       1     |     1     |         1     |
| real        |       0.147 |     0.147 |         0.147 |
| smooth      |       0.149 |     0.147 |         0.147 |
| threshold   |       0.121 |     0.121 |         0.121 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm         |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| incumbent   |     0.416 |           0.798 |        0.298 |        0.502 |       0.334 |       2.035 |            0.644 |
| surr_gp     |     0.845 |           0.865 |        0.878 |        0.96  |       0.122 |       0.852 |            0.736 |
| surr_tabpfn |     0.873 |           0.923 |        0.925 |        0.99  |       0.126 |       0.692 |            0.781 |

![regret curves](../../results/phase2_surrogates-pilot/report/regret_curves.png)

## phase2_batch

280 campaigns: 10 problems x 7 arms, 4 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm           |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:--------------|--------------------:|-------------:|------:|----------------:|----------:|
| batch_fantasy |               0.214 |        0.306 | 0.273 |           0.275 |     5.31  |
| batch_lp      |               0.244 |        0.319 | 0.317 |           0.275 |     0.827 |
| incumbent     |               0.215 |        0.324 | 0.269 |           0.275 |     0.8   |
| noise_xgb     |               0.217 |        0.306 | 0.271 |           0.275 |     0.827 |
| pool_local    |               0.215 |        0.289 | 0.312 |           0.175 |     0.883 |
| surr_gp       |               0.215 |        0.315 | 0.268 |           0.275 |     0.647 |
| surr_gp_noise |               0.214 |        0.309 | 0.266 |           0.275 |     0.641 |

Family F3 (paired differences, negative = arm better):

| arm           | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:--------------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| batch_fantasy | incumbent   |           10 |        40 |      -0.001 |   -0.004 |     0.002 |            0.2 |           0.2 |     0.625 |    1     | False         |
| batch_lp      | incumbent   |           10 |        40 |       0.029 |    0.008 |     0.056 |            0   |           0.6 |     0.031 |    0.094 | False         |
| noise_xgb     | incumbent   |           10 |        40 |       0.002 |   -0.001 |     0.008 |            0   |           0.1 |     1     |    1     | False         |

Family F3b (paired differences, negative = arm better):

| arm           | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:--------------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| surr_gp_noise | surr_gp     |           10 |        40 |      -0.002 |   -0.006 |         0 |            0.2 |           0.1 |       0.5 |      0.5 | False         |

Family F3c (paired differences, negative = arm better):

| arm        | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:-----------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| pool_local | incumbent   |           10 |        40 |          -0 |   -0.035 |     0.046 |            0.5 |           0.4 |     0.734 |    0.734 | False         |

Final normalised regret by landscape:

| landscape   |   batch_fantasy |   batch_lp |   incumbent |   noise_xgb |   pool_local |   surr_gp |   surr_gp_noise |
|:------------|----------------:|-----------:|------------:|------------:|-------------:|----------:|----------------:|
| interaction |           0.011 |      0.055 |       0.011 |       0.011 |        0.095 |     0.01  |           0.01  |
| multipeak   |           0.468 |      0.523 |       0.476 |       0.492 |        0.442 |     0.468 |           0.468 |
| needle      |           1     |      1     |       1     |       1     |        1     |     1     |           1     |
| real        |           0.064 |      0.119 |       0.064 |       0.064 |        0.019 |     0.08  |           0.064 |
| smooth      |           0.135 |      0.16  |       0.132 |       0.132 |        0.103 |     0.132 |           0.132 |
| threshold   |           0.108 |      0.123 |       0.11  |       0.11  |        0.097 |     0.107 |           0.107 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm           |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:--------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| batch_fantasy |     0.419 |           0.791 |        0.273 |        0.493 |       0.348 |       2.103 |            0.621 |
| batch_lp      |     0.652 |           0.86  |        0.287 |        0.48  |       0.336 |       2.365 |            0.875 |
| incumbent     |     0.398 |           0.79  |        0.27  |        0.485 |       0.35  |       2.026 |            0.572 |
| noise_xgb     |     0.477 |           0.794 |        0.268 |        0.498 |       0.35  |       2.256 |            0.566 |
| pool_local    |     0.231 |           0.663 |        0.27  |        0.48  |       0.349 |       2.516 |            0.213 |
| surr_gp       |     0.841 |           0.875 |        0.853 |        0.96  |       0.123 |       0.905 |            0.709 |
| surr_gp_noise |     0.875 |           0.875 |        0.833 |        0.947 |       0.107 |       0.979 |            0.779 |

![regret curves](../../results/phase2_batch-pilot/report/regret_curves.png)

## phase3_schedule

120 campaigns: 10 problems x 3 arms, 4 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm                     |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:------------------------|--------------------:|-------------:|------:|----------------:|----------:|
| incumbent_check         |               0.215 |        0.289 | 0.312 |           0.175 |     1.187 |
| tuned_schedule          |               0.149 |        0.181 | 0.218 |           0.425 |     1.512 |
| winners_legacy_schedule |               0.145 |        0.174 | 0.216 |           0.475 |     1.511 |

Family F4 (paired differences, negative = arm better):

| arm            | reference               |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:---------------|:------------------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| tuned_schedule | winners_legacy_schedule |           10 |        40 |       0.004 |   -0.018 |     0.026 |            0.2 |           0.4 |     0.312 |    0.312 | False         |

Final normalised regret by landscape:

| landscape   |   incumbent_check |   tuned_schedule |   winners_legacy_schedule |
|:------------|------------------:|-----------------:|--------------------------:|
| interaction |             0.095 |            0.002 |                     0     |
| multipeak   |             0.442 |            0.28  |                     0.273 |
| needle      |             1     |            1     |                     1     |
| real        |             0.019 |            0.004 |                     0.011 |
| smooth      |             0.103 |            0.015 |                     0.003 |
| threshold   |             0.097 |            0.056 |                     0.053 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm                     |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:------------------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| incumbent_check         |     0.231 |           0.663 |        0.27  |        0.48  |       0.349 |       2.516 |            0.217 |
| tuned_schedule          |     0.77  |           0.798 |        0.842 |        0.95  |       0.098 |       0.938 |            0.311 |
| winners_legacy_schedule |     0.782 |           0.802 |        0.85  |        0.953 |       0.106 |       0.91  |            0.306 |

![regret curves](../../results/phase3_schedule-pilot/report/regret_curves.png)

## phase3_hparams

80 campaigns: 10 problems x 2 arms, 4 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm     |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:--------|--------------------:|-------------:|------:|----------------:|----------:|
| tuned   |               0.137 |        0.175 | 0.215 |           0.5   |     1.527 |
| untuned |               0.149 |        0.181 | 0.218 |           0.425 |     1.717 |

Family F5 (paired differences, negative = arm better):

| arm   | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| tuned | untuned     |           10 |        40 |      -0.011 |   -0.039 |     0.002 |            0.4 |           0.2 |     0.438 |    0.438 | False         |

Final normalised regret by landscape:

| landscape   |   tuned |   untuned |
|:------------|--------:|----------:|
| interaction |   0     |     0.002 |
| multipeak   |   0.276 |     0.28  |
| needle      |   1     |     1     |
| real        |   0.012 |     0.004 |
| smooth      |   0.012 |     0.015 |
| threshold   |   0.02  |     0.056 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm     |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:--------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| tuned   |     0.801 |           0.832 |        0.872 |        0.967 |       0.111 |       0.831 |            0.326 |
| untuned |     0.77  |           0.798 |        0.842 |        0.95  |       0.098 |       0.938 |            0.311 |

![regret curves](../../results/phase3_hparams-pilot/report/regret_curves.png)

## phase3_ablation

160 campaigns: 5 problems x 16 arms, 2 seeds each.

Mean endpoints per arm (normalised regret: 0 = optimum, 1 = median random recipe):

| arm                               |   final_norm_regret |   rec_regret |   auc |   found_optimum |   minutes |
|:----------------------------------|--------------------:|-------------:|------:|----------------:|----------:|
| abl_batch                         |               0.193 |        0.302 | 0.267 |             0.1 |     6.028 |
| abl_batch+pool                    |               0.143 |        0.213 | 0.272 |             0.1 |     6.163 |
| abl_batch+pool+schedule           |               0.113 |        0.168 | 0.262 |             0.1 |     6.072 |
| abl_batch+pool+schedule+surrogate |               0.11  |        0.16  | 0.22  |             0.4 |     1.024 |
| abl_batch+pool+surrogate          |               0.08  |        0.112 | 0.207 |             0.2 |     1.111 |
| abl_batch+schedule                |               0.194 |        0.327 | 0.27  |             0.1 |     6.226 |
| abl_batch+schedule+surrogate      |               0.198 |        0.31  | 0.265 |             0.1 |     1.037 |
| abl_batch+surrogate               |               0.2   |        0.309 | 0.268 |             0.1 |     0.736 |
| abl_pool                          |               0.185 |        0.266 | 0.315 |             0.1 |     1.014 |
| abl_pool+schedule                 |               0.184 |        0.277 | 0.311 |             0.1 |     1.037 |
| abl_pool+schedule+surrogate       |               0.089 |        0.163 | 0.221 |             0.2 |     0.694 |
| abl_pool+surrogate                |               0.091 |        0.135 | 0.22  |             0.3 |     0.559 |
| abl_schedule                      |               0.194 |        0.321 | 0.264 |             0.1 |     0.96  |
| abl_schedule+surrogate            |               0.193 |        0.305 | 0.26  |             0.1 |     0.445 |
| abl_surrogate                     |               0.193 |        0.27  | 0.26  |             0.1 |     0.402 |
| incumbent                         |               0.196 |        0.303 | 0.265 |             0.1 |     1.094 |

Family main (paired differences, negative = arm better):

| arm                               | reference   |   n_problems |   n_pairs |   mean_diff |   ci_low |   ci_high |   share_better |   share_worse |   p_value |   p_holm | significant   |
|:----------------------------------|:------------|-------------:|----------:|------------:|---------:|----------:|---------------:|--------------:|----------:|---------:|:--------------|
| abl_batch                         | incumbent   |            5 |        10 |      -0.003 |   -0.012 |     0.007 |            0.4 |           0.2 |     0.5   |    1     | False         |
| abl_pool                          | incumbent   |            5 |        10 |      -0.01  |   -0.064 |     0.07  |            0.8 |           0.2 |     0.625 |    1     | False         |
| abl_schedule                      | incumbent   |            5 |        10 |      -0.001 |   -0.005 |     0     |            0.2 |           0   |     1     |    1     | False         |
| abl_surrogate                     | incumbent   |            5 |        10 |      -0.003 |   -0.013 |     0.007 |            0.4 |           0.2 |     0.5   |    1     | False         |
| abl_batch+pool                    | incumbent   |            5 |        10 |      -0.052 |   -0.122 |     0.026 |            0.6 |           0.4 |     0.312 |    1     | False         |
| abl_batch+schedule                | incumbent   |            5 |        10 |      -0.001 |   -0.01  |     0.007 |            0.4 |           0.2 |     0.75  |    1     | False         |
| abl_batch+surrogate               | incumbent   |            5 |        10 |       0.005 |   -0.013 |     0.028 |            0.4 |           0.4 |     1     |    1     | False         |
| abl_pool+schedule                 | incumbent   |            5 |        10 |      -0.012 |   -0.065 |     0.057 |            0.6 |           0.4 |     0.625 |    1     | False         |
| abl_pool+surrogate                | incumbent   |            5 |        10 |      -0.105 |   -0.166 |    -0.044 |            1   |           0   |     0.062 |    0.938 | False         |
| abl_schedule+surrogate            | incumbent   |            5 |        10 |      -0.003 |   -0.013 |     0.007 |            0.4 |           0.2 |     0.5   |    1     | False         |
| abl_batch+pool+schedule           | incumbent   |            5 |        10 |      -0.083 |   -0.152 |    -0.025 |            0.8 |           0   |     0.125 |    1     | False         |
| abl_batch+pool+surrogate          | incumbent   |            5 |        10 |      -0.116 |   -0.196 |    -0.044 |            1   |           0   |     0.062 |    0.938 | False         |
| abl_batch+schedule+surrogate      | incumbent   |            5 |        10 |       0.003 |   -0.013 |     0.025 |            0.4 |           0.2 |     1     |    1     | False         |
| abl_pool+schedule+surrogate       | incumbent   |            5 |        10 |      -0.106 |   -0.179 |    -0.038 |            1   |           0   |     0.062 |    0.938 | False         |
| abl_batch+pool+schedule+surrogate | incumbent   |            5 |        10 |      -0.086 |   -0.155 |    -0.025 |            1   |           0   |     0.062 |    0.938 | False         |

Final normalised regret by landscape:

| landscape   |   abl_batch |   abl_batch+pool |   abl_batch+pool+schedule |   abl_batch+pool+schedule+surrogate |   abl_batch+pool+surrogate |   abl_batch+schedule |   abl_batch+schedule+surrogate |   abl_batch+surrogate |   abl_pool |   abl_pool+schedule |   abl_pool+schedule+surrogate |   abl_pool+surrogate |   abl_schedule |   abl_schedule+surrogate |   abl_surrogate |   incumbent |
|:------------|------------:|-----------------:|--------------------------:|------------------------------------:|---------------------------:|---------------------:|-------------------------------:|----------------------:|-----------:|--------------------:|------------------------------:|---------------------:|---------------:|-------------------------:|----------------:|------------:|
| interaction |       0.018 |            0.068 |                     0.018 |                               0     |                      0     |                0.018 |                          0.018 |                 0.018 |      0.117 |               0.075 |                         0     |                0     |          0.018 |                    0.018 |           0.018 |       0.018 |
| multipeak   |       0.46  |            0.366 |                     0.361 |                               0.277 |                      0.277 |                0.466 |                          0.46  |                 0.46  |      0.441 |               0.408 |                         0.275 |                0.284 |          0.466 |                    0.46  |           0.46  |       0.472 |
| real        |       0.048 |            0.084 |                     0.01  |                               0     |                      0.007 |                0.048 |                          0.086 |                 0.086 |      0.012 |               0.024 |                         0.013 |                0.009 |          0.048 |                    0.048 |           0.048 |       0.048 |
| smooth      |       0.28  |            0.121 |                     0.079 |                               0.178 |                      0.04  |                0.28  |                          0.268 |                 0.28  |      0.254 |               0.313 |                         0.072 |                0.101 |          0.268 |                    0.28  |           0.28  |       0.268 |
| threshold   |       0.159 |            0.078 |                     0.095 |                               0.093 |                      0.075 |                0.159 |                          0.158 |                 0.158 |      0.103 |               0.098 |                         0.087 |                0.061 |          0.171 |                    0.158 |           0.158 |       0.171 |

Surrogate quality, median over guided rounds (held-out R2 / Spearman; coverage of nominal 80% and 95% intervals; calibration error; z_sd > 1 = overconfident):

| arm                               |   test_r2 |   test_spearman |   test_cov80 |   test_cov95 |   test_mace |   test_z_sd |   batch_spearman |
|:----------------------------------|----------:|----------------:|-------------:|-------------:|------------:|------------:|-----------------:|
| abl_batch                         |     0.407 |           0.755 |        0.292 |        0.503 |       0.335 |       2.183 |            0.591 |
| abl_batch+pool                    |     0.289 |           0.698 |        0.315 |        0.548 |       0.326 |       2.249 |            0.432 |
| abl_batch+pool+schedule           |     0.317 |           0.702 |        0.322 |        0.545 |       0.315 |       2.198 |            0.465 |
| abl_batch+pool+schedule+surrogate |     0.769 |           0.829 |        0.858 |        0.957 |       0.111 |       0.873 |            0.234 |
| abl_batch+pool+surrogate          |     0.835 |           0.847 |        0.867 |        0.962 |       0.108 |       0.823 |            0.18  |
| abl_batch+schedule                |     0.399 |           0.76  |        0.282 |        0.493 |       0.336 |       2.387 |            0.622 |
| abl_batch+schedule+surrogate      |     0.884 |           0.888 |        0.913 |        0.977 |       0.161 |       0.706 |            0.708 |
| abl_batch+surrogate               |     0.874 |           0.895 |        0.915 |        0.977 |       0.16  |       0.755 |            0.666 |
| abl_pool                          |     0.282 |           0.662 |        0.303 |        0.53  |       0.327 |       2.463 |            0.193 |
| abl_pool+schedule                 |     0.304 |           0.666 |        0.315 |        0.522 |       0.325 |       2.542 |            0.172 |
| abl_pool+schedule+surrogate       |     0.767 |           0.817 |        0.878 |        0.95  |       0.133 |       0.871 |            0.315 |
| abl_pool+surrogate                |     0.785 |           0.823 |        0.885 |        0.948 |       0.116 |       0.827 |            0.308 |
| abl_schedule                      |     0.422 |           0.772 |        0.315 |        0.53  |       0.33  |       2.089 |            0.649 |
| abl_schedule+surrogate            |     0.895 |           0.878 |        0.913 |        0.98  |       0.157 |       0.724 |            0.702 |
| abl_surrogate                     |     0.903 |           0.877 |        0.903 |        0.982 |       0.164 |       0.761 |            0.682 |
| incumbent                         |     0.404 |           0.776 |        0.307 |        0.528 |       0.335 |       2.026 |            0.596 |

Ablation: mean paired difference per subset:

| subset                        |    mean |    sem |   count |
|:------------------------------|--------:|-------:|--------:|
| incumbent                     |  0      | 0      |       5 |
| batch                         | -0.0026 | 0.0045 |       5 |
| batch+pool                    | -0.0522 | 0.0399 |       5 |
| batch+pool+schedule           | -0.0828 | 0.0324 |       5 |
| batch+pool+surrogate          | -0.1159 | 0.0416 |       5 |
| batch+schedule                | -0.0013 | 0.004  |       5 |
| batch+surrogate               |  0.0049 | 0.0094 |       5 |
| batch+schedule+surrogate      |  0.0025 | 0.0093 |       5 |
| pool                          | -0.0101 | 0.0285 |       5 |
| pool+schedule                 | -0.012  | 0.0269 |       5 |
| pool+surrogate                | -0.1045 | 0.0337 |       5 |
| schedule                      | -0.0012 | 0.0012 |       5 |
| surrogate                     | -0.0027 | 0.0046 |       5 |
| schedule+surrogate            | -0.0027 | 0.0046 |       5 |
| pool+schedule+surrogate       | -0.1062 | 0.0386 |       5 |
| batch+pool+schedule+surrogate | -0.0859 | 0.0301 |       5 |

Ablation: main effects and interactions:

| term               |   effect_on_regret |   ci_low |   ci_high |
|:-------------------|-------------------:|---------:|----------:|
| batch              |            -0.0076 |  -0.0194 |    0.0086 |
| pool               |            -0.0161 |  -0.0577 |    0.0386 |
| schedule           |            -0.0008 |  -0.0035 |    0.0035 |
| surrogate          |            -0.0153 |  -0.0224 |   -0.005  |
| batchxpool         |            -0.0322 |  -0.0651 |   -0.0058 |
| batchxschedule     |            -0.0029 |  -0.0118 |    0.0046 |
| batchxsurrogate    |             0.0306 |   0.0119 |    0.0558 |
| poolxschedule      |            -0.0042 |  -0.0193 |    0.0109 |
| poolxsurrogate     |            -0.0693 |  -0.1038 |   -0.0339 |
| schedulexsurrogate |             0.0109 |   0.0054 |    0.0164 |

Ablation: Shapley shares of the full improvement:

| intervention   |   shapley_improvement |   ci_low |   ci_high |
|:---------------|----------------------:|---------:|----------:|
| batch          |                0.0049 |  -0.0022 |    0.0101 |
| pool           |                0.0636 |   0.0154 |    0.1045 |
| schedule       |               -0.0042 |  -0.0177 |    0.0067 |
| surrogate      |                0.0217 |   0.0077 |    0.0402 |

![regret curves](../../results/phase3_ablation-pilot/report/regret_curves.png)

## Decision step

Chosen interventions (`artifacts/winners-pilot.json`, significance required: False):

```json
{
 "F1:init": {
  "init": {
   "type": "sobol"
  }
 },
 "F2:surrogate": {
  "surrogate": {
   "type": "gp_matern",
   "nu": 2.5,
   "n_restarts": 0
  }
 },
 "F3:selector": {
  "selector": {
   "type": "sequential_fantasy"
  }
 },
 "F3b:use_noise": {
  "use_noise": true
 },
 "F3c:pool": {
  "pool": {
   "type": "local",
   "local_share": 0.5,
   "n_parents": 8,
   "max_changes": 3
  }
 }
}
```

## Tuning: schedule

Winner `schedule_adaptive_start1.41`: `{"type": "adaptive", "start": 1.41}`

Best candidates on the dev split (mean final normalised regret):

| arm                                   |   mean |    sem |   count |
|:--------------------------------------|-------:|-------:|--------:|
| schedule_adaptive_start1.41           | 0.182  | 0.1062 |       9 |
| schedule_constant_beta1.0             | 0.1896 | 0.1054 |       9 |
| schedule_linear_start2.0_end0.25      | 0.1954 | 0.1056 |       9 |
| schedule_gp_ucb_scale0.2              | 0.196  | 0.1044 |       9 |
| schedule_exponential_start2.0_end0.25 | 0.1967 | 0.1056 |       9 |
| schedule_legacy                       | 0.2086 | 0.1035 |       9 |

## Tuning: surrogate

Winner `surrogate_gp_matern_nu1.5_n_restarts0`: `{"type": "gp_matern", "nu": 1.5, "n_restarts": 0}`

Best candidates on the dev split (mean final normalised regret):

| arm                                   |   mean |    sem |   count |
|:--------------------------------------|-------:|-------:|--------:|
| surrogate_gp_matern_nu1.5_n_restarts0 | 0.1698 | 0.1078 |       9 |
| surrogate_gp_matern_nu2.5_n_restarts0 | 0.182  | 0.1062 |       9 |
