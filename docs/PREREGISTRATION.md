# METIS benchmark: preregistered protocol

Status: **frozen 2026-10-14**, before any confirmatory campaign was run. The metric
definitions (`metis_bench/metrics.py`) and the statistical procedure
(`metis_bench/stats.py`) are pinned by SHA-256 in `metis_bench/FROZEN.json`, and
`tests/test_frozen.py` fails if either file changes. Changes after this date go in
the amendment log (section 9) with a reason, a date and the new hash, and are reported
as deviations.

Pilot runs (fewer seeds, cheaper incumbent tuning) exist only to test the machinery. They
are never used for confirmatory claims or to choose between arms.

## 1. Questions

| ID | Question | Phase |
| --- | --- | --- |
| H1 | Does a space-filling initial design (scrambled Sobol, maximin LHS) beat METIS's uniform random day 1? Does "fill the unsampled region" help when a campaign starts from imported data? | 2 (step 8) |
| H2 | Do TabPFN or a Matern GP beat the XGBoost ensemble as surrogate, in final regret, predictive accuracy and interval calibration? | 2 (step 9) |
| H3 | Does proper batch selection with a diversity penalty beat top-32-by-UCB and the v1 `batch_ucb` heuristic? Does feeding replicate standard deviations into the surrogate help? | 2 (step 10) |
| H4 | Does a tuned or adaptive exploration schedule beat the hand-set 1.41 / 1.0 / 0.5 list? | 3 (step 11) |
| H5 | Do tuned hyperparameters improve the step 9 winner? | 3 (step 12) |
| H6 | Which intervention carries the effect, alone and in combination? | 3 (step 13) |

## 2. Problems

* **Synthetic suite** (`suite/suite.json`): 9 landscape subtypes (smooth: broad,
  plateau; threshold: switch on, switch off, rise then crash; needle; multiple peaks with
  a decoy; interactions: multiplicative pairs, required ratio). Each has 5 specs chosen by
  the calibration procedure (section 2.1), and each spec is instantiated at 12 and 27
  factors: 90 functions. Every function's global maximum on its droplet grid is exact
  (dynamic programming over the volume budget, verified by brute force in tests).
* **Real oracles**: `borkowski` (gold XGBoost on 1,017 recipes, 11 factors x 4 levels,
  maximum by full enumeration of 4^11 recipes) and `pure_pool` (gold XGBoost on 1,094
  PURE recipes; candidates are the measured recipes; exact maximum over the pool).
  `pure_grid` has no proven maximum and is excluded from confirmatory analysis.
* **Noise presets** (`metis_bench/noise.py`): `realistic` is primary. `none` and `high`
  are robustness analyses. Triplicate wells in every case.

### 2.1 Calibration and realism

Descriptors (zero fraction, skew, top/median ratio, first-order and pairwise Sobol sums,
share of factors for 80% of variance, correlation length, share of distinct hill-climbing
peaks, threshold strength, learnability at 32/100/300 designs, noise ratio) are computed
for every synthetic function and for both real datasets. For the real data they come from
raw samples where possible and from XGBoost and GP fits otherwise; both model versions
are reported. Realism score = RMS standardised distance to the nearest real reference
vector. Per subtype, 3 of 5 specs are drawn with probability proportional to
exp(-d^2 / 2 tau^2) (tau = median distance in the subtype), and 2 are chosen for coverage
(farthest point in descriptor space). Every result carries its function's realism score.

### 2.2 Dev / test split

Within each subtype, specs are sorted by id and every third goes to **dev** (both of its
dimensionalities). The rest are **test**. Phase 3 tuning (schedules, hyperparameters) may
look only at dev. All confirmatory comparisons use the test split plus the real oracles,
which are never used for tuning.

## 3. Campaign protocol

* 32 recipes per round, 10 rounds. Round 1 is the arm's initial design; rounds 2 to 10 are
  model guided (as METIS: day 1 random, then guided days).
* Candidate pool: 10,000 fresh unmeasured feasible recipes per round, drawn uniformly per
  factor with volume rejection (METIS's sampler). PURE pool: all unmeasured pool recipes.
* Surrogates are refitted from scratch on all replicate means each round.
* **Incumbent fidelity**: confirmatory runs use the notebook incumbent unchanged:
  RandomizedSearchCV with 200 draws, 5-fold, MAE, over the v1 grid, and the top 20
  settings as the ensemble.
* Exploration: UCB `1 * mean + beta * std`, with the v1 schedule beta = 1.41 for rounds
  2 and 3, 1.0 for rounds 4 to 7, and 0.5 for rounds 8 to 10 unless the arm says otherwise.
* **Seeds**: 50 per (problem, arm), seeds 0 to 49. A seed fixes the initial design (per
  initialiser type), the imported data, and every noise draw, so arms are paired
  (common random numbers).
* Imported-data scenario (H1 fill): 32 one-factor-at-a-time recipes around two base
  recipes in the lower half of the space are measured before round 1.

## 4. Metrics (frozen in `metis_bench/metrics.py`)

* `f_ref` = median noise-free yield of 10,000 random feasible designs (fixed seed).
  Normalised regret `(f_max - v) / (f_max - f_ref)`: 0 = optimum, 1 = median random
  design.
* **Primary endpoint**: `final_norm_regret`, the normalised simple regret of the best
  *true* yield among all recipes measured after round 10.
* Secondary endpoints:
  * `final_norm_rec_regret`: regret of the recipe with the best *measured* mean (what an
    experimenter would carry forward).
  * `auc_norm_regret`: mean normalised regret over rounds 1 to 10.
  * `final_norm_cum_regret`: mean normalised regret over all measured recipes.
  * `rounds_to_target`: first round with normalised regret <= 0.05 (11 if never).
* Surrogate quality, every model-guided round, on 1,000 fixed random feasible designs
  (noise-free truth): R2, Spearman rho, RMSE; interval calibration of `mean +/- z * std`:
  coverage at 50/80/95%, mean absolute calibration error over levels 0.1 to 0.9, Gaussian
  NLPD, sharpness, standardised-error SD. The same on the batch actually chosen
  (on-policy).

## 5. Statistical procedure (frozen in `metis_bench/stats.py`)

* Unit of analysis for suite-level claims: the problem. Seeds are within-problem
  replicates.
* Per problem: mean paired difference (arm - reference) over seeds; two-sided Wilcoxon
  signed-rank across seeds.
* Across problems: two-sided Wilcoxon signed-rank on per-problem mean differences;
  Hodges-Lehmann estimate; share of problems improved; hierarchical bootstrap 95% CI
  (10,000 resamples: problems, then seeds within problems; seed 20241014).
* Multiplicity: Holm-Bonferroni within each family of section 6, alpha = 0.05.
* Practical relevance: |mean difference in normalised regret| >= 0.05.
* Stratified reporting (descriptive, not tested): landscape class, dimension, noise
  preset, realism tercile, real oracles separately.

## 6. Arms and comparison families

Reference arm everywhere: `incumbent` (random init, XGB ensemble, UCB with the v1
schedule, top-k, replicate stds ignored). Exact arm definitions are in `experiments/*.yaml`.

| Family | Arms (each vs reference) | Endpoint |
| --- | --- | --- |
| F1 init | `init_sobol`, `init_lhs` | primary |
| F1b imported data | `import_fill` vs `import_v1` | primary |
| F2 surrogate | `surr_gp`, `surr_tabpfn` | primary; accuracy and calibration secondary |
| F3 batching and noise | `batch_fantasy` (v1 heuristic), `batch_lp` (local penalisation, radius floor 0.15), `noise_xgb` (replicate stds as XGB weights) | primary |
| F3b noise feeding with GP | `surr_gp_noise` vs `surr_gp` | primary |
| F3c candidate pool (amendment 1) | `pool_local` (half the pool from mutations of the 8 best measured recipes) | primary |
| F4 schedule | best dev schedule vs the v1 schedule, both on top of the Phase 2 winners | primary |
| F5 hyperparameters | tuned vs untuned winning surrogate, both with the Phase 2 winners and the tuned schedule | primary |
| F6 ablation | every non-empty subset of the winning interventions vs `incumbent` | primary; main effects and interactions by linear model on paired differences; Shapley shares |

Descriptive baselines, outside every family and never tested: `random_search` in
`phase2_init` (the floor any model-guided arm should clear), and `fig1b_check` (the
Borkowski oracle with Fig. 1b's 100 recipes per round, METIS v1 vs random search, to
confirm the oracle reproduces the paper's qualitative behaviour).

Decision rule (`python -m metis_bench decide`), applied per intervention part (initial
design, surrogate, selector, noise feeding, schedule): among arms that differ from
their family reference in exactly that part and are significantly better (Holm), the
one with the lowest mean difference wins. If none qualifies, the incumbent's choice
stays; in particular, step 12 then tunes the incumbent XGBoost ensemble. Phase 3 builds
on these winners (`artifacts/winners.json`).

## 7. Analysis outputs

Per family: the table from `stats.compare` (mean difference, Hodges-Lehmann, CI, share
better/worse, raw and Holm p), regret curves by round, stratified tables, and for F2 the
per-round accuracy and calibration curves.

## 8. What is not claimed

* Synthetic landscapes are calibrated toward two cell-free datasets. Claims hold for
  landscapes like these, and results carry realism scores so the reader can weigh them.
* The PURE gold model reaches only R2 of about 0.4 (one-factor-at-a-time data), so the
  `pure_pool` oracle is a weak proxy for PURE itself.

## 9. Amendment log

| Date | Change | Reason | New hashes |
| --- | --- | --- | --- |
| 2026-10-14 | Initial freeze | | see FROZEN.json |
| 2026-10-14 | Amendment 1: family F3c, candidate pools (`pool_local` vs `incumbent`); `pool` added as an intervention part for the decision rule and the ablation | The Phase 2 pilot showed every arm plateauing at the best recipe its random candidate pools contained: on a 12-factor function, 100,000 random recipes reach about 85% of the optimum, while local moves from the best reach 98%. Added before any confirmatory campaign; metrics and statistics unchanged | unchanged |
