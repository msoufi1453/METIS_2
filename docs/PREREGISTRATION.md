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
| H1 | Does a space-filling initial design (scrambled Sobol, maximin LHS) beat METIS's uniform random day 1? Does "fill the unsampled region" help when a campaign starts from imported data? (Amendment 6: Sobol only; the imported-data question is dropped.) | 2 (step 8) |
| H2 | Do TabPFN or a Matern GP beat the XGBoost ensemble as surrogate, in final regret, predictive accuracy and interval calibration? | 2 (step 9) |
| H3 | Does proper batch selection with a diversity penalty beat top-32-by-UCB and the v1 `batch_ucb` heuristic? Does feeding replicate standard deviations into the surrogate help? (Amendment 6: the replicate-noise question is dropped.) | 2 (step 10) |
| H4 | Does a tuned or adaptive exploration schedule beat the hand-set 1.41 / 1.0 / 0.5 list? (Amendment 6: not tested; every model keeps its own schedule.) | 3 (step 11) |
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
* **Noise presets** (`metis_bench/noise.py`): `realistic` is used for every
  confirmatory experiment, including the ablation (amendment 3). `none` and `high` remain
  available for exploratory robustness checks. Triplicate wells in every case.

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
look only at dev (after amendment 6, hyperparameters only). All confirmatory comparisons use the test split plus the real oracles,
which are never used for tuning. (One exception, amendment 4: TabPFN's own exploration
schedule was tuned on the measured PURE and Borkowski data the two real oracles are built
from.)

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
  TabPFN arms that set no schedule use TabPFN's own tuned schedule instead (amendment 4):
  `exploit_ramp` with slope 4.5, i.e. UCB `0.5 * mean + 0.5 * std` in round 2 and the mean
  alone from round 3 (tuned 2026-10-28, before any confirmatory campaign).
* **Seeds**: 20 per (problem, arm), seeds 0 to 19 (amendment 3; was 50); 10 (seeds 0 to 9) for
  families F1, F3 and F3c (amendment 6). Tuning uses
  10 seeds, 1000 to 1009 (was 20). A seed fixes the initial design (per
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
| F1 init | `init_sobol` (`init_lhs` dropped, amendment 6) | primary |
| ~~F1b imported data~~ | dropped (amendment 6) | |
| F2 surrogate | `surr_gp`, `surr_tabpfn` (TabPFN-3.5 with its tuned schedule, amendments 4 and 5) | primary; accuracy and calibration secondary |
| F3 batching | `batch_fantasy` (v1 heuristic), `batch_lp` (local penalisation, radius floor 0.15); `noise_xgb` dropped (amendment 6) | primary |
| ~~F3b noise feeding with GP~~ | dropped (amendment 6) | |
| F3c candidate pool (amendment 1) | `pool_local` (half the pool from mutations of the 8 best measured recipes) | primary |
| ~~F4 schedule~~ | dropped (amendment 6): XGBoost and the GP keep the v1 schedule, TabPFN its amendment-4 schedule | |
| F5 hyperparameters | tuned vs untuned winning surrogate, both with the Phase 2 winners and the model's own schedule | primary |
| F6 ablation | every non-empty subset of the winning interventions vs `incumbent`, on 30 problems (28 test functions sampled evenly over subtype x dimension, plus both real oracles), realistic noise (amendment 3) | primary; main effects and interactions by linear model on paired differences; Shapley shares |

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
| 2026-10-14 | Amendment 2: `metrics.accuracy` returns NaN for R2 and Spearman when the noise-free test truth is flat (sum of squares <= 1e-12 x sum of squared values) | Needle test sets sit entirely on the flat background; a rounding-level denominator gave R2 near -1e29 in the pilot. Secondary surrogate metric only; no confirmatory data existed | metrics.py 0b35edd53939... (FROZEN.json) |
| 2026-10-18 | Amendment 3: compute budget. (a) 20 paired seeds per (problem, arm) instead of 50; (b) tuning: 10 seeds instead of 20, and 15 schedule candidates instead of 41 (at least one per family); (c) ablation: realistic noise only (was realistic, none and high) and 30 problems (28 test functions stratified over subtype x dimension by a fixed hash, plus both real oracles) instead of 56; (d) identical campaigns (same resolved arm, problem, seed and settings) are computed once and shared between experiments; (e) the TabPFN arm keeps full coverage (all 56 problems x 20 seeds) and may run on a GPU; (f) `incumbent_check` in phase3_schedule resets the candidate pool to random, so it is the true v1 incumbent | The full design was about 19,000 core-hours (most of it the ablation at three noise levels). The amended design is about 2,700 core-hours plus the TabPFN arm. The main test treats problems (56) as the unit, so fewer seeds mainly widens per-problem averages and does not bias paired comparisons. (d) and (f) do not change any result. Made before any confirmatory campaign | unchanged |
| 2026-10-28 | Amendment 4: TabPFN's own exploration schedule. (a) New schedule family `exploit_ramp`: exploitation weight min(1, slope x (round - 1) / 9), exploration weight 1 - exploitation. (b) `experiments/tabpfn_schedule_tune.yaml` picks one of 10 slopes (0.8 to 4.5; every one starts mostly exploring and ends mostly or fully exploiting) for TabPFN-3.5 on the measured-data replay (PURE, Borkowski, CETCH yield and efficiency; docs/REPLAY.md), 5 seeds, by the lowest mean `auc_norm_regret` with datasets weighted equally. Every arm reaches the pool's best recipe there, so the primary endpoint ties. The v1 schedule runs as a reference and cannot win. (c) The winner (`experiments/tuned/tabpfn_schedule.json`) is the schedule of every TabPFN arm that sets none, including `surr_tabpfn` in F2 and any Phase 3 arm built from a TabPFN winner; confirmatory runs refuse to start without it. (d) The decision rule counts TabPFN with its own schedule as one surrogate intervention. XGBoost and the GP keep the v1 schedule | In the replay, the XGBoost ensemble's 80% intervals held 30-41% of held-out recipes and TabPFN-3.5's 88-93%. The v1 weights were set for XGBoost, so under them TabPFN explores far more, and F2 would partly compare the size of the two models' uncertainties rather than the models. Caveats: the tuning data are the measurements the `borkowski` and `pure_pool` oracles are built from, so TabPFN's F2 result on those two problems is not independent of the tuning (the stratified tables give the synthetic-only result); the schedule is tuned with TabPFN-3.5, while `surr_tabpfn` uses TabPFN v2 unless changed. Made before any confirmatory campaign | unchanged |
| 2026-10-28 | Amendment 5: every TabPFN arm uses TabPFN-3.5 (`tabpfn==9.2.0`, `version: v3.5`, environment extra `tabpfn35`) instead of TabPFN v2 (`tabpfn==2.0.9`). The Phase 3 TabPFN hyperparameter candidates also name `device: auto`, which they had lost by replacing the whole surrogate setting | TabPFN-3.5 is the model the schedule of amendment 4 was tuned with, and the current TabPFN release. Its weights are under a non-commercial licence and need a one-time licence acceptance. Made before any confirmatory campaign | unchanged |
| 2026-10-28 | Amendment 6: fewer XGBoost campaigns. (a) Dropped from Phase 2: `init_lhs` (F1 keeps `init_sobol`), the imported-data family F1b (`import_v1`, `import_fill`), `noise_xgb` (F3) and family F3b (`surr_gp_noise` vs `surr_gp`). (b) Families F1, F3 and F3c use 10 seeds (0 to 9); F2 (XGBoost vs TabPFN vs GP) keeps all 56 problems x 20 seeds, and the incumbent's seeds 0 to 9 are shared. (c) Phase 3 no longer tunes the exploration schedule (step 11, family F4): every model keeps its own (XGBoost and the GP: v1; TabPFN: amendment 4), so steps 12 and 13 build on the Phase 2 winners and the hyperparameter tuning only. (d) Reports analyse only the configured arms and seeds, so files from the earlier plan are ignored | The measured-data replay timed a full XGBoost campaign at 15-30 CPU-minutes, 2-4 times the amendment-3 estimate. Phase 2 had 10,040 XGBoost campaigns (about 2,500-5,000 core-hours) and Phase 3's schedule tuning about 7,600 more if XGBoost wins. After (a) and (b) Phase 2 has 3,360. The main question (H2) keeps full coverage; the remaining side questions lose power (per-problem means from 10 seeds instead of 20). The v1 schedule was set for XGBoost by the METIS authors. Made before any confirmatory campaign was analysed | unchanged |
