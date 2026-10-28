# Replay of the paper's measured-data experiments (exploratory)

A quick backup to the synthetic benchmark: run METIS against the published data, as the
METIS paper did for PURE (Fig. 1). This is **exploratory**. It is not part of the
preregistered study (docs/PREREGISTRATION.md), uses 3 seeds, and makes no significance
claims. The synthetic benchmark is unchanged and waits for GPUs.

## Set-up

| | |
|---|---|
| Oracle | the measured data. Only recipes that were measured can be chosen, and each returns its measured yield (mean of replicates; identical recipes merged by mean). No added noise. |
| Datasets | PURE (Matsuura et al. 2009, 1,094 recipes, 69 factors); Borkowski et al. 2020 (1,007 recipes, 11 factors); CETCH yield (5 days, 625 recipes) and CETCH efficiency (8 days, 1,000 recipes), 32 columns each. |
| Concatenated tables | `data/replay/<dataset>.csv` (all days in one file, with a `day` column; written by `metis_bench.oracles.measured.write_concatenated`). |
| Held-out set | a fixed 20% of recipes per dataset (split seed 41, as in the PURE notebook), never offered to the optimiser; used only to score R² and Spearman after each fit. |
| Candidate pool | the other 80%. Every round scores all unmeasured pool recipes, so the 10,000-random-candidate generator does not apply. |
| Optimum | the best measured yield in the pool, so "found the optimum" is exact for the data. |
| Rounds | 10. Round 1 is random; rounds 2-10 are model-guided (UCB, METIS v1 exploration schedule, top-k batch). |
| Batch size | about pool/20, so 10 rounds measure half the pool: PURE 44, Borkowski 40, CETCH yield 25, CETCH efficiency 40. The paper's 80-100 per round would use up these pools before round 10. |
| Seeds | 3 (0, 1, 2). A seed fixes the round-1 recipes, shared by all models. |

Models:

| arm | surrogate |
|---|---|
| `incumbent` | METIS v1: XGBoost ensemble, 200 random-search draws, 20 members |
| `linear` | ordinary least squares (`LinearRegression`), the paper's linear model. Its 20 copies are identical, so there is no uncertainty and UCB ranks by the mean. |
| `tabpfn35` | TabPFN-3.5 (open weights, `tabpfn==9.2.0`), 8 estimators, predictive std from its output distribution |

## Running it

```bash
uv sync --extra tabpfn35 --extra cpu      # or cu126 / cu128 on an NVIDIA GPU
python -m metis_bench run experiments/replay_paper.yaml --jobs 6
python -m metis_bench replay experiments/replay_paper.yaml
```

Outputs land in `results/replay_paper/replay/`: `yields_by_round.png` (the paper-style
figure), `best_so_far.png`, `heldout_r2.png`, `summary.md` and `summary.csv`.
`python -m metis_bench report experiments/replay_paper.yaml` also works, but with 3 seeds
its paired tests have almost no power.

### TabPFN-3.5 licence (one time, by you)

The TabPFN-3.5 weights are under a **non-commercial licence**, and downloading them needs
a free Prior Labs account and a one-time licence acceptance. In your own terminal:

```bash
.venv\Scripts\python.exe -c "from tabpfn import TabPFNRegressor; import numpy as np; TabPFNRegressor(device='cpu').fit(np.random.rand(20,3), np.random.rand(20))"
```

This opens a browser to log in and accept, then caches a token in
`~/.cache/tabpfn/auth_token`. On a machine without a browser, set the `TABPFN_TOKEN`
environment variable instead. After that, `run` picks up the `tabpfn35` arm; finished
arms are skipped.

The main benchmark uses TabPFN v2 (`--extra tabpfn`, package 2.0.9). The two extras
cannot be installed together, and `TabPFNSurrogate` refuses to run if the installed
package does not match the arm's `version`.

## Data note: CETCH `Yield_all_data.csv`

The repository's `Examples/CETCH/Data/Yield_optimization/Yield_all_data.csv` stacks the
day files by column position, but days 2-5 list the one-hot HEPES pH columns in a
different order from day 1. As a result, 500 of its 625 rows carry the wrong pH. The
replay concatenates the per-day `Results_<day>.csv` files by column name instead
(checked in `tests/test_replay.py`).

## Results (2026-10-28; XGBoost, linear and random; TabPFN-3.5 pending the licence step)

Full table: [replay/summary.md](replay/summary.md). Figures:
[recipes per round](replay/yields_by_round.png), [best so far](replay/best_so_far.png),
[held-out R²](replay/heldout_r2.png). Mean (sd) over 3 seeds:

| dataset | model | found best (seeds) | round found | AUC norm. regret | held-out R² |
|---|---|---|---|---|---|
| PURE | XGBoost (incumbent) | 3/3 | 5.7 (2.5) | 0.105 | 0.36 |
| PURE | linear | 3/3 | 6.7 (1.5) | 0.113 | -4.61 |
| PURE | random | 1/3 | 10 | 0.198 | - |
| Borkowski | XGBoost (incumbent) | 3/3 | 2.0 (0.0) | 0.025 | 0.85 |
| Borkowski | linear | 3/3 | 2.3 (0.6) | 0.030 | 0.80 |
| Borkowski | random | 2/3 | 3.0 | 0.075 | - |
| CETCH yield | XGBoost (incumbent) | 3/3 | 4.3 (2.1) | 0.122 | 0.57 |
| CETCH yield | linear | 3/3 | 6.0 (1.0) | 0.168 | 0.37 |
| CETCH yield | random | 1/3 | 7 | 0.294 | - |
| CETCH efficiency | XGBoost (incumbent) | 3/3 | 3.3 (0.6) | 0.082 | 0.56 |
| CETCH efficiency | linear | 3/3 | 4.3 (2.1) | 0.107 | 0.33 |
| CETCH efficiency | random | 0/3 | - | 0.244 | - |

What it shows:

* Both models beat random on every dataset. Random selection found the best recipe in
  4 of 12 runs; each model found it in all 12.
* XGBoost found the best recipe earlier than linear on average on every dataset, and
  predicts held-out recipes better (R² 0.36-0.85 vs -4.6-0.80). The gap is small on
  Borkowski (11 factors), where both models succeed by round 2-3, and on PURE.
* Linear R² on PURE is very negative because round-2 fits have 44 recipes for 69
  factors. The rankings it produces are still useful (Spearman 0.60 vs 0.63).
* Wall time: XGBoost's 200-draw hyperparameter search costs 15-40 CPU-minutes per
  campaign; linear takes seconds.

Caveats:

* **Three seeds.** No significance claims; the seed-to-seed spread in "round found" is
  as large as most of the gaps.
* **Finite pools make this easier than a real campaign.** With half of each pool
  measured by round 10, every model reaches the best recipe eventually, and yields fall
  in late rounds as the good recipes run out (clearest on Borkowski). Use "round found"
  and AUC, not the final yield.
* **The pools were themselves collected by active learning.** CETCH and Borkowski
  recipes after day 1 were chosen by an optimiser, so they are concentrated in good
  regions. A replay therefore cannot show how a model explores outside what was measured;
  that is what the synthetic benchmark is for.
