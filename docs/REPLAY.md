# Replay of the paper's measured-data experiments (exploratory)

A quick backup to the synthetic benchmark: run METIS against the published data, as the
METIS paper did for PURE and Borkowski (Fig. 1). This is **exploratory**. It is not part
of the preregistered study (docs/PREREGISTRATION.md), uses 3 seeds, and makes no
significance claims. The synthetic benchmark is unchanged.

## Set-up

| | |
|---|---|
| Oracle | the measured data. Only recipes that were measured can be chosen, and each returns its measured yield (mean of replicates; identical recipes merged by mean). No added noise. |
| Datasets | PURE (Matsuura et al. 2009, 1,094 recipes, 69 factors); Borkowski et al. 2020 (1,007 recipes, 11 factors); CETCH yield (5 days, 625 recipes) and CETCH efficiency (8 days, 1,000 recipes), 32 columns each. |
| Concatenated tables | `data/replay/<dataset>.csv` (all days in one file, with a `day` column; written by `metis_bench.oracles.measured.write_concatenated`). |
| Held-out set | a fixed 20% of recipes per dataset (split seed 41, as in the PURE notebook), never offered to the optimiser; used only to score R², Spearman and interval coverage after each fit. |
| Candidate pool | the other 80%. Every round scores all unmeasured pool recipes, so the 10,000-random-candidate generator does not apply. |
| Optimum | the best measured yield in the pool, so "found the optimum" is exact for the data. |
| Rounds | 10. Round 1 is random; rounds 2-10 are model-guided (UCB with the METIS v1 exploration weights 1.41, 1, 0.5; top-k batch). |
| Batch size | about pool/20, so 10 rounds measure half the pool: PURE 44, Borkowski 40, CETCH yield 25, CETCH efficiency 40. The paper's 80-100 per round would use up these pools before round 10. |
| Seeds | 3 (0, 1, 2). A seed fixes the round-1 recipes, shared by all models. |

Models:

| arm | shown as | surrogate |
|---|---|---|
| `incumbent` | XGBoost (METIS) | METIS v1: XGBoost ensemble, 200 random-search draws, 20 members |
| `linear` | Linear | ordinary least squares (`LinearRegression`), the paper's linear model. Its 20 copies are identical, so there is no uncertainty and UCB ranks by the mean. |
| `tabpfn35` | TabPFN-3.5 | TabPFN-3.5 (open weights, `tabpfn==9.2.0`), 8 estimators, uncertainty from its predictive distribution |
| `random` | Random | no model: a random batch each round (context: how easy is each pool?) |

## Running it

```bash
uv sync --extra tabpfn35 --extra cpu      # or cu126 (NVIDIA up to RTX 40) / cu128 (RTX 50)
python -m metis_bench run experiments/replay_paper.yaml --jobs 6
python -m metis_bench replay experiments/replay_paper.yaml
```

The arms can run on different machines (`--arms tabpfn35` on a GPU, `--skip-arms tabpfn35`
elsewhere); copying the `results/replay_paper/<dataset>/<arm>/` folders together before
`replay` is all the merging needed.

Outputs land in `results/replay_paper/replay/`:

| file | what it shows |
|---|---|
| `yields_by_round.png` | every recipe each model chose, by round: box plots with the recipes on top, the style of the METIS notebooks' per-day figures; the model comparison of the paper's Fig. 1b |
| `best_so_far.png` | best yield found so far, as a fraction of the pool's best |
| `model_accuracy.png` | R² and Spearman correlation on the held-out recipes after each round |
| `model_comparison.png` | one dot per seed: round the best recipe was found, final R², 80%-interval coverage |
| `summary.md` / `summary.csv` | the numbers, plus each model paired by seed against XGBoost |

`python -m metis_bench report experiments/replay_paper.yaml` also works, but with 3 seeds
its paired tests have almost no power.

### TabPFN-3.5 licence and Windows notes

The TabPFN-3.5 weights are under a **non-commercial licence**, and downloading them needs
a free Prior Labs account and a one-time licence acceptance. Either run this once in an
interactive terminal (a browser opens; the token is then cached in
`~/.cache/tabpfn/auth_token`):

```bash
.venv\Scripts\python.exe -c "from tabpfn import TabPFNRegressor; import numpy as np; TabPFNRegressor(device='cpu').fit(np.random.rand(20,3), np.random.rand(20))"
```

or accept the licence on https://ux.priorlabs.ai (Licenses tab), copy the API key from the
account page and set it in the shell before `run`: `$env:TABPFN_TOKEN="<key>"` in
PowerShell. Never put the key in a file inside the repository.

On Windows, keep the repository outside OneDrive. If `uv sync` fails with
`os error 396` (hard links refused), add `--link-mode copy`, or set `UV_LINK_MODE=copy`
for the user.

The synthetic benchmark uses TabPFN-3.5 too (amendment 5; `--extra tabpfn35`). The older
TabPFN v2 extra (`--extra tabpfn`, package 2.0.9) cannot be installed alongside it, and
`TabPFNSurrogate` refuses to run if the installed package does not match the arm's
`version`.

## TabPFN's own exploration schedule (amendment 4)

The results below show why TabPFN needs its own schedule: METIS's exploration weights
were set for XGBoost, whose uncertainty is far too small, so with the same weights the
honest-uncertainty TabPFN explores much more. `experiments/tabpfn_schedule_tune.yaml`
therefore tunes TabPFN-3.5's schedule on these four datasets; XGBoost keeps its own, and
the linear model has no uncertainty for a schedule to act on.

Candidates: 10 linear "exploitation ramps". Each round, exploitation = min(1, slope ×
(round − 1) / 9) and exploration = 1 − exploitation, and UCB scores
exploitation × prediction + exploration × uncertainty. Every ramp starts mostly exploring
and eases into exploitation:

| slope | 0.8 | 0.9 | 1.0 | 1.125 | 1.3 | 1.5 | 1.8 | 2.25 | 3.0 | 4.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| exploitation in round 2 | 9% | 10% | 11% | 13% | 14% | 17% | 20% | 25% | 33% | 50% |
| only exploits from round | never (80% at the end) | never (90%) | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3 |

The winner is the ramp with the lowest mean `auc_norm_regret` (how fast good recipes are
found; every arm finds the best recipe eventually here), datasets weighted equally, over
5 seeds. METIS's v1 schedule runs alongside as a reference but cannot win. That is 220
campaigns, about 30 minutes on an RTX 5090 laptop GPU (12 of them are reused from the
replay run if its results are on the machine):

```bash
uv sync --extra tabpfn35 --extra cu128
python -m metis_bench tune experiments/tabpfn_schedule_tune.yaml --jobs 1
```

It writes `experiments/tuned/tabpfn_schedule.json` and a report in
`results/tabpfn_schedule_tune/tuning/` (`tuning_report.md`, `tuning_curve.png`). Commit the
JSON file: from then on every TabPFN arm that sets no schedule uses it, on every machine,
including `surr_tabpfn` in the synthetic study. Confirmatory runs refuse to start without
it (pilots fall back to the v1 schedule and say so). The replay's own `tabpfn35` arm keeps
METIS's schedule explicitly, so the comparison below is unchanged.

Caveats for the tuned schedule: it is chosen on these replay pools, which were collected
by optimisers and are half measured by round 10, so they reward exploiting; on the
synthetic landscapes the best ramp may be slower. The report flags a winner at the edge
of the grid. The tuning data are also the source of the `borkowski` and `pure_pool`
oracles, so TabPFN's synthetic-study result on those two problems is not independent of
the tuning.

## Data note: CETCH `Yield_all_data.csv`

The repository's `Examples/CETCH/Data/Yield_optimization/Yield_all_data.csv` stacks the
day files by column position, but days 2-5 list the one-hot HEPES pH columns in a
different order from day 1. As a result, 500 of its 625 rows carry the wrong pH. The
replay concatenates the per-day `Results_<day>.csv` files by column name instead
(checked in `tests/test_replay.py`).

## Results (2026-10-28, all four models)

Figures: [recipes per round](replay/yields_by_round.png) (paper style),
[best so far](replay/best_so_far.png), [model accuracy](replay/model_accuracy.png),
[models side by side](replay/model_comparison.png). Full table, including standard
deviations and the paired comparison: [replay/summary.md](replay/summary.md).

Mean over 3 seeds. *Found best* counts seeds that measured the best recipe in the pool;
*round found* averages over those seeds. *80% coverage* is the share of held-out recipes
inside the model's 80% interval after the last round (0.80 = honest uncertainty).

| dataset | model | found best | round found | held-out R² | 80% coverage | time per campaign |
|---|---|---|---|---|---|---|
| PURE | XGBoost (METIS) | 3/3 | 5.7 | 0.36 | 0.30 | 19 min |
| PURE | Linear | 3/3 | 6.7 | −4.61 | none | 3 s |
| PURE | TabPFN-3.5 | 3/3 | 3.7 | 0.39 | 0.93 | 10 s |
| PURE | Random | 1/3 | 10 | – | – | 1 s |
| Borkowski | XGBoost (METIS) | 3/3 | 2.0 | 0.85 | 0.32 | 16 min |
| Borkowski | Linear | 3/3 | 2.3 | 0.80 | none | 4 s |
| Borkowski | TabPFN-3.5 | 3/3 | 4.3 | 0.93 | 0.88 | 6 s |
| Borkowski | Random | 2/3 | 3.0 | – | – | 1 s |
| CETCH yield | XGBoost (METIS) | 3/3 | 4.3 | 0.57 | 0.41 | 38 min |
| CETCH yield | Linear | 3/3 | 6.0 | 0.37 | none | 3 s |
| CETCH yield | TabPFN-3.5 | 3/3 | 4.0 | 0.68 | 0.92 | 6 s |
| CETCH yield | Random | 1/3 | 7 | – | – | 1 s |
| CETCH efficiency | XGBoost (METIS) | 3/3 | 3.3 | 0.56 | 0.40 | 37 min |
| CETCH efficiency | Linear | 3/3 | 4.3 | 0.33 | none | 20 s |
| CETCH efficiency | TabPFN-3.5 | 3/3 | 2.7 | 0.72 | 0.91 | 8 s |
| CETCH efficiency | Random | 0/3 | – | – | – | 1 s |

TabPFN-3.5 against XGBoost, paired by seed (same round-1 recipes):

| dataset | best found sooner / same / later | mean change in round found | change in held-out R² |
|---|---|---|---|
| PURE | 2 / 1 / 0 | −2.0 | +0.03 |
| Borkowski | 0 / 0 / 3 | +2.3 | +0.07 |
| CETCH yield | 2 / 0 / 1 | −0.3 | +0.11 |
| CETCH efficiency | 2 / 1 / 0 | −0.7 | +0.16 |
| all 12 pairs | 6 / 2 / 4 | −0.2 | +0.09 |

What it shows:

* **TabPFN-3.5 is the most accurate model on every dataset.** It has the highest held-out
  R² and Spearman correlation after round 10 everywhere, with the largest margins on the
  two CETCH datasets, and it leads from the first guided round (one random batch of
  data) on all four by R².
* **Better predictions did not reliably mean finding the best recipe sooner.** Paired by
  seed, TabPFN-3.5 found it sooner than XGBoost in 6 of 12 runs, at the same round in 2
  and later in 4: about 2 rounds sooner on PURE, slightly sooner on the CETCH datasets,
  and about 2 rounds later on Borkowski in all three seeds.
* **The difference comes from uncertainty, not accuracy.** METIS ranks recipes by
  prediction + exploration weight × uncertainty, with weights that were set for
  XGBoost. XGBoost's ensemble is badly overconfident: its 80% intervals contain only
  30-41% of held-out recipes, so the uncertainty term is small and it picks almost
  greedily. TabPFN-3.5's intervals are honest, if slightly cautious (88-93%), so the
  same weights make it explore. In Borkowski round 2, 31% of TabPFN-3.5's picks were
  below a quarter of the best yield (random: 38%; XGBoost: 0%), visible as the tall
  round-2 box in `yields_by_round.png`. On Borkowski (11 factors) a greedy model reaches
  the best recipe in round 2, so exploring costs about two rounds; on PURE (69 factors)
  it pays off.
* **Linear is the weakest model** (no uncertainty at all, so it is purely greedy), but
  still beats random. Random found the best recipe in only 4 of 12 runs; every model found
  it in all 12.
* **Speed:** TabPFN-3.5 took 6-10 s per campaign on an RTX 5090 laptop GPU (the very
  first run 100 s, including the weight download and GPU warm-up). XGBoost took 16-39 min
  per campaign on one CPU core, mostly its 200-draw hyperparameter search every round.
  Different hardware, so this is a practical comparison, not an algorithmic one.

What it means for the synthetic benchmark: comparing models under one fixed exploration
schedule mixes up how accurate a model is with how large its uncertainty is. The
preregistered surrogate comparison (Phase 2, family F2) runs every surrogate with
METIS's schedule, so it carries this confound. Tuning the exploration weight per
surrogate would separate the two; that would need a preregistration amendment.

Caveats:

* **Three seeds.** No significance claims; the seed-to-seed spread in "round found" is
  as large as most of the gaps.
* **Two machines.** The TabPFN-3.5 runs were made on a teammate's machine (Python
  3.11.18). Their result files carry no code version, since that machine had no
  command-line git. Their configuration fingerprints match this repository's exactly,
  their round-1 recipes are identical to the XGBoost runs with the same seeds, and every
  yield they recorded matches our copy of the data.
* **Finite pools make this easier than a real campaign.** With half of each pool
  measured by round 10, every model reaches the best recipe eventually, and yields fall
  in late rounds as the good recipes run out (clearest on Borkowski). Use "round found"
  and AUC, not the final yield.
* **The pools were themselves collected by active learning.** CETCH and Borkowski
  recipes after day 1 were chosen by an optimiser, so they are concentrated in good
  regions. A replay therefore cannot show how a model explores outside what was measured;
  that is what the synthetic benchmark is for.
