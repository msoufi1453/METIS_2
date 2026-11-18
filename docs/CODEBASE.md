# METIS codebase guide

What every part of the repository does, how the parts connect, and where to look when
something breaks or is slow. Docstrings in the code give the details of each function;
this page is the map.

## 1. The repository at a glance

| Path | What it is | Touch it when |
| --- | --- | --- |
| `metis/` | The library: design space, models ("surrogates"), scoring ("acquisition"), batch rules, pools, hill-climbing, the v1 notebook API | changing how recipes are chosen |
| `metis/campaign/` | Campaigns in a few lines; the engine of the app (variables, liquid handling, rounds, workbook, pipetting, analysis) | changing what the app computes |
| `metis/idot/` | Unmodified copy of github.com/msoufi1453/idot-protocol-builder (iDOT `.idotx` files) | never here: change the original repo and copy it over |
| `gui/` | The Streamlit app; it only draws what `metis.campaign` computes | changing screens or controls |
| `metis_bench/` | The benchmark: oracles (true yields), noise, campaign runner, metrics, statistics, reports | running or changing experiments |
| `experiments/*.yaml` | One file per experiment: problems, arms, seeds, settings | defining a new comparison |
| `experiments/tuned/` | Tuned settings other runs read (`tabpfn_schedule.json`: TabPFN's schedule, slope 4.5) | after a tuning run |
| `tests/` | pytest suite (see section 8) | always, before committing |
| `docs/` | `APP.md` (app guide), `BENCHMARK.md` (runbook), `PREREGISTRATION.md` (frozen protocol and amendments), `REPLAY.md`, `MODULES.md`, `PILOT_RESULTS.md`, `TODO.md` | |
| `data/replay/` | The four measured datasets (PURE, Borkowski, CETCH yield and efficiency) | |
| `suite/` | The 90 synthetic test functions (`suite.json`), written by `metis_bench calibrate` | |
| `artifacts/oracles/` | Cached gold-regressor oracles (committed so every machine uses the same ones) | |
| `results/` | Campaign results; ignored by git except `results/replay_small/` and the figures in `results/tabpfn_modules/replay/` | |
| `scripts/` | `run_pipeline.ps1` / `.sh` (whole benchmark on one machine), `slurm_tabpfn_modules.sh` (cluster job) | |
| `Examples/`, `*.ipynb`, `utils.py` | The original METIS notebooks and data; `utils.py` re-exports the v1 API from `metis` so they still run | |

Two checkouts exist on the laptop: `METIS` (branch `benchmark`) and `METIS_2` (branch
`main`, pushed to GitHub, pulled by the GPU laptop). New work goes into `METIS_2`.

## 2. Environment and everyday commands

Python 3.11, every package pinned in `uv.lock` (the byte-for-byte regression test depends
on it). Extras pick the optional parts:

| Extra | Gives | Notes |
| --- | --- | --- |
| `tabpfn35` | TabPFN-3.5 (package 9.2.0) | used by the app and every benchmark TabPFN arm; licence once, or `TABPFN_TOKEN` |
| `tabpfn` | TabPFN v2 (2.0.9) | old configurations only; conflicts with `tabpfn35` |
| `cpu` / `cu126` / `cu128` | PyTorch build | one of them; `cu128` for RTX 50 GPUs |
| `gui` | Streamlit 1.45.1 | the app |

```bash
uv sync --extra tabpfn35 --extra cpu --extra gui          # this laptop
.venv/Scripts/python -m streamlit run gui/app.py           # the app (localhost only, .streamlit/config.toml)
.venv/Scripts/python -m pytest                             # all tests
.venv/Scripts/python -m metis_bench run experiments/replay_small.yaml --jobs 6
```

On OneDrive folders use `uv sync --link-mode copy` (hard links fail there).

## 3. How one round is chosen (the core loop)

Everything, in the app and in the benchmark, follows the same steps:

1. **Encode** the measured recipes as rows of numbers (`metis.space.DesignSpace` in the
   benchmark, `metis.campaign.RecipeSpace` in the app) and turn them into model inputs
   (concentrations as numbers, categories one-hot).
2. **Fit** the surrogate to all results so far (`metis.surrogates`). It returns a mean and
   an uncertainty (`std`) for any recipe.
3. **Draw a pool** of candidate recipes that fit the pipettable volume and were not measured
   (`metis.pools`; the app uses `RecipeSpace.sample`). The model can only pick from it.
4. **Score** each candidate: UCB = `exploitation * mean + exploration * std`
   (`metis.acquisition.UCB`). The exploitation share comes from a schedule.
5. Optional: **hill-climb** (`metis.hillclimb`): from the best-scoring candidates, move one
   variable one step at a time while the score improves; the end points join the pool.
6. **Select the batch** (`metis.acquisition.selectors`, `metis.campaign.batch`): top-k, or
   a rule that spreads the batch out (distance penalty, fantasy batch, both).
7. **Measure** (the lab, or an oracle plus noise in the benchmark) and repeat.

`metis.optimize.suggest` runs steps 2, 4 and 6 for given candidates; everything else
calls it.

## 4. The `metis` library

| Module | Key contents | Remember |
| --- | --- | --- |
| `space.py` | `Factor` (levels, droplet cost, alternatives, log scale), `DesignSpace` (index rows, `features`, `embedding`, `feasible`, `sample`, `neighbors`) | a recipe is a row of level indices; `keys()` packs rows into bytes (max 255 levels per factor) |
| `surrogates/` | `Prediction(mean, std)`; `XGBEnsemble` (METIS v1), `RegressorEnsemble`, `GPSurrogate`, `TabPFNSurrogate` | every model: `fit(X, y, noise_var=None)`, `predict(X)`, `predict_interval`, `refit`, `clone` |
| `acquisition/functions.py` | `UCB`, `ExpectedImprovement` | higher score = better |
| `acquisition/selectors.py` | `TopK` (v1), `SequentialFantasy`, `SoftPenalty`, `LocalPenalization`, `GPBUCB`, `RandomSelector` | all take a `SelectionContext` and return a `Selection` |
| `pools.py` | `RandomPool` (v1), `LocalPool` (half mutations of the 8 best) | the pool caps what any model can find (docs/PILOT_RESULTS.md) |
| `hillclimb.py` | `climb()` | one model call per step for all climbers; caches every scored recipe |
| `schedules.py` | exploration schedules; `ExploitRampSchedule(slope)` is TabPFN's tuned one (slope 4.5) | round 2 = first model-guided round |
| `initialization.py` | random, Sobol, maximin Latin hypercube, fill-around-imported-data | |
| `design.py`, `io.py`, `echo.py`, `compat.py`, `optimize.bayesian_optimization` | the v1 notebook functions, unchanged in behaviour | `tests/test_regression_legacy.py` checks them byte for byte |

**The models.**
- **`XGBEnsemble`:** reruns a 200-draw x 5-fold hyperparameter search every fit, then keeps
  the 20 best settings as members. The spread between members is the uncertainty, and it
  is far too narrow: its 80% intervals covered 30-41% of held-out values in the replay.
- **`TabPFNSurrogate`:** fitting only stores the data and prediction does the work. The
  std comes from TabPFN's predictive distribution and includes measurement noise; its 80%
  intervals covered 88-93% in the replay.
- **`GPSurrogate`:** reports the latent std (noise removed).

## 5. `metis.campaign`: the app's engine

```python
campaign = Campaign(variables, target='Yield', liquid_handling=LiquidHandling(10))
batch = campaign.propose(10)      # Latin hypercube without data, model-guided with >= 2 results
campaign.add_round(batch, replicates=2)
campaign.save('round1.xlsx')      # fill in results, then Campaign.load(...) and propose again
```

| Module | Role |
| --- | --- |
| `variables.py` | `Continuous`, `Discrete` (1-100 steps; one step = fixed component), `Categorical` (with optional pipetted volume); validated on construction (`VariableError`) |
| `liquid.py` | `LiquidHandling`: one droplet of stock `S` adds `S * d / (1000 * V)` to a `V` uL sample (`d` nL droplets, default 4); `check()` returns `Issue`s (errors block a round) |
| `space.py` | `RecipeSpace`: one axis per variable (`_Interval`, `_DropletGrid`, `_Levels`, `_Categories`); `sample`, `lhs`, `snap`, `neighbors`, `features`, `embedding` |
| `campaign.py` | `Campaign` (data table, rounds, propose, fit, predict, workbook I/O), `Proposal` (a round's recipes and predictions; `.climb()`), `Fitted` |
| `models.py` | the model registry: only `tabpfn35` (8 estimators, device auto; 10,000 candidates with a GPU, 1,000 without); `register_model` adds others |
| `batch.py` | the batch modules: distance penalty (radius 1-3 x the candidates' typical spacing), fantasy, both together |
| `analysis.py` | PCA map, importance (R2 drop when a variable is shuffled), pairwise interactions, progress |
| `workbook.py` | the Excel file: Experiments, Instructions, Metadata, State (JSON, read back on upload) |
| `pipetting.py` | wells (96 per plate, `1A` = column 1 row A, filled by column), pipetting sheet (exact Decimal uL), iDOT protocol |
| `day0.py` | reading data that predates the campaign, and guessing variables from it |
| `tutorial.py` | the Borkowski example and its simulated lab (distance-weighted mean of the 5 nearest measured recipes) |
| `columns.py` | column names (Sample ID, Round, Data point, Replicate, Plate, Well, prediction columns) |

**Rules worth knowing.**
- **Rounds and rows.** Round 0 is day-0 data; generated rounds count from 1. Every
  measurement is its own row, and replicates share a Sample ID.
- **Exploitation default.** 0.5 in the first model-guided round, then 1.0 (the schedule
  tuned on the replay data, amendment 4).
- **Reproducible rounds.** Round `r` draws its random numbers from `[seed, r]`, so the same
  workbook gives the same proposal.
- **Rounds the app refuses.** It blocks a round if fewer than 1% of random recipes fit the
  sample, or if a liquid-handling check returns an error.
- **The workbook State sheet.** It is the source of truth on upload. Editing it breaks
  loading (`WorkbookError`), and changing its layout needs `FORMAT` in `campaign.py`
  raised.
- **Recipe identity.** Recipes are compared in single precision, and anything compared
  after an Excel round trip must be rounded (Excel keeps about 15 digits).

## 6. The app (`gui/`)

| File | Screen or role |
| --- | --- |
| `app.py` | entry point: one run = init state, take uploads, draw the screen |
| `home.py` | three buttons: new campaign, continue (upload workbook), tutorial |
| `setup_page.py` + `form.py` | the set-up form (variables, liquid handling, strategy); with data, four analysis tabs next to it |
| `round_page.py` | the proposed recipes (editable; edits are snapped and re-predicted), pipetting, map, downloads |
| `insight.py` + `charts.py` | PCA map, progress, importance, interactions (Altair) |
| `state.py` | session state: form fields `f_*`, variable rows `row<n>_*`, the round in progress |
| `texts.py` | the long texts and the tutorial's instructions |

**Streamlit gotchas (each has caused a bug).**
- **Lost widget values.** A widget's value is dropped when it is not drawn, so `state.init`
  re-assigns every form key on every run.
- **Tabs jumping back.** The open tab resets when anything above `st.tabs` appears or
  disappears, so conditional messages go into fixed `st.empty()` slots.
- **Generate.** Right after "Generate", the set-up and round screens are drawn in the same
  run, so their widget keys must differ.
- **Click selection.** It only works on single-layer charts, so the PCA map draws
  climbing paths as dots.
- **Tests and uploads.** Tests cannot drive uploads or the data editor: uploads go through
  `inbox_workbook` / `inbox_day0`, and edits through `round_page.apply_edit`.

## 7. The benchmark (`metis_bench`)

**Flow:** experiment YAML -> `experiments.tasks()` (problem x arm x seed) ->
`campaign.run_tasks()` -> `run_campaign()` -> one JSON per campaign ->
`report` / `replay` figures and tables.

| Command (`python -m metis_bench ...`) | Does |
| --- | --- |
| `oracles` | build and cache the gold-regressor oracles |
| `calibrate` | describe synthetic and real landscapes, select the 90-function suite |
| `run <yaml> [--jobs N --gpu-jobs M --arms a,b --skip-arms c --shard i/n --pilot]` | run the campaigns (resumable: finished files are skipped) |
| `report <yaml>` | the preregistered statistics report |
| `replay <yaml>` | figures for measured-data runs (or the TabPFN module report) |
| `tune <yaml>`, `decide <reports>` | Phase 3 tuning, the Phase 2 decision rule |
| `tasks <yaml>`, `summary` | list task files; one overview of all reports |

| Module | Role |
| --- | --- |
| `arms.py` | an arm is a dict (surrogate, acquisition, selector, schedule, pool, init, optional modules); missing keys take METIS v1 defaults; a TabPFN arm without a schedule takes the tuned one |
| `campaign.py` | `run_campaign`, `run_tasks` (CPU pool plus a separate GPU pool), shared results (`results/_shared`, `SHARED_VERSION`), `code_hash` provenance |
| `problems.py`, `oracles/` | problem = oracle x noise; oracles: synthetic functions (exact optimum), gold XGBoost models of Borkowski/PURE, measured data (replay), cell-growth grid |
| `noise.py` | triplicate wells, heteroscedastic noise, outliers, plate effects; presets `realistic`, `homoscedastic`, `high`, `none` |
| `metrics.py`, `stats.py` | **frozen by hash** (`FROZEN.json`, `tests/test_frozen.py`): any change, even a comment, needs a preregistration amendment |
| `report.py`, `replay.py`, `modules_report.py`, `summary.py` | tables and figures |
| `tuning.py`, `ablation.py`, `calibration.py`, `descriptors.py`, `synthetic/` | Phase 3, ablation, suite construction |

**Pairing.** A seed fixes the initial design, the imported data and every noise draw, so
two arms with the same seed differ only through the recipes they choose.

**Shared campaigns.** Identical campaigns (same resolved arm, problem, seed and settings)
run once and are copied to every experiment that needs them. Raise `SHARED_VERSION` when
campaign logic changes results.

**GPU guard.** TabPFN tasks stop with exit code 2 if PyTorch sees no GPU
(`--allow-cpu-tabpfn` overrides).

## 8. Tests

| File | Guards |
| --- | --- |
| `test_regression_legacy.py` | one seeded v1 round reproduces the original output byte for byte |
| `test_notebook_compat.py` | every `utils` call in the notebooks still binds |
| `test_frozen.py` | metrics and statistics unchanged |
| `test_campaign_api.py`, `test_gui.py` | the campaign package and the app (fast stand-in model, `tests/helpers.py`) |
| `test_space.py`, `test_interfaces.py`, `test_synthetic.py` | design space, plug-in interfaces, exact synthetic maxima |
| `test_campaign.py`, `test_modules.py`, `test_replay.py`, `test_gpu_runs.py` | benchmark runner, TabPFN modules, replay, GPU pools |
| `test_phase2_phase3.py`, `test_phase3_configs.py`, `test_amendment*.py`, `test_tabpfn_schedule.py`, `test_calibration.py`, `test_metrics_stats.py` | experiment files and amendments |

The marker `tabpfn` needs the real TabPFN; `pytest -m "not tabpfn"` skips it.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `needs the tabpfn 9.x package` | the v2 extra is installed | `uv sync --extra tabpfn35 --extra cpu` (or `cu128`) |
| TabPFN asks for a licence | first use on this machine | accept it once, or `setx TABPFN_TOKEN "..."` and reopen the shell (never in a repo file) |
| TabPFN run exits with code 2 | PyTorch sees no GPU | check `torch.cuda.is_available()`; reinstall with the right CUDA extra |
| `Failed to hardlink` / missing `RECORD` | repo inside OneDrive | move it out, or `uv sync --link-mode copy` |
| `test_regression_legacy` fails | both checks fail: the environment drifted; only the live check fails: a code change altered v1 behaviour | re-sync from `uv.lock`; or find the code change |
| `test_frozen` fails | `metrics.py` or `stats.py` edited | revert, or write an amendment and update `FROZEN.json` |
| Upload says the State sheet was changed | the workbook's State sheet was edited | use the downloaded workbook; edit only results (and recipe values) |
| "Fewer than 1 in 100 random recipes fit" | maxima too high for the pipettable volume | lower maxima or use stronger stocks (a larger sample alone does not help: the volumes grow with it) |
| Long runs stall overnight | the laptop slept (Modern Standby) | keep it plugged in and awake; runs resume on wake |
| No output in a redirected log | Python buffers stdout | `python -u`, or count result files |

## 10. Where the time goes (speed levers)

| Step | Cost | Lever |
| --- | --- | --- |
| XGBoost fit | 200 settings x 5 folds every round: 17-28 min per 10-round campaign on this laptop, whatever the batch size | fewer draws (`n_iter`), more CPU cores (`--jobs`) |
| TabPFN-3.5 predict | GPU 0.5-0.8 s per replay round; this CPU about 38 s per 1,000 candidates (8 estimators), 4 s with 1 | GPU; fewer candidates; `n_estimators` (changes the model) |
| Fantasy batch | one refit and one prediction per pick | use with a GPU, or the distance penalty instead |
| Hill-climbing | one model call per step for all climbers (up to 20 steps) | `n_starts`, `max_steps` |
| Importance / interactions (app) | one model call over (rows x variables x repeats) or (pairs x variable pairs) recipes | lighter settings without a GPU (already automatic) |
| Pool size | linear in candidates | 1,000 on CPU, 10,000 on GPU (app defaults) |
