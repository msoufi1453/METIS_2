# METIS benchmark: runbook

Everything runs from the repository root in the pinned environment:

```bash
uv sync --all-extras
```

The protocol, metrics and statistics are fixed in [PREREGISTRATION.md](PREREGISTRATION.md).

## 0. Guards

```bash
uv run --all-extras pytest
```

* `tests/test_regression_legacy.py`: one seeded METIS round reproduces the v1 output byte
  for byte (design, volumes, tuning, `bayesian_optimization`, `batch_ucb`, Echo files).
* `tests/test_notebook_compat.py`: every `utils` call in the 10 notebooks binds to the shim.
* `tests/test_synthetic.py`: every landscape's dynamic-programming maximum equals brute force.
* `tests/test_frozen.py`: metric and statistics definitions match the preregistered hashes.

## 1. Oracles and calibration (Phase 1)

```bash
uv run --all-extras python -m metis_bench oracles      # gold regressors, cached in artifacts/oracles/
uv run --all-extras python -m metis_bench calibrate    # real + synthetic descriptors, suite selection
```

`calibrate` writes `suite/`:

| File | Content |
| --- | --- |
| `suite.json` | the 90 functions (45 specs x 12/27 dims): spec, split, realism score, nearest real reference |
| `suite_descriptors.csv` | every descriptor for every function |
| `real_descriptors.csv` | Borkowski and PURE: raw, XGBoost and GP rows, plus the combined reference vectors |
| `descriptor_map.png/.csv` | 2D PCA map of all functions and both datasets |
| `screened_candidates.csv` | all screened candidates (for auditing the realism-biased selection) |

## 2. Experiments

Each experiment file in `experiments/` defines the problems, arms and comparison
families. The same commands serve every phase:

```bash
uv run --all-extras python -m metis_bench run    experiments/phase2_surrogates.yaml --jobs 8
uv run --all-extras python -m metis_bench report experiments/phase2_surrogates.yaml
```

* `--pilot` applies the file's `pilot` block (few problems and seeds, cheaper incumbent
  tuning) and writes to `results/<name>-pilot/`. Pilots check the machinery only.
* Runs are resumable: finished campaigns (`results/<name>/<problem>/<arm>/seed_XXX.json`)
  are skipped.
* Clusters: `--shard i/n` runs every n-th task starting at i, so an array job of n
  workers each runs `run <file> --shard $TASK_ID/n`. `tasks <file>` lists the task files.

Order (each step needs the previous one's output):

```bash
# Phase 2: the three interventions against the incumbent
run experiments/phase2_init.yaml;        report experiments/phase2_init.yaml
run experiments/phase2_import.yaml;      report experiments/phase2_import.yaml
run experiments/phase2_surrogates.yaml;  report experiments/phase2_surrogates.yaml
run experiments/phase2_batch.yaml;       report experiments/phase2_batch.yaml
decide results/phase2_init/report results/phase2_surrogates/report results/phase2_batch/report
#   -> artifacts/winners.json (preregistered decision rule, per intervention part)

# Phase 3: tuning on the dev split, confirmation on the test split, ablation
tune   experiments/phase3_schedule_tune.yaml   # -> artifacts/tuning/schedule.json
run    experiments/phase3_schedule.yaml;  report experiments/phase3_schedule.yaml
tune   experiments/phase3_hparams_tune.yaml    # -> artifacts/tuning/surrogate.json
run    experiments/phase3_hparams.yaml;   report experiments/phase3_hparams.yaml
run    experiments/phase3_ablation.yaml;  report experiments/phase3_ablation.yaml
```

(prefix each line with `uv run --all-extras python -m metis_bench`).

## 3. Reading a report

`results/<name>/report/report.md` holds, per comparison family:

* the primary-endpoint table: mean paired difference to the reference (negative =
  lower regret = better), Hodges-Lehmann estimate, hierarchical bootstrap CI, share of
  problems improved or worsened, Wilcoxon p-value and Holm-adjusted p-value;
* secondary endpoints (`family_<F>_<metric>.csv`);
* means by landscape class, dimension, noise preset and realism tercile;
* surrogate accuracy and interval calibration per round (`surrogate_quality.png`,
  `rounds.csv`);
* for the ablation: per-subset effects, main effects and interactions, and Shapley shares.

`runs.csv` has one row per campaign, labelled with the function's realism score, so any
result can be reweighted or filtered by realism.

## 4. Compute

One confirmatory campaign with the full incumbent (RandomizedSearchCV, 200 draws x 5 folds
per round) takes about 5 to 10 CPU-minutes. GP campaigns take about 1 minute. TabPFN on
CPU is the expensive arm: about 4 minutes per round single-threaded (320 training recipes,
2,300 predictions, 4 estimators), so about 40 minutes per campaign. Run it on GPUs
(`surrogate: {type: tabpfn, device: cuda}`). With 64 test-split problems x 50 seeds, each arm is roughly
3,200 campaigns, so Phase 2 alone is several thousand CPU-hours: run it on a cluster with
`--shard`. Pilots (`--pilot`) finish in about an hour on 8 cores.

## 5. Extending

* New surrogate: subclass `metis.surrogates.Surrogate` (`fit(X, y, noise_var=None)`,
  `predict(X) -> Prediction(mean, std)`) and register it in `metis_bench/arms.py`.
* New selector: subclass `metis.acquisition.BatchSelector` (`select(ctx, k)`).
* New landscape: add a builder to `metis_bench/synthetic/suite.py` that returns
  components of lookup-table terms over disjoint dimension groups, so the exact maximum
  stays computable, and add a brute-force case to `tests/test_synthetic.py`.
