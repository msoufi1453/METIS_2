# METIS benchmark: runbook

Everything runs from the repository root in the pinned environment:

```bash
uv sync --extra tabpfn --extra cpu      # CPU machines
uv sync --extra tabpfn --extra cu126    # NVIDIA GPU up to RTX 40 series (cu128 for RTX 50)
```

The protocol, metrics and statistics are fixed in [PREREGISTRATION.md](PREREGISTRATION.md).

## 0. Guards

```bash
uv run --extra tabpfn --extra cpu pytest
```

* `tests/test_regression_legacy.py`: one seeded METIS round reproduces the v1 output byte
  for byte (design, volumes, tuning, `bayesian_optimization`, `batch_ucb`, Echo files).
* `tests/test_notebook_compat.py`: every `utils` call in the 10 notebooks binds to the shim.
* `tests/test_synthetic.py`: every landscape's dynamic-programming maximum equals brute force.
* `tests/test_frozen.py`: metric and statistics definitions match the preregistered hashes.

## 1. Oracles and calibration (Phase 1)

```bash
uv run --extra tabpfn --extra cpu python -m metis_bench oracles      # gold regressors, cached in artifacts/oracles/
uv run --extra tabpfn --extra cpu python -m metis_bench calibrate    # real + synthetic descriptors, suite selection
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
uv run --extra tabpfn --extra cpu python -m metis_bench run    experiments/phase2_surrogates.yaml --jobs 8
uv run --extra tabpfn --extra cpu python -m metis_bench report experiments/phase2_surrogates.yaml
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

(prefix each line with `uv run --extra tabpfn --extra cpu python -m metis_bench`).

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

Per-campaign cost (estimates; only the pilot settings were timed): the full incumbent
(RandomizedSearchCV, 200 draws x 5 folds every round) about 8 CPU-minutes, XGBoost with
fantasy batches about 16, the GP about 1.5, TabPFN on CPU about 40 (single-threaded). The
confirmatory design after amendment 3 (20 seeds, 56 test problems, shared identical
campaigns, a smaller tuning grid, a 30-problem realistic-noise ablation) is roughly
2,700 laptop-core-hours, plus the TabPFN arm (1,120 campaigns) on a GPU.

Time one full-fidelity incumbent campaign on the target hardware before trusting these
numbers: everything else scales from it.

**Identical campaigns run once.** A campaign's result depends only on its resolved arm,
problem, seed and settings, never on the experiment or the arm's name. Results are
stored under `results/_shared/` and copied into every experiment that needs the same
campaign (for example the incumbent, which appears in four experiment files). Bump
`SHARED_VERSION` in `metis_bench/campaign.py` if campaign logic changes, so stale results
are not reused.

**TabPFN on a separate GPU machine.** The TabPFN arm is the only one that uses a GPU.
Run it alone there and everything else on CPU workers, then copy the result folders
together before `report`:

```bash
# GPU machine (NVIDIA up to RTX 40 series; use cu128 for RTX 50 series)
uv sync --extra tabpfn --extra cu126
uv run --extra tabpfn --extra cu126 python -m metis_bench oracles
uv run --extra tabpfn --extra cu126 python -m metis_bench run experiments/phase2_surrogates.yaml --arms surr_tabpfn --jobs 2

# CPU workers
uv run --extra tabpfn --extra cpu python -m metis_bench run experiments/phase2_surrogates.yaml --skip-arms surr_tabpfn --jobs 8

# then copy results/phase2_surrogates/ from both machines into one folder and run `report`
```

Check the GPU is used: `uv run --extra tabpfn --extra cu126 python -c "import torch; print(torch.cuda.is_available())"`
should print `True`. Several TabPFN workers can share one GPU if its memory allows (start
with `--jobs 2` and watch `nvidia-smi`). If TabPFN wins Phase 2, the Phase 3 arms that
use it need the GPU machine too.

## 5. Extending

* New surrogate: subclass `metis.surrogates.Surrogate` (`fit(X, y, noise_var=None)`,
  `predict(X) -> Prediction(mean, std)`) and register it in `metis_bench/arms.py`.
* New selector: subclass `metis.acquisition.BatchSelector` (`select(ctx, k)`).
* New landscape: add a builder to `metis_bench/synthetic/suite.py` that returns
  components of lookup-table terms over disjoint dimension groups, so the exact maximum
  stays computable, and add a brute-force case to `tests/test_synthetic.py`.
