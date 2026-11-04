# TabPFN module tests and the GUI

Exploratory tests (amendment 7 in [PREREGISTRATION.md](PREREGISTRATION.md)): three optional
modules, each compared one at a time with plain TabPFN-3.5. Experiment file:
[`experiments/tabpfn_modules.yaml`](../experiments/tabpfn_modules.yaml).

## What is compared

Every arm is TabPFN-3.5 (8 estimators, its tuned exploration schedule, amendment 4) with
one change:

| arm | change | what it targets |
|---|---|---|
| `control` | none: the 10 highest UCB scores | |
| `penalty_low` | soft distance penalty, radius = 1 x the dataset's spacing | batches of near-copies |
| `penalty_high` | the same, radius = 3 x the spacing | |
| `fantasy` | fantasy batch (kriging believer) | batches of near-copies |
| `hillclimb` | discrete hill-climbing from the 10 best pool candidates, at most 20 moves | the best recipes are missing from a random pool |

* **Soft distance penalty** (`metis.acquisition.SoftPenalty`). After each pick, every
  candidate's score (shifted to be positive) is multiplied by `1 - exp(-d^2 / 2r^2)`,
  with `d` its distance to the pick: 0 at the pick, 0.39 at `d = r`, 0.86 at `2r`, 0.99
  at `3r`. Distances are in the design-space embedding (each factor scaled to [0, 1], a
  change of category = distance 1). The radius is a multiple of the dataset's *spacing*,
  the median distance from a recipe to its nearest other recipe, so "low" and "high" mean
  the same on every dataset. Spacing: Borkowski 0.222, cell growth 1.000 (for most
  recipes the nearest one is the same medium with another strain, exactly 1 apart),
  synthetic 0.512.
* **Fantasy batch** (`metis.acquisition.SequentialFantasy`). Pick the best candidate,
  add it to the data with its *predicted* value, re-fit, re-score, pick the next. For
  TabPFN a re-fit is one more row of in-context data, not retraining.
* **Discrete hill-climbing** (`metis.hillclimb`). Score the pool, start from its 10 best
  candidates, and move each to its best-scoring neighbour while that improves the score,
  for at most 20 moves. A neighbour is one factor one level up or down, or another
  category; on the cell-growth grid it is the same medium with another strain, or the same
  strain in one of the 8 media closest in composition. Measured recipes are never
  proposed. The end points join the pool, and the batch is then the 10 best scores as
  usual. Every climb's move count is logged.

Each module can be limited to some rounds, e.g. a penalty from round 4 on:
`selector: {type: soft_penalty, radius_scale: 1.0, rounds: [4, 10]}`; the same `rounds`
key works for `sequential_fantasy` and `hillclimb`.

## Problems

| problem | recipes | pool each round | batch | hill-climbing |
|---|---|---|---|---|
| Borkowski (cell-free, 11 factors) | 1,007 measured | all unmeasured (about 800) | 10 (1%) | no: neighbours were mostly never measured |
| Cell growth (Aida & Ying 2024): 115 strains x 135 media | 13,944 measured, an exhaustive grid | 1,000 random unmeasured | 10 (0.07%) | yes |
| Synthetic, 12 factors (the pool-ceiling function of PILOT_RESULTS.md, noise-free) | 2.2e10 grid points, exact optimum | 10,000 random feasible | 10 | yes |

As in the replay, 20% of each measured dataset is held out for accuracy, and only measured
recipes can be chosen. The cell-growth pool is a random 1,000 of its ~11,000 candidates per
round, so that the best recipes are usually not in it and hill-climbing has room to help.
Ten rounds, three seeds (seed 0 keeps every round's scored pool for the PCA figures).

The cell-growth data is not in this repository (licence: data/README.md). Without it the
experiment stops with a message saying where to put the file.

## Running (GPU machine)

```powershell
cd C:\Users\maxho\METIS_2
# GitHub Desktop: Fetch origin, then Pull origin
uv sync --link-mode copy --extra tabpfn35 --extra cu128      # adds openpyxl, needed to read the cell-growth file
# save the cell-growth spreadsheet as data\external\cell_growth_Data_Data.xlsx
.\.venv\Scripts\python.exe -m pytest tests/test_modules.py -q
.\.venv\Scripts\python.exe -m metis_bench run experiments/tabpfn_modules.yaml --pilot --jobs 1    # about 10 min
.\.venv\Scripts\python.exe -m metis_bench run experiments/tabpfn_modules.yaml --jobs 1
.\.venv\Scripts\python.exe -m metis_bench replay experiments/tabpfn_modules.yaml
```

42 campaigns (hill-climbing runs on two of the three problems). Plain and penalty
campaigns take about a minute or two on an RTX 5090 laptop; fantasy and hill-climbing call
TabPFN several times per round (10 fits per round for fantasy). Expect roughly 1 to 3
hours in all. Finished campaigns are skipped on a re-run.

Not on a CPU: timed on a 4-core laptop CPU (i7-8665U), TabPFN-3.5 with 8 estimators
scores about 11 recipes per second (3 per second on cell growth, which has 162 inputs), so
the full test would take about 180 hours, most of it the fantasy arms.

**SLURM GPU cluster.** `scripts/slurm_tabpfn_modules.sh` runs the pilot, the full test and
the figures on one GPU; its header lists the one-time setup on a login node (environment,
data file, `TABPFN_TOKEN`, caching the weights, since compute nodes often have no
internet). Submit with `sbatch scripts/slurm_tabpfn_modules.sh`, adding the partition or
account the cluster requires.

Figures and tables land in `results/tabpfn_modules/replay/`, the only part of the results
that is committed (the raw results contain measured values from the cell-growth data).

## Figures

* `best_so_far.png`: best measured value so far as a fraction of the dataset's best,
  mean over seeds with a band of one standard error.
* `yields_by_round.png`: the measured value of every recipe each arm chose, per round.
* `batch_spread.png`: mean and smallest distance between the recipes of a batch, in units
  of the spacing. Low = clustered; this is where the penalty and fantasy should differ from
  the control.
* `hillclimb_steps.png`: moves each climb took to reach a local top of the score (dashed
  line: the cap of 20), with how many climbed recipes made it into the batch.
* `pca_<dataset>.png`: seed 0, rounds 2, 4, 6, 8 and 10, one row per arm. Every candidate
  of the round's pool on the dataset's first two principal components, coloured by its
  score percentile in that pool; grey: recipes measured before the round; rings: the batch;
  triangles: batch recipes found by hill-climbing. The PCA is fitted once per dataset, so
  all panels share axes.
* `summary.md`: per dataset and arm, and paired by seed against the control.

With three seeds these are exploratory: look for consistent direction across datasets,
not significance.

## The GUI

```powershell
uv sync --extra tabpfn35 --extra cpu --extra gui        # or cu128 instead of cpu on the GPU laptop
.\.venv\Scripts\python.exe -m streamlit run gui/app.py
```

It opens in the browser:

1. **Data**: upload a CSV or Excel table (one row per recipe, one column per factor, one
   result column), or pick an example (Borkowski, PURE, CETCH). Choose the result to
   maximise, the columns to ignore (IDs) and which columns are numeric or categorical
   factors. Rows without a result are candidates.
2. **Explore (PCA)**: the recipes on their first two principal components, coloured by
   the result; which factors spread the recipes out (PCA loadings, a property of the
   design); and which factors the model relies on (the drop in R^2 when a factor is
   shuffled).
3. **Next batch**: model (TabPFN-3.5, XGBoost as in METIS v1, Gaussian process, linear),
   round number and planned rounds (for the exploration schedule), batch size, candidates
   (new combinations of the observed levels, or the table's rows without a result), and
   the modules: batch rule (top scores, distance penalty low/high, fantasy batch),
   hill-climbing on or off, and the PCA map of the scored candidates. The batch can be
   downloaded as CSV.

The logic is in `gui/core.py` (tested in `tests/test_gui.py`, including a headless run of
the page); `gui/app.py` is the page. TabPFN-3.5 needs its licence accepted or
`TABPFN_TOKEN` set in the shell that starts the app.
