# Open tasks (not started)

Recorded 2026-10-18. Nothing here is implemented yet. Both tasks need a preregistration
amendment (docs/PREREGISTRATION.md, section 9) before any confirmatory run.

## Task 1: fantasised batches with TabPFN vs filtering a random pool

Question: does choosing the batch sequentially by fantasising (Ginsbourger et al. 2010:
pick the top point, pretend its outcome is the posterior mean or a constant "lie", refit,
re-score, pick the next) beat filtering a random pool by one UCB ranking, and does the
answer depend on the surrogate?

Arms (all else equal, v1 schedule, random init, realistic noise):

| arm | surrogate | batch rule |
| --- | --- | --- |
| `incumbent` (exists) | XGBoost ensemble | top-32 of a random pool |
| `tabpfn_topk` | TabPFN | top-32 of a random pool |
| `tabpfn_fantasy` | TabPFN | sequential fantasies (kriging believer) |
| `xgb_fantasy` (exists as `batch_fantasy`) | XGBoost ensemble | sequential fantasies |

The fantasy selector already exists (`metis.acquisition.SequentialFantasy`) and
refits through `Surrogate.refit`; TabPFN's `refit` is `fit`, so it should plug in.

Cost caveats to check before claiming TabPFN makes this "nearly free":
- Conditioning on an extra point needs no retraining, but each fantasy still needs a
  fresh forward pass over the whole pool. Pilot timing was about 4 CPU-minutes per round
  for about 2,300 predictions; 31 re-scorings of a 10,000-recipe pool is far more.
- TabPFN's `fit_mode="fit_with_cache"` may cache the context so only query rows cost.
  Measure whether it is usable with the fantasised rows appended, and on a GPU.
- If it is not cheap, score only a shortlist (for example the top 500 by UCB) at each
  fantasy step, and say so in the amendment.

Also decide: fantasy value (posterior mean vs a pessimistic constant) and whether
TabPFN's std, which includes measurement noise, should be used as is.

## Task 2: measure batch collapse in every experiment

The suspicion: top-k by a per-recipe score picks near-duplicates. Make it a measured
result instead of a complaint.

For every guided round of every campaign, record, for the 32 selected recipes and for
random 32-subsets of the same candidate pool (use about 100 draws for a distribution):
- the distribution of pairwise distances (embedding distance from `DesignSpace.embedding`),
- summary numbers: median pairwise distance, minimum pairwise distance, and the ratio
  selected / random.

Report these across rounds per arm (the plot that would show collapse), stratified by
landscape and dimension. Arms with no candidate pool (random search) report n/a.

Implementation notes:
- Logged next to the existing per-round surrogate metrics in `campaign.py`.
- `metrics.py` is frozen (hash in `FROZEN.json`). Put the diversity measures in a new
  module, list them in the preregistration as secondary diagnostics, and avoid touching
  `metrics.py`, so the frozen hash stays valid.
- Cheap: 32 x 32 distances per draw. Runs for every arm, including existing Phase 2 arms.
- Pilot first: rerun `phase2_batch` in pilot mode to see whether the effect is visible.
