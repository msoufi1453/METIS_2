#!/usr/bin/env bash
# Phase 2 -> decision -> Phase 3, in order. Each step needs the previous step's outputs.
#
#   scripts/run_pipeline.sh            # confirmatory (long: use a cluster, see docs/BENCHMARK.md)
#   PILOT=1 JOBS=7 scripts/run_pipeline.sh   # machinery check on a laptop
#
# Finished campaigns are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-8}
FLAG=""
SUFFIX=""
if [[ "${PILOT:-0}" == "1" ]]; then FLAG="--pilot"; SUFFIX="-pilot"; fi
# TORCH=cpu (default), cu126 or cu128 picks the PyTorch build for the TabPFN arm.
bench() { uv run --extra tabpfn --extra "${TORCH:-cpu}" python -W ignore -m metis_bench "$@"; }

# TabPFN arms use TabPFN's own schedule, tuned beforehand on the replay data with TabPFN-3.5
# on a GPU (experiments/tabpfn_schedule_tune.yaml, amendment 4). Pilots fall back to v1.
if [[ "${PILOT:-0}" != "1" && ! -f experiments/tuned/tabpfn_schedule.json ]]; then
  echo "experiments/tuned/tabpfn_schedule.json is missing: run the TabPFN schedule tuning first (docs/REPLAY.md)" >&2
  exit 1
fi

for exp in phase2_init phase2_import phase2_surrogates phase2_batch; do
  bench run "experiments/$exp.yaml" $FLAG --jobs "$JOBS"
  bench report "experiments/$exp.yaml" $FLAG
done

DECIDE_FLAGS="$FLAG"
# Pilots have too few seeds for significance; pick by mean so Phase 3 has something to run.
if [[ "${PILOT:-0}" == "1" ]]; then DECIDE_FLAGS="$FLAG --ignore-significance"; fi
bench decide $DECIDE_FLAGS \
  "results/phase2_init$SUFFIX/report" "results/phase2_surrogates$SUFFIX/report" "results/phase2_batch$SUFFIX/report"

bench tune experiments/phase3_schedule_tune.yaml $FLAG --jobs "$JOBS"
bench run experiments/phase3_schedule.yaml $FLAG --jobs "$JOBS"
bench report experiments/phase3_schedule.yaml $FLAG

bench tune experiments/phase3_hparams_tune.yaml $FLAG --jobs "$JOBS"
bench run experiments/phase3_hparams.yaml $FLAG --jobs "$JOBS"
bench report experiments/phase3_hparams.yaml $FLAG

bench run experiments/phase3_ablation.yaml $FLAG --jobs "$JOBS"
bench report experiments/phase3_ablation.yaml $FLAG
