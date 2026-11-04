#!/usr/bin/env bash
# TabPFN-3.5 module tests (docs/MODULES.md) on a SLURM GPU cluster: one GPU, about 1-3 hours.
#
# Once, on a login node (it has internet; compute nodes often do not):
#   curl -LsSf https://astral.sh/uv/install.sh | sh            # if uv is missing
#   uv sync --extra tabpfn35 --extra cu126                      # cu128 for Blackwell GPUs (RTX 50, B200)
#   copy the cell-growth spreadsheet to data/external/cell_growth_Data_Data.xlsx (data/README.md)
#   export TABPFN_TOKEN=...                                     # your token; never put it in the repository
#   TORCH=cu126 scripts/slurm_tabpfn_modules.sh --download-only # caches the TabPFN-3.5 weights
# Then submit from the repository folder (the job inherits TABPFN_TOKEN from this shell):
#   sbatch scripts/slurm_tabpfn_modules.sh
# Add the partition or account your cluster requires, e.g. sbatch --partition=gpu --account=abc123 ...
#
# Finished campaigns are skipped, so a job that hits its time limit can simply be resubmitted.
#SBATCH --job-name=tabpfn-modules
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=tabpfn_modules_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

bench() { uv run --extra tabpfn35 --extra "${TORCH:-cu126}" python -W ignore -m metis_bench "$@"; }

if [[ "${1:-}" == "--download-only" ]]; then
  # One tiny fit and prediction downloads the weights into the TabPFN cache in $HOME.
  uv run --extra tabpfn35 --extra "${TORCH:-cu126}" python -c "
import numpy as np
from metis.surrogates import TabPFNSurrogate
s = TabPFNSurrogate(n_estimators=1, device='cpu', version='v3.5').fit(np.random.rand(10, 3), np.random.rand(10))
print('TabPFN-3.5 ready:', s.predict(np.random.rand(2, 3)).mean)"
  exit 0
fi

nvidia-smi
bench run experiments/tabpfn_modules.yaml --pilot --jobs 1    # a few minutes: stops early if anything is wrong
bench run experiments/tabpfn_modules.yaml --jobs 1
bench replay experiments/tabpfn_modules.yaml
echo "Figures and tables: results/tabpfn_modules/replay/"
