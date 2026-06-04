#!/bin/bash
#SBATCH --job-name=combined-kde-shared-axis
#SBATCH --output=logs/combined-kde-shared-axis-%j.out
#SBATCH --error=logs/combined-kde-shared-axis-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cordex-ml-benchmark

set -euo pipefail

echo "[run_combined_regions_kde_shared_axis] START $(date)"

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/manuscript_figures/Figures1-10"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/combined_regions_kde_shared_axis.py"

mkdir -p "${SCRIPT_DIR}/logs"

export PATH="$HOME/.local/bin:$PATH"
export PIXI_HOME="${PIXI_HOME:-/lustre/gmeteo/WORK/rampaln/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/lustre/gmeteo/WORK/rampaln/.pixi/cache}"

TRUTH_ROOT="${TRUTH_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/target}"
PRED_ROOT="${PRED_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/staging}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/plots}"
SEED="${SEED:-42}"
POOL="${POOL:-10}"
N_SAMPLE="${N_SAMPLE:-3}"
BOT_POOL="${BOT_POOL:-15}"
BOT_EXCLUDE="${BOT_EXCLUDE:-3}"

#VARIABLES_LIST="${VARIABLES_LIST:-pr tasmax}"
VARIABLES_LIST="${VARIABLES_LIST:-tasmax}"
EXPERIMENTS_LIST="${EXPERIMENTS_LIST:-ESD_pseudo_reality Emulator_hist_future}"
read -r -a VARIABLES <<< "${VARIABLES_LIST}"
read -r -a EXPERIMENTS <<< "${EXPERIMENTS_LIST}"

cd "${REPO_ROOT}"
echo "[run_combined_regions_kde_shared_axis] Using script: ${PY_SCRIPT}"

for var in "${VARIABLES[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    echo "[run_combined_regions_kde_shared_axis] Running climatology: var=${var} exp=${exp}"
    pixi run python "${PY_SCRIPT}" \
      --variable "${var}" \
      --experiment "${exp}" \
      --mode climatology \
      --truth-root "${TRUTH_ROOT}" \
      --pred-root "${PRED_ROOT}" \
      --out-dir "${OUT_ROOT}/combined" \
      --seed "${SEED}" \
      --pool "${POOL}" \
      --n-sample "${N_SAMPLE}" \
      --bot-pool "${BOT_POOL}" \
      --bot-exclude "${BOT_EXCLUDE}"
  done
done

echo "[run_combined_regions_kde_shared_axis] END $(date)"
