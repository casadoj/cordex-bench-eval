#!/bin/bash
#SBATCH --job-name=cordex-dist-cc-prithvi
#SBATCH --output=logs/dist-cc-%j.out
#SBATCH --error=logs/dist-cc-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cordex-ml-benchmark

set -euo pipefail

# When running under SLURM, BASH_SOURCE[0] resolves to the spool copy of the
# script, not the original path.  SLURM_SUBMIT_DIR is always the directory
# from which sbatch was called (i.e. the diagnostics/ folder), so prefer it.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

export PATH="$HOME/.local/bin:$PATH"
export PIXI_HOME="${PIXI_HOME:-/lustre/gmeteo/WORK/rampaln/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/lustre/gmeteo/WORK/rampaln/.pixi/cache}"
export PYTHONUNBUFFERED=1

PIXI_PROJECT="${PIXI_PROJECT:-/lustre/gmeteo/WORK/rampaln/cordex-ml-bench}"
TRUTH_ROOT="${TRUTH_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/target}"
PRED_ROOT="${PRED_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/staging}"
CATALOG_PATH="${CATALOG_PATH:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/catalog/catalog.json}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/diagnostics/outputs}"

cd "$PIXI_PROJECT"
echo "PIXI_HOME=$PIXI_HOME"
pixi info

#pixi run python "$SCRIPT_DIR/compute_diagnostics_v2.py" \
#    --truth-root "$TRUTH_ROOT" \
#    --pred-root  "$PRED_ROOT" \
#    --out-root   "$OUT_ROOT" \
#    "$@"
#pixi run python "$SCRIPT_DIR/compute_distribution_cc_diagnostics.py" \
#    --truth-root "$TRUTH_ROOT" \
#    --pred-root "$PRED_ROOT" \
#    --catalog-path "$CATALOG_PATH" \
#    --out-root "$OUT_ROOT" \
#    "$@"
pixi run python "$SCRIPT_DIR/compute_diagnostic_single_model.py" \
    --truth-root "$TRUTH_ROOT" \
    --pred-root  "$PRED_ROOT" \
    --out-root   "$OUT_ROOT" \
    "$@"
