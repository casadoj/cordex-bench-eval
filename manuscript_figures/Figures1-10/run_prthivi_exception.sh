#!/bin/bash
#SBATCH --job-name=cordex-combined
#SBATCH --output=logs/cordex-combined-%j.out
#SBATCH --error=logs/cordex-combined-%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cordex-ml-benchmark

# ============================================================================
#  CORDEX-ML-Bench :: combined region plots (climatology + daily + cc_signal)
# ============================================================================

set -euo pipefail
mkdir -p logs
echo "=== Job started on $(hostname) at $(date) ==="

# --- pixi ---
export PATH="$HOME/.local/bin:$PATH"
export PIXI_HOME=/lustre/gmeteo/WORK/rampaln/.pixi
export PIXI_CACHE_DIR=/lustre/gmeteo/WORK/rampaln/.pixi/cache

PIXI_PROJECT=/lustre/gmeteo/WORK/rampaln/cordex-ml-bench
cd "$PIXI_PROJECT"
pixi add python=3.11 xarray dask netcdf4 zarr matplotlib numpy cartopy pandas >/dev/null

# --- paths ---
SCRIPTS_DIR="/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/manuscript_figures/Figures1-10"
TRUTH_ROOT="${TRUTH_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/target}"
PRED_ROOT="${PRED_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/staging}"
OUT_ROOT="${OUT_ROOT:-/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/manuscript_figures/Figures1-10/plots}"

# --- knobs ---
SEED="${SEED:-42}"
POOL="${POOL:-10}"
N_SAMPLE="${N_SAMPLE:-3}"
BOT_POOL="${BOT_POOL:-15}"
BOT_EXCLUDE="${BOT_EXCLUDE:-3}"

VARIABLES=(tasmax)
EXPERIMENTS=(ESD_pseudo_reality Emulator_hist_future)

COMMON_ARGS=(
    --truth-root "$TRUTH_ROOT"
    --pred-root  "$PRED_ROOT"
    --out-dir    "$OUT_ROOT/combined"
    --seed "$SEED"
    --pool "$POOL"
    --n-sample "$N_SAMPLE"
    --bot-pool "$BOT_POOL"
    --bot-exclude "$BOT_EXCLUDE"
)

SCRIPT="$SCRIPTS_DIR/combined_regions_updated_prithvi_exception.py"


# ---------------------------------------------------------------------------
# 2) Daily (ranks 1, 2, 3)
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  Daily snapshots (rank 1-3)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 3) Climate-change signal (mid + end century; pr also in relative %)
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  cc_signal  (mid_century + end_century)"
echo "============================================================"

for var in "${VARIABLES[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    for future in mid_century end_century; do

      echo; echo ">>> combined  $var / $exp / cc_signal / $future"
      pixi run python "$SCRIPT" \
          --variable "$var" \
          --experiment "$exp" \
          --mode cc_signal \
          --future-period "$future" \
          "${COMMON_ARGS[@]}"

      # pr only: relative (%) version
      if [[ "$var" == "pr" ]]; then
        echo; echo ">>> combined  $var / $exp / cc_signal / $future (relative %)"
        pixi run python "$SCRIPT" \
            --variable "$var" \
            --experiment "$exp" \
            --mode cc_signal \
            --future-period "$future" \
            --relative \
            "${COMMON_ARGS[@]}"
      fi

    done
  done
done

echo
echo "=== Job finished at $(date) ==="
echo "Outputs under: $OUT_ROOT/combined"