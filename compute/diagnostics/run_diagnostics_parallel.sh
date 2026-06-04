#!/bin/bash
#SBATCH --job-name=pritivi
#SBATCH --output=logs/dist-cc-%j.out
#SBATCH --error=logs/dist-cc-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cordex-ml-benchmark
mkdir -p logs
set -euo pipefail
# "DeepESD-IFCAv1"
# "DeepESD-IFCAv1-orog"
# "DeepESDcrps_IFCAv2"
# "DeepESDcrps_IFCAv2-orog"
# "ViT-IFCAv1"
# "ViT-IFCAv1-orog"
# "FlowMatching-v1"
# "FlowMatching-v1-orog"
# "UiBCorrDiff-orog"
# "RCMGEM-mv-orog"
# "CorrDiff-TW1_NTNU-CWA"
# "CorrDiff-TW1_NTNU-CWA-orog"
# "Prithvi-UNet"
# "Prithvi-UNet-orog"
# "XGBoost_IDL"
# "DeepESD_IDL"
# "deepsensor_bas"
# "deepsensor_orog_bas"
# "ResGAN"
# "ResGAN-orog"
# "CNRM-UNeT"
# "Rossby-UNet"
# "Rossby-UNet-orog"
# "EnScale"
# "EnScale-orog"
# "EnScale-linex"
# "spaGAN"
# "spaGAN-orog"
# "DetUNet"
# "DetUNet-orog"
# "ParamUNET"
# "ResGAN-v2"
# "ResGAN-v2-orog"
# "DetUNet-v2"
# "DetUNet-v2-orog"
# "ANN"
# "ANN-orog"

#MODELS=(
#    "Prithvi-UNet"
#    "Prithvi-UNet-orog"
#)
MODELS=(
    "Prithvi-UNet-orog_old"
)
export PATH="$HOME/.local/bin:$PATH"
export PIXI_HOME="${PIXI_HOME:-/lustre/gmeteo/WORK/rampaln/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/lustre/gmeteo/WORK/rampaln/.pixi/cache}"
export PYTHONUNBUFFERED=1

PIXI_PROJECT="${PIXI_PROJECT:-/lustre/gmeteo/WORK/rampaln/cordex-ml-bench}"
#cd "$PIXI_PROJECT"
echo "PIXI_HOME=$PIXI_HOME"
pixi info

SCRIPT_DIR="/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/compute/diagnostics"
PYTHON_SCRIPT="$SCRIPT_DIR/compute_diagnostics.py"

for model in "${MODELS[@]}"; do
    echo "Running: $model"
    sed "s/^model_to_evaluate = .*/model_to_evaluate = \"${model}\"/" "$PYTHON_SCRIPT" \
        > "$SCRIPT_DIR/.temp_${model}.py"
    pixi run python "$SCRIPT_DIR/.temp_${model}.py"
    rm "$SCRIPT_DIR/.temp_${model}.py"
done

echo "Done"
