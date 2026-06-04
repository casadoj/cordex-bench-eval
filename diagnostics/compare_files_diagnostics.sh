#!/bin/bash

# Configuration
WORKDIR="/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/diagnostics"
PATTERN="metric*.nc"

# All model directories
models=(
    ANN ANN-orog CNRM-UNeT CorrDiff-TW1_NTNU-CWA CorrDiff-TW1_NTNU-CWA-orog
    DeepESDcrps_IFCAv2 DeepESDcrps_IFCAv2-orog DeepESD_IDL DeepESD-IFCAv1 DeepESD-IFCAv1-orog
    deepsensor_bas deepsensor_orog_bas DetUNet DetUNet-orog DetUNet-v2 DetUNet-v2-orog
    EnScale EnScale-linex EnScale-orog FlowMatching-v1 FlowMatching-v1-orog
    GNN4CD-orog ParamDiffusion-orog ParamUNET Prithvi-UNet Prithvi-UNet-orog
    RCMFlow-orog RCMGEM-mv-orog ResGAN ResGAN-orog ResGAN-v2 ResGAN-v2-orog
    Rossby-UNet Rossby-UNet-orog spaGAN spaGAN-orog UiBCorrDiff-orog
    ViT-IFCAv1 ViT-IFCAv1-orog XGBoost_IDL
)

# ── Section 1: file counts ──────────────────────────────────────────────────
echo "=================================================="
echo "  metric*.nc file count per model"
echo "=================================================="

declare -A counts

for model in "${models[@]}"; do
    dir="$WORKDIR/$model"
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f -name "$PATTERN" | wc -l)
        counts["$model"]=$count
        printf "  %-35s %4d files\n" "$model" "$count"
    else
        printf "  %-35s  DIR NOT FOUND\n" "$model"
        counts["$model"]=-1
    fi
done

# ── Section 2: pairwise diff vs ANN ────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Pairwise diff: each model vs ANN (reference)"
echo "  (only models with differences are shown)"
echo "=================================================="

REFMODEL="ANN"
REFDIR="$WORKDIR/$REFMODEL"

if [ ! -d "$REFDIR" ]; then
    echo "ERROR: reference directory $REFDIR not found."
    exit 1
fi

ref_files=$(find "$REFDIR" -type f -name "$PATTERN" | sed "s|^$REFDIR/||" | sort)

for model in "${models[@]}"; do
    [ "$model" == "$REFMODEL" ] && continue

    dir="$WORKDIR/$model"
    [ ! -d "$dir" ] && continue

    model_files=$(find "$dir" -type f -name "$PATTERN" | sed "s|^$dir/||" | sort)

    only_in_ref=$(diff  <(echo "$ref_files") <(echo "$model_files") | grep '^<' | sed 's/^< //')
    only_in_model=$(diff <(echo "$ref_files") <(echo "$model_files") | grep '^>' | sed 's/^> //')

    # Skip models with identical file sets
    [ -z "$only_in_ref" ] && [ -z "$only_in_model" ] && continue

    echo ""
    echo "--- $REFMODEL  vs  $model ---"
    if [ -n "$only_in_ref" ]; then
        echo "  Only in $REFMODEL (missing from $model):"
        echo "$only_in_ref" | sed 's/^/    /'
    fi
    if [ -n "$only_in_model" ]; then
        echo "  Only in $model (not in $REFMODEL):"
        echo "$only_in_model" | sed 's/^/    /'
    fi
done

echo ""
echo "Done."