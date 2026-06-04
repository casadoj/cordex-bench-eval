#!/bin/bash

SRC="/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/diagnostics/outputs/netcdf/DetUNet_orog"
DST="/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/cordex-bench-eval/diagnostics/outputs/netcdf/DetUNet-orog"

# ── Dry-run by default; pass --execute to actually move ────────────────────
DRY_RUN=true
if [[ "$1" == "--execute" ]]; then
    DRY_RUN=false
fi

if [ ! -d "$SRC" ]; then
    echo "ERROR: source directory not found: $SRC"
    exit 1
fi

echo "Source : $SRC"
echo "Dest   : $DST"
if $DRY_RUN; then
    echo "Mode   : DRY RUN (pass --execute to actually move files)"
else
    echo "Mode   : EXECUTE"
fi
echo ""

moved=0
skipped=0
errors=0

while IFS= read -r -d '' src_file; do
    # Compute relative path from source root
    rel="${src_file#$SRC/}"
    dst_file="$DST/$rel"
    dst_dir="$(dirname "$dst_file")"

    if [ -e "$dst_file" ]; then
        echo "  SKIP (already exists): $rel"
        (( skipped++ ))
        continue
    fi

    if $DRY_RUN; then
        echo "  MOVE: $rel"
        echo "     -> $dst_file"
    else
        mkdir -p "$dst_dir" && mv "$src_file" "$dst_file"
        if [ $? -eq 0 ]; then
            echo "  MOVED: $rel"
            (( moved++ ))
        else
            echo "  ERROR moving: $rel"
            (( errors++ ))
        fi
    fi
    $DRY_RUN && (( moved++ ))

done < <(find "$SRC" -type f -print0)

echo ""
echo "=================================================="
if $DRY_RUN; then
    echo "  Dry run complete."
    echo "  Files that would be moved : $moved"
    echo "  Files that would be skipped (already exist): $skipped"
    echo ""
    echo "  To execute, run:"
    echo "    bash move_spagan_orog.sh --execute"
else
    echo "  Done."
    echo "  Moved   : $moved"
    echo "  Skipped : $skipped"
    echo "  Errors  : $errors"
fi
echo "=================================================="