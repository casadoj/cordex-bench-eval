#!/usr/bin/env bash
# ==============================================================================
# convert_to_eval_experiment_format.sh
#
# Reorganises distribution-diagnostic NetCDF files from the per-period/flavour/
# GCM layout produced by compute_distribution_cc_diagnostics.py into the
# eval-experiment layout expected by ml-benchmark:
#
#   SOURCE: outputs/netcdf/distribution/{model}/{domain}/{experiment}/{variable}/
#               {period}/{flavour}/{gcm}/metric_*.nc
#   TARGET: outputs/netcdf/{model}/{domain}/{experiment}/{variable}/
#               {eval_experiment}/metric_*.nc
#
# For Emulator_hist_future the mapping is 1-to-1 (symlinks are created).
# For ESD_pseudo_reality future periods (mid_century + end_century both map to
# Perfect/Imperfect_extrapolation*) the two period files are merged by summing
# histogram counts and recomputing the distribution metrics.
#
# Usage:
#   bash convert_to_eval_experiment_format.sh [--dry-run] [--overwrite]
#
#   --dry-run    Print actions without writing anything.
#   --overwrite  Recreate targets even when they already exist.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${SCRIPT_DIR}/outputs/netcdf/distribution"
DST_ROOT="${SCRIPT_DIR}/outputs/netcdf"

DRY_RUN=false
OVERWRITE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --overwrite) OVERWRITE=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# GCM look-up tables (domain key → GCM name)
# ---------------------------------------------------------------------------
declare -A PRIMARY_GCM=([ALPS]="CNRM-CM5"   [NZ]="ACCESS-CM2"  [SA]="ACCESS-CM2")
declare -A EVAL_GCM=(   [ALPS]="MPI-ESM-LR" [NZ]="EC-Earth3"   [SA]="NorESM2-MM")

# ---------------------------------------------------------------------------
# Python merge helper (written to a temp file at start-up)
# ---------------------------------------------------------------------------
MERGE_PY=$(mktemp /tmp/merge_distribution_XXXXXX.py)
trap 'rm -f "$MERGE_PY"' EXIT

cat > "$MERGE_PY" << 'PYEOF'
"""
Merge two distribution-metric NetCDF files (mid_century + end_century) by
summing histogram counts, then recomputing probabilities and the metric value.

Usage:  python merge_distribution.py <file1> <file2> <outfile>
"""
import sys
import os
import numpy as np
import xarray as xr


def _probabilities(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total,
                     out=np.zeros_like(counts, dtype=np.float64),
                     where=total > 0)


def _recompute_lhd(pred_counts: np.ndarray,
                   truth_counts: np.ndarray,
                   min_count: int = 10):
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    truth_counts = np.asarray(truth_counts, dtype=np.float64)
    if pred_counts.ndim == 1:
        valid = (pred_counts > min_count) & (truth_counts > min_count)
        if not valid.any():
            return np.asarray(np.nan), valid
        value = np.sqrt(
            np.mean(10.0 * np.log10(pred_counts[valid] / truth_counts[valid]) ** 2)
        )
        return np.asarray(value), valid
    values, valid_masks = [], []
    for mc in pred_counts:
        valid = (mc > min_count) & (truth_counts > min_count)
        valid_masks.append(valid)
        if not valid.any():
            values.append(np.nan)
        else:
            values.append(
                np.sqrt(np.mean(
                    10.0 * np.log10(mc[valid] / truth_counts[valid]) ** 2
                ))
            )
    return np.asarray(values, dtype=np.float64), np.asarray(valid_masks, dtype=bool)


def _recompute_pss(pred_counts: np.ndarray, truth_counts: np.ndarray) -> np.ndarray:
    truth_p = _probabilities(np.asarray(truth_counts, dtype=np.float64))
    pred_p  = _probabilities(np.asarray(pred_counts,  dtype=np.float64))
    if pred_p.ndim == 1:
        return np.asarray(np.minimum(pred_p, truth_p).sum())
    return np.minimum(pred_p, truth_p[None, :]).sum(axis=1)


def merge(file1: str, file2: str, outfile: str) -> None:
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    tc = ds1["truth_count"].values + ds2["truth_count"].values
    pc = ds1["prediction_count"].values + ds2["prediction_count"].values

    tp = _probabilities(tc)
    pp = _probabilities(pc)

    new_vars: dict = {
        "truth_count":           (ds1["truth_count"].dims,          tc),
        "truth_probability":     (ds1["truth_probability"].dims,     tp),
        "prediction_count":      (ds1["prediction_count"].dims,      pc),
        "prediction_probability":(ds1["prediction_probability"].dims, pp),
    }

    if "lhd" in ds1.data_vars:
        min_count = int(ds1.attrs.get("min_count_for_lhd", 10))
        lhd_vals, valid_bins = _recompute_lhd(pc, tc, min_count=min_count)
        if "lhd_by_member" in ds1.data_vars:
            new_vars["lhd_by_member"] = (ds1["lhd_by_member"].dims, lhd_vals)
            new_vars["lhd"]           = ([], np.asarray(np.nanmean(lhd_vals)))
        else:
            new_vars["lhd"]           = ([], lhd_vals)
        new_vars["valid_bin"] = (ds1["valid_bin"].dims, valid_bins)

    if "perkins_skill_score" in ds1.data_vars:
        pss_vals = _recompute_pss(pc, tc)
        if "perkins_skill_score_by_member" in ds1.data_vars:
            new_vars["perkins_skill_score_by_member"] = (ds1["perkins_skill_score_by_member"].dims, pss_vals)
            new_vars["perkins_skill_score"] = ([], np.asarray(np.nanmean(pss_vals)))
        else:
            new_vars["perkins_skill_score"] = ([], pss_vals)

    coords = {k: v for k, v in ds1.coords.items()}
    out_ds = xr.Dataset(new_vars, coords=coords)

    attrs = dict(ds1.attrs)
    attrs["period"] = "mid_century+end_century"
    out_ds.attrs = attrs

    ds1.close()
    ds2.close()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out_ds.to_netcdf(outfile)
    print(f"  [merged] {outfile}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: merge_distribution.py <file1> <file2> <outfile>", file=sys.stderr)
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
PYEOF

# ---------------------------------------------------------------------------
# Helper to run Python (respects pixi environment when available)
# ---------------------------------------------------------------------------
run_python() {
    if command -v pixi &>/dev/null; then
        pixi run --manifest-path "${REPO_ROOT}/pyproject.toml" python "$@"
    else
        python "$@"
    fi
}

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
n_symlink=0; n_merged=0; n_skipped=0; n_missing=0; n_error=0

# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------
process_file() {
    local src_file="$1"
    local rel="${src_file#${SRC_ROOT}/}"

    # Split into 8 components: model/domain/experiment/variable/period/flavour/gcm/filename
    IFS='/' read -r model domain experiment variable period flavour gcm filename <<< "$rel"

    # Derive domain key (e.g. ALPS_Domain → ALPS, NZ_Domain → NZ)
    local domain_key
    domain_key="${domain%_Domain}"
    domain_key="${domain_key^^}"

    local primary="${PRIMARY_GCM[$domain_key]:-}"
    local eval_gcm="${EVAL_GCM[$domain_key]:-}"

    if [[ -z "$primary" ]]; then
        echo "  [skip] unknown domain key '${domain_key}': ${rel}"
        (( n_skipped++ )) || true
        return
    fi

    # ------------------------------------------------------------------
    # Determine target eval_experiment and whether a merge is needed
    # ------------------------------------------------------------------
    local eval_exp=""
    local needs_merge=false

    if [[ "$experiment" == "Emulator_hist_future" ]]; then
        case "${period}/${flavour}/${gcm}" in
            "historical/perfect/${primary}")        eval_exp="Perfect_cross_validation" ;;
            "historical/imperfect/${primary}")      eval_exp="Imperfect_cross_validation" ;;
            "mid_century/perfect/${primary}")       eval_exp="Perfect_interpolation" ;;
            "mid_century/imperfect/${primary}")     eval_exp="Imperfect_interpolation" ;;
            "mid_century/perfect/${eval_gcm}")      eval_exp="Perfect_interpolation_GCM_transferability" ;;
            "mid_century/imperfect/${eval_gcm}")    eval_exp="Imperfect_interpolation_GCM_transferability" ;;
            # end_century and historical non-primary-GCM → no mapping
        esac

    elif [[ "$experiment" == "ESD_pseudo_reality" ]]; then
        case "${period}/${flavour}/${gcm}" in
            "historical/perfect/${primary}")        eval_exp="Perfect_cross_validation" ;;
            "historical/imperfect/${primary}")      eval_exp="Imperfect_cross_validation" ;;
            "mid_century/perfect/${primary}")       eval_exp="Perfect_extrapolation";                    needs_merge=true ;;
            "mid_century/imperfect/${primary}")     eval_exp="Imperfect_extrapolation";                  needs_merge=true ;;
            "mid_century/perfect/${eval_gcm}")      eval_exp="Perfect_extrapolation_GCM_transferability"; needs_merge=true ;;
            "end_century/"*)
                # end_century is processed during the mid_century pass → skip
                (( n_skipped++ )) || true
                return ;;
            # historical non-primary GCM and imperfect EVAL_GCM future → no mapping
        esac
    fi

    if [[ -z "$eval_exp" ]]; then
        (( n_skipped++ )) || true
        return
    fi

    local dst_file="${DST_ROOT}/${model}/${domain}/${experiment}/${variable}/${eval_exp}/${filename}"

    # Skip if already exists (unless --overwrite)
    if [[ -e "$dst_file" ]] && ! $OVERWRITE; then
        (( n_skipped++ )) || true
        return
    fi

    if $DRY_RUN; then
        if $needs_merge; then
            echo "  [DRY merge] mid+end_century → ${dst_file#${DST_ROOT}/}"
        else
            echo "  [DRY link]  ${rel} → ${dst_file#${DST_ROOT}/}"
        fi
        return
    fi

    mkdir -p "$(dirname "$dst_file")"

    if $needs_merge; then
        local end_src="${SRC_ROOT}/${model}/${domain}/${experiment}/${variable}/end_century/${flavour}/${gcm}/${filename}"
        if [[ ! -f "$end_src" ]]; then
            echo "  [MISSING end_century] ${end_src}"
            (( n_missing++ )) || true
            # Fall back: symlink mid_century only
            ln -sfn "$(realpath "$src_file")" "$dst_file"
            (( n_symlink++ )) || true
        else
            if run_python "$MERGE_PY" "$src_file" "$end_src" "$dst_file"; then
                (( n_merged++ )) || true
            else
                echo "  [ERROR] merge failed for ${dst_file}"
                (( n_error++ )) || true
            fi
        fi
    else
        ln -sfn "$(realpath "$src_file")" "$dst_file"
        (( n_symlink++ )) || true
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Converting distribution diagnostics → eval_experiment layout"
echo "  source : ${SRC_ROOT}"
echo "  target : ${DST_ROOT}"
$DRY_RUN  && echo "  mode   : DRY RUN (no files written)"
$OVERWRITE && echo "  mode   : OVERWRITE enabled"
echo "============================================================"

if [[ ! -d "$SRC_ROOT" ]]; then
    echo "ERROR: source directory not found: $SRC_ROOT" >&2
    exit 1
fi

while IFS= read -r -d '' file; do
    process_file "$file"
done < <(find "$SRC_ROOT" -name "*.nc" -print0 | sort -z)

echo ""
echo "------------------------------------------------------------"
echo " Results"
echo "   Symlinked : ${n_symlink}"
echo "   Merged    : ${n_merged}"
echo "   Skipped   : ${n_skipped}"
echo "   Missing   : ${n_missing}  (end_century file absent; mid_century used)"
echo "   Errors    : ${n_error}"
echo "------------------------------------------------------------"

# ---------------------------------------------------------------------------
# Check for missing files: every expected target should exist
# ---------------------------------------------------------------------------
echo ""
echo "Checking for missing target files..."
missing_targets=0

check_expected() {
    local dst_file="$1"
    if [[ ! -e "$dst_file" ]]; then
        echo "  [MISSING TARGET] ${dst_file#${DST_ROOT}/}"
        (( missing_targets++ )) || true
    fi
}

# Walk existing source structure and verify corresponding targets exist
while IFS= read -r -d '' file; do
    local_rel="${file#${SRC_ROOT}/}"
    IFS='/' read -r model domain experiment variable period flavour gcm filename <<< "$local_rel"
    domain_key="${domain%_Domain}"; domain_key="${domain_key^^}"
    primary="${PRIMARY_GCM[$domain_key]:-}"; eval_gcm="${EVAL_GCM[$domain_key]:-}"
    [[ -z "$primary" ]] && continue

    eval_exp=""
    if [[ "$experiment" == "Emulator_hist_future" ]]; then
        case "${period}/${flavour}/${gcm}" in
            "historical/perfect/${primary}")     eval_exp="Perfect_cross_validation" ;;
            "historical/imperfect/${primary}")   eval_exp="Imperfect_cross_validation" ;;
            "mid_century/perfect/${primary}")    eval_exp="Perfect_interpolation" ;;
            "mid_century/imperfect/${primary}")  eval_exp="Imperfect_interpolation" ;;
            "mid_century/perfect/${eval_gcm}")   eval_exp="Perfect_interpolation_GCM_transferability" ;;
            "mid_century/imperfect/${eval_gcm}") eval_exp="Imperfect_interpolation_GCM_transferability" ;;
        esac
    elif [[ "$experiment" == "ESD_pseudo_reality" ]]; then
        case "${period}/${flavour}/${gcm}" in
            "historical/perfect/${primary}")   eval_exp="Perfect_cross_validation" ;;
            "historical/imperfect/${primary}") eval_exp="Imperfect_cross_validation" ;;
            "mid_century/perfect/${primary}")  eval_exp="Perfect_extrapolation" ;;
            "mid_century/imperfect/${primary}") eval_exp="Imperfect_extrapolation" ;;
            "mid_century/perfect/${eval_gcm}") eval_exp="Perfect_extrapolation_GCM_transferability" ;;
        esac
    fi

    [[ -z "$eval_exp" ]] && continue
    check_expected "${DST_ROOT}/${model}/${domain}/${experiment}/${variable}/${eval_exp}/${filename}"
done < <(find "$SRC_ROOT" -name "*.nc" -print0 | sort -z)

if [[ $missing_targets -eq 0 ]]; then
    echo "  All expected target files are present."
else
    echo "  ${missing_targets} target file(s) missing."
fi
echo "============================================================"
