"""
Fix transposed x/y spatial dimensions in CORDEX benchmark submission NetCDF files.

Some submissions were generated with a bug in the submission script that stored the
data variable as (time, x, y, ...) instead of the correct (time, y, x, ...) order. 
The 1-D x/y coordinate values and the 2-D lat/lon auxiliary coordinates are correct
and untouched; only the in-file dimension order of the data variables needs to be corrected.

This script detects the issue and overwrites the affected files in-place so that all
downstream operations (evaluation, plotting, alignment) work without any code changes.

Usage
-----
# Fix specific submissions by name (resolved from the catalog):
    python fix_transposed_grid.py --submission EnScale --submission EnScale-orog

# Fix all .nc files under an arbitrary directory:
    python fix_transposed_grid.py --path /path/to/staged/EnScale

# Dry-run (detect only, no writes):
    python fix_transposed_grid.py --submission EnScale --dry-run

# Custom config file:
    python fix_transposed_grid.py --submission EnScale --config /other/config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import xarray as xr

########################################################################################
# Default config path — override with --config on the CLI
_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "config.json"
########################################################################################


def _needs_transposition(ds: xr.Dataset) -> bool:
    """Return True if any data variable has the x dimension before y.

    A correct submission stores spatial data as (..., y, x, ...).  When the
    submission script used the wrong spatial_dims tuple ('x', 'y') instead of
    ('y', 'x'), the data variable ends up with x before y while the lat/lon
    coordinate arrays (which were copied from the template) remain in (y, x)
    order — causing a 90° rotation in maps and a silent spatial mismatch during
    evaluation alignment.
    """
    for var in ds.data_vars:
        dims = ds[var].dims
        if "x" in dims and "y" in dims and dims.index("x") < dims.index("y"):
            return True
    return False


def _correct_transposed_file(nc_path: Path, dry_run: bool = False) -> bool:
    """Detect and correct a single NetCDF file with transposed x/y dimensions.

    The file is rewritten atomically (via a sibling .tmp.nc file that is then
    renamed over the original) so a crash during writing cannot corrupt the
    existing data.

    Parameters
    ----------
    nc_path : Path
        Path to the NetCDF file to inspect (and possibly overwrite).
    dry_run : bool
        When True the file is never written; the function only returns whether
        a correction would have been applied.

    Returns
    -------
    bool
        True if the file was (or, in dry-run mode, would have been) corrected.
    """
    with xr.open_dataset(nc_path) as raw_ds:
        # Promote any spatial helper variables that were not encoded as coords
        spatial_vars = [v for v in ("lat", "lon", "latitude", "longitude", "x", "y")
                        if v in raw_ds.data_vars]
        if spatial_vars:
            raw_ds = raw_ds.set_coords(spatial_vars)

        if not _needs_transposition(raw_ds):
            return False

        # Load fully into memory before closing the file handle so we can
        # safely overwrite the path afterwards.
        ds = raw_ds.load()

    if dry_run:
        return True

    # Transpose each data variable to canonical (time, y, x, member) order.
    # Any dimension not present in a given variable is simply skipped.
    _CANONICAL = ("time", "y", "x", "member")
    for var in list(ds.data_vars):
        var_dims = ds[var].dims
        if "x" in var_dims and "y" in var_dims and var_dims.index("x") < var_dims.index("y"):
            new_dims = [d for d in _CANONICAL if d in var_dims]
            ds[var] = ds[var].transpose(*new_dims)

    tmp_path = nc_path.with_suffix(".tmp.nc")
    try:
        ds.to_netcdf(tmp_path)
        tmp_path.replace(nc_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return True


def fix_transposed_grid(submission_path: Path, dry_run: bool = False) -> List[Path]:
    """Walk all NetCDF files under *submission_path* and correct transposed grids.

    Parameters
    ----------
    submission_path : Path
        Root directory of the staged submission (or any directory containing
        NetCDF files to inspect).
    dry_run : bool
        When True no files are modified; only detection is performed.

    Returns
    -------
    list[Path]
        Paths of files that were (or, in dry-run mode, would have been)
        corrected.
    """
    submission_path = Path(submission_path).expanduser().resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission path not found: {submission_path}")

    corrected: List[Path] = []
    nc_files = sorted(submission_path.rglob("*.nc"))
    if not nc_files:
        print(f"  No .nc files found under {submission_path}")
        return corrected

    for nc_path in nc_files:
        fixed = _correct_transposed_file(nc_path, dry_run=dry_run)
        if fixed:
            corrected.append(nc_path)
            tag = "[dry-run] would fix" if dry_run else "fixed"
            print(f"  {tag}: {nc_path}")
        else:
            print(f"  ok:  {nc_path}")

    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Correct NetCDF files whose data variables have x before y "
                     "in their dimension order (transposed spatial grid bug)."))
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--submission",
                        metavar="NAME",
                        action="append",
                        dest="submissions",
                        help="Submission name as registered in the catalog (repeatable).",)
    target.add_argument("--path",
                        metavar="DIR",
                        type=Path,
                        help="Arbitrary directory to scan instead of resolving from the catalog.")
    parser.add_argument("--config",
                        metavar="FILE",
                        type=Path,
                        default=_DEFAULT_CONFIG,
                        help=f"Path to config.json (default: {_DEFAULT_CONFIG}).")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Detect issues and print what would be changed without writing anything.")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode: no files will be modified.\n")

    # Resolve target directories
    if args.path is not None:
        targets = [args.path]
    else:
        # Resolve submission names to staging paths via catalog
        from cordex_bench_eval.config import load_config_file
        from cordex_bench_eval.catalog import SubmissionCatalog

        config = load_config_file(args.config)
        catalog = SubmissionCatalog(config.catalog_path)

        targets = []
        for name in args.submissions:
            entry = catalog.get(name)
            if entry is None:
                print(f"[ERROR] Submission '{name}' not found in catalog.", file=sys.stderr)
                sys.exit(1)
            targets.append(Path(entry.location))

    # Run the correction
    total_fixed: List[Path] = []
    for submission_path in targets:
        print(f"\nProcessing: {submission_path}")
        corrected = fix_transposed_grid(submission_path, dry_run=args.dry_run)
        total_fixed.extend(corrected)

    action = "would be corrected" if args.dry_run else "corrected"
    print(f"\nDone. {len(total_fixed)} file(s) {action}.")


if __name__ == "__main__":
    main()
