"""
Fix tasmax units: convert Celsius to Kelvin for selected submissions/paths.

Some submissions have tasmax stored in degrees Celsius instead of Kelvin.
This utility adds 273.15 to every tasmax value and updates the ``units``
attribute accordingly.

A basic sanity check is performed: files whose tasmax values are already
plausibly in Kelvin (median > 100) are skipped to avoid double-conversion.

Targets are explicit only:
- pass one or more catalog submission names with --submission
- and/or pass one or more directories with --path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "config.json"
_CELSIUS_TO_KELVIN = 273.15
_KELVIN_THRESHOLD = 100  # median above this ⇒ already Kelvin


def _fix_file(nc_path: Path, dry_run: bool = False) -> bool:
    """Convert tasmax from Celsius to Kelvin in *nc_path*."""
    with xr.open_dataset(nc_path) as raw_ds:
        if "tasmax" not in raw_ds.data_vars:
            return False

        median_val = float(np.nanmedian(raw_ds["tasmax"].values))
        if median_val > _KELVIN_THRESHOLD:
            return False

        ds = raw_ds.load()

    if dry_run:
        return True

    ds["tasmax"].values = ds["tasmax"].values + _CELSIUS_TO_KELVIN
    ds["tasmax"].attrs["units"] = "K"

    tmp_path = nc_path.with_suffix(".tmp.nc")
    try:
        ds.to_netcdf(tmp_path)
        tmp_path.replace(nc_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return True


def _collect_targets(args: argparse.Namespace) -> List[Path]:
    targets: List[Path] = []

    if args.paths:
        targets.extend(Path(p).expanduser().resolve() for p in args.paths)

    if args.submissions:
        from cordex_bench_eval.catalog import SubmissionCatalog
        from cordex_bench_eval.config import load_config_file

        config = load_config_file(args.config)
        catalog = SubmissionCatalog(config.catalog_path)

        for name in args.submissions:
            entry = catalog.get(name)
            if entry is None:
                print(f"[ERROR] Submission '{name}' not found in catalog.", file=sys.stderr)
                sys.exit(1)
            targets.append(Path(entry.location).expanduser().resolve())

    deduped: List[Path] = []
    seen = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        deduped.append(target)

    return deduped


def _scan_target(target: Path, dry_run: bool = False) -> List[Path]:
    if not target.exists():
        raise FileNotFoundError(f"Target path not found: {target}")

    nc_files = sorted(target.rglob("*.nc"))
    if not nc_files:
        print(f"  no .nc files under: {target}")
        return []

    corrected: List[Path] = []
    for nc_path in nc_files:
        if _fix_file(nc_path, dry_run=dry_run):
            corrected.append(nc_path)
            tag = "would-fix" if dry_run else "fixed"
            print(f"  {tag}: {nc_path}")
        else:
            print(f"  skip (no tasmax or already Kelvin): {nc_path}")
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tasmax from Celsius to Kelvin in selected submissions."
    )
    parser.add_argument(
        "--submission",
        dest="submissions",
        action="append",
        metavar="NAME",
        help="Catalog submission name (repeatable).",
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        metavar="DIR",
        type=Path,
        help="Directory to scan recursively for .nc files (repeatable).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        metavar="FILE",
        help=f"Path to config.json for catalog resolution (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files that would be fixed; do not write changes.",
    )
    args = parser.parse_args()

    if not args.submissions and not args.paths:
        parser.error("provide at least one --submission and/or --path target")

    targets = _collect_targets(args)
    if not targets:
        print("No targets resolved.")
        return

    if args.dry_run:
        print("Dry-run mode enabled; no files will be modified.\n")

    total_corrected: List[Path] = []
    for target in targets:
        print(f"Processing: {target}")
        corrected = _scan_target(target, dry_run=args.dry_run)
        total_corrected.extend(corrected)

    action = "would be corrected" if args.dry_run else "corrected"
    print(f"\nDone. {len(total_corrected)} file(s) {action}.")


if __name__ == "__main__":
    main()
