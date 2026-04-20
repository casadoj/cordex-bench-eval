"""
Fix flipped grids for selected submissions/paths.

This utility corrects files where the data values are mirrored along a spatial
axis.  It supports two operations controlled by --axis:

  --axis x   flip along x (the last spatial axis, i.e. left-right mirror)
  --axis y   flip along y (the second-to-last spatial axis, i.e. top-bottom mirror)

Use --domain to restrict the correction to specific domains (ALPS, NZ, SA).
When omitted, all domains are processed.

Targets are explicit only:
- pass one or more catalog submission names with --submission
- and/or pass one or more directories with --path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import xarray as xr

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "config.json"

_DOMAIN_DIR_NAMES = {
    "ALPS": "ALPS_Domain",
    "NZ": "NZ_Domain",
    "SA": "SA_Domain",
}


def _fix_file(nc_path: Path, flip_axes: List[int], dry_run: bool = False) -> bool:
    """Apply flip(s) to pr/tasmax along the given axes (negative indexing)."""
    with xr.open_dataset(nc_path) as raw_ds:
        vars_to_fix = [var for var in ("pr", "tasmax") if var in raw_ds.data_vars]
        if not vars_to_fix:
            return False

        ds = raw_ds.load()

    if dry_run:
        return True

    for var in vars_to_fix:
        values = ds[var].values
        for ax in flip_axes:
            values = np.flip(values, axis=ax)
        ds[var].values = values

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


def _matches_domain(nc_path: Path, domain_dirs: Set[str]) -> bool:
    return any(d in nc_path.parts for d in domain_dirs)


def _scan_target(
    target: Path,
    flip_axes: List[int],
    domain_dirs: Set[str],
    dry_run: bool = False,
) -> List[Path]:
    if not target.exists():
        raise FileNotFoundError(f"Target path not found: {target}")

    nc_files = sorted(target.rglob("*.nc"))
    if not nc_files:
        print(f"  no .nc files under: {target}")
        return []

    corrected: List[Path] = []
    for nc_path in nc_files:
        if not _matches_domain(nc_path, domain_dirs):
            print(f"  skip (domain filter): {nc_path}")
            continue
        if _fix_file(nc_path, flip_axes, dry_run=dry_run):
            corrected.append(nc_path)
            tag = "would-fix" if dry_run else "fixed"
            print(f"  {tag}: {nc_path}")
        else:
            print(f"  ok:      {nc_path}")
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flip pr/tasmax data along a chosen spatial axis to correct mirrored grids."
        )
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
        "--axis",
        action="append",
        choices=["x", "y"],
        required=True,
        help="Axis to flip: 'x' (horizontal, axis=-1) or 'y' (vertical, axis=-2). Repeatable.",
    )
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=["ALPS", "NZ", "SA"],
        help="Domain(s) to process (repeatable). If omitted, all domains are processed.",
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

    axis_map = {"x": -1, "y": -2}
    flip_axes = [axis_map[a] for a in args.axis]
    axis_desc = ", ".join(f"{a} (axis={axis_map[a]})" for a in args.axis)

    if args.domains:
        domain_dirs = {_DOMAIN_DIR_NAMES[d] for d in args.domains}
        domain_desc = ", ".join(args.domains)
    else:
        domain_dirs = set(_DOMAIN_DIR_NAMES.values())
        domain_desc = "all"

    targets = _collect_targets(args)
    if not targets:
        print("No targets resolved.")
        return

    if args.dry_run:
        print("Dry-run mode enabled; no files will be modified.\n")

    print(f"Flip axis/axes: {axis_desc}")
    print(f"Domains: {domain_desc}\n")

    total_corrected: List[Path] = []
    for target in targets:
        print(f"Processing: {target}")
        corrected = _scan_target(target, flip_axes, domain_dirs, dry_run=args.dry_run)
        total_corrected.extend(corrected)

    action = "would be corrected" if args.dry_run else "corrected"
    print(f"\nDone. {len(total_corrected)} file(s) {action}.")


if __name__ == "__main__":
    main()
