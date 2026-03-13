"""
Fix ALPS submission files missing explicit lat/lon coordinates.

This utility is intentionally aligned with the issue43 fixer:
- targets are explicit with --submission and/or --path
- only ALPS_Domain files are processed
- supports --config and --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "config.json"


def _is_alps_path(path: Path) -> bool:
    return "alps_domain" in str(path).lower()


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


def _iter_nc_files(target: Path) -> List[Path]:
    if not target.exists():
        raise FileNotFoundError(f"Target path not found: {target}")
    if target.is_file() and target.suffix == ".nc":
        return [target]
    return sorted(target.rglob("*.nc"))


def _load_latlon_if_available(nc_path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    with xr.open_dataset(nc_path, decode_cf=False) as raw_ds:
        ds = raw_ds.set_coords([name for name in ("lat", "lon") if name in raw_ds.data_vars])
        if "lat" not in ds.coords or "lon" not in ds.coords:
            return None
        lat = ds["lat"].load().values
        lon = ds["lon"].load().values
    if lat.ndim != 2 or lon.ndim != 2:
        return None
    return lat, lon


def _discover_reference_latlon(targets: List[Path]) -> Tuple[np.ndarray, np.ndarray, Path]:
    for target in targets:
        for nc_path in _iter_nc_files(target):
            if not _is_alps_path(nc_path):
                continue
            coords = _load_latlon_if_available(nc_path)
            if coords is not None:
                lat, lon = coords
                return lat, lon, nc_path
    raise RuntimeError(
        "Could not find an ALPS file with existing lat/lon coordinates among provided targets."
    )


def _fix_file(nc_path: Path, ref_lat: np.ndarray, ref_lon: np.ndarray, dry_run: bool = False) -> bool:
    with xr.open_dataset(nc_path, decode_cf=False) as ds:
        ds = ds.set_coords([name for name in ("lat", "lon") if name in ds.data_vars])

        if "lat" in ds.coords and "lon" in ds.coords:
            return False
        if "y" not in ds.dims or "x" not in ds.dims:
            raise ValueError("Expected x/y dimensions.")

        target_shape = (int(ds.sizes["y"]), int(ds.sizes["x"]))
        if ref_lat.shape != target_shape:
            raise ValueError(f"Reference shape {ref_lat.shape} does not match target shape {target_shape}.")
        lat_values = ref_lat
        lon_values = ref_lon

        if dry_run:
            return True

        out = ds.load().assign_coords(lat=(("y", "x"), lat_values), lon=(("y", "x"), lon_values))

    tmp_path = nc_path.with_suffix(".tmp.nc")
    try:
        out.to_netcdf(tmp_path)
        tmp_path.replace(nc_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return True


def _scan_target(target: Path, ref_lat: np.ndarray, ref_lon: np.ndarray, dry_run: bool = False) -> List[Path]:
    nc_files = _iter_nc_files(target)
    if not nc_files:
        print(f"  no .nc files under: {target}")
        return []

    corrected: List[Path] = []
    for nc_path in nc_files:
        if not _is_alps_path(nc_path):
            print(f"  skip (non-ALPS_Domain): {nc_path}")
            continue
        if _fix_file(nc_path, ref_lat=ref_lat, ref_lon=ref_lon, dry_run=dry_run):
            corrected.append(nc_path)
            tag = "would-fix" if dry_run else "fixed"
            print(f"  {tag}: {nc_path}")
        else:
            print(f"  ok:      {nc_path}")
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add missing ALPS lat/lon coordinates using an existing ALPS file as reference."
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

    ref_lat, ref_lon, ref_path = _discover_reference_latlon(targets)
    print(f"Using reference coordinates from: {ref_path}\n")

    total_corrected: List[Path] = []
    for target in targets:
        print(f"Processing: {target}")
        corrected = _scan_target(target, ref_lat=ref_lat, ref_lon=ref_lon, dry_run=args.dry_run)
        total_corrected.extend(corrected)

    action = "would be corrected" if args.dry_run else "corrected"
    print(f"\nDone. {len(total_corrected)} file(s) {action}.")


if __name__ == "__main__":
    main()
