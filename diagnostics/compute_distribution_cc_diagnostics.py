#!/usr/bin/env python
"""
Compute distribution and climate-change diagnostics for CORDEX-ML submissions.

This script reads prediction files from the staging tree and target files from
the benchmark target tree, but writes all new outputs to a local diagnostics
folder by default.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_TRUTH_ROOT = Path("/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/target")
DEFAULT_PRED_ROOT = Path("/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/staging")
DEFAULT_CATALOG_PATH = Path("/lustre/gmeteo/WORK/abadj/cordex-bench-eval/catalog/catalog.json")
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "outputs"

DOMAIN_GCMS: dict[str, list[str]] = {
    "ALPS": ["CNRM-CM5", "MPI-ESM-LR"],
    "NZ": ["ACCESS-CM2", "EC-Earth3"],
    "SA": ["ACCESS-CM2", "NorESM2-MM"],
}
PRIMARY_GCM: dict[str, str] = {
    "ALPS": "CNRM-CM5",
    "NZ": "ACCESS-CM2",
    "SA": "ACCESS-CM2",
}
EVALUATION_GCM: dict[str, str] = {
    "ALPS": "MPI-ESM-LR",
    "NZ": "EC-Earth3",
    "SA": "NorESM2-MM",
}
PERIOD_YEARS: dict[str, str] = {
    "historical": "1981-2000",
    "mid_century": "2041-2060",
    "end_century": "2080-2099",
}
EXPERIMENTS = ("ESD_pseudo_reality", "Emulator_hist_future")
PERIODS = ("historical", "mid_century", "end_century")
FUTURE_PERIODS = ("mid_century", "end_century")
FLAVOURS = ("perfect", "imperfect")
VARIABLES = ("pr", "tasmax")
SEASONS = ("DJF", "JJA")


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    probabilistic: bool


@dataclass(frozen=True)
class Scenario:
    model: str
    domain: str
    domain_key: str
    experiment: str
    period: str
    flavour: str
    gcm: str
    pred_path: Path
    truth_path: Path


@dataclass(frozen=True)
class ClimateChangeScenario:
    model: str
    domain: str
    domain_key: str
    experiment: str
    future_period: str
    flavour: str
    gcm: str
    pred_hist_path: Path
    pred_future_path: Path
    truth_hist_path: Path
    truth_future_path: Path


def _normalize_domain_key(domain: str) -> str:
    value = Path(domain).name
    for suffix in ("_domain", "_Domain"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    value = value.upper()
    if value not in DOMAIN_GCMS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {sorted(DOMAIN_GCMS)}")
    return value


def _domain_dir_name(domain_key: str) -> str:
    return f"{domain_key}_Domain"


def _truth_domain_dir_name(domain_key: str) -> str:
    return f"{domain_key}_domain"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_catalog(path: Path) -> list[CatalogEntry]:
    payload = json.loads(path.read_text())
    entries: list[CatalogEntry] = []
    for raw in payload.get("entries", []):
        extra = raw.get("extra", {})
        entries.append(
            CatalogEntry(
                name=raw["name"],
                probabilistic=str(extra.get("probabilistic", "false")).lower() == "true",
            )
        )
    return entries


def _selected_gcms(domain_key: str, mode: str) -> list[str]:
    if mode == "all":
        return DOMAIN_GCMS[domain_key]
    if mode == "primary":
        return [PRIMARY_GCM[domain_key]]
    if mode == "evaluation":
        return [EVALUATION_GCM[domain_key]]
    raise ValueError(f"Unsupported gcm mode: {mode}")


def _truth_file(root: Path, domain_key: str, period: str, gcm: str) -> Path:
    years = PERIOD_YEARS[period]
    return (
        root
        / _truth_domain_dir_name(domain_key)
        / "test"
        / period
        / "target"
        / f"pr_tasmax_{gcm}_{years}.nc"
    )


def _prediction_file(
    root: Path,
    model: str,
    domain_key: str,
    experiment: str,
    period: str,
    flavour: str,
    gcm: str,
) -> Path:
    years = PERIOD_YEARS[period]
    return (
        root
        / model
        / _domain_dir_name(domain_key)
        / experiment
        / period
        / flavour
        / f"Predictions_pr_tasmax_{gcm}_{years}.nc"
    )


def _iter_distribution_scenarios(args: argparse.Namespace, models: Sequence[str]) -> Iterator[Scenario]:
    for model in models:
        for domain in args.domains:
            domain_key = _normalize_domain_key(domain)
            domain_name = _domain_dir_name(domain_key)
            for experiment in args.experiments:
                for flavour in args.flavours:
                    for gcm in _selected_gcms(domain_key, args.gcm_mode):
                        for period in args.periods:
                            pred_path = _prediction_file(
                                args.pred_root, model, domain_key, experiment, period, flavour, gcm
                            )
                            truth_path = _truth_file(args.truth_root, domain_key, period, gcm)
                            if pred_path.exists() and truth_path.exists():
                                yield Scenario(
                                    model=model,
                                    domain=domain_name,
                                    domain_key=domain_key,
                                    experiment=experiment,
                                    period=period,
                                    flavour=flavour,
                                    gcm=gcm,
                                    pred_path=pred_path,
                                    truth_path=truth_path,
                                )
                            else:
                                missing = pred_path if not pred_path.exists() else truth_path
                                print(f"[skip] missing file for distribution: {missing}")


def _iter_cc_scenarios(args: argparse.Namespace, models: Sequence[str]) -> Iterator[ClimateChangeScenario]:
    for model in models:
        for domain in args.domains:
            domain_key = _normalize_domain_key(domain)
            domain_name = _domain_dir_name(domain_key)
            for experiment in args.experiments:
                for flavour in args.flavours:
                    for gcm in _selected_gcms(domain_key, args.gcm_mode):
                        for future_period in args.future_periods:
                            pred_hist_path = _prediction_file(
                                args.pred_root, model, domain_key, experiment, "historical", flavour, gcm
                            )
                            pred_future_path = _prediction_file(
                                args.pred_root, model, domain_key, experiment, future_period, flavour, gcm
                            )
                            truth_hist_path = _truth_file(args.truth_root, domain_key, "historical", gcm)
                            truth_future_path = _truth_file(args.truth_root, domain_key, future_period, gcm)
                            required = [pred_hist_path, pred_future_path, truth_hist_path, truth_future_path]
                            missing = [path for path in required if not path.exists()]
                            if missing:
                                print(f"[skip] missing file for climate-change: {missing[0]}")
                                continue
                            yield ClimateChangeScenario(
                                model=model,
                                domain=domain_name,
                                domain_key=domain_key,
                                experiment=experiment,
                                future_period=future_period,
                                flavour=flavour,
                                gcm=gcm,
                                pred_hist_path=pred_hist_path,
                                pred_future_path=pred_future_path,
                                truth_hist_path=truth_hist_path,
                                truth_future_path=truth_future_path,
                            )


def _open_variable(path: Path, variable: str, *, domain_key: str, is_truth: bool) -> tuple[xr.Dataset, xr.DataArray]:
    ds = xr.open_dataset(path)
    if variable not in ds.data_vars:
        ds.close()
        raise KeyError(f"Variable {variable!r} not found in {path}")
    da = _normalize_units(ds[variable], variable=variable, domain_key=domain_key, is_truth=is_truth)
    return ds, da


def _normalize_units(
    da: xr.DataArray,
    *,
    variable: str,
    domain_key: str,
    is_truth: bool,
) -> xr.DataArray:
    out = da
    units = str(out.attrs.get("units", "")).lower()

    if variable == "tasmax":
        convert = units in {"k", "kelvin"}
        if not convert:
            try:
                sample = out.isel(time=0) if "time" in out.dims else out
                convert = float(sample.max(skipna=True).values) > 150.0
            except (TypeError, ValueError, RuntimeError):
                convert = False
        if convert:
            out = out - 273.15
            out.attrs["units"] = "degC"
        return out

    if variable == "pr" and is_truth and domain_key == "SA":
        convert = "kg m-2 s-1" in units or "kg m**-2 s**-1" in units
        if not convert:
            try:
                sample = out.isel(time=0) if "time" in out.dims else out
                convert = float(sample.max(skipna=True).values) < 1.0
            except (TypeError, ValueError, RuntimeError):
                convert = False
        if convert:
            out = out * 86400.0
            out.attrs["units"] = "mm day-1"
    return out


def _member_values(da: xr.DataArray) -> np.ndarray | None:
    if "member" not in da.dims:
        return None
    if "member" in da.coords:
        return np.asarray(da["member"].values)
    return np.arange(da.sizes["member"])


def _histogram_counts(
    da: xr.DataArray,
    bins: np.ndarray,
    *,
    time_chunk: int,
    member_index: int | None = None,
) -> np.ndarray:
    work = da.isel(member=member_index) if member_index is not None and "member" in da.dims else da
    counts = np.zeros(len(bins) - 1, dtype=np.float64)

    if "time" in work.dims:
        n_time = work.sizes["time"]
        for start in range(0, n_time, time_chunk):
            stop = min(start + time_chunk, n_time)
            arr = np.asarray(work.isel(time=slice(start, stop)).values, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                counts += np.histogram(arr, bins=bins)[0]
    else:
        arr = np.asarray(work.values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            counts += np.histogram(arr, bins=bins)[0]
    return counts


def _probabilities(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts, dtype=np.float64), where=total > 0)


def _lhd(pred_counts: np.ndarray, truth_counts: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    truth_counts = np.asarray(truth_counts, dtype=np.float64)
    if pred_counts.ndim == 1:
        valid = (pred_counts > min_count) & (truth_counts > min_count)
        if not valid.any():
            return np.asarray(np.nan), valid
        value = np.sqrt(np.mean(10.0 * np.log10(pred_counts[valid] / truth_counts[valid]) ** 2))
        return np.asarray(value), valid

    values = []
    valid_masks = []
    for member_counts in pred_counts:
        valid = (member_counts > min_count) & (truth_counts > min_count)
        valid_masks.append(valid)
        if not valid.any():
            values.append(np.nan)
        else:
            values.append(
                np.sqrt(np.mean(10.0 * np.log10(member_counts[valid] / truth_counts[valid]) ** 2))
            )
    return np.asarray(values, dtype=np.float64), np.asarray(valid_masks, dtype=bool)


def _perkins_skill_score(pred_counts: np.ndarray, truth_counts: np.ndarray) -> np.ndarray:
    truth_p = _probabilities(truth_counts)
    pred_p = _probabilities(pred_counts)
    if pred_p.ndim == 1:
        return np.asarray(np.minimum(pred_p, truth_p).sum())
    return np.minimum(pred_p, truth_p[None, :]).sum(axis=1)


def _kl_divergence(pred_counts: np.ndarray, truth_counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    truth_p = _probabilities(truth_counts)
    pred_p = _probabilities(pred_counts)
    if pred_p.ndim == 1:
        mask = truth_p > 0
        return np.asarray(np.sum(truth_p[mask] * np.log((truth_p[mask] + eps) / (pred_p[mask] + eps))))
    values = []
    mask = truth_p > 0
    for member_p in pred_p:
        values.append(np.sum(truth_p[mask] * np.log((truth_p[mask] + eps) / (member_p[mask] + eps))))
    return np.asarray(values, dtype=np.float64)


def _distribution_output_dir(args: argparse.Namespace, scenario: Scenario, variable: str) -> Path:
    return (
        args.out_root
        / "netcdf"
        / "distribution"
        / scenario.model
        / scenario.domain
        / scenario.experiment
        / variable
        / scenario.period
        / scenario.flavour
        / _safe_name(scenario.gcm)
    )


def _distribution_plot_path(args: argparse.Namespace, scenario: Scenario, variable: str, metric_name: str) -> Path:
    return (
        args.out_root
        / "plots"
        / "distribution"
        / scenario.domain
        / scenario.experiment
        / variable
        / scenario.period
        / scenario.flavour
        / _safe_name(scenario.gcm)
        / f"{_safe_name(scenario.model)}_{metric_name}.png"
    )


def _compute_distribution_for_scenario(
    args: argparse.Namespace,
    scenario: Scenario,
    variable: str,
) -> Path:
    metric_name = "lhd" if variable == "pr" else args.tasmax_metric
    metric_path = _distribution_output_dir(args, scenario, variable) / f"metric_{metric_name}.nc"
    plot_path = _distribution_plot_path(args, scenario, variable, metric_name)
    if metric_path.exists() and (args.no_plots or plot_path.exists()) and not args.recompute:
        print(f"[reuse] {metric_path}")
        return metric_path

    if variable == "pr":
        bins = np.linspace(args.pr_lower, args.pr_upper, args.pr_bins + 1, dtype=np.float64)
    else:
        bins = np.linspace(args.tasmax_lower, args.tasmax_upper, args.tasmax_bins + 1, dtype=np.float64)

    truth_ds = pred_ds = None
    try:
        truth_ds, truth_da = _open_variable(
            scenario.truth_path, variable, domain_key=scenario.domain_key, is_truth=True
        )
        pred_ds, pred_da = _open_variable(
            scenario.pred_path, variable, domain_key=scenario.domain_key, is_truth=False
        )

        truth_counts = _histogram_counts(truth_da, bins, time_chunk=args.time_chunk)
        member_values = _member_values(pred_da)
        if member_values is None:
            pred_counts = _histogram_counts(pred_da, bins, time_chunk=args.time_chunk)
            metric_values, valid_bins = _distribution_metric(
                variable, args.tasmax_metric, pred_counts, truth_counts, args.min_count
            )
            metric_mean = float(metric_values)
        else:
            pred_counts = np.vstack(
                [
                    _histogram_counts(pred_da, bins, time_chunk=args.time_chunk, member_index=i)
                    for i in range(pred_da.sizes["member"])
                ]
            )
            metric_values, valid_bins = _distribution_metric(
                variable, args.tasmax_metric, pred_counts, truth_counts, args.min_count
            )
            metric_mean = float(np.nanmean(metric_values))
    finally:
        if truth_ds is not None:
            truth_ds.close()
        if pred_ds is not None:
            pred_ds.close()

    ds = _build_distribution_dataset(
        variable=variable,
        metric_name=metric_name,
        bins=bins,
        truth_counts=truth_counts,
        pred_counts=pred_counts,
        metric_values=metric_values,
        metric_mean=metric_mean,
        valid_bins=valid_bins,
        member_values=member_values,
        scenario=scenario,
        min_count=args.min_count,
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(metric_path)
    ds.close()
    print(f"[write] {metric_path}")

    if not args.no_plots:
        _plot_distribution(
            output_path=plot_path,
            variable=variable,
            metric_name=metric_name,
            bins=bins,
            truth_counts=truth_counts,
            pred_counts=pred_counts,
            member_values=member_values,
            metric_values=metric_values,
            metric_mean=metric_mean,
            scenario=scenario,
        )
    return metric_path


def _distribution_metric(
    variable: str,
    tasmax_metric: str,
    pred_counts: np.ndarray,
    truth_counts: np.ndarray,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if variable == "pr":
        return _lhd(pred_counts, truth_counts, min_count)
    if tasmax_metric == "perkins_skill_score":
        return _perkins_skill_score(pred_counts, truth_counts), None
    if tasmax_metric == "kl_divergence":
        return _kl_divergence(pred_counts, truth_counts), None
    raise ValueError(f"Unknown tasmax metric: {tasmax_metric}")


def _build_distribution_dataset(
    *,
    variable: str,
    metric_name: str,
    bins: np.ndarray,
    truth_counts: np.ndarray,
    pred_counts: np.ndarray,
    metric_values: np.ndarray,
    metric_mean: float,
    valid_bins: np.ndarray | None,
    member_values: np.ndarray | None,
    scenario: Scenario,
    min_count: int,
) -> xr.Dataset:
    coords: dict[str, tuple[str, np.ndarray] | np.ndarray] = {
        "bin": np.arange(len(bins) - 1),
        "bin_lower": ("bin", bins[:-1]),
        "bin_upper": ("bin", bins[1:]),
        "bin_center": ("bin", 0.5 * (bins[:-1] + bins[1:])),
    }
    data_vars: dict[str, tuple[Sequence[str], np.ndarray] | xr.DataArray] = {
        "truth_count": (("bin",), truth_counts.astype(np.float64)),
        "truth_probability": (("bin",), _probabilities(truth_counts)),
    }

    if member_values is None:
        data_vars["prediction_count"] = (("bin",), pred_counts.astype(np.float64))
        data_vars["prediction_probability"] = (("bin",), _probabilities(pred_counts))
        data_vars[metric_name] = ((), np.asarray(metric_values, dtype=np.float64))
        if valid_bins is not None:
            data_vars["valid_bin"] = (("bin",), np.asarray(valid_bins, dtype=bool))
    else:
        coords["member"] = np.asarray(member_values)
        data_vars["prediction_count"] = (("member", "bin"), pred_counts.astype(np.float64))
        data_vars["prediction_probability"] = (("member", "bin"), _probabilities(pred_counts))
        data_vars[f"{metric_name}_by_member"] = (("member",), np.asarray(metric_values, dtype=np.float64))
        data_vars[metric_name] = ((), np.asarray(metric_mean, dtype=np.float64))
        if valid_bins is not None:
            data_vars["valid_bin"] = (("member", "bin"), np.asarray(valid_bins, dtype=bool))

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update(
        diagnostic="distribution",
        metric=metric_name,
        variable=variable,
        model=scenario.model,
        domain=scenario.domain,
        experiment=scenario.experiment,
        period=scenario.period,
        predictor_flavour=scenario.flavour,
        gcm=scenario.gcm,
        pred_path=str(scenario.pred_path),
        truth_path=str(scenario.truth_path),
        min_count_for_lhd=int(min_count),
    )
    if variable == "pr":
        ds.attrs["lhd_formula"] = "sqrt(mean(10 * log10(pred_count / truth_count) ** 2)) over bins with both counts > min_count"
    return ds


def _plot_distribution(
    *,
    output_path: Path,
    variable: str,
    metric_name: str,
    bins: np.ndarray,
    truth_counts: np.ndarray,
    pred_counts: np.ndarray,
    member_values: np.ndarray | None,
    metric_values: np.ndarray,
    metric_mean: float,
    scenario: Scenario,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    truth_prob = _probabilities(truth_counts)
    pred_prob = _probabilities(pred_counts)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ax.step(centers, truth_prob, where="mid", color="black", linewidth=2.2, label="ground truth")

    if member_values is None:
        label = f"{scenario.model} ({metric_name}={float(metric_values):.3g})"
        ax.step(centers, pred_prob, where="mid", color="#1f77b4", linewidth=1.8, label=label)
    else:
        for member, member_prob, member_metric in zip(member_values, pred_prob, metric_values):
            ax.step(
                centers,
                member_prob,
                where="mid",
                color="#1f77b4",
                linewidth=0.8,
                alpha=0.35,
                label=f"member {member} ({member_metric:.3g})",
            )
        ax.step(
            centers,
            np.nanmean(pred_prob, axis=0),
            where="mid",
            color="#d62728",
            linewidth=2.0,
            label=f"member mean ({metric_name}={metric_mean:.3g})",
        )

    if variable == "pr":
        ax.set_yscale("log")
        ax.set_xlabel("Precipitation [mm day-1]")
    else:
        ax.set_xlabel("tasmax [degC]")
    ax.set_ylabel("Histogram probability")
    ax.set_title(
        f"{scenario.model} | {scenario.domain} | {scenario.experiment}\n"
        f"{variable} distribution, {scenario.period}, {scenario.flavour}, {scenario.gcm}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def _season_subset(da: xr.DataArray, season: str, min_days: int) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError("Seasonal diagnostics require a time dimension.")
    season = season.upper()
    if season == "DJF":
        months_wanted = np.asarray([12, 1, 2])
    elif season == "JJA":
        months_wanted = np.asarray([6, 7, 8])
    else:
        raise ValueError(f"Unsupported season {season!r}; expected DJF or JJA.")

    months = np.asarray(da["time"].dt.month.values)
    years = np.asarray(da["time"].dt.year.values)
    mask = np.isin(months, months_wanted)
    if not mask.any():
        raise ValueError(f"No {season} days found in data.")

    season_year = years.copy()
    if season == "DJF":
        season_year = season_year + (months == 12)

    selected_years = season_year[mask]
    unique_years, counts = np.unique(selected_years, return_counts=True)
    valid_years = unique_years[counts >= min_days]
    if valid_years.size == 0:
        raise ValueError(f"No complete-enough {season} seasons found with min_days={min_days}.")

    keep = mask & np.isin(season_year, valid_years)
    return da.isel(time=np.flatnonzero(keep)).assign_coords(
        season_year=("time", season_year[keep].astype(np.int32))
    )


def _seasonal_field(
    da: xr.DataArray,
    *,
    variable: str,
    statistic: str,
    season: str,
    min_days: int,
) -> xr.DataArray:
    selected = _season_subset(da, season, min_days)
    grouped = selected.groupby("season_year")
    if statistic == "mean":
        out = grouped.mean("time", skipna=True).mean("season_year", skipna=True)
        out.name = f"{variable}_mean_{season}"
    elif statistic == "extreme":
        out = grouped.max("time", skipna=True).mean("season_year", skipna=True)
        out.name = "RX1DAY" if variable == "pr" else "TXX"
    else:
        raise ValueError(f"Unknown statistic {statistic!r}.")
    return _canonicalize_spatial_order(out)


def _canonicalize_spatial_order(da: xr.DataArray) -> xr.DataArray:
    leading = [dim for dim in da.dims if dim not in {"y", "x", "lat", "lon"}]
    if "y" in da.dims and "x" in da.dims:
        return da.transpose(*leading, "y", "x")
    if "lat" in da.dims and "lon" in da.dims:
        return da.transpose(*leading, "lat", "lon")
    return da


def _cc_signal(hist: xr.DataArray, future: xr.DataArray, *, variable: str) -> xr.DataArray:
    if variable == "pr":
        denom = hist.where(hist > 0.1)
        out = 100.0 * (future - hist) / denom
        out.attrs["units"] = "%"
        return out
    out = future - hist
    out.attrs["units"] = "degC"
    return out


def _align_to_truth(pred: xr.DataArray, truth: xr.DataArray, domain_key: str) -> xr.DataArray:
    pred = _canonicalize_spatial_order(pred)
    truth = _canonicalize_spatial_order(truth)
    try:
        if domain_key == "ALPS" and "y" in pred.dims and "x" in pred.dims:
            pred = pred.sel(y=truth["y"].values, x=truth["x"].values, method="nearest")
        elif "lat" in pred.dims and "lon" in pred.dims:
            pred = pred.sel(lat=truth["lat"].values, lon=truth["lon"].values, method="nearest")
    except Exception as exc:
        warnings.warn(f"Coordinate nearest-neighbor alignment failed; falling back to xarray.align: {exc}")

    pred, _ = xr.align(pred, truth, join="inner")
    return _canonicalize_spatial_order(pred)


def _cc_output_dir(
    args: argparse.Namespace,
    scenario: ClimateChangeScenario,
    variable: str,
    statistic: str,
    season: str,
) -> Path:
    return (
        args.out_root
        / "netcdf"
        / "climate_change"
        / scenario.model
        / scenario.domain
        / scenario.experiment
        / variable
        / statistic
        / season
        / f"{scenario.future_period}_minus_historical"
        / scenario.flavour
        / _safe_name(scenario.gcm)
    )


def _cc_plot_path(
    args: argparse.Namespace,
    scenario: ClimateChangeScenario,
    variable: str,
    statistic: str,
    season: str,
) -> Path:
    return (
        args.out_root
        / "plots"
        / "climate_change"
        / scenario.domain
        / scenario.experiment
        / variable
        / statistic
        / season
        / f"{scenario.future_period}_minus_historical"
        / scenario.flavour
        / _safe_name(scenario.gcm)
        / f"{_safe_name(scenario.model)}.png"
    )


def _compute_cc_for_scenario(
    args: argparse.Namespace,
    scenario: ClimateChangeScenario,
    variable: str,
    statistic: str,
    season: str,
) -> Path:
    metric_path = _cc_output_dir(args, scenario, variable, statistic, season) / "metric_cc_signal.nc"
    plot_path = _cc_plot_path(args, scenario, variable, statistic, season)
    if metric_path.exists() and (args.no_plots or plot_path.exists()) and not args.recompute:
        print(f"[reuse] {metric_path}")
        return metric_path

    min_days = args.djf_min_days if season == "DJF" else args.jja_min_days
    datasets: list[xr.Dataset] = []
    try:
        truth_hist_ds, truth_hist_da = _open_variable(
            scenario.truth_hist_path, variable, domain_key=scenario.domain_key, is_truth=True
        )
        truth_future_ds, truth_future_da = _open_variable(
            scenario.truth_future_path, variable, domain_key=scenario.domain_key, is_truth=True
        )
        pred_hist_ds, pred_hist_da = _open_variable(
            scenario.pred_hist_path, variable, domain_key=scenario.domain_key, is_truth=False
        )
        pred_future_ds, pred_future_da = _open_variable(
            scenario.pred_future_path, variable, domain_key=scenario.domain_key, is_truth=False
        )
        datasets.extend([truth_hist_ds, truth_future_ds, pred_hist_ds, pred_future_ds])

        truth_hist = _seasonal_field(
            truth_hist_da, variable=variable, statistic=statistic, season=season, min_days=min_days
        )
        truth_future = _seasonal_field(
            truth_future_da, variable=variable, statistic=statistic, season=season, min_days=min_days
        )
        pred_hist = _seasonal_field(
            pred_hist_da, variable=variable, statistic=statistic, season=season, min_days=min_days
        )
        pred_future = _seasonal_field(
            pred_future_da, variable=variable, statistic=statistic, season=season, min_days=min_days
        )

        truth_signal = _cc_signal(truth_hist, truth_future, variable=variable).load()
        pred_signal = _cc_signal(pred_hist, pred_future, variable=variable)
        pred_signal = _align_to_truth(pred_signal, truth_signal, scenario.domain_key).load()
        truth_signal = _canonicalize_spatial_order(truth_signal)

        error = (pred_signal - truth_signal).load()
    finally:
        for ds in datasets:
            ds.close()

    ds = _build_cc_dataset(
        variable=variable,
        statistic=statistic,
        season=season,
        scenario=scenario,
        truth_signal=truth_signal,
        pred_signal=pred_signal,
        error=error,
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(metric_path)
    ds.close()
    print(f"[write] {metric_path}")

    if not args.no_plots:
        _plot_cc_signal(
            output_path=plot_path,
            variable=variable,
            statistic=statistic,
            season=season,
            scenario=scenario,
            truth_signal=truth_signal,
            pred_signal=pred_signal,
            error=error,
        )
    return metric_path


def _member_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean("member", skipna=True) if "member" in da.dims else da


def _spatial_dims(da: xr.DataArray) -> list[str]:
    return [dim for dim in da.dims if dim != "member"]


def _spatial_rmse(error: xr.DataArray) -> xr.DataArray:
    dims = _spatial_dims(error)
    return np.sqrt((error**2).mean(dim=dims, skipna=True))


def _build_cc_dataset(
    *,
    variable: str,
    statistic: str,
    season: str,
    scenario: ClimateChangeScenario,
    truth_signal: xr.DataArray,
    pred_signal: xr.DataArray,
    error: xr.DataArray,
) -> xr.Dataset:
    metric_label = _metric_label(variable, statistic)
    units = "%" if variable == "pr" else "degC"
    truth_da = truth_signal.rename("truth_signal")
    pred_da = pred_signal.rename("model_signal")
    error_da = error.rename("signal_error")
    truth_da.attrs.update(long_name=f"Ground-truth {season} {metric_label} climate-change signal", units=units)
    pred_da.attrs.update(long_name=f"Model {season} {metric_label} climate-change signal", units=units)
    error_da.attrs.update(long_name=f"Model minus ground-truth {season} {metric_label} signal", units=units)

    data_vars: dict[str, xr.DataArray] = {
        "truth_signal": truth_da,
        "model_signal": pred_da,
        "signal_error": error_da,
        "model_signal_mean": _member_mean(pred_da).rename("model_signal_mean"),
        "signal_error_mean": _member_mean(error_da).rename("signal_error_mean"),
        "signal_error_rmse": _spatial_rmse(error_da).rename("signal_error_rmse"),
        "truth_signal_area_mean": _field_mean(truth_da).rename("truth_signal_area_mean"),
        "model_signal_area_mean": _field_mean(pred_da).rename("model_signal_area_mean"),
        "signal_error_area_mean": _field_mean(error_da).rename("signal_error_area_mean"),
    }
    if "member" in error_da.dims:
        data_vars["signal_error_rmse_mean"] = data_vars["signal_error_rmse"].mean("member", skipna=True)

    ds = xr.Dataset(data_vars)
    ds.attrs.update(
        diagnostic="climate_change",
        variable=variable,
        statistic=statistic,
        metric=metric_label,
        season=season,
        signal_units=units,
        model=scenario.model,
        domain=scenario.domain,
        experiment=scenario.experiment,
        future_period=scenario.future_period,
        predictor_flavour=scenario.flavour,
        gcm=scenario.gcm,
        pred_hist_path=str(scenario.pred_hist_path),
        pred_future_path=str(scenario.pred_future_path),
        truth_hist_path=str(scenario.truth_hist_path),
        truth_future_path=str(scenario.truth_future_path),
    )
    return ds


def _metric_label(variable: str, statistic: str) -> str:
    if statistic == "extreme":
        return "RX1DAY" if variable == "pr" else "TXX"
    return f"{variable}_mean"


def _field_mean(field: xr.DataArray) -> xr.DataArray:
    dims = _spatial_dims(field)
    return field.mean(dim=dims, skipna=True)


def _plot_cc_signal(
    *,
    output_path: Path,
    variable: str,
    statistic: str,
    season: str,
    scenario: ClimateChangeScenario,
    truth_signal: xr.DataArray,
    pred_signal: xr.DataArray,
    error: xr.DataArray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_mean = _member_mean(pred_signal)
    error_mean = _member_mean(error)

    panels = [
        ("Ground truth signal", truth_signal),
        ("Model signal", model_mean),
        ("Model - truth", error_mean),
    ]
    stack = np.concatenate([np.asarray(field.values, dtype=float).ravel() for _, field in panels])
    stack = stack[np.isfinite(stack)]
    if stack.size == 0:
        warnings.warn(f"No finite values to plot for {output_path}")
        return

    if variable == "pr":
        limit = max(float(np.nanpercentile(np.abs(stack), 98)), 1.0)
        ranges = [(-limit, limit), (-limit, limit), (-limit, limit)]
        cmap = "BrBG"
        units = "%"
    else:
        signal_stack = np.concatenate(
            [
                np.asarray(truth_signal.values, dtype=float).ravel(),
                np.asarray(model_mean.values, dtype=float).ravel(),
            ]
        )
        signal_stack = signal_stack[np.isfinite(signal_stack)]
        sig_min = min(0.0, float(np.nanpercentile(signal_stack, 2)))
        sig_max = max(1.0, float(np.nanpercentile(signal_stack, 98)))
        err_limit = max(float(np.nanpercentile(np.abs(np.asarray(error_mean.values).ravel()), 98)), 0.5)
        ranges = [(sig_min, sig_max), (sig_min, sig_max), (-err_limit, err_limit)]
        cmap = "coolwarm"
        units = "degC"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150, constrained_layout=True)
    for ax, (title, field), (vmin, vmax) in zip(axes, panels, ranges):
        image = _draw_field(ax, field, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.075, pad=0.08)
        cbar.set_label(units)

    fig.suptitle(
        f"{scenario.model} | {scenario.domain} | {scenario.experiment} | {scenario.gcm}\n"
        f"{season} {statistic} {variable}: {scenario.future_period} - historical ({scenario.flavour})"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def _draw_field(ax: plt.Axes, field: xr.DataArray, *, vmin: float, vmax: float, cmap: str):
    field = _canonicalize_spatial_order(field)
    if "lon" in field.coords and "lat" in field.coords:
        lon = np.asarray(field["lon"].values)
        lat = np.asarray(field["lat"].values)
        if lon.ndim == 2 and lat.ndim == 2:
            image = ax.pcolormesh(lon, lat, field.values, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
        else:
            image = ax.pcolormesh(lon, lat, field.values, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    else:
        image = ax.imshow(field.values, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return image


def _resolve_models(args: argparse.Namespace, catalog_entries: Sequence[CatalogEntry]) -> list[str]:
    available = [entry.name for entry in catalog_entries]
    if args.models:
        requested = list(dict.fromkeys(args.models))
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(f"Requested models not found in catalog: {missing}")
        models = requested
    else:
        models = available
    if args.max_models is not None:
        models = models[: args.max_models]
    return models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--truth-root", type=Path, default=DEFAULT_TRUTH_ROOT)
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--models", nargs="*", default=None, help="Model names to process; defaults to all catalog models.")
    parser.add_argument("--max-models", type=int, default=None, help="Process only the first N selected models.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["ALPS_Domain", "NZ_Domain", "SA_Domain"],
        help="Domain names or keys.",
    )
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    parser.add_argument("--future-periods", nargs="+", default=list(FUTURE_PERIODS), choices=list(FUTURE_PERIODS))
    parser.add_argument("--flavours", nargs="+", default=list(FLAVOURS), choices=list(FLAVOURS))
    parser.add_argument("--variables", nargs="+", default=list(VARIABLES), choices=list(VARIABLES))
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS), choices=list(SEASONS))
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=["distribution", "climate_change"],
        choices=["distribution", "climate_change"],
    )
    parser.add_argument("--gcm-mode", default="all", choices=["all", "primary", "evaluation"])
    parser.add_argument(
        "--tasmax-metric",
        default="perkins_skill_score",
        choices=["perkins_skill_score", "kl_divergence"],
        help="Distribution metric for tasmax.",
    )
    parser.add_argument("--pr-bins", type=int, default=60)
    parser.add_argument("--pr-lower", type=float, default=0.0)
    parser.add_argument("--pr-upper", type=float, default=600.0)
    parser.add_argument("--tasmax-bins", type=int, default=110)
    parser.add_argument("--tasmax-lower", type=float, default=-50.0)
    parser.add_argument("--tasmax-upper", type=float, default=60.0)
    parser.add_argument("--min-count", type=int, default=10, help="Minimum counts in both histograms for LHD bins.")
    parser.add_argument("--time-chunk", type=int, default=365, help="Number of time steps per histogram read.")
    parser.add_argument("--djf-min-days", type=int, default=80)
    parser.add_argument("--jja-min-days", type=int, default=80)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.truth_root = args.truth_root.expanduser().resolve()
    args.pred_root = args.pred_root.expanduser().resolve()
    args.catalog_path = args.catalog_path.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()

    entries = _load_catalog(args.catalog_path)
    models = _resolve_models(args, entries)
    print(f"[setup] models={len(models)} out_root={args.out_root}")

    if "distribution" in args.diagnostics:
        for scenario in _iter_distribution_scenarios(args, models):
            for variable in args.variables:
                try:
                    _compute_distribution_for_scenario(args, scenario, variable)
                except Exception as exc:
                    print(
                        f"[error] distribution {scenario.model}|{scenario.domain}|{scenario.experiment}|"
                        f"{scenario.period}|{scenario.flavour}|{scenario.gcm}|{variable}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    if "climate_change" in args.diagnostics:
        for scenario in _iter_cc_scenarios(args, models):
            for variable in args.variables:
                statistics = ["mean", "extreme"]
                for statistic in statistics:
                    for season in args.seasons:
                        try:
                            _compute_cc_for_scenario(args, scenario, variable, statistic, season)
                        except Exception as exc:
                            print(
                                f"[error] climate_change {scenario.model}|{scenario.domain}|{scenario.experiment}|"
                                f"{scenario.future_period}|{scenario.flavour}|{scenario.gcm}|"
                                f"{variable}|{statistic}|{season}: {type(exc).__name__}: {exc}"
                            )


if __name__ == "__main__":
    main()
