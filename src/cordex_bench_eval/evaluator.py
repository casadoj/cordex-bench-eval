"""
Evaluator for computing metrics on model predictions.

This module provides the Evaluator class which computes configurable metrics
on prediction datasets against ground truth, and saves results as NetCDF files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import numpy as np
import re
from itertools import product
from pathlib import Path
from typing import Any, Callable, List, Set

import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader

_VALID_SEASONS = frozenset({"winter", "summer", "spring", "autumn"})

class Evaluator:
    """
    Compute evaluation metrics for model predictions against ground truth data.
    
    Computes metrics defined in a JSON registry and saves each metric as a
    separate NetCDF file under the diagnostic_path directory.
    
    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Configuration containing paths including diagnostic_path.
    catalog : SubmissionCatalog
        Submission catalog for loading predictions.
    metrics_registry_path : str | Path
        Path to JSON file defining available metrics.
    ml_benchmark_path : str | Path, optional
        Path to ml-benchmark evaluation module (adds to sys.path).
    """

    def __init__(self, config: SubmissionWorkflowConfig, catalog: SubmissionCatalog,
                 metrics_registry_path: str | Path, ml_benchmark_path: str | Path | None = None):
        if config.diagnostic_path is None:
            raise ValueError("diagnostic_path must be set in config.")

        self.config = config
        self.catalog = catalog
        self.metrics_registry = _load_metrics_registry(metrics_registry_path)
        self.submission_loader = SubmissionDataLoader(catalog)
        self.gt_loader = GroundTruthDataLoader(config.groundtruth_path)

        # Add ml-benchmark path to sys.path if provided
        if ml_benchmark_path:
            import sys
            sys.path.insert(0, str(Path(ml_benchmark_path).expanduser().resolve()))

    def evaluate(self, submission_name: str, domain: str, training_experiment: str,
                 evaluation_experiment: str, variable: str, metrics: List[str] | None = None,
                 season: str | None = None, recompute_existing: bool = False,
                 **open_kwargs: Any) -> dict[str, list[str]]:
        """
        Compute metrics for a prediction and save results.
        
        Parameters
        ----------
        submission_name : str
            Name of the submission (catalog entry).
        domain : str
            Domain name (e.g., "ALPS_Domain").
        training_experiment : str
            Training experiment (e.g., "Emulator_hist_future", "ESD_pseudo_reality").
        evaluation_experiment : str
            Evaluation experiment (e.g., "Perfect_cross_validation").
        variable : str
            Variable name (e.g., "pr", "tasmax").
        metrics : list of str, optional
            Metric names to compute. If None, uses defaults from registry.
        season : str, optional
            Season filter ("winter", "summer", "spring", "autumn").
        recompute_existing : bool, optional
            If True, recompute and overwrite metric files even when they already
            exist. If False (default), existing metric files are reused without
            being loaded into memory.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        dict[str, list[str]]
            Lightweight execution summary with metric names grouped by status:
            ``computed``, ``reused``, and ``failed``. Metric datasets are always
            saved to disk and are not retained in memory by this method.
        """
        if metrics is None:
            metrics = self.metrics_registry.get("defaults", {}).get(variable, [])
        vars_needed = self._variables_needed_for_metrics(metrics, variable)
        load_variable: str | None = None if len(vars_needed) > 1 else variable
        total_metrics = len(metrics)
        context_label = (
            f"{submission_name}|{domain}|{training_experiment}|{evaluation_experiment}|{variable}"
        )
        print(
            f"[Evaluator] Start context {context_label} "
            f"(metrics={total_metrics}, season={season})"
        )
        pred_ds: xr.Dataset | None = None
        gt_ds: xr.Dataset | None = None

        # Compute each metric without retaining metric datasets in memory.
        computed_metrics: list[str] = []
        reused_metrics: list[str] = []
        failed_metrics: list[str] = []
        for metric_index, metric_name in enumerate(metrics, start=1):
            metric_path = self._metric_path(submission_name=submission_name,
                                            domain=domain,
                                            training_experiment=training_experiment,
                                            evaluation_experiment=evaluation_experiment,
                                            variable=variable,
                                            metric_name=metric_name,
                                            season=season)
            try:
                if metric_name not in self.metrics_registry.get("metrics", {}):
                    raise ValueError(f"Unknown metric: {metric_name}")

                if metric_path.exists() and not recompute_existing:
                    reused_metrics.append(metric_name)
                    print(
                        f"[Evaluator] [{context_label}] "
                        f"Metric {metric_index}/{total_metrics}: {metric_name} "
                        f"(loaded existing: {metric_path})"
                    )
                    continue

                print(
                    f"[Evaluator] [{context_label}] "
                    f"Metric {metric_index}/{total_metrics}: {metric_name} (start)"
                )
                if pred_ds is None or gt_ds is None:
                    print(f"[Evaluator] Loading datasets for {context_label}")
                    pred_ds, gt_ds = self._load_and_align_datasets(submission_name=submission_name,
                                                                   domain=domain,
                                                                   training_experiment=training_experiment,
                                                                   evaluation_experiment=evaluation_experiment,
                                                                   context_variable=variable,
                                                                   load_variable=load_variable,
                                                                   **open_kwargs)
                    print(f"[Evaluator] Datasets loaded and aligned for {context_label}")

                metric_config = self.metrics_registry["metrics"][metric_name]
                member_handling = metric_config.get("member_handling", "mean")
                selection_context = "|".join([
                    submission_name,
                    domain,
                    training_experiment,
                    evaluation_experiment,
                    variable,
                    metric_name,
                ])
                metric_pred_ds, metric_gt_ds = self._apply_member_handling_to_datasets(
                    pred_ds=pred_ds,
                    gt_ds=gt_ds,
                    member_handling=member_handling,
                    selection_context=selection_context,
                )
                self._validate_metric_inputs(metric_pred_ds,
                                             metric_gt_ds,
                                             variable=variable,
                                             metric_config=metric_config,
                                             context_label=selection_context)

                metric_ds = self._compute_metric(pred_ds=metric_pred_ds, gt_ds=metric_gt_ds, variable=variable,
                                                 metric_name=metric_name, metric_config=metric_config,
                                                 season=season)

                # Save metric to NetCDF
                metric_path.parent.mkdir(parents=True, exist_ok=True)
                metric_ds.to_netcdf(metric_path)
                print(
                    f"[Evaluator] [{context_label}] "
                    f"Metric {metric_index}/{total_metrics}: {metric_name} (saved: {metric_path})"
                )

                computed_metrics.append(metric_name)
            except Exception as exc:
                failed_metrics.append(metric_name)
                failed_file = self._extract_failed_file_path(exc)
                print(
                    f"[Evaluator][ERROR] [{context_label}] "
                    f"Metric {metric_index}/{total_metrics}: {metric_name} failed "
                    f"({type(exc).__name__}: {exc})"
                )
                print(
                    f"[Evaluator][ERROR] [{context_label}] "
                    f"Metric {metric_name} output path: {metric_path}"
                )
                if failed_file is not None:
                    print(
                        f"[Evaluator][ERROR] [{context_label}] "
                        f"Metric {metric_name} failed file: {failed_file}"
                    )
                continue

        if pred_ds is not None:
            pred_ds.close()
        if gt_ds is not None:
            gt_ds.close()

        success_count = len(computed_metrics) + len(reused_metrics)
        if failed_metrics:
            print(
                f"[Evaluator] Completed context {context_label} "
                f"(success={success_count}/{total_metrics}, "
                f"computed={len(computed_metrics)}, reused={len(reused_metrics)}, "
                f"failed={len(failed_metrics)}: {', '.join(failed_metrics)})"
            )
        else:
            print(
                f"[Evaluator] Completed context {context_label} "
                f"(success={success_count}/{total_metrics}, "
                f"computed={len(computed_metrics)}, reused={len(reused_metrics)}, failed=0)"
            )
        return {
            "computed": computed_metrics,
            "reused": reused_metrics,
            "failed": failed_metrics,
        }

    def evaluate_multiple(self, submission_names: List[str], domain: str | List[str],
                          training_experiment: str | List[str],
                          evaluation_experiment: str | List[str],
                          variable: str, metrics: List[str] | None = None,
                          season: str | None = None, recompute_existing: bool = False,
                          **open_kwargs: Any) -> dict[str, dict[str, list[str]]]:
        """
        Compute metrics for multiple submissions and save results.

        Parameters
        ----------
        submission_names : list of str
            Names of submissions (catalog entries).
        domain : str or sequence of str
            Domain name(s) (e.g., "ALPS_Domain").
        training_experiment : str or sequence of str
            Training experiment(s) (e.g., "Emulator_hist_future", "ESD_pseudo_reality").
        evaluation_experiment : str or sequence of str
            Evaluation experiment(s) (e.g., "Perfect_cross_validation").
        variable : str
            Variable name (e.g., "pr", "tasmax").
        metrics : list of str, optional
            Metric names to compute. If None, uses defaults from registry.
        season : str, optional
            Season filter ("winter", "summer", "spring", "autumn").
        recompute_existing : bool, optional
            If True, recompute and overwrite metric files even when they already
            exist. If False (default), existing metric files are reused without
            being loaded into memory.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Dictionary mapping:
            - submission names to per-context execution summaries when single values are provided
            - composite context keys
              "{submission}|{domain}|{training_experiment}|{evaluation_experiment}|{variable}"
              to per-context execution summaries when any multi-value input is provided
        """
        def _as_list(value: str | List[str]) -> List[str]:
            if isinstance(value, str):
                return [value]
            return list(value)

        domains = _as_list(domain)
        training_experiments = _as_list(training_experiment)
        evaluation_experiments = _as_list(evaluation_experiment)

        multi_context = any([
            len(domains) > 1,
            len(training_experiments) > 1,
            len(evaluation_experiments) > 1,
        ])
        total_contexts = (
            len(submission_names)
            * len(domains)
            * len(training_experiments)
            * len(evaluation_experiments)
        )
        print(
            f"[Evaluator] Starting evaluation: contexts={total_contexts}, "
            f"variable={variable}, season={season}"
        )

        results: dict[str, dict[str, list[str]]] = {}
        for context_index, (submission_name, domain_name, training_name, evaluation_name) in enumerate(
            product(submission_names, domains, training_experiments, evaluation_experiments), start=1
        ):
            print(
                f"[Evaluator] Context {context_index}/{total_contexts}: "
                f"{submission_name}|{domain_name}|{training_name}|{evaluation_name}|{variable} (start)"
            )
            if multi_context:
                key = "|".join([submission_name, domain_name, training_name, evaluation_name, variable])
            else:
                key = submission_name

            results[key] = self.evaluate(submission_name=submission_name,
                                         domain=domain_name,
                                         training_experiment=training_name,
                                         evaluation_experiment=evaluation_name,
                                         variable=variable,
                                         metrics=metrics,
                                         season=season,
                                         recompute_existing=recompute_existing,
                                         **open_kwargs)
            print(f"[Evaluator] Context {context_index}/{total_contexts}: {key} (done)")
        print("[Evaluator] Evaluation finished.")
        return results

    def available_metric_files(self, submission_name: str, domain: str, training_experiment: str,
                               evaluation_experiment: str, variable: str,
                               season: str | None = None) -> List[str]:
        """
        List saved metric files for a prediction.

        Parameters
        ----------
        season : str, optional
            Season filter ("winter", "summer", "spring", "autumn"). If None,
            list annual (non-seasonal) metric files.

        Returns
        -------
        list of str
            Metric filenames (without path).
        """
        metric_dir = self._metric_dir(submission_name=submission_name,
                                      domain=domain,
                                      training_experiment=training_experiment,
                                      evaluation_experiment=evaluation_experiment,
                                      variable=variable) 

        if not metric_dir.exists():
            return []

        if season is None:
            return sorted([
                f.name
                for f in metric_dir.glob("metric_*.nc")
                if not any(f.stem.endswith(f"_{known_season}") for known_season in _VALID_SEASONS)
            ])

        season_token = str(season).strip().lower()
        if not season_token:
            return sorted([
                f.name
                for f in metric_dir.glob("metric_*.nc")
                if not any(f.stem.endswith(f"_{known_season}") for known_season in _VALID_SEASONS)
            ])
        return sorted([f.name for f in metric_dir.glob(f"metric_*_{season_token}.nc")])

    def load_metric(self, submission_name: str, domain: str, training_experiment: str,
                    evaluation_experiment: str, variable: str, metric_name: str,
                    season: str | None = None,
                    **open_kwargs: Any) -> xr.Dataset:
        """
        Load a previously saved metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to load.
        season : str, optional
            Season filter ("winter", "summer", "spring", "autumn"). If None,
            loads annual metric file.
        
        Returns
        -------
        xr.Dataset
            Loaded metric dataset.
        """
        metric_path = self._metric_path(submission_name=submission_name,
                                        domain=domain,
                                        training_experiment=training_experiment,
                                        evaluation_experiment=evaluation_experiment,
                                        variable=variable,
                                        metric_name=metric_name,
                                        season=season)

        if not metric_path.exists():
            raise FileNotFoundError(f"Metric file not found: {metric_path}")

        return xr.open_dataset(metric_path, **open_kwargs)

    def _variables_needed_for_metrics(self, metrics: List[str], context_variable: str) -> Set[str]:
        """Union of the context variable and any variables required by multivariable metrics."""
        needed: Set[str] = {context_variable}
        registry = self.metrics_registry.get("metrics", {})
        for name in metrics:
            cfg = registry.get(name, {})
            if cfg.get("type") == "bias_multivariable_correlation":
                if "var_x" not in cfg or "var_y" not in cfg:
                    raise ValueError(
                        f"Metric '{name}' of type 'bias_multivariable_correlation' "
                        "requires 'var_x' and 'var_y'."
                    )
                needed.add(cfg["var_x"])
                needed.add(cfg["var_y"])
        return needed

    def _load_and_align_datasets(self, submission_name: str, domain: str,
                                 training_experiment: str, evaluation_experiment: str,
                                 context_variable: str,
                                 load_variable: str | None,
                                 **open_kwargs: Any) -> tuple[xr.Dataset, xr.Dataset]:
        """Load prediction and ground truth datasets, aligned on coordinates.

        Parameters
        ----------
        context_variable
            Evaluation context variable (used in log labels and paths elsewhere).
        load_variable
            Variable name passed to loaders, or ``None`` to load all variables in the
            file (e.g. both ``pr`` and ``tasmax`` when a multivariable metric is requested).
        """
        context_label = (
            f"{submission_name}|{domain}|{training_experiment}|{evaluation_experiment}|{context_variable}"
        )
        # Load prediction
        pred_ds = self.submission_loader.load_dataset(submission_name=submission_name,
                                                      domain=domain,
                                                      training_experiment=training_experiment,
                                                      evaluation_experiment=evaluation_experiment,
                                                      variable=load_variable,
                                                      **open_kwargs)

        # Load ground truth
        gt_ds = self.gt_loader.load_test_target(domain=domain,
                                                training_experiment=training_experiment,
                                                evaluation_experiment=evaluation_experiment,
                                                variable=load_variable,
                                                merge_mid_end=True,
                                                **open_kwargs)

        pred_ds, gt_ds = self._align_prediction_and_ground_truth(pred_ds=pred_ds,
                                                                 gt_ds=gt_ds,
                                                                 context_label=context_label)

        return pred_ds, gt_ds

    def _apply_member_handling_to_datasets(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset,
                                           member_handling: str, selection_context: str) -> tuple[xr.Dataset, xr.Dataset]:
        """Apply member-dimension handling to already-aligned datasets."""
        normalized = self._normalize_member_handling(member_handling)
        pred_processed = self._reduce_member_dim(pred_ds,
                                                 strategy=normalized,
                                                 selection_context=f"{selection_context}|pred")
        gt_processed = self._reduce_member_dim(gt_ds,
                                               strategy=normalized,
                                               selection_context=f"{selection_context}|gt")
        # Datasets are aligned once in _load_and_align_datasets; reducing member does not
        # change time/lat/lon coordinates, so re-aligning here is unnecessary overhead.
        return pred_processed, gt_processed

    def _align_prediction_and_ground_truth(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset,
                                           context_label: str) -> tuple[xr.Dataset, xr.Dataset]:
        """Normalize and align prediction/ground-truth datasets with fast-path + fallback."""
        pred_norm = self._normalize_alignment_coords(pred_ds)
        gt_norm = self._normalize_alignment_coords(gt_ds)

        # Fast path (close to original behavior): align directly and keep going when valid.
        pred_fast, gt_fast = xr.align(pred_norm, gt_norm, join="inner", exclude=["time"])
        pred_fast, gt_fast = xr.align(pred_fast, gt_fast, join="inner")
        if self._is_valid_alignment(pred_fast, gt_fast):
            return pred_fast, gt_fast

        # Fallback path: explicit day-level matching for problematic calendar/encoding cases.
        pred_aligned, gt_aligned = xr.align(pred_norm, gt_norm, join="inner", exclude=["time"])
        pred_aligned, gt_aligned = self._align_common_time(pred_aligned, gt_aligned)
        pred_aligned, gt_aligned = xr.align(pred_aligned, gt_aligned, join="inner")
        self._validate_aligned_datasets(pred_aligned, gt_aligned, context_label=context_label)
        return pred_aligned, gt_aligned

    def _normalize_alignment_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalize time and spatial coordinates to improve robust alignment."""
        normalized = ds

        for coord_name in ("lat", "latitude", "lon", "longitude"):
            if coord_name in normalized.coords and coord_name in normalized.dims:
                coord = normalized[coord_name]
                if np.issubdtype(coord.dtype, np.number):
                    normalized = normalized.assign_coords({coord_name: coord.round(6)})
        return normalized

    def _align_common_time(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
        """Align datasets on common calendar days while preserving datetime time coordinates."""
        if "time" not in pred_ds.dims or "time" not in gt_ds.dims:
            return pred_ds, gt_ds

        pred_time_values = pred_ds["time"].values.tolist()
        gt_time_values = gt_ds["time"].values.tolist()

        pred_day_tokens = [self._to_day_token(value) for value in pred_time_values]
        gt_day_tokens = [self._to_day_token(value) for value in gt_time_values]
        gt_tokens_set = set(gt_day_tokens)

        common_tokens: list[str] = []
        pred_indices: list[int] = []
        gt_first_index: dict[str, int] = {}
        for idx, token in enumerate(gt_day_tokens):
            gt_first_index.setdefault(token, idx)

        seen_tokens: set[str] = set()
        for idx, token in enumerate(pred_day_tokens):
            if token in gt_tokens_set and token not in seen_tokens:
                seen_tokens.add(token)
                common_tokens.append(token)
                pred_indices.append(idx)

        if not common_tokens:
            # Fallback: some files decode to different absolute calendars/origins but still
            # represent the same daily sequence. If lengths match, align positionally.
            pred_len = pred_ds.sizes["time"]
            gt_len = gt_ds.sizes["time"]
            if pred_len > 0 and pred_len == gt_len:
                gt_canonical_tokens = [self._to_day_token(value) for value in gt_time_values]
                canonical_time = np.array(gt_canonical_tokens, dtype="datetime64[D]")
                pred_fallback = pred_ds.assign_coords(time=canonical_time)
                gt_fallback = gt_ds.assign_coords(time=canonical_time)
                return pred_fallback, gt_fallback
            return pred_ds.isel(time=slice(0, 0)), gt_ds.isel(time=slice(0, 0))

        gt_indices = [gt_first_index[token] for token in common_tokens]
        pred_aligned = pred_ds.isel(time=pred_indices)
        gt_aligned = gt_ds.isel(time=gt_indices)

        canonical_time = np.array(common_tokens, dtype="datetime64[D]")
        pred_aligned = pred_aligned.assign_coords(time=canonical_time)
        gt_aligned = gt_aligned.assign_coords(time=canonical_time)
        return pred_aligned, gt_aligned

    @staticmethod
    def _to_day_token(value: Any) -> str:
        """Convert various datetime-like objects to a canonical YYYY-MM-DD token."""
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit="D")
        if hasattr(value, "strftime"):
            return str(value.strftime("%Y-%m-%d"))
        return str(value)[:10]

    def _validate_aligned_datasets(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset, context_label: str) -> None:
        """Validate aligned datasets before metric computation."""
        if "time" in pred_ds.dims and pred_ds.sizes["time"] == 0:
            raise ValueError(f"Empty time overlap after alignment for context '{context_label}' (prediction).")
        if "time" in gt_ds.dims and gt_ds.sizes["time"] == 0:
            raise ValueError(f"Empty time overlap after alignment for context '{context_label}' (ground truth).")

        for dim_name in ("lat", "latitude", "lon", "longitude"):
            if dim_name in pred_ds.dims and dim_name in gt_ds.dims:
                pred_size = pred_ds.sizes[dim_name]
                gt_size = gt_ds.sizes[dim_name]
                if pred_size == 0 or gt_size == 0:
                    raise ValueError(
                        f"Empty spatial dimension '{dim_name}' after alignment for context '{context_label}' "
                        f"(pred={pred_size}, gt={gt_size})."
                    )
                if pred_size != gt_size:
                    raise ValueError(
                        f"Mismatched spatial dimension '{dim_name}' after alignment for context '{context_label}' "
                        f"(pred={pred_size}, gt={gt_size})."
                    )

    @staticmethod
    def _is_valid_alignment(pred_ds: xr.Dataset, gt_ds: xr.Dataset) -> bool:
        """Fast non-raising validity check for alignment results."""
        if "time" in pred_ds.dims and pred_ds.sizes["time"] == 0:
            return False
        if "time" in gt_ds.dims and gt_ds.sizes["time"] == 0:
            return False
        for dim_name in ("lat", "latitude", "lon", "longitude"):
            if dim_name in pred_ds.dims and dim_name in gt_ds.dims:
                pred_size = pred_ds.sizes[dim_name]
                gt_size = gt_ds.sizes[dim_name]
                if pred_size == 0 or gt_size == 0 or pred_size != gt_size:
                    return False
        return True

    def _validate_metric_inputs(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset,
                                variable: str, metric_config: dict,
                                context_label: str) -> None:
        """Ensure metric inputs have non-empty time and consistent spatial dimensions."""
        self._validate_aligned_datasets(pred_ds, gt_ds, context_label=context_label)

        mtype = metric_config.get("type")
        if mtype == "bias_multivariable_correlation":
            try:
                vars_to_check = [metric_config["var_x"], metric_config["var_y"]]
            except KeyError as exc:
                raise ValueError(
                    "Metric type 'bias_multivariable_correlation' requires "
                    "'var_x' and 'var_y' in the metric configuration."
                ) from exc
        else:
            vars_to_check = [variable]

        for check_var in vars_to_check:
            for ds_label, ds in (("prediction", pred_ds), ("ground_truth", gt_ds)):
                if check_var not in ds.data_vars:
                    raise KeyError(
                        f"Variable '{check_var}' not found in {ds_label} dataset "
                        f"for context '{context_label}'."
                    )
                var_da = ds[check_var]
                for dim_name in ("lat", "latitude", "lon", "longitude"):
                    if dim_name in var_da.dims and dim_name in ds.dims:
                        if var_da.sizes[dim_name] != ds.sizes[dim_name]:
                            raise ValueError(
                                f"Inconsistent dimension '{dim_name}' for variable '{check_var}' "
                                f"in {ds_label} dataset for context '{context_label}' "
                                f"(variable={var_da.sizes[dim_name]}, dataset={ds.sizes[dim_name]})."
                            )

    def _reduce_member_dim(self, ds: xr.Dataset, strategy: str, selection_context: str) -> xr.Dataset:
        """Reduce/select the member dimension when present."""
        if "member" not in ds.dims:
            return ds

        if strategy == "mean":
            return ds.mean(dim="member")

        if strategy == "random":
            idx = self._deterministic_member_index(ds.sizes["member"], selection_context)
            return ds.isel(member=idx, drop=True)

        raise ValueError(f"Unknown member handling strategy: {strategy}")

    def _normalize_member_handling(self, member_handling: str) -> str:
        """Normalize accepted member handling aliases."""
        normalized = str(member_handling).strip().lower()
        if normalized in {"mean", "ensemble_mean"}:
            return "mean"
        if normalized in {"random", "random_member"}:
            return "random"
        raise ValueError(
            f"Unsupported member_handling='{member_handling}'. Expected one of "
            "'mean', 'ensemble_mean', 'random', or 'random_member'."
        )

    def _deterministic_member_index(self, n_members: int, selection_context: str) -> int:
        """Compute a deterministic member index in [0, n_members)."""
        if n_members <= 0:
            raise ValueError("Member dimension must contain at least one value.")
        digest = hashlib.sha256(selection_context.encode("utf-8")).hexdigest()
        return int(digest, 16) % n_members

    @staticmethod
    def _extract_failed_file_path(exc: Exception) -> str | None:
        """Try to extract a path-like token from an exception message."""
        message = str(exc)
        quoted_matches = re.findall(r"['\"]([^'\"]+)['\"]", message)
        for candidate in quoted_matches:
            if "/" in candidate or "\\" in candidate:
                return candidate

        if ":" in message:
            tail = message.rsplit(":", maxsplit=1)[-1].strip()
            if "/" in tail or "\\" in tail:
                return tail
        return None

    def _compute_metric(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset,
                        variable: str, metric_name: str, metric_config: dict,
                        season: str | None = None) -> xr.Dataset:
        """Compute a single metric using the appropriate diagnostic function."""
        metric_type = metric_config["type"]

        # Import ml-benchmark diagnostics module
        try:
            import evaluation.diagnostics as diagnostics
        except ImportError:
            raise ImportError("Could not import ml-benchmark evaluation modules.")

        if metric_type == "rmse":
            dim = metric_config.get("dim")
            return diagnostics.rmse(x0=gt_ds, x1=pred_ds,
                                    var=variable, season=season,
                                    dim=dim)

        elif metric_type == "psd":
            psd_x0, psd_x1 = diagnostics.psd(x0=gt_ds, x1=pred_ds,
                                             var=variable, season=season)
            return xr.Dataset({"PSD_x0": psd_x0, "PSD_x1": psd_x1})

        elif metric_type == "bias_index":
            index_fn = _import_function(metric_config["index_fn"])
            index_kwargs = metric_config.get("kwargs", {}).copy()
            index_kwargs["var"] = variable
            return diagnostics.bias_index(x0=gt_ds, x1=pred_ds,
                                          index_fn=index_fn, season=season,
                                          **index_kwargs)

        elif metric_type == "ratio_index":
            index_fn = _import_function(metric_config["index_fn"])
            index_kwargs = metric_config.get("kwargs", {}).copy()
            index_kwargs["var"] = variable
            return diagnostics.ratio_index(x0=gt_ds, x1=pred_ds,
                                          index_fn=index_fn, season=season,
                                          **index_kwargs)

        elif metric_type == "bias_multivariable_correlation":
            try:
                var_x = metric_config["var_x"]
                var_y = metric_config["var_y"]
            except KeyError as exc:
                raise ValueError(
                    "Metric type 'bias_multivariable_correlation' requires "
                    "'var_x' and 'var_y' in the metric configuration."
                ) from exc
            return diagnostics.bias_multivariable_correlation(
                x0=gt_ds,
                x1=pred_ds,
                var_x=var_x,
                var_y=var_y,
                season=season,
            )

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def _metric_dir(self, submission_name: str, domain: str, training_experiment: str,
                    evaluation_experiment: str, variable: str) -> Path:
        """Build directory path for storing metrics."""
        from .data_loader import _normalize_training_experiment
        experiment_name = _normalize_training_experiment(training_experiment)
        return (self.config.diagnostic_path
                / submission_name
                / domain
                / experiment_name
                / variable
                / evaluation_experiment)

    def _metric_path(self, submission_name: str, domain: str, training_experiment: str,
                     evaluation_experiment: str, variable: str, metric_name: str,
                     season: str | None = None) -> Path:
        """Build file path for a metric NetCDF."""
        metric_dir = self._metric_dir(submission_name=submission_name,
                                      domain=domain,
                                      training_experiment=training_experiment,
                                      evaluation_experiment=evaluation_experiment,
                                      variable=variable)
        if season is None:
            return metric_dir / f"metric_{metric_name}.nc"

        season_token = str(season).strip().lower()
        if not season_token:
            return metric_dir / f"metric_{metric_name}.nc"
        return metric_dir / f"metric_{metric_name}_{season_token}.nc"


def _load_metrics_registry(metrics_path: str | Path) -> dict:
    """Load metrics configuration from JSON file."""
    path = Path(metrics_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metrics registry not found: {path}")
    return json.loads(path.read_text())


def _import_function(module_path: str) -> Callable:
    """
    Import a function from a dotted module path.
    
    Example: "evaluation.indices.mean" -> imports mean from evaluation.indices
    """
    module_name, func_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)
