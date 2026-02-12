"""
Evaluator for computing metrics on model predictions.

This module provides the Evaluator class which computes configurable metrics
on prediction datasets against ground truth, and saves results as NetCDF files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Callable, List

import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader

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
                 **open_kwargs: Any) -> dict[str, xr.Dataset]:
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
            exist. If False (default), existing metric files are loaded and
            reused.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        dict[str, xr.Dataset]
            Dictionary mapping metric names to computed result datasets.
        """
        if metrics is None:
            metrics = self.metrics_registry.get("defaults", {}).get(variable, [])
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

        # Compute each metric
        results = {}
        for metric_index, metric_name in enumerate(metrics, start=1):
            if metric_name not in self.metrics_registry.get("metrics", {}):
                raise ValueError(f"Unknown metric: {metric_name}")

            metric_path = self._metric_path(submission_name=submission_name,
                                            domain=domain,
                                            training_experiment=training_experiment,
                                            evaluation_experiment=evaluation_experiment,
                                            variable=variable,
                                            metric_name=metric_name)
            if metric_path.exists() and not recompute_existing:
                with xr.open_dataset(metric_path, **open_kwargs) as cached_ds:
                    results[metric_name] = cached_ds.load()
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
                                                               variable=variable,
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

            results[metric_name] = metric_ds

        print(f"[Evaluator] Completed context {context_label}")
        return results

    def evaluate_multiple(self, submission_names: List[str], domain: str | List[str],
                          training_experiment: str | List[str],
                          evaluation_experiment: str | List[str],
                          variable: str, metrics: List[str] | None = None,
                          season: str | None = None, recompute_existing: bool = False,
                          **open_kwargs: Any) -> dict[str, dict[str, xr.Dataset]]:
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
            exist. If False (default), existing metric files are loaded and
            reused.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().

        Returns
        -------
        dict[str, dict[str, xr.Dataset]]
            Dictionary mapping:
            - submission names to metric result dictionaries when single values are provided
            - composite context keys
              "{submission}|{domain}|{training_experiment}|{evaluation_experiment}|{variable}"
              to metric result dictionaries when any multi-value input is provided
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

        results: dict[str, dict[str, xr.Dataset]] = {}
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
                               evaluation_experiment: str, variable: str) -> List[str]:
        """
        List saved metric files for a prediction.
        
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

        return sorted([f.name for f in metric_dir.glob("metric_*.nc")])

    def load_metric(self, submission_name: str, domain: str, training_experiment: str,
                    evaluation_experiment: str, variable: str, metric_name: str,
                    **open_kwargs: Any) -> xr.Dataset:
        """
        Load a previously saved metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to load.
        
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
                                        metric_name=metric_name)

        if not metric_path.exists():
            raise FileNotFoundError(f"Metric file not found: {metric_path}")

        return xr.open_dataset(metric_path, **open_kwargs)

    def _load_and_align_datasets(self, submission_name: str, domain: str,
                                 training_experiment: str, evaluation_experiment: str,
                                 variable: str, **open_kwargs: Any) -> tuple[xr.Dataset, xr.Dataset]:
        """Load prediction and ground truth datasets, aligned on coordinates."""
        # Load prediction
        pred_ds = self.submission_loader.load_dataset(submission_name=submission_name,
                                                      domain=domain,
                                                      training_experiment=training_experiment,
                                                      evaluation_experiment=evaluation_experiment,
                                                      variable=variable,
                                                      **open_kwargs)

        # Load ground truth
        gt_ds = self.gt_loader.load_test_target(domain=domain,
                                                training_experiment=training_experiment,
                                                evaluation_experiment=evaluation_experiment,
                                                variable=variable,
                                                merge_mid_end=True,
                                                **open_kwargs)

        # Align on spatial coordinates first, then time
        pred_ds, gt_ds = xr.align(pred_ds, gt_ds, join="inner", exclude=["time"])
        pred_ds, gt_ds = xr.align(pred_ds, gt_ds, join="inner")

        return pred_ds, gt_ds

    def _apply_member_handling_to_datasets(self, pred_ds: xr.Dataset, gt_ds: xr.Dataset,
                                           member_handling: str, selection_context: str) -> tuple[xr.Dataset, xr.Dataset]:
        """Apply member-dimension handling to prediction/ground-truth datasets."""
        normalized = self._normalize_member_handling(member_handling)
        pred_processed = self._reduce_member_dim(pred_ds,
                                                 strategy=normalized,
                                                 selection_context=f"{selection_context}|pred")
        gt_processed = self._reduce_member_dim(gt_ds,
                                               strategy=normalized,
                                               selection_context=f"{selection_context}|gt")
        pred_processed, gt_processed = xr.align(pred_processed, gt_processed, join="inner", exclude=["time"])
        pred_processed, gt_processed = xr.align(pred_processed, gt_processed, join="inner")
        return pred_processed, gt_processed

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
                     evaluation_experiment: str, variable: str, metric_name: str) -> Path:
        """Build file path for a metric NetCDF."""
        metric_dir = self._metric_dir(submission_name=submission_name,
                                      domain=domain,
                                      training_experiment=training_experiment,
                                      evaluation_experiment=evaluation_experiment,
                                      variable=variable)
                                      
        return metric_dir / f"metric_{metric_name}.nc"


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
