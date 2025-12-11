"""
Evaluator for computing metrics on model predictions.

This module provides the Evaluator class which computes configurable metrics
on prediction datasets against ground truth, and saves results as NetCDF files.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List

import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .data_loader import GroundTruthDataLoader, SubmissionDataLoader


# =============================================================================
# EVALUATOR
# =============================================================================


class Evaluator:
    """
    Compute evaluation metrics for model predictions.
    
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
    
    Example
    -------
    >>> evaluator = Evaluator(
    ...     config=config,
    ...     catalog=catalog,
    ...     metrics_registry_path="configs/metrics.json",
    ...     ml_benchmark_path="/code/ml-benchmark",
    ... )
    >>> results = evaluator.evaluate(
    ...     submission_name="DeepESD-v1_IFCA_m1",
    ...     domain="ALPS_Domain",
    ...     experiment="Emulator_Hist_Future",
    ...     variable="pr",
    ...     prediction="Extrapolation_perfect",
    ...     use_orography=True,
    ... )
    """

    def __init__(
        self,
        config: SubmissionWorkflowConfig,
        catalog: SubmissionCatalog,
        metrics_registry_path: str | Path,
        ml_benchmark_path: str | Path | None = None,
    ):
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

    def evaluate(
        self,
        submission_name: str,
        domain: str,
        experiment: str,
        variable: str,
        prediction: str,
        metrics: List[str] | None = None,
        use_orography: bool | None = None,
        season: str | None = None,
        run_id: str | None = None,
        **open_kwargs: Any,
    ) -> dict[str, xr.Dataset]:
        """
        Compute metrics for a prediction and save results.
        
        Parameters
        ----------
        submission_name : str
            Name of the submission (catalog entry).
        domain : str
            Domain name (e.g., "ALPS_Domain").
        experiment : str
            Experiment name (e.g., "Emulator_Hist_Future").
        variable : str
            Variable name (e.g., "pr", "tasmax").
        prediction : str
            Prediction name (e.g., "Extrapolation_perfect").
        metrics : list of str, optional
            Metric names to compute. If None, uses defaults from registry.
        use_orography : bool, optional
            Toggle orography variant of experiment.
        season : str, optional
            Season filter ("winter", "summer", "spring", "autumn").
        run_id : str, optional
            Identifier for this run. If None, uses timestamp.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        dict[str, xr.Dataset]
            Dictionary mapping metric names to computed result datasets.
        """
        if run_id is None:
            run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        if metrics is None:
            metrics = self.metrics_registry.get("defaults", {}).get(variable, [])

        # Load and align datasets
        pred_ds, gt_ds = self._load_and_align_datasets(
            submission_name=submission_name,
            domain=domain,
            experiment=experiment,
            variable=variable,
            prediction=prediction,
            use_orography=use_orography,
            **open_kwargs,
        )

        # Compute each metric
        results = {}
        for metric_name in metrics:
            if metric_name not in self.metrics_registry.get("metrics", {}):
                raise ValueError(f"Unknown metric: {metric_name}")

            metric_config = self.metrics_registry["metrics"][metric_name]
            metric_ds = self._compute_metric(
                pred_ds=pred_ds,
                gt_ds=gt_ds,
                variable=variable,
                metric_name=metric_name,
                metric_config=metric_config,
                season=season,
            )

            # Save metric to NetCDF
            metric_path = self._metric_path(
                submission_name=submission_name,
                domain=domain,
                experiment=experiment,
                variable=variable,
                prediction=prediction,
                metric_name=metric_name,
                run_id=run_id,
                use_orography=use_orography,
            )
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_ds.to_netcdf(metric_path)

            results[metric_name] = metric_ds

        return results

    def available_metric_files(
        self,
        submission_name: str,
        domain: str,
        experiment: str,
        variable: str,
        prediction: str,
        use_orography: bool | None = None,
    ) -> List[str]:
        """
        List saved metric files for a prediction.
        
        Returns
        -------
        list of str
            Metric filenames (without path).
        """
        metric_dir = self._metric_dir(
            submission_name=submission_name,
            domain=domain,
            experiment=experiment,
            variable=variable,
            prediction=prediction,
            use_orography=use_orography,
        )

        if not metric_dir.exists():
            return []

        return sorted([f.name for f in metric_dir.glob("metric_*.nc")])

    def load_metric(
        self,
        submission_name: str,
        domain: str,
        experiment: str,
        variable: str,
        prediction: str,
        metric_name: str,
        run_id: str | None = None,
        use_orography: bool | None = None,
        **open_kwargs: Any,
    ) -> xr.Dataset:
        """
        Load a previously saved metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to load.
        run_id : str, optional
            Specific run_id to load. If None, loads the most recent.
        
        Returns
        -------
        xr.Dataset
            Loaded metric dataset.
        """
        if run_id is None:
            # Find most recent metric file
            metric_files = self.available_metric_files(
                submission_name=submission_name,
                domain=domain,
                experiment=experiment,
                variable=variable,
                prediction=prediction,
                use_orography=use_orography,
            )
            matching = [f for f in metric_files if f.startswith(f"metric_{metric_name}_")]
            if not matching:
                raise FileNotFoundError(
                    f"No metric file found for '{metric_name}' in prediction '{prediction}'."
                )
            # Extract run_id from filename: metric_<name>_<runid>.nc
            run_id = matching[-1].replace(f"metric_{metric_name}_", "").replace(".nc", "")

        metric_path = self._metric_path(
            submission_name=submission_name,
            domain=domain,
            experiment=experiment,
            variable=variable,
            prediction=prediction,
            metric_name=metric_name,
            run_id=run_id,
            use_orography=use_orography,
        )

        if not metric_path.exists():
            raise FileNotFoundError(f"Metric file not found: {metric_path}")

        return xr.open_dataset(metric_path, **open_kwargs)

    # -------------------------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------------------------

    def _load_and_align_datasets(
        self,
        submission_name: str,
        domain: str,
        experiment: str,
        variable: str,
        prediction: str,
        use_orography: bool | None = None,
        **open_kwargs: Any,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Load prediction and ground truth datasets, aligned on coordinates."""
        # Load prediction
        pred_ds = self.submission_loader.load_dataset(
            submission_name=submission_name,
            domain=domain,
            experiment=experiment,
            variable=variable,
            prediction=prediction,
            use_orography=use_orography,
            **open_kwargs,
        )

        # Resolve ground truth experiment name
        gt_experiment = _normalize_experiment_for_groundtruth(experiment)

        # Load ground truth
        gt_ds = self.gt_loader.load_test_target(
            domain=domain,
            training_experiment=gt_experiment,
            evaluation_experiment=prediction,
            variable=variable,
            merge_mid_end=True,
            **open_kwargs,
        )

        # Align on spatial coordinates first, then time
        pred_ds, gt_ds = xr.align(pred_ds, gt_ds, join="inner", exclude=["time"])
        pred_ds, gt_ds = xr.align(pred_ds, gt_ds, join="inner")

        return pred_ds, gt_ds

    def _compute_metric(
        self,
        pred_ds: xr.Dataset,
        gt_ds: xr.Dataset,
        variable: str,
        metric_name: str,
        metric_config: dict,
        season: str | None = None,
    ) -> xr.Dataset:
        """Compute a single metric using the appropriate diagnostic function."""
        metric_type = metric_config["type"]

        # Import ml-benchmark diagnostics module
        try:
            import evaluation.diagnostics as diagnostics
        except ImportError:
            raise ImportError(
                "Could not import ml-benchmark evaluation modules. "
                "Ensure evaluation/ directory is in Python path or set ml_benchmark_path "
                "when initializing Evaluator."
            )

        if metric_type == "rmse":
            dim = metric_config.get("dim")
            return diagnostics.rmse(
                x0=gt_ds,
                x1=pred_ds,
                var=variable,
                season=season,
                dim=dim,
            )

        elif metric_type == "psd":
            psd_x0, psd_x1 = diagnostics.psd(
                x0=gt_ds,
                x1=pred_ds,
                var=variable,
                season=season,
            )
            return xr.Dataset({"PSD_x0": psd_x0, "PSD_x1": psd_x1})

        elif metric_type == "bias_index":
            index_fn = _import_function(metric_config["index_fn"])
            index_kwargs = metric_config.get("kwargs", {}).copy()
            index_kwargs["var"] = variable
            return diagnostics.bias_index(
                x0=gt_ds,
                x1=pred_ds,
                index_fn=index_fn,
                season=season,
                **index_kwargs,
            )

        elif metric_type == "ratio_index":
            index_fn = _import_function(metric_config["index_fn"])
            index_kwargs = metric_config.get("kwargs", {}).copy()
            index_kwargs["var"] = variable
            return diagnostics.ratio_index(
                x0=gt_ds,
                x1=pred_ds,
                index_fn=index_fn,
                season=season,
                **index_kwargs,
            )

        elif metric_type == "corr_bias":
            kwargs = metric_config.get("kwargs", {})
            var_x = kwargs.get("var_x", variable)
            var_y = kwargs.get("var_y")
            if var_y is None:
                raise ValueError(
                    f"corr_bias requires var_y in kwargs for metric '{metric_name}'"
                )
            return diagnostics.bias_multivariable_correlation(
                x0=gt_ds,
                x1=pred_ds,
                var_x=var_x,
                var_y=var_y,
                season=season,
            )

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def _metric_dir(
        self,
        submission_name: str,
        domain: str,
        experiment: str,
        variable: str,
        prediction: str,
        use_orography: bool | None = None,
    ) -> Path:
        """Build directory path for storing metrics."""
        experiment_name = self.submission_loader._resolve_experiment_name(
            experiment, use_orography
        )
        return (
            self.config.diagnostic_path
            / submission_name
            / domain
            / experiment_name
            / variable
            / prediction
        )

    def _metric_path(
        self,
        submission_name: str,
        domain: str,
        experiment: str,
        variable: str,
        prediction: str,
        metric_name: str,
        run_id: str,
        use_orography: bool | None = None,
    ) -> Path:
        """Build file path for a metric NetCDF."""
        metric_dir = self._metric_dir(
            submission_name=submission_name,
            domain=domain,
            experiment=experiment,
            variable=variable,
            prediction=prediction,
            use_orography=use_orography,
        )
        return metric_dir / f"metric_{metric_name}_{run_id}.nc"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


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


def _normalize_experiment_for_groundtruth(experiment: str) -> str:
    """
    Convert experiment name to ground truth naming convention.
    
    Examples:
        "Emulator_Hist_Future_NO_OROG" -> "Emulator_hist_future"
        "ESD_Pseudo-Reality_OROG" -> "ESD_pseudo_reality"
    """
    # Strip orography suffix
    base = experiment
    for suffix in ("_OROG", "_NO_OROG"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    # Replace dashes with underscores
    base = base.replace("-", "_")

    # Convert to ground truth format: first part stays, rest lowercase
    parts = base.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1].lower()}"
    return base.lower()
