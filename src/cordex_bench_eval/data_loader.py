"""
Data loaders for submissions and ground truth.

This module provides two main loaders:
- SubmissionDataLoader: Load prediction NetCDF files from staged submissions
- GroundTruthDataLoader: Load ground truth data for training and testing experiments
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import xarray as xr
from .catalog import SubmissionCatalog
from .mappings import (
    EVAL_SETTINGS as _EVAL_SETTINGS,
    GCM_EVAL as _GCM_EVAL,
    GCM_TRAIN as _GCM_TRAIN,
    PERIOD_DATES as _PERIOD_DATES,
    TRAIN_PERIODS as _TRAIN_PERIODS,
)

class SubmissionDataLoader:
    """
    Load NetCDF files from staged submissions.
    
    Submissions are organized as:
        <submission>/<domain>/<training_exp>/<period>/<condition>/<filename>.nc
    
    Where filename is: Predictions_pr_tasmax_<GCM>_<dates>.nc
    
    Note: Orography information is now stored in the catalog metadata rather
    than in the directory structure.
    
    Parameters
    ----------
    catalog : SubmissionCatalog
        Catalog containing registered submissions.
    """

    def __init__(self, catalog: SubmissionCatalog):
        self.catalog = catalog

    def available_domains(self, submission_name: str) -> List[str]:
        """List all domain folders for a submission."""
        return _list_dir_names(self._entry_root(submission_name))

    def available_experiments(self, submission_name: str, domain: str) -> List[str]:
        """List experiment folders within a domain."""
        domain_dir = self._entry_root(submission_name) / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Domain '{domain}' does not exist for '{submission_name}'.")
        return _list_dir_names(domain_dir)

    def available_periods(self, submission_name: str, domain: str, experiment: str) -> List[str]:
        """List period folders within an experiment."""
        experiment_dir = self._entry_root(submission_name) / domain / experiment
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment}' under '{domain}' does not exist "
                                    f"for '{submission_name}'.")
        return _list_dir_names(experiment_dir)

    def available_conditions(self, submission_name: str, domain: str, experiment: str,
                             period: str) -> List[str]:
        """List condition folders (perfect/imperfect) within a period."""
        period_dir = self._entry_root(submission_name) / domain / experiment / period
        if not period_dir.exists():
            raise FileNotFoundError(f"Period '{period}' under '{experiment}' does not exist.")
        return _list_dir_names(period_dir)

    def available_predictions(self, submission_name: str, domain: str, experiment: str,
                              period: str, condition: str) -> List[str]:
        """List available prediction files for a condition."""
        target_dir = self._target_dir(submission_name, domain, experiment, period, condition)
        return [path.name for path in sorted(target_dir.glob("*.nc"))]

    def dataset_path(self, submission_name: str, domain: str, training_experiment: str,
                     evaluation_experiment: str) -> Path:
        """
        Get the path to a prediction NetCDF file based on evaluation experiment.
        
        This method maps the evaluation experiment to the corresponding period,
        condition, and GCM to locate the correct file.
        
        Parameters
        ----------
        submission_name : str
            Name of the submission in the catalog.
        domain : str
            Domain name (e.g., "ALPS_Domain").
        training_experiment : str
            Training experiment (e.g., "Emulator_hist_future", "ESD_pseudo_reality").
        evaluation_experiment : str
            Evaluation experiment (e.g., "Perfect_cross_validation").
        
        Returns
        -------
        Path
            Path to the NetCDF file.
        """
        # Normalize experiment name
        experiment_name = _normalize_training_experiment(training_experiment)
        
        # Get period, condition, and GCM info from evaluation experiment
        period, condition, same_gcm_as_train = self._resolve_evaluation_experiment(training_experiment,
                                                                                   evaluation_experiment)
        
        # Get GCM name for this domain
        domain_key = _normalize_domain_key(domain)
        gcm_name = _GCM_TRAIN[domain_key] if same_gcm_as_train else _GCM_EVAL[domain_key]
        
        # Handle combined mid+end century periods - need to return list of paths
        if period == "mid_end_century":
            # For mid_end_century, we'll merge files - return first one and handle in load_dataset
            target_dir = self._target_dir(submission_name, domain, experiment_name,
                                          "mid_century", condition)
            date_tag = _PERIOD_DATES["mid_century"]
            filename = f"Predictions_pr_tasmax_{gcm_name}_{date_tag}.nc"
            return target_dir / filename
        
        target_dir = self._target_dir(submission_name, domain, experiment_name, period, condition)
        date_tag = _PERIOD_DATES[period]
        filename = f"Predictions_pr_tasmax_{gcm_name}_{date_tag}.nc"
        
        path = target_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        return path

    def load_dataset(self, submission_name: str, domain: str, training_experiment: str,
                     evaluation_experiment: str, variable: str | None = None,
                     merge_mid_end: bool = True, **open_kwargs: Any) -> xr.Dataset:
        """
        Load a prediction dataset for a specific evaluation experiment.
        
        Parameters
        ----------
        submission_name : str
            Name of the submission in the catalog.
        domain : str
            Domain name (e.g., "ALPS_Domain").
        training_experiment : str
            Training experiment (e.g., "Emulator_hist_future").
        evaluation_experiment : str
            Evaluation experiment (e.g., "Perfect_cross_validation").
        variable : str, optional
            Variable to select ("pr" or "tasmax"). If None, returns both.
        merge_mid_end : bool, default=True
            If True and period spans mid+end century, merge both files.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        xr.Dataset
            Loaded xarray dataset.
        """
        # Normalize experiment name
        experiment_name = _normalize_training_experiment(training_experiment)
        
        # Get period, condition, and GCM info from evaluation experiment
        period, condition, same_gcm_as_train = self._resolve_evaluation_experiment(training_experiment, evaluation_experiment)
        
        # Get GCM name for this domain
        domain_key = _normalize_domain_key(domain)
        gcm_name = _GCM_TRAIN[domain_key] if same_gcm_as_train else _GCM_EVAL[domain_key]
        
        # Handle combined mid+end century periods
        if period == "mid_end_century":
            mid_path = self._get_prediction_path(submission_name, domain, experiment_name,
                                                  "mid_century", condition, gcm_name)
            mid_ds = xr.open_dataset(mid_path, **open_kwargs).load()
            
            if merge_mid_end:
                end_path = self._get_prediction_path(submission_name, domain, experiment_name,
                                                      "end_century", condition, gcm_name)
                end_ds = xr.open_dataset(end_path, **open_kwargs).load()
                ds = xr.merge([mid_ds, end_ds], join="outer", compat="no_conflicts")
                mid_ds.close()
                end_ds.close()
            else:
                ds = mid_ds
        else:
            path = self._get_prediction_path(submission_name, domain, experiment_name,
                                              period, condition, gcm_name)
            ds = xr.open_dataset(path, **open_kwargs)
        
        return self._maybe_subset_variable(ds, variable)

    def _entry_root(self, submission_name: str) -> Path:
        """Get the root directory for a submission."""
        entry = self.catalog.get(submission_name)
        if entry is None:
            raise KeyError(f"No submission named '{submission_name}' in catalog.")
        root = Path(entry.location)
        if not root.exists():
            raise FileNotFoundError(f"Submission folder '{root}' does not exist.")
        return root

    def _target_dir(self, submission_name: str, domain: str, experiment: str,
                    period: str, condition: str) -> Path:
        """Get the directory containing prediction files."""
        target_dir = self._entry_root(submission_name) / domain / experiment / period / condition
        if not target_dir.exists():
            raise FileNotFoundError(f"Folder {target_dir} does not exist.")
        return target_dir

    def _get_prediction_path(self, submission_name: str, domain: str, experiment: str,
                              period: str, condition: str, gcm_name: str) -> Path:
        """Build the full path to a prediction file."""
        target_dir = self._target_dir(submission_name, domain, experiment, period, condition)
        date_tag = _PERIOD_DATES[period]
        filename = f"Predictions_pr_tasmax_{gcm_name}_{date_tag}.nc"
        path = target_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        return path

    def _resolve_evaluation_experiment(self, training_experiment: str,
                                       evaluation_experiment: str) -> tuple[str, str, bool]:
        """
        Look up evaluation experiment settings from mapping tables.
        
        Returns
        -------
        tuple[str, str, bool]
            (period, condition, same_gcm_as_train)
        """
        # Normalize training experiment name
        train_exp = _normalize_training_experiment(training_experiment)
        
        if train_exp not in _EVAL_SETTINGS:
            raise KeyError(f"Unknown training experiment '{training_experiment}'.")
        if evaluation_experiment not in _EVAL_SETTINGS[train_exp]:
            available = ", ".join(sorted(_EVAL_SETTINGS[train_exp]))
            raise KeyError(f"Evaluation experiment '{evaluation_experiment}' not supported for "
                           f"'{training_experiment}'. Available: {available}")
        return _EVAL_SETTINGS[train_exp][evaluation_experiment]

    @staticmethod
    def _maybe_subset_variable(ds: xr.Dataset, variable: str | None) -> xr.Dataset:
        """Optionally subset dataset to a single variable."""
        if variable is None:
            return ds
        
        # Ensure common spatial coordinate names are treated as coordinates
        # so they are preserved during subsetting
        spatial_vars = [v for v in ["lat", "lon", "latitude", "longitude"] if v in ds.data_vars]
        if spatial_vars:
            ds = ds.set_coords(spatial_vars)

        if variable not in ds.data_vars:
            raise KeyError(f"Variable '{variable}' not found in dataset.")
        return ds[[variable]]


class GroundTruthDataLoader:
    """
    Load ground truth NetCDF files for training and testing.
    
    Ground truth is organized as:
        <root>/<DOMAIN>_domain/train/<experiment>/target/pr_tasmax_<GCM>_<period>.nc
        <root>/<DOMAIN>_domain/test/<period>/target/pr_tasmax_<GCM>_<dates>.nc
    
    Parameters
    ----------
    root : str | Path
        Root directory containing ground truth data.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Ground-truth root '{self.root}' does not exist.")

    def available_domains(self) -> List[str]:
        """List available domain directories."""
        return _list_dir_names(self.root)

    def available_train_experiments(self, domain: str) -> List[str]:
        """List available training experiments for a domain."""
        return _list_dir_names(self._train_dir(domain))

    def available_test_periods(self, domain: str) -> List[str]:
        """List available test periods for a domain."""
        return _list_dir_names(self._test_dir(domain))

    def to_mm_day(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert the dataset to mm/day."""
        if "pr" in ds.data_vars:
            ds["pr"] = ds["pr"] * 86400
        return ds

    def load_train_target(self, domain: str, training_experiment: str,
                          variable: str | None = None, **open_kwargs: Any) -> xr.Dataset:
        """
        Load training target data.
        
        Parameters
        ----------
        domain : str
            Domain name (e.g., "ALPS", "NZ", "SA").
        training_experiment : str
            Training experiment name (e.g., "Emulator_hist_future").
        variable : str, optional
            Variable to select ("pr" or "tasmax"). If None, returns both.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        xr.Dataset
            Ground truth training dataset.
        """
        path = self._train_target_path(domain, training_experiment)
        ds = xr.open_dataset(path, **open_kwargs)

        return self._maybe_subset_variable(ds, variable)

    def load_test_target(self, domain: str, training_experiment: str, evaluation_experiment: str,
                         variable: str | None = None, merge_mid_end: bool = True,
                         **open_kwargs: Any) -> xr.Dataset:
        """
        Load test target data for evaluation.
        
        Parameters
        ----------
        domain : str
            Domain name (e.g., "ALPS", "NZ", "SA").
        training_experiment : str
            Training experiment (determines which GCM to use).
        evaluation_experiment : str
            Evaluation experiment (e.g., "Perfect_cross_validation").
        variable : str, optional
            Variable to select. If None, returns both.
        merge_mid_end : bool, default=True
            If True and period spans mid+end century, merge both files.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        xr.Dataset
            Ground truth test dataset.
        """
        period, _, same_gcm_as_train = self._resolve_experiment(training_experiment, evaluation_experiment)
        domain_key = _normalize_domain_key(domain)
        gcm_name = _GCM_TRAIN[domain_key] if same_gcm_as_train else _GCM_EVAL[domain_key]

        # Handle combined mid+end century period
        if period == "mid_end_century":
            mid_path = self._test_target_path(domain, "mid_century", gcm_name)
            mid_ds = xr.open_dataset(mid_path, **open_kwargs).load()
            if merge_mid_end:
                end_path = self._test_target_path(domain, "end_century", gcm_name)
                end_ds = xr.open_dataset(end_path, **open_kwargs).load()
                ds = xr.merge([mid_ds, end_ds], join="outer", compat="no_conflicts")
                mid_ds.close()
                end_ds.close()
            else:
                ds = mid_ds
        else:
            ds = xr.open_dataset(self._test_target_path(domain, period, gcm_name), **open_kwargs)

        # Test data for SA domain is in kg m-2 s-1, we need to convert to mm/day
        if domain == "SA_Domain":
            ds = self.to_mm_day(ds)

        return self._maybe_subset_variable(ds, variable)

    def _train_dir(self, domain: str) -> Path:
        """Get the train directory for a domain."""
        train_dir = self._domain_dir(domain) / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Train folder not found: {train_dir}")
        return train_dir

    def _test_dir(self, domain: str) -> Path:
        """Get the test directory for a domain."""
        test_dir = self._domain_dir(domain) / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test folder not found: {test_dir}")
        return test_dir

    def _domain_dir(self, domain: str) -> Path:
        """Get the domain directory."""
        normalized = _normalize_domain_key(domain)
        domain_dir = self.root / f"{normalized}_domain"
        if not domain_dir.exists():
            raise FileNotFoundError(f"Domain '{domain}' not found in {self.root}.")
        return domain_dir

    def _train_target_path(self, domain: str, training_experiment: str) -> Path:
        """Build path to training target file."""
        domain_key = _normalize_domain_key(domain)
        train_exp = _normalize_training_experiment(training_experiment)
        if train_exp not in _TRAIN_PERIODS:
            raise KeyError(f"Unknown training experiment '{training_experiment}'.")
        gcm_name = _GCM_TRAIN[domain_key]
        period = _TRAIN_PERIODS[train_exp]
        target_dir = self._train_dir(domain) / train_exp / "target"
        if not target_dir.exists():
            raise FileNotFoundError(f"Target folder not found: {target_dir}")
        path = target_dir / f"pr_tasmax_{gcm_name}_{period}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {path}")
        return path

    def _test_target_path(self, domain: str, period: str, gcm_name: str) -> Path:
        """Build path to test target file."""
        if period not in _PERIOD_DATES:
            raise KeyError(f"Unknown period '{period}'.")
        date_tag = _PERIOD_DATES[period]
        target_dir = self._test_dir(domain) / period / "target"
        if not target_dir.exists():
            raise FileNotFoundError(f"Target folder not found: {target_dir}")
        path = target_dir / f"pr_tasmax_{gcm_name}_{date_tag}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {path}")
        return path

    def _resolve_experiment(self, training_experiment: str,
                            evaluation_experiment: str) -> tuple[str, str, bool]:
        """
        Look up experiment settings from mapping tables.
        
        Returns
        -------
        tuple[str, str, bool]
            (period, condition, same_gcm_as_train)
        """
        train_exp = _normalize_training_experiment(training_experiment)
        if train_exp not in _EVAL_SETTINGS:
            raise KeyError(f"Unknown training experiment '{training_experiment}'.")
        if evaluation_experiment not in _EVAL_SETTINGS[train_exp]:
            available = ", ".join(sorted(_EVAL_SETTINGS[train_exp]))
            raise KeyError(f"Experiment '{evaluation_experiment}' not supported for "
                           f"'{training_experiment}'. Available: {available}")
        return _EVAL_SETTINGS[train_exp][evaluation_experiment]

    @staticmethod
    def _maybe_subset_variable(ds: xr.Dataset, variable: str | None) -> xr.Dataset:
        """Optionally subset dataset to a single variable."""
        if variable is None:
            return ds
        
        # Ensure common spatial coordinate names are treated as coordinates
        # so they are preserved during subsetting
        spatial_vars = [v for v in ["lat", "lon", "latitude", "longitude"] if v in ds.data_vars]
        if spatial_vars:
            ds = ds.set_coords(spatial_vars)

        if variable not in ds.data_vars:
            raise KeyError(f"Variable '{variable}' not found in dataset.")
        return ds[[variable]]


def _list_dir_names(path: Path) -> List[str]:
    """List subdirectory names (not files) in a directory."""
    return [child.name for child in sorted(path.iterdir()) if child.is_dir()]


def _normalize_domain_key(domain: str) -> str:
    """
    Normalize domain strings to uppercase keys.
    
    Handles inputs like "ALPS", "ALPS_domain", "ALPS_Domain" -> "ALPS"
    """
    value = Path(domain).name
    for suffix in ("_domain", "_Domain"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.upper()


def _normalize_training_experiment(experiment: str) -> str:
    """
    Normalize training experiment name to canonical form.
    
    Handles inputs like "Emulator_Hist_Future", "Emulator_hist_future" -> "Emulator_hist_future"
    Also strips orography suffix if present.
    """
    # Strip orography suffix first
    base = experiment
    for suffix in ("_OROG", "_NO_OROG"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    
    # Normalize case: first part stays, rest lowercase
    parts = base.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1].lower()}"
    return base.lower()

