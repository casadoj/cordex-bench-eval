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

class SubmissionDataLoader:
    """
    Load NetCDF files from staged submissions.
    
    Submissions are organized as:
        <submission>/<domain>/<experiment>/<variable>/predictions_<prediction>.nc
    
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

    def available_variables(self, submission_name: str, domain: str, experiment: str) -> List[str]:
        """List variable folders within an experiment."""
        experiment_dir = self._entry_root(submission_name) / domain / experiment
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment}' under '{domain}' does not exist "
                                    f"for '{submission_name}'.")
        return _list_dir_names(experiment_dir)

    def available_predictions(self, submission_name: str, domain: str, experiment: str,
                              variable: str) -> List[str]:
        """List available predictions (derived from filenames) for a variable."""
        target_dir = self._target_dir(submission_name, domain, experiment, variable)
        return [_normalize_prediction_name(path.name)
                for path in sorted(target_dir.glob("*.nc"))]

    def dataset_path(self, submission_name: str, domain: str, experiment: str, variable: str,
                     prediction: str | None = None, use_orography: bool | None = None) -> Path:
        """
        Get the path to a NetCDF file.
        
        Parameters
        ----------
        submission_name : str
            Name of the submission in the catalog.
        domain : str
            Domain name (e.g., "ALPS_Domain").
        experiment : str
            Base experiment name (e.g., "Emulator_Hist_Future").
        variable : str
            Variable name (e.g., "pr", "tasmax").
        prediction : str, optional
            Prediction name. Required if multiple .nc files exist.
        use_orography : bool, optional
            If True, appends "_OROG" to experiment.
            If False, appends "_NO_OROG".
            If None, uses experiment name as-is.
        
        Returns
        -------
        Path
            Path to the NetCDF file.
        """
        experiment_name = self._resolve_experiment_name(experiment, use_orography)
        target_dir = self._target_dir(submission_name, domain, experiment_name, variable)
        return self._select_prediction_file(target_dir, prediction)

    def load_dataset(self, submission_name: str, domain: str, experiment: str, variable: str,
                     prediction: str | None = None, use_orography: bool | None = None,
                     **open_kwargs: Any) -> xr.Dataset:
        """
        Load a prediction dataset.
        
        Parameters
        ----------
        submission_name : str
            Name of the submission in the catalog.
        domain : str
            Domain name (e.g., "ALPS_Domain").
        experiment : str
            Base experiment name.
        variable : str
            Variable name.
        prediction : str, optional
            Prediction name. Required if multiple .nc files exist.
        use_orography : bool, optional
            Toggle orography variant of experiment.
        **open_kwargs
            Additional arguments passed to xarray.open_dataset().
        
        Returns
        -------
        xr.Dataset
            Loaded xarray dataset.
        """
        path = self.dataset_path(submission_name, domain, experiment, variable,
                                 prediction, use_orography)
        return xr.open_dataset(path, **open_kwargs)

    def _entry_root(self, submission_name: str) -> Path:
        """Get the root directory for a submission."""
        entry = self.catalog.get(submission_name)
        if entry is None:
            raise KeyError(f"No submission named '{submission_name}' in catalog.")
        root = Path(entry.location)
        if not root.exists():
            raise FileNotFoundError(f"Submission folder '{root}' does not exist.")
        return root

    def _target_dir(self, submission_name: str, domain: str, experiment: str, variable: str) -> Path:
        """Get the directory containing prediction files."""
        target_dir = self._entry_root(submission_name) / domain / experiment / variable
        if not target_dir.exists():
            raise FileNotFoundError(f"Folder {target_dir} does not exist.")
        return target_dir

    def _select_prediction_file(self, target_dir: Path, prediction: str | None) -> Path:
        """Select the correct NetCDF file from a directory."""
        nc_files = sorted(target_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {target_dir}.")

        # If only one file and no prediction specified, return it
        if prediction is None:
            if len(nc_files) == 1:
                return nc_files[0]
            available = ", ".join(_normalize_prediction_name(p.name) for p in nc_files)
            raise ValueError(f"Multiple NetCDF files in {target_dir}. "
                             f"Specify prediction (available: {available}).")

        # Find matching file
        normalized = _normalize_prediction_name(prediction)
        for path in nc_files:
            if _normalize_prediction_name(path.name) == normalized:
                return path

        available = ", ".join(_normalize_prediction_name(p.name) for p in nc_files)
        raise FileNotFoundError(f"No NetCDF file named '{prediction}' in {target_dir}. "
                                f"Available predictions: {available}.")

    @staticmethod
    def _resolve_experiment_name(experiment: str, use_orography: bool | None) -> str:
        """
        Build the full experiment name with orography suffix.
        
        If use_orography is None, returns the experiment unchanged.
        Otherwise, strips any existing suffix and adds the appropriate one.
        """
        if use_orography is None:
            return experiment

        base = experiment
        for suffix in ("_OROG", "_NO_OROG"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        return f"{base}_{'OROG' if use_orography else 'NO_OROG'}"

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
            Evaluation experiment (e.g., "Extrapolation_perfect").
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
            end_path = self._test_target_path(domain, "end_century", gcm_name)
            mid_ds = xr.open_dataset(mid_path, **open_kwargs).load()
            if merge_mid_end:
                end_ds = xr.open_dataset(end_path, **open_kwargs).load()
                ds = xr.merge([mid_ds, end_ds], join="outer", compat="no_conflicts")
                mid_ds.close()
                end_ds.close()
            else:
                ds = mid_ds
        else:
            ds = xr.open_dataset(self._test_target_path(domain, period, gcm_name), **open_kwargs)

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
        if training_experiment not in _TRAIN_PERIODS:
            raise KeyError(f"Unknown training experiment '{training_experiment}'.")
        gcm_name = _GCM_TRAIN[domain_key]
        period = _TRAIN_PERIODS[training_experiment]
        target_dir = self._train_dir(domain) / training_experiment / "target"
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

    def _resolve_experiment(
        self, training_experiment: str, evaluation_experiment: str
    ) -> tuple[str, str, bool]:
        """
        Look up experiment settings from mapping tables.
        
        Returns
        -------
        tuple[str, str, bool]
            (period, experiment_type, same_gcm_as_train)
        """
        if training_experiment not in _EVAL_SETTINGS:
            raise KeyError(f"Unknown training experiment '{training_experiment}'.")
        if evaluation_experiment not in _EVAL_SETTINGS[training_experiment]:
            available = ", ".join(sorted(_EVAL_SETTINGS[training_experiment]))
            raise KeyError(f"Experiment '{evaluation_experiment}' not supported for "
                           f"'{training_experiment}'. Available: {available}")
        return _EVAL_SETTINGS[training_experiment][evaluation_experiment]

    @staticmethod
    def _maybe_subset_variable(ds: xr.Dataset, variable: str | None) -> xr.Dataset:
        """Optionally subset dataset to a single variable."""
        if variable is None:
            return ds
        if variable not in ds.data_vars:
            raise KeyError(f"Variable '{variable}' not found in dataset.")
        return ds[[variable]]

def _list_dir_names(path: Path) -> List[str]:
    """List subdirectory names (not files) in a directory."""
    return [child.name for child in sorted(path.iterdir()) if child.is_dir()]

def _normalize_prediction_name(value: str) -> str:
    """
    Normalize prediction names for comparison.
    
    Strips 'predictions_' prefix and '.nc' suffix.
    Example: 'predictions_Extrapolation_perfect.nc' -> 'Extrapolation_perfect'
    """
    name = Path(value).name
    if name.endswith(".nc"):
        name = name[:-3]
    if name.startswith("predictions_"):
        name = name[len("predictions_"):]
    return name

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


##################
# MAPPING TABLES #
##################

# Training periods by experiment type
_TRAIN_PERIODS = {"Emulator_hist_future": "1961-1980_2080-2099",
                  "ESD_pseudo_reality": "1961-1980"}

# Date tags for test periods
_PERIOD_DATES = {"historical": "1981-2000",
                 "mid_century": "2041-2060",
                 "end_century": "2080-2099",
                 "mid_end_century": ["2041-2060", "2080-2099"]}

# GCM models used for training by domain
_GCM_TRAIN = {"NZ": "ACCESS-CM2",
              "ALPS": "CNRM-CM5",
              "SA": "ACCESS-CM2"}

# GCM models used for evaluation (hard-transferability)
_GCM_EVAL = {"NZ": "EC-Earth3",
             "ALPS": "MPI-ESM-LR",
            "SA": "NorESM2-MM"}

# Evaluation settings: training_experiment -> evaluation_experiment -> (period, mode, same_gcm_as_train)
_EVAL_SETTINGS = {
    "ESD_pseudo_reality": {
        "PP_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Extrapolation_perfect": ("mid_end_century", "perfect", True),
        "Extrapolation_imperfect": ("mid_end_century", "imperfect", True),
    },
    "Emulator_hist_future": {
        "PP_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Extrapolation_perfect": ("mid_end_century", "perfect", True),
        "Extrapolation_imperfect": ("mid_end_century", "imperfect", True),
        "Extrapolation_perfect_hard": ("mid_end_century", "perfect", False),
        "Extrapolation_imperfect_hard": ("mid_end_century", "imperfect", False),
    },
}
