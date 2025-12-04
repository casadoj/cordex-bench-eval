from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import xarray as xr
import cf_xarray as cfxr # TODO: Should I keep this? 

from .catalog import SubmissionCatalog


class SubmissionDataLoader:
    """Locate and load NetCDF files for staged submissions."""

    def __init__(self, catalog: SubmissionCatalog):
        self.catalog = catalog

    def available_domains(self, submission_name: str) -> List[str]:
        """Return all domain folders for a submission."""

        root = self._entry_root(submission_name)
        return _list_dir_names(root)

    def available_experiments(self, submission_name: str, domain: str) -> List[str]:
        """Return experiment folders within a domain."""

        domain_dir = self._entry_root(submission_name) / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Domain '{domain}' does not exist for '{submission_name}'.")
        return _list_dir_names(domain_dir)

    def available_variables(self,
                            submission_name: str,
                            domain: str,
                            experiment: str) -> List[str]:
        """Return variable folders within a domain/experiment."""

        experiment_dir = self._entry_root(submission_name) / domain / experiment
        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment}' under '{domain}' does not exist for '{submission_name}'."
            )
        return _list_dir_names(experiment_dir)

    def available_predictions(self,
                              submission_name: str,
                              domain: str,
                              experiment: str,
                              variable: str) -> List[str]:
        """Return prediction names (derived from filenames) for a variable folder."""

        target_dir = self._target_dir(submission_name, domain, experiment, variable)
        return [_normalize_prediction_name(path.name) for path in sorted(target_dir.glob("*.nc"))]

    def dataset_path(self,
                     submission_name: str,
                     domain: str,
                     experiment: str,
                     variable: str,
                     prediction: str | None = None) -> Path:
        """
        Resolve the NetCDF file path for the requested subset.

        Files sit under submission/domain/experiment/variable and follow the naming
        convention ``predictions_<prediction>.nc``. If multiple NetCDF files exist
        and ``prediction`` is not provided, the available names are surfaced in the
        error message to guide the caller.
        """

        target_dir = self._target_dir(submission_name, domain, experiment, variable)
        return self._select_prediction_file(target_dir, prediction)

    def load_dataset(self,
                     submission_name: str,
                     domain: str,
                     experiment: str,
                     variable: str,
                     prediction: str | None = None,
                     **open_kwargs: Any) -> xr.Dataset:
        """
        Load the requested NetCDF file with xarray.open_dataset().

        Additional keyword arguments are forwarded to xarray.
        """

        path = self.dataset_path(submission_name, domain, experiment, variable, prediction)
        return xr.open_dataset(path, **open_kwargs)

    def _entry_root(self, submission_name: str) -> Path:
        entry = self.catalog.get(submission_name)
        if entry is None:
            raise KeyError(f"No submission named '{submission_name}' in catalog.")
        root = Path(entry.location)
        if not root.exists():
            raise FileNotFoundError(f"Submission folder '{root}' does not exist on disk.")
        return root

    def _target_dir(self,
                    submission_name: str,
                    domain: str,
                    experiment: str,
                    variable: str) -> Path:
        target_dir = self._entry_root(submission_name) / domain / experiment / variable
        if not target_dir.exists():
            raise FileNotFoundError(f"Folder {target_dir} does not exist.")
        return target_dir

    def _select_prediction_file(self, target_dir: Path, prediction: str | None) -> Path:
        nc_files = sorted(target_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {target_dir}.")

        if prediction is None:
            if len(nc_files) == 1:
                return nc_files[0]
            available = ", ".join(_normalize_prediction_name(path.name) for path in nc_files)
            raise ValueError(
                f"Multiple NetCDF files in {target_dir}. Specify prediction "
                f"(available: {available})."
            )

        normalized = _normalize_prediction_name(prediction)
        for path in nc_files:
            if _normalize_prediction_name(path.name) == normalized:
                return path

        available = ", ".join(_normalize_prediction_name(path.name) for path in nc_files)
        raise FileNotFoundError(
            f"No NetCDF file named '{prediction}' in {target_dir}. "
            f"Available predictions: {available}."
        )


def _list_dir_names(path: Path) -> List[str]:
    return [child.name for child in sorted(path.iterdir()) if child.is_dir()]


def _normalize_prediction_name(value: str) -> str:
    """Strip prefixes/suffixes so Extrapolation_perfect maps to predictions file."""

    name = Path(value).name
    if name.endswith(".nc"):
        name = name[:-3]
    if name.startswith("predictions_"):
        name = name[len("predictions_"):]
    return name

