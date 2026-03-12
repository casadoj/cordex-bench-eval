"""
Submission validation utilities for staged CORDEX benchmark submissions.

This module provides a validator that checks whether staged submissions
contain all files required by configured experiments, with correct naming and
minimal NetCDF structure needed by evaluators and figure generators.

Validation coverage includes:
- expected file presence across domain/training/evaluation combinations,
- file naming convention checks,
- lightweight NetCDF readability/corruption probes,
- required variables (`pr`, `tasmax`) and dimensions (`time` + spatial axes),
- time coverage against filename date ranges,
- reporting of explicit missing file paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .catalog import SubmissionCatalog
from .config import SubmissionWorkflowConfig
from .mappings import EVAL_SETTINGS, GCM_EVAL, GCM_TRAIN, PERIOD_DATES


@dataclass(slots=True)
class ValidationIssue:
    """
    Structured validation issue.

    Attributes
    ----------
    severity : str
        Issue severity ("ERROR" or "WARNING").
    code : str
        Stable machine-readable issue code.
    message : str
        Human-readable issue description.
    context : dict[str, str]
        Optional context fields such as domain/experiment/file.
    """

    severity: str
    code: str
    message: str
    context: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SubmissionValidationReport:
    """
    Validation report for one submission.

    Attributes
    ----------
    submission_name : str
        Submission identifier.
    submission_path : str
        Absolute path to the submission root directory.
    issues : list[ValidationIssue]
        All collected validation issues.
    checked_files : int
        Number of existing NetCDF files checked.
    missing_files : int
        Number of required files that were missing.
    """

    submission_name: str
    submission_path: str
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_files: int = 0
    missing_files: int = 0

    @property
    def is_valid(self) -> bool:
        """Return True when no ERROR-level issues were recorded."""
        return all(issue.severity != "ERROR" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Count ERROR-level issues."""
        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        """Count WARNING-level issues."""
        return sum(1 for issue in self.issues if issue.severity == "WARNING")

    @property
    def missing_file_paths(self) -> list[str]:
        """
        Return all missing/invalid expected file paths collected by validation.

        Returns
        -------
        list[str]
            Sorted unique file paths from missing-file related issue codes.
        """
        missing_codes = {"MISSING_REQUIRED_FILE", "EXPECTED_FILE_NOT_A_FILE"}
        paths = {
            issue.context["file"]
            for issue in self.issues
            if issue.code in missing_codes and "file" in issue.context
        }
        return sorted(paths)


@dataclass(slots=True)
class _ExpectedFile:
    """Expected prediction file definition for one scenario."""

    domain: str
    training_experiment: str
    evaluation_experiment: str
    period: str
    condition: str
    gcm_name: str
    date_tag: str
    filename: str
    relative_path: Path


class SubmissionValidator:
    """
    Validate staged submissions against required scenarios and file schemas.

    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Workflow configuration with staging and catalog paths.
    catalog : SubmissionCatalog, optional
        Submission catalog. If omitted, it is loaded from config.catalog_path.
    """

    _REQUIRED_VARIABLES = ("pr", "tasmax")
    _TIME_DIM = "time"
    _TIME_COVERAGE_BOUNDARY_TOLERANCE = 1

    def __init__(self, config: SubmissionWorkflowConfig, catalog: SubmissionCatalog | None = None):
        self.config = config
        self.catalog = catalog or SubmissionCatalog(config.catalog_path)

    def validate_submission(self, submission_name: str, domains: Sequence[str] | None = None,
                            training_experiments: Sequence[str] | None = None,
                            evaluation_experiments: Sequence[str] | None = None,
                            check_corruption: bool = True) -> SubmissionValidationReport:
        """
        Validate a single submission by name.

        The validator first resolves the submission path from the catalog. If the
        submission is not in the catalog, it falls back to
        `<staging_path>/<submission_name>`.

        Parameters
        ----------
        submission_name : str
            Name of the submission.
        domains : sequence of str, optional
            Domains to validate (default: all known domains).
        training_experiments : sequence of str, optional
            Training experiments to validate (default: all known experiments).
        evaluation_experiments : sequence of str, optional
            Evaluation experiments to validate for each training experiment
            (default: all supported for each training experiment).
        check_corruption : bool, default=True
            If True, run a lightweight content probe per required variable to
            catch likely corrupted files without full-data reads.

        Returns
        -------
        SubmissionValidationReport
            Validation outcome and issue details.
        """
        entry = self.catalog.get(submission_name)
        if entry is not None:
            submission_path = Path(entry.location).expanduser().resolve()
        else:
            submission_path = (self.config.staging_path / submission_name).expanduser().resolve()

        return self._validate_submission_path(submission_name=submission_name,
                                              submission_path=submission_path,
                                              domains=domains,
                                              training_experiments=training_experiments,
                                              evaluation_experiments=evaluation_experiments,
                                              check_corruption=check_corruption)

    def validate_all_staged(self, domains: Sequence[str] | None = None,
                            training_experiments: Sequence[str] | None = None,
                            evaluation_experiments: Sequence[str] | None = None,
                            check_corruption: bool = True) -> dict[str, SubmissionValidationReport]:
        """
        Validate all staged submissions found in the staging directory.

        Parameters
        ----------
        domains : sequence of str, optional
            Domains to validate.
        training_experiments : sequence of str, optional
            Training experiments to validate.
        evaluation_experiments : sequence of str, optional
            Evaluation experiments to validate.
        check_corruption : bool, default=True
            If True, run a lightweight content probe per required variable to
            catch likely corrupted files without full-data reads.

        Returns
        -------
        dict[str, SubmissionValidationReport]
            Mapping of submission names to validation reports.
        """
        reports: dict[str, SubmissionValidationReport] = {}
        for submission_dir in sorted(self.config.staging_path.iterdir()):
            if not submission_dir.is_dir():
                continue
            report = self._validate_submission_path(submission_name=submission_dir.name,
                                                    submission_path=submission_dir,
                                                    domains=domains,
                                                    training_experiments=training_experiments,
                                                    evaluation_experiments=evaluation_experiments,
                                                    check_corruption=check_corruption)
            reports[submission_dir.name] = report
        return reports

    def _validate_submission_path(self, submission_name: str, submission_path: Path,
                                  domains: Sequence[str] | None, training_experiments: Sequence[str] | None,
                                  evaluation_experiments: Sequence[str] | None,
                                  check_corruption: bool) -> SubmissionValidationReport:
        """Validate one staged submission path."""
        report = SubmissionValidationReport(submission_name=submission_name,
                                            submission_path=str(submission_path))

        if not submission_path.exists():
            self._add_issue(report=report,
                            severity="ERROR",
                            code="MISSING_SUBMISSION_ROOT",
                            message="Submission directory does not exist.",
                            context={"path": str(submission_path)})
            return report
        if not submission_path.is_dir():
            self._add_issue(report=report,
                            severity="ERROR",
                            code="INVALID_SUBMISSION_ROOT",
                            message="Submission root exists but is not a directory.",
                            context={"path": str(submission_path)})
            return report

        expected_files = self._iter_expected_files(domains=domains,
                                                   training_experiments=training_experiments,
                                                   evaluation_experiments=evaluation_experiments)
        for expected in expected_files:
            resolved_path = submission_path / expected.relative_path
            if not resolved_path.exists():
                report.missing_files += 1
                self._add_issue(report=report,
                                severity="ERROR",
                                code="MISSING_REQUIRED_FILE",
                                message="Required prediction file is missing.",
                                context={"file": str(resolved_path),
                                         "domain": expected.domain,
                                         "training_experiment": expected.training_experiment,
                                         "evaluation_experiment": expected.evaluation_experiment})
                continue
            if not resolved_path.is_file():
                report.missing_files += 1
                self._add_issue(report=report,
                                severity="ERROR",
                                code="EXPECTED_FILE_NOT_A_FILE",
                                message="Expected prediction path exists but is not a file.",
                                context={"file": str(resolved_path)})
                continue

            report.checked_files += 1
            self._validate_netcdf_file(path=resolved_path,
                                       expected=expected,
                                       report=report,
                                       check_corruption=check_corruption)

        return report

    def _iter_expected_files(self, domains: Sequence[str] | None, training_experiments: Sequence[str] | None,
                             evaluation_experiments: Sequence[str] | None) -> list[_ExpectedFile]:
        """Build expected file definitions from mapping tables."""
        domain_values = [self._normalize_domain_name(domain) for domain in (domains or self._default_domains())]
        training_values = list(training_experiments or sorted(EVAL_SETTINGS.keys()))
        evaluation_filter = set(evaluation_experiments or [])

        expected_files: list[_ExpectedFile] = []
        for domain in domain_values:
            domain_key = self._normalize_domain_key(domain)
            if domain_key not in GCM_TRAIN or domain_key not in GCM_EVAL:
                continue

            for training_experiment in training_values:
                if training_experiment not in EVAL_SETTINGS:
                    continue

                eval_settings = EVAL_SETTINGS[training_experiment]
                eval_names = sorted(eval_settings.keys())
                if evaluation_filter:
                    eval_names = [name for name in eval_names if name in evaluation_filter]

                for evaluation_experiment in eval_names:
                    period, condition, same_gcm_as_train = eval_settings[evaluation_experiment]
                    gcm_name = GCM_TRAIN[domain_key] if same_gcm_as_train else GCM_EVAL[domain_key]
                    for period_name in self._expand_period(period):
                        date_tag = PERIOD_DATES[period_name]
                        filename = f"Predictions_pr_tasmax_{gcm_name}_{date_tag}.nc"
                        relative_path = Path(domain) / training_experiment / period_name / condition / filename
                        expected_files.append(_ExpectedFile(domain=domain,
                                                            training_experiment=training_experiment,
                                                            evaluation_experiment=evaluation_experiment,
                                                            period=period_name,
                                                            condition=condition,
                                                            gcm_name=gcm_name,
                                                            date_tag=date_tag,
                                                            filename=filename,
                                                            relative_path=relative_path))
        return expected_files

    def _validate_netcdf_file(self, path: Path, expected: _ExpectedFile,
                              report: SubmissionValidationReport,
                              check_corruption: bool) -> None:
        """Validate NetCDF variables and dimensions for one required file."""
        try:
            with xr.open_dataset(path, decode_cf=False) as ds:
                # With decode_cf=False, auxiliary spatial variables such as
                # lat/lon may remain in data_vars; promote them so variable
                # structure checks can detect explicit geographic coordinates.
                spatial_coord_vars = [
                    name
                    for name in ("lat", "lon", "latitude", "longitude", "x", "y")
                    if name in ds.data_vars
                ]
                if spatial_coord_vars:
                    ds = ds.set_coords(spatial_coord_vars)

                missing_vars = [var for var in self._REQUIRED_VARIABLES if var not in ds.data_vars]
                if missing_vars:
                    self._add_issue(report=report,
                                    severity="ERROR",
                                    code="MISSING_REQUIRED_VARIABLES",
                                    message=f"Dataset is missing required variables: {', '.join(missing_vars)} (in file: {path}).",
                                    context={"file": str(path)})

                for variable in self._REQUIRED_VARIABLES:
                    if variable not in ds.data_vars:
                        continue
                    if check_corruption:
                        self._probe_variable_readability(path=path,
                                                         variable=variable,
                                                         data_array=ds[variable],
                                                         report=report)
                    self._validate_variable_structure(path=path,
                                                      variable=variable,
                                                      data_array=ds[variable],
                                                      report=report,
                                                      domain=expected.domain)

                self._validate_time_coverage(ds=ds, path=path, expected=expected, report=report)
                self._validate_file_naming(path=path, expected=expected, report=report)
        except Exception as exc:
            self._add_issue(report=report,
                            severity="ERROR",
                            code="UNREADABLE_NETCDF_FILE",
                            message=f"Failed to open NetCDF dataset: {exc}",
                            context={"file": str(path)})

    def _probe_variable_readability(self, path: Path, variable: str,
                                    data_array: xr.DataArray,
                                    report: SubmissionValidationReport) -> None:
        """
        Perform a tiny eager read to detect likely NetCDF content corruption.

        This check intentionally reads only the first cell (across all dims) to
        keep runtime low while still forcing backend I/O.
        """
        try:
            if not data_array.dims:
                _ = data_array.load()
                return
            first_cell = data_array.isel({dim: 0 for dim in data_array.dims})
            _ = first_cell.load()
        except Exception as exc:
            self._add_issue(report=report,
                            severity="ERROR",
                            code="CORRUPTED_NETCDF_CONTENT",
                            message=(
                                f"Lightweight content probe failed for variable '{variable}': {exc}"
                            ),
                            context={"file": str(path), "variable": variable})

    def _validate_file_naming(self, path: Path, expected: _ExpectedFile,
                              report: SubmissionValidationReport) -> None:
        """Validate filename against expected naming pattern for scenario."""
        if path.name == expected.filename:
            return
        self._add_issue(report=report,
                        severity="ERROR",
                        code="INVALID_FILENAME",
                        message="Prediction filename does not match expected naming convention.",
                        context={"file": str(path),
                                 "expected_filename": expected.filename,
                                 "domain": expected.domain,
                                 "training_experiment": expected.training_experiment,
                                 "evaluation_experiment": expected.evaluation_experiment})

    def _validate_variable_structure(self, path: Path, variable: str, data_array: xr.DataArray,
                                     report: SubmissionValidationReport, domain: str) -> None:
        """Validate variable dimensions needed by evaluation and daily plotting."""
        if self._TIME_DIM not in data_array.dims:
            self._add_issue(report=report,
                            severity="ERROR",
                            code="MISSING_TIME_DIMENSION",
                            message=f"Variable '{variable}' must contain a '{self._TIME_DIM}' dimension (in file: {path}).",
                            context={"file": str(path), "variable": variable})
        elif int(data_array.sizes.get(self._TIME_DIM, 0)) <= 0:
            self._add_issue(report=report,
                            severity="ERROR",
                            code="EMPTY_TIME_DIMENSION",
                            message=f"Variable '{variable}' has an empty '{self._TIME_DIM}' dimension (in file: {path}).",
                            context={"file": str(path), "variable": variable})

        lat_dim = self._infer_spatial_dim(tuple(data_array.dims), axis="y")
        lon_dim = self._infer_spatial_dim(tuple(data_array.dims), axis="x")
        if lat_dim is None:
            self._add_issue(report=report,
                            severity="ERROR",
                            code="MISSING_SPATIAL_LAT_DIMENSION",
                            message=f"Unable to infer latitude dimension for variable '{variable}' (in file: {path}).",
                            context={"file": str(path), "variable": variable})
        if lon_dim is None:
            self._add_issue(report=report,
                            severity="ERROR",
                            code="MISSING_SPATIAL_LON_DIMENSION",
                            message=f"Unable to infer longitude dimension for variable '{variable}' (in file: {path}).",
                            context={"file": str(path), "variable": variable})

        coord_names = {name.lower() for name in data_array.coords}
        has_lat_lon_coords = {"lat", "lon"}.issubset(coord_names) or {
            "latitude", "longitude"
        }.issubset(coord_names)
        has_xy_coords = {"x", "y"}.issubset(coord_names)
        domain_key = self._normalize_domain_key(domain)

        if domain_key == "ALPS":
            if not has_xy_coords:
                self._add_issue(report=report,
                                severity="ERROR",
                                code="MISSING_REQUIRED_X_Y_COORDINATES",
                                message=(
                                    f"Variable '{variable}' in domain '{domain}' must expose "
                                    f"explicit x/y coordinates (in file: {path})."
                                ),
                                context={"file": str(path), "variable": variable, "domain": domain})
        if not has_lat_lon_coords and (lat_dim is not None and lon_dim is not None):
            self._add_issue(report=report,
                            severity="WARNING",
                            code="MISSING_LAT_LON_COORDINATES",
                            message=(
                                f"Variable '{variable}' does not expose explicit lat/lon coordinates (in file: {path}). "
                                "Plotting may rely on raw dimension coordinates."
                            ),
                            context={"file": str(path), "variable": variable,
                                     "lat_dim": lat_dim, "lon_dim": lon_dim})

    def _validate_time_coverage(self, ds: xr.Dataset, path: Path,
                                expected: _ExpectedFile,
                                report: SubmissionValidationReport) -> None:
        """Check that all years in the expected date range are present in the time coordinate."""
        if self._TIME_DIM not in ds or ds[self._TIME_DIM].size == 0:
            return

        time_var = ds[self._TIME_DIM]
        try:
            units = time_var.attrs.get("units", "")
            calendar = time_var.attrs.get("calendar", "standard")
            decoded = xr.coding.times.decode_cf_datetime(
                time_var.values, units, calendar
            )
        except Exception:
            return

        start_year, end_year = int(expected.date_tag[:4]), int(expected.date_tag[5:9])
        expected_years = set(range(start_year, end_year + 1))
        if isinstance(decoded, np.ndarray) and np.issubdtype(decoded.dtype, np.datetime64):
            actual_years = set(pd.DatetimeIndex(decoded).year)
        else:
            actual_years = {int(v.year) for v in decoded}
        all_missing = expected_years - actual_years
        extra = sorted(actual_years - expected_years)
        actual_sorted = sorted(actual_years)

        tolerated_start = set(range(start_year, start_year + self._TIME_COVERAGE_BOUNDARY_TOLERANCE))
        tolerated_end = set(range(end_year - self._TIME_COVERAGE_BOUNDARY_TOLERANCE + 1, end_year + 1))
        tolerated_boundary = tolerated_start | tolerated_end
        tolerated = sorted(all_missing & tolerated_boundary)
        missing = sorted(all_missing - tolerated_boundary)

        if missing:
            self._add_issue(
                report=report, severity="ERROR",
                code="INCOMPLETE_TIME_COVERAGE",
                message=(
                    f"Time coordinate is missing years implied by the filename date "
                    f"range {expected.date_tag}: {missing}. "
                    f"File contains years: {actual_sorted} (in file: {path})."
                ),
                context={"file": str(path),
                         "expected_range": expected.date_tag,
                         "missing_years": str(missing),
                         "actual_years": str(actual_sorted)},
            )
        if tolerated:
            self._add_issue(
                report=report, severity="WARNING",
                code="TOLERATED_BOUNDARY_YEAR_OFFSET",
                message=(
                    f"Time coordinate is missing boundary year(s) of the filename date "
                    f"range {expected.date_tag}: {tolerated}. This is within the allowed "
                    f"boundary tolerance. "
                    f"File contains years: {actual_sorted} (in file: {path})."
                ),
                context={"file": str(path),
                         "expected_range": expected.date_tag,
                         "tolerated_years": str(tolerated),
                         "actual_years": str(actual_sorted)},
            )
        if extra:
            self._add_issue(
                report=report, severity="WARNING",
                code="UNEXPECTED_TIME_COVERAGE",
                message=(
                    f"Time coordinate contains years outside the filename date "
                    f"range {expected.date_tag}: {extra}. "
                    f"File contains years: {actual_sorted} (in file: {path})."
                ),
                context={"file": str(path),
                         "expected_range": expected.date_tag,
                         "extra_years": str(extra),
                         "actual_years": str(actual_sorted)},
            )

    @staticmethod
    def _add_issue(report: SubmissionValidationReport, severity: str, code: str,
                   message: str, context: dict[str, str] | None = None) -> None:
        """Append a structured issue to the report."""
        report.issues.append(ValidationIssue(severity=severity,
                                             code=code,
                                             message=message,
                                             context=context or {}))

    @staticmethod
    def _expand_period(period: str) -> list[str]:
        """Expand combined periods used by mapping tables."""
        if period == "mid_end_century":
            return ["mid_century", "end_century"]
        return [period]

    @staticmethod
    def _infer_spatial_dim(dimensions: tuple[str, ...], axis: str) -> str | None:
        """Infer spatial dimension names from common aliases."""
        candidates = ("lat", "latitude", "y") if axis == "y" else ("lon", "longitude", "x")
        for dim in dimensions:
            lowered = dim.lower()
            if any(token in lowered for token in candidates):
                return dim
        return None

    @staticmethod
    def _normalize_domain_key(domain: str) -> str:
        """Normalize domain names to mapping keys (ALPS/NZ/SA)."""
        value = Path(domain).name
        for suffix in ("_domain", "_Domain"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        return value.upper()

    @classmethod
    def _normalize_domain_name(cls, domain: str) -> str:
        """Normalize domain value to canonical `<KEY>_Domain` form."""
        return f"{cls._normalize_domain_key(domain)}_Domain"

    @staticmethod
    def _default_domains() -> Iterable[str]:
        """Return canonical default domain names from mapping tables."""
        return [f"{domain}_Domain" for domain in sorted(GCM_TRAIN.keys())]
