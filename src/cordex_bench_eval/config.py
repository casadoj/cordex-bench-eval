"""
Configuration management for the CORDEX benchmark evaluation workflow.

This module provides the SubmissionWorkflowConfig class that defines where
submissions, ground truth data, and evaluation outputs are stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


def _normalize_extensions(exts: Iterable[str]) -> tuple[str, ...]:
    """
    Normalize archive extensions to lowercase with leading dots.
    
    Examples:
        >>> _normalize_extensions(["ZIP", ".tar.gz"])
        ('.tar.gz', '.zip')
    """
    cleaned = set()
    for ext in exts:
        suffix = ext.lower().lstrip(".")
        if suffix:
            cleaned.add(f".{suffix}")
    if not cleaned:
        raise ValueError("archive_extensions cannot be empty")
    return tuple(sorted(cleaned))


class SubmissionWorkflowConfig:
    """
    Configuration for the submission evaluation workflow.
    
    This class defines all the paths and settings needed to run the evaluation
    pipeline, including where to find submissions, ground truth data, and
    where to store evaluation results.
    
    Attributes
    ----------
    groundtruth_path : Path
        Directory containing ground truth NetCDF files.
    incoming_path : Path
        Directory where new submission archives (.zip) are placed.
    staging_path : Path
        Directory where submissions are extracted for processing.
    catalog_path : Path
        Path to the JSON file that tracks all registered submissions.
    mlbenchmark_path : Path
        Path to the ml-benchmark repository (for evaluation metrics).
    diagnostic_path : Path | None
        Directory where evaluation results are saved.
    archive_extensions : tuple[str, ...]
        Allowed archive file extensions (default: .zip).
    overwrite_existing : bool
        Whether to overwrite existing staged submissions.
    
    Example
    -------
    >>> config = SubmissionWorkflowConfig(
    ...     groundtruth_path="/data/target",
    ...     incoming_path="/data/incoming",
    ...     staging_path="/data/staging",
    ...     catalog_path="/data/catalog.json",
    ...     mlbenchmark_path="/code/ml-benchmark",
    ...     diagnostic_path="/data/diagnostics",
    ... )
    """

    def __init__(
        self,
        groundtruth_path: str | Path,
        incoming_path: str | Path,
        staging_path: str | Path,
        catalog_path: str | Path,
        mlbenchmark_path: str | Path,
        diagnostic_path: str | Path | None = None,
        archive_extensions: Sequence[str] | None = None,
        overwrite_existing: bool = False,
    ) -> None:
        self.groundtruth_path = Path(groundtruth_path).expanduser().resolve()
        self.incoming_path = Path(incoming_path).expanduser().resolve()
        self.staging_path = Path(staging_path).expanduser().resolve()
        self.catalog_path = Path(catalog_path).expanduser().resolve()
        self.mlbenchmark_path = Path(mlbenchmark_path).expanduser().resolve()
        self.diagnostic_path = (
            Path(diagnostic_path).expanduser().resolve() if diagnostic_path else None
        )
        self.archive_extensions = _normalize_extensions(archive_extensions or (".zip",))
        self.overwrite_existing = bool(overwrite_existing)
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.groundtruth_path.mkdir(parents=True, exist_ok=True)
        self.incoming_path.mkdir(parents=True, exist_ok=True)
        self.staging_path.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlbenchmark_path.mkdir(parents=True, exist_ok=True)
        if self.diagnostic_path is not None:
            self.diagnostic_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary (for JSON serialization)."""
        return {
            "groundtruth_path": str(self.groundtruth_path),
            "incoming_path": str(self.incoming_path),
            "staging_path": str(self.staging_path),
            "catalog_path": str(self.catalog_path),
            "mlbenchmark_path": str(self.mlbenchmark_path),
            "diagnostic_path": str(self.diagnostic_path) if self.diagnostic_path else None,
            "archive_extensions": list(self.archive_extensions),
            "overwrite_existing": self.overwrite_existing,
        }


def load_config_dict(data: dict) -> SubmissionWorkflowConfig:
    """
    Create a SubmissionWorkflowConfig from a dictionary.
    
    Parameters
    ----------
    data : dict
        Configuration dictionary with keys matching constructor parameters.
    
    Returns
    -------
    SubmissionWorkflowConfig
        Configured workflow settings.
    """
    return SubmissionWorkflowConfig(
        groundtruth_path=data["groundtruth_path"],
        incoming_path=data["incoming_path"],
        staging_path=data["staging_path"],
        catalog_path=data["catalog_path"],
        mlbenchmark_path=data["mlbenchmark_path"],
        diagnostic_path=data.get("diagnostic_path"),
        archive_extensions=data.get("archive_extensions", (".zip",)),
        overwrite_existing=data.get("overwrite_existing", False),
    )


def load_config_file(file_path: str | Path) -> SubmissionWorkflowConfig:
    """
    Load configuration from a JSON file.
    
    Parameters
    ----------
    file_path : str | Path
        Path to the JSON configuration file.
    
    Returns
    -------
    SubmissionWorkflowConfig
        Configured workflow settings.
    
    Example
    -------
    >>> config = load_config_file("configs/config.json")
    """
    data = json.loads(Path(file_path).read_text())
    return load_config_dict(data)
