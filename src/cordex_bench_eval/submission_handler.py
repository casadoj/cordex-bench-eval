"""
Submission ingestion handler.

This module handles the process of discovering new submission archives,
extracting them to the staging area, and registering them in the catalog.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import xarray as xr

from .catalog import SubmissionCatalog, CatalogEntry
from .config import SubmissionWorkflowConfig

class SubmissionHandler:
    """
    Coordinate discovery, extraction, and cataloguing of submission archives.
    
    This class watches the incoming directory for new .zip archives, extracts
    them to the staging directory, and registers them in the catalog.
    
    Parameters
    ----------
    config : SubmissionWorkflowConfig
        Configuration specifying paths and settings.
    """

    def __init__(self, config: SubmissionWorkflowConfig):
        self.config = config
        self.catalog = SubmissionCatalog(config.catalog_path)

    def ingest(self) -> List[CatalogEntry]:
        """
        Process all new submission archives.
        
        Discovers archives in the incoming directory, extracts them to staging,
        and registers them in the catalog.
        
        Returns
        -------
        List[CatalogEntry]
            List of newly registered submissions.
        """
        ingested: List[CatalogEntry] = []
        for archive in self._discover_archives():
            dest = self._stage_archive(archive)
            metadata = self._extract_metadata(dest)
            entry = self.catalog.register(archive.stem, dest, extra=metadata)
            ingested.append(entry)
        if ingested:
            self.catalog.save()
        return ingested

    def _discover_archives(self) -> Iterable[Path]:
        """Find all archive files in the incoming directory."""
        allowed = tuple(ext.lower() for ext in self.config.archive_extensions)
        for candidate in sorted(self.config.incoming_path.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in allowed:
                yield candidate

    def _stage_archive(self, archive_path: Path) -> Path:
        """Extract archive to staging directory."""
        destination = self.config.staging_path / archive_path.stem
        if destination.exists():
            if not self.config.overwrite_existing:
                raise FileExistsError(
                    f"The destination folder {destination} already exists. "
                    "Enable overwrite_existing to replace it."
                )
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(destination)
        return destination

    def _extract_metadata(self, staged_dir: Path) -> Dict[str, str]:
        """
        Extract submission metadata from a NetCDF file.
        
        Looks for standard attributes in the first NetCDF file found,
        including CORDEX ML-Benchmark-specific attributes.
        
        Parameters
        ----------
        staged_dir : Path
            Root directory of the staged submission.
        
        Returns
        -------
        dict
            Extracted metadata attributes.
        """
        desired_attrs = ("esd_method_id",
                         "esd_method_version",
                         "training_methodology_id",
                         "project_id",
                         "activity_id",
                         "product",
                         "benchmark_id",
                         "institution_id",
                         "institution",
                         "contact",
                         "creation_date",
                         "emulator_id",
                         "emulator",
                         "training_id",
                         "training",
                         "stochastic_output",
                         "version_realization",
                         "version_realization_info",
                         "reference_url",
                         "reproducibility_url")

        nc_file = next(staged_dir.rglob("*.nc"), None)
        if nc_file is None:
            return {}
        try:
            with xr.open_dataset(nc_file, decode_cf=False) as ds:
                return {key: str(ds.attrs[key]) for key in desired_attrs if key in ds.attrs}
        except Exception:
            return {}
