from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import xarray as xr
from .catalog import SubmissionCatalog, CatalogEntry
from .config import SubmissionWorkflowConfig


class SubmissionHandler:
    """Coordinate discovery, extraction, and cataloguing of submission archives."""

    def __init__(self, config: SubmissionWorkflowConfig):
        self.config = config
        self.catalog = SubmissionCatalog(config.catalog_path)

    def ingest(self) -> List[CatalogEntry]:
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
        allowed = tuple(ext.lower() for ext in self.config.archive_extensions)
        for candidate in sorted(self.config.incoming_dir.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in allowed:
                yield candidate

    def _stage_archive(self, archive_path: Path) -> Path:
        destination = self.config.staging_dir / archive_path.stem
        if destination.exists():
            if not self.config.overwrite_existing:
                raise FileExistsError(
                    f"The destination folder {destination} already exists. "
                    "Enable overwrite_existing to replace it.")
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(destination)
        return destination

    def _extract_metadata(self, staged_dir: Path) -> Dict[str, str]:
        """
        Read submission-level metadata from a single NetCDF file.
        The metadata must be uniform across all files in the submission.
        """

        desired_attrs = (
            "esd_method_id",
            "esd_method_version",
            "training_methodology_id",
        )

        nc_file = next(staged_dir.rglob("*.nc"), None)
        if nc_file is None:
            return {}
        try:
            with xr.open_dataset(nc_file, decode_cf=False) as ds:
                extracted = {key: str(ds.attrs[key])
                             for key in desired_attrs if key in ds.attrs}
                return extracted
        except Exception as exc:
            return {}