"""
Submission catalog for tracking registered submissions.

The catalog is a simple JSON file that stores metadata about each submission
that has been ingested into the staging area.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

@dataclass(slots=True)
class CatalogEntry:
    """
    Metadata for a single staged submission.
    
    Attributes
    ----------
    name : str
        Unique identifier for the submission.
    location : str
        Path to the staged submission directory.
    created_at : str
        ISO-format timestamp when the submission was registered.
    status : str
        Current status of the submission.
    extra : dict
        Additional metadata from NetCDF attributes.
    """
    name: str
    location: str
    created_at: str
    status: str = "staged"
    extra: Dict[str, str] = field(default_factory=dict)

def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()

class SubmissionCatalog:
    """
    JSON-backed catalog that records staged submissions.
    
    The catalog persists to a JSON file and provides methods to register
    new submissions and query existing ones.
    
    Parameters
    ----------
    path : str | Path
        Path to the catalog JSON file (will be created if it doesn't exist).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._data: Dict[str, CatalogEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load existing catalog entries from disk."""
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for entry in raw.get("entries", []):
            catalog_entry = CatalogEntry(name=entry["name"],
                                         location=entry["location"],
                                         created_at=entry["created_at"],
                                         status=entry.get("status", "staged"),
                                         extra=entry.get("extra", {}))
            self._data[catalog_entry.name] = catalog_entry

    def save(self) -> None:
        """Persist catalog to disk."""
        payload = {"entries": [asdict(entry) for entry in self._data.values()]}
        self.path.write_text(json.dumps(payload, indent=2))

    def register(self, name: str, location: Path,
                 *, extra: Dict[str, str] | None = None) -> CatalogEntry:
        """
        Register a new submission in the catalog.
        
        Parameters
        ----------
        name : str
            Unique name for the submission.
        location : Path
            Path to the staged submission directory.
        extra : dict, optional
            Additional metadata to store.
        
        Returns
        -------
        CatalogEntry
            The newly created catalog entry.
        """
        entry = CatalogEntry(name=name,
                             location=str(location),
                             created_at=_now_iso(),
                             extra=extra or {})
        self._data[name] = entry
        return entry

    def all_entries(self) -> Iterable[CatalogEntry]:
        """Return all registered submissions."""
        return self._data.values()

    def get(self, name: str) -> CatalogEntry | None:
        """
        Look up a submission by name.
        
        Parameters
        ----------
        name : str
            Submission name to look up.
        
        Returns
        -------
        CatalogEntry | None
            The matching entry, or None if not found.
        """
        return self._data.get(name)
    
    def uses_orography(self, name: str) -> bool:
        """
        Check if a submission uses orography as a predictor.
        
        Parameters
        ----------
        name : str
            Submission name to look up.
        
        Returns
        -------
        bool
            True if the submission uses orography, False otherwise.
        """
        entry = self.get(name)
        if entry is None:
            return False
        return entry.extra.get("uses_orography", "false").lower() == "true"