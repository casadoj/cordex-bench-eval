from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


@dataclass(slots=True)
class CatalogEntry:
    """Metadata describing a staged submission."""

    name: str
    location: str
    created_at: str
    status: str = "staged"
    extra: Dict[str, str] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SubmissionCatalog:
    """JSON-backed catalogue that records staged submissions."""

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, CatalogEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        entries = raw.get("entries", [])
        for entry in entries:
            catalog_entry = CatalogEntry(
                name=entry["name"],
                location=entry["location"],
                created_at=entry["created_at"],
                status=entry.get("status", "staged"),
                extra=entry.get("extra", {}),
            )
            self._data[catalog_entry.name] = catalog_entry

    def save(self) -> None:
        payload = {"entries": [asdict(entry) for entry in self._data.values()]}
        self.path.write_text(json.dumps(payload, indent=2))

    def register(self, name: str, location: Path, *, extra: Dict[str, str] | None = None) -> CatalogEntry:
        entry = CatalogEntry(
            name=name,
            location=str(location),
            created_at=_now_iso(),
            extra=extra or {},
        )
        self._data[name] = entry
        return entry

    def all_entries(self) -> Iterable[CatalogEntry]:
        return self._data.values()

    def get(self, name: str) -> CatalogEntry | None:
        """Return a catalog entry by name if it exists."""
        return self._data.get(name)

