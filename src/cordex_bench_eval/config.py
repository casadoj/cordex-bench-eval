from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


def _normalize_extensions(exts: Iterable[str]) -> tuple[str, ...]:
    """Normalize archive extensions to lowercase and remove leading dots."""

    cleaned = set()
    for ext in exts:
        suffix = ext.lower().lstrip(".")
        if suffix:
            cleaned.add(f".{suffix}")
    if not cleaned:
        raise ValueError("archive_extensions cannot be empty")
    return tuple(sorted(cleaned))


class SubmissionWorkflowConfig:
    """Configuration describing where submissions live."""

    def __init__(self,
                incoming_dir: str | Path,
                staging_dir: str | Path,
                catalog_path: str | Path,
                archive_extensions: Sequence[str] | None = None,
                overwrite_existing: bool = False) -> None:

        self.incoming_dir = Path(incoming_dir).expanduser().resolve()
        self.staging_dir = Path(staging_dir).expanduser().resolve()
        self.catalog_path = Path(catalog_path).expanduser().resolve()
        self.archive_extensions = _normalize_extensions(archive_extensions or (".zip",))
        self.overwrite_existing = bool(overwrite_existing)
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {"incoming_dir": str(self.incoming_dir),
                "staging_dir": str(self.staging_dir),
                "catalog_path": str(self.catalog_path),
                "archive_extensions": list(self.archive_extensions),
                "overwrite_existing": self.overwrite_existing}


def load_config_dict(data: dict) -> SubmissionWorkflowConfig:
    """Build a SubmissionWorkflowConfig from a dictionary."""

    return SubmissionWorkflowConfig(incoming_dir=data["incoming_dir"],
                                    staging_dir=data["staging_dir"],
                                    catalog_path=data["catalog_path"],
                                    archive_extensions=data.get("archive_extensions", (".zip",)),
                                    overwrite_existing=data.get("overwrite_existing", False))


def load_config_file(file_path: str | Path) -> SubmissionWorkflowConfig:
    """Build a SubmissionWorkflowConfig from a JSON file on disk."""

    data = json.loads(Path(file_path).read_text())
    return load_config_dict(data)