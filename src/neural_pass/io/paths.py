"""Project path helpers for neural_pass data layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths under the project root."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def aovs_dir(self) -> Path:
        return self.data_dir / "aovs"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "masks"

    @property
    def generations_dir(self) -> Path:
        return self.data_dir / "generations"

    @property
    def edits_dir(self) -> Path:
        return self.data_dir / "edits"

    @property
    def accepted_dir(self) -> Path:
        return self.data_dir / "accepted"

    @property
    def styles_dir(self) -> Path:
        return self.root / "configs" / "styles"

    def ensure_data_dirs(self) -> None:
        """Create data directories if missing."""
        for d in (
            self.aovs_dir,
            self.masks_dir,
            self.generations_dir,
            self.edits_dir,
            self.accepted_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
