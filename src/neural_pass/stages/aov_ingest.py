"""Load and validate Arnold condition AOVs.

Expected AOV kinds (vision): position, normal, depth, motion_vector, …
Inputs land under `data/aovs/` (or a path passed on the CLI).

STUB: does not parse EXR/TIFF; returns a placeholder inventory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Documented AOV names the real ingest will look for.
EXPECTED_AOVS = ("position", "normal", "depth", "motion_vector")


def load_aovs(aov_dir: str | Path) -> dict[str, Any]:
    """Load/validate Arnold AOVs from a directory (stub).

    Returns a placeholder dict describing what was found / expected.
    Does not invent image data or fake metrics.
    """
    directory = Path(aov_dir)
    print(f"  [aov_ingest] scanning {directory} (stub)")

    found: list[str] = []
    missing: list[str] = []
    if directory.is_dir():
        names = {p.stem.lower() for p in directory.iterdir() if p.is_file()}
        for name in EXPECTED_AOVS:
            # Accept either exact stem or stem prefix (e.g. position.exr)
            if any(n == name or n.startswith(name) for n in names):
                found.append(name)
            else:
                missing.append(name)
    else:
        print(f"  [aov_ingest] WARNING: directory does not exist: {directory}")
        missing = list(EXPECTED_AOVS)

    # TODO: open EXRs, check channels/resolution, validate world-space ranges.
    return {
        "dir": str(directory),
        "expected": list(EXPECTED_AOVS),
        "found": found,
        "missing": missing,
        "status": "stub",
        "note": "No AOV pixels loaded — implement real ingest later.",
    }
