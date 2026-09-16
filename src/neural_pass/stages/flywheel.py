"""Record accepted finals as labeled training data.

Flywheel idea: more use → more corrections → more accepted pairs →
model closer to project/user → less editing next time.

Pairs are keyed by style id and land under `data/accepted/`.

STUB: builds a metadata record only; does not copy image files.
"""

from __future__ import annotations

from typing import Any, Mapping

from neural_pass.io.paths import ProjectPaths
from neural_pass.io.style_config import StyleConfig


def record_accepted(
    *,
    style: StyleConfig,
    generation: Mapping[str, Any],
    edit: Mapping[str, Any],
    conditioning: Mapping[str, Any],
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Record an accepted result as a labeled train sample (stub)."""
    dest = paths.accepted_dir / style.style_id
    print(f"  [flywheel] would record sample under {dest} (stub)")

    # TODO: copy generation + edit (+ conditioning refs) into dest; write manifest.
    return {
        "style_id": style.style_id,
        "dest_dir": str(dest),
        "generation_ref": generation.get("output_path"),
        "edit_ref": edit.get("edit_path"),
        "conditioning_ref": conditioning.get("mask_path"),
        "written": False,
        "status": "stub",
        "note": "No files written — implement labeled-pair packaging later.",
    }
