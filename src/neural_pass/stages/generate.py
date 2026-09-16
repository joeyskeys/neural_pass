"""Call the style model for base generation.

Style is selected via the style YAML (base / fine-tuned checkpoint ids).
User workflow: run basic generation → iteratively modify until satisfied.

STUB: does not run a model; writes no images; returns a placeholder record.
"""

from __future__ import annotations

from typing import Any, Mapping

from neural_pass.io.paths import ProjectPaths
from neural_pass.io.style_config import StyleConfig
from neural_pass.models.base import PlaceholderStyleModel


def generate_image(
    *,
    style: StyleConfig,
    conditioning: Mapping[str, Any],
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Run style-model generation (stub)."""
    out_dir = paths.generations_dir
    print(
        f"  [generate] style={style.style_id!r} "
        f"base_model={style.base_model_id!r} → {out_dir} (stub)"
    )

    model = PlaceholderStyleModel(style)
    # load/generate are stubs — they return placeholders / raise as documented.
    model.load()
    result = model.generate(conditioning)

    return {
        "style_id": style.style_id,
        "output_dir": str(out_dir),
        "output_path": None,  # no file written in stub mode
        "model_result": result,
        "status": "stub",
        "note": "No image generated — implement StyleModel.generate later.",
    }
