"""Post-train / fine-tune a base model into a customized style model.

Each style YAML describes one style and points at a base_model_id plus
train hyperparameters. Accepted flywheel pairs under data/accepted/<style_id>
are the intended supervision source.

STUB: validates config presence and prints intent; no optimizer steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_pass.io.paths import ProjectPaths
from neural_pass.io.style_config import StyleConfig, load_style_config
from neural_pass.models.base import PlaceholderStyleModel


def train_style(
    *,
    style_config_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune / post-train for a style config (stub)."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = ProjectPaths(root)
    style: StyleConfig = load_style_config(style_config_path)

    data_dir = paths.accepted_dir / style.style_id
    print(
        f"  [train_style] style={style.style_id!r} "
        f"base={style.base_model_id!r} data={data_dir} (stub)"
    )
    print(f"  [train_style] hyperparams={style.train_hyperparams}")

    model = PlaceholderStyleModel(style)
    model.load()
    # fine_tune is stubbed on the model protocol.
    train_result = model.fine_tune({"data_dir": str(data_dir), **style.train_hyperparams})

    return {
        "style_id": style.style_id,
        "base_model_id": style.base_model_id,
        "style_model_path": style.style_model_path,
        "data_dir": str(data_dir),
        "train_result": train_result,
        "status": "stub",
        "note": "No weights updated — implement fine_tune later.",
    }


def main_stub() -> None:
    raise SystemExit("Use scripts/train_style.py — this entry is a package metadata stub.")
