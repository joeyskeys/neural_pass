"""Load and lightly validate style YAML configs.

Each config = one style and requires post-training / fine-tuning of a base
model into a customized style model (see train_style stage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class StyleConfig:
    """In-memory style config used by generate + train stubs."""

    style_id: str
    name: str
    description: str
    prompt: str
    base_model_id: str
    style_model_path: str
    conditioning: Mapping[str, Any] = field(default_factory=dict)
    train_hyperparams: Mapping[str, Any] = field(default_factory=dict)
    generate_params: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


def load_style_config(path: str | Path) -> StyleConfig:
    """Load a style YAML and validate required fields."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Style config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Style config must be a mapping: {cfg_path}")

    style = raw.get("style") or {}
    model = raw.get("model") or {}
    train = raw.get("train") or {}
    generate = raw.get("generate") or {}

    style_id = style.get("id")
    base_model_id = model.get("base_model_id")
    if not style_id:
        raise ValueError(f"style.id is required in {cfg_path}")
    if not base_model_id:
        raise ValueError(f"model.base_model_id is required in {cfg_path}")

    return StyleConfig(
        style_id=str(style_id),
        name=str(style.get("name") or style_id),
        description=str(style.get("description") or "").strip(),
        prompt=str(style.get("prompt") or "").strip(),
        base_model_id=str(base_model_id),
        style_model_path=str(model.get("style_model_path") or f"models/styles/{style_id}"),
        conditioning=model.get("conditioning") or {},
        train_hyperparams=train,
        generate_params=generate,
        raw=raw,
    )
