"""Orchestrator for the neural post-render pass.

Wires stages in order:
  aov_ingest → conditioning → generate → edit_loop → flywheel

Training (`train_style`) is intentionally separate so generation and
fine-tuning stay clear. All stage bodies are stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from neural_pass.io.paths import ProjectPaths
from neural_pass.io.style_config import StyleConfig, load_style_config
from neural_pass.stages import aov_ingest, conditioning, edit_loop, flywheel, generate


@dataclass
class PassResult:
    """Placeholder result for a dry-run / stubbed pass."""

    style_id: str
    stages_run: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: str = "stubbed — no real model output"


def run_pass(
    *,
    style_config_path: str | Path,
    aov_dir: str | Path,
    mask_path: str | Path,
    project_root: str | Path | None = None,
    accept: bool = False,
    edit_path: str | Path | None = None,
) -> PassResult:
    """Run the neural pass pipeline (stubbed stage sequence).

    Prints / records stage order without performing real inference.
    """
    root = Path(project_root) if project_root else Path.cwd()
    paths = ProjectPaths(root)
    style: StyleConfig = load_style_config(style_config_path)

    print(f"[pipeline] style={style.style_id!r} from {style_config_path}")
    result = PassResult(style_id=style.style_id)
    ctx: MutableMapping[str, Any] = {
        "style": style,
        "paths": paths,
        "aov_dir": Path(aov_dir),
        "mask_path": Path(mask_path),
    }

    # 1) Load / validate Arnold AOVs
    print("[pipeline] stage: aov_ingest")
    aovs = aov_ingest.load_aovs(ctx["aov_dir"])
    ctx["aovs"] = aovs
    result.stages_run.append("aov_ingest")
    result.artifacts["aovs"] = aovs

    # 2) Pack AOVs + mask into conditioning bundle
    print("[pipeline] stage: conditioning")
    bundle = conditioning.build_conditioning(aovs=aovs, mask_path=ctx["mask_path"])
    ctx["conditioning"] = bundle
    result.stages_run.append("conditioning")
    result.artifacts["conditioning"] = bundle

    # 3) Style model base generation
    print("[pipeline] stage: generate")
    generation = generate.generate_image(style=style, conditioning=bundle, paths=paths)
    ctx["generation"] = generation
    result.stages_run.append("generate")
    result.artifacts["generation"] = generation

    # 4) User modification loop hooks
    print("[pipeline] stage: edit_loop")
    edit_state = edit_loop.run_edit_loop(
        generation=generation,
        edit_path=Path(edit_path) if edit_path else None,
        accept=accept,
    )
    ctx["edit"] = edit_state
    result.stages_run.append("edit_loop")
    result.artifacts["edit"] = edit_state

    # 5) Record accepted result as labeled flywheel sample (only if accepted)
    if edit_state.get("accepted"):
        print("[pipeline] stage: flywheel")
        sample = flywheel.record_accepted(
            style=style,
            generation=generation,
            edit=edit_state,
            conditioning=bundle,
            paths=paths,
        )
        result.stages_run.append("flywheel")
        result.artifacts["flywheel_sample"] = sample
    else:
        print("[pipeline] stage: flywheel (skipped — not accepted)")

    print(f"[pipeline] done. stages={result.stages_run}")
    return result


def main_stub() -> None:
    """Console-script placeholder; prefer scripts/run_pass.py."""
    raise SystemExit(
        "Use scripts/run_pass.py — this entry is a package metadata stub."
    )
