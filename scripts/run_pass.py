#!/usr/bin/env python3
"""CLI entry: run the neural pass pipeline (stubs) and print stage flow.

Example:
  python scripts/run_pass.py \\
    --style configs/styles/example_style.yaml \\
    --aov-dir data/aovs \\
    --mask data/masks/example_mask.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add src/ to path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neural_pass.pipeline import run_pass  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="neural_pass dry-run / stub pipeline after Arnold",
    )
    p.add_argument(
        "--style",
        required=True,
        help="Path to style YAML (e.g. configs/styles/example_style.yaml)",
    )
    p.add_argument(
        "--aov-dir",
        required=True,
        help="Directory of Arnold condition AOVs",
    )
    p.add_argument(
        "--mask",
        required=True,
        help="Path to user-provided mask",
    )
    p.add_argument(
        "--project-root",
        default=str(_REPO_ROOT),
        help="Project root (default: repo root)",
    )
    p.add_argument(
        "--edit",
        default=None,
        help="Optional path to a user-edited image (edit_loop hook)",
    )
    p.add_argument(
        "--accept",
        action="store_true",
        help="Mark result accepted and invoke flywheel stub",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== neural_pass run_pass (stub dry-run) ===")
    print("Stage sequence: aov_ingest \u2192 conditioning \u2192 generate \u2192 edit_loop \u2192 [flywheel]")
    result = run_pass(
        style_config_path=args.style,
        aov_dir=args.aov_dir,
        mask_path=args.mask,
        project_root=args.project_root,
        accept=args.accept,
        edit_path=args.edit,
    )
    print("--- summary ---")
    print(f"style_id: {result.style_id}")
    print(f"stages_run: {result.stages_run}")
    print(f"notes: {result.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
