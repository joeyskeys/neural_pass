#!/usr/bin/env python3
"""CLI entry for style post-training / fine-tuning (stub).

Example:
  python scripts/train_style.py --style configs/styles/example_style.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neural_pass.stages.train_style import train_style  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="neural_pass style post-training (stub)",
    )
    p.add_argument(
        "--style",
        required=True,
        help="Path to style YAML",
    )
    p.add_argument(
        "--project-root",
        default=str(_REPO_ROOT),
        help="Project root (default: repo root)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== neural_pass train_style (stub) ===")
    result = train_style(
        style_config_path=args.style,
        project_root=args.project_root,
    )
    print("--- summary ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
