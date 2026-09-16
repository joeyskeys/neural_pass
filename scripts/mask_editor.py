#!/usr/bin/env python3
"""Launch the neural_pass mask-painting GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without install: PYTHONPATH=src or this fallback.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neural_pass.gui.app import run


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
