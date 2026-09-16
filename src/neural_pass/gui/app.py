"""Application entry for the mask editor GUI."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="neural_pass mask editor")
    parser.add_argument(
        "--aov-dir",
        type=str,
        default=None,
        help="Folder containing AOV buffers (position/normal/depth/motion_vector)",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional mask image to load (PNG/TIFF/NPY)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Create QApplication, show MainWindow, run event loop. Returns exit code."""
    # Import Qt only when launching so `from neural_pass.gui.app import run` works
    # in headless / import-check environments without initializing a display.
    from PySide6.QtWidgets import QApplication

    from neural_pass.gui.main_window import MainWindow

    args = _parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(aov_dir=args.aov_dir, mask_path=args.mask)
    win.show()
    return app.exec()


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)
