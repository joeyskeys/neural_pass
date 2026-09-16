"""Mask-painting GUI for neural_pass AOVs."""

from __future__ import annotations

__all__ = ["run"]


def run(argv=None):
    """Launch the mask editor (lazy import so headless import-checks stay light)."""
    from neural_pass.gui.app import run as _run

    return _run(argv)
