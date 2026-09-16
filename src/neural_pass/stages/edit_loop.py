"""User modification loop hooks.

After base generation, the user iteratively edits until satisfied.
Hooks: load an edited image, compare to generation, mark accept/reject.

STUB: no UI; optional --edit path and --accept flag drive placeholder state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def run_edit_loop(
    *,
    generation: Mapping[str, Any],
    edit_path: Optional[Path] = None,
    accept: bool = False,
) -> dict[str, Any]:
    """Run edit-loop hooks (stub).

    If `edit_path` is given, records that an edit was provided (without
    loading pixels). `accept=True` marks the result accepted for flywheel.
    """
    print(
        f"  [edit_loop] generation_status={generation.get('status')} "
        f"edit_path={edit_path} accept={accept} (stub)"
    )

    edit_exists = bool(edit_path and Path(edit_path).is_file())
    # TODO: load edit, compute optional preview/diff, interactive session.
    return {
        "edit_path": str(edit_path) if edit_path else None,
        "edit_loaded": edit_exists,
        "accepted": bool(accept),
        "compared": False,
        "status": "stub",
        "note": "Edit loop is hooks-only — no UI or pixel compare yet.",
    }


def load_edit(edit_path: str | Path) -> dict[str, Any]:
    """Load a user-modified image (stub)."""
    path = Path(edit_path)
    print(f"  [edit_loop] load_edit {path} (stub)")
    # TODO: decode image; raise if missing.
    if not path.is_file():
        raise FileNotFoundError(f"Edit file not found: {path}")
    raise NotImplementedError("TODO: implement edit image loading")


def compare_to_generation(
    generation: Mapping[str, Any],
    edit: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare edit vs generation (stub)."""
    raise NotImplementedError("TODO: implement edit vs generation compare")
