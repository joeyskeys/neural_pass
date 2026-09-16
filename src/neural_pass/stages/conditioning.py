"""Pack Arnold AOVs + user mask into a model conditioning bundle.

Vision: condition tensors (or an equivalent multi-channel stack) that the
style generative model consumes alongside any text/style prompt from config.

STUB: returns a placeholder bundle; does not allocate real tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def build_conditioning(
    *,
    aovs: Mapping[str, Any],
    mask_path: str | Path,
) -> dict[str, Any]:
    """Build conditioning bundle from AOVs + mask (stub)."""
    mask = Path(mask_path)
    print(f"  [conditioning] packing AOVs + mask={mask} (stub)")

    # TODO: resize/align AOVs to mask, stack channels, normalize, device placement.
    return {
        "aov_summary": {
            "found": list(aovs.get("found", [])),
            "missing": list(aovs.get("missing", [])),
        },
        "mask_path": str(mask),
        "mask_exists": mask.is_file(),
        "tensor": None,  # placeholder — no fake tensor invented
        "status": "stub",
        "note": "Conditioning tensor not built — implement packing later.",
    }
