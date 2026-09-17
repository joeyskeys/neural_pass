"""Discover and load AOV buffers for the mask editor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neural_pass.stages.aov_ingest import EXPECTED_AOVS

SUPPORTED_EXTENSIONS = (".exr", ".tif", ".tiff", ".png", ".npy", ".hdr")


@dataclass
class AOVSet:
    """Loaded AOV float arrays keyed by logical name (HxW or HxWxC)."""

    directory: Path
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    paths: Dict[str, Path] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        # Prefer EXPECTED_AOVS order, then any extras.
        ordered = [n for n in EXPECTED_AOVS if n in self.arrays]
        extras = sorted(n for n in self.arrays if n not in EXPECTED_AOVS)
        return ordered + extras

    def shape_hw(self) -> Optional[Tuple[int, int]]:
        for arr in self.arrays.values():
            return int(arr.shape[0]), int(arr.shape[1])
        return None

    def channel_count(self, name: str) -> int:
        arr = self.arrays[name]
        if arr.ndim == 2:
            return 1
        return int(arr.shape[2])

    def get_channel(self, name: str, channel: int) -> np.ndarray:
        """Return a 2-D float view/copy of one channel (does not mutate source)."""
        arr = self.arrays[name]
        if arr.ndim == 2:
            if channel != 0:
                raise IndexError(f"{name} is single-channel; got channel={channel}")
            return np.asarray(arr, dtype=np.float32)
        if channel < 0 or channel >= arr.shape[2]:
            raise IndexError(f"{name} has {arr.shape[2]} channels; got channel={channel}")
        return np.asarray(arr[:, :, channel], dtype=np.float32)


def discover_aov_files(aov_dir: Path) -> Tuple[Dict[str, Path], List[str], List[str]]:
    """Map EXPECTED_AOVS (+ extras) to files via stem prefix matching.

    Returns (mapping, missing_expected, warnings).
    """
    aov_dir = Path(aov_dir)
    mapping: Dict[str, Path] = {}
    warnings: List[str] = []

    if not aov_dir.is_dir():
        return {}, list(EXPECTED_AOVS), [f"AOV directory does not exist: {aov_dir}"]

    files = [
        p
        for p in sorted(aov_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    stems = [(p, p.stem.lower()) for p in files]

    claimed: set = set()
    for name in EXPECTED_AOVS:
        matches = [
            p
            for p, stem in stems
            if p not in claimed and (stem == name or stem.startswith(name))
        ]
        if not matches:
            continue
        # Prefer exact stem, then shortest stem, then name order.
        ext_rank = {
            ".npy": 0,
            ".exr": 1,
            ".tif": 2,
            ".tiff": 2,
            ".hdr": 3,
            ".png": 4,
        }
        matches.sort(
            key=lambda p: (
                0 if p.stem.lower() == name else 1,
                ext_rank.get(p.suffix.lower(), 9),
                len(p.stem),
                p.name,
            )
        )
        chosen = matches[0]
        mapping[name] = chosen
        claimed.add(chosen)
        if len(matches) > 1:
            warnings.append(
                f"Multiple files for '{name}'; using {chosen.name} "
                f"(also: {', '.join(m.name for m in matches[1:])})"
            )

    missing = [n for n in EXPECTED_AOVS if n not in mapping]

    # Extras: unmatched files whose stem is not covered.
    for p, stem in stems:
        if p in claimed:
            continue
        # Skip if already matched as prefix of an expected name.
        logical = stem
        for name in EXPECTED_AOVS:
            if stem == name or stem.startswith(name):
                logical = name
                break
        if logical in mapping:
            continue
        mapping[logical] = p

    return mapping, missing, warnings


def _load_openexr(path: Path) -> np.ndarray:
    """Load EXR via OpenEXR bindings (classic or OpenEXR 3.x File API)."""
    import OpenEXR  # type: ignore

    # OpenEXR 3.x / modern bindings: File + channels as arrays
    if hasattr(OpenEXR, "File"):
        f = OpenEXR.File(str(path))
        header = f.header() if callable(getattr(f, "header", None)) else getattr(f, "header", {})
        channels = getattr(f, "channels", None)
        if channels is None and hasattr(f, "parts"):
            # Multi-part: take first part channels
            parts = f.parts
            channels = parts[0].channels if parts else {}
        if not channels:
            raise RuntimeError("OpenEXR.File opened but no channels found")
        names = list(channels.keys())
        preferred = ["R", "G", "B", "A", "X", "Y", "Z", "U", "V"]
        ordered = [c for c in preferred if c in channels] + [c for c in names if c not in preferred]
        planes = []
        for c in ordered:
            pix = channels[c]
            arr = np.asarray(getattr(pix, "pixels", pix), dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            planes.append(arr)
        if len(planes) == 1:
            return planes[0]
        return np.stack(planes, axis=-1)

    # Classic InputFile + Imath API
    import Imath  # type: ignore

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = list(header["channels"].keys())
    preferred = ["R", "G", "B", "A", "X", "Y", "Z", "U", "V"]
    ordered = [c for c in preferred if c in channels]
    ordered += [c for c in channels if c not in ordered]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    planes = []
    for c in ordered:
        raw = exr.channel(c, pt)
        planes.append(np.frombuffer(raw, dtype=np.float32).reshape(height, width))
    if len(planes) == 1:
        return planes[0]
    return np.stack(planes, axis=-1)


def _load_imageio(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    arr = imageio.imread(str(path))
    return np.asarray(arr)


def load_array(path: Path) -> np.ndarray:
    """Load an image / EXR / npy file as float32 HxW or HxWxC."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(path))
    elif suffix == ".exr":
        try:
            arr = _load_openexr(path)
        except Exception:
            try:
                arr = _load_imageio(path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load EXR {path}. Install OpenEXR or imageio with "
                    f"FreeImage/OpenEXR support. Underlying error: {exc}"
                ) from exc
    else:
        try:
            arr = _load_imageio(path)
        except Exception as exc:
            # Fallback: Pillow via imageio may fail; try numpy for .png rarely.
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc

    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def load_aov_set(aov_dir: str | Path) -> AOVSet:
    """Discover and load AOVs from a folder."""
    directory = Path(aov_dir)
    mapping, missing, warnings = discover_aov_files(directory)
    arrays: Dict[str, np.ndarray] = {}
    paths: Dict[str, Path] = {}
    load_warnings = list(warnings)

    shapes: List[Tuple[int, int]] = []
    for name, path in mapping.items():
        try:
            arr = load_array(path)
            arrays[name] = arr
            paths[name] = path
            shapes.append((int(arr.shape[0]), int(arr.shape[1])))
        except Exception as exc:
            load_warnings.append(f"Could not load {name} from {path.name}: {exc}")
            if name in EXPECTED_AOVS and name not in missing:
                missing.append(name)

    if shapes and len(set(shapes)) > 1:
        load_warnings.append(
            "AOV resolutions differ: " + ", ".join(f"{n}={arrays[n].shape[:2]}" for n in arrays)
        )

    return AOVSet(
        directory=directory,
        arrays=arrays,
        paths=paths,
        missing=missing,
        warnings=load_warnings,
    )


def normalize_for_display(
    channel: np.ndarray,
    *,
    percentile: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Display-only normalize a 2-D float channel to uint8 RGB grayscale.

    Does not modify the input array.
    """
    data = np.asarray(channel, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        out = np.zeros(data.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, percentile)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            scaled = np.zeros_like(data, dtype=np.float32)
        else:
            scaled = (data - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~np.isfinite(data)] = 0.0
        out = (scaled * 255.0 + 0.5).astype(np.uint8)
    # HxWx3 for QImage Format_RGB888
    return np.stack([out, out, out], axis=-1)


def overlay_mask_rgb(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    """Composite translucent mask over an RGB uint8 image (display only)."""
    base = np.asarray(base_rgb, dtype=np.float32)
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if m.shape[:2] != base.shape[:2]:
        raise ValueError(f"mask shape {m.shape} != base {base.shape[:2]}")
    col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    a = (m * alpha)[..., None]
    out = base * (1.0 - a) + col * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
