#!/usr/bin/env python3
"""Write tiny synthetic AOV sample buffers for GUI demos (no Arnold required).

Outputs under data/aovs/_sample/ as .npy (float) and .png (8-bit preview).
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neural_pass.stages.aov_ingest import EXPECTED_AOVS


def _write_png_gray_or_rgb(path: Path, u8: np.ndarray) -> None:
    """Minimal PNG writer (no external deps). u8 is HxW or HxWx3."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(u8)
    if arr.ndim == 2:
        height, width = arr.shape
        color_type = 0  # grayscale
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(height))
    elif arr.ndim == 3 and arr.shape[2] == 3:
        height, width, _ = arr.shape
        color_type = 2  # RGB
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(height))
    else:
        raise ValueError(f"unsupported png shape {arr.shape}")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _to_u8_preview(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2:
        planes = [data]
    else:
        planes = [data[:, :, c] for c in range(data.shape[2])]

    u8_planes = []
    for plane in planes:
        finite = plane[np.isfinite(plane)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if hi <= lo:
            u8_planes.append(np.zeros(plane.shape, dtype=np.uint8))
        else:
            u8_planes.append((np.clip((plane - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))

    if len(u8_planes) == 1:
        return u8_planes[0]
    if len(u8_planes) == 2:
        z = np.zeros_like(u8_planes[0])
        return np.stack([u8_planes[0], u8_planes[1], z], axis=-1)
    return np.stack(u8_planes[:3], axis=-1)


def _write_png(path: Path, arr: np.ndarray) -> None:
    u8 = _to_u8_preview(arr)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), u8)
    except Exception:
        _write_png_gray_or_rgb(path, u8)


def make_samples(out_dir: Path, size: int = 64) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    fx = xx / max(size - 1, 1)
    fy = yy / max(size - 1, 1)

    position = np.stack([fx, fy, 0.25 + 0.5 * fx * fy], axis=-1)

    nx = fx * 2 - 1
    ny = fy * 2 - 1
    nz = np.ones_like(fx)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-6)
    normal = n

    depth = np.sqrt((fx - 0.5) ** 2 + (fy - 0.5) ** 2)
    motion_vector = np.stack([(fx - 0.5) * 0.1, (0.5 - fy) * 0.1], axis=-1)

    buffers = {
        "position": position,
        "normal": normal,
        "depth": depth,
        "motion_vector": motion_vector,
    }
    assert set(EXPECTED_AOVS) <= set(buffers.keys())

    for name, arr in buffers.items():
        npy_path = out_dir / f"{name}.npy"
        png_path = out_dir / f"{name}.png"
        np.save(str(npy_path), arr.astype(np.float32))
        _write_png(png_path, arr)
        print(f"  wrote {npy_path.name} {tuple(arr.shape)} and {png_path.name}")

    print(f"Sample AOVs ready under {out_dir}")


def main() -> int:
    out = _ROOT / "data" / "aovs" / "_sample"
    print(f"Generating sample AOVs → {out}")
    make_samples(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
