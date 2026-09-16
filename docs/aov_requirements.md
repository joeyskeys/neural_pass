# AOV requirements

Condition buffers expected by `neural_pass` (see `stages.aov_ingest.EXPECTED_AOVS`).

| Logical name     | Typical channels | Notes |
|------------------|------------------|-------|
| `position`       | XYZ (3)          | World- or camera-space position |
| `normal`         | XYZ (3)          | Shading / geometric normals |
| `depth`          | Z (1)            | Camera depth / distance |
| `motion_vector`  | XY or XYZ (2–3)  | Screen- or world-space motion |

## File discovery

The mask editor and ingest stage match files by **stem prefix**:

- Exact stem: `position.exr`, `normal.png`
- Prefixed stem: `position.beauty.exr`, `normal_001.tif`

Supported extensions for the GUI loader: `.exr`, `.tif`, `.tiff`, `.png`, `.npy`.

Missing AOVs are allowed: the GUI loads whatever is present and warns in the status bar.

## Sample data

Generate tiny demo buffers (no Arnold required):

```bash
python scripts/make_sample_aovs.py
```

Outputs under `data/aovs/_sample/`.
