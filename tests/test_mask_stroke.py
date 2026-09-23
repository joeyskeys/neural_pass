"""Brush strokes stay continuous, and OpenEXR 3.x channels() still load."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_pass.gui.aov_loader import load_array  # noqa: E402
from neural_pass.gui.canvas import (  # noqa: E402
    MaskCanvas,
    _iter_stroke_points,
    _stroke_spacing,
)
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _segment_is_solid(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(math.ceil(dist)))
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if mask[y, x] < 0.5:
            return False
    return True


class _Mouse:
    def __init__(self, x: float, y: float, button=Qt.MouseButton.LeftButton):
        self._pos = QPointF(x, y)
        self._button = button

    def position(self):
        return self._pos

    def button(self):
        return self._button


class StrokePointTests(unittest.TestCase):
    def test_spacing_is_half_radius_and_at_least_one_pixel(self):
        self.assertEqual(_stroke_spacing(12), 6.0)
        self.assertEqual(_stroke_spacing(1), 1.0)
        self.assertLessEqual(_stroke_spacing(40), 0.5 * 40)
        self.assertGreaterEqual(_stroke_spacing(40), 0.25 * 40)

    def test_fast_segment_includes_endpoint_and_stays_within_spacing(self):
        spacing = 6.0
        pts = list(_iter_stroke_points(0, 0, 100, 40, spacing))
        self.assertEqual(pts[-1], (100, 40))
        self.assertNotIn((0, 0), pts)
        prev = (0, 0)
        for p in pts:
            gap = math.hypot(p[0] - prev[0], p[1] - prev[1])
            # Rounding to integer pixels can add a little over the float step.
            self.assertLessEqual(gap, spacing + 1.5)
            prev = p

    def test_slow_segment_is_a_single_dab(self):
        pts = list(_iter_stroke_points(10, 20, 12, 21, spacing=8.0))
        self.assertEqual(pts, [(12, 21)])

    def test_zero_length_segment_adds_no_dabs(self):
        self.assertEqual(list(_iter_stroke_points(4, 4, 4, 4, spacing=3.0)), [])


class MaskCanvasStrokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _canvas(self, h: int, w: int, radius: int = 8) -> MaskCanvas:
        canvas = MaskCanvas()
        canvas.set_brush_radius(radius)
        canvas.set_aov_channel(np.zeros((h, w), dtype=np.float32))
        canvas._label.setMinimumSize(w, h)
        canvas._label.resize(w, h)
        canvas.resize(w, h)
        canvas._refresh()
        pix = canvas._label.pixmap()
        self.assertIsNotNone(pix)
        self.assertFalse(pix.isNull())
        self.assertEqual((pix.width(), pix.height()), (w, h))
        self.assertEqual(canvas._label.width(), w)
        self.assertEqual(canvas._label.height(), h)
        return canvas

    def test_fast_drag_paints_a_continuous_stroke(self):
        h, w = 120, 360
        canvas = self._canvas(h, w, radius=8)
        y = 60
        canvas._on_press(_Mouse(12, y))
        # One mouse-move sample, far enough that endpoint dabs would not meet.
        canvas._on_move(_Mouse(340, y))
        canvas._on_release(_Mouse(340, y))
        mask = canvas.mask
        self.assertTrue(_segment_is_solid(mask, 12, y, 340, y))
        # Endpoint-only stamping would leave the midpoint untouched.
        self.assertGreater(abs(340 - 12), 2 * canvas.brush_radius)
        self.assertEqual(mask[y, (12 + 340) // 2], 1.0)
        self.assertEqual(mask[0, 180], 0.0)
        self.assertIsNone(canvas._last_paint_ij)
        self.assertFalse(canvas._painting)

    def test_diagonal_fast_drag_and_eraser_are_continuous(self):
        h, w = 160, 320
        canvas = self._canvas(h, w, radius=10)
        canvas._on_press(_Mouse(16, 20))
        canvas._on_move(_Mouse(300, 140))
        canvas._on_release(_Mouse(300, 140))
        self.assertTrue(_segment_is_solid(canvas.mask, 16, 20, 300, 140))

        canvas.set_mode("eraser")
        canvas._on_press(_Mouse(16, 20))
        canvas._on_move(_Mouse(300, 140))
        canvas._on_release(_Mouse(300, 140))
        # Eraser clears the same centerline.
        dist = math.hypot(300 - 16, 140 - 20)
        steps = int(math.ceil(dist))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(16 + (300 - 16) * t))
            y = int(round(20 + (140 - 20) * t))
            self.assertEqual(canvas.mask[y, x], 0.0)
        # Paint away from the stroke is untouched.
        self.assertEqual(canvas.mask[0, 0], 0.0)

    def test_right_button_erases_along_the_segment(self):
        h, w = 80, 200
        canvas = self._canvas(h, w, radius=6)
        canvas.mask[:] = 1.0
        y = 40
        canvas._on_press(_Mouse(10, y, Qt.MouseButton.RightButton))
        canvas._on_move(_Mouse(180, y, Qt.MouseButton.RightButton))
        canvas._on_release(_Mouse(180, y, Qt.MouseButton.RightButton))
        self.assertEqual(canvas.mask[y, 100], 0.0)
        self.assertEqual(canvas.mask[2, 2], 1.0)

    def test_slow_drag_does_not_overstamp(self):
        canvas = self._canvas(80, 200, radius=20)
        dabs = []
        orig = canvas._dab

        def _count(ix, iy):
            dabs.append((ix, iy))
            orig(ix, iy)

        canvas._dab = _count
        canvas._on_press(_Mouse(20, 40))
        # 1px steps: each sample is well under half the radius, so one dab each.
        for x in range(21, 51):
            canvas._on_move(_Mouse(x, 40))
        self.assertEqual(len(dabs), 1 + 30)

    def test_release_does_not_join_the_next_stroke(self):
        canvas = self._canvas(80, 220, radius=4)
        canvas._on_press(_Mouse(10, 40))
        canvas._on_release(_Mouse(10, 40))
        canvas._on_press(_Mouse(160, 40))
        canvas._on_move(_Mouse(180, 40))
        canvas._on_release(_Mouse(180, 40))
        self.assertEqual(canvas.mask[40, 80], 0.0)
        self.assertEqual(canvas.mask[40, 170], 1.0)

    def test_move_without_press_does_not_paint(self):
        canvas = self._canvas(40, 40, radius=5)
        self.assertFalse(canvas._on_move(_Mouse(20, 20)))
        self.assertEqual(canvas.mask.sum(), 0.0)


class OpenEXRChannelTests(unittest.TestCase):
    def test_openexr_3_channels_method(self):
        import OpenEXR

        h, w = 2, 3
        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
        }
        channels = {
            "R": np.full((h, w), 0.2, np.float32),
            "G": np.full((h, w), 0.4, np.float32),
            "B": np.full((h, w), 0.6, np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.exr"
            with OpenEXR.File(header, channels) as outfile:
                outfile.write(str(path))
            # Guard the hypothesis: 3.x exposes channels as a method.
            opened = OpenEXR.File(str(path))
            self.assertTrue(callable(opened.channels))
            arr = load_array(path)
        self.assertEqual(arr.shape, (h, w, 3))
        np.testing.assert_allclose(arr[0, 0], [0.2, 0.4, 0.6])

    def test_openexr_3_single_channel(self):
        import OpenEXR

        z = np.arange(6, dtype=np.float32).reshape(2, 3)
        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "z.exr"
            with OpenEXR.File(header, {"Z": z}) as outfile:
                outfile.write(str(path))
            arr = load_array(path)
        np.testing.assert_allclose(arr, z)

    def test_dict_channels_attribute_still_works(self):
        """Older bindings expose File.channels as a dict, not a method."""
        real = sys.modules.get("OpenEXR")
        fake = types.ModuleType("OpenEXR")
        pixels = np.arange(6, dtype=np.float32).reshape(2, 3)

        class _Pix:
            def __init__(self, data):
                self.pixels = data

        class _File:
            def __init__(self, path):
                self.channels = {"Z": _Pix(pixels)}

        fake.File = _File
        sys.modules["OpenEXR"] = fake
        try:
            arr = load_array(Path("not-a-real-file.exr"))
        finally:
            if real is None:
                sys.modules.pop("OpenEXR", None)
            else:
                sys.modules["OpenEXR"] = real
        np.testing.assert_allclose(arr, pixels)


if __name__ == "__main__":
    unittest.main()
