"""Painting canvas: AOV view + translucent mask overlay."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget, QVBoxLayout

from neural_pass.gui.aov_loader import normalize_for_display, overlay_mask_rgb


def _numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    h, w, c = rgb.shape
    assert c == 3
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    bytes_per_line = 3 * w
    # Copy so QImage owns stable bytes after the Python buffer may move.
    data = rgb.tobytes()
    img = QImage(data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return img.copy()


class MaskCanvas(QWidget):
    """Shows current AOV channel with mask overlay; LMB paint / RMB erase."""

    mask_changed = Signal()
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMinimumSize(256, 256)
        self._label.setStyleSheet("background-color: #1e1e1e; color: #aaa;")
        self._label.setText("Open an AOV folder to begin")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._base_channel: Optional[np.ndarray] = None  # float HxW
        self._base_rgb: Optional[np.ndarray] = None  # uint8 HxWx3
        self._mask: Optional[np.ndarray] = None  # float HxW [0,1]
        self._brush_radius = 12
        self._mode = "brush"  # or "eraser"
        self._painting = False
        self._paint_value = 1.0
        self._scale = 1.0

        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---- public API ----

    @property
    def brush_radius(self) -> int:
        return self._brush_radius

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("brush", "eraser"):
            raise ValueError(mode)
        self._mode = mode

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_aov_channel(self, channel: Optional[np.ndarray]) -> None:
        """Set the display AOV channel (float HxW). Resizes mask if needed."""
        if channel is None:
            self._base_channel = None
            self._base_rgb = None
            self._refresh()
            return
        ch = np.asarray(channel, dtype=np.float32)
        if ch.ndim != 2:
            raise ValueError("channel must be HxW")
        self._base_channel = ch
        self._base_rgb = normalize_for_display(ch)
        h, w = ch.shape
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.float32)
        self._refresh()

    def set_mask(self, mask: np.ndarray) -> None:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim != 2:
            raise ValueError("mask must be HxW")
        self._mask = np.clip(m, 0.0, 1.0)
        if self._base_channel is not None and self._mask.shape != self._base_channel.shape:
            # Resize mask to AOV via nearest (simple crop/pad)
            h, w = self._base_channel.shape
            new = np.zeros((h, w), dtype=np.float32)
            hh = min(h, m.shape[0])
            ww = min(w, m.shape[1])
            new[:hh, :ww] = self._mask[:hh, :ww]
            self._mask = new
            self.status_message.emit("Mask resized to match AOV resolution")
        self._refresh()
        self.mask_changed.emit()

    def clear_mask(self) -> None:
        if self._mask is not None:
            self._mask[...] = 0.0
            self._refresh()
            self.mask_changed.emit()

    def ensure_mask_shape(self, h: int, w: int) -> None:
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.float32)

    # ---- painting ----

    def eventFilter(self, obj, event):
        if obj is self._label:
            et = event.type()
            from PySide6.QtCore import QEvent

            if et == QEvent.Type.MouseButtonPress:
                return self._on_press(event)
            if et == QEvent.Type.MouseMove:
                return self._on_move(event)
            if et == QEvent.Type.MouseButtonRelease:
                return self._on_release(event)
        return super().eventFilter(obj, event)

    def _widget_to_image(self, pos) -> Optional[Tuple[int, int]]:
        if self._mask is None:
            return None
        pix = self._label.pixmap()
        if pix is None or pix.isNull():
            return None
        lw, lh = self._label.width(), self._label.height()
        pw, ph = pix.width(), pix.height()
        # Pixmap is centered in the label.
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        x = pos.x() - ox
        y = pos.y() - oy
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return None
        # Map from displayed pixmap size to image size.
        h, w = self._mask.shape
        ix = int(x * w / pw)
        iy = int(y * h / ph)
        ix = min(max(ix, 0), w - 1)
        iy = min(max(iy, 0), h - 1)
        return ix, iy

    def _stroke_value(self, button) -> Optional[float]:
        if button == Qt.MouseButton.LeftButton:
            return 1.0 if self._mode == "brush" else 0.0
        if button == Qt.MouseButton.RightButton:
            return 0.0  # RMB always erase
        return None

    def _on_press(self, event) -> bool:
        val = self._stroke_value(event.button())
        if val is None or self._mask is None:
            return False
        self._painting = True
        self._paint_value = val
        self._stamp(event.position().toPoint())
        return True

    def _on_move(self, event) -> bool:
        if not self._painting or self._mask is None:
            return False
        self._stamp(event.position().toPoint())
        return True

    def _on_release(self, event) -> bool:
        if self._painting:
            self._painting = False
            self.mask_changed.emit()
            return True
        return False

    def _stamp(self, pos) -> None:
        ij = self._widget_to_image(pos)
        if ij is None:
            return
        ix, iy = ij
        r = self._brush_radius
        h, w = self._mask.shape
        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - iy) ** 2 + (xx - ix) ** 2
        disk = dist2 <= r * r
        if self._paint_value >= 0.5:
            self._mask[y0:y1, x0:x1][disk] = 1.0
        else:
            self._mask[y0:y1, x0:x1][disk] = 0.0
        self._refresh()

    def _refresh(self) -> None:
        if self._base_rgb is None or self._mask is None:
            if self._base_rgb is None:
                self._label.setPixmap(QPixmap())
                self._label.setText("Open an AOV folder to begin")
            return
        composed = overlay_mask_rgb(self._base_rgb, self._mask)
        qimg = _numpy_rgb_to_qimage(composed)
        pix = QPixmap.fromImage(qimg)
        # Fit into label while keeping aspect.
        scaled = pix.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()
