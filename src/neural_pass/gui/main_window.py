"""Main window for the neural_pass mask editor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QToolBar,
    QLabel,
    QSpinBox,
    QMessageBox,
    QWidget,
    QVBoxLayout,
)

from neural_pass.gui.aov_loader import AOVSet, load_aov_set, load_array
from neural_pass.gui.canvas import MaskCanvas

SHORTCUT_HELP = (
    "Tab: next AOV | Shift+Tab: prev AOV | C: next channel | "
    "[ / ]: brush size | B: brush | E: eraser | "
    "Ctrl+O: open AOV folder | Ctrl+S: save mask | "
    "LMB paint / RMB erase"
)


class MainWindow(QMainWindow):
    def __init__(self, aov_dir: Optional[str] = None, mask_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("neural_pass — Mask Editor")
        self.resize(1100, 800)

        self._aovs: Optional[AOVSet] = None
        self._aov_index = 0
        self._channel_index = 0
        self._mask_path: Optional[Path] = Path(mask_path) if mask_path else None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.canvas = MaskCanvas(self)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        self._view_label = QLabel("")
        self.statusBar().addPermanentWidget(self._view_label)
        self.statusBar().showMessage(SHORTCUT_HELP)

        self.canvas.status_message.connect(self.statusBar().showMessage)
        self.canvas.mask_changed.connect(self._on_mask_changed)

        self._build_menus()
        self._build_toolbar()
        self._bind_shortcuts()

        if aov_dir:
            self.load_aov_directory(Path(aov_dir))
        if mask_path:
            self.open_mask(Path(mask_path))

    # ---- UI construction ----

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_aov = QAction("Open AOV &Folder…", self)
        open_aov.setShortcut(QKeySequence.StandardKey.Open)
        open_aov.triggered.connect(self._dialog_open_aov)
        file_menu.addAction(open_aov)

        open_mask = QAction("Open &Mask…", self)
        open_mask.triggered.connect(self._dialog_open_mask)
        file_menu.addAction(open_mask)

        save_mask = QAction("&Save Mask", self)
        save_mask.setShortcut(QKeySequence.StandardKey.Save)
        save_mask.triggered.connect(self.save_mask)
        file_menu.addAction(save_mask)

        save_as = QAction("Save Mask &As…", self)
        save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as.triggered.connect(self._dialog_save_mask_as)
        file_menu.addAction(save_as)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = self.menuBar().addMenu("&Edit")
        clear = QAction("&Clear Mask", self)
        clear.triggered.connect(self.canvas.clear_mask)
        edit_menu.addAction(clear)

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts = QAction("&Keyboard Shortcuts", self)
        shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Tools", self)
        self.addToolBar(tb)

        brush = QAction("Brush (B)", self)
        brush.triggered.connect(lambda: self._set_mode("brush"))
        tb.addAction(brush)

        eraser = QAction("Eraser (E)", self)
        eraser.triggered.connect(lambda: self._set_mode("eraser"))
        tb.addAction(eraser)

        tb.addSeparator()
        tb.addWidget(QLabel(" Brush "))
        self._brush_spin = QSpinBox(self)
        self._brush_spin.setRange(1, 256)
        self._brush_spin.setValue(self.canvas.brush_radius)
        self._brush_spin.valueChanged.connect(self.canvas.set_brush_radius)
        tb.addWidget(self._brush_spin)

        tb.addSeparator()
        next_aov = QAction("Next AOV (Tab)", self)
        next_aov.triggered.connect(lambda: self.cycle_aov(1))
        tb.addAction(next_aov)
        prev_aov = QAction("Prev AOV (Shift+Tab)", self)
        prev_aov.triggered.connect(lambda: self.cycle_aov(-1))
        tb.addAction(prev_aov)
        next_ch = QAction("Next Channel (C)", self)
        next_ch.triggered.connect(self.cycle_channel)
        tb.addAction(next_ch)

    def _bind_shortcuts(self) -> None:
        # Tab / Shift+Tab — use Shortcuts so they work even when menus steal focus.
        QShortcut(QKeySequence(Qt.Key.Key_Tab), self, activated=lambda: self.cycle_aov(1))
        QShortcut(
            QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_Tab),
            self,
            activated=lambda: self.cycle_aov(-1),
        )
        # Backtab is what many platforms emit for Shift+Tab
        QShortcut(QKeySequence(Qt.Key.Key_Backtab), self, activated=lambda: self.cycle_aov(-1))

        QShortcut(QKeySequence(Qt.Key.Key_C), self, activated=self.cycle_channel)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self, activated=lambda: self._nudge_brush(-2))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self, activated=lambda: self._nudge_brush(2))
        QShortcut(QKeySequence(Qt.Key.Key_B), self, activated=lambda: self._set_mode("brush"))
        QShortcut(QKeySequence(Qt.Key.Key_E), self, activated=lambda: self._set_mode("eraser"))

    # ---- actions ----

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "\n".join(
                [
                    "Tab — next AOV buffer",
                    "Shift+Tab — previous AOV buffer",
                    "C — next channel within current AOV",
                    "[ / ] — decrease / increase brush size",
                    "B — brush mode",
                    "E — eraser mode",
                    "Ctrl+O — open AOV folder",
                    "Ctrl+S — save mask",
                    "LMB — paint in current mode",
                    "RMB — erase",
                ]
            ),
        )

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.statusBar().showMessage(f"Mode: {mode}  |  {SHORTCUT_HELP}", 4000)

    def _nudge_brush(self, delta: int) -> None:
        val = max(1, min(256, self.canvas.brush_radius + delta))
        self._brush_spin.setValue(val)
        self.statusBar().showMessage(f"Brush size: {val}", 2000)

    def _dialog_open_aov(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open AOV Folder")
        if path:
            self.load_aov_directory(Path(path))

    def _dialog_open_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask", "", "Images (*.png *.tif *.tiff *.npy);;All (*.*)"
        )
        if path:
            self.open_mask(Path(path))

    def _dialog_save_mask_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", str(self._mask_path or "mask.png"), "PNG (*.png)"
        )
        if path:
            self._mask_path = Path(path)
            self.save_mask()

    def load_aov_directory(self, directory: Path) -> None:
        aovs = load_aov_set(directory)
        self._aovs = aovs
        self._aov_index = 0
        self._channel_index = 0

        msgs = []
        if aovs.missing:
            msgs.append("Missing AOVs: " + ", ".join(aovs.missing))
        msgs.extend(aovs.warnings)
        if not aovs.arrays:
            self.canvas.set_aov_channel(None)
            self.statusBar().showMessage(
                "No AOVs loaded. " + (" | ".join(msgs) if msgs else ""), 8000
            )
            self._view_label.setText("")
            return

        shape = aovs.shape_hw()
        if shape:
            self.canvas.ensure_mask_shape(*shape)
        self._refresh_view()
        warn = " | ".join(msgs) if msgs else "OK"
        self.statusBar().showMessage(
            f"Loaded {len(aovs.arrays)} AOV(s) from {directory} — {warn}", 8000
        )

    def open_mask(self, path: Path) -> None:
        try:
            arr = load_array(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open Mask", f"Failed to load mask:\n{exc}")
            return
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        # If loaded as 0–1 float from uint8 path already scaled; clamp.
        if arr.max() > 1.5:
            arr = arr / 255.0
        self.canvas.set_mask(arr)
        self._mask_path = path
        self.statusBar().showMessage(f"Opened mask {path}", 4000)

    def save_mask(self) -> None:
        mask = self.canvas.mask
        if mask is None:
            QMessageBox.information(self, "Save Mask", "No mask to save yet.")
            return
        if self._mask_path is None:
            self._dialog_save_mask_as()
            return
        path = self._mask_path
        path.parent.mkdir(parents=True, exist_ok=True)
        u8 = (np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        try:
            import imageio.v2 as imageio

            imageio.imwrite(str(path), u8)
        except Exception:
            # Minimal PNG writer via Qt if imageio missing at save time.
            from PySide6.QtGui import QImage

            h, w = u8.shape
            img = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
            if not img.save(str(path)):
                QMessageBox.warning(self, "Save Mask", f"Failed to save {path}")
                return
        self.statusBar().showMessage(f"Saved mask → {path}", 5000)

    def cycle_aov(self, delta: int) -> None:
        if not self._aovs or not self._aovs.names:
            return
        n = len(self._aovs.names)
        self._aov_index = (self._aov_index + delta) % n
        self._channel_index = 0
        self._refresh_view()

    def cycle_channel(self) -> None:
        if not self._aovs or not self._aovs.names:
            return
        name = self._aovs.names[self._aov_index]
        nch = self._aovs.channel_count(name)
        self._channel_index = (self._channel_index + 1) % nch
        self._refresh_view()

    def _refresh_view(self) -> None:
        if not self._aovs or not self._aovs.names:
            return
        name = self._aovs.names[self._aov_index]
        nch = self._aovs.channel_count(name)
        self._channel_index = self._channel_index % nch
        channel = self._aovs.get_channel(name, self._channel_index)
        self.canvas.set_aov_channel(channel)
        src = self._aovs.paths.get(name)
        src_s = src.name if src else "?"
        self._view_label.setText(
            f"  AOV: {name}  ch {self._channel_index}/{nch - 1}  ({src_s})  "
            f"mode={self.canvas.mode}  brush={self.canvas.brush_radius}  "
        )

    def _on_mask_changed(self) -> None:
        # Keep status shortcuts visible; light touch.
        pass
