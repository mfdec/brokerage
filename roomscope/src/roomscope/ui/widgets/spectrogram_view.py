"""Scrolling log-frequency spectrogram: frequency on x (aligned with the RTA), time running down."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from roomscope.core.units import format_freq
from roomscope.ui import theme
from roomscope.ui.widgets.plots import apply_log_freq_axis, style_plot

COLORMAPS = ["inferno", "viridis", "magma", "turbo", "plasma"]


def colormap_lut(name: str, n: int = 256) -> np.ndarray:
    return pg.colormap.get(name).getLookupTable(0.0, 1.0, n)


class SpectrogramView(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="seconds ago")
        self.pi.setMenuEnabled(False)
        self.pi.vb.setMouseEnabled(x=True, y=False)
        self.pi.invertY(True)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.pi.addItem(self.img)
        self.pi.showGrid(x=True, y=False, alpha=0.25)
        self.levels = (-100.0, 0.0)
        self.set_colormap("inferno")
        self._freqs: np.ndarray | None = None
        self._data: np.ndarray | None = None
        self._hop = 0.05
        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse)

    def set_colormap(self, name: str) -> None:
        self.img.setLookupTable(colormap_lut(name))

    def set_levels(self, lo: float, hi: float) -> None:
        self.levels = (float(lo), float(hi))
        if self._data is not None:
            self.img.setLevels(self.levels)

    def set_data(self, freqs: np.ndarray, data: np.ndarray, hop_s: float) -> None:
        """``data``: (n_time, n_freq) dB, row 0 newest."""
        first = self._freqs is None or self._freqs.shape != freqs.shape or self._data.shape != data.shape
        self._freqs, self._data, self._hop = freqs, data, hop_s
        self.img.setImage(data, autoLevels=False, levels=self.levels)
        if first:
            x0, x1 = np.log10(freqs[0]), np.log10(freqs[-1])
            dx = (x1 - x0) / max(freqs.size - 1, 1)
            self.img.setRect(QtCore.QRectF(x0 - dx / 2, 0.0, x1 - x0 + dx, data.shape[0] * hop_s))
            apply_log_freq_axis(self.pi, freqs[0], freqs[-1])
            self.pi.setYRange(0.0, data.shape[0] * hop_s, padding=0)
            self.pi.setLimits(yMin=0.0, yMax=data.shape[0] * hop_s)

    def _on_mouse(self, evt) -> None:
        pos = evt[0]
        if self._freqs is None or not self.sceneBoundingRect().contains(pos):
            return
        mp = self.pi.vb.mapSceneToView(pos)
        f, t = 10.0 ** mp.x(), mp.y()
        row = int(t / self._hop)
        if not (0 <= row < self._data.shape[0]) or not (self._freqs[0] <= f <= self._freqs[-1]):
            return
        col = int(np.argmin(np.abs(np.log(self._freqs / f))))
        self.pi.setTitle(f"{format_freq(self._freqs[col])} Hz   {t:.1f} s ago   {self._data[row, col]:.1f} dB",
                         size="10pt", color=theme.TEXT)
