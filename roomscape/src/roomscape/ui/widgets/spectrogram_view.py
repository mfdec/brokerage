"""Scrolling log-frequency spectrogram: frequency on x (aligned with the RTA), time running down."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from roomscape.core.units import format_freq
from roomscape.ui import theme
from roomscape.ui.widgets.plots import apply_linear_freq_axis, apply_log_freq_axis, style_plot

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
        self._scale = "log"
        self._hop = 0.05
        self.time_fmt = "{t:.1f} s ago"
        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse)

    def set_colormap(self, name: str) -> None:
        self.img.setLookupTable(colormap_lut(name))

    def set_levels(self, lo: float, hi: float) -> None:
        self.levels = (float(lo), float(hi))
        if self._data is not None:
            self.img.setLevels(self.levels)

    def set_time_label(self, label: str, fmt: str) -> None:
        self.pi.setLabel("left", label)
        self.time_fmt = fmt

    def set_data(self, freqs: np.ndarray, data: np.ndarray, hop_s: float, reset: bool = False, scale: str = "log") -> None:
        """``data``: (n_time, n_freq) dB, row 0 newest (or t = 0 for decay maps); ``scale``: "log" or "linear" frequency axis."""
        first = (reset or self._freqs is None or self._freqs.shape != freqs.shape or self._data.shape != data.shape
                 or scale != self._scale)
        self._freqs, self._data, self._hop, self._scale = freqs, data, hop_s, scale
        self.img.setImage(data, autoLevels=False, levels=self.levels)
        if first:
            if scale == "linear":
                x0, x1 = float(freqs[0]), float(freqs[-1])         # (the axis itself starts at 0 Hz, see below)
            else:
                x0, x1 = np.log10(freqs[0]), np.log10(freqs[-1])
            dx = (x1 - x0) / max(freqs.size - 1, 1)
            self.img.setRect(QtCore.QRectF(x0 - dx / 2, 0.0, x1 - x0 + dx, data.shape[0] * hop_s))
            if scale == "linear":
                apply_linear_freq_axis(self.pi, 0.0, float(freqs[-1]))
            else:
                apply_log_freq_axis(self.pi, freqs[0], freqs[-1])
            self.pi.setYRange(0.0, data.shape[0] * hop_s, padding=0)
            self.pi.setLimits(yMin=0.0, yMax=data.shape[0] * hop_s)

    def _on_mouse(self, evt) -> None:
        pos = evt[0]
        if self._freqs is None or not self.sceneBoundingRect().contains(pos):
            return
        mp = self.pi.vb.mapSceneToView(pos)
        linear = self._scale == "linear"
        f, t = (mp.x() if linear else 10.0 ** mp.x()), mp.y()
        row = int(t / self._hop)
        if not (0 <= row < self._data.shape[0]) or not (self._freqs[0] - 1e-9 <= f <= self._freqs[-1]):
            return
        col = int(np.argmin(np.abs(self._freqs - f))) if linear else int(np.argmin(np.abs(np.log(self._freqs / f))))
        label = f"{self._freqs[col]:.1f}" if linear and self._freqs[col] < 1000 else format_freq(self._freqs[col])
        self.pi.setTitle(f"{label} Hz   {self.time_fmt.format(t=t)}   {self._data[row, col]:.1f} dB",
                         size="10pt", color=theme.TEXT)
