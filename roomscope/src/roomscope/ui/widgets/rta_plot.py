"""Live RTA plot: instantaneous / averaged / peak curves or fractional-octave bars, with hover read-out."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from roomscope.core.spectrum import RtaFrame
from roomscope.core.units import format_freq
from roomscope.ui import theme
from roomscope.ui.widgets.plots import apply_log_freq_axis, style_plot


class RtaPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="dBFS", bottom="Frequency (Hz)")
        self.pi.setTitle(" ", size="10pt", color=theme.MUTED)
        self.f_min, self.f_max = 20.0, 20000.0
        apply_log_freq_axis(self.pi, self.f_min, self.f_max)
        self.pi.setMenuEnabled(False)
        self.pi.vb.setMouseEnabled(x=True, y=False)

        self.inst = self.pi.plot(pen=pg.mkPen((76, 194, 255, 80), width=1))
        self.avg = self.pi.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.peak = self.pi.plot(pen=pg.mkPen(theme.AMBER, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.bars = pg.BarGraphItem(x=[0], height=[0], width=0.1, y0=-300, brush=pg.mkBrush(76, 194, 255, 150),
                                    pen=pg.mkPen(theme.ACCENT))
        self.pi.addItem(self.bars)
        self.bars.setVisible(False)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.PenStyle.DotLine))
        self.pi.addItem(self.cursor, ignoreBounds=True)
        self.cursor.setVisible(False)

        self.show_inst = True
        self.show_peak = False
        self._last: RtaFrame | None = None
        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse)

    # ------------------------------------------------------------------ #
    def set_range(self, top: float, span: float) -> None:
        self.pi.setYRange(top - span, top, padding=0)

    def set_unit(self, unit: str) -> None:
        self.pi.setLabel("left", unit)

    def set_frame(self, fr: RtaFrame) -> None:
        self._last = fr
        x = np.log10(fr.freqs)
        if fr.kind == "bars":
            frac = 1 if len(fr.freqs) < 14 else 3
            w = np.log10(2.0 ** (1.0 / frac)) * 0.86
            base = self.pi.viewRange()[1][0]
            self.bars.setOpts(x=x, height=np.maximum(fr.avg_db - base, 0.0), width=w, y0=base)
            self.bars.setVisible(True)
            self.inst.setData([], [])
            self.avg.setData([], [])
            self.peak.setData(x, fr.peak_db) if self.show_peak else self.peak.setData([], [])
        else:
            self.bars.setVisible(False)
            self.inst.setData(x, fr.inst_db) if self.show_inst else self.inst.setData([], [])
            self.avg.setData(x, fr.avg_db)
            self.peak.setData(x, fr.peak_db) if self.show_peak else self.peak.setData([], [])

    # ------------------------------------------------------------------ #
    def _on_mouse(self, evt) -> None:
        pos = evt[0]
        if self._last is None or not self.sceneBoundingRect().contains(pos):
            self.cursor.setVisible(False)
            return
        mp = self.pi.vb.mapSceneToView(pos)
        f = 10.0 ** mp.x()
        fr = self._last
        if f < fr.freqs[0] or f > fr.freqs[-1]:
            self.cursor.setVisible(False)
            return
        i = int(np.argmin(np.abs(np.log(fr.freqs / f))))
        self.cursor.setPos(np.log10(fr.freqs[i]))
        self.cursor.setVisible(True)
        self.pi.setTitle(f"{format_freq(fr.freqs[i])} Hz   {fr.avg_db[i]:.1f} {fr.unit}", size="10pt", color=theme.TEXT)
