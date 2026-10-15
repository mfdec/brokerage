"""Plot widgets for a room measurement: RT60 bars, Schroeder decay, frequency response, impulse response."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from roomscape.core.decay import BandDecay, DecayResult
from roomscape.core.measurement import FrequencyResponse, MeasurementResult
from roomscape.core.units import format_freq, nominal_frequency
from roomscape.ui import theme
from roomscape.ui.widgets.plots import apply_linear_freq_axis, apply_log_freq_axis, style_plot

DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine


def band_label(b: BandDecay, third: bool) -> str:
    return format_freq(nominal_frequency(b.fc) if third else _octave_nominal(b.fc))


def _octave_nominal(fc: float) -> float:
    for nominal in (2, 4, 8, 16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
        if abs(np.log2(fc / nominal)) < 0.12:
            return nominal
    return fc


class Rt60Plot(pg.PlotWidget):
    """RT60 per band: bars = best estimate, diamonds = EDT; faded bars are 'indicative' only."""

    bandSelected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="seconds", bottom="Band centre (Hz)")
        self.pi.setMenuEnabled(False)
        self.pi.vb.setMouseEnabled(x=False, y=False)
        self.pi.setTitle("RT60", size="10pt", color=theme.MUTED)
        self.target = pg.LinearRegionItem(values=(0.2, 0.5), orientation="horizontal", movable=False,
                                          brush=pg.mkBrush(95, 211, 141, 28), pen=pg.mkPen(95, 211, 141, 90))
        self.pi.addItem(self.target)
        self.bars = pg.BarGraphItem(x=[0], height=[0], width=0.7, brush=pg.mkBrush(theme.ACCENT))
        self.pi.addItem(self.bars)
        self.edt = pg.ScatterPlotItem(symbol="d", size=9, brush=pg.mkBrush(theme.AMBER), pen=pg.mkPen("#1a1a1a"))
        self.pi.addItem(self.edt)
        self.sel = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.TEXT, style=DOT))
        self.pi.addItem(self.sel)
        self.pi.getAxis("left").setWidth(54)
        self._n = 0
        self.proxy = pg.SignalProxy(self.scene().sigMouseClicked, rateLimit=20, slot=self._clicked)

    def set_target(self, lo: float, hi: float) -> None:
        self.target.setRegion((lo, hi))

    def set_result(self, res: DecayResult, selected: int = 0) -> None:
        bands = res.bands
        n = len(bands)
        self._n = n
        t60 = np.array([b.t60 for b in bands])
        edt = np.array([b.edt for b in bands])
        ok = np.array([b.reliable for b in bands])
        x = np.arange(n, dtype=float)
        h = np.nan_to_num(t60, nan=0.0)
        brushes = [pg.mkBrush(76, 194, 255, 220) if r else pg.mkBrush(255, 180, 84, 120) for r in ok]
        self.bars.setOpts(x=x, height=h, width=0.7, brushes=brushes)
        good = np.isfinite(edt)
        self.edt.setData(x[good], edt[good])
        third = res.bands_per_octave == 3
        self.pi.getAxis("bottom").setTicks([[(i, band_label(b, third)) for i, b in enumerate(bands)]])
        top = max(1.0, float(np.nanmax(np.concatenate((h, edt[good])))) * 1.25) if n else 1.0
        self.pi.setXRange(-0.7, n - 0.3, padding=0)
        self.pi.setYRange(0, top, padding=0)
        self.sel.setPos(selected)

    def _clicked(self, evt) -> None:
        ev = evt[0]
        if self._n == 0 or not self.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = self.pi.vb.mapSceneToView(ev.scenePos()).x()
        i = int(np.clip(round(x), 0, self._n - 1))
        self.sel.setPos(i)
        self.bandSelected.emit(i)


class SchroederPlot(pg.PlotWidget):
    """Backward-integrated decay of one band with the fitted RT60 line and the noise-truncation point."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="dB", bottom="Time after direct sound (s)")
        self.pi.setMenuEnabled(False)
        self.pi.setTitle(" ", size="10pt", color=theme.MUTED)
        self.pi.setYRange(-80, 3, padding=0)
        self.pi.getAxis("left").setWidth(54)
        self.curve = self.pi.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.fit = self.pi.plot(pen=pg.mkPen(theme.AMBER, width=1.5, style=DASH))
        self.cut = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.MUTED, style=DOT))
        self.pi.addItem(self.cut)

    def set_band(self, b: BandDecay, fs: int, title: str = "") -> None:
        if b.curve_db is None:
            self.curve.setData([], [])
            self.fit.setData([], [])
            return
        c = b.curve_db
        step = max(1, c.size // 3000)
        t = np.arange(0, c.size, step) / fs
        self.curve.setData(t, c[::step])
        if np.isfinite(b.t60):
            self.fit.setData([0.0, b.t60], [0.0, -60.0])
        else:
            self.fit.setData([], [])
        if b.trunc is not None:
            self.cut.setPos(b.trunc.index / fs)
        txt = f"{title}  EDT {b.edt:.2f}  T20 {b.t20:.2f}  T30 {b.t30:.2f}  s" if title else ""
        self.pi.setTitle(txt, size="10pt", color=theme.TEXT)
        self.pi.setXRange(0, max(b.t60 * 1.6 if np.isfinite(b.t60) else 0.5, 0.3), padding=0.0)


class ResponsePlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="dBFS", bottom="Frequency (Hz)")
        self.pi.setMenuEnabled(False)
        self.pi.vb.setMouseEnabled(x=True, y=True)
        apply_log_freq_axis(self.pi, 20, 20000)
        self.pi.getAxis("left").setWidth(54)
        self.curve = self.pi.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self._fr: FrequencyResponse | None = None
        self._axis_key = ("log", 20.0, 20000.0)
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._mouse)

    def set_response(self, fr: FrequencyResponse, autorange: bool = True) -> None:
        self._fr = fr
        self.pi.setLabel("left", fr.unit)
        linear = fr.scale == "linear"
        key = (fr.scale, 0.0 if linear else float(fr.freqs[0]), float(fr.freqs[-1]))
        if key != self._axis_key:                                       # the axis changed (log <-> linear, or its limits)
            self._axis_key = key
            if linear:
                apply_linear_freq_axis(self.pi, 0.0, float(fr.freqs[-1]))
            else:
                apply_log_freq_axis(self.pi, float(fr.freqs[0]), float(fr.freqs[-1]))
        self.curve.setData(fr.freqs if linear else np.log10(np.maximum(fr.freqs, 1e-3)), fr.level_db)
        if autorange:
            lo, hi = float(np.percentile(fr.level_db, 2)) - 6, float(np.max(fr.level_db)) + 6
            self.pi.setYRange(lo, hi, padding=0)

    def _mouse(self, evt) -> None:
        if self._fr is None or not self.sceneBoundingRect().contains(evt[0]):
            return
        fr = self._fr
        x = self.pi.vb.mapSceneToView(evt[0]).x()
        linear = fr.scale == "linear"
        f = x if linear else 10.0 ** x
        if not (fr.freqs[0] - 1e-9 <= f <= fr.freqs[-1]):
            return
        i = int(np.argmin(np.abs(fr.freqs - f))) if linear else int(np.argmin(np.abs(np.log(fr.freqs / f))))
        how = f"{fr.window_s * 1000:.0f} ms window" if linear else f"1/{fr.smoothing_oct:g} oct, {fr.window_s * 1000:.0f} ms window"
        label = f"{fr.freqs[i]:.1f}" if linear and fr.freqs[i] < 1000 else format_freq(fr.freqs[i])
        self.pi.setTitle(f"{label} Hz   {fr.level_db[i]:.1f} {fr.unit}   ({how})", size="10pt", color=theme.TEXT)


class ImpulsePlot(pg.PlotWidget):
    """Energy-time curve (dB re peak, 1 ms blocks) with the noise floor and truncation point."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="dB re peak", bottom="Time after direct sound (ms)")
        self.pi.setMenuEnabled(False)
        self.pi.setYRange(-110, 3, padding=0)
        self.pi.getAxis("left").setWidth(54)
        self.curve = self.pi.plot(pen=pg.mkPen(theme.ACCENT, width=1.5))
        self.noise = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(theme.AMBER, style=DASH))
        self.cut = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.MUTED, style=DOT))
        self.pi.addItem(self.noise)
        self.pi.addItem(self.cut)

    def set_result(self, res: MeasurementResult) -> None:
        bb = res.decay.broadband
        if bb.energy_db is None:
            return
        peak = float(np.max(bb.energy_db))
        e = bb.energy_db - peak
        self.curve.setData(np.arange(e.size) + 0.5, e)
        if bb.trunc is not None and bb.trunc.noise_power > 0:
            self.noise.setPos(float(10.0 * np.log10(bb.trunc.noise_power)) - peak)
            self.cut.setPos(bb.trunc.index / res.ir.fs * 1000.0)
        self.pi.setXRange(0, max(e.size, 50), padding=0)
