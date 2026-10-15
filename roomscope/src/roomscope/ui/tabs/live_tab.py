"""Live tab: real-time analyzer (RTA) with level meter."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtWidgets

from roomscope.core.meter import LevelMeter
from roomscope.core.spectrum import RESOLUTIONS, WINDOWS, RtaEngine, RtaSettings
from roomscope.core.weighting import WEIGHTINGS
from roomscope.ui.state import AppState
from roomscope.ui.widgets.level_meter import LevelMeterWidget
from roomscope.ui.widgets.rta_plot import RtaPlot

FFT_SIZES = [4096, 8192, 16384, 32768, 65536]
AVERAGING = [("Off", 0.0), ("0.25 s", 0.25), ("0.5 s", 0.5), ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0)]
FRAME_MS = 33


class LiveTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.engine: RtaEngine | None = None
        self.meter: LevelMeter | None = None
        self._meter_cursor = 0
        self._last_t = time.monotonic()
        self._unit = state.calibration.unit
        self.frame_ms = 0.0               # last analysis+paint cost, for the status bar
        self.fps = 0.0

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self._build_controls(), 0)
        right = QtWidgets.QVBoxLayout()
        self.plot = RtaPlot()
        right.addWidget(self.plot, 3)
        self.lower = QtWidgets.QTabWidget()           # spectrogram / waterfall views are added here (v0.2)
        self.lower.setVisible(False)
        right.addWidget(self.lower, 2)
        root.addLayout(right, 1)

        self._apply_settings(initial=True)
        state.sourceChanged.connect(self._source_changed)
        state.calibrationChanged.connect(self._calibration_changed)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ------------------------------------------------------------------ UI
    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(250)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        self.meter_widget = LevelMeterWidget()
        v.addWidget(self.meter_widget)

        box = QtWidgets.QGroupBox("Analyzer")
        f = QtWidgets.QFormLayout(box)
        self.fft = QtWidgets.QComboBox()
        for n in FFT_SIZES:
            self.fft.addItem(f"{n // 1024}k  ({n})", n)
        self.fft.setCurrentIndex(FFT_SIZES.index(16384))
        self.window = QtWidgets.QComboBox()
        self.window.addItems(list(WINDOWS))
        self.avg = QtWidgets.QComboBox()
        for lbl, val in AVERAGING:
            self.avg.addItem(lbl, val)
        self.avg.setCurrentIndex(2)
        self.res = QtWidgets.QComboBox()
        self.res.addItems(RESOLUTIONS)
        self.res.setCurrentText("1/12")
        self.res.setToolTip("Fractional-octave band level (true RTA). 'FFT' shows raw bins; use Flat-top for exact tone levels.")
        self.weight = QtWidgets.QComboBox()
        self.weight.addItems(list(WEIGHTINGS))
        f.addRow("FFT size", self.fft)
        f.addRow("Window", self.window)
        f.addRow("Averaging", self.avg)
        f.addRow("Resolution", self.res)
        f.addRow("Weighting", self.weight)
        v.addWidget(box)

        disp = QtWidgets.QGroupBox("Display")
        d = QtWidgets.QFormLayout(disp)
        self.top = QtWidgets.QDoubleSpinBox()
        self.top.setRange(-60, 200)
        self.top.setValue(0.0)
        self.top.setSuffix(" dB")
        self.span = QtWidgets.QDoubleSpinBox()
        self.span.setRange(20, 200)
        self.span.setValue(100.0)
        self.span.setSuffix(" dB")
        self.show_inst = QtWidgets.QCheckBox("Show instantaneous")
        self.show_inst.setChecked(True)
        self.show_peak = QtWidgets.QCheckBox("Peak hold")
        d.addRow("Top", self.top)
        d.addRow("Range", self.span)
        d.addRow(self.show_inst)
        d.addRow(self.show_peak)
        v.addWidget(disp)
        v.addStretch(1)

        for w in (self.fft, self.window, self.avg, self.res, self.weight):
            w.currentIndexChanged.connect(lambda _=0: self._apply_settings())
        for w in (self.top, self.span):
            w.valueChanged.connect(lambda _=0: self._apply_settings())
        self.show_inst.toggled.connect(lambda _=False: self._apply_settings())
        self.show_peak.toggled.connect(lambda _=False: self._apply_settings())
        return panel

    # ------------------------------------------------------------------ settings & wiring
    def settings(self) -> RtaSettings:
        return RtaSettings(
            fft_size=int(self.fft.currentData()), window=self.window.currentText(),
            averaging_s=float(self.avg.currentData()), resolution=self.res.currentText(),
            weighting=self.weight.currentText(),
        )

    def _apply_settings(self, initial: bool = False) -> None:
        s = self.settings()
        if self.engine is not None:
            self.engine.set_settings(s)
        if self.meter is not None and self.meter.weighting != s.weighting:
            self.meter.set_weighting(s.weighting)
        self.plot.set_range(self.top.value(), self.span.value())
        self.plot.show_inst = self.show_inst.isChecked()
        self.plot.show_peak = self.show_peak.isChecked()

    def _source_changed(self) -> None:
        src = self.state.source
        if src is None:
            return
        self.engine = RtaEngine(src.fs, self.settings(), self.state.calibration)
        self.meter = LevelMeter(src.fs, self.weight.currentText())
        self._meter_cursor = src.ring.total
        self._last_t = time.monotonic()

    def _calibration_changed(self) -> None:
        cal = self.state.calibration
        if self.engine is not None:
            self.engine.set_calibration(cal)
        if cal.unit != self._unit:
            self._unit = cal.unit
            self.top.setValue(110.0 if cal.unit == "dB SPL" else 0.0)
        self.plot.set_unit(cal.unit)

    # ------------------------------------------------------------------ frame loop
    def _tick(self) -> None:
        src = self.state.source
        if src is None or not src.running or self.engine is None or not self.isVisible():
            return
        t0 = time.perf_counter()
        now = time.monotonic()
        dt, self._last_t = now - self._last_t, now
        fr = self.engine.process(src.ring.read_latest(self.engine.settings.fft_size), dt)
        self.plot.set_frame(fr)

        new, self._meter_cursor, _over = src.ring.read_since(self._meter_cursor)
        if self.meter is not None and new.size:
            r = self.meter.process(new)
            off = self.state.calibration.offset
            self.meter_widget.set_values(r.tw_db + off, r.tw_db, r.peak_db, r.leq_db + off, r.clipped,
                                         self.state.calibration.unit, self.weight.currentText(), off)
        cost = (time.perf_counter() - t0) * 1000.0
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * cost
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-3))
