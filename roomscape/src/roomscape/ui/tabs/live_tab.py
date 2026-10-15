"""Live tab: real-time analyzer (RTA), level meter, scrolling spectrogram and 3D waterfall."""

from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from roomscape.core.meter import LevelMeter
from roomscape.core.spectrogram import MODES, SpectrogramEngine, SpectrogramHistory, SpectrogramSettings
from roomscape.core.spectrum import RESOLUTIONS, WINDOWS, RtaEngine, RtaSettings
from roomscape.core.weighting import WEIGHTINGS
from roomscape.ui.state import AppState
from roomscape.ui.widgets.level_meter import LevelMeterWidget
from roomscape.ui.widgets.rta_plot import RtaPlot
from roomscape.ui.widgets.spectrogram_view import COLORMAPS, SpectrogramView
from roomscape.ui.widgets.waterfall3d import Waterfall3D

FFT_SIZES = [4096, 8192, 16384, 32768, 65536]
AVERAGING = [("Off", 0.0), ("0.25 s", 0.25), ("0.5 s", 0.5), ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0)]
FRAME_MS = 33
HOP_S = 0.05                      # spectrogram column interval
HISTORY_S = (10, 20, 30)
LEFT_AXIS_W = 62
# (label, scale, lowest Hz, highest Hz): the log views reach down to 2 Hz, the linear ones start at 0 Hz.
AXES = [
    ("Log  20 Hz - 20 kHz", "log", 20.0, 20000.0),
    ("Log  2 Hz - 20 kHz", "log", 2.0, 20000.0),
    ("Linear  0 - 300 Hz", "linear", 0.0, 300.0),
    ("Linear  0 - 1 kHz", "linear", 0.0, 1000.0),
    ("Linear  0 - 5 kHz", "linear", 0.0, 5000.0),
    ("Linear  0 - 20 kHz", "linear", 0.0, 20000.0),
]
_LINEAR_MODE_OF = {300.0: "Linear 0-300 Hz", 1000.0: "Linear 0-1 kHz", 5000.0: "Linear 0-5 kHz", 20000.0: "Linear 0-20 kHz"}
_MIN_FFT_FOR = {300.0: 65536, 1000.0: 32768}          # finer bins for the room-mode views


class LiveTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.engine: RtaEngine | None = None
        self.meter: LevelMeter | None = None
        self.spec_engine: SpectrogramEngine | None = None
        self.history: SpectrogramHistory | None = None
        self._meter_cursor = 0
        self._spec_end = 0
        self._last_t = time.monotonic()
        self._unit = state.calibration.unit
        self._auto_top = -20.0
        self._tick_n = 0
        self._axis_key = None
        self.frame_ms = 0.0               # last analysis+paint cost, for the status bar
        self.fps = 0.0

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self._build_controls(), 0)
        right = QtWidgets.QVBoxLayout()
        self.plot = RtaPlot()
        self.plot.pi.getAxis("left").setWidth(LEFT_AXIS_W)
        # pyqtgraph lines linked views up using each widget's position inside its parent; the spectrogram sits at (0, 0) in a
        # tab page, so the RTA gets a zero-margin holder of its own - otherwise the two frequency axes are offset on screen
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.plot)
        right.addWidget(holder, 5)
        right.addWidget(self._build_lower(), 4)
        root.addLayout(right, 1)

        self._apply_settings()
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
        self.axis = QtWidgets.QComboBox()
        for label, scale, lo, hi in AXES:
            self.axis.addItem(label, (scale, lo, hi))
        self.axis.setToolTip("Log: the usual view (the second entry reaches down to 2 Hz).\nLinear: evenly spaced frequency starting at 0 Hz, "
                             "best for room modes and infrasound. It shows FFT bins, so pick a large FFT size.\n"
                             "The microphone itself is only specified down to about 20 Hz and the USB audio path removes DC, "
                             "so what you see near 0 Hz is mostly noise and rumble.")
        d.addRow("Frequency axis", self.axis)
        d.addRow("Top", self.top)
        d.addRow("Range", self.span)
        d.addRow(self.show_inst)
        d.addRow(self.show_peak)
        v.addWidget(disp)

        sp = QtWidgets.QGroupBox("Spectrogram / waterfall")
        s = QtWidgets.QFormLayout(sp)
        self.sp_mode = QtWidgets.QComboBox()
        self.sp_mode.addItems(list(MODES))
        self.sp_mode.setToolTip("Balanced: long windows for bass, short for treble.\nBass detail: finest low-frequency resolution.\nFast: best timing.")
        self.sp_cmap = QtWidgets.QComboBox()
        self.sp_cmap.addItems(COLORMAPS)
        self.sp_hist = QtWidgets.QComboBox()
        for sec in HISTORY_S:
            self.sp_hist.addItem(f"{sec} s", sec)
        self.sp_contrast = QtWidgets.QComboBox()
        for db in (40, 50, 60, 70, 80, 100):
            self.sp_contrast.addItem(f"{db} dB", db)
        self.sp_contrast.setCurrentIndex(2)
        self.sp_auto = QtWidgets.QCheckBox("Auto level")
        self.sp_auto.setChecked(True)
        s.addRow("Resolution", self.sp_mode)
        s.addRow("Colors", self.sp_cmap)
        s.addRow("History", self.sp_hist)
        s.addRow("Contrast", self.sp_contrast)
        s.addRow(self.sp_auto)
        v.addWidget(sp)
        v.addStretch(1)

        for w in (self.fft, self.window, self.avg, self.res, self.weight):
            w.currentIndexChanged.connect(lambda _=0: self._apply_settings())
        self.axis.currentIndexChanged.connect(lambda _=0: self._axis_changed())
        for w in (self.top, self.span):
            w.valueChanged.connect(lambda _=0: self._apply_settings())
        self.show_inst.toggled.connect(lambda _=False: self._apply_settings())
        self.show_peak.toggled.connect(lambda _=False: self._apply_settings())
        self.sp_mode.currentIndexChanged.connect(lambda _=0: self._rebuild_spectrogram())
        self.sp_hist.currentIndexChanged.connect(lambda _=0: self._rebuild_spectrogram())
        self.sp_cmap.currentTextChanged.connect(self._set_colormap)
        return panel

    def _build_lower(self) -> QtWidgets.QWidget:
        self.lower = QtWidgets.QTabWidget()
        self.spectro = SpectrogramView()
        self.spectro.pi.getAxis("left").setWidth(LEFT_AXIS_W)
        self.spectro.pi.setXLink(self.plot.pi)
        self.waterfall = Waterfall3D()
        self.lower.addTab(self.spectro, "Spectrogram")
        self.lower.addTab(self.waterfall, "3D waterfall")
        return self.lower

    def _set_colormap(self, name: str) -> None:
        self.spectro.set_colormap(name)
        self.waterfall.set_colormap(name)

    # ------------------------------------------------------------------ settings & wiring
    def axis_spec(self) -> tuple[str, float, float]:
        return self.axis.currentData()

    def settings(self) -> RtaSettings:
        scale, lo, hi = self.axis_spec()
        return RtaSettings(
            fft_size=int(self.fft.currentData()), window=self.window.currentText(),
            averaging_s=float(self.avg.currentData()), resolution="FFT" if scale == "linear" else self.res.currentText(),
            weighting=self.weight.currentText(), f_min=lo, f_max=hi,
        )

    def _axis_changed(self) -> None:
        scale, lo, hi = self.axis_spec()
        want = _MIN_FFT_FOR.get(hi) if scale == "linear" else None
        if want and int(self.fft.currentData()) < want:                      # room-mode views need fine frequency bins
            self.fft.setCurrentIndex(FFT_SIZES.index(want))
        self.res.setEnabled(scale == "log")
        self.sp_mode.setEnabled(scale == "log")
        self._apply_settings()
        self._rebuild_spectrogram()

    def _apply_settings(self) -> None:
        s = self.settings()
        if self.engine is not None:
            self.engine.set_settings(s)
        if self.meter is not None and self.meter.weighting != s.weighting:
            self.meter.set_weighting(s.weighting)
        key = self.axis_spec()
        if key != self._axis_key:
            self._axis_key = key
            self.plot.set_axis(*key)
        self.plot.set_range(self.top.value(), self.span.value())
        self.plot.show_inst = self.show_inst.isChecked()
        self.plot.show_peak = self.show_peak.isChecked()

    def _rebuild_spectrogram(self) -> None:
        src = self.state.source
        if src is None:
            return
        scale, lo, hi = self.axis_spec()
        if scale == "linear":
            cfg = SpectrogramSettings(mode=_LINEAR_MODE_OF[hi])
        else:
            cfg = SpectrogramSettings(mode=self.sp_mode.currentText(), f_min=lo, f_max=hi)
        self.spec_engine = SpectrogramEngine(src.fs, cfg, self.state.calibration)
        rows = int(int(self.sp_hist.currentData()) / HOP_S)
        self.history = SpectrogramHistory(self.spec_engine.grid.size, rows, HOP_S)
        self._spec_end = src.ring.total
        self.waterfall.set_freq_scale(scale)
        self.spectro.set_data(self.spec_engine.grid, self.history.data, HOP_S, reset=True, scale=scale)

    def _source_changed(self) -> None:
        src = self.state.source
        if src is None:
            return
        self.engine = RtaEngine(src.fs, self.settings(), self.state.calibration)
        self.meter = LevelMeter(src.fs, self.weight.currentText())
        self._meter_cursor = src.ring.total
        self._last_t = time.monotonic()
        self._rebuild_spectrogram()

    def _calibration_changed(self) -> None:
        cal = self.state.calibration
        if self.engine is not None:
            self.engine.set_calibration(cal)
        if self.spec_engine is not None:
            self.spec_engine.set_calibration(cal)
        if cal.unit != self._unit:
            self._unit = cal.unit
            self.top.setValue(110.0 if cal.unit == "dB SPL" else 0.0)
            self._auto_top = 90.0 if cal.unit == "dB SPL" else -20.0
        self.plot.set_unit(cal.unit)
        self.waterfall.level_unit = cal.unit

    # ------------------------------------------------------------------ frame loop
    def _tick(self) -> None:
        src = self.state.source
        if src is None or not src.running or self.engine is None or not self.isVisible():
            return
        t0 = time.perf_counter()
        now = time.monotonic()
        dt, self._last_t = now - self._last_t, now
        self._tick_n += 1
        fr = self.engine.process(src.ring.read_latest(self.engine.settings.fft_size), dt)
        self.plot.set_frame(fr)

        new, self._meter_cursor, _over = src.ring.read_since(self._meter_cursor)
        if self.meter is not None and new.size:
            r = self.meter.process(new)
            off = self.state.calibration.offset
            self.meter_widget.set_values(r.tw_db + off, r.tw_db, r.peak_db, r.leq_db + off, r.clipped,
                                         self.state.calibration.unit, self.weight.currentText(), off)
        self._update_spectro(src)
        cost = (time.perf_counter() - t0) * 1000.0
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * cost
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-3))

    def _update_spectro(self, src) -> None:
        if self.spec_engine is None or self.history is None:
            return
        hop = int(HOP_S * src.fs)
        end = src.ring.total
        n_new = (end - self._spec_end) // hop
        if n_new < 1:
            return
        col = self.spec_engine.compute(src.ring.read_latest(self.spec_engine.required_samples))
        for _ in range(min(n_new, 8)):            # repeat on stalls so the time axis stays honest
            self.history.push(col)
        self._spec_end += n_new * hop
        contrast = float(self.sp_contrast.currentData())
        if self.sp_auto.isChecked():
            self._auto_top += 0.15 * (float(col.max()) + 4.0 - self._auto_top)
            top = self._auto_top
        else:
            top = self.top.value()
        lo, hi = top - contrast, top
        idx = self.lower.currentIndex()
        if idx == 0:
            self.spectro.set_levels(lo, hi)
            self.spectro.set_data(self.spec_engine.grid, self.history.data, HOP_S, scale=self.axis_spec()[0])
        elif self._tick_n % 2 == 0:               # waterfall repaints at half rate
            n_sl = 48
            sl, ago = self.history.slices(n_sl, min(self.history.span_s, 8.0))
            self.waterfall.time_unit, self.waterfall.time_axis_label = "s", "seconds ago"
            self.waterfall.set_data(self.spec_engine.grid, sl, ago, lo, hi, span=float(max(ago.max(), 1e-3)))
