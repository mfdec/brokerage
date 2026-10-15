"""Measure tab: sweep measurement, room decay (RT60), waterfall, frequency and impulse response."""

from __future__ import annotations

import threading

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from roomscope.audio.devices import SIM_DEVICE, DeviceInfo, list_output_devices
from roomscope.audio.engine import AudioError
from roomscope.audio.measure import CHANNEL_MODES, HardwarePlayer, MeasurementCancelled, record_sweep
from roomscope.core.export import export_measurement
from roomscope.core.measurement import MeasurementError, MeasurementResult, analyze_recording, frequency_response
from roomscope.core.sweep import SweepSpec
from roomscope.core.units import format_freq
from roomscope.sim.world import SimPlayer
from roomscope.ui import theme
from roomscope.ui.state import AppState
from roomscope.ui.widgets.decay_plots import ImpulsePlot, ResponsePlot, Rt60Plot, SchroederPlot, band_label
from roomscope.ui.widgets.spectrogram_view import COLORMAPS, SpectrogramView
from roomscope.ui.widgets.waterfall3d import Waterfall3D

WATERFALL_PRESETS = {
    "Bass 20-500 Hz (400 ms window)": dict(f_lo=20.0, f_hi=500.0, window_s=0.40, dt_s=0.010, n_slices=40),
    "Full range 20 Hz-20 kHz (150 ms window)": dict(f_lo=20.0, f_hi=20000.0, window_s=0.15, dt_s=0.005, n_slices=40),
    "Bass detail 20-200 Hz (800 ms window)": dict(f_lo=20.0, f_hi=200.0, window_s=0.80, dt_s=0.020, n_slices=40),
}
SMOOTHING = [("1/3 oct", 3.0), ("1/6 oct", 6.0), ("1/12 oct", 12.0), ("1/24 oct", 24.0), ("none", 0.0)]
RESPONSE_WINDOWS = [("5 ms", 0.005), ("20 ms", 0.02), ("100 ms", 0.1), ("500 ms", 0.5), ("1 s", 1.0), ("full", 3.0)]


class MeasureWorker(QtCore.QThread):
    progress = QtCore.Signal(float, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, source, player, spec: SweepSpec, state: AppState, bands_per_octave: int, parent=None,
                 position: tuple[float, float] | None = None, label: str = ""):
        super().__init__(parent)
        self.source, self.player, self.spec = source, player, spec
        self.calibration = state.calibration
        self.bands = bands_per_octave
        self.position, self.label = position, label
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            rec = record_sweep(self.source, self.player, self.spec, progress=self.progress.emit, cancel=self._cancel)
            self.progress.emit(1.0, "Analyzing...")
            self.done.emit(analyze_recording(rec, self.spec, self.calibration, self.bands,
                                             label=self.label, position=self.position))
        except MeasurementCancelled:
            self.failed.emit("Measurement cancelled.")
        except (MeasurementError, AudioError) as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - surfaced to the user instead of killing the thread silently
            self.failed.emit(f"Unexpected error: {type(e).__name__}: {e}")


class MeasureTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.result: MeasurementResult | None = None
        self.worker: MeasureWorker | None = None
        self._outputs: list[DeviceInfo] = []
        self._closing = False

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self._build_controls(), 0)
        root.addWidget(self._build_results(), 1)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(160)
        self._debounce.timeout.connect(self._refresh_derived)
        state.sourceChanged.connect(self._refresh_enabled)
        state.measurementsChanged.connect(self._refresh_history)
        state.calibrationChanged.connect(self._refresh_derived)
        self.refresh_outputs()
        self._refresh_enabled()
        self._show(None)

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(292)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        how = QtWidgets.QLabel(
            "Plays a sweep through your speakers and records it with the mic that is listening on the Input bar. "
            "Keep the room quiet during the measurement.")
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(how)

        out = QtWidgets.QGroupBox("Output (speakers)")
        of = QtWidgets.QFormLayout(out)
        self.out_dev = QtWidgets.QComboBox()
        self.out_ch = QtWidgets.QComboBox()
        self.out_ch.addItems(CHANNEL_MODES)
        rescan = QtWidgets.QPushButton("Rescan outputs")
        rescan.clicked.connect(self.refresh_outputs)
        of.addRow("Device", self.out_dev)
        of.addRow("Channel", self.out_ch)
        of.addRow(rescan)
        v.addWidget(out)

        sw = QtWidgets.QGroupBox("Sweep")
        sf = QtWidgets.QFormLayout(sw)
        self.dur = QtWidgets.QComboBox()
        for lbl, val in (("2 s (quick)", 2.0), ("6 s (recommended)", 6.0), ("12 s (quietest result)", 12.0)):
            self.dur.addItem(lbl, val)
        self.dur.setCurrentIndex(1)
        self.level = QtWidgets.QDoubleSpinBox()
        self.level.setRange(-60, -1)
        self.level.setValue(-15.0)
        self.level.setSuffix(" dBFS")
        self.level.setToolTip("Digital level of the sweep. Start low and raise your amplifier volume until the "
                              "result reports a good signal-to-noise ratio.")
        self.tail = QtWidgets.QDoubleSpinBox()
        self.tail.setRange(0.5, 8.0)
        self.tail.setValue(2.5)
        self.tail.setSuffix(" s")
        self.tail.setToolTip("Recording time after the sweep, to capture the room decay. Use about 1.5 x the expected RT60.")
        self.bands = QtWidgets.QComboBox()
        self.bands.addItem("Octave bands", 1)
        self.bands.addItem("1/3-octave bands", 3)
        sf.addRow("Duration", self.dur)
        sf.addRow("Level", self.level)
        sf.addRow("Decay tail", self.tail)
        sf.addRow("RT60 bands", self.bands)
        v.addWidget(sw)

        run = QtWidgets.QHBoxLayout()
        self.measure_btn = QtWidgets.QPushButton("Measure")
        self.measure_btn.setObjectName("primary")
        self.measure_btn.clicked.connect(lambda _checked=False: self._start())
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        run.addWidget(self.measure_btn, 2)
        run.addWidget(self.cancel_btn, 1)
        v.addLayout(run)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        v.addWidget(self.progress)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        hist = QtWidgets.QGroupBox("Measurements")
        hf = QtWidgets.QVBoxLayout(hist)
        self.history = QtWidgets.QComboBox()
        self.history.currentIndexChanged.connect(self._history_changed)
        hf.addWidget(self.history)
        clear = QtWidgets.QPushButton("Clear all")
        clear.clicked.connect(self.state.clear_measurements)
        export = QtWidgets.QPushButton("Export CSV / WAV...")
        export.setToolTip("Write the shown measurement to a folder: impulse response (WAV), frequency response and RT60 table (CSV), details (JSON).")
        export.clicked.connect(self._export)
        hf.addWidget(export)
        hf.addWidget(clear)
        v.addWidget(hist)

        tgt = QtWidgets.QGroupBox("RT60 reference band")
        tf = QtWidgets.QFormLayout(tgt)
        self.tgt_lo = QtWidgets.QDoubleSpinBox()
        self.tgt_hi = QtWidgets.QDoubleSpinBox()
        for sp, val in ((self.tgt_lo, 0.30), (self.tgt_hi, 0.60)):
            sp.setRange(0.05, 3.0)
            sp.setSingleStep(0.05)
            sp.setValue(val)
            sp.setSuffix(" s")
            sp.valueChanged.connect(self._target_changed)
        tf.addRow("From", self.tgt_lo)
        tf.addRow("To", self.tgt_hi)
        note = QtWidgets.QLabel("Shaded on the RT60 chart; set to whatever you are aiming for.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        tf.addRow(note)
        v.addWidget(tgt)
        v.addStretch(1)
        return panel

    # ------------------------------------------------------------------ results
    def _build_results(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        self.summary = QtWidgets.QLabel("No measurement yet.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 11pt;")
        v.addWidget(self.summary)
        self.warn = QtWidgets.QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet(f"color: {theme.AMBER};")
        v.addWidget(self.warn)

        self.pages = QtWidgets.QTabWidget()
        v.addWidget(self.pages, 1)

        # decay page
        decay = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.rt60 = Rt60Plot()
        self.rt60.bandSelected.connect(self._select_band)
        decay.addWidget(self.rt60)
        lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.schroeder = SchroederPlot()
        lower.addWidget(self.schroeder)
        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Band Hz", "EDT s", "T20 s", "T30 s", "RT60 s", "C50 dB", "C80 dB", "B*T", "Notes"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._table_selected)
        lower.addWidget(self.table)
        lower.setSizes([420, 640])
        decay.addWidget(lower)
        decay.setSizes([300, 300])
        self.pages.addTab(decay, "Room decay (RT60)")

        # waterfall page
        wf = QtWidgets.QWidget()
        wv = QtWidgets.QVBoxLayout(wf)
        row = QtWidgets.QHBoxLayout()
        self.wf_preset = QtWidgets.QComboBox()
        self.wf_preset.addItems(list(WATERFALL_PRESETS))
        self.wf_smooth = QtWidgets.QComboBox()
        for lbl, val in SMOOTHING:
            self.wf_smooth.addItem(lbl, val)
        self.wf_smooth.setCurrentIndex(2)
        self.wf_range = QtWidgets.QComboBox()
        for db in (30, 40, 50, 60, 70, 80):
            self.wf_range.addItem(f"{db} dB", db)
        self.wf_range.setCurrentIndex(3)
        self.wf_cmap = QtWidgets.QComboBox()
        self.wf_cmap.addItems(COLORMAPS)
        for lbl, w in (("Range", self.wf_preset), ("Smoothing", self.wf_smooth), ("Contrast", self.wf_range), ("Colors", self.wf_cmap)):
            row.addWidget(QtWidgets.QLabel(lbl))
            row.addWidget(w)
        row.addStretch(1)
        wv.addLayout(row)
        self.wf_views = QtWidgets.QTabWidget()
        self.wf3d = Waterfall3D()
        self.wf3d.time_unit, self.wf3d.time_axis_label = "ms", "ms after the direct sound (decay)"
        self.wf2d = SpectrogramView()
        self.wf2d.set_time_label("ms after direct sound", "{t:.0f} ms")
        self.wf_views.addTab(self.wf3d, "3D waterfall")
        self.wf_views.addTab(self.wf2d, "Decay map (2D)")
        wv.addWidget(self.wf_views, 1)
        for w in (self.wf_preset, self.wf_smooth, self.wf_range):
            w.currentIndexChanged.connect(lambda _=0: self._debounce.start())
        self.wf_cmap.currentTextChanged.connect(self._wf_cmap)
        self.pages.addTab(wf, "Waterfall")

        # response page
        rp = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rp)
        row = QtWidgets.QHBoxLayout()
        self.rs_smooth = QtWidgets.QComboBox()
        for lbl, val in SMOOTHING:
            self.rs_smooth.addItem(lbl, val)
        self.rs_smooth.setCurrentIndex(1)
        self.rs_win = QtWidgets.QComboBox()
        for lbl, val in RESPONSE_WINDOWS:
            self.rs_win.addItem(lbl, val)
        self.rs_win.setCurrentIndex(3)
        for lbl, w in (("Smoothing", self.rs_smooth), ("Window", self.rs_win)):
            row.addWidget(QtWidgets.QLabel(lbl))
            row.addWidget(w)
        row.addStretch(1)
        rv.addLayout(row)
        self.response_plot = ResponsePlot()
        rv.addWidget(self.response_plot, 1)
        for w in (self.rs_smooth, self.rs_win):
            w.currentIndexChanged.connect(lambda _=0: self._debounce.start())
        self.pages.addTab(rp, "Frequency response")

        self.impulse = ImpulsePlot()
        self.pages.addTab(self.impulse, "Impulse response")
        return box

    # ------------------------------------------------------------------ outputs / state
    def refresh_outputs(self) -> None:
        self._outputs = list_output_devices()
        self.out_dev.clear()
        for d in self._outputs:
            self.out_dev.addItem(d.label + ("   (default)" if d.is_default_output else ""), d)
        pick = next((i for i, d in enumerate(self._outputs) if d.is_default_output and not d.is_simulated), 0)
        self.out_dev.setCurrentIndex(pick)

    def _refresh_enabled(self) -> None:
        sim = self.state.world is not None and self.state.running
        self.out_dev.setEnabled(not sim)
        self.out_ch.setEnabled(not sim)
        if sim:
            self.out_dev.setCurrentIndex(0)
        busy = self.worker is not None and self.worker.isRunning()
        self.measure_btn.setEnabled(self.state.running and not busy)
        if not self.state.running and not busy:
            self.status.setText("Start the microphone input (top bar) to measure.")
        elif not busy:
            self.status.setText("")

    def _make_player(self):
        if self.state.world is not None:
            return SimPlayer(self.state.world)
        dev: DeviceInfo = self.out_dev.currentData()
        if dev is None or dev.is_simulated:
            raise MeasurementError("Choose an output device (your speakers) first.")
        return HardwarePlayer(dev, self.out_ch.currentText())

    # ------------------------------------------------------------------ run
    def start_with_position(self, xy: tuple[float, float]) -> bool:
        """Measure now and tag the result with a position in the room (used by the walk-the-room map)."""
        return self._start(position=(float(xy[0]), float(xy[1])))

    def _start(self, position: tuple[float, float] | None = None) -> bool:
        src = self.state.source
        if src is None or not src.running or (self.worker is not None and self.worker.isRunning()):
            return False
        try:
            player = self._make_player()
        except MeasurementError as e:
            QtWidgets.QMessageBox.warning(self, "Measure", str(e))
            return False
        spec = SweepSpec(fs=src.fs, duration=float(self.dur.currentData()), level_dbfs=self.level.value(),
                         tail_s=self.tail.value())
        label = f"@({position[0]:.1f}, {position[1]:.1f})" if position is not None else ""
        self.worker = MeasureWorker(src, player, spec, self.state, int(self.bands.currentData()), self,
                                    position=position, label=label)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._worker_finished)
        self.measure_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("Starting...")
        self.worker.start()
        return True

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def _on_progress(self, frac: float, text: str) -> None:
        self.progress.setValue(int(frac * 1000))
        self.status.setText(text)

    def _on_done(self, result: MeasurementResult) -> None:
        self.state.add_measurement(result)
        self.status.setText("Done.")

    def shutdown(self) -> None:
        """Stop a running measurement and wait for its thread (the window is closing)."""
        self._closing = True
        w = self.worker
        if w is not None and w.isRunning():
            w.cancel()
            w.wait(8000)

    def _on_failed(self, msg: str) -> None:
        self.status.setText(msg)
        self.progress.setValue(0)
        if "cancelled" not in msg.lower() and not self._closing:
            QtWidgets.QMessageBox.warning(self, "Measurement failed", msg)

    def _worker_finished(self) -> None:
        self.cancel_btn.setEnabled(False)
        self.worker = None
        self._refresh_enabled()

    def _export(self) -> None:
        res = self.result
        if res is None:
            self.status.setText("Nothing to export yet: run a measurement first.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export the shown measurement to this folder")
        if not folder:
            return
        cal = self.state.calibration
        fr = frequency_response(res.ir, res.spec, cal if cal is not None else res.calibration,
                                window_s=float(self.rs_win.currentData()), smoothing_oct=float(self.rs_smooth.currentData()))
        files = export_measurement(res, folder, fr)
        self.status.setText(f"Exported {len(files)} files to {folder}")

    def wait_idle(self, timeout_ms: int = 60000) -> None:
        """Block until a running measurement finishes (used by tests / scripted runs)."""
        w = self.worker
        if w is not None:
            w.wait(timeout_ms)
            QtWidgets.QApplication.processEvents()

    # ------------------------------------------------------------------ history & display
    def _refresh_history(self) -> None:
        self.history.blockSignals(True)
        self.history.clear()
        for m in self.state.measurements:
            tm = QtCore.QDateTime.fromSecsSinceEpoch(int(m.created)).toString("HH:mm:ss")
            bb = m.decay.broadband
            self.history.addItem(f"{m.label}  {tm}  RT60 {bb.t60:.2f} s", m)
        if self.state.measurements:
            self.history.setCurrentIndex(self.history.count() - 1)
        self.history.blockSignals(False)
        self._show(self.state.measurements[-1] if self.state.measurements else None)

    def _history_changed(self) -> None:
        m = self.history.currentData()
        if m is not None:
            self._show(m)

    def _show(self, res: MeasurementResult | None) -> None:
        self.result = res
        if res is None:
            self.summary.setText("No measurement yet. Start the input, set the output device and press Measure.")
            self.warn.setText("")
            self.table.setRowCount(0)
            return
        self.summary.setText(res.summary())
        self.warn.setText("\n".join("! " + w for w in res.warnings))
        d = res.decay
        k = int(np.argmin(np.abs(np.log(d.centers / 1000.0))))
        self.rt60.set_target(self.tgt_lo.value(), self.tgt_hi.value())
        self.rt60.set_result(d, k)
        self._fill_table(res)
        self._select_band(k)
        self._refresh_derived()
        self.impulse.set_result(res)

    def _fill_table(self, res: MeasurementResult) -> None:
        third = res.decay.bands_per_octave == 3
        rows = res.decay.bands + [res.decay.broadband]
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        f = lambda x: "-" if not np.isfinite(x) else f"{x:.2f}"          # noqa: E731
        for r, b in enumerate(rows):
            label = "Broadband" if r == len(rows) - 1 else band_label(b, third)
            cells = [label, f(b.edt), f(b.t20), f(b.t30), f(b.t60), "-" if not np.isfinite(b.c50_db) else f"{b.c50_db:.1f}",
                     "-" if not np.isfinite(b.c80_db) else f"{b.c80_db:.1f}", "-" if not np.isfinite(b.bt) else f"{b.bt:.0f}",
                     "; ".join(b.notes)]
            for c, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                if c in range(1, 8):
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if not b.reliable:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(theme.MUTED)))
                elif c == 4:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(theme.ACCENT)))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _select_band(self, i: int) -> None:
        if self.result is None:
            return
        d = self.result.decay
        bands = d.bands + [d.broadband]
        i = int(np.clip(i, 0, len(bands) - 1))
        third = d.bands_per_octave == 3
        title = "Broadband" if i == len(bands) - 1 else f"{band_label(bands[i], third)} Hz"
        self.schroeder.set_band(bands[i], self.result.ir.fs, title)
        self.table.blockSignals(True)
        self.table.selectRow(i)
        self.table.blockSignals(False)
        if i < len(d.bands):
            self.rt60.sel.setPos(i)

    def _table_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._select_band(rows[0].row())

    def _target_changed(self) -> None:
        self.rt60.set_target(min(self.tgt_lo.value(), self.tgt_hi.value()), max(self.tgt_lo.value(), self.tgt_hi.value()))

    def _wf_cmap(self, name: str) -> None:
        self.wf3d.set_colormap(name)
        self.wf2d.set_colormap(name)

    def _refresh_derived(self) -> None:
        """Waterfall + frequency response, recomputed from the stored impulse response."""
        res = self.result
        if res is None:
            return
        cal = self.state.calibration
        preset = dict(WATERFALL_PRESETS[self.wf_preset.currentText()])
        wf = res.waterfall(smoothing_oct=float(self.wf_smooth.currentData()), **preset)
        hi = float(np.max(wf.slices_db[0])) + 2.0
        lo = hi - float(self.wf_range.currentData())
        t_ms = wf.times_s * 1000.0
        self.wf3d.level_unit = res.response.unit
        self.wf3d.set_data(wf.freqs, wf.slices_db, t_ms, lo, hi, span=float(t_ms[-1]))
        self.wf2d.set_levels(lo, hi)
        self.wf2d.set_data(wf.freqs, wf.slices_db.astype(np.float32), float(wf.times_s[1] - wf.times_s[0]) * 1000.0, reset=True)
        fr = frequency_response(res.ir, res.spec, cal if cal is not None else res.calibration,
                                window_s=float(self.rs_win.currentData()), smoothing_oct=float(self.rs_smooth.currentData()))
        self.response_plot.set_response(fr)
