"""Dialogs: absolute-level (SPL) calibration."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from roomscope.core.calibration import spl_offset_from_calibrator
from roomscope.core.spectrum import tone_level_dbfs
from roomscope.ui.state import AppState


class SplCalibrationDialog(QtWidgets.QDialog):
    """Set the dBFS -> dB SPL offset by hand or from an acoustic calibrator (94/114 dB at 1 kHz)."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Absolute level (SPL) reference")
        self.setMinimumWidth(460)
        lay = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Relative results (decay times, room maps, response shape) do not need this. Set it only if you want "
            "levels in dB SPL. Keep the Windows input level and the mic position/hardware unchanged afterwards."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        box = QtWidgets.QGroupBox("From an acoustic calibrator")
        form = QtWidgets.QFormLayout(box)
        self.cal_level = QtWidgets.QDoubleSpinBox()
        self.cal_level.setRange(60, 140)
        self.cal_level.setValue(94.0)
        self.cal_level.setSuffix(" dB SPL")
        self.cal_freq = QtWidgets.QDoubleSpinBox()
        self.cal_freq.setRange(50, 4000)
        self.cal_freq.setValue(1000.0)
        self.cal_freq.setSuffix(" Hz")
        self.measure_btn = QtWidgets.QPushButton("Measure calibrator (3 s)")
        self.measure_btn.clicked.connect(self._measure)
        self.result = QtWidgets.QLabel("Fit the calibrator on the mic, start listening, then press Measure.")
        self.result.setWordWrap(True)
        form.addRow("Calibrator level", self.cal_level)
        form.addRow("Tone frequency", self.cal_freq)
        form.addRow(self.measure_btn)
        form.addRow(self.result)
        lay.addWidget(box)

        manual = QtWidgets.QGroupBox("Manual offset (dB SPL = dBFS + offset)")
        mf = QtWidgets.QFormLayout(manual)
        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setRange(-50, 200)
        self.offset.setDecimals(2)
        cur = state.calibration.spl_offset_db
        self.offset.setValue(0.0 if cur is None else cur)
        mf.addRow("Offset", self.offset)
        lay.addWidget(manual)

        btns = QtWidgets.QDialogButtonBox()
        self.apply_btn = btns.addButton("Apply offset", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.clear_btn = btns.addButton("Use dBFS (clear)", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        btns.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_btn.clicked.connect(self._apply)
        self.clear_btn.clicked.connect(self._clear)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _measure(self) -> None:
        src = self.state.source
        if src is None or not src.running:
            self.result.setText("Start the input first (top bar).")
            return
        x = src.ring.read_latest(int(3 * src.fs))
        if src.seconds_recorded < 3:
            self.result.setText("Wait a few seconds for the buffer to fill, then try again.")
            return
        f0 = self.cal_freq.value()
        level = tone_level_dbfs(x, src.fs, f0)
        mic = self.state.calibration.mic
        if mic is not None:
            level -= float(mic.correction_db(f0))
        off = spl_offset_from_calibrator(level, self.cal_level.value())
        self.offset.setValue(off)
        self.result.setText(f"Tone reads {level:.2f} dBFS  ->  offset {off:+.2f} dB. Press 'Apply offset' to keep it.")

    def _apply(self) -> None:
        self.state.set_spl_offset(self.offset.value())
        self.accept()

    def _clear(self) -> None:
        self.state.set_spl_offset(None)
        self.accept()
