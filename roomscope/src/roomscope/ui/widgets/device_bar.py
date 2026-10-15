"""Top bar: input device, sample rate, start/stop, mic-calibration file and SPL reference."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from roomscope.audio.devices import (
    DeviceInfo, backend_error, check_input, likely_measurement_mic, list_input_devices, rates_for,
)
from roomscope.audio.engine import AudioError
from roomscope.core.calibration import CalibrationError
from roomscope.ui.dialogs import SplCalibrationDialog
from roomscope.ui.state import AppState


class DeviceBar(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.devices: list[DeviceInfo] = []
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 2)
        lay.setSpacing(8)

        lay.addWidget(QtWidgets.QLabel("Input"))
        self.dev = QtWidgets.QComboBox()
        self.dev.setMinimumWidth(330)
        self.dev.currentIndexChanged.connect(self._device_changed)
        lay.addWidget(self.dev, 1)
        self.fs = QtWidgets.QComboBox()
        lay.addWidget(self.fs)
        self.ch = QtWidgets.QSpinBox()
        self.ch.setPrefix("Ch ")
        self.ch.setRange(1, 1)
        self.ch.setToolTip("Which input channel carries the measurement mic")
        lay.addWidget(self.ch)
        rescan = QtWidgets.QPushButton("Rescan")
        rescan.setToolTip("Re-read the list of audio devices")
        rescan.clicked.connect(self.refresh_devices)
        lay.addWidget(rescan)
        self.start = QtWidgets.QPushButton("Start listening")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self._toggle)
        lay.addWidget(self.start)

        lay.addSpacing(14)
        self.cal_btn = QtWidgets.QToolButton()
        self.cal_btn.setText("Mic cal file")
        self.cal_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(self.cal_btn)
        menu.addAction("Load calibration file...", self._load_cal)
        menu.addAction("Clear calibration file", self.state.clear_mic_cal)
        self.cal_btn.setMenu(menu)
        lay.addWidget(self.cal_btn)
        spl = QtWidgets.QPushButton("SPL reference")
        spl.clicked.connect(lambda: SplCalibrationDialog(self.state, self).exec())
        lay.addWidget(spl)
        self.cal_label = QtWidgets.QLabel()
        self.cal_label.setStyleSheet("color: #8b95a3;")
        lay.addWidget(self.cal_label)

        state.sourceChanged.connect(self._refresh_button)
        state.calibrationChanged.connect(self._refresh_cal)
        self.refresh_devices()
        self._refresh_cal()

    # ------------------------------------------------------------------ devices
    def refresh_devices(self) -> None:
        keep = self.dev.currentData().name if self.dev.currentData() else self.state.settings.value("audio/device", "", str)
        self.devices = list_input_devices()
        self.dev.blockSignals(True)
        self.dev.clear()
        for d in self.devices:
            self.dev.addItem(d.label + ("   (default)" if d.is_default_input else ""), d)
        pick = 0
        names = [d.name for d in self.devices]
        if keep in names:
            pick = names.index(keep)
        else:
            mic = likely_measurement_mic(self.devices)
            if mic is not None:
                pick = self.devices.index(mic)
            elif len(self.devices) == 1:
                pick = 0
        self.dev.setCurrentIndex(pick)
        self.dev.blockSignals(False)
        self._device_changed()
        err = backend_error()
        if err and len(self.devices) == 1:
            self.state.message.emit(f"No audio backend: {err}. Only the simulated room is available.")

    def _device_changed(self) -> None:
        d: DeviceInfo | None = self.dev.currentData()
        if d is None:
            return
        self.fs.clear()
        for r in rates_for(d):
            self.fs.addItem(f"{r} Hz", r)
        saved = self.state.settings.value("audio/fs", 48000, int)
        i = self.fs.findData(saved)
        self.fs.setCurrentIndex(max(i, 0))
        self.ch.setRange(1, max(d.max_input, 1))
        self.ch.setVisible(d.max_input > 1)

    # ------------------------------------------------------------------ start / stop
    def _toggle(self) -> None:
        if self.state.running:
            self.state.close_source()
            return
        d: DeviceInfo = self.dev.currentData()
        fs = int(self.fs.currentData())
        problem = check_input(d, fs)
        if problem:
            QtWidgets.QMessageBox.warning(self, "Input device", f"{d.name} cannot record at {fs} Hz:\n\n{problem}")
            return
        try:
            self.state.open_source(d, fs, self.ch.value() - 1)
        except AudioError as e:
            QtWidgets.QMessageBox.critical(self, "Cannot start input", str(e))

    def _refresh_button(self) -> None:
        running = self.state.running
        self.start.setText("Stop" if running else "Start listening")
        self.start.setObjectName("danger" if running else "primary")
        self.start.style().unpolish(self.start)
        self.start.style().polish(self.start)
        self.dev.setEnabled(not running)
        self.fs.setEnabled(not running)
        self.ch.setEnabled(not running)

    # ------------------------------------------------------------------ calibration
    def _load_cal(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Microphone calibration file", "", "Calibration files (*.txt *.csv *.cal *.frd);;All files (*)")
        if not path:
            return
        try:
            self.state.load_mic_cal(path)
        except CalibrationError as e:
            QtWidgets.QMessageBox.warning(self, "Calibration file", str(e))

    def _refresh_cal(self) -> None:
        self.cal_label.setText(self.state.calibration.describe())
