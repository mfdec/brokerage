"""Application-wide state shared by all tabs (audio source, calibration, persisted settings)."""

from __future__ import annotations

from PySide6 import QtCore

from roomscope.audio.devices import DeviceInfo
from roomscope.audio.engine import AudioError, AudioSource, HardwareInput, SimulatedInput
from roomscope.core.calibration import Calibration, CalibrationError, load_calibration
from roomscope.sim.demo import DemoRoomSignal


class AppState(QtCore.QObject):
    sourceChanged = QtCore.Signal()
    calibrationChanged = QtCore.Signal()
    message = QtCore.Signal(str)

    def __init__(self, settings: QtCore.QSettings | None = None, parent=None):
        super().__init__(parent)
        self.settings = settings or QtCore.QSettings("RoomScope", "RoomScope")
        self.calibration = Calibration()
        self.source: AudioSource | None = None
        self.device: DeviceInfo | None = None
        self._restore_calibration()

    # ------------------------------------------------------------------ audio
    def open_source(self, device: DeviceInfo, fs: int, channel: int = 0) -> AudioSource:
        self.close_source()
        if device.is_simulated:
            src: AudioSource = SimulatedInput(DemoRoomSignal(fs), fs, name=device.name)
        else:
            src = HardwareInput(device, fs, channel)
        try:
            src.start()
        except AudioError:
            self.sourceChanged.emit()
            raise
        self.source, self.device = src, device
        self.settings.setValue("audio/device", device.name)
        self.settings.setValue("audio/fs", fs)
        self.sourceChanged.emit()
        self.message.emit(f"Listening: {device.name} @ {fs} Hz")
        return src

    def close_source(self) -> None:
        if self.source is not None:
            self.source.stop()
            self.source = None
            self.sourceChanged.emit()

    @property
    def running(self) -> bool:
        return self.source is not None and self.source.running

    # ------------------------------------------------------------------ calibration
    def load_mic_cal(self, path: str) -> None:
        cal = load_calibration(path)                      # raises CalibrationError
        self.calibration = Calibration(mic=cal, spl_offset_db=self.calibration.spl_offset_db)
        self.settings.setValue("cal/path", path)
        self.calibrationChanged.emit()
        self.message.emit(f"Loaded mic calibration '{cal.name}' ({cal.freqs.size} points, {cal.span[0]:g}-{cal.span[1]:g} Hz)")

    def clear_mic_cal(self) -> None:
        self.calibration = Calibration(mic=None, spl_offset_db=self.calibration.spl_offset_db)
        self.settings.remove("cal/path")
        self.calibrationChanged.emit()

    def set_spl_offset(self, offset: float | None) -> None:
        self.calibration = Calibration(mic=self.calibration.mic, spl_offset_db=offset)
        if offset is None:
            self.settings.remove("cal/spl_offset")
        else:
            self.settings.setValue("cal/spl_offset", float(offset))
        self.calibrationChanged.emit()

    def _restore_calibration(self) -> None:
        mic = None
        path = self.settings.value("cal/path", "", str)
        if path:
            try:
                mic = load_calibration(path)
            except CalibrationError:
                mic = None
        off = self.settings.value("cal/spl_offset", None)
        try:
            off = None if off in (None, "") else float(off)
        except (TypeError, ValueError):
            off = None
        self.calibration = Calibration(mic=mic, spl_offset_db=off)
