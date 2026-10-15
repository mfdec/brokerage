"""Main window: device bar, feature tabs, status bar."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from roomscape import __version__
from roomscape.ui.state import AppState
from roomscape.ui.tabs.camera_tab import CameraTab
from roomscape.ui.tabs.live_tab import LiveTab
from roomscape.ui.tabs.decay_panel import DecayPanel
from roomscape.ui.tabs.mapping_panel import MappingPanel
from roomscape.ui.tabs.measure_tab import MeasureTab
from roomscape.ui.tabs.modes_tab import ModesTab
from roomscape.ui.widgets.device_bar import DeviceBar


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, state: AppState | None = None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle(f"RoomScape {__version__}")
        self.resize(1360, 860)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.device_bar = DeviceBar(self.state)
        lay.addWidget(self.device_bar)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.live = LiveTab(self.state)
        self.tabs.addTab(self.live, "Live")
        self.measure = MeasureTab(self.state)
        self.tabs.addTab(self.measure, "Measure / Decay")
        self.modes = ModesTab(self.state)
        self.tabs.addTab(self.modes, "Room modes")
        self.camera_tab = CameraTab(self.state)
        self.camera_tab.measure_tab = self.measure
        self.camera_tab.output_provider = lambda: (self.measure.out_dev.currentData(), self.measure.out_ch.currentText())
        self.mapping = MappingPanel(self.camera_tab)
        self.decay = DecayPanel(self.camera_tab, self.mapping)
        self.tabs.addTab(self.camera_tab, "Camera map")

        self.status_left = QtWidgets.QLabel("Ready")
        self.status_right = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addPermanentWidget(self.status_right)
        self.state.message.connect(self.status_left.setText)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

    def _update_status(self) -> None:
        src = self.state.source
        if src is None or not src.running:
            self.status_right.setText("not listening")
            return
        sim = " (simulated)" if src.is_simulated else ""
        self.status_right.setText(
            f"{src.name}{sim}  {src.fs} Hz   {src.seconds_recorded:,.0f} s   xruns {src.xruns}   "
            f"{self.live.fps:.0f} fps / {self.live.frame_ms:.1f} ms"
        )

    def closeEvent(self, ev) -> None:
        self.measure.shutdown()
        self.camera_tab.shutdown()
        self.state.close_source()
        super().closeEvent(ev)
