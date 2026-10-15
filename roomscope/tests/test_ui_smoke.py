"""Headless GUI smoke tests (Qt offscreen): the windows build, run against the simulated mic and don't raise."""

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from roomscope.audio.devices import SIM_DEVICE
from roomscope.core.calibration import parse_calibration_text


@pytest.fixture(scope="module")
def qapp():
    from roomscope.ui.theme import apply_theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app)
    return app


@pytest.fixture
def window(qapp, tmp_path):
    from roomscope.ui.main_window import MainWindow
    from roomscope.ui.state import AppState

    st = AppState(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    win = MainWindow(st)
    win.show()
    yield win
    st.close_source()
    win.close()


def spin(ms: int) -> None:
    QtTest.QTest.qWait(ms)


def test_live_tab_runs_against_simulated_mic(window):
    window.state.open_source(SIM_DEVICE, 48000)
    spin(900)
    live = window.live
    assert live.plot._last is not None and live.plot._last.avg_db.size > 100
    assert live.meter_widget._level > -150
    assert live.fps > 5
    assert window.state.source.xruns == 0
    window.state.close_source()
    assert not window.state.running


def test_resolution_modes_render(window):
    window.state.open_source(SIM_DEVICE, 48000)
    live = window.live
    for res in ("FFT", "1/1", "1/3", "1/48", "1/12"):
        live.res.setCurrentText(res)
        spin(150)
        assert live.plot._last is not None and live.plot._last.kind == ("bars" if res in ("1/1", "1/3") else "line")


def test_calibration_switches_units(window):
    window.state.open_source(SIM_DEVICE, 48000)
    spin(300)
    assert window.live.plot._last.unit == "dBFS"
    window.state.set_spl_offset(110.0)
    spin(300)
    assert window.live.plot._last.unit == "dB SPL"
    assert window.live.top.value() == 110.0
    window.state.calibration.mic is None
    window.state.set_spl_offset(None)
    spin(200)
    assert window.live.plot._last.unit == "dBFS"


def test_mic_calibration_file_load_and_persist(window, tmp_path):
    f = tmp_path / "cal.txt"
    f.write_text('"test mic"\n20 0.0\n1000 0.5\n20000 -2.0\n')
    window.state.load_mic_cal(str(f))
    assert window.state.calibration.mic is not None
    assert "cal" in window.device_bar.cal_label.text()
    from roomscope.ui.state import AppState

    again = AppState(window.state.settings)                  # same settings store -> restored
    assert again.calibration.mic is not None and again.calibration.mic.freqs.size == 3


def test_spectrogram_and_waterfall_update_live(window):
    window.state.open_source(SIM_DEVICE, 48000)
    live = window.live
    spin(1200)
    assert live.history is not None and live.history.count > 10
    assert live.spectro._data is not None
    live.lower.setCurrentIndex(1)                         # 3D waterfall
    spin(700)
    assert live.waterfall.slices is not None and live.waterfall.slices.shape[0] >= 8
    # switching mode / history rebuilds the engine without error and keeps running
    live.sp_mode.setCurrentText("Bass detail")
    live.sp_hist.setCurrentIndex(0)
    spin(500)
    assert live.spec_engine.settings.mode == "Bass detail"
    assert window.state.source.xruns == 0
