"""Headless GUI smoke tests (Qt offscreen): the windows build, run against the simulated mic and don't raise."""

import pytest
from PySide6 import QtCore, QtWidgets

from helpers import spin, spin_until

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


def test_measure_tab_runs_a_simulated_sweep_and_fills_every_page(window):
    window.state.open_source(SIM_DEVICE, 48000)
    spin(600)
    mt = window.measure
    window.tabs.setCurrentWidget(mt)
    assert mt.measure_btn.isEnabled()
    mt.dur.setCurrentIndex(0)                                   # 2 s quick sweep
    mt.tail.setValue(1.2)
    mt._start()
    assert spin_until(lambda: len(window.state.measurements) == 1, 60)
    spin_until(lambda: mt.worker is None, 5)
    res = window.state.measurements[0]
    assert mt.result is res and "RT60" in mt.summary.text()
    assert mt.table.rowCount() == len(res.decay.bands) + 1
    assert mt.rt60._n == len(res.decay.bands)
    assert mt.wf3d.slices is not None and mt.wf3d.slices.shape[0] == 40
    assert mt.response_plot._fr is not None and mt.response_plot._fr.level_db.size > 100
    assert mt.schroeder.curve.getData()[0] is not None
    # switching display options recomputes without error
    mt.wf_preset.setCurrentIndex(1)
    mt.bands.setCurrentIndex(1)
    spin(500)
    for i in range(mt.pages.count()):
        mt.pages.setCurrentIndex(i)
        spin(80)
        window.grab()


def test_measure_tab_disables_without_input_and_reports_why(window):
    mt = window.measure
    assert not mt.measure_btn.isEnabled()
    assert "Start the microphone" in mt.status.text()


def test_modes_tab_follows_room_edits_and_drives_the_simulated_room(window):
    st = window.state
    mo = window.modes
    window.tabs.setCurrentWidget(mo)
    spin(400)
    n0 = len(mo.modes)
    assert n0 > 50 and mo.table.rowCount() == n0 and "Schroeder" in mo.summary.text()
    first = mo.modes[0].freq
    assert first == pytest.approx(343.4 / (2 * 5.0), rel=0.01)                      # 5 m length -> 34.3 Hz
    # editing the length through the UI updates state, modes and (when running) the simulated room
    st.open_source(SIM_DEVICE, 48000)
    mo.lx.setValue(6.0)
    spin(500)
    assert st.room.dims.lx == 6.0 and st.world.room.dims.lx == 6.0
    assert mo.modes[0].freq == pytest.approx(343.4 / 12.0, rel=0.01)
    # dragging the speaker marker in the field view moves the speaker (clamped inside the room)
    mo.field.src_t.setPos(2.0, 1.5)
    mo.field._emit_positions()
    spin(400)
    assert st.room.source[:2] == pytest.approx((2.0, 1.5), abs=1e-6)
    # clicking a table row jumps the field view to that mode's frequency
    mo.table.selectRow(3)
    spin(300)
    assert mo.field_f.value() == pytest.approx(mo.modes[3].freq, abs=0.06)
    # the settings persist
    from roomscope.ui.state import AppState

    again = AppState(st.settings)
    assert again.room.dims.lx == 6.0
    st.close_source()


def test_closing_the_window_during_a_measurement_cancels_it_cleanly(window):
    window.state.open_source(SIM_DEVICE, 48000)
    spin(600)
    mt = window.measure
    mt.dur.setCurrentIndex(0)
    assert mt._start()
    worker = mt.worker
    spin(400)
    assert worker.isRunning()
    window.close()                                                # must not hang, pop a dialog or leave the thread running
    assert not worker.isRunning()
    assert window.state.source is None
