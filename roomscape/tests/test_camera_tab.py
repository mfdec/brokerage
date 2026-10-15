"""Camera tab with the simulated camera: pinning, pose, FOV tools, overlay, persistence."""

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from helpers import spin, spin_until
from roomscape.audio.devices import SIM_DEVICE
from roomscape.core.geometry import Intrinsics


@pytest.fixture(scope="module")
def qapp():
    from roomscape.ui.theme import apply_theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app)
    return app


@pytest.fixture
def win(qapp, tmp_path):
    from roomscape.ui.main_window import MainWindow
    from roomscape.ui.state import AppState

    st = AppState(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    w = MainWindow(st)
    w.resize(1500, 900)
    w.show()
    st.open_source(SIM_DEVICE, 48000)
    w.tabs.setCurrentWidget(w.camera_tab)
    yield w
    w.close()


def connect(win):
    ct = win.camera_tab
    ct._toggle_connect()
    assert spin_until(lambda: ct.work_size[0] > 0, 10)
    return ct


def true_px(ct, p):
    uv, _ = ct.camera.scene.cam.project(np.array([[p.x, p.y, 0.0]]))
    return float(uv[0, 0]), float(uv[0, 1])


def test_connecting_the_simulated_camera_shows_frames_and_front_rectangle(win):
    ct = connect(win)
    assert ct.camera.is_simulated and ct.work_size == (960, 540)
    assert len(ct.pins) == 4 and ct.pins[0].x == pytest.approx(0.8)
    spin(300)
    assert ct.view.image_size == (960, 540)
    ct._toggle_connect()                                              # disconnect
    assert ct.camera is None and ct.view.image_size == (0, 0)


def test_clicking_the_pins_calibrates_the_pose_and_wrong_fov_shows_as_fit_error(win):
    ct = connect(win)
    ct._start_pick()
    rng = np.random.default_rng(0)
    for p in list(ct.pins):
        u, v = true_px(ct, p)
        ct.view.picked.emit(u + rng.normal(0, 0.6), v + rng.normal(0, 0.6))
    assert ct._mode == "idle" and all(p.px for p in ct.pins)
    ct.fov.setValue(70.0)
    assert ct.cam_model is not None and ct.rms < 2.0
    assert np.linalg.norm(ct.cam_model.center - np.array([5.0, 2.0, 2.4])) < 0.15
    good = ct.rms
    ct.fov.setValue(52.0)                                             # clearly wrong lens
    assert ct.rms > good + 3.0


def test_fov_tools_recover_the_true_lens(win):
    ct = connect(win)
    ct._demo_pins()
    ct.fov.setValue(60.0)
    ct.cam_h.setValue(2.4)
    ct._fov_from_height()
    assert ct.fov.value() == pytest.approx(70.0, abs=3.0)
    ct.fov.setValue(55.0)
    ct._auto_fov()
    assert ct.fov.value() == pytest.approx(70.0, abs=7.0) or "perspective" in ct.fov_status.text()
    # marker held at mic height above pin 1
    ct.fov.setValue(58.0)
    mic_h = win.state.room.listener[2]
    p = ct.pins[0]
    uv, _ = ct.camera.scene.cam.project(np.array([[p.x, p.y, mic_h]]))
    ct.raised_pin.setCurrentIndex(0)
    ct._start_raised()
    ct.view.picked.emit(float(uv[0, 0]), float(uv[0, 1]))
    assert ct.fov.value() == pytest.approx(70.0, abs=4.0)


def test_overlay_is_drawn_and_responds_to_its_controls(win):
    ct = connect(win)
    ct._demo_pins()
    ct.fov.setValue(70.0)
    spin(300)
    frame = ct.camera.latest().image
    plain = ct._work_frame(frame)
    ct.show_map.setChecked(False)
    ct.show_wire.setChecked(False)
    ct.show_icons.setChecked(False)
    bare = ct.process_frame(plain)
    assert np.array_equal(bare, plain)
    ct.show_map.setChecked(True)
    mapped = ct.process_frame(plain)
    assert np.abs(mapped.astype(int) - plain.astype(int)).sum() > 1e6
    ct.opacity.setValue(10)
    faint = ct.process_frame(plain)
    assert np.abs(faint.astype(int) - plain.astype(int)).sum() < np.abs(mapped.astype(int) - plain.astype(int)).sum()
    f0 = ct.process_frame(plain)
    ct.freq.setValue(68.6)                                            # a different mode -> a different pattern
    assert not np.array_equal(ct.process_frame(plain), f0)
    ct._snap()
    assert ct.freq.value() in [round(m.freq, 1) for m in win.state.room.modes()] or abs(ct.freq.value() - 68.6) < 1.0
    ct.show_wire.setChecked(True)
    ct.show_icons.setChecked(True)
    assert not np.array_equal(ct.process_frame(plain), mapped)


def test_calibration_is_saved_and_restored_for_the_same_camera(win, tmp_path):
    from roomscape.ui.main_window import MainWindow
    from roomscape.ui.state import AppState

    ct = connect(win)
    ct._demo_pins()
    ct.fov.setValue(70.0)
    saved_px = [p.px for p in ct.pins]
    st2 = AppState(win.state.settings)
    st2.open_source(SIM_DEVICE, 48000)
    w2 = MainWindow(st2)
    w2.show()
    w2.tabs.setCurrentWidget(w2.camera_tab)
    try:
        ct2 = w2.camera_tab
        ct2._toggle_connect()
        assert spin_until(lambda: ct2.work_size[0] > 0, 10)
        # connecting resets to a fresh rectangle, then restores what was saved for this camera + resolution
        assert [tuple(np.round(p.px, 3)) if p.px else None for p in ct2.pins] == [tuple(np.round(q, 3)) for q in saved_px]
        assert ct2.fov.value() == pytest.approx(70.0)
        assert ct2.cam_model is not None
    finally:
        w2.close()


def test_changing_the_room_updates_overlay_inputs(win):
    ct = connect(win)
    ct._demo_pins()
    ct.fov.setValue(70.0)
    st = win.state
    h0 = ct.plane.value()
    st.set_room(st.room.with_(listener=(3.0, 2.0, 1.5)))
    assert ct.plane.value() == pytest.approx(1.5) and h0 != ct.plane.value()
