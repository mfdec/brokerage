"""Walk-the-room mapping panel, end to end in the simulation: virtual camera + virtual walker + simulated room."""

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from helpers import spin, spin_until
from roomscape.audio.devices import SIM_DEVICE
from roomscape.sim.world import LawnmowerPath


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
    w.resize(1500, 950)
    w.show()
    st.open_source(SIM_DEVICE, 48000)
    w.tabs.setCurrentWidget(w.camera_tab)
    yield w
    w.close()


def calibrated(win):
    ct = win.camera_tab
    ct._toggle_connect()
    assert spin_until(lambda: ct.work_size[0] > 0, 10)
    ct._demo_pins()
    ct.fov.setValue(70.0)
    assert ct.cam_model is not None
    return ct


def marker_px(ct, xyz):
    uv, _ = ct.camera.scene.cam.project(np.array([xyz], float))
    return float(uv[0, 0]), float(uv[0, 1])


def test_panel_is_wired_into_the_camera_tab(win):
    ct, mp = win.camera_tab, win.mapping
    assert ct.measure_tab is win.measure and ct.texture_provider == mp.texture
    assert mp.on_frame in ct.pre_frame_hooks and mp.strip.parent() is not None
    assert mp.source.currentText().startswith("Model") and ct.extra_box.count() >= 1
    # nothing to map yet: clear, human-readable guidance instead of errors
    mp._toggle_mapping()
    assert not mp.mapping and "camera" in mp.map_status.text().lower()
    mp._pick_marker()
    assert "camera" in mp.track_label.text().lower()


def test_clicking_the_marker_then_tracking_it_in_room_coordinates(win):
    ct = calibrated(win)
    mp, world = win.mapping, win.state.world
    world.set_mic_position((3.0, 2.2, 1.2))
    spin(400)
    mp._pick_marker()
    assert ct._mode == "external"
    ct.view.picked.emit(*marker_px(ct, (3.0, 2.2, 1.2)))
    assert ct._mode == "idle" and mp.tracker.target is not None
    assert spin_until(lambda: mp.track.valid, 6)
    spin(500)
    x, y = mp.track.xy
    assert np.hypot(x - 3.0, y - 2.2) < 0.10
    assert spin_until(lambda: "Tracking" in mp.track_label.text(), 3)
    # the colour is remembered for this camera and picture size
    raw = win.state.settings.value(ct._calib_key() + "/marker", "", str)
    assert raw and len(raw) > 5
    # moving the marker moves the track
    world.set_mic_position((2.0, 1.5, 1.2))
    assert spin_until(lambda: np.hypot(mp.track.xy[0] - 2.0, mp.track.xy[1] - 1.5) < 0.12, 6)


def test_virtual_walk_paints_a_map_that_matches_the_modal_ground_truth(win):
    ct = calibrated(win)
    mp, world, st = win.mapping, win.state.world, win.state
    mp._demo_walk()
    assert mp.tracker.target is not None
    # a shorter walk than the button's, to keep the test quick (the sim runs in real time)
    world.walker = LawnmowerPath(1.0, 3.0, 1.2, 2.4, z=1.2, speed=0.6, row_spacing=0.6, pause_s=0.0,
                                 start_delay_s=world.time_s + 0.5)
    mp.level.setValue(-20.0)
    mp._toggle_stim()
    assert mp.stim is not None and mp.stim.playing and mp.play_btn.text().startswith("Stop")
    mp._toggle_mapping()
    assert mp.mapping and mp.mapper is not None
    assert spin_until(lambda: world.time_s > world.walker.duration + 1.5, 40)

    m = mp.mapper
    n = sum(m.counts.values())
    assert m.counts["accepted"] >= 8 and m.accepted_fraction > 0.5, (m.counts, n)
    assert m.field.coverage() > 0.05
    assert "coverage" in mp.map_status.text()

    # measured map vs. the simulation's true standing-wave pattern at the first axial mode
    fm = m.field
    f0 = 34.3
    meas = fm.relative_db(f0 - 2.0, f0 + 2.0)
    lines = [34.0, 36.0]
    X, Y = np.meshgrid(fm.xs, fm.ys)
    pts = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, 1.2)))
    H = world.room.frequency_response_many(pts, lines)
    truth = 10 * np.log10(np.mean(np.abs(H) ** 2, axis=1)).reshape(X.shape)
    ok = np.isfinite(meas) & (fm.w >= 1.0)
    assert ok.sum() > 40
    r = np.corrcoef(meas[ok], truth[ok])[0, 1]
    assert r > 0.85, r
    rmse = np.sqrt(np.mean((meas[ok] - meas[ok].mean() - (truth[ok] - truth[ok].mean())) ** 2))
    assert rmse < 4.0, rmse

    # the picture overlay and the plan views follow the chosen source
    ct.freq.setValue(f0)
    mp.source.setCurrentIndex(1)
    tex = mp.texture()
    assert tex is not None and tex.shape[:2] == (fm.ny, fm.nx) and tex[..., 3].max() > 0
    mp.source.setCurrentIndex(2)
    assert mp.texture() is not None
    mp.source.setCurrentIndex(0)
    assert mp.texture() is None
    mp._update_plan()
    for k in ("measured", "model", "diff"):
        assert mp.strip.views[k].img.image is not None
    out = ct.process_frame(ct._work_frame(ct.camera.latest().image), ct.camera.latest())
    assert out.shape[:2] == (540, 960)

    # stop and clean up
    mp._toggle_mapping()
    assert not mp.mapping
    mp._toggle_stim()
    assert mp.stim is None and world._stim is None


def test_save_and_load_the_map(win, tmp_path, monkeypatch):
    calibrated(win)
    mp = win.mapping
    mp._new_mapper()
    mp.mapper.field.add(2.0, 2.0, np.full(mp.mapper.field.freqs.size, 1e-4), 2.0)
    path = str(tmp_path / "m.npz")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, "")))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
    mp._save()
    mp._clear()
    assert mp.mapper.field.coverage() == 0.0
    mp._export_csv()
    assert "Nothing to export" in mp.map_status.text()
    mp._load()
    assert mp.mapper.field.coverage() > 0.0
    csv_path = str(tmp_path / "m.csv")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (csv_path, "")))
    mp._export_csv()
    lines = open(csv_path).read().splitlines()
    assert lines[0].startswith("x_m,y_m,weight,") and len(lines) > 5 and "Exported" in mp.map_status.text()


def test_spot_sweep_pauses_the_map_is_tagged_with_the_position_and_resumes_the_signal(win):
    ct = calibrated(win)
    mp, world, st, mt = win.mapping, win.state.world, win.state, win.measure
    mt.dur.setCurrentIndex(0)                                     # 2 s sweep
    mt.tail.setValue(1.5)
    world.set_mic_position((3.2, 2.0, 1.2))
    spin(400)
    mp._pick_marker()
    ct.view.picked.emit(*marker_px(ct, (3.2, 2.0, 1.2)))
    assert spin_until(lambda: mp.track.valid, 6)
    mp.level.setValue(-20.0)
    mp._toggle_stim()
    mp._toggle_mapping()
    assert spin_until(lambda: mp.mapper.counts["accepted"] >= 2, 15)       # standing still: frames pile up at one spot
    before = dict(mp.mapper.counts)
    mp._sweep_here()
    assert mt.worker is not None and mp.stim is None                     # the mapping signal makes way for the sweep
    assert spin_until(lambda: "Measuring the decay" in mp.map_status.text(), 5)
    assert spin_until(lambda: len(st.measurements) == 1, 40)
    mt.wait_idle()
    assert spin_until(lambda: mp.stim is not None and mp.stim.playing, 5)  # ...and comes back afterwards
    res = st.measurements[0]
    assert res.position is not None and np.hypot(res.position[0] - 3.2, res.position[1] - 2.0) < 0.12
    assert res.label.startswith("@(") and np.isfinite(res.decay.broadband.t60)
    assert sum(mp.mapper.counts.values()) - sum(before.values()) <= 1      # the sweep audio never reached the map
    # the result is pinned on the picture, and goes away with the measurement
    img = ct._work_frame(ct.camera.latest().image)
    with_pin = ct.process_frame(img.copy(), ct.camera.latest())
    st.clear_measurements()
    without = ct.process_frame(img.copy(), ct.camera.latest())
    assert not np.array_equal(with_pin, without)
    mp._toggle_mapping()
    mp._toggle_stim()


def test_mapping_refuses_to_start_without_a_marker_colour(win):
    calibrated(win)
    mp = win.mapping
    mp._toggle_mapping()
    assert not mp.mapping and "marker" in mp.map_status.text().lower()


def test_mapping_settings_are_remembered_and_the_camera_delay_belongs_to_the_camera(win):
    from roomscape.ui.main_window import MainWindow
    from roomscape.ui.state import AppState

    calibrated(win)
    mp = win.mapping
    mp.level.setValue(-27.0)
    mp.fh.setValue(180.0)
    mp.vmax.setValue(0.6)
    mp.tilt.setCurrentIndex(1)
    mp.auto_pin.setChecked(True)
    mp.delay.setValue(240.0)
    st2 = AppState(win.state.settings)
    st2.open_source(SIM_DEVICE, 48000)
    w2 = MainWindow(st2)
    w2.show()
    w2.tabs.setCurrentWidget(w2.camera_tab)
    try:
        mp2 = w2.mapping
        assert mp2.level.value() == -27.0 and mp2.fh.value() == 180.0 and mp2.vmax.value() == pytest.approx(0.6)
        assert mp2.tilt.currentIndex() == 1 and mp2.auto_pin.isChecked() and not mp2.adapt.isChecked()
        assert mp2.delay.value() == 0.0                                   # a delay belongs to a camera: restored on connect
        w2.camera_tab._toggle_connect()
        assert spin_until(lambda: w2.camera_tab.work_size[0] > 0, 10)
        assert mp2.delay.value() == 240.0
    finally:
        w2.close()


def test_live_settings_reach_a_running_mapper_and_out_of_range_frequencies_say_so(win):
    from roomscape.video.sim import MARKER_BGR

    ct = calibrated(win)
    mp = win.mapping
    mp.tracker.set_target_bgr(MARKER_BGR)
    mp._toggle_stim()
    mp._toggle_mapping()
    assert mp.mapping
    mp.vmax.setValue(0.5)
    mp.snr.setValue(14.0)
    mp.delay.setValue(200.0)
    s = mp.mapper.s
    assert s.v_max == pytest.approx(0.5) and s.snr_min_db == 14.0 and s.camera_delay_s == pytest.approx(0.2)

    mp.source.setCurrentIndex(1)                                    # the measured map, but nothing was played up here
    ct.freq.setValue(450.0)
    tex = mp.texture()
    assert tex is not None and np.all(tex[..., 3] == 0)
    assert "covers 20-300 Hz" in ct.overlay_note.text()
    ct.freq.setValue(100.0)
    mp.texture()
    assert "covers" not in ct.overlay_note.text()
    mp.source.setCurrentIndex(0)
    assert mp.texture() is None and "Warm" in ct.overlay_note.text()
    mp._toggle_mapping()
    mp._toggle_stim()
