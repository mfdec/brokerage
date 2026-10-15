"""Decay camera panel end to end in the simulation: virtual slow walk, repeating bursts, decay overlay and animation."""

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from helpers import spin, spin_until
from roomscape.audio.devices import SIM_DEVICE
from roomscape.sim.world import LawnmowerPath
from roomscape.video.sim import MARKER_BGR


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
    return ct


def test_panel_is_wired_and_refuses_politely_without_a_camera(win):
    dp, ct = win.decay, win.camera_tab
    assert dp.texture in ct.overlay_providers and dp.on_frame in ct.pre_frame_hooks and win.mapping.strip.views["decay"] is not None
    dp._toggle()
    assert not dp.running and "camera" in dp.status.text().lower()
    assert dp.texture() is None                                      # overlay "Off": the level map / model stays in charge


def test_virtual_slow_walk_maps_decay_times_and_animates_the_energy_left(win):
    ct = calibrated(win)
    dp, mp, world = win.decay, win.mapping, win.state.world
    mp.tracker.set_target_bgr(MARKER_BGR)
    world.set_mic_position((1.0, 1.2, 1.2))
    world.walker = LawnmowerPath(1.0, 2.6, 1.2, 1.9, z=1.2, speed=0.3, row_spacing=0.6, pause_s=0.0, start_delay_s=world.time_s + 0.5)
    ct.freq.setValue(63.0)
    dp._toggle()
    assert dp.running and dp.stim is not None and dp.run_btn.text().startswith("Stop")
    assert spin_until(lambda: dp.mapper.counts["accepted"] >= 6, 45), dp.mapper.counts
    assert dp.mapper.locked
    f = dp.fields()
    assert f is not None and f.n_used >= 5
    assert "Bursts used" in dp.status.text()
    # decay-time overlay
    dp.source.setCurrentIndex(1)
    assert mp.source.currentIndex() == 0                              # one overlay at a time
    tex = dp.texture()
    assert tex is not None and tex.shape[:2] == (f.edt.ny, f.edt.nx) and tex[..., 3].max() > 0
    assert "EDT" in ct.overlay_note.text()
    arr = f.edt_s()
    assert 0.1 < np.nanmedian(arr) < 2.5
    img = ct._work_frame(ct.camera.latest().image)
    shown = ct.process_frame(img.copy(), ct.camera.latest())
    assert shown.shape[:2] == (540, 960)
    # the animation
    dp._toggle_anim()
    assert dp.anim_playing and dp.source.currentIndex() == 3
    t0 = dp.anim_t
    assert spin_until(lambda: dp.anim_t > t0 + 0.2, 5)
    tex_a = dp.texture()
    dp.time_slider.setValue(120)
    assert dp.anim_t == pytest.approx(1.2)
    tex_b = dp.texture()
    assert tex_a is not None and tex_b is not None and not np.array_equal(tex_a, tex_b)
    assert "energy still to come" in ct.overlay_note.text()
    dp._toggle_anim()
    # plan view shows up
    dp._update_plan()
    assert win.mapping.strip.views["decay"].isVisibleTo(win.mapping.strip) and win.mapping.strip.views["decay"].img.image is not None
    # the level map taking over stops the bursts
    mp._toggle_stim()
    spin(300)
    assert not dp.running and "level map" in dp.status.text()
    mp._toggle_stim()
    dp.stop()


def test_save_load_and_csv_of_the_decay_map(win, tmp_path, monkeypatch):
    from roomscape.core.decaymap import DecayWalkMapper, DecaySettings

    calibrated(win)
    dp = win.decay
    dp._export_csv()
    assert "Nothing to export" in dp.status.text()
    from test_decaymap import ModalRoom, run_walk

    mapper, _ = run_walk([(1.5 + 0.3 * k, 2.0, 1.2) for k in range(6)], rm=ModalRoom())
    dp.mapper = mapper
    win.camera_tab.freq.setValue(63.0)
    path, csv_path = str(tmp_path / "d.npz"), str(tmp_path / "d.csv")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, "")))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
    dp._save()
    n = len(mapper.frames)
    dp._clear()
    assert not mapper.frames and dp.fields() is None
    dp._load()
    assert len(dp.mapper.frames) == n and dp.fields() is not None and dp.fields().n_used >= 3
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (csv_path, "")))
    dp._export_csv()
    assert open(csv_path).read().startswith("x_m,y_m,weight,edt_s,t20_s,") and "Exported" in dp.status.text()
