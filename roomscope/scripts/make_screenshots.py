"""Regenerate the README screenshots from the built-in simulation (no hardware needed, runs headless).

    python scripts/make_screenshots.py [output_dir]          # default: docs/screenshots

Takes about two minutes: it runs a simulated sweep measurement and a full virtual walk through the room.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from roomscope.audio.devices import SIM_DEVICE  # noqa: E402
from roomscope.ui.main_window import MainWindow  # noqa: E402
from roomscope.ui.state import AppState  # noqa: E402
from roomscope.ui.theme import apply_theme  # noqa: E402


def spin(ms: float) -> None:
    """Process Qt events for ``ms`` (QTest.qWait would hold the GIL and starve the audio threads)."""
    end = time.monotonic() + ms / 1000.0
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.004)


def spin_until(pred, timeout_s: float = 60.0) -> bool:
    end = time.monotonic() + timeout_s
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.004)
    return False


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "screenshots")
    out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication([])
    apply_theme(app)
    settings = QtCore.QSettings(os.path.join(tempfile.mkdtemp(), "shots.ini"), QtCore.QSettings.Format.IniFormat)
    st = AppState(settings)
    win = MainWindow(st)
    win.resize(1500, 920)
    win.show()
    st.open_source(SIM_DEVICE, 48000)

    def shot(name: str) -> None:
        win.grab().save(str(out / name))
        print("saved", name)

    # ---- Live tab
    spin(9000)
    shot("01_live_spectrogram.png")
    win.live.lower.setCurrentIndex(1)
    spin(2500)
    shot("02_live_waterfall.png")

    # ---- Measure / decay
    win.tabs.setCurrentWidget(win.measure)
    mt = win.measure
    mt.dur.setCurrentIndex(1)
    mt.tail.setValue(2.5)
    mt._start()
    spin_until(lambda: len(st.measurements) == 1, 120)
    spin_until(lambda: mt.worker is None, 10)
    spin(800)
    for i, name in enumerate(("03_decay_rt60.png", "04_decay_waterfall.png", "05_frequency_response.png")):
        mt.pages.setCurrentIndex(i)
        spin(600)
        shot(name)

    # ---- Room modes
    win.tabs.setCurrentWidget(win.modes)
    spin(800)
    shot("06_room_modes.png")

    # ---- Camera: modelled sound field on the picture
    win.tabs.setCurrentWidget(win.camera_tab)
    ct, mp = win.camera_tab, win.mapping
    ct._toggle_connect()
    spin_until(lambda: ct.work_size[0] > 0, 15)
    ct._demo_pins()
    ct.fov.setValue(70.0)
    ct.freq.setValue(34.3)
    spin(900)
    shot("07_camera_model_overlay.png")

    # ---- Walk-the-room map (virtual walker, 32 s)
    mp.level.setValue(-20)
    mp._demo_walk()
    mp._toggle_stim()
    mp._toggle_mapping()
    world = st.world
    mp.source.setCurrentIndex(1)
    spin_until(lambda: world.time_s > 0.55 * world.walker.duration, 60)
    spin(300)
    shot("08_walk_mapping_in_progress.png")
    spin_until(lambda: world.time_s > world.walker.duration + 1.5, 60)
    spin(600)
    shot("09_walk_map_result.png")
    mp.source.setCurrentIndex(2)
    spin(500)
    shot("10_walk_map_vs_model.png")
    sc = ct.findChild(QtWidgets.QScrollArea)
    sc.verticalScrollBar().setValue(sc.verticalScrollBar().maximum())
    spin(300)
    shot("11_walk_controls.png")

    mp._toggle_mapping()
    mp._toggle_stim()
    ct.shutdown()
    st.close_source()
    return 0


if __name__ == "__main__":
    sys.exit(main())
