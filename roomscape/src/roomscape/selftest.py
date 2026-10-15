"""``roomscape --selftest``: a headless end-to-end check of the installed program, no hardware needed.

It drives the real windows with the simulated microphone, room, camera and a virtual walker: libraries, signal processing,
live analyzer, a sweep measurement with room decay, data export, the camera overlay and a short walk-the-room mapping run.
A report and a screenshot are written to a folder; the exit code is 0 when everything worked.  Handy right after installing
(or unpacking the Windows build) and for bug reports.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path


def _spin(ms: float) -> None:
    from PySide6 import QtWidgets  # noqa: PLC0415

    end = time.monotonic() + ms / 1000.0
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.004)


def _spin_until(pred, timeout_s: float) -> bool:
    from PySide6 import QtWidgets  # noqa: PLC0415

    end = time.monotonic() + timeout_s
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.004)
    return False


class _Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = 0

    def step(self, name: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            self.lines.append(f"[ ok ] {name} ({time.perf_counter() - t0:.1f} s)" + (f": {detail}" if detail else ""))
        except Exception as e:  # noqa: BLE001 - every failure is reported, none may stop the remaining steps
            self.failed += 1
            self.lines.append(f"[FAIL] {name}: {type(e).__name__}: {e}")
            self.lines.append(textwrap.indent("".join(traceback.format_exc(limit=5)), "        "))


def run(out_dir: str | None = None, echo: bool = True) -> int:
    """Run all checks; returns 0 if all passed.  Writes ``roomscape-selftest.txt`` and ``roomscape-selftest.png`` to ``out_dir``."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"                       # no screen needed (a no-op if Qt is already running)
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="roomscape-selftest-"))
    out.mkdir(parents=True, exist_ok=True)
    rep = _Report()
    ctx: dict = {}
    t_start = time.perf_counter()

    def libraries() -> str:
        import importlib  # noqa: PLC0415

        vers = []
        for mod in ("numpy", "scipy", "cv2", "PySide6", "pyqtgraph", "sounddevice"):
            m = importlib.import_module(mod)
            vers.append(f"{mod} {getattr(m, '__version__', '')}".strip())
        import sounddevice as sd  # noqa: PLC0415

        vers.append(f"PortAudio {sd.get_portaudio_version()[1]}")
        return ", ".join(vers)

    def dsp() -> str:
        from roomscape.diagnostics import check_dsp  # noqa: PLC0415

        c = check_dsp()[0]
        if c.status != "ok":
            raise RuntimeError(c.detail)
        return c.detail

    def window() -> str:
        from PySide6 import QtCore, QtWidgets  # noqa: PLC0415

        from roomscape.ui.main_window import MainWindow  # noqa: PLC0415
        from roomscape.ui.state import AppState  # noqa: PLC0415
        from roomscape.ui.theme import apply_theme  # noqa: PLC0415

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        apply_theme(app)
        settings = QtCore.QSettings(str(out / "selftest-settings.ini"), QtCore.QSettings.Format.IniFormat)
        ctx["state"] = AppState(settings)
        ctx["win"] = win = MainWindow(ctx["state"])
        win.resize(1500, 920)
        win.show()
        _spin(300)
        return f"{win.windowTitle()}, {win.tabs.count()} tabs"

    def live() -> str:
        from roomscape.audio.devices import SIM_DEVICE  # noqa: PLC0415

        st, win = ctx["state"], ctx["win"]
        st.open_source(SIM_DEVICE, 48000)
        _spin(2500)
        if st.source.seconds_recorded < 1.5:
            raise RuntimeError("the simulated microphone delivered no audio")
        if win.live.fps < 2.0:
            raise RuntimeError(f"the live display is not updating ({win.live.fps:.1f} fps)")
        return f"{win.live.fps:.0f} fps, {st.source.xruns} dropouts"

    def measurement() -> str:
        st, win = ctx["state"], ctx["win"]
        mt = win.measure
        mt.dur.setCurrentIndex(0)
        mt.tail.setValue(1.5)
        if not mt._start():
            raise RuntimeError("the measurement could not start")
        if not _spin_until(lambda: len(st.measurements) == 1, 90):
            raise RuntimeError(mt.status.text() or "the measurement did not finish")
        mt.wait_idle()
        ctx["result"] = res = st.measurements[0]
        t_mid = res.decay.t_mid()
        if not (t_mid == t_mid) or not (0.15 < t_mid < 1.5):
            raise RuntimeError(f"implausible decay time {t_mid}")
        return f"T_mid {t_mid:.2f} s, RT60 {res.decay.broadband.t60:.2f} s, S/N {res.snr_db:.0f} dB"

    def export() -> str:
        from roomscape.core.export import export_measurement  # noqa: PLC0415

        files = export_measurement(ctx["result"], out / "export")
        if len(files) != 4 or not all(p.exists() and p.stat().st_size > 0 for p in files):
            raise RuntimeError("export wrote incomplete files")
        return ", ".join(p.suffix for p in files)

    def camera() -> str:
        win = ctx["win"]
        win.tabs.setCurrentWidget(win.camera_tab)
        ct = win.camera_tab
        ct._toggle_connect()
        if not _spin_until(lambda: ct.work_size[0] > 0, 15):
            raise RuntimeError("the simulated camera delivered no picture")
        ct._demo_pins()
        ct.fov.setValue(70.0)
        _spin(500)
        if ct.cam_model is None or not ct.rms < 3.0:
            raise RuntimeError(f"camera pose not found (fit error {ct.rms})")
        ct.freq.setValue(34.3)
        ct.process_frame(ct._work_frame(ct.camera.latest().image), ct.camera.latest())
        return f"{ct.work_size[0]}x{ct.work_size[1]}, pose fit error {ct.rms:.1f} px"

    def mapping() -> str:
        win = ctx["win"]
        mp = win.mapping
        mp.level.setValue(-20.0)
        mp._demo_walk()
        mp._toggle_stim()
        mp._toggle_mapping()
        ok = _spin_until(lambda: mp.mapper is not None and mp.mapper.counts["accepted"] >= 3, 40)
        counts = dict(mp.mapper.counts) if mp.mapper is not None else {}
        mp.source.setCurrentIndex(1)
        _spin(600)
        ctx["map_status"] = mp.map_status.text()
        mp._toggle_mapping()
        mp._toggle_stim()
        if not ok:
            raise RuntimeError(f"too few mapped frames: {counts}")
        return f"{counts['accepted']} frames mapped, coverage {mp.mapper.field.coverage() * 100:.0f} %"

    def screenshot() -> str:
        win = ctx["win"]
        path = out / "roomscape-selftest.png"
        if not win.grab().save(str(path)) or path.stat().st_size < 10_000:
            raise RuntimeError("could not render the window")
        return str(path)

    def shutdown() -> str:
        win, st = ctx["win"], ctx["state"]
        win.measure.shutdown()
        win.camera_tab.shutdown()
        st.close_source()
        win.close()
        _spin(100)
        return ""

    for name, fn in (("Libraries", libraries), ("Signal processing", dsp), ("Main window", window), ("Live analyzer (simulated mic)", live),
                     ("Sweep measurement + room decay", measurement), ("Export", export), ("Camera overlay (simulated camera)", camera),
                     ("Walk-the-room mapping (virtual walker)", mapping), ("Screenshot", screenshot), ("Shutdown", shutdown)):
        if name not in ("Libraries", "Signal processing", "Main window") and "win" not in ctx:
            rep.lines.append(f"[skip] {name}")
            continue
        if name in ("Export",) and "result" not in ctx:
            rep.lines.append(f"[skip] {name}")
            continue
        rep.step(name, fn)

    verdict = "ALL CHECKS PASSED" if rep.failed == 0 else f"{rep.failed} CHECK(S) FAILED"
    text = "\n".join(["RoomScape self-test", *rep.lines, "", f"{verdict} in {time.perf_counter() - t_start:.0f} s", f"Files: {out}"]) + "\n"
    (out / "roomscape-selftest.txt").write_text(text, encoding="utf-8")
    if echo and sys.stdout is not None:
        print(text)
    return 0 if rep.failed == 0 else 1
