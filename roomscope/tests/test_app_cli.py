"""Command-line entry points, the built-in self-test and the icon."""

import struct
from pathlib import Path

import pytest
from PySide6 import QtWidgets

import roomscope
from roomscope import app, diagnostics, selftest


def test_version_flag_prints_the_version(capsys):
    with pytest.raises(SystemExit) as e:
        app.main(["--version"])
    assert e.value.code == 0
    assert f"RoomScope {roomscope.__version__}" in capsys.readouterr().out


def test_doctor_flag_hands_its_options_to_the_doctor(monkeypatch):
    seen = []
    monkeypatch.setattr(diagnostics, "main", lambda argv: seen.append(list(argv)) or 3)
    assert app.main(["--doctor", "--skip-dsp", "--mic"]) == 3
    assert seen == [["--skip-dsp", "--mic"]]
    assert app.main(["--doctor"]) == 3 and seen[-1] == []


def test_console_helper_is_a_no_op_where_a_console_exists():
    assert app._ensure_console() is False                       # not Windows, or stdout already present


def test_selftest_passes_end_to_end_and_writes_report_and_screenshot(tmp_path, capsys):
    rc = selftest.run(str(tmp_path), echo=False)
    report = (tmp_path / "roomscope-selftest.txt").read_text()
    assert rc == 0 and "ALL CHECKS PASSED" in report, report
    for step in ("Libraries", "Signal processing", "Main window", "Live analyzer", "Sweep measurement", "Export",
                 "Camera overlay", "Walk-the-room mapping", "Screenshot", "Shutdown"):
        assert f"[ ok ] {step}" in report, report
    assert (tmp_path / "roomscope-selftest.png").stat().st_size > 10_000
    assert len(list((tmp_path / "export").iterdir())) == 4


def test_selftest_reports_a_failing_step_without_stopping_the_others(tmp_path, monkeypatch):
    import roomscope.diagnostics as dx

    monkeypatch.setattr(dx, "check_dsp", lambda seed=0: [dx.Check(dx.FAIL, "Signal-processing self-test", "boom")])
    monkeypatch.setattr(selftest, "_spin_until", lambda pred, t: False)               # keep this test quick
    rc = selftest.run(str(tmp_path), echo=False)
    report = (tmp_path / "roomscope-selftest.txt").read_text()
    assert rc == 1 and "[FAIL] Signal processing" in report and "boom" in report
    assert "[ ok ] Libraries" in report and "[ ok ] Main window" in report and "CHECK(S) FAILED" in report


def test_icon_is_painted_in_every_size():
    from roomscope.ui.icon import SIZES, make_icon, make_pixmap

    assert QtWidgets.QApplication.instance() is not None or QtWidgets.QApplication([])
    for s in SIZES:
        img = make_pixmap(s).toImage()
        assert img.width() == s and img.pixelColor(s // 2, s // 2).alpha() > 0
        assert img.pixelColor(0, 0).alpha() == 0                      # rounded corners stay transparent
    assert not make_icon().isNull()


def test_packaged_ico_has_png_frames_for_each_size():
    from roomscope.ui.icon import SIZES

    data = Path(__file__).resolve().parent.parent.joinpath("packaging", "roomscope.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind, count) == (0, 1, len(SIZES))
    for i, size in enumerate(SIZES):
        w, h, _c, _r, planes, bits, nbytes, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        assert (w or 256) == size and (h or 256) == size and bits == 32
        assert data[offset:offset + 8] == b"\x89PNG\r\n\x1a\n" and offset + nbytes <= len(data)
