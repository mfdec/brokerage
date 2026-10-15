"""Exporting a measurement: WAV impulse response, CSV tables, JSON details."""

import csv
import json

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from scipy.io import wavfile

from helpers import spin_until
from roomscape.audio.devices import SIM_DEVICE
from roomscape.core.calibration import Calibration
from roomscape.core.export import export_measurement
from roomscape.core.measurement import analyze_recording, frequency_response
from roomscape.core.sweep import SweepSpec
from test_measurement_e2e import known_system_recording

FS = 48000


def make_result(label="@(3.0, 2.0)"):
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=1.0, level_dbfs=-12)
    return analyze_recording(known_system_recording(spec, gain=0.5), spec, Calibration(spl_offset_db=100.0), 1,
                             label=label, position=(3.0, 2.0))


def test_export_writes_a_wav_two_csvs_and_a_json_that_round_trip(tmp_path):
    res = make_result()
    files = export_measurement(res, tmp_path / "out")
    assert len(files) == 4 and all(p.exists() for p in files)
    wav, resp, rt60, info = files
    assert wav.name.endswith("_impulse.wav") and "@(3.0,_2.0)" in wav.name

    fs, h = wavfile.read(wav)
    assert fs == FS and h.dtype == np.float32 and h.size == res.ir.h.size
    assert np.allclose(h, res.ir.h, atol=1e-6) and int(np.argmax(np.abs(h))) == res.ir.t0

    rows = list(csv.reader(open(resp)))
    assert rows[0] == ["frequency_hz", "level_dB_SPL"]
    freqs = np.array([float(r[0]) for r in rows[1:]])
    lvl = np.array([float(r[1]) for r in rows[1:]])
    ref = frequency_response(res.ir, res.spec, res.calibration)
    assert np.all(np.diff(freqs) > 0) and np.allclose(lvl, ref.level_db, atol=0.0006) and freqs.size == ref.freqs.size

    rows = list(csv.reader(open(rt60)))
    assert rows[0][:7] == ["band_hz", "f_lo_hz", "f_hi_hz", "edt_s", "t20_s", "t30_s", "rt60_s"]
    assert rows[-1][0] == "broadband" and len(rows) - 2 == len(res.decay.bands)
    assert all(len(r) == len(rows[0]) for r in rows)

    d = json.loads(info.read_text())
    assert d["position_m"] == [3.0, 2.0] and d["sample_rate_hz"] == FS and d["level_unit"] == "dB SPL"
    assert d["direct_sound_index"] == res.ir.t0 and d["sweep"]["f1_hz"] == 20.0 and d["spl_offset_db"] == 100.0


def test_export_without_position_or_calibration_and_with_odd_labels(tmp_path):
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=1.0, level_dbfs=-12)
    res = analyze_recording(known_system_recording(spec), spec, label="../evil name?*")
    files = export_measurement(res, tmp_path)
    assert all(p.parent == tmp_path for p in files)                       # a label can never escape the chosen folder
    assert json.loads(files[3].read_text())["position_m"] is None
    assert json.loads(files[3].read_text())["level_unit"] == "dBFS"


@pytest.fixture(scope="module")
def qapp():
    from roomscape.ui.theme import apply_theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app)
    return app


def test_export_button_on_the_measure_tab(qapp, tmp_path, monkeypatch):
    from roomscape.ui.main_window import MainWindow
    from roomscape.ui.state import AppState

    st = AppState(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    win = MainWindow(st)
    win.show()
    try:
        mt = win.measure
        mt._export()
        assert "run a measurement" in mt.status.text()
        st.add_measurement(make_result())
        assert spin_until(lambda: mt.result is not None, 10)
        out = tmp_path / "exported"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out)))
        mt._export()
        assert len(list(out.iterdir())) == 4 and "Exported 4 files" in mt.status.text()
    finally:
        win.close()
