"""The frequency floor: linear axes that start at 0 Hz, log axes down to 2 Hz, sweeps and decay bands below 20 Hz."""

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from scipy.signal import fftconvolve

from helpers import spin, spin_until
from roomscape.audio.devices import SIM_DEVICE
from roomscape.core.calibration import Calibration
from roomscape.core.decay import analyze_decay, csd_waterfall
from roomscape.core.measurement import analyze_recording, frequency_response
from roomscape.core.spectrogram import LINEAR_MODES, SpectrogramEngine, SpectrogramSettings
from roomscape.core.spectrum import RtaEngine, RtaSettings
from roomscape.core.sweep import ImpulseResponse, SweepDeconvolver, SweepSpec, recording_length, sweep_signal
from test_measurement_e2e import known_system_recording

FS = 48000


def tone(f, seconds=2.0, amp=0.1, dc=0.0, noise_dbfs=-100.0, fs=FS, seed=0):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t) + dc + 10 ** (noise_dbfs / 20) * np.random.default_rng(seed).standard_normal(t.size)


# ------------------------------------------------------------------ analyzer and spectrogram
def test_rta_fft_view_runs_from_the_first_bin_above_dc_and_ignores_a_dc_offset():
    cfg = RtaSettings(fft_size=65536, window="Flat-top", averaging_s=0.0, resolution="FFT", f_min=0.0, f_max=300.0, weighting="A")
    eng = RtaEngine(FS, cfg)
    fr = eng.process(tone(7.0, dc=0.3), 0.1)
    assert 0.0 < fr.freqs[0] < 1.0 and fr.freqs[-1] <= 300.0 and np.all(np.diff(fr.freqs) > 0)
    assert np.all(np.isfinite(fr.avg_db))                                    # A-weighting is finite all the way down
    flat = RtaEngine(FS, RtaSettings(fft_size=65536, window="Flat-top", averaging_s=0.0, resolution="FFT", f_min=0.0, f_max=300.0))
    z = flat.process(tone(7.0, dc=0.3), 0.1)
    k = int(np.argmax(z.avg_db))
    assert z.freqs[k] == pytest.approx(7.0, abs=0.75)
    assert z.avg_db[k] == pytest.approx(20 * np.log10(0.1 / np.sqrt(2)), abs=0.2)     # the 0.3 DC offset is removed, the tone is exact
    assert z.avg_db[:3].max() < -40.0                                         # a 0.3 DC offset would read +9.5 dBFS here: it is gone


def test_rta_log_axis_reaches_2_hz_with_finite_levels():
    eng = RtaEngine(FS, RtaSettings(fft_size=65536, averaging_s=0.0, resolution="1/12", f_min=2.0, f_max=20000.0))
    fr = eng.process(tone(5.0), 0.1)
    assert fr.freqs[0] == pytest.approx(2.0, rel=0.02) and np.all(np.isfinite(fr.avg_db))
    bars = RtaEngine(FS, RtaSettings(fft_size=65536, averaging_s=0.0, resolution="1/3", f_min=2.0, f_max=20000.0)).process(tone(5.0), 0.1)
    assert bars.freqs[0] < 3.0 and bars.kind == "bars"


@pytest.mark.parametrize("fs", [44100, 48000])
def test_linear_spectrogram_modes_start_at_zero_and_find_the_tone_bin(fs):
    for name, (top, _n) in LINEAR_MODES.items():
        eng = SpectrogramEngine(fs, SpectrogramSettings(mode=name))
        assert eng.linear and eng.grid[0] == pytest.approx(fs / eng._plan[0][0]) and eng.grid[-1] <= top    # first bin above DC
        assert np.allclose(np.diff(eng.grid), np.diff(eng.grid)[0])           # evenly spaced
    eng = SpectrogramEngine(fs, SpectrogramSettings(mode="Linear 0-300 Hz"))
    col = eng.compute(tone(40.0, seconds=3.0, fs=fs))
    assert eng.grid[int(np.argmax(col))] == pytest.approx(40.0, abs=1.0)
    assert eng.required_samples >= 60000
    log = SpectrogramEngine(fs, SpectrogramSettings(mode="Balanced", f_min=2.0, f_max=20000.0))
    assert not log.linear and log.grid[0] == pytest.approx(2.0) and np.all(np.isfinite(log.compute(tone(5.0, seconds=3.0, fs=fs))))


# ------------------------------------------------------------------ measurement: waterfall, response, decay bands, sweeps
def damped_ir(components, fs=FS, seconds=4.0, direct=1.0):
    """Impulse response: a direct impulse plus damped sinusoids ``(f_hz, t60_s, amplitude)``."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    h = np.zeros(n)
    h[0] = direct
    for f, t60, a in components:
        h += a * np.sin(2 * np.pi * f * t) * np.exp(-6.9078 * t / t60)
    return h


def test_linear_csd_waterfall_starts_at_zero_hz_and_shows_the_decay_of_a_12_hz_resonance():
    ir = ImpulseResponse.from_array(damped_ir([(12.0, 1.5, 0.02)]), FS, 0)
    wf = csd_waterfall(ir, n_slices=20, dt_s=0.05, window_s=1.5, f_lo=0.0, f_hi=100.0, scale="linear")
    assert wf.freqs[0] == 0.0 and wf.freqs[-1] == pytest.approx(100.0, abs=0.5) and np.allclose(np.diff(wf.freqs), np.diff(wf.freqs)[0])
    k = int(np.argmax(wf.slices_db[0, 2:])) + 2                                  # (ignore the direct impulse's broadband floor)
    assert wf.freqs[k] == pytest.approx(12.0, abs=1.5)
    slope = np.polyfit(wf.times_s[:12], wf.slices_db[:12, k], 1)[0]               # dB per second
    assert slope == pytest.approx(-60.0 / 1.5, rel=0.3)
    log = csd_waterfall(ir, n_slices=5, window_s=0.8, f_lo=20.0, f_hi=200.0)
    assert log.freqs[0] == pytest.approx(20.0)                                  # the default (log) behaviour is unchanged


def test_linear_frequency_response_matches_the_log_one_where_they_overlap():
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=1.0, level_dbfs=-12)
    res = analyze_recording(known_system_recording(spec, gain=0.5), spec, Calibration())
    lin = frequency_response(res.ir, spec, Calibration(), window_s=0.005, f_lo=0.0, f_hi=1000.0, scale="linear")
    log = frequency_response(res.ir, spec, Calibration(), window_s=0.005)
    assert lin.scale == "linear" and lin.freqs[0] == 0.0 and 1000.0 - 1e-6 <= lin.freqs[-1] <= 1000.0 + np.diff(lin.freqs)[0] and log.scale == "log"
    i, j = int(np.argmin(np.abs(lin.freqs - 500))), int(np.argmin(np.abs(log.freqs - 500)))
    assert lin.level_db[i] == pytest.approx(log.level_db[j], abs=0.5)
    assert np.all(np.isfinite(lin.level_db))


def test_octave_decay_bands_continue_below_31_hz_when_asked():
    ir = ImpulseResponse.from_array(damped_ir([(8.0, 1.2, 0.05), (16.0, 0.9, 0.05), (63.0, 0.6, 0.05)], seconds=5.0), FS, 0)
    d = analyze_decay(ir, bands_per_octave=1, f_lo=7.0, f_hi=100.0)
    centers = d.centers
    assert centers[0] == pytest.approx(7.8, abs=0.3) and np.any(np.isclose(centers, 15.6, atol=0.3)) and np.any(np.isclose(centers, 62.5, atol=0.5))
    near = lambda f: min(d.bands, key=lambda b: abs(np.log(b.fc / f)))              # noqa: E731
    assert near(8.0).t60 == pytest.approx(1.2, rel=0.2)
    assert near(16.0).t60 == pytest.approx(0.9, rel=0.2)
    assert near(63.0).t60 == pytest.approx(0.6, rel=0.2)


def test_a_sweep_that_starts_at_5_hz_is_recovered_from_15_hz_up_and_adds_low_decay_bands():
    spec = SweepSpec(fs=FS, f1=5.0, duration=6.0, tail_s=1.5, level_dbfs=-12)
    x = sweep_signal(spec)
    first = x[: int(0.4 * FS)]
    zc = np.count_nonzero(np.diff(np.signbit(first[int(0.01 * FS):])))
    assert zc / 2 / 0.39 < 12.0                                                    # at the start the sweep is still in the single-digit Hz
    h = np.zeros(2000)
    h[100] = 0.5
    sig = fftconvolve(x.astype(float), h)
    rec = np.zeros(recording_length(spec))
    m = min(sig.size, rec.size - 6000)
    rec[6000:6000 + m] = sig[:m]
    ir = SweepDeconvolver(spec).extract(rec)
    n = 1 << 16
    H = np.abs(np.fft.rfft(ir.h[ir.t0 - 100: ir.t0 - 100 + 2000], n))
    f = np.fft.rfftfreq(n, 1 / FS)
    sel = (f > 15) & (f < 2000)
    assert np.abs(20 * np.log10(H[sel] / 0.5)).max() < 0.6                         # unity (0.5) gain from 3 x the start frequency up
    res = analyze_recording(known_system_recording(spec, gain=0.5), spec, Calibration())
    assert res.decay.centers[0] < 10.0                                             # decay bands below 10 Hz were produced
    plain = analyze_recording(known_system_recording(SweepSpec(fs=FS, duration=2.0, tail_s=1.0), gain=0.5),
                              SweepSpec(fs=FS, duration=2.0, tail_s=1.0), Calibration())
    assert plain.decay.centers[0] > 25.0                                           # the default start keeps the familiar bands


# ------------------------------------------------------------------ the windows
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

    st = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)
    w = MainWindow(AppState(st))
    w.resize(1500, 900)
    w.show()
    yield w
    w.close()


def test_live_tab_linear_axis_from_0_hz_and_log_axis_down_to_2_hz(win):
    st, live = win.state, win.live
    st.open_source(SIM_DEVICE, 48000)
    spin(600)
    assert live.plot.scale == "log" and live.res.isEnabled()
    live.axis.setCurrentIndex(2)                                                   # Linear 0 - 300 Hz
    assert live.plot.scale == "linear" and live.plot.f_min == 0.0 and live.plot.f_max == 300.0
    assert int(live.fft.currentData()) >= 65536 and not live.res.isEnabled() and not live.sp_mode.isEnabled()
    assert spin_until(lambda: live.plot._last is not None and live.plot._last.freqs[-1] <= 300.0, 5)
    fr = live.plot._last
    assert 0.0 < fr.freqs[0] < 1.0 and fr.kind == "line"
    assert live.spec_engine.linear and live.spec_engine.grid[-1] <= 300.0 and live.waterfall.freq_scale == "linear"
    vr = live.plot.pi.viewRange()[0]
    assert vr[0] == pytest.approx(0.0, abs=1.0) and vr[1] == pytest.approx(300.0, abs=2.0)           # the picture really starts at 0 Hz
    spin(500)
    live.lower.setCurrentIndex(1)                                                  # the 3D waterfall in linear mode
    spin(700)
    assert not win.live.waterfall.grab().isNull() and live.waterfall.last_paint_ms > 0
    live.lower.setCurrentIndex(0)
    live.axis.setCurrentIndex(1)                                                   # Log 2 Hz - 20 kHz
    assert live.plot.scale == "log" and live.res.isEnabled() and live.sp_mode.isEnabled()
    assert spin_until(lambda: live.plot._last is not None and live.plot._last.freqs[0] < 3.0, 5)
    assert live.spec_engine.grid[0] == pytest.approx(2.0) and not live.spec_engine.linear
    live.axis.setCurrentIndex(0)
    assert spin_until(lambda: live.plot._last is not None and live.plot._last.freqs[0] >= 19.0, 5)


def test_measure_tab_linear_waterfall_and_response_axes_and_the_sweep_start(win):
    from test_export import make_result

    st, mt = win.state, win.measure
    st.add_measurement(make_result())
    assert spin_until(lambda: mt.result is not None, 10)
    mt.wf_preset.setCurrentText("Linear 0-300 Hz (800 ms window)")
    mt.rs_axis.setCurrentIndex(2)
    assert spin_until(lambda: mt.wf3d.freq_scale == "linear" and mt.response_plot._fr is not None and mt.response_plot._fr.scale == "linear", 5)
    assert mt.wf2d._scale == "linear" and mt.wf3d.freqs[0] == 0.0
    assert mt.response_plot._fr.freqs[0] == 0.0 and 299.0 <= mt.response_plot._fr.freqs[-1] <= 301.0
    vr = mt.response_plot.pi.viewRange()[0]
    assert vr[0] == pytest.approx(0.0, abs=1e-6)
    mt.pages.setCurrentIndex(1)
    spin(500)
    assert not mt.wf3d.grab().isNull()
    mt.wf_preset.setCurrentIndex(0)
    mt.rs_axis.setCurrentIndex(1)                                                  # log, down to 2 Hz
    assert spin_until(lambda: mt.wf3d.freq_scale == "log" and mt.response_plot._fr.scale == "log", 5)
    assert mt.response_plot._fr.freqs[0] == pytest.approx(2.0)

    # the sweep start frequency reaches the measurement
    st.open_source(SIM_DEVICE, 48000)
    spin(500)
    mt.f1.setCurrentIndex(2)
    mt.dur.setCurrentIndex(0)
    assert mt._start()
    assert mt.worker.spec.f1 == 5.0
    mt.shutdown()


def test_modes_tab_and_camera_overlay_reach_below_20_hz(win):
    st = win.state
    st.set_room(st.room.with_(dims=type(st.room.dims)(12.0, 4.0, 2.5)))               # a 12 m room: the first mode is at ~14 Hz
    mo = win.modes
    mo.recompute()
    assert min(m.freq for m in mo.modes) < 16.0
    vr = mo.map.pi.viewRange()[0]
    assert vr[0] == pytest.approx(0.0, abs=1e-6)
    ct = win.camera_tab
    ct.freq.setValue(5.0)
    assert ct.freq.value() == 5.0 and ct.freq_slider.value() == 10
    assert win.mapping.fl.minimum() <= 5.0
