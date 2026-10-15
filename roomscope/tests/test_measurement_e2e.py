import threading
import time

import numpy as np
import pytest
from scipy.signal import fftconvolve

from roomscope.audio.engine import SimulatedInput
from roomscope.audio.measure import MeasurementCancelled, record_sweep
from roomscope.core.calibration import Calibration
from roomscope.core.measurement import MeasurementError, analyze_recording, frequency_response
from roomscope.core.sweep import SweepSpec, recording_length, sweep_signal
from roomscope.sim.world import SimPlayer, SimWorld

FS = 48000


class NullPlayer:
    """A 'speaker' that is off: nothing is played."""

    def play(self, signal, fs): pass
    def stop(self): pass
    playing = False


def known_system_recording(spec, gain=0.5, delay=3000, noise_dbfs=-100):
    h = np.zeros(delay + 10)
    h[delay] = gain
    y = np.zeros(recording_length(spec))
    sig = fftconvolve(sweep_signal(spec).astype(np.float64), h)
    n = min(sig.size, y.size - 4000)
    y[4000:4000 + n] = sig[:n]
    return y + 10 ** (noise_dbfs / 20) * np.random.default_rng(1).standard_normal(y.size)


def test_simulated_measurement_end_to_end_recovers_the_room():
    world = SimWorld(FS)
    src = SimulatedInput(world, FS)
    src.start()
    try:
        time.sleep(0.8)
        spec = SweepSpec(fs=FS, duration=2.0, tail_s=1.2, level_dbfs=-12)
        rec = record_sweep(src, SimPlayer(world), spec)
    finally:
        src.stop()
    res = analyze_recording(rec, spec)
    expected = int(0.5 * FS) + world.latency + int(world.room.distance(world.mic) / world.room.c * FS)
    assert abs(res.ir.latency_samples - expected) < 2500            # pre-roll + simulated output latency + flight time
    assert not res.warnings and res.snr_db > 60
    for b in res.decay.bands:
        if 250 <= b.fc <= 4000:
            assert b.t60 == pytest.approx(float(world.room.t60(b.fc)), rel=0.12), (b.fc, b.t60, b.notes)
    assert res.decay.t_mid() == pytest.approx(0.39, rel=0.12)
    assert "RT60" in res.summary() and "T_mid" in res.summary()


def test_response_level_is_calibrated_and_offset_applies_exactly():
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=0.8, level_dbfs=-12)
    y = known_system_recording(spec, gain=0.5)
    plain = analyze_recording(y, spec)
    i = int(np.argmin(np.abs(plain.response.freqs - 1000)))
    # a 0.5-gain system fed a sine at the sweep amplitude reads 20log10(0.5 * A/sqrt(2)) dBFS
    assert plain.response.level_db[i] == pytest.approx(20 * np.log10(0.5 * spec.amplitude / np.sqrt(2)), abs=0.3)
    assert plain.response.unit == "dBFS"
    cal = analyze_recording(y, spec, Calibration(spl_offset_db=100.0))
    assert cal.response.unit == "dB SPL"
    assert np.allclose(cal.response.level_db - plain.response.level_db, 100.0, atol=1e-9)
    # window length changes the view but not a flat system's level
    short = frequency_response(plain.ir, spec, Calibration(), window_s=0.005)
    assert short.level_db[i] == pytest.approx(plain.response.level_db[i], abs=0.3)


def test_waterfall_levels_follow_the_response_offset():
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=0.8)
    res = analyze_recording(known_system_recording(spec), spec)
    wf0 = res.waterfall(n_slices=5, dt_s=0.01, window_s=0.2, f_lo=50, f_hi=500)
    res2 = analyze_recording(known_system_recording(spec), spec, Calibration(spl_offset_db=90.0))
    wf1 = res2.waterfall(n_slices=5, dt_s=0.01, window_s=0.2, f_lo=50, f_hi=500)
    assert np.allclose(wf1.slices_db - wf0.slices_db, 90.0, atol=1e-6)


def test_silence_is_reported_as_no_sweep_found():
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=0.5)
    quiet = 1e-5 * np.random.default_rng(0).standard_normal(recording_length(spec))
    with pytest.raises(MeasurementError, match="No clear sweep"):
        analyze_recording(quiet, spec)


def test_clipping_and_low_snr_produce_warnings():
    spec = SweepSpec(fs=FS, duration=2.0, tail_s=0.8)
    y = known_system_recording(spec, gain=0.5)
    clipped = analyze_recording(np.clip(y * 8, -0.9999, 0.9999), spec)
    assert any("clipped" in w for w in clipped.warnings)
    noisy = analyze_recording(known_system_recording(spec, noise_dbfs=-10), spec)      # ~34 dB S/N
    assert any("signal-to-noise" in w for w in noisy.warnings)


def test_record_requires_running_input_and_matching_rate():
    world = SimWorld(FS)
    src = SimulatedInput(world, FS)
    with pytest.raises(MeasurementError, match="Start the microphone"):
        record_sweep(src, SimPlayer(world), SweepSpec(fs=FS))
    src.start()
    try:
        with pytest.raises(MeasurementError, match="runs at"):
            record_sweep(src, SimPlayer(world), SweepSpec(fs=44100))
    finally:
        src.stop()


def test_cancel_stops_playback_and_raises():
    world = SimWorld(FS)
    src = SimulatedInput(world, FS)
    src.start()
    cancel = threading.Event()
    threading.Timer(0.6, cancel.set).start()
    try:
        time.sleep(0.5)
        with pytest.raises(MeasurementCancelled):
            record_sweep(src, SimPlayer(world), SweepSpec(fs=FS, duration=2.0), cancel=cancel)
        assert not world.playing                                   # player was stopped
    finally:
        src.stop()
