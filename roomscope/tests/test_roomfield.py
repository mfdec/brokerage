import numpy as np
import pytest

from roomscope.core.measurement import analyze_recording
from roomscope.core.modes import Mode, RoomDims, room_modes
from roomscope.core.roomfield import (
    Peak, RoomConfig, excitation_db, find_response_peaks, match_peaks, modal_field_db, predicted_response_db,
)
from roomscope.sim.world import SimWorld

CFG = RoomConfig(RoomDims(5.0, 4.0, 2.5), source=(0.0, 0.0, 0.0), listener=(5.0, 4.0, 2.5), t60=0.5)


def test_corner_to_corner_excites_every_mode_equally_in_magnitude():
    modes = CFG.modes(120)
    exc = excitation_db(CFG, modes)
    # in the opposite corners |psi| = 1 for every mode, so only the Lambda normalisation differs (0..~12 dB)
    assert exc.max() == pytest.approx(0.0) and exc.min() > -13.0


def test_modes_with_a_node_at_the_speaker_are_barely_excited():
    cfg = CFG.with_(source=(2.5, 2.0, 1.25), listener=(0.4, 0.3, 0.4))               # speaker at the room centre
    modes = cfg.modes(120)
    exc = excitation_db(cfg, modes)
    lookup = {m.n: e for m, e in zip(modes, exc)}
    assert lookup[(1, 0, 0)] < -60 and lookup[(0, 1, 0)] < -60 and lookup[(0, 0, 1)] < -60     # odd axial modes: node at centre
    assert lookup[(2, 0, 0)] > -15                                                              # even modes: antinode at centre


def test_isolated_mode_field_follows_the_mode_shape():
    cfg = CFG.with_(t60=4.0, source=(0.0, 0.0, 0.0))                                  # high Q: one mode dominates
    f_n = cfg.c / (2 * 5.0)
    xs, ys, db = modal_field_db(cfg, f_n, z=0.0, n=(80, 60), modes=[Mode((1, 0, 0), f_n)])
    assert db.shape == (60, 80) and np.median(db) == pytest.approx(0.0, abs=1e-9)
    lin = 10 ** (db / 20)
    shape = np.abs(np.cos(np.pi * xs / 5.0))[None, :] * np.ones((60, 1))
    assert np.corrcoef(lin.ravel(), shape.ravel())[0, 1] > 0.99
    assert db[:, 0].mean() > db[:, 40].mean() + 15                                  # wall = antinode, mid-room = node


def test_predicted_response_peaks_sit_on_the_modes():
    cfg = RoomConfig(RoomDims(5.0, 4.0, 2.5), source=(0.3, 0.3, 0.3), listener=(4.5, 3.5, 2.0), t60=0.7)
    f = np.arange(20, 150, 0.25)
    y = predicted_response_db(cfg, f)
    peaks = find_response_peaks(f, y, 20, 150, prominence_db=6.0)
    matches = match_peaks(peaks, cfg.modes(160), cfg)
    assert len(peaks) >= 6
    assert all(m.matched and abs(m.error_hz) <= 2.0 for m in matches)


def test_peak_finder_recovers_lorentzians_and_flags_unmatched():
    f = np.arange(20, 200, 0.25)
    y = np.zeros_like(f)
    for f0, a, w in ((34.3, 12, 2.0), (42.9, 9, 2.0), (95.0, 8, 2.0)):               # 95 Hz is not a mode of this room
        y += a / (1 + ((f - f0) / w) ** 2)
    y += np.random.default_rng(0).normal(0, 0.3, f.size)
    peaks = find_response_peaks(f, y, 20, 200, prominence_db=3.0)
    got = sorted(round(p.freq) for p in peaks)
    assert got[:2] == [34, 43] and 95 in got
    ms = {round(m.peak.freq): m for m in match_peaks(peaks, room_modes(RoomDims(5.0, 4.0, 2.5), 200, 343.4), CFG)}
    assert ms[34].best.n == (1, 0, 0) and ms[43].best.n == (0, 1, 0)
    assert not ms[95].matched or abs(ms[95].error_hz) < 2.5


def test_measured_bass_peaks_agree_with_the_modal_model():
    """The synthetic room is built from the same modal model; after sweep -> IR -> analysis the measured
    low-frequency peaks must land on predicted modes (end-to-end consistency of model and measurement)."""
    from test_measurement_e2e import FS  # reuse
    from scipy.signal import fftconvolve

    from roomscope.core.sweep import SweepSpec, recording_length, sweep_signal

    world = SimWorld(FS, seed=1)
    spec = SweepSpec(fs=FS, duration=3.0, tail_s=1.5, level_dbfs=-12)
    h = world.ir()
    y = np.zeros(recording_length(spec))
    sig = fftconvolve(sweep_signal(spec).astype(np.float64), h)
    y[8000:8000 + min(sig.size, y.size - 8000)] = sig[: y.size - 8000]
    y += 1e-6 * np.random.default_rng(0).standard_normal(y.size)
    res = analyze_recording(y, spec)
    f, lvl = res.bass(f_hi=200)
    cfg = RoomConfig(world.room.dims, world.room.temp_c, tuple(world.room.src), tuple(world.mic), 0.5, 220.0)
    peaks = find_response_peaks(f, lvl, 25, 190, prominence_db=5.0)
    matches = match_peaks(peaks, cfg.modes(240), cfg, tol_frac=0.03, tol_min_hz=2.0)
    assert len(peaks) >= 5
    assert sum(m.matched for m in matches) / len(matches) >= 0.8, [(round(m.peak.freq, 1), m.matched) for m in matches]
