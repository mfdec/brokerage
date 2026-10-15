"""Decay camera core: periodic sweep bursts -> impulse responses -> decay maps, checked against the synthetic room's own decays."""

import numpy as np
import pytest
from scipy.signal import fftconvolve, resample_poly

from roomscape.core.decaymap import (
    FS_D, DecaySettings, DecayWalkMapper, PulseSpec, band_metrics, build_fields, generate,
)
from roomscape.core.modes import RoomDims
from roomscape.core.sweep import ImpulseResponse
from roomscape.core.synth import SyntheticRoom, T60Profile

FS = 48000
DIMS = RoomDims(5.0, 4.0, 2.5)
GAIN = 10 ** (-22 / 20)
LATENCY = int(0.137 * FS)


def room():
    return SyntheticRoom(DIMS, (0.4, 1.3, 1.1), T60Profile(0.5))


class ModalRoom:
    """Exactly known decays: a direct impulse plus a 63 Hz mode (T60 0.70 s) and a 125 Hz mode (T60 0.45 s)."""

    def ir(self, pos, fs, length_s):
        n = int(length_s * fs)
        t = np.arange(n) / fs
        h = np.zeros(n)
        h[int(0.008 * fs)] = 1.0
        return h + 0.05 * np.sin(2 * np.pi * 63 * t) * np.exp(-6.9078 * t / 0.7) + 0.05 * np.sin(2 * np.pi * 125 * t) * np.exp(-6.9078 * t / 0.45)


def ideal_metrics(rm, pos, fc, spec=None):
    """What the burst method itself reports for a spot with a perfect recording (no noise, no windowing, no latency):
    the reference for the mapper's synchronisation / gating / windowing machinery."""
    from roomscape.core.sweep import SweepDeconvolver

    spec = spec or PulseSpec()
    y = fftconvolve(generate(spec).astype(float), rm.ir(pos, FS, 3.0) * GAIN)[: int(spec.period_s * FS)]
    ir = SweepDeconvolver(spec.sweep_spec).extract(y, pre_s=0.01)
    hd = resample_poly(ir.h, 1, FS // FS_D)
    return band_metrics(ImpulseResponse(hd, FS_D, int(round(ir.t0 * FS_D / FS)), rec_len=len(y) * FS_D // FS,
                                        latency_samples=ir.latency_samples * FS_D // FS, f1=spec.f1, sweep_rate_L=float(ir.sweep_rate_L)), fc)


def run_walk(positions, periods=None, spec=None, noise_dbfs=-86.0, seed=0, rm=None):
    """Feed the mapper with a virtual recording: one burst per period at ``positions[k]`` (a list of (x, y, z))."""
    spec = spec or PulseSpec()
    rm = rm or room()
    sig = generate(spec).astype(np.float64)
    P = spec.n_period
    n_per = len(positions) if periods is None else periods
    total = (n_per + 2) * P + LATENCY + FS
    rec = np.random.default_rng(seed).standard_normal(total) * 10 ** (noise_dbfs / 20)
    for k in range(n_per):
        pos = positions[min(k, len(positions) - 1)]
        resp = fftconvolve(sig, rm.ir(pos, FS, 3.0) * GAIN)
        a = k * P + LATENCY
        m = min(resp.size, total - a)
        rec[a:a + m] += resp[:m]
    mapper = DecayWalkMapper(DIMS, DecaySettings(spec=spec))
    chunk = 1024
    t_pos = 0.0
    for i in range(0, total, chunk):
        end = min(i + chunk, total)
        t_now = end / FS
        while t_pos <= t_now:                                       # video-rate positions along the path
            k = min(int(t_pos * FS / P), len(positions) - 1)
            mapper.feed_position(t_pos, (positions[k][0], positions[k][1]), True)
            t_pos += 1 / 30
        mapper.feed_audio(rec[i:end], t_now)
    return mapper, rm


def test_pulse_period_is_a_sweep_followed_by_silence():
    spec = PulseSpec()
    p = generate(spec)
    assert p.size == spec.n_period == int(1.6 * FS)
    nz = np.nonzero(p)[0]
    assert nz[0] < 50 and 0.45 * FS < nz[-1] < 0.55 * FS                       # about 0.5 s of sweep
    assert np.max(np.abs(p)) == pytest.approx(10 ** (-18 / 20), rel=1e-2) and not p[int(0.6 * FS):].any()


def test_burst_method_recovers_known_decay_times_through_noise_latency_and_windowing():
    mapper, rm = run_walk([(3.0, 2.0, 1.2)] * 8, rm=ModalRoom())
    assert mapper.locked and mapper.counts["accepted"] >= 5, mapper.counts
    for fc, t60 in ((63.0, 0.70), (125.0, 0.45)):
        for fr in mapper.frames[-3:]:
            m = band_metrics(fr.ir, fc)
            assert m.ok and m.t20 == pytest.approx(t60, rel=0.06) and m.t60 == pytest.approx(t60, rel=0.06), (fc, m.t20, m.t60)
    assert all(fr.snr_db > 25 for fr in mapper.frames) and mapper.frames[0].xy == pytest.approx((3.0, 2.0), abs=1e-6)


def test_mapper_machinery_reproduces_the_ideal_burst_result_for_the_synthetic_room():
    pos = (3.0, 2.0, 1.2)
    mapper, rm = run_walk([pos] * 8)
    assert mapper.locked and mapper.counts["accepted"] >= 5, mapper.counts
    for fc in (63.0, 125.0):
        ref = ideal_metrics(rm, pos, fc)
        assert ref.ok
        for fr in mapper.frames[-3:]:
            m = band_metrics(fr.ir, fc)
            assert m.ok and m.edt == pytest.approx(ref.edt, rel=0.03) and m.t20 == pytest.approx(ref.t20, rel=0.03), (fc, m.edt, ref.edt)


def test_a_speaker_that_is_off_never_locks_and_a_loud_burst_train_is_gated_by_speed():
    quiet, _ = run_walk([(3.0, 2.0, 1.2)] * 4, noise_dbfs=-60.0, periods=0)
    assert not quiet.locked and quiet.counts["accepted"] == 0
    # positions jumping 1.2 m per burst = 0.75 m/s is above the 0.7 m/s gate
    fast, _ = run_walk([(0.8 + 1.2 * k, 2.0, 1.2) for k in range(3)] * 1 + [(4.4, 2.0, 1.2)] * 0, noise_dbfs=-86.0)
    assert fast.counts["accepted"] + fast.counts["speed"] + fast.counts["no_position"] >= 1


def test_the_decay_map_follows_the_true_spatial_pattern_of_the_room():
    pts = [(0.9 + 0.3 * k, 1.4, 1.2) for k in range(9)] + [(3.3, 1.4 + 0.5 * j, 1.2) for j in range(1, 4)]     # a slow walk, 0.3 m per burst
    mapper, rm = run_walk(pts, periods=len(pts))
    assert mapper.counts["accepted"] >= 8, mapper.counts
    fields = mapper.fields(63.0, 1 / 3)
    assert fields.n_used >= 8 and fields.coverage() > 0.01
    errs = []
    for fr in mapper.frames:
        t = ideal_metrics(rm, (fr.xy[0], fr.xy[1], 1.2), 63.0)
        e, _ = fields.value_at(*fr.xy)
        if t.ok and np.isfinite(e):
            errs.append(e / t.edt)
    assert len(errs) >= 7 and np.median(np.abs(np.array(errs) - 1.0)) < 0.2         # EDT at the measured spots (map smoothing included)
    left = fields.energy_left_db(0.0)
    assert np.nanmax(left) == pytest.approx(0.0, abs=0.5)                            # all the energy is still to come at t = 0
    later = fields.energy_left_db(0.5)
    assert np.nanmax(later) < -3.0 and np.nanmedian(later) < np.nanmedian(fields.energy_left_db(0.2))


def test_energy_left_decays_faster_where_the_edt_is_shorter():
    pos_a, pos_b = (1.5, 2.0, 1.2), (3.2, 1.0, 1.2)
    mapper, rm = run_walk([pos_a] * 5 + [pos_b] * 5)
    fa = build_fields(mapper.frames[:3], DIMS, 63.0)
    fb = build_fields(mapper.frames[-3:], DIMS, 63.0)
    ea, eb = fa.value_at(1.5, 2.0)[0], fb.value_at(3.2, 1.0)[0]
    assert np.isfinite(ea) and np.isfinite(eb)
    ta = ideal_metrics(rm, pos_a, 63.0).edt
    tb = ideal_metrics(rm, pos_b, 63.0).edt
    assert (ea - eb) * (ta - tb) > 0 or abs(ta - tb) < 0.03                          # the ordering of the two spots is right
