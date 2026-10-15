import numpy as np
import pytest
from scipy.signal import fftconvolve

from roomscope.core.sweep import (
    ImpulseResponse, SweepDeconvolver, SweepSpec, recording_length, sweep_signal, sweep_unit,
)


def simulate(spec, h, latency=7000, noise_dbfs=None, drive=lambda x: x, seed=0):
    x = sweep_signal(spec).astype(np.float64)
    n = recording_length(spec)
    y = np.zeros(n)
    sig = fftconvolve(drive(x), h)
    m = min(sig.size, n - latency)
    y[latency:latency + m] = sig[:m]
    if noise_dbfs is not None:
        y += 10 ** (noise_dbfs / 20) * np.random.default_rng(seed).standard_normal(n)
    return y


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_sweep_level_duration_and_zero_crossing_end(fs):
    spec = SweepSpec(fs=fs, duration=3.0, level_dbfs=-12.0)
    x = sweep_signal(spec)
    assert np.max(np.abs(x)) == pytest.approx(10 ** (-12 / 20), rel=1e-3)
    assert abs(x.size / fs - 3.0) < 0.01
    assert abs(x[-1]) < 0.02 * np.max(np.abs(x))                 # faded out / on a zero crossing


def test_sweep_energy_falls_3db_per_octave():
    spec = SweepSpec(fs=48000, duration=6.0)
    x = sweep_unit(spec)
    P = np.abs(np.fft.rfft(x, 2 ** 20)) ** 2
    f = np.fft.rfftfreq(2 ** 20, 1 / 48000)
    def band(lo, hi):
        s = (f >= lo) & (f < hi)
        return 10 * np.log10(P[s].sum())
    # equal-octave energy of a log sweep is constant, i.e. PSD falls 3 dB/oct
    levels = [band(f0, 2 * f0) for f0 in (100, 200, 400, 800, 1600, 3200, 6400)]
    assert max(levels) - min(levels) < 0.6


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_recovers_known_impulse_response_gain_latency_and_noise(fs):
    spec = SweepSpec(fs=fs, duration=4.0, tail_s=1.0, level_dbfs=-12)
    h = np.zeros(int(0.06 * fs))
    h[int(0.002 * fs)] = 0.5
    h[int(0.0125 * fs)] = -0.2
    lat = int(0.17 * fs)
    ir = SweepDeconvolver(spec).extract(simulate(spec, h, latency=lat, noise_dbfs=-90))
    assert ir.latency_samples == lat + int(0.002 * fs)
    # Compare spectra in-band: |H_est| must equal |H_true| (system gain). The matched-filter inverse leaves a
    # small spectral ripple (<0.3 dB, far inside any mic's tolerance); the comb in H_true makes notches sensitive.
    n = 8192
    t_true = int(0.002 * fs)
    H_true = np.abs(np.fft.rfft(h, n))
    H_est = np.abs(np.fft.rfft(ir.h[ir.t0 - t_true: ir.t0 - t_true + h.size], n))
    f = np.fft.rfftfreq(n, 1 / fs)
    sel = (f > 200) & (f < 12000)
    err = 20 * np.log10(H_est[sel] / H_true[sel])
    assert np.abs(err).max() < 0.3, np.abs(err).max()
    assert np.sqrt(np.mean(err ** 2)) < 0.1


def test_harmonic_distortion_is_rejected():
    fs = 48000
    spec = SweepSpec(fs=fs, duration=4.0, tail_s=1.0, level_dbfs=-6)
    h = np.zeros(2000)
    h[100] = 0.6
    h[900] = -0.25
    clean = SweepDeconvolver(spec).extract(simulate(spec, h))
    dirty = SweepDeconvolver(spec).extract(simulate(spec, h, drive=lambda x: x + 0.08 * x ** 2 + 0.05 * x ** 3))
    seg = slice(clean.t0, clean.t0 + 1500)
    # the linear IR after the peak is unchanged by even/odd harmonic distortion
    err = clean.h[seg] - dirty.h[seg]
    assert np.sum(err ** 2) / np.sum(clean.h[seg] ** 2) < 10 ** (-40 / 10)
    assert dirty.latency_samples == clean.latency_samples


def test_valid_samples_shrink_toward_high_frequencies():
    fs = 48000
    spec = SweepSpec(fs=fs, duration=6.0, tail_s=2.0)
    h = np.zeros(500)
    h[50] = 1.0
    ir = SweepDeconvolver(spec).extract(simulate(spec, h, latency=9000))
    lo, hi = ir.valid_samples(50), ir.valid_samples(20000)
    assert lo > hi > int(1.5 * fs)                          # HF keeps ~the tail, LF keeps far more
    assert (lo - hi) / fs == pytest.approx(ir.sweep_rate_L * np.log(20000 / 50), rel=0.02)
    assert ir.valid_samples(10) <= ir.h.size - ir.t0


def test_from_array_defaults_to_peak_and_full_validity():
    h = np.zeros(1000)
    h[123] = -2.0
    ir = ImpulseResponse.from_array(h, 48000)
    assert ir.t0 == 123 and ir.valid_samples(5000) == 1000 - 123
