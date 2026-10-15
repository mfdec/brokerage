import numpy as np
import pytest

from roomscape.core.calibration import Calibration, parse_calibration_text
from roomscape.core.multisine import MultisineSpec, analyze_frame, crest_factor_db, generate, line_params

FS = 48000


@pytest.mark.parametrize("pink", [True, False])
def test_stimulus_level_lines_and_crest_factor(pink):
    spec = MultisineSpec(FS, 0.5, 20, 300, pink, -20.0)
    x = generate(spec)
    assert x.size == 24000 and np.max(np.abs(x)) == pytest.approx(0.1, rel=1e-3)
    assert crest_factor_db(x) < 9.0                                       # Schroeder phases keep the peak low (pure noise ~12+ dB)
    f = spec.line_freqs
    assert f[0] == 20.0 and f[-1] == 300.0 and len(f) == 141 and np.allclose(np.diff(f), 2.0)
    # the spectrum has exactly these lines
    X = np.abs(np.fft.rfft(x))
    bins = np.rint(f / (FS / x.size)).astype(int)
    off = np.ones(X.size, bool)
    off[bins] = False
    assert X[bins].min() > 1000 * X[off].max()


def test_pink_tilt_is_minus_3_db_per_octave_and_flat_is_flat():
    _, a_pink, _ = line_params(MultisineSpec(FS, 0.5, 20, 300, True))
    _, a_flat, _ = line_params(MultisineSpec(FS, 0.5, 20, 300, False))
    f = MultisineSpec().line_freqs
    i40, i80 = np.argmin(np.abs(f - 40)), np.argmin(np.abs(f - 80))
    assert 20 * np.log10(a_pink[i80] / a_pink[i40]) == pytest.approx(-3.0, abs=0.1)
    assert np.allclose(a_flat, a_flat[0])


def mic_signal(spec, gains_db, noise_dbfs=None, periods=2, seed=0, offset=0):
    """What a mic would record: the stimulus with each line scaled by ``gains_db`` (+ noise)."""
    f, a, th = line_params(spec)
    t = (offset + np.arange(periods * spec.n_period)) / spec.fs
    g = 10 ** (np.asarray(gains_db) / 20)
    x = (np.cos(2 * np.pi * np.outer(t, f) + th) @ (a * g))
    if noise_dbfs is not None:
        x = x + 10 ** (noise_dbfs / 20) * np.random.default_rng(seed).standard_normal(x.size)
    return x, f, a * g


def test_line_levels_are_recovered_exactly_when_clean():
    spec = MultisineSpec(FS)
    gains = np.random.default_rng(1).uniform(-12, 6, 141)                 # a wildly uneven "room"
    x, f, amps = mic_signal(spec, gains, noise_dbfs=-150)
    r = analyze_frame(x, spec)
    expected = 20 * np.log10(amps / np.sqrt(2))
    assert np.abs(r.power_db - expected).max() < 0.02
    assert r.snr_db > 60 and not r.clipped


def test_levels_do_not_depend_on_where_in_the_period_the_frame_starts():
    spec = MultisineSpec(FS)
    a, _, _ = mic_signal(spec, np.zeros(141), noise_dbfs=-120, offset=0)
    b, _, _ = mic_signal(spec, np.zeros(141), noise_dbfs=-120, offset=7777)
    ra, rb = analyze_frame(a, spec), analyze_frame(b, spec)
    assert np.allclose(ra.power_db, rb.power_db, atol=0.02)               # magnitude-only: no sync needed


@pytest.mark.parametrize("noise_dbfs,tol", [(-60, 0.3), (-50, 1.0)])
def test_noise_is_estimated_from_the_empty_bins_and_subtracted(noise_dbfs, tol):
    spec = MultisineSpec(FS, level_dbfs=-20)
    x, f, amps = mic_signal(spec, np.zeros(141), noise_dbfs=noise_dbfs)
    r = analyze_frame(x, spec)
    expected = 20 * np.log10(amps / np.sqrt(2))
    # per-line SNR at 30 dB noise-vs-line is poor for some lines, so judge the typical (median) error
    assert np.median(np.abs(r.power_db - expected)) < tol
    # broadband SNR: total line power vs total noise power in the same bins
    p_lines = np.sum(amps ** 2 / 2)
    p_noise = 10 ** (noise_dbfs / 10) * (141 / 24000)
    assert r.snr_db == pytest.approx(10 * np.log10(p_lines / p_noise), abs=1.5)


def test_clock_mismatch_of_40_ppm_is_harmless_at_bass_frequencies():
    spec = MultisineSpec(FS)
    f, a, th = line_params(spec)
    t = np.arange(2 * spec.n_period) / (FS * (1 + 40e-6))               # capture clock slightly off the playback clock
    x = np.cos(2 * np.pi * np.outer(t, f) * (1) + th) @ a
    r = analyze_frame(x, spec)
    assert np.abs(r.power_db - 20 * np.log10(a / np.sqrt(2))).max() < 0.15


def test_movement_modulation_lowers_the_reported_snr():
    spec = MultisineSpec(FS)
    steady, _, _ = mic_signal(spec, np.zeros(141), noise_dbfs=-120)
    t = np.arange(steady.size) / FS
    shaken = steady * (1 + 0.25 * np.sin(2 * np.pi * 1.0 * t))           # mic swinging through the field at 1 Hz
    assert analyze_frame(shaken, spec).snr_db < analyze_frame(steady, spec).snr_db - 15


def test_calibration_and_clipping_and_input_validation():
    spec = MultisineSpec(FS)
    x, f, amps = mic_signal(spec, np.zeros(141), noise_dbfs=-140)
    cal = Calibration(mic=parse_calibration_text("20 -3\n20000 -3\n"), spl_offset_db=100.0)
    base = analyze_frame(x, spec)
    c = analyze_frame(x, spec, cal)
    assert np.allclose(c.power_db - base.power_db, 103.0, atol=0.02)
    hot = analyze_frame(np.clip(x * 100, -1, 1), spec)
    assert hot.clipped
    with pytest.raises(ValueError):
        analyze_frame(x[:1000], spec)


def test_other_sample_rates():
    for fs in (44100, 96000):
        spec = MultisineSpec(fs)
        x, f, amps = mic_signal(spec, np.zeros(len(spec.line_freqs)), noise_dbfs=-140)
        r = analyze_frame(x, spec)
        assert np.abs(r.power_db - 20 * np.log10(amps / np.sqrt(2))).max() < 0.02
