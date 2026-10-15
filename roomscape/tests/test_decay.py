import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from roomscape.core.decay import (
    BandDecay, DecayResult, analyze_band, analyze_decay, csd_waterfall, fit_decay_time, lundeby, schroeder_db,
)
from roomscape.core.sweep import ImpulseResponse

FS = 48000
PRE = 144


def exp_noise_ir(t60, secs=3.0, noise_db=None, seed=0):
    r = np.random.default_rng(seed)
    n = int(secs * FS)
    t = np.arange(n) / FS
    h = r.standard_normal(n) * np.exp(-6.9078 * t / t60)
    h[0] = 8.0
    if noise_db is not None:
        h = h + r.standard_normal(n) * np.sqrt(np.mean(h[: int(0.02 * FS)] ** 2)) * 10 ** (noise_db / 20)
    return ImpulseResponse.from_array(np.concatenate((np.zeros(PRE), h)), FS, t0=PRE)


@pytest.mark.parametrize("t60", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("noise_db", [None, -60, -45, -25])
def test_t60_accurate_with_and_without_noise(t60, noise_db):
    b = analyze_band(exp_noise_ir(t60, noise_db=noise_db), 1000, 20, 20000, filtered=False)
    assert b.t60 == pytest.approx(t60, rel=0.03), (b.basis, b.dynamic_range_db)
    assert b.edt == pytest.approx(t60, rel=0.04)


def test_basis_selection_follows_dynamic_range():
    hi = analyze_band(exp_noise_ir(0.5, noise_db=-60), 1000, 20, 20000, filtered=False)
    mid = analyze_band(exp_noise_ir(0.5, noise_db=-38), 1000, 20, 20000, filtered=False)
    lo = analyze_band(exp_noise_ir(0.5, noise_db=-25), 1000, 20, 20000, filtered=False)
    assert (hi.basis, mid.basis, lo.basis) == ("T30", "T20", "EDT")
    assert hi.reliable and mid.reliable and not lo.reliable
    assert any("noise" in n for n in lo.notes)


def test_noise_handling_removes_the_bias_of_plain_schroeder_integration():
    ir = exp_noise_ir(0.5, noise_db=-30)
    e = ir.h[ir.t0:] ** 2
    naive = np.cumsum(e[::-1])[::-1]
    naive_db = 10 * np.log10(naive / naive[0])
    t20_naive, _ = fit_decay_time(naive_db, FS, -5, -25)
    b = analyze_band(ir, 1000, 20, 20000, filtered=False)
    assert abs(t20_naive / 0.5 - 1) > 0.15           # un-truncated integration is badly biased by the noise floor ...
    assert abs(b.t20 / 0.5 - 1) < 0.03               # ... the Lundeby + compensation path is not


def test_pure_noise_is_flagged_unreliable():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(FS) ** 2
    tr = lundeby(e, FS)
    assert not tr.reliable


def test_schroeder_curve_is_monotonic_and_starts_at_zero():
    ir = exp_noise_ir(0.4, noise_db=-50)
    e = ir.h[ir.t0:] ** 2
    tr = lundeby(e, FS)
    c = schroeder_db(e, tr, FS)
    assert c[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(c) <= 1e-6)


def per_band_ir(t60_of, secs=3.0, seed=5):
    """Noise IR whose decay time depends on frequency (octave bands 63 Hz .. 16 kHz)."""
    r = np.random.default_rng(seed)
    n = int(secs * FS)
    t = np.arange(n) / FS
    noise = r.standard_normal(n)
    h = np.zeros(n)
    for fc in (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
        sos = butter(4, [fc / np.sqrt(2), min(fc * np.sqrt(2), 0.49 * FS)], btype="band", fs=FS, output="sos")
        h += sosfiltfilt(sos, noise) * np.exp(-6.9078 * t / t60_of(fc))
    h[0] += 6.0
    return ImpulseResponse.from_array(np.concatenate((np.zeros(PRE), h)), FS, t0=PRE)


def test_octave_bands_recover_frequency_dependent_decay():
    """Tolerances widen toward the bass: a 44 Hz-wide band has few independent samples (B*T ~ 16-40), so a
    single noise realisation scatters by ~10-15 % there (the reason ISO 3382 asks for B*T > 16)."""
    spec_t60 = {63: 0.9, 125: 0.7, 250: 0.55, 500: 0.45, 1000: 0.4, 2000: 0.35, 4000: 0.3, 8000: 0.25}
    ir = per_band_ir(lambda fc: spec_t60.get(round(fc), spec_t60[min(spec_t60, key=lambda k: abs(np.log(k / fc)))]))
    res = analyze_decay(ir, 1, 60, 8000)
    tol = {63: 0.20, 125: 0.12}
    for fc, t60 in spec_t60.items():
        b = res.band_near(fc)
        assert b.t60 == pytest.approx(t60, rel=tol.get(fc, 0.08)), (fc, b.t60, t60, b.notes)
    assert res.t_mid() == pytest.approx((0.45 + 0.4) / 2, rel=0.06)
    assert res.bass_ratio() == pytest.approx((0.7 + 0.55) / (0.45 + 0.4), rel=0.12)


def test_bandwidth_time_product_flag():
    ir = per_band_ir(lambda fc: 0.4)
    oct_ = analyze_decay(ir, 1, 100, 200)
    third = analyze_decay(ir, 3, 55, 70)
    b125 = oct_.band_near(125)
    assert b125.bt == pytest.approx((b125.f_hi - b125.f_lo) * b125.t60) and b125.bt > 16 and b125.reliable
    t = third.band_near(63)                                   # 14 Hz wide: B*T ~ 5
    assert t.bt < 16 and any("narrow" in n for n in t.notes) and not t.reliable


def test_clarity_matches_analytic_exponential():
    t60 = 0.5
    ir = exp_noise_ir(t60, secs=4.0, noise_db=None, seed=2)
    ir.h[ir.t0] = 0.0                                        # no direct spike: pure exponential energy
    b = analyze_band(ir, 1000, 20, 20000, filtered=False)
    a50 = np.exp(-13.8155 * 0.05 / t60)
    a80 = np.exp(-13.8155 * 0.08 / t60)
    assert b.c50_db == pytest.approx(10 * np.log10((1 - a50) / a50), abs=0.4)
    assert b.c80_db == pytest.approx(10 * np.log10((1 - a80) / a80), abs=0.4)
    assert b.d50_pct == pytest.approx(100 * (1 - a50), abs=1.5)


def test_short_recording_is_reported_not_crashed():
    ir = ImpulseResponse.from_array(np.r_[1.0, np.zeros(500)], FS)
    b = analyze_band(ir, 1000, 700, 1400)
    assert np.isnan(b.t60) and any("short" in n for n in b.notes)


def test_csd_ridge_decays_at_the_modal_rate_and_sits_at_the_mode():
    f0, t60 = 100.0, 0.8
    n = int(1.5 * FS)
    t = np.arange(n) / FS
    h = np.sin(2 * np.pi * f0 * t) * np.exp(-6.9078 * t / t60)
    ir = ImpulseResponse.from_array(np.concatenate((np.zeros(PRE), h)), FS, t0=PRE)
    wf = csd_waterfall(ir, n_slices=30, dt_s=0.02, window_s=0.5, f_lo=40, f_hi=250, smoothing_oct=24)
    peak_i = np.argmax(wf.slices_db, axis=1)
    assert np.all(np.abs(wf.freqs[peak_i] / f0 - 1) < 0.06)
    ridge = wf.slices_db[np.arange(30), peak_i]
    slope = np.polyfit(wf.times_s[2:], ridge[2:], 1)[0]
    assert slope == pytest.approx(-60.0 / t60, rel=0.06)
    assert wf.slices_db.shape == (30, wf.freqs.size) and wf.times_s[1] == pytest.approx(0.02)


def test_decay_result_helpers_on_handmade_bands():
    mk = lambda fc, t: BandDecay(fc=fc, f_lo=fc / 1.41, f_hi=fc * 1.41, t60=t)          # noqa: E731
    res = DecayResult(FS, [mk(125.0, 0.8), mk(250.0, 0.6), mk(500.0, 0.4), mk(1000.0, 0.4)], mk(500, 0.5), 1)
    assert res.t_mid() == pytest.approx(0.4)
    assert res.bass_ratio() == pytest.approx(1.4 / 0.8)
    assert res.band_near(300).fc == 250.0
