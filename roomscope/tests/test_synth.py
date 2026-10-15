import numpy as np
import pytest

from roomscope.core.decay import analyze_band, analyze_decay
from roomscope.core.modes import RoomDims
from roomscope.core.sweep import ImpulseResponse
from roomscope.core.synth import SyntheticRoom, T60Profile, fractional_delay_impulse

FS = 48000
MIC = (3.5, 2.0, 1.2)


@pytest.fixture(scope="module")
def room():
    return SyntheticRoom(RoomDims(5.0, 4.0, 2.5), (0.4, 1.3, 1.1), seed=3)


@pytest.fixture(scope="module")
def ir(room):
    return room.ir(MIC, FS, 3.0)


def test_t60_profile_interpolates_log_and_clamps():
    p = T60Profile({100.0: 1.0, 10000.0: 0.5})
    assert p(1000.0) == pytest.approx(0.75)
    assert p(20.0) == pytest.approx(1.0) and p(20000.0) == pytest.approx(0.5)
    assert T60Profile(0.42)(1234.0) == pytest.approx(0.42)


def test_fractional_delay_places_peak_between_samples():
    n = 4096
    x = fractional_delay_impulse(n, 1000.37, 1.0)
    assert np.argmax(x) == 1000
    spec = np.fft.rfft(x)
    k = np.arange(1, 200)
    delay = -np.unwrap(np.angle(spec[1:200])) / (2 * np.pi * k / n)
    assert np.allclose(delay, 1000.37, atol=0.02)
    assert np.allclose(np.abs(spec[1:200]), 1.0, atol=1e-3)           # flat passband


def test_direct_sound_arrival_time_and_energy(room, ir):
    r = room.distance(MIC)
    assert abs(np.argmax(np.abs(ir)) - r / room.c * FS) < 2           # peak lands at r/c
    assert 0.6 / r < np.max(np.abs(ir)) < 1.1 / r                     # sample peak = sinc offset x 1/r, plus a little tail
    win = slice(int(r / room.c * FS) - 40, int(r / room.c * FS) + 40)
    assert np.sum(ir[win] ** 2) == pytest.approx(1 / r ** 2, rel=0.5)


def test_direct_to_reverberant_ratio_follows_critical_distance(room, ir):
    """The critical-distance formula assumes a mean absorption, and our HF decays are shorter than the mean,
    so a few dB of difference is expected."""
    r = room.distance(MIC)
    e_dir = (1 / r) ** 2
    drr_db = 10 * np.log10(e_dir / (np.sum(ir ** 2) - e_dir))
    assert drr_db == pytest.approx(20 * np.log10(room.r_crit / r), abs=4.0)


def test_spectrum_has_no_step_at_the_modal_diffuse_crossover(room, ir):
    H = np.abs(np.fft.rfft(ir, 2 ** 17)) ** 2
    f = np.fft.rfftfreq(2 ** 17, 1 / FS)
    m = lambda lo, hi: 10 * np.log10(H[(f >= lo) & (f < hi)].mean())          # noqa: E731
    fx = room.f_x
    assert abs(m(0.6 * fx, 0.9 * fx) - m(1.1 * fx, 1.6 * fx)) < 4.0
    assert abs(m(2 * fx, 4 * fx) - m(8 * fx, 16 * fx)) < 3.0               # flat reverberant spectrum above


def test_low_frequency_spectrum_matches_the_analytic_modal_response(room, ir):
    """Truncating the modal sum leaves tiny acausal ringing (-32 dB) that the cropped IR drops, so agreement is
    statistical: typical error a few percent, and the level shape (dB) agrees to well under 1 dB rms."""
    f = np.fft.rfftfreq(2 ** 17, 1 / FS)
    k = np.where((f > 20) & (f < 0.7 * room.f_x))[0]
    H = np.fft.rfft(ir, 2 ** 17)[k]
    ref = room.frequency_response(MIC, f[k])
    assert np.median(np.abs(H - ref) / np.abs(ref)) < 0.06
    db = 20 * np.log10(np.abs(H) / np.abs(ref))
    assert abs(np.mean(db)) < 0.3 and np.sqrt(np.mean(db ** 2)) < 1.2


def test_octave_band_decay_matches_the_specified_profile(room, ir):
    res = analyze_decay(ImpulseResponse.from_array(ir, FS), 1, 60, 8000)
    for b in res.bands:
        tol = 0.25 if b.fc < 100 else (0.15 if b.fc < 250 else 0.08)      # bass: few modes -> beating
        assert b.t60 == pytest.approx(float(room.t60(b.fc)), rel=tol), (b.fc, b.t60, b.notes)


def test_position_changes_the_bass_response_but_not_the_decay_time(room):
    a, b = (3.5, 2.0, 1.2), (1.2, 0.6, 1.2)
    ia, ib = room.ir(a, FS, 3.0), room.ir(b, FS, 3.0)
    f = np.fft.rfftfreq(2 ** 17, 1 / FS)
    sel = (f > 30) & (f < 150)
    la = 20 * np.log10(np.abs(np.fft.rfft(ia, 2 ** 17))[sel])
    lb = 20 * np.log10(np.abs(np.fft.rfft(ib, 2 ** 17))[sel])
    assert np.std(la - lb) > 3.0                               # standing-wave pattern differs by position
    ta = analyze_decay(ImpulseResponse.from_array(ia, FS), 1, 500, 2000).t_mid(500, 1000)
    tb = analyze_decay(ImpulseResponse.from_array(ib, FS), 1, 500, 2000).t_mid(500, 1000)
    assert ta == pytest.approx(tb, rel=0.1)


def test_ir_is_deterministic_per_seed_and_position(room):
    a, b = room.ir(MIC, FS, 1.0), room.ir(MIC, FS, 1.0)
    c = SyntheticRoom(room.dims, room.source, room.t60, seed=4).ir(MIC, FS, 1.0)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
