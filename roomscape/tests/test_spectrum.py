import numpy as np
import pytest

from roomscape.core.calibration import Calibration, parse_calibration_text
from roomscape.core.spectrum import RtaEngine, RtaSettings, power_spectrum, tone_level_dbfs
from roomscape.core.units import band_centers, fractional_octave_power, power_to_db

FS = 48000


def sine(f, amp, n, fs=FS, phase=0.3):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f * t + phase)


def pink_noise(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    spec /= np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return x / x.std()


def test_parseval_white_noise_power():
    rng = np.random.default_rng(1)
    x = 0.1 * rng.standard_normal(FS * 4)
    p, _ = power_spectrum(x, "Hann")
    assert p.sum() == pytest.approx(x.var(), rel=0.02)


@pytest.mark.parametrize("window", ["Hann", "Blackman-Harris", "Flat-top", "Hamming", "Rectangular"])
def test_tone_power_is_conserved(window):
    x = sine(1234.5, 0.1, 16384)
    p, _ = power_spectrum(x, window)
    assert power_to_db(p.sum()) == pytest.approx(20 * np.log10(0.1 / np.sqrt(2)), abs=0.02)


def test_fft_mode_peak_bin_reads_true_level_on_bin_center():
    n = 16384
    f0 = 341 * FS / n                                    # exactly on a bin
    eng = RtaEngine(FS, RtaSettings(fft_size=n, resolution="FFT", averaging_s=0))
    fr = eng.process(sine(f0, 0.1, n), 0.03)
    assert fr.inst_db.max() == pytest.approx(-23.01, abs=0.05)
    assert fr.freqs[np.argmax(fr.inst_db)] == pytest.approx(f0, abs=eng._cache["df"])


def test_flattop_fft_mode_is_scalloping_free():
    n = 16384
    eng = RtaEngine(FS, RtaSettings(fft_size=n, resolution="FFT", window="Flat-top", averaging_s=0))
    # half-bin offset is the worst case for every window
    f0 = (341.5) * FS / n
    fr = eng.process(sine(f0, 0.1, n), 0.03)
    assert fr.inst_db.max() == pytest.approx(-23.01, abs=0.1)


@pytest.mark.parametrize("res,tol", [("1/3", 0.1), ("1/12", 0.15), ("1/24", 0.3), ("1/48", 0.8)])
def test_fractional_octave_modes_read_tone_level_off_bin(res, tol):
    """Bands wider than the Hann main lobe capture the whole tone; the finest bands may clip its skirts."""
    n = 16384
    eng = RtaEngine(FS, RtaSettings(fft_size=n, resolution=res, averaging_s=0))
    fr = eng.process(sine(1000.0, 0.1, n), 0.03)
    assert -23.01 - tol <= fr.inst_db.max() <= -23.01 + 0.05


def test_pink_noise_stays_flat_into_the_bass_with_sub_resolution_bands():
    """16k FFT at 48 kHz = 2.9 Hz bins; a 1/12-octave band at 40 Hz is only 2.3 Hz wide.  Noise must still read flat."""
    rng = np.random.default_rng(11)
    x = 0.1 * pink_noise(FS * 60, rng)
    eng = RtaEngine(FS, RtaSettings(fft_size=32768, resolution="1/12", averaging_s=120.0))
    step = 1600
    for i in range(0, x.size - 32768, step):
        fr = eng.process(x[i:i + 32768], step / FS)
    sel = (fr.freqs > 35) & (fr.freqs < 10000)
    dev = fr.avg_db[sel] - fr.avg_db[sel].mean()
    assert np.abs(dev).max() < 1.8, np.abs(dev).max()


def test_white_noise_overall_level():
    rng = np.random.default_rng(2)
    sigma = 0.05
    x = sigma * rng.standard_normal(FS * 2)
    eng = RtaEngine(FS, RtaSettings(fft_size=32768, averaging_s=0))
    fr = eng.process(x, 0.03)
    expected = 10 * np.log10(sigma ** 2 * (20000 - 20) / (FS / 2))
    assert fr.overall_db == pytest.approx(expected, abs=0.3)


def test_pink_noise_is_flat_in_third_octave_bands():
    rng = np.random.default_rng(3)
    x = 0.1 * pink_noise(FS * 40, rng)
    p, _ = power_spectrum(x, "Hann")
    df = FS / x.size
    fc = band_centers(3, 63, 8000)
    levels = power_to_db(fractional_octave_power(p, df, fc, 3))
    assert levels.max() - levels.min() < 1.5
    assert np.abs(levels - levels.mean()).max() < 0.8


def test_mic_calibration_and_spl_offset_are_applied():
    n = 16384
    cal_text = "20 -3\n20000 -3\n"                       # mic reads 3 dB low
    cal = Calibration(mic=parse_calibration_text(cal_text), spl_offset_db=100.0)
    eng = RtaEngine(FS, RtaSettings(fft_size=n, resolution="1/12", averaging_s=0), cal)
    fr = eng.process(sine(1000.0, 0.1, n), 0.03)
    assert fr.unit == "dB SPL"
    assert fr.inst_db.max() == pytest.approx(-23.01 + 3.0 + 100.0, abs=0.2)


def test_a_weighting_applied_to_display():
    n = 16384
    eng = RtaEngine(FS, RtaSettings(fft_size=n, resolution="FFT", window="Flat-top", weighting="A", averaging_s=0))
    fr = eng.process(sine(100.0, 0.1, n), 0.03)
    assert fr.inst_db.max() == pytest.approx(-23.01 - 19.1, abs=0.15)


def test_exponential_averaging_converges_and_smooths():
    rng = np.random.default_rng(4)
    eng = RtaEngine(FS, RtaSettings(fft_size=8192, resolution="1/3", averaging_s=1.0))
    last = None
    for _ in range(120):
        last = eng.process(0.1 * rng.standard_normal(8192), 1 / 30)
    # averaged curve varies much less than the instantaneous one
    assert np.std(np.diff(last.avg_db)) < np.std(np.diff(last.inst_db))


def test_tone_level_helper_for_calibrator_wizard():
    x = sine(1000.0, 0.02, FS * 2) + 0.0005 * np.random.default_rng(5).standard_normal(FS * 2)
    assert tone_level_dbfs(x, FS, 1000.0) == pytest.approx(20 * np.log10(0.02 / np.sqrt(2)), abs=0.1)


def test_exponential_average_is_unbiased_at_start():
    from roomscape.core.spectrum import SpectrumAverager

    av = SpectrumAverager(time_const_s=100.0)
    a, _ = av.update(np.array([4.0]), 0.03)
    assert a[0] == pytest.approx(4.0)                                  # first frame is exactly itself
    for _ in range(999):
        a, _ = av.update(np.array([2.0]), 0.03)
    # ~30 s of 2.0 after one 4.0: weights are ~equal, so the mean is ~2.0 + 2/1000 - not dominated by the seed
    assert a[0] == pytest.approx(2.0 + 2.0 / 1000, abs=0.01)
