import numpy as np
import pytest

from roomscape.core.calibration import Calibration
from roomscape.core.spectrogram import MODES, SpectrogramEngine, SpectrogramHistory, SpectrogramSettings

FS = 48000


def pink(n, seed):
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std()


@pytest.mark.parametrize("mode,f0,tol", [
    ("Balanced", 80.0, 1.2), ("Balanced", 1000.0, 0.7), ("Balanced", 4000.0, 0.7),
    ("Bass detail", 80.0, 1.2), ("Bass detail", 1000.0, 0.7), ("Bass detail", 4000.0, 0.7),
    # 'Fast' trades frequency resolution for timing: a pure tone's skirts spill outside a 1/24-octave band
    ("Fast", 1000.0, 2.0), ("Fast", 4000.0, 2.0),
])
def test_tone_levels_in_every_resolution_group(mode, f0, tol):
    eng = SpectrogramEngine(FS, SpectrogramSettings(mode=mode))
    t = np.arange(eng.required_samples) / FS
    col = eng.compute(0.1 * np.sin(2 * np.pi * f0 * t + 0.4))
    peak = col.max()
    assert -23.01 - tol <= peak <= -23.01 + 0.1, (mode, f0, peak)
    assert abs(np.log2(eng.grid[np.argmax(col)] / f0)) < 1.0 / 12          # within a half grid step or so


def test_pink_noise_is_flat_across_stitched_groups():
    eng = SpectrogramEngine(FS)
    cols = [eng.compute(0.1 * pink(eng.required_samples + 4000, s)) for s in range(80)]
    mean = 10 * np.log10(np.mean(10 ** (np.array(cols) / 10), axis=0))            # power-average over frames
    sel = (eng.grid > 60) & (eng.grid < 12000)
    assert mean[sel].max() - mean[sel].min() < 3.0
    i_lo = np.argmin(np.abs(eng.grid - 140))
    i_hi = np.argmin(np.abs(eng.grid - 160))
    assert abs(mean[i_lo] - mean[i_hi]) < 2.0                                      # no jump at the 150 Hz hand-over
    i_lo = np.argmin(np.abs(eng.grid - 1400))
    i_hi = np.argmin(np.abs(eng.grid - 1600))
    assert abs(mean[i_lo] - mean[i_hi]) < 2.0


def test_chirp_ridge_follows_instantaneous_frequency():
    eng = SpectrogramEngine(FS, SpectrogramSettings(mode="Fast"))
    dur, f1, f2 = 4.0, 200.0, 8000.0
    t = np.arange(int(dur * FS)) / FS
    k = np.log(f2 / f1) / dur
    x = 0.1 * np.sin(2 * np.pi * f1 / k * (np.exp(k * t) - 1))
    for t_probe in (1.0, 2.0, 3.0):
        end = int(t_probe * FS) + eng.required_samples // 2
        col = eng.compute(x[:end])
        f_true = f1 * np.exp(k * (end - eng.required_samples / 2) / FS)
        assert abs(np.log2(eng.grid[np.argmax(col)] / f_true)) < 0.35, (t_probe, f_true)


def test_calibration_offset_and_mic_cal_apply():
    from roomscape.core.calibration import parse_calibration_text

    cal = Calibration(mic=parse_calibration_text("20 -3\n20000 -3\n"), spl_offset_db=100.0)
    eng = SpectrogramEngine(FS, calibration=cal)
    t = np.arange(eng.required_samples) / FS
    col = eng.compute(0.1 * np.sin(2 * np.pi * 1000 * t))
    assert col.max() == pytest.approx(-23.01 + 3 + 100, abs=0.8)


def test_history_is_newest_first_and_slices_are_evenly_spaced():
    h = SpectrogramHistory(n_freq=4, n_rows=100, hop_s=0.05)
    for i in range(30):
        h.push(np.full(4, float(i)))
    assert h.data[0, 0] == 29 and h.data[1, 0] == 28 and h.data[29, 0] == 0
    assert h.data[40, 0] == -200.0                         # untouched fill
    sl, ago = h.slices(n_slices=5, span_s=1.0)             # 1 s / 0.05 s = 20 rows over 5 slices -> stride 4
    assert sl[:, 0].tolist() == [29, 25, 21, 17, 13]
    assert ago.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert h.span_s == pytest.approx(5.0)


def test_short_buffer_is_zero_padded():
    eng = SpectrogramEngine(FS)
    col = eng.compute(np.zeros(100))
    assert col.shape == eng.grid.shape and np.all(col < -150)
