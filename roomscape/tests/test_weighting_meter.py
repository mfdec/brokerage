import numpy as np
import pytest
from scipy.signal import sosfreqz

from roomscape.core.meter import LevelMeter
from roomscape.core.weighting import weighting_db, weighting_sos

# IEC 61672-1 table values (dB) at the exact 10**(n/10) Hz frequencies
_EXACT = {31.5: 10 ** 1.5, 63: 10 ** 1.8, 125: 10 ** 2.1, 250: 10 ** 2.4, 500: 10 ** 2.7, 1000: 1000.0,
          2000: 10 ** 3.3, 4000: 10 ** 3.6, 8000: 10 ** 3.9, 16000: 10 ** 4.2}
A_TABLE = {31.5: -39.4, 63: -26.2, 125: -16.1, 250: -8.6, 500: -3.2, 1000: 0.0,
           2000: 1.2, 4000: 1.0, 8000: -1.1, 16000: -6.6}
C_TABLE = {31.5: -3.0, 63: -0.8, 125: -0.2, 1000: 0.0, 4000: -0.8, 8000: -3.0, 16000: -8.5}


@pytest.mark.parametrize("nominal,expected", A_TABLE.items())
def test_a_weighting_matches_standard_table(nominal, expected):
    assert weighting_db(_EXACT[nominal], "A") == pytest.approx(expected, abs=0.06)


@pytest.mark.parametrize("nominal,expected", C_TABLE.items())
def test_c_weighting_matches_standard_table(nominal, expected):
    assert weighting_db(_EXACT[nominal], "C") == pytest.approx(expected, abs=0.06)


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
@pytest.mark.parametrize("kind", ["A", "C"])
def test_digital_filter_tracks_analog_weighting(fs, kind):
    """Bilinear design: tight up to 4 kHz, <0.8 dB at 8 kHz even at 44.1 kHz (IEC class-2 allows +-1.5 dB there)."""
    sos = weighting_sos(fs, kind)
    f = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000])
    _w, h = sosfreqz(sos, worN=2 * np.pi * f / fs)
    err = np.abs(20 * np.log10(np.abs(h)) - weighting_db(f, kind))
    assert err[:-1].max() < 0.1, (fs, kind, err)
    assert err[-1] < 0.8, (fs, kind, err)


def test_meter_rms_and_peak_of_sine():
    fs = 48000
    t = np.arange(fs) / fs
    x = 0.5 * np.sin(2 * np.pi * 1000 * t)
    r = LevelMeter(fs).process(x)
    assert r.rms_db == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.01)
    assert r.peak_db == pytest.approx(20 * np.log10(0.5), abs=0.01)
    assert not r.clipped


def test_meter_time_weighting_is_chunk_size_independent():
    fs = 48000
    rng = np.random.default_rng(0)
    x = 0.1 * rng.standard_normal(fs)
    one = LevelMeter(fs).process(x).tw_db
    m = LevelMeter(fs)
    for chunk in np.array_split(x, 37):
        last = m.process(chunk)
    assert last.tw_db == pytest.approx(one, abs=1e-6)


def test_meter_clip_detection():
    x = np.zeros(1000)
    x[10] = 1.0
    m = LevelMeter(48000)
    assert m.process(x).clipped
    assert not m.process(np.zeros(1000)).clipped


def test_meter_leq_tracks_average_power():
    fs = 48000
    m = LevelMeter(fs)
    loud = 0.2 * np.ones(fs)
    quiet = np.zeros(fs)
    m.process(loud)
    r = m.process(quiet)
    assert r.leq_db == pytest.approx(10 * np.log10(0.04 / 2), abs=0.01)


def test_a_weighted_meter_attenuates_low_frequency():
    fs = 48000
    t = np.arange(fs) / fs
    low = 0.1 * np.sin(2 * np.pi * 63 * t)
    mz, ma = LevelMeter(fs, "Z"), LevelMeter(fs, "A")
    dz = mz.process(low).rms_db
    ma.process(low)                      # let the filter settle
    da = ma.process(low).rms_db
    assert dz - da == pytest.approx(26.2, abs=0.6)
