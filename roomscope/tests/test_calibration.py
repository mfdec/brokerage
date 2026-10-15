import numpy as np
import pytest

from roomscope.core.calibration import (
    Calibration, CalibrationError, parse_calibration_text, spl_offset_from_calibrator,
)

REW_STYLE = '''"Dayton Audio iMM-6C  Serial 99999"
"Sens Factor =-1.378dB, SERNO: 7000001"
* comment line
20.000\t-0.45\t12.3
100.000\t0.00\t3.1
1000.000\t0.10\t0.0
10000.000\t1.50\t-40.0
20000.000\t-4.00\t-90.0
'''

CSV_STYLE = '''Frequency (Hz),Magnitude (dB)
20,-0.5
200,0.0
2000,1.0
20000,-3.0
'''


def test_parse_rew_style_with_phase_and_sens_factor():
    cal = parse_calibration_text(REW_STYLE, name="mic")
    assert cal.freqs.tolist() == [20, 100, 1000, 10000, 20000]
    assert cal.mag_db[0] == pytest.approx(-0.45)
    assert cal.phase_deg is not None and cal.phase_deg[-1] == pytest.approx(-90.0)
    assert cal.sens_factor_db == pytest.approx(-1.378)
    assert "iMM-6C" in cal.header


def test_parse_csv_with_header_row():
    cal = parse_calibration_text(CSV_STYLE)
    assert cal.freqs.size == 4 and cal.phase_deg is None
    assert cal.mag_db[-1] == pytest.approx(-3.0)


def test_semicolon_decimal_comma():
    cal = parse_calibration_text("20,5;0,3\n200,0;-0,2\n2000;1,0\n")
    assert cal.freqs.tolist() == [20.5, 200.0, 2000.0]
    assert cal.mag_db.tolist() == [0.3, -0.2, 1.0]


def test_numeric_looking_header_is_not_data():
    cal = parse_calibration_text("1/3 octave calibration, rev 2\n50 0.5\n500 0.0\n")
    assert cal.freqs.tolist() == [50.0, 500.0]


def test_quoted_csv_rows():
    cal = parse_calibration_text('"Freq","dB"\n"20","0.5"\n"200","-0.5"\n')
    assert cal.freqs.tolist() == [20.0, 200.0]


def test_interpolation_log_axis_and_flat_extrapolation():
    cal = parse_calibration_text("100 0\n1000 2\n")
    assert cal.correction_db(np.sqrt(100 * 1000)) == pytest.approx(1.0)
    assert cal.correction_db(10) == pytest.approx(0.0)      # below the file -> held
    assert cal.correction_db(5000) == pytest.approx(2.0)    # above the file -> held


def test_apply_to_power_removes_mic_response():
    cal = parse_calibration_text("20 -3\n20000 -3\n")        # mic reads 3 dB low everywhere
    p = cal.apply_to_power(np.array([100.0, 1000.0]), np.array([1.0, 1.0]))
    assert 10 * np.log10(p[0]) == pytest.approx(3.0, abs=1e-9)


def test_errors():
    with pytest.raises(CalibrationError):
        parse_calibration_text("hello\nworld\n")
    with pytest.raises(CalibrationError):
        parse_calibration_text("20 -80\n200 -75\n")          # looks like an SPL curve


def test_spl_offset_and_units():
    off = spl_offset_from_calibrator(-30.0, 94.0)
    assert off == 124.0
    assert Calibration().unit == "dBFS"
    assert Calibration(spl_offset_db=off).unit == "dB SPL"
