"""dB helpers, speed of sound and fractional-octave band utilities (pure NumPy)."""

from __future__ import annotations

import numpy as np

_P_FLOOR = 1e-20
_A_FLOOR = 1e-10


def power_to_db(p):
    """10*log10(power) with a -200 dB floor so silence never produces -inf."""
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=np.float64), _P_FLOOR))


def amp_to_db(a):
    """20*log10(|amplitude|) with a -200 dB floor."""
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(a, dtype=np.float64)), _A_FLOOR))


def db_to_power(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)


def db_to_amp(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def speed_of_sound(temp_c: float = 20.0) -> float:
    """Speed of sound in air (m/s); 343.4 m/s at 20 C."""
    return 331.3 + 0.606 * float(temp_c)


# --------------------------------------------------------------------------- #
# Fractional-octave bands
# --------------------------------------------------------------------------- #

# Nominal (preferred-number) 1/3-octave centre frequencies, 12.5 Hz .. 20 kHz.
_NOMINAL_THIRD = np.array(
    [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
     800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
)


def band_edge_factor(bands_per_octave: float) -> float:
    """Ratio upper_edge / centre (= centre / lower_edge) of a 1/N-octave band."""
    return 2.0 ** (1.0 / (2.0 * bands_per_octave))


def band_centers(bands_per_octave: int, f_lo: float, f_hi: float, ref: float = 1000.0) -> np.ndarray:
    """Exact base-2 band centres ``ref * 2**(i/N)`` inside [f_lo, f_hi]."""
    n = float(bands_per_octave)
    i_lo = int(np.ceil(n * np.log2(f_lo / ref) - 1e-9))
    i_hi = int(np.floor(n * np.log2(f_hi / ref) + 1e-9))
    return ref * 2.0 ** (np.arange(i_lo, i_hi + 1) / n)


def nominal_frequency(fc: float) -> float:
    """Snap an exact 1/3-octave centre to its preferred nominal value (e.g. 31.25 -> 31.5)."""
    j = int(np.argmin(np.abs(np.log(_NOMINAL_THIRD / fc))))
    return float(_NOMINAL_THIRD[j])


def format_freq(f: float) -> str:
    """Compact label: 31.5, 125, 1k, 1.25k, 16k."""
    if f >= 1000:
        v = f / 1000.0
        return f"{v:g}k" if abs(v - round(v, 2)) < 1e-9 else f"{v:.2f}k"
    return f"{f:g}"


def log_grid(f_lo: float, f_hi: float, points_per_octave: int) -> np.ndarray:
    """Log-spaced frequency grid including both end points (approximately)."""
    n = max(2, int(round(points_per_octave * np.log2(f_hi / f_lo))) + 1)
    return np.geomspace(f_lo, f_hi, n)


def integrate_bands(power: np.ndarray, df: float, lo_hz, hi_hz) -> np.ndarray:
    """Sum per-bin power over arbitrary frequency intervals with fractional bin edges.

    ``power[k]`` belongs to the bin centred on ``k * df`` and covering
    ``[(k-0.5) df, (k+0.5) df]``.  The result is the (linearly interpolated)
    area, so a tone's power is conserved wherever the interval boundaries fall
    in the empty part of the spectrum.
    """
    cum = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    x = np.arange(cum.size)
    lo = np.asarray(lo_hz, dtype=np.float64) / df + 0.5
    hi = np.asarray(hi_hz, dtype=np.float64) / df + 0.5
    return np.interp(hi, x, cum) - np.interp(lo, x, cum)


def fractional_octave_power(
    power: np.ndarray, df: float, f_centers, bands_per_octave: float, min_width_hz: float = 0.0
) -> np.ndarray:
    """Band power in 1/N-octave bands around ``f_centers`` (sliding or fixed centres).

    ``min_width_hz`` widens bands that are narrower than the FFT resolution so a
    tone at low frequency still reads its true level.
    """
    fc = np.asarray(f_centers, dtype=np.float64)
    k = band_edge_factor(bands_per_octave)
    lo, hi = fc / k, fc * k
    if min_width_hz > 0:
        width = hi - lo
        mid = 0.5 * (hi + lo)
        need = np.maximum(width, min_width_hz)
        lo, hi = mid - 0.5 * need, mid + 0.5 * need
    return integrate_bands(power, df, np.maximum(lo, 0.5 * df), hi)
