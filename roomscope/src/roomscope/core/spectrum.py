"""Level-accurate FFT spectrum analysis and the live RTA engine.

Scaling conventions (all levels are RMS, referenced to digital full scale):

* ``power_spectrum`` returns a *Parseval-consistent* one-sided power per FFT bin:
  summing it over a frequency range gives that range's mean-square signal power,
  independent of FFT size and window.  That is what a true RTA band level is.
* A steady sine of peak amplitude A reads ``A**2 / 2`` summed over its bins, i.e.
  ``20*log10(A/sqrt(2))`` dBFS.  A full-scale sine therefore reads -3.01 dBFS.
* The "FFT" display multiplies the per-bin power by the window's equivalent noise
  bandwidth so a tone's *peak bin* also reads its true RMS level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import fft as sfft
from scipy.signal import get_window

from .calibration import Calibration
from .units import (
    band_centers,
    fractional_octave_power,
    log_grid,
    power_to_db,
)
from .weighting import weighting_db

WINDOWS = {
    "Hann": "hann",
    "Blackman-Harris": "blackmanharris",
    "Flat-top": "flattop",
    "Hamming": "hamming",
    "Rectangular": "boxcar",
}

RESOLUTIONS = ["FFT", "1/1", "1/3", "1/6", "1/12", "1/24", "1/48"]


@lru_cache(maxsize=32)
def window_info(name: str, n: int):
    """(window, sum(w), sum(w^2), ENBW in bins) for a periodic spectral-analysis window."""
    w = get_window(WINDOWS.get(name, name), n, fftbins=True).astype(np.float64)
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    return w, s1, s2, n * s2 / (s1 * s1)


def power_spectrum(x: np.ndarray, window: str = "Hann") -> tuple[np.ndarray, float]:
    """One-sided, Parseval-consistent power per bin and the window ENBW (in bins)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    w, _s1, s2, enbw = window_info(window, n)
    spec = sfft.rfft(x * w)
    p = (spec.real * spec.real + spec.imag * spec.imag) * (2.0 / (n * s2))
    p[0] *= 0.5
    if n % 2 == 0:
        p[-1] *= 0.5
    return p, enbw


def bin_freqs(n_fft: int, fs: float) -> np.ndarray:
    return np.arange(n_fft // 2 + 1) * (fs / n_fft)


class SpectrumAverager:
    """Exponential power averaging with optional decaying peak hold."""

    def __init__(self, time_const_s: float = 0.5, peak_decay_db_per_s: float = 8.0):
        self.time_const_s = float(time_const_s)
        self.peak_decay_db_per_s = float(peak_decay_db_per_s)
        self.avg: np.ndarray | None = None
        self.peak: np.ndarray | None = None

    def reset(self) -> None:
        self.avg = None
        self.peak = None

    def update(self, power: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self.avg is None or self.avg.shape != power.shape:
            self.avg = power.copy()
            self.peak = power.copy()
            return self.avg, self.peak
        if self.time_const_s <= 0:
            self.avg = power.copy()
        else:
            a = float(np.exp(-max(dt, 0.0) / self.time_const_s))
            self.avg = a * self.avg + (1.0 - a) * power
        decay = 10.0 ** (-self.peak_decay_db_per_s * max(dt, 0.0) / 10.0)
        self.peak = np.maximum(self.peak * decay, self.avg)
        return self.avg, self.peak


@dataclass
class RtaSettings:
    fft_size: int = 16384
    window: str = "Hann"
    averaging_s: float = 0.5          # 0 = no averaging
    resolution: str = "1/12"          # one of RESOLUTIONS
    weighting: str = "Z"              # Z (flat), A or C
    f_min: float = 20.0
    f_max: float = 20000.0


@dataclass
class RtaFrame:
    freqs: np.ndarray
    inst_db: np.ndarray
    avg_db: np.ndarray
    peak_db: np.ndarray
    kind: str                         # "line" or "bars"
    unit: str                         # "dBFS" or "dB SPL"
    overall_db: float                 # level summed over [f_min, f_max]
    resolution_hz: float              # FFT bin spacing


class RtaEngine:
    """Turns raw audio into calibrated, smoothed spectra for display."""

    def __init__(self, fs: float, settings: RtaSettings | None = None, calibration: Calibration | None = None):
        self.fs = float(fs)
        self.settings = settings or RtaSettings()
        self.calibration = calibration or Calibration()
        self._averager = SpectrumAverager(self.settings.averaging_s)
        self._cache_key = None
        self._cache: dict = {}

    # -- configuration ---------------------------------------------------- #
    def set_settings(self, s: RtaSettings) -> None:
        self.settings = s
        self._averager.time_const_s = s.averaging_s
        self._averager.reset()

    def set_calibration(self, c: Calibration) -> None:
        self.calibration = c
        self._cache_key = None
        self._averager.reset()

    def reset(self) -> None:
        self._averager.reset()

    # -- processing -------------------------------------------------------- #
    def _prepare(self, n_fft: int) -> dict:
        s = self.settings
        key = (n_fft, self.fs, s.window, s.resolution, s.weighting, s.f_min, s.f_max,
               id(self.calibration.mic), )
        if key == self._cache_key:
            return self._cache
        freqs = bin_freqs(n_fft, self.fs)
        df = self.fs / n_fft
        _w, _s1, _s2, enbw = window_info(s.window, n_fft)
        gain_db = weighting_db(freqs, s.weighting)
        if self.calibration.mic is not None:
            gain_db = gain_db - self.calibration.mic.correction_db(freqs)
        gain = 10.0 ** (gain_db / 10.0)
        f_max = min(s.f_max, 0.5 * self.fs)
        if s.resolution == "FFT":
            sel = (freqs >= s.f_min) & (freqs <= f_max)
            grid, kind, frac = freqs[sel], "line", 0
        elif s.resolution in ("1/1", "1/3"):
            frac = int(s.resolution.split("/")[1])
            grid, kind = band_centers(frac, s.f_min, f_max), "bars"
        else:
            frac = int(s.resolution.split("/")[1])
            grid, kind = log_grid(s.f_min, f_max, max(48, 2 * frac)), "line"   # >=2x band overlap
        self._cache = dict(freqs=freqs, df=df, enbw=enbw, gain=gain, grid=grid, kind=kind,
                           frac=frac, sel=(freqs >= s.f_min) & (freqs <= f_max))
        self._cache_key = key
        return self._cache

    def _display(self, p: np.ndarray, c: dict) -> np.ndarray:
        if c["kind"] == "line" and c["frac"] == 0:        # FFT bins, tone-accurate
            out = power_to_db(p[c["sel"]] * c["enbw"])
        else:
            bw = c["enbw"] * c["df"]
            out = power_to_db(fractional_octave_power(p, c["df"], c["grid"], c["frac"], min_width_hz=bw))
        return out + self.calibration.offset

    def process(self, samples: np.ndarray, dt: float) -> RtaFrame:
        n = self.settings.fft_size
        x = np.asarray(samples, dtype=np.float64)
        if x.size < n:
            x = np.concatenate((np.zeros(n - x.size), x))
        x = x[-n:]
        x = x - x.mean()
        p, _ = power_spectrum(x, self.settings.window)
        c = self._prepare(n)
        p = p * c["gain"]
        avg, peak = self._averager.update(p, dt)
        overall = power_to_db(p[c["sel"]].sum()) + self.calibration.offset
        return RtaFrame(
            freqs=c["grid"], inst_db=self._display(p, c), avg_db=self._display(avg, c),
            peak_db=self._display(peak, c), kind=c["kind"], unit=self.calibration.unit,
            overall_db=float(overall), resolution_hz=c["df"],
        )


def tone_level_dbfs(x: np.ndarray, fs: float, f0: float, window: str = "Flat-top", half_width_hz: float = 30.0) -> float:
    """RMS level (dBFS) of the tone near ``f0`` (used by the SPL calibrator wizard)."""
    p, enbw = power_spectrum(np.asarray(x, dtype=np.float64) - np.mean(x), window)
    freqs = bin_freqs(len(x), fs)
    sel = (freqs >= f0 - half_width_hz) & (freqs <= f0 + half_width_hz)
    if not sel.any():
        raise ValueError("Analysis band contains no FFT bins")
    return float(power_to_db(p[sel].sum()))
