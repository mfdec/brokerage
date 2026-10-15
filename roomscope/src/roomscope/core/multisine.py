"""Periodic multisine stimulus and per-frame analysis for walk-the-room mapping.

Why a multisine instead of noise?  Its spectrum is *deterministic*: every analysis frame sees exactly the
same line amplitudes, so the only things that change a line's level are the room (position!) and noise.
The period is chosen so that an analysis frame of two periods puts the signal on the even FFT bins and leaves
the odd bins empty - those odd bins measure the noise (and the modulation caused by moving the microphone)
in the same frame, for free.

With the default 0.5 s period (2 Hz line spacing) a frame of 1.0 s has 1 Hz FFT bins:
lines on even bins, noise estimate on odd bins.  Magnitude-only analysis, so no synchronisation between
playback and capture is needed - they may even run on different sound cards (a clock mismatch of tens of ppm
is <2 % of a bin at bass frequencies).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Calibration
from .units import power_to_db


@dataclass(frozen=True)
class MultisineSpec:
    fs: int = 48000
    period_s: float = 0.5
    f_lo: float = 20.0
    f_hi: float = 300.0
    pink: bool = True                 # amplitude ~ f^-1/2 (more energy where speakers are weak and modes live)
    level_dbfs: float = -20.0         # peak level of the played signal

    @property
    def n_period(self) -> int:
        return int(round(self.period_s * self.fs))

    @property
    def spacing_hz(self) -> float:
        return self.fs / self.n_period

    @property
    def line_freqs(self) -> np.ndarray:
        df = self.spacing_hz
        k = np.arange(int(np.ceil(self.f_lo / df)), int(np.floor(self.f_hi / df)) + 1)
        return k * df


def line_params(spec: MultisineSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frequencies, amplitudes, phases) with x(t) = sum A_k cos(2 pi f_k t + theta_k), peak = ``level_dbfs``."""
    f = spec.line_freqs
    a = f ** -0.5 if spec.pink else np.ones_like(f)
    p = a ** 2 / np.sum(a ** 2)
    theta = 2.0 * np.pi * np.array([np.sum((k - np.arange(k)) * p[:k]) for k in range(f.size)])     # Schroeder phases
    t = np.arange(spec.n_period) / spec.fs
    x = (a[None, :] * np.cos(2.0 * np.pi * f[None, :] * t[:, None] + theta[None, :])).sum(axis=1)
    return f, a * (10.0 ** (spec.level_dbfs / 20.0) / np.max(np.abs(x))), theta


def generate(spec: MultisineSpec) -> np.ndarray:
    """One period of the stimulus (float32); play it in a loop."""
    f, a, theta = line_params(spec)
    t = np.arange(spec.n_period) / spec.fs
    return (a[None, :] * np.cos(2.0 * np.pi * f[None, :] * t[:, None] + theta[None, :])).sum(axis=1).astype(np.float32)


def crest_factor_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(20.0 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2))))


@dataclass
class FrameResult:
    freqs: np.ndarray             # line frequencies (Hz)
    power_db: np.ndarray          # calibrated level of each line (dBFS / dB SPL), noise-subtracted
    line_snr_db: np.ndarray       # line power over the neighbouring empty-bin noise
    snr_db: float                 # broadband: total line power over total noise power (same bandwidth)
    clipped: bool

    @property
    def usable(self) -> bool:
        return np.isfinite(self.snr_db)


def analyze_frame(x: np.ndarray, spec: MultisineSpec, cal: Calibration | None = None, periods: int = 2) -> FrameResult:
    """Line levels of ``periods`` whole stimulus periods of mic signal (rectangular window, exact bins)."""
    n = periods * spec.n_period
    x = np.asarray(x, dtype=np.float64)
    if x.size < n:
        raise ValueError(f"need {n} samples, got {x.size}")
    x = x[-n:]
    clipped = bool(np.max(np.abs(x)) >= 0.999)
    X = np.fft.rfft(x - x.mean())
    P = (X.real ** 2 + X.imag ** 2) * (2.0 / n ** 2)                        # mean-square per bin (sine A -> A^2/2)
    df = spec.fs / n
    bins = np.rint(spec.line_freqs / df).astype(int)
    step = bins[1] - bins[0] if bins.size > 1 else periods
    lines = P[bins]
    if step >= 2:                                                           # empty bins between the lines
        noise = 0.5 * (P[bins - 1] + P[np.minimum(bins + 1, P.size - 1)])
    else:                                                                   # no empty bins: cannot estimate noise
        noise = np.full_like(lines, np.nan)
    clean = np.maximum(lines - np.nan_to_num(noise, nan=0.0), 1e-30)
    snr = 10.0 * np.log10(np.maximum(lines, 1e-30) / np.maximum(noise, 1e-30))
    total = float(np.sum(lines))
    n_tot = float(np.sum(noise)) if np.all(np.isfinite(noise)) else float("nan")
    broad = float(10.0 * np.log10(max(total, 1e-30) / max(n_tot, 1e-30))) if np.isfinite(n_tot) else float("nan")
    cal = cal or Calibration()
    p_cal = cal.correct_power(spec.line_freqs, clean)
    return FrameResult(spec.line_freqs, power_to_db(p_cal) + cal.offset, snr, broad, clipped)
