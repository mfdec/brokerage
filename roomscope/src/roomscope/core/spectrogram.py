"""Multi-resolution log-frequency spectrogram frames and a scrolling history buffer.

Low frequencies need long windows to resolve room modes; high frequencies need short ones
to show timing.  Each frame therefore stitches three FFTs of different length onto one
log-frequency grid.  Levels are fractional-octave band power (RMS dB), so noise and tones
read consistently across the three resolution groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Calibration
from .spectrum import bin_freqs, power_spectrum, window_info
from .units import fractional_octave_power, log_grid, power_to_db

# (lower edge Hz, upper edge Hz, FFT length at 48 kHz)
MODES: dict[str, tuple[tuple[float, float, int], ...]] = {
    "Balanced": ((0.0, 150.0, 32768), (150.0, 1500.0, 8192), (1500.0, 1e9, 2048)),
    "Bass detail": ((0.0, 200.0, 65536), (200.0, 2000.0, 16384), (2000.0, 1e9, 4096)),
    "Fast": ((0.0, 300.0, 8192), (300.0, 3000.0, 4096), (3000.0, 1e9, 1024)),
}


def _pow2(x: float) -> int:
    return int(2 ** round(np.log2(max(x, 64))))


@dataclass(frozen=True)
class SpectrogramSettings:
    mode: str = "Balanced"
    f_min: float = 20.0
    f_max: float = 20000.0
    ppo: int = 24                     # grid points (and band fraction) per octave


class SpectrogramEngine:
    def __init__(self, fs: float, settings: SpectrogramSettings | None = None, calibration: Calibration | None = None):
        self.fs = float(fs)
        self.settings = settings or SpectrogramSettings()
        self.calibration = calibration or Calibration()
        f_max = min(self.settings.f_max, 0.49 * self.fs)
        self.grid = log_grid(self.settings.f_min, f_max, self.settings.ppo)
        self._plan = []
        for lo, hi, n0 in MODES[self.settings.mode]:
            n = _pow2(n0 * self.fs / 48000.0)
            idx = np.where((self.grid >= lo) & (self.grid < hi))[0]
            if idx.size:
                self._plan.append((n, idx))
        self._gain_key = None
        self._gains: dict[int, np.ndarray] = {}

    @property
    def required_samples(self) -> int:
        return max(n for n, _ in self._plan)

    def set_calibration(self, c: Calibration) -> None:
        self.calibration = c
        self._gain_key = None

    def _gain(self, n: int) -> np.ndarray:
        key = id(self.calibration.mic)
        if key != self._gain_key:
            self._gains.clear()
            self._gain_key = key
        g = self._gains.get(n)
        if g is None:
            mic = self.calibration.mic
            g = np.ones(n // 2 + 1) if mic is None else 10.0 ** (-mic.correction_db(bin_freqs(n, self.fs)) / 10.0)
            self._gains[n] = g
        return g

    def compute(self, x: np.ndarray) -> np.ndarray:
        """One spectrogram column (dB) from the newest ``required_samples`` samples (zero-padded if short)."""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(self.grid.size)
        for n, idx in self._plan:
            seg = x[-n:]
            if seg.size < n:
                seg = np.concatenate((np.zeros(n - seg.size), seg))
            p, enbw = power_spectrum(seg - seg.mean(), "Hann")
            p *= self._gain(n)
            df = self.fs / n
            band = fractional_octave_power(p, df, self.grid[idx], self.settings.ppo, min_width_hz=enbw * df)
            out[idx] = power_to_db(band) + self.calibration.offset
        return out


class SpectrogramHistory:
    """Newest-first scrolling history: ``data[0]`` is the latest column."""

    def __init__(self, n_freq: int, n_rows: int, hop_s: float, fill: float = -200.0):
        self.hop_s = float(hop_s)
        self.data = np.full((n_rows, n_freq), fill, dtype=np.float32)
        self.count = 0

    def push(self, column: np.ndarray) -> None:
        self.data[1:] = self.data[:-1]
        self.data[0] = column
        self.count += 1

    def slices(self, n_slices: int, span_s: float) -> tuple[np.ndarray, np.ndarray]:
        """``n_slices`` columns spread evenly over the newest ``span_s`` seconds; returns (levels, seconds_ago)."""
        stride = max(1, int(round(span_s / self.hop_s / max(n_slices, 1))))
        rows = np.arange(n_slices) * stride
        rows = rows[rows < self.data.shape[0]]
        return self.data[rows], rows * self.hop_s

    @property
    def span_s(self) -> float:
        return self.data.shape[0] * self.hop_s
