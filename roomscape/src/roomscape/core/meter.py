"""Streaming broadband level meter: RMS with time weighting, peak, clip detection, Leq."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, sosfilt

from .units import power_to_db
from .weighting import weighting_sos

CLIP_LEVEL = 0.999


@dataclass
class MeterReading:
    rms_db: float        # RMS of the latest chunk (dBFS)
    tw_db: float         # exponentially time-weighted level (dBFS)
    peak_db: float       # sample peak of the latest chunk (dBFS)
    leq_db: float        # equivalent continuous level since reset (dBFS)
    clipped: bool


class LevelMeter:
    """Feed consecutive audio chunks; time weighting is per-sample exact across chunk boundaries."""

    def __init__(self, fs: float, weighting: str = "Z", time_const_s: float = 0.125):
        self.fs = float(fs)
        self.set_weighting(weighting)
        self.set_time_constant(time_const_s)
        self.reset()

    def set_weighting(self, weighting: str) -> None:
        self.weighting = weighting
        self._sos = weighting_sos(self.fs, weighting)
        self._zi = None if self._sos is None else np.zeros((self._sos.shape[0], 2))

    def set_time_constant(self, tau: float) -> None:
        self.tau = float(tau)
        self._alpha = float(np.exp(-1.0 / (self.fs * self.tau)))
        self._tw_zi = None

    def reset(self) -> None:
        self._tw_zi = None
        self._sum_sq = 0.0
        self._count = 0
        if self._sos is not None:
            self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, x: np.ndarray) -> MeterReading:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return MeterReading(-200.0, -200.0, -200.0, -200.0, False)
        peak = float(np.max(np.abs(x)))
        clipped = peak >= CLIP_LEVEL
        y = x
        if self._sos is not None:
            y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        sq = y * y
        a = self._alpha
        if self._tw_zi is None:
            self._tw_zi = np.array([a * float(sq[0])])
        tw, self._tw_zi = lfilter([1.0 - a], [1.0, -a], sq, zi=self._tw_zi)
        self._sum_sq += float(sq.sum())
        self._count += sq.size
        return MeterReading(
            rms_db=float(power_to_db(sq.mean())),
            tw_db=float(power_to_db(tw[-1])),
            peak_db=float(power_to_db(peak * peak)),
            leq_db=float(power_to_db(self._sum_sq / max(self._count, 1))),
            clipped=clipped,
        )
