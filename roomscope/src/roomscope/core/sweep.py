"""Exponential sine sweep (Farina) measurement: stimulus, inverse filter, impulse-response extraction.

A logarithmic sweep pushes harmonic distortion products to *negative* time after deconvolution, so
the linear impulse response is clean even through a distorting speaker; the recording only has to
contain the sweep plus the room's decay.  The direct-sound peak is found automatically, so the
playback and capture do not need to be sample-synchronous (they may even be separate devices).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class SweepSpec:
    fs: int = 48000
    f1: float = 20.0
    f2: float = 20000.0
    duration: float = 6.0            # nominal seconds (adjusted a hair so the sweep ends on a zero crossing)
    level_dbfs: float = -12.0        # peak level of the played sweep
    fade_in_s: float = 0.010
    fade_out_s: float = 0.002
    tail_s: float = 2.5              # keep recording this long after the sweep to capture the room decay

    @property
    def f2_eff(self) -> float:
        return float(min(self.f2, 0.46 * self.fs))

    @property
    def amplitude(self) -> float:
        return float(10.0 ** (self.level_dbfs / 20.0))


def _adjusted_duration(f1: float, f2: float, duration: float) -> float:
    """Nearest duration for which the sweep's final phase is a multiple of pi (ends on a zero crossing)."""
    ln_r = np.log(f2 / f1)
    m = max(round(duration * 2.0 * (f2 - f1) / ln_r), 1)
    return float(m * ln_r / (2.0 * (f2 - f1)))


def sweep_unit(spec: SweepSpec) -> np.ndarray:
    """The sweep with unit peak amplitude (float64)."""
    f1, f2 = spec.f1, spec.f2_eff
    T = _adjusted_duration(f1, f2, spec.duration)
    n = int(round(T * spec.fs))
    t = np.arange(n) / spec.fs
    L = T / np.log(f2 / f1)
    x = np.sin(2.0 * np.pi * f1 * L * (np.exp(t / L) - 1.0))
    n_in, n_out = int(spec.fade_in_s * spec.fs), int(spec.fade_out_s * spec.fs)
    if n_in > 1:
        x[:n_in] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    if n_out > 1:
        x[-n_out:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n_out) / n_out)
    return x


def sweep_signal(spec: SweepSpec) -> np.ndarray:
    """The sweep as it should be played (float32 at ``spec.level_dbfs``)."""
    return (spec.amplitude * sweep_unit(spec)).astype(np.float32)


def recording_length(spec: SweepSpec, lead_s: float = 0.5, margin_s: float = 0.6) -> int:
    """Samples to record: lead-in (for latency), the sweep, its decay tail and a latency margin."""
    return int(round((lead_s + sweep_unit(spec).size / spec.fs + spec.tail_s + margin_s) * spec.fs))


@dataclass
class ImpulseResponse:
    """Linear impulse response (system gain: output / input) with the direct-sound peak at ``h[t0]``."""

    h: np.ndarray
    fs: int
    t0: int
    latency_samples: int = 0            # direct-sound arrival relative to the start of the recording
    rec_len: int = 0                    # length of the recording it came from
    f1: float = 20.0                    # sweep start frequency (for per-band valid lengths)
    sweep_rate_L: float = 0.0           # L = T / ln(f2/f1) of the sweep, 0 if unknown

    @property
    def duration(self) -> float:
        return (self.h.size - self.t0) / self.fs

    def valid_samples(self, f_hi: float) -> int:
        """Samples after the peak whose content at frequencies up to ``f_hi`` is fully observed.

        The sweep reaches frequency f at time L*ln(f/f1); the room's response to it is only recorded up to
        the end of the recording, so low frequencies keep a much longer valid decay than the top octave.
        """
        full = self.h.size - self.t0
        if self.sweep_rate_L <= 0 or self.rec_len <= 0:
            return full
        t_f = self.sweep_rate_L * np.log(max(f_hi, self.f1) / self.f1)
        avail = self.rec_len - self.latency_samples - int(t_f * self.fs)
        return int(max(0, min(full, avail)))

    @classmethod
    def from_array(cls, h: np.ndarray, fs: int, t0: int | None = None) -> "ImpulseResponse":
        h = np.asarray(h, dtype=np.float64)
        return cls(h=h, fs=int(fs), t0=int(np.argmax(np.abs(h))) if t0 is None else int(t0))


class SweepDeconvolver:
    """Pre-computes the inverse filter and gain normalisation for a sweep specification."""

    def __init__(self, spec: SweepSpec):
        self.spec = spec
        s = sweep_unit(spec)
        self.n = s.size
        T = self.n / spec.fs
        self.L = T / np.log(spec.f2_eff / spec.f1)
        t = np.arange(self.n) / spec.fs
        self.inv = s[::-1] * np.exp(-t / self.L)               # time-reversed, +6 dB/oct envelope (flat product)
        n_fft = int(2 ** np.ceil(np.log2(2 * self.n)))
        f = np.fft.rfftfreq(n_fft, 1.0 / spec.fs)
        prod = np.abs(np.fft.rfft(s, n_fft) * np.fft.rfft(self.inv, n_fft))
        band = (f > 2.0 * spec.f1) & (f < 0.5 * spec.f2_eff)
        self.gain = float(np.median(prod[band]))               # |S * INV| is constant in band: sets IR units

    def deconvolve(self, recording: np.ndarray) -> np.ndarray:
        """Full linear convolution with the inverse filter; lag-0 of the impulse response is at index ``n-1``."""
        y = np.asarray(recording, dtype=np.float64)
        y = y - y.mean()
        return fftconvolve(y, self.inv, mode="full") / (self.gain * self.spec.amplitude)

    def extract(self, recording: np.ndarray, pre_s: float = 0.003, max_len_s: float | None = None) -> ImpulseResponse:
        """Impulse response from a recording of the played sweep.

        ``pre_s`` of pre-peak signal is kept (pre-ringing / noise reference).  The harmonic-distortion
        images sit hundreds of ms earlier and are excluded by construction.
        """
        fs = self.spec.fs
        full = self.deconvolve(recording)
        lag0 = self.n - 1
        search_from = max(lag0 - int(0.05 * fs), 0)
        p = search_from + int(np.argmax(np.abs(full[search_from:])))
        pre = min(int(pre_s * fs), p)
        end = full.size if max_len_s is None else min(full.size, p + int(max_len_s * fs))
        h = full[p - pre:end].copy()
        return ImpulseResponse(
            h=h, fs=fs, t0=pre, latency_samples=max(p - lag0, 0), rec_len=len(recording),
            f1=self.spec.f1, sweep_rate_L=float(self.L),
        )


def extract_impulse_response(recording: np.ndarray, spec: SweepSpec, **kw) -> ImpulseResponse:
    return SweepDeconvolver(spec).extract(recording, **kw)
