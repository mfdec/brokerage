"""Synthetic room for simulation and verification: impulse responses with *known* decay times.

The impulse response is the sum of three physically-motivated parts:

* direct sound: windowed-sinc fractional delay, 1/r amplitude;
* bass region (below the crossover): the exact modal Green's function of the rectangular room
  (``core.modes.forced_response``), so peaks, dips and per-position level differences are right;
* diffuse tail (above the crossover): Gaussian noise shaped per octave band with a per-band
  exponential envelope exp(-6.91 t / T60(f)).

Levels: the IR is a *gain* (output/input) so its DFT bins are the transfer function.  The direct path
has gain 1/r.  By Parseval the mean |DFT|^2 per bin of a time-domain tail equals its energy, so giving the
tail energy E_rev = E_dir (r/r_c)^2 = 1/r_c^2 makes the direct-to-reverberant ratio (r_c/r)^2 as for a
diffuse field; the modal part is scaled to the same per-bin level around the crossover so the two regions
join without a step.  Because T60(f) is specified up front, a measurement pipeline can be checked
against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .modes import RoomDims, critical_distance, forced_response, room_modes, schroeder_frequency
from .units import speed_of_sound

DEFAULT_T60_BANDS = {63.0: 0.65, 125.0: 0.55, 250.0: 0.45, 500.0: 0.40, 1000.0: 0.38,
                     2000.0: 0.36, 4000.0: 0.32, 8000.0: 0.26, 16000.0: 0.18}
_DIFFUSE_BANDS = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)


class T60Profile:
    """Reverberation time vs frequency (log-frequency interpolation, held flat outside the data)."""

    def __init__(self, bands: dict[float, float] | float = DEFAULT_T60_BANDS):
        if isinstance(bands, (int, float)):
            bands = {63.0: float(bands), 16000.0: float(bands)}
        self.freqs = np.array(sorted(bands))
        self.values = np.array([bands[f] for f in self.freqs], dtype=np.float64)

    def __call__(self, f):
        return np.interp(np.log(np.maximum(np.asarray(f, dtype=np.float64), 1.0)), np.log(self.freqs), self.values)

    def mean(self, f_lo: float = 250.0, f_hi: float = 4000.0) -> float:
        return float(np.mean(self(np.geomspace(f_lo, f_hi, 16))))


def fractional_delay_impulse(n: int, delay_samples: float, amp: float = 1.0, taps: int = 48) -> np.ndarray:
    """Band-limited impulse at a non-integer sample position (Hann-windowed sinc)."""
    out = np.zeros(n)
    k0 = int(np.floor(delay_samples))
    k = np.arange(k0 - taps // 2 + 1, k0 + taps // 2 + 1)
    k = k[(k >= 0) & (k < n)]
    x = k - delay_samples
    out[k] = amp * np.sinc(x) * (0.5 + 0.5 * np.cos(np.pi * x / (taps / 2)))
    return out


@dataclass
class SyntheticRoom:
    dims: RoomDims
    source: tuple[float, float, float]
    t60: T60Profile = field(default_factory=T60Profile)
    temp_c: float = 20.0
    seed: int = 0
    crossover_hz: float | None = None          # modal -> diffuse hand-over (default from the Schroeder frequency)

    def __post_init__(self):
        self.c = speed_of_sound(self.temp_c)
        self.src = np.asarray(self.source, dtype=np.float64)
        t_mid = self.t60.mean()
        self.schroeder_hz = float(schroeder_frequency(t_mid, self.dims.volume))
        self.f_x = float(self.crossover_hz or np.clip(1.5 * self.schroeder_hz, 220.0, 420.0))
        self.f_modal_max = 1.35 * self.f_x
        self.modes = room_modes(self.dims, self.f_modal_max, self.c)
        self.r_crit = float(critical_distance(self.dims.volume, t_mid))
        self._modal_scale: float | None = None

    # ------------------------------------------------------------------ building blocks
    def distance(self, rcv) -> float:
        return float(max(np.linalg.norm(np.asarray(rcv, dtype=np.float64) - self.src), 0.25))

    def _modal_raw(self, rcv, freqs) -> np.ndarray:
        return forced_response(self.dims, self.modes, self.src, np.atleast_2d(rcv), freqs, self.t60, self.c)[0]

    def _crossfade(self, freqs: np.ndarray, kind: str) -> np.ndarray:
        """Power-complementary low-pass (modal) / high-pass (diffuse) weights around the crossover."""
        lo, hi = 0.75 * self.f_x, 1.35 * self.f_x
        ramp = np.clip((np.log(np.maximum(freqs, 1e-3)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0.0, 1.0)
        return np.cos(0.5 * np.pi * ramp) if kind == "lp" else np.sin(0.5 * np.pi * ramp)

    @property
    def modal_scale(self) -> float:
        """Gain that makes the modal region as loud (per bin, on average) as the diffuse region, 1/r_c."""
        if self._modal_scale is None:
            rng = np.random.default_rng(1234)
            pts = rng.uniform(0.15, 0.85, (24, 3)) * self.dims.as_array()
            f = np.linspace(0.55 * self.f_x, 1.0 * self.f_x, 160)
            mean_sq = np.mean([np.mean(np.abs(self._modal_raw(p, f)) ** 2) for p in pts])
            self._modal_scale = float(1.0 / (self.r_crit * np.sqrt(max(mean_sq, 1e-30))))
        return self._modal_scale

    # ------------------------------------------------------------------ outputs
    def frequency_response(self, rcv, freqs) -> np.ndarray:
        """Complex low-frequency response (direct + modal) at ``rcv``; excludes the random diffuse tail."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        r = self.distance(rcv)
        direct = np.exp(-2j * np.pi * freqs * r / self.c) / r
        modal = self.modal_scale * self._modal_raw(rcv, freqs) * self._crossfade(freqs, "lp")
        return direct + modal

    def ir(self, rcv, fs: int, length_s: float = 2.5) -> np.ndarray:
        n = int(round(length_s * fs))
        n_fft = int(2 ** np.ceil(np.log2(n)))
        t = np.arange(n_fft) / fs
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        r = self.distance(rcv)
        t_dir = r / self.c
        direct = fractional_delay_impulse(n_fft, t_dir * fs, 1.0 / r)

        # diffuse tail: octave-band noise, each band with its own exponential decay
        key = tuple(np.round(np.asarray(rcv, dtype=np.float64), 2))
        rng = np.random.default_rng([self.seed, abs(hash(key)) % (2 ** 31)])
        noise = rng.standard_normal(n_fft)
        tau = np.maximum(t - t_dir, 0.0)
        tail = np.zeros(n_fft)
        for fc in _DIFFUSE_BANDS:
            if fc >= 0.45 * fs:
                continue
            sos = signal.butter(4, [fc / np.sqrt(2), min(fc * np.sqrt(2), 0.49 * fs)], btype="band", fs=fs, output="sos")
            tail += signal.sosfiltfilt(sos, noise) * np.exp(-6.9078 * tau / float(self.t60(fc)))
        tail *= 1.0 - np.exp(-tau / 0.004)                           # reflections build up over a few ms
        spec_t = np.fft.rfft(tail)
        band = (freqs >= self.f_x) & (freqs <= 2.0 * self.f_x)
        spec_t *= (1.0 / self.r_crit) / np.sqrt(np.mean(np.abs(spec_t[band]) ** 2))   # mean |T|^2 = 1/r_c^2 per bin
        spec_t *= self._crossfade(freqs, "hp")

        modal = np.zeros(freqs.size, dtype=np.complex128)
        sel = freqs <= self.f_modal_max
        modal[sel] = self.modal_scale * self._modal_raw(rcv, freqs[sel]) * self._crossfade(freqs[sel], "lp")
        h = np.fft.irfft(np.fft.rfft(direct) + spec_t + modal, n_fft)
        return h[:n]
