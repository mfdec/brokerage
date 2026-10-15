"""One complete room measurement: recording -> impulse response -> decay, response and waterfall."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .calibration import Calibration
from .decay import DecayResult, Waterfall, analyze_decay, csd_waterfall, fractional_octave_mean
from .sweep import ImpulseResponse, SweepDeconvolver, SweepSpec
from .units import log_grid, power_to_db


class MeasurementError(RuntimeError):
    """The recording does not contain a usable sweep (speaker off, wrong mic, ...)."""


@dataclass
class FrequencyResponse:
    freqs: np.ndarray
    level_db: np.ndarray          # level of a sine at the sweep amplitude, after the system (dBFS or dB SPL)
    unit: str
    window_s: float
    smoothing_oct: float


def level_offset_db(spec: SweepSpec, cal: Calibration) -> float:
    """dB added to 20*log10|H| to express the response as the level of a sine at the sweep amplitude."""
    return float(20.0 * np.log10(spec.amplitude / np.sqrt(2.0)) + cal.offset)


def frequency_response(
    ir: ImpulseResponse, spec: SweepSpec, cal: Calibration, window_s: float = 0.5, smoothing_oct: float = 6.0,
    f_lo: float = 20.0, f_hi: float = 20000.0, points_per_octave: int = 48,
) -> FrequencyResponse:
    """Windowed-IR frequency response (``window_s`` after the direct sound; short windows give a quasi-anechoic view)."""
    fs, t0 = ir.fs, ir.t0
    n_w = max(min(int(window_s * fs), ir.h.size - t0), 16)
    seg = ir.h[: t0 + n_w].copy()
    n_fall = max(int(0.15 * n_w), 8)
    seg[-n_fall:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n_fall) / n_fall)
    n_fft = int(2 ** np.ceil(np.log2(4 * seg.size)))
    spec_ = np.fft.rfft(seg, n_fft)
    p = spec_.real ** 2 + spec_.imag ** 2
    f_hi = min(f_hi, 0.46 * fs)
    grid = log_grid(f_lo, f_hi, points_per_octave)
    lvl = power_to_db(fractional_octave_mean(p, fs / n_fft, grid, smoothing_oct)) + level_offset_db(spec, cal)
    if cal.mic is not None:
        lvl = lvl - cal.mic.correction_db(grid)
    return FrequencyResponse(grid, lvl, cal.unit, window_s, smoothing_oct)


def bass_response(ir: ImpulseResponse, spec: SweepSpec, cal: Calibration, f_lo: float = 20.0, f_hi: float = 320.0,
                  window_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Unsmoothed low-frequency response (linear frequency axis, ~0.2 Hz steps) for mode analysis."""
    fs, t0 = ir.fs, ir.t0
    n_w = max(min(int(window_s * fs), ir.h.size - t0), 16)
    seg = ir.h[: t0 + n_w].copy()
    n_fall = max(int(0.15 * n_w), 8)
    seg[-n_fall:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n_fall) / n_fall)
    n_fft = int(2 ** np.ceil(np.log2(max(4 * fs, 2 * seg.size))))
    h = np.fft.rfft(seg, n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    sel = (f >= f_lo) & (f <= f_hi)
    lvl = 20.0 * np.log10(np.maximum(np.abs(h[sel]), 1e-30)) + level_offset_db(spec, cal)
    if cal.mic is not None:
        lvl = lvl - cal.mic.correction_db(f[sel])
    return f[sel], lvl


@dataclass
class MeasurementResult:
    spec: SweepSpec
    ir: ImpulseResponse
    decay: DecayResult
    response: FrequencyResponse
    calibration: Calibration
    snr_db: float
    created: float = field(default_factory=time.time)
    label: str = ""
    position: tuple[float, float] | None = None       # (x, y) metres in the room, set for walk-the-room pins
    warnings: list[str] = field(default_factory=list)

    def bass(self, f_hi: float = 320.0, window_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        return bass_response(self.ir, self.spec, self.calibration, f_hi=f_hi, window_s=window_s)

    def waterfall(self, **kw) -> Waterfall:
        """Cumulative spectral decay of this measurement (see ``decay.csd_waterfall`` for parameters)."""
        kw.setdefault("level_offset_db", level_offset_db(self.spec, self.calibration))
        if self.calibration.mic is not None:
            kw.setdefault("mic_correction", self.calibration.mic.correction_db)
        return csd_waterfall(self.ir, **kw)

    def summary(self) -> str:
        d = self.decay
        bb = d.broadband
        t_mid = d.t_mid()
        parts = [f"RT60 (broadband) {bb.t60:.2f} s ({bb.basis})" if np.isfinite(bb.t60) else "RT60 n/a"]
        if np.isfinite(t_mid):
            parts.append(f"T_mid {t_mid:.2f} s")
        br = d.bass_ratio()
        if np.isfinite(br):
            parts.append(f"bass ratio {br:.2f}")
        parts.append(f"S/N {self.snr_db:.0f} dB")
        return "   |   ".join(parts)


def analyze_recording(
    recording: np.ndarray, spec: SweepSpec, calibration: Calibration | None = None, bands_per_octave: int = 1,
    label: str = "", position: tuple[float, float] | None = None,
) -> MeasurementResult:
    cal = calibration or Calibration()
    rec = np.asarray(recording, dtype=np.float64)
    warnings: list[str] = []
    if np.max(np.abs(rec)) >= 0.999:
        warnings.append("The input clipped during the sweep: lower the sweep level / mic input level and measure again.")
    ir = SweepDeconvolver(spec).extract(rec)
    h = ir.h
    peak = float(np.abs(h[ir.t0]))
    # noise reference: last stretch of the *valid* impulse response (the end of the raw convolution holds no data)
    valid = max(ir.valid_samples(spec.f2_eff), 1024)
    seg = h[ir.t0 + max(valid - max(int(0.2 * valid), 512), 1): ir.t0 + valid]
    noise_rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-30
    snr = float(20.0 * np.log10(peak / noise_rms))
    if snr < 20.0:
        raise MeasurementError(
            f"No clear sweep found (the strongest response is only {snr:.0f} dB above the noise). Check that the speaker "
            "is on and the volume is up, that the output goes to the room, and that the right microphone is selected."
        )
    if snr < 40.0:
        warnings.append(f"Low signal-to-noise ratio ({snr:.0f} dB): T30 may not be available. Raise the level or reduce noise.")
    decay = analyze_decay(ir, bands_per_octave=bands_per_octave, f_lo=30.0, f_hi=8000.0)
    bb = decay.broadband
    if bb.trunc is not None and bb.trunc.index >= 0.95 * (ir.h.size - ir.t0) and np.isfinite(bb.t60) and bb.t60 > 0.6 * spec.tail_s:
        warnings.append("The decay was still running when the recording ended: increase the tail length.")
    resp = frequency_response(ir, spec, cal)
    return MeasurementResult(spec=spec, ir=ir, decay=decay, response=resp, calibration=cal, snr_db=snr,
                             label=label, position=position, warnings=warnings)
