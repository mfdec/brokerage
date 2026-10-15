"""Room decay analysis: Schroeder integration, Lundeby noise handling, EDT/T20/T30/RT60, clarity, waterfall.

Method (ISO 3382 style):

1. Band-pass the impulse response (Butterworth octave / third-octave bands) and square it.
2. Find where the decay meets the noise floor with the Lundeby iteration (block-averaged dB curve,
   least-squares decay line, noise level from the region after the crossing point, repeat).
3. Schroeder backward integration up to that crossing point, with the expected noise energy
   subtracted and the extrapolated decay tail added, gives a smooth, noise-robust decay curve.
4. EDT (0..-10 dB), T20 (-5..-25 dB) and T30 (-5..-35 dB) are least-squares line fits of that curve,
   each scaled to a 60 dB decay.  Dynamic-range, curvature and bandwidth-time checks flag estimates
   that should not be trusted.

The cumulative spectral decay (CSD) "waterfall" shows the spectrum of the response remaining after
each slice time, which is how modal ringing in the bass shows up as ridges that fade slowly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

from .sweep import ImpulseResponse
from .units import band_centers, band_edge_factor, integrate_bands, log_grid

_TINY = 1e-60


def _db(p) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, _TINY))


# --------------------------------------------------------------------------- #
# Noise floor / truncation (Lundeby)
# --------------------------------------------------------------------------- #
@dataclass
class Truncation:
    index: int                 # samples after the peak that are used (decay meets noise here)
    noise_power: float         # mean-square noise floor
    slope_db_per_s: float      # fitted decay slope (negative); 0 if no decay was found
    dynamic_range_db: float    # peak block level above the noise floor
    reliable: bool


def lundeby(energy: np.ndarray, fs: int, n_per_10db: int = 5, max_iter: int = 12) -> Truncation:
    """Noise-floor estimate and truncation point of a squared impulse response (Lundeby et al., 1995)."""
    e = np.asarray(energy, dtype=np.float64)
    n = e.size
    if n < int(0.05 * fs):
        return Truncation(n, 0.0, 0.0, 0.0, False)
    noise = float(e[-max(int(0.1 * n), int(0.02 * fs)):].mean())
    floor_rel = 1e-14                          # ideal noise-free decays: never trust less than -140 dB
    blk_s, cross, prev = 0.01, n, -1
    slope, icpt, dyn = 0.0, 0.0, 0.0
    for it in range(max_iter):
        blk = max(int(round(blk_s * fs)), 4)
        nb = n // blk
        if nb < 8:
            break
        eb = e[: nb * blk].reshape(nb, blk).mean(axis=1)
        eb_db = _db(eb)
        tb = (np.arange(nb) + 0.5) * blk / fs
        i0 = int(np.argmax(eb_db[: max(nb // 4, 1)]))
        noise = max(noise, floor_rel * float(eb[i0]))
        noise_db = float(_db(noise))
        dyn = float(eb_db[i0] - noise_db)
        below = np.nonzero(eb_db[i0:] <= noise_db + 10.0)[0]
        i1 = i0 + (int(below[0]) if below.size else nb - 1)
        if i1 - i0 < 4:                      # less than ~10 dB of decay above the noise
            return Truncation(n, noise, 0.0, dyn, False)
        x, y = tb[i0:i1 + 1], eb_db[i0:i1 + 1]
        slope, icpt = np.polyfit(x, y, 1)
        if slope >= -0.5:
            return Truncation(n, noise, 0.0, dyn, False)
        cross = int(((noise_db - icpt) / slope) * fs)
        cross = min(max(cross, int(0.01 * fs)), n)
        # noise level from the region well after the crossing point
        start = cross + int(8.0 / abs(slope) * fs)
        if cross < n and start < n - int(0.02 * fs):
            noise = float(e[start:].mean())
        blk_s = float(np.clip(10.0 / abs(slope) / n_per_10db, 0.003, 0.2))
        if it >= 2 and abs(cross - prev) < blk:
            break
        prev = cross
    reliable = slope < 0 and dyn > 0
    return Truncation(int(cross), noise, float(slope), dyn, reliable)


def schroeder_db(energy: np.ndarray, trunc: Truncation, fs: int, compensate_noise: bool = True) -> np.ndarray:
    """Backward-integrated decay curve in dB (0 dB at the start), noise-compensated and tail-corrected."""
    n = max(int(trunc.index), 2)
    seg = np.asarray(energy[:n], dtype=np.float64)
    c = np.cumsum(seg[::-1])[::-1]
    if compensate_noise and trunc.noise_power > 0:
        c = c - trunc.noise_power * (n - np.arange(n))
    if trunc.reliable and trunc.slope_db_per_s < 0 and trunc.index < energy.size:
        c = c + trunc.noise_power * fs * 10.0 / (abs(trunc.slope_db_per_s) * np.log(10.0))   # extrapolated tail
    c = np.maximum(c, c[0] * 1e-12 if c[0] > 0 else _TINY)
    # noise compensation subtracts an *expected* energy, so enforce the physical requirement of a monotonic decay
    return np.minimum.accumulate(_db(c / c[0]))


def fit_decay_time(curve_db: np.ndarray, fs: int, hi_db: float, lo_db: float):
    """Line fit of the decay curve between ``hi_db`` and ``lo_db`` -> (T60 seconds, correlation) or (nan, nan)."""
    below_hi = np.nonzero(curve_db <= hi_db)[0]
    below_lo = np.nonzero(curve_db <= lo_db)[0]
    if below_hi.size == 0 or below_lo.size == 0:
        return np.nan, np.nan
    i0, i1 = int(below_hi[0]), int(below_lo[0])
    if i1 - i0 < 4:
        return np.nan, np.nan
    t = np.arange(i0, i1 + 1) / fs
    y = curve_db[i0:i1 + 1]
    slope, icpt = np.polyfit(t, y, 1)
    if slope >= 0:
        return np.nan, np.nan
    r = float(np.corrcoef(t, y)[0, 1])
    return float(-60.0 / slope), r


# --------------------------------------------------------------------------- #
# Per-band analysis
# --------------------------------------------------------------------------- #
@dataclass
class BandDecay:
    fc: float
    f_lo: float
    f_hi: float
    edt: float = np.nan
    t20: float = np.nan
    t30: float = np.nan
    t60: float = np.nan                 # best estimate (T30, else T20, else EDT)
    basis: str = ""
    dynamic_range_db: float = 0.0
    curvature_pct: float = np.nan       # 100*(T30/T20 - 1): >10 % means a non-exponential (curved) decay
    c50_db: float = np.nan
    c80_db: float = np.nan
    d50_pct: float = np.nan
    bt: float = np.nan                  # bandwidth x T60; >= ~16 is the usual reliability rule of thumb
    reliable: bool = False
    notes: list[str] = field(default_factory=list)
    curve_db: np.ndarray | None = None  # Schroeder curve (dB), fs = result.fs
    trunc: Truncation | None = None
    energy_db: np.ndarray | None = None # block-averaged energy-time curve for plotting (dB, 1 ms blocks)


def _bandpass(h: np.ndarray, fs: int, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    f_hi = min(f_hi, 0.49 * fs)
    sos = butter(order, [f_lo, f_hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, h)


def analyze_band(ir: ImpulseResponse, fc: float, f_lo: float, f_hi: float, filtered: bool = True) -> BandDecay:
    fs = ir.fs
    out = BandDecay(fc=fc, f_lo=f_lo, f_hi=f_hi)
    n_valid = ir.valid_samples(f_hi)
    if n_valid < int(0.05 * fs):
        out.notes.append("recording too short for this band")
        return out
    h = ir.h[: ir.t0 + n_valid]
    y = _bandpass(h, fs, f_lo, f_hi) if filtered else h
    e = y[ir.t0:] ** 2
    tr = lundeby(e, fs)
    out.trunc = tr
    out.dynamic_range_db = tr.dynamic_range_db
    curve = schroeder_db(e, tr, fs)
    out.curve_db = curve
    blk = max(int(0.001 * fs), 1)
    nb = e.size // blk
    out.energy_db = _db(e[: nb * blk].reshape(nb, blk).mean(axis=1))

    out.edt, _ = fit_decay_time(curve, fs, 0.0, -10.0)
    out.t20, _ = fit_decay_time(curve, fs, -5.0, -25.0)
    out.t30, _ = fit_decay_time(curve, fs, -5.0, -35.0)
    dr = tr.dynamic_range_db if tr.reliable else 0.0
    if np.isfinite(out.t30) and dr >= 45.0:
        out.t60, out.basis = out.t30, "T30"
    elif np.isfinite(out.t20) and dr >= 35.0:
        out.t60, out.basis = out.t20, "T20"
        out.notes.append("T30 not available (noise floor): showing T20")
    elif np.isfinite(out.edt):
        out.t60, out.basis = out.edt, "EDT"
        out.notes.append(f"only {dr:.0f} dB of decay above the noise: showing EDT")
    if np.isfinite(out.t20) and np.isfinite(out.t30) and out.t20 > 0:
        out.curvature_pct = 100.0 * (out.t30 / out.t20 - 1.0)
        if abs(out.curvature_pct) > 10.0:
            out.notes.append("curved decay (T30 differs from T20 by >10 %)")
    if np.isfinite(out.t60):
        out.bt = (f_hi - f_lo) * out.t60
        if out.bt < 16.0:
            out.notes.append(f"band too narrow for this decay time (B*T={out.bt:.0f} < 16): treat as indicative")
    # clarity / definition (noise-compensated energy ratios)
    n50, n80 = int(0.050 * fs), int(0.080 * fs)
    total = max(int(tr.index), n80 + 1)
    comp = np.maximum(e[:total] - tr.noise_power, 0.0)
    e_tot = comp.sum()
    if e_tot > 0:
        e50, e80 = comp[:n50].sum(), comp[:n80].sum()
        out.c50_db = float(_db(e50 / max(e_tot - e50, _TINY)))
        out.c80_db = float(_db(e80 / max(e_tot - e80, _TINY)))
        out.d50_pct = float(100.0 * e50 / e_tot)
    out.reliable = bool(out.basis in ("T20", "T30") and tr.reliable and (np.isnan(out.bt) or out.bt >= 16.0))
    return out


@dataclass
class DecayResult:
    fs: int
    bands: list[BandDecay]
    broadband: BandDecay
    bands_per_octave: int

    @property
    def centers(self) -> np.ndarray:
        return np.array([b.fc for b in self.bands])

    def t60_array(self) -> np.ndarray:
        return np.array([b.t60 for b in self.bands])

    def band_near(self, f: float) -> BandDecay:
        return self.bands[int(np.argmin(np.abs(np.log(self.centers / f))))]

    def t_mid(self, f_lo: float = 500.0, f_hi: float = 1000.0) -> float:
        """Mean RT60 of the 500 Hz and 1 kHz octave bands (ISO 'T_mid'), from the octave analysis."""
        vals = [b.t60 for b in self.bands if f_lo * 0.99 <= b.fc <= f_hi * 1.01 and np.isfinite(b.t60)]
        return float(np.mean(vals)) if vals else float("nan")

    def bass_ratio(self) -> float:
        """Beranek's bass ratio: (T125 + T250) / (T500 + T1000), > 1 means a boomy room."""
        try:
            t = [self.band_near(f).t60 for f in (125, 250, 500, 1000)]
        except ValueError:
            return float("nan")
        return float((t[0] + t[1]) / (t[2] + t[3])) if all(np.isfinite(t)) else float("nan")


def analyze_decay(ir: ImpulseResponse, bands_per_octave: int = 1, f_lo: float = 50.0, f_hi: float = 8000.0) -> DecayResult:
    """RT60 / EDT / clarity for every octave (1) or third-octave (3) band plus the broadband response."""
    fs = ir.fs
    f_hi = min(f_hi, 0.42 * fs)
    k = band_edge_factor(bands_per_octave)
    centers = band_centers(bands_per_octave, f_lo, f_hi)
    bands = [analyze_band(ir, float(fc), float(fc / k), float(fc * k)) for fc in centers]
    broadband = analyze_band(ir, float(np.sqrt(20.0 * min(20000.0, 0.45 * fs))), 20.0, min(20000.0, 0.45 * fs), filtered=False)
    return DecayResult(fs=fs, bands=bands, broadband=broadband, bands_per_octave=bands_per_octave)


# --------------------------------------------------------------------------- #
# Cumulative spectral decay (waterfall)
# --------------------------------------------------------------------------- #
@dataclass
class Waterfall:
    freqs: np.ndarray            # (F,) Hz
    slices_db: np.ndarray        # (S, F) dB (gain of the windowed response remaining after each slice time)
    times_s: np.ndarray          # (S,) slice start times after the direct sound
    window_s: float


def fractional_octave_mean(power: np.ndarray, df: float, centers, bands_per_octave: float) -> np.ndarray:
    """Mean power per bin inside 1/N-octave windows (smoothing of a spectrum, not band *power*).

    Windows narrower than one FFT bin are widened to one bin.  ``bands_per_octave <= 0`` interpolates.
    """
    fc = np.asarray(centers, dtype=np.float64)
    if bands_per_octave <= 0:
        return np.interp(fc, np.arange(power.size) * df, power)
    k = band_edge_factor(bands_per_octave)
    lo, hi = fc / k, fc * k
    used = np.maximum(hi - lo, df)
    mid = 0.5 * (hi + lo)
    return integrate_bands(power, df, np.maximum(mid - 0.5 * used, 0.5 * df), mid + 0.5 * used) / (used / df)


def csd_waterfall(
    ir: ImpulseResponse,
    n_slices: int = 40,
    dt_s: float = 0.010,
    window_s: float = 0.40,
    rise_s: float = 0.003,
    f_lo: float = 20.0,
    f_hi: float = 500.0,
    points_per_octave: int = 24,
    smoothing_oct: float = 12.0,
    level_offset_db: float = 0.0,
    mic_correction=None,
    scale: str = "log",
    max_columns: int = 400,
) -> Waterfall:
    """Spectra of the impulse response remaining after each slice time (Berman & Fincham CSD).

    ``scale="log"``: log-spaced columns from ``f_lo`` with fractional-octave smoothing.  ``scale="linear"``: columns evenly
    spaced from ``f_lo`` (may be 0 Hz) to ``f_hi``, each the mean power of the FFT bins around it (``smoothing_oct`` is ignored).
    """
    fs = ir.fs
    h = ir.h
    t0 = ir.t0
    n_w = int(window_s * fs)
    rise = int(min(rise_s * fs, t0))
    n_fft = int(2 ** np.ceil(np.log2(2 * (n_w + rise))))
    fall = 0.5 + 0.5 * np.cos(np.pi * np.arange(n_w) / n_w)                      # 1 -> 0 over the window
    win = np.concatenate((0.5 - 0.5 * np.cos(np.pi * np.arange(rise) / max(rise, 1)), fall))
    df = fs / n_fft
    if scale == "linear":
        step = max((f_hi - f_lo) / max(max_columns - 1, 1), df)
        freqs = np.arange(f_lo, f_hi + 0.5 * step, step)
        edge_lo, edge_hi = np.maximum(freqs - 0.5 * step, 0.0), freqs + 0.5 * step
    else:
        freqs = log_grid(f_lo, f_hi, points_per_octave)
    slices = np.empty((n_slices, freqs.size))
    for s in range(n_slices):
        start = t0 - rise + int(round(s * dt_s * fs))
        seg = np.zeros(win.size)
        lo, hi = max(start, 0), min(start + win.size, h.size)
        if hi > lo:
            seg[lo - start: hi - start] = h[lo:hi]
        spec = np.fft.rfft(seg * win, n_fft)
        p = spec.real ** 2 + spec.imag ** 2
        if scale == "linear":
            slices[s] = _db(integrate_bands(p, df, edge_lo, edge_hi) / ((edge_hi - edge_lo) / df))
        else:
            slices[s] = _db(fractional_octave_mean(p, df, freqs, smoothing_oct))
    slices += level_offset_db
    if mic_correction is not None:
        slices -= np.asarray(mic_correction(freqs))[None, :]
    return Waterfall(freqs=freqs, slices_db=slices, times_s=np.arange(n_slices) * dt_s, window_s=window_s)
