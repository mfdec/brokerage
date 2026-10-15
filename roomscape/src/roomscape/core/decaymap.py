"""Decay camera: how long the room rings at each spot, measured while you walk.

A repeating low-frequency sweep (default 20-300 Hz, 0.5 s of sweep followed by silence, one burst every 1.6 s) plays through the
speakers.  Every burst recorded by the walking microphone is deconvolved into an impulse response for that spot, decimated to a
1 kHz sample rate (the bass region is all that matters here) and kept together with the marker position and a quality figure.
Any band can then be analysed afterwards (Schroeder decay curve, EDT, T20): the map of decay times, and the energy *left* at a
chosen time after the burst, are rasterised from the stored frames on demand.  Played back as an animation, "energy left" shows where
the sound lingers.

Pure logic with no threads or devices, like ``mapping.WalkMapper``: feed positions (video rate) and audio chunks with timestamps from
one clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .calibration import Calibration
from .decay import analyze_band
from .fieldmap import FieldMap
from .modes import RoomDims
from .sweep import ImpulseResponse, SweepDeconvolver, SweepSpec, sweep_signal

FS_D = 1000                       # sample rate of the stored impulse responses (Hz)
T_GRID_DT = 0.05                  # spacing of the stored "energy left" time grid (s)
T_GRID_MAX = 1.5


# =============================================================================== stimulus
@dataclass(frozen=True)
class PulseSpec:
    fs: int = 48000
    f1: float = 20.0
    f2: float = 300.0
    sweep_s: float = 0.5
    period_s: float = 1.6
    level_dbfs: float = -18.0

    @property
    def sweep_spec(self) -> SweepSpec:
        return SweepSpec(fs=self.fs, f1=self.f1, f2=self.f2, duration=self.sweep_s, level_dbfs=self.level_dbfs,
                         fade_in_s=0.01, fade_out_s=0.005, tail_s=self.period_s - self.sweep_s)

    @property
    def n_period(self) -> int:
        return int(round(self.period_s * self.fs))


def generate(spec: PulseSpec) -> np.ndarray:
    """One period of the stimulus: the sweep followed by silence (float32)."""
    out = np.zeros(spec.n_period, dtype=np.float32)
    sw = sweep_signal(spec.sweep_spec)
    out[: min(sw.size, out.size)] = sw[: out.size]
    return out


# =============================================================================== per-burst frames
@dataclass
class DecayFrame:
    t_end: float
    xy: tuple[float, float]
    path_m: float                 # distance walked while the burst was recorded
    ir: ImpulseResponse           # impulse response at FS_D, direct sound at ir.t0
    snr_db: float


@dataclass
class BandMetrics:
    edt: float = np.nan
    t20: float = np.nan
    t60: float = np.nan
    edc_db: np.ndarray = field(default_factory=lambda: np.full(int(T_GRID_MAX / T_GRID_DT) + 1, np.nan))   # energy left vs time
    dyn_db: float = 0.0
    ok: bool = False


def band_metrics(ir: ImpulseResponse, fc: float, bw_oct: float = 1.0 / 3.0) -> BandMetrics:
    """Decay of one band of a (decimated) impulse response: EDT, T20, and the Schroeder curve on the fixed time grid."""
    k = 2.0 ** (0.5 * bw_oct)
    f_hi = min(fc * k, 0.45 * ir.fs)
    f_lo = max(fc / k, 0.5)
    if f_hi <= f_lo * 1.05:
        return BandMetrics()
    b = analyze_band(ir, float(fc), float(f_lo), float(f_hi))
    m = BandMetrics(edt=b.edt, t20=b.t20, t60=b.t60, dyn_db=b.dynamic_range_db)
    if b.curve_db is not None and b.curve_db.size > 8:
        t = np.arange(0.0, T_GRID_MAX + 1e-9, T_GRID_DT)
        c = b.curve_db
        m.edc_db = np.interp(t * ir.fs, np.arange(c.size), c, right=float(c[-1]))
    m.ok = bool(np.isfinite(m.edt) and b.dynamic_range_db >= 15.0)
    return m


# =============================================================================== spatial fields
@dataclass
class DecayFields:
    """Decay maps of one band: weighted means on the room grid (NaN where too little data landed)."""

    dims: RoomDims
    fc: float
    bw_oct: float
    edt: FieldMap
    t20: FieldMap
    edc: FieldMap
    times: np.ndarray
    n_frames: int = 0
    n_used: int = 0

    def edt_s(self, min_weight: float = 0.25) -> np.ndarray:
        return self.edt.mean(min_weight)[..., 0]

    def t20_s(self, min_weight: float = 0.25) -> np.ndarray:
        return self.t20.mean(min_weight)[..., 0]

    def energy_left_db(self, t: float, min_weight: float = 0.25) -> np.ndarray:
        """Mean energy still to come ``t`` seconds after the direct sound, dB re the total at that spot (0 dB at t = 0)."""
        lin = self.edc.mean(min_weight)
        i = float(np.clip(t / T_GRID_DT, 0.0, lin.shape[-1] - 1))
        i0 = int(np.floor(i))
        i1 = min(i0 + 1, lin.shape[-1] - 1)
        v = (1.0 - (i - i0)) * lin[..., i0] + (i - i0) * lin[..., i1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10.0 * np.log10(np.maximum(v, 1e-12))

    def confidence(self) -> np.ndarray:
        return self.edc.confidence()

    def export_csv(self, path: str, min_weight: float = 0.25) -> int:
        """One row per mapped cell: position, EDT, T20 and the energy left (dB) at each stored time after the burst."""
        import csv  # noqa: PLC0415

        edt, t20 = self.edt_s(min_weight), self.t20_s(min_weight)
        left = np.stack([self.energy_left_db(t, min_weight) for t in self.times], axis=-1)
        cells = np.argwhere(np.isfinite(left[..., 0]))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_m", "y_m", "weight", "edt_s", "t20_s"] + [f"left_dB_at_{t:.2f}s" for t in self.times])
            for iy, ix in cells:
                row = [f"{self.edc.xs[ix]:.3f}", f"{self.edc.ys[iy]:.3f}", f"{self.edc.w[iy, ix]:.2f}",
                       "" if not np.isfinite(edt[iy, ix]) else f"{edt[iy, ix]:.3f}", "" if not np.isfinite(t20[iy, ix]) else f"{t20[iy, ix]:.3f}"]
                wr = left[iy, ix]
                w.writerow(row + [f"{v:.2f}" for v in wr])
        return len(cells)

    def coverage(self, min_weight: float = 0.5) -> float:
        return self.edc.coverage(min_weight)

    def value_at(self, x: float, y: float) -> tuple[float, float]:
        """(EDT, T20) of the cell containing (x, y); NaN where unmeasured."""
        e, t = self.edt.sample(x, y)[0], self.t20.sample(x, y)[0]
        return float(e), float(t)


def build_fields(frames, dims: RoomDims, fc: float, bw_oct: float = 1.0 / 3.0, cell: float = 0.1, sigma0: float = 0.22,
                 min_snr_db: float = 0.0, cache: dict | None = None) -> DecayFields:
    """Rasterise the stored frames for one band.  ``cache`` (any dict) remembers each frame's band metrics so that only new frames
    cost anything when the map is refreshed during a walk."""
    n_t = int(round(T_GRID_MAX / T_GRID_DT)) + 1
    edt = FieldMap(dims, np.array([0.0]), cell, sigma0)
    t20 = FieldMap(dims, np.array([0.0]), cell, sigma0)
    edc = FieldMap(dims, np.arange(n_t, dtype=float), cell, sigma0)
    out = DecayFields(dims, float(fc), float(bw_oct), edt, t20, edc, np.arange(n_t) * T_GRID_DT)
    for fr in frames:
        out.n_frames += 1
        if fr.snr_db < min_snr_db:
            continue
        key = (id(fr), round(float(fc), 4), round(float(bw_oct), 4))
        m = cache.get(key) if cache is not None else None
        if m is None:
            m = band_metrics(fr.ir, fc, bw_oct)
            if cache is not None:
                cache[key] = m
        if not m.ok:
            continue
        x, y = fr.xy
        foot = max(sigma0, 0.5 * fr.path_m)
        edt.add(x, y, np.array([m.edt]), 1.0, foot)
        if np.isfinite(m.t20):
            t20.add(x, y, np.array([m.t20]), 1.0, foot)
        lin = 10.0 ** (np.nan_to_num(m.edc_db, nan=-120.0) / 10.0)
        edc.add(x, y, lin, 1.0, foot)
        out.n_used += 1
    return out


# =============================================================================== the walking mapper
@dataclass
class DecaySettings:
    spec: PulseSpec = field(default_factory=PulseSpec)
    v_max: float = 0.7
    snr_min_db: float = 25.0
    min_valid_frac: float = 0.7
    camera_delay_s: float = 0.0
    lead_s: float = 0.10


class DecayWalkMapper:
    def __init__(self, dims: RoomDims, settings: DecaySettings | None = None, calibration: Calibration | None = None):
        self.s = settings or DecaySettings()
        self.dims = dims
        self.cal = calibration or Calibration()
        sp = self.s.spec
        self.deconv = SweepDeconvolver(sp.sweep_spec)
        self.P = sp.n_period
        self.lead = int(round(self.s.lead_s * sp.fs))
        cap = int(3.5 * self.P) + sp.fs
        self._buf = np.zeros(cap, dtype=np.float32)
        self._total = 0
        self._t_latest = 0.0
        self._next_d: int | None = None            # absolute index at which the next burst's direct sound should arrive
        self._lost = 0
        self._pos: deque = deque(maxlen=4000)
        self.frames: list[DecayFrame] = []
        self.counts = {"accepted": 0, "no_position": 0, "speed": 0, "snr": 0, "clipped": 0, "lost": 0}
        self.locked = False
        self.last_snr = float("nan")
        self.last_speed = 0.0
        self.running = True

    # ------------------------------------------------------------------ inputs
    def reset_audio_clock(self) -> None:
        self._total, self._next_d, self.locked, self._lost = 0, None, False, 0

    def clear(self) -> None:
        self.frames.clear()
        self._pos.clear()
        for k in self.counts:
            self.counts[k] = 0

    def feed_position(self, t: float, xy, valid: bool) -> None:
        if self.running:
            self._pos.append((t - self.s.camera_delay_s, xy[0] if xy else np.nan, xy[1] if xy else np.nan, bool(valid and xy)))

    def feed_audio(self, samples: np.ndarray, t_latest: float) -> int:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        cap = self._buf.size
        if x.size >= cap:
            x = x[-cap:]
        i = self._total % cap
        first = min(x.size, cap - i)
        self._buf[i:i + first] = x[:first]
        if first < x.size:
            self._buf[:x.size - first] = x[first:]
        self._total += x.size
        self._t_latest = t_latest
        n = 0
        while self.running:
            if self._next_d is None:
                if not self._acquire():
                    break
                continue
            end = self._next_d - self.lead + self.P
            if end > self._total:
                break
            if self._total - end > cap - self.P:                 # fell far behind
                self._next_d = None
                self.locked = False
                break
            self._analyse(self._next_d)
            n += 1
        return n

    # ------------------------------------------------------------------ buffer access
    def _get(self, start: int, stop: int) -> np.ndarray:
        cap = self._buf.size
        return self._buf[np.arange(start, stop) % cap]

    # ------------------------------------------------------------------ synchronisation
    def _acquire(self) -> bool:
        """Find where the periodic bursts arrive in the stream (needs a little over two periods of audio)."""
        need = int(2.2 * self.P)
        if self._total < need:
            return False
        start = self._total - need
        seg = self._get(start, self._total).astype(np.float64)
        full = self.deconv.deconvolve(seg)
        lag0 = self.deconv.n - 1
        a = np.abs(full[lag0: lag0 + need])
        p = int(np.argmax(a))
        peak = float(a[p])
        floor = float(np.median(a)) + 1e-30
        if peak < 12.0 * floor:                                   # nothing clearly periodic yet (speaker off / too quiet)
            self.locked = False
            self._total_skip(int(0.5 * self.P))
            return False
        self._next_d = start + p + self.P                          # the burst after the one we just found
        self.locked = True
        self._lost = 0
        return True

    def _total_skip(self, n: int) -> None:
        """Without a lock keep only the newest audio: nothing to do (the buffer is a ring), kept for clarity."""

    # ------------------------------------------------------------------ analysis of one burst
    def _analyse(self, d_est: int) -> None:
        sp = self.s.spec
        fs = sp.fs
        start = d_est - self.lead
        seg = self._get(start, start + self.P).astype(np.float64)
        clipped = bool(np.max(np.abs(seg)) >= 0.999)
        ir = self.deconv.extract(seg, pre_s=0.01)
        lat = ir.latency_samples
        if abs(lat - self.lead) > int(0.05 * fs):                  # the burst is not where it should be: re-acquire
            self._lost += 1
            self.counts["lost"] += 1
            self._next_d = d_est + self.P if self._lost < 3 else None
            self.locked = self._lost < 3
            return
        self._lost = 0
        self._next_d = start + lat + self.P                        # follows clock drift between output and input
        t1 = self._t_latest - (self._total - (start + self.P)) / fs
        t0 = t1 - self.P / fs
        g = gcd(FS_D, fs)
        hd = resample_poly(ir.h, FS_D // g, fs // g)
        t0_d = int(round(ir.t0 * FS_D / fs))
        ird = ImpulseResponse(h=hd, fs=FS_D, t0=t0_d, latency_samples=int(round(lat * FS_D / fs)), rec_len=int(round(self.P * FS_D / fs)),
                              f1=sp.f1, sweep_rate_L=float(ir.sweep_rate_L))
        n_valid = max(ird.valid_samples(sp.f2), int(0.2 * FS_D))
        peak = float(np.max(np.abs(hd[max(t0_d - 4, 0): t0_d + 5])))
        tail = hd[t0_d + int(0.75 * n_valid): t0_d + n_valid]
        snr = float(20.0 * np.log10(peak / (np.sqrt(np.mean(tail ** 2)) + 1e-30))) if tail.size > 8 else float("nan")
        self.last_snr = snr

        pts = [(t, x, y, ok) for (t, x, y, ok) in self._pos if t0 <= t <= t1]
        valid = [(x, y) for (_t, x, y, ok) in pts if ok]
        reason, xy, path = "accepted", None, 0.0
        if not pts or not valid or len(valid) / len(pts) < self.s.min_valid_frac:
            reason = "no_position"
        else:
            v = np.array(valid)
            xy = (float(v[:, 0].mean()), float(v[:, 1].mean()))
            path = float(np.sum(np.hypot(*np.diff(v, axis=0).T))) if len(v) > 1 else 0.0
            self.last_speed = path / max(t1 - t0, 1e-6)
            if self.last_speed > self.s.v_max:
                reason = "speed"
            elif clipped:
                reason = "clipped"
            elif not np.isfinite(snr) or snr < self.s.snr_min_db:
                reason = "snr"
        self.counts[reason] += 1
        if reason == "accepted":
            self.frames.append(DecayFrame(t1, xy, path, ird, snr))

    # ------------------------------------------------------------------ results
    def fields(self, fc: float, bw_oct: float = 1.0 / 3.0, cell: float = 0.1) -> DecayFields:
        return build_fields(self.frames, self.dims, fc, bw_oct, cell, 0.22, 0.0)

    @property
    def accepted_fraction(self) -> float:
        n = sum(self.counts.values())
        return self.counts["accepted"] / n if n else 0.0


# =============================================================================== persistence
def save_frames(frames: list[DecayFrame], path: str, spec: PulseSpec, dims: RoomDims) -> None:
    """Keep the stored frames (positions + decimated impulse responses) so the map can be re-analysed later."""
    lens = np.array([f.ir.h.size for f in frames], dtype=np.int64)
    np.savez_compressed(
        path, xy=np.array([f.xy for f in frames], dtype=float).reshape(-1, 2), path_m=np.array([f.path_m for f in frames]),
        snr=np.array([f.snr_db for f in frames]), t_end=np.array([f.t_end for f in frames]), lens=lens,
        t0=np.array([f.ir.t0 for f in frames], dtype=np.int64), lat=np.array([f.ir.latency_samples for f in frames], dtype=np.int64),
        rec_len=np.array([f.ir.rec_len for f in frames], dtype=np.int64), L=np.array([f.ir.sweep_rate_L for f in frames]),
        h=np.concatenate([f.ir.h for f in frames]) if frames else np.zeros(0), dims=dims.as_array(),
        spec=np.array([spec.fs, spec.f1, spec.f2, spec.sweep_s, spec.period_s, spec.level_dbfs]))


def load_frames(path: str) -> tuple[list[DecayFrame], PulseSpec, RoomDims]:
    d = np.load(path)
    sp = d["spec"]
    spec = PulseSpec(int(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4]), float(sp[5]))
    frames, off = [], 0
    for i, n in enumerate(d["lens"]):
        h = d["h"][off:off + int(n)]
        off += int(n)
        ir = ImpulseResponse(h=h.copy(), fs=FS_D, t0=int(d["t0"][i]), latency_samples=int(d["lat"][i]), rec_len=int(d["rec_len"][i]),
                             f1=spec.f1, sweep_rate_L=float(d["L"][i]))
        frames.append(DecayFrame(float(d["t_end"][i]), (float(d["xy"][i, 0]), float(d["xy"][i, 1])), float(d["path_m"][i]), ir, float(d["snr"][i])))
    return frames, spec, RoomDims(*[float(v) for v in d["dims"]])
