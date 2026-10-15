"""Walk-the-room mapping: fuse the marker's position track with the microphone's line levels into a field map.

Pure logic with no threads or devices: feed it positions (video rate) and audio chunks (any chunking) with
timestamps from one clock; it produces analysis frames every ``hop_s`` and paints accepted ones into a
``FieldMap``.  The same object runs live (timestamps from ``time.monotonic``) and in offline tests (virtual time).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .calibration import Calibration
from .fieldmap import FieldMap
from .modes import RoomDims
from .multisine import FrameResult, MultisineSpec, analyze_frame


@dataclass
class MapperSettings:
    spec: MultisineSpec = field(default_factory=MultisineSpec)
    periods: int = 2                 # analysis frame = this many stimulus periods (2 -> noise bins are free)
    hop_s: float = 0.5               # new frame every hop
    v_max: float = 0.9               # m/s; faster movement smears the map and shakes the mic
    snr_min_db: float = 10.0
    min_valid_frac: float = 0.7      # fraction of the frame during which the marker was tracked
    sigma0: float = 0.22             # footprint of one frame (m), grows with the distance walked during it
    cell: float = 0.1
    camera_delay_s: float = 0.0      # video arrives this late (network cameras): positions are shifted back by it
    line_snr_min_db: float = 3.0


@dataclass
class FrameLog:
    t_end: float
    xy: tuple[float, float] | None
    accepted: bool
    reason: str
    snr_db: float


class WalkMapper:
    def __init__(self, dims: RoomDims, settings: MapperSettings | None = None, calibration: Calibration | None = None):
        self.s = settings or MapperSettings()
        self.dims = dims
        self.cal = calibration or Calibration()
        self.field = FieldMap(dims, self.s.spec.line_freqs, self.s.cell, self.s.sigma0)
        fs = self.s.spec.fs
        self.frame_n = self.s.periods * self.s.spec.n_period
        self.hop_n = int(round(self.s.hop_s * fs))
        cap = self.frame_n + 4 * self.hop_n + fs
        self._buf = np.zeros(cap, dtype=np.float32)
        self._total = 0                        # samples received
        self._end = self.frame_n               # absolute index of the end of the next frame to analyse
        self._t_latest = 0.0                   # time of the newest sample
        self._pos: deque = deque(maxlen=4000)  # (t, x, y, valid)
        self.counts = {"accepted": 0, "no_position": 0, "speed": 0, "snr": 0, "clipped": 0}
        self.log: deque[FrameLog] = deque(maxlen=400)
        self.last: FrameResult | None = None
        self.last_xy: tuple[float, float] | None = None
        self.last_speed = 0.0
        self.running = True

    # ------------------------------------------------------------------ inputs
    def reset_audio_clock(self) -> None:
        """Call when the input stream restarts so old samples are not mixed with new ones."""
        self._total, self._end = 0, self.frame_n

    def clear(self) -> None:
        self.field.clear()
        self._pos.clear()
        for k in self.counts:
            self.counts[k] = 0
        self.log.clear()

    def feed_position(self, t: float, xy: tuple[float, float] | None, valid: bool) -> None:
        if not self.running:
            return
        self._pos.append((t - self.s.camera_delay_s, xy[0] if xy else np.nan, xy[1] if xy else np.nan, bool(valid and xy)))

    def feed_audio(self, samples: np.ndarray, t_latest: float) -> int:
        """Append audio (mono float); ``t_latest`` is the time of its last sample.  Returns frames analysed."""
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
        while self._end <= self._total and self.running:
            if self._total - self._end > cap - self.frame_n:       # fell far behind: skip ahead
                self._end = self._total - self.frame_n
            self._analyse(self._end)
            self._end += self.hop_n
            n += 1
        return n

    # ------------------------------------------------------------------ analysis
    def _window(self, end: int) -> np.ndarray:
        cap = self._buf.size
        idx = (np.arange(end - self.frame_n, end)) % cap
        return self._buf[idx]

    def _analyse(self, end: int) -> None:
        fs = self.s.spec.fs
        t1 = self._t_latest - (self._total - end) / fs
        t0 = t1 - self.frame_n / fs
        res = analyze_frame(self._window(end), self.s.spec, self.cal, self.s.periods)
        self.last = res
        pts = [(t, x, y, ok) for (t, x, y, ok) in self._pos if t0 <= t <= t1]
        valid = [(x, y) for (_t, x, y, ok) in pts if ok]
        xy = None
        reason = "accepted"
        speed = 0.0
        if not pts or len(valid) / len(pts) < self.s.min_valid_frac or not valid:
            reason = "no_position"
        else:
            v = np.array(valid)
            xy = (float(v[:, 0].mean()), float(v[:, 1].mean()))
            path = float(np.sum(np.hypot(*np.diff(v, axis=0).T))) if len(v) > 1 else 0.0
            speed = path / max(t1 - t0, 1e-6)
            self.last_xy, self.last_speed = xy, speed
            if speed > self.s.v_max:
                reason = "speed"
            elif res.clipped:
                reason = "clipped"
            elif not np.isfinite(res.snr_db) or res.snr_db < self.s.snr_min_db:
                reason = "snr"
        self.log.append(FrameLog(t1, xy, reason == "accepted", reason, res.snr_db))
        self.counts[reason] += 1
        if reason != "accepted":
            return
        p = 10.0 ** (res.power_db / 10.0)
        p[res.line_snr_db < self.s.line_snr_min_db] = np.nan
        footprint = max(self.s.sigma0, 0.5 * path)
        self.field.add(xy[0], xy[1], p, 1.0, footprint)

    # ------------------------------------------------------------------ status
    def stationary_for(self, seconds: float, radius: float = 0.1) -> bool:
        """True if the marker has stayed within ``radius`` for the last ``seconds`` (e.g. to trigger a spot sweep)."""
        if not self._pos:
            return False
        t_now = self._pos[-1][0]
        pts = [(x, y) for (t, x, y, ok) in self._pos if t >= t_now - seconds and ok]
        if not pts or self._pos[0][0] > t_now - seconds * 0.95:
            return False
        a = np.array(pts)
        return bool(len(a) >= 3 and np.ptp(a[:, 0]) <= 2 * radius and np.ptp(a[:, 1]) <= 2 * radius)

    @property
    def accepted_fraction(self) -> float:
        n = sum(self.counts.values())
        return self.counts["accepted"] / n if n else 0.0
