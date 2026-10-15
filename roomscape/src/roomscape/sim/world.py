"""Simulated acoustic world: a synthetic room with a speaker and a (movable) microphone.

Implements the ``SampleSource`` protocol for ``SimulatedInput`` (steady demo signal + ambient noise),
and a ``play`` entry point so measurement sweeps played by the app come back through the room's
impulse response at the microphone, with latency and noise - exactly like a real measurement.
"""

from __future__ import annotations

import threading

import numpy as np
from scipy.signal import fftconvolve

from roomscape.core.modes import RoomDims
from roomscape.core.multisine import MultisineSpec, line_params
from roomscape.core.synth import SyntheticRoom, T60Profile
from roomscape.sim.demo import DemoRoomSignal

DEFAULT_DIMS = RoomDims(5.0, 4.0, 2.5)
DEFAULT_SOURCE = (0.4, 1.3, 1.1)
DEFAULT_MIC = (3.5, 2.0, 1.2)


def default_room(seed: int = 0) -> SyntheticRoom:
    return SyntheticRoom(DEFAULT_DIMS, DEFAULT_SOURCE, T60Profile(), seed=seed)


class LawnmowerPath:
    """A person sweeping the mic over a rectangle in rows (constant speed, short pauses at the turns)."""

    def __init__(self, x0: float, x1: float, y0: float, y1: float, z: float = 1.2, speed: float = 0.45,
                 row_spacing: float = 0.5, pause_s: float = 0.4, start_delay_s: float = 0.0):
        ys = np.arange(y0, y1 + 1e-9, row_spacing)
        pts = []
        for i, y in enumerate(ys):
            a, b = (x0, x1) if i % 2 == 0 else (x1, x0)
            pts += [(a, y), (b, y)]
        self.z = z
        self.pts = np.array(pts, dtype=float)
        t, bt, bx, by = start_delay_s, [start_delay_s], [self.pts[0, 0]], [self.pts[0, 1]]
        for i in range(1, len(self.pts)):
            t += float(np.hypot(*(self.pts[i] - self.pts[i - 1]))) / speed
            bt.append(t), bx.append(self.pts[i, 0]), by.append(self.pts[i, 1])
            if pause_s > 0 and i < len(self.pts) - 1:             # stand still for a moment at each turn
                t += pause_s
                bt.append(t), bx.append(self.pts[i, 0]), by.append(self.pts[i, 1])
        self._t, self._x, self._y = np.array(bt), np.array(bx), np.array(by)

    @property
    def duration(self) -> float:
        return float(self._t[-1])

    def position(self, t: float) -> tuple[float, float, float]:
        return float(np.interp(t, self._t, self._x)), float(np.interp(t, self._t, self._y)), self.z


class SimWorld:
    def __init__(self, fs: int = 48000, room: SyntheticRoom | None = None, mic_pos=DEFAULT_MIC,
                 noise_dbfs: float = -86.0, latency_s: float = 0.085, demo_level_dbfs: float = -24.0, seed: int = 0,
                 gain_db: float = -22.0):
        self.fs = int(fs)
        self.room = room or default_room(seed)
        self.mic = np.asarray(mic_pos, dtype=np.float64)
        self.noise_amp = 10.0 ** (noise_dbfs / 20.0)
        self.gain = 10.0 ** (gain_db / 20.0)       # speaker level -> mic digital level (a -12 dBFS sweep lands near -35 dBFS)
        self.latency = int(latency_s * self.fs)
        self.demo_level_dbfs = demo_level_dbfs
        self.demo_on = True
        self._rng = np.random.default_rng(seed + 99)
        self._lock = threading.Lock()
        self._pos = 0
        self._pending: list[tuple[int, np.ndarray]] = []
        self._ir_cache: dict[tuple, np.ndarray] = {}
        self._demo: DemoRoomSignal | None = None
        self.walker: LawnmowerPath | None = None      # when set, the microphone follows it (audio clock = world time)
        self._stim = None                             # (freqs, amplitudes, phases) of the active multisine
        self._pulse = None                            # [one period, period samples, next start (absolute sample)] of a pulse train
        self._H_key = None
        self._H = None

    # ------------------------------------------------------------------ geometry
    def ir(self, pos=None, length_s: float = 3.0) -> np.ndarray:
        p = self.mic if pos is None else np.asarray(pos, dtype=np.float64)
        key = (tuple(np.round(p, 2)), length_s)
        h = self._ir_cache.get(key)
        if h is None:
            h = self.room.ir(p, self.fs, length_s) * self.gain
            if len(self._ir_cache) > 32:
                self._ir_cache.clear()
            self._ir_cache[key] = h
        return h

    def set_room(self, room: SyntheticRoom, mic_pos=None) -> None:
        """Swap in a different synthetic room (and optionally move the mic); caches are rebuilt lazily."""
        with self._lock:
            self.room = room
            if mic_pos is not None:
                self.mic = np.asarray(mic_pos, dtype=np.float64)
            self._ir_cache.clear()
            self._demo = None

    # ------------------------------------------------------------------ walk-the-room support
    @property
    def time_s(self) -> float:
        return self._pos / self.fs

    def marker_xyz(self) -> tuple[float, float, float]:
        m = self.mic
        return float(m[0]), float(m[1]), float(m[2])

    def set_pulse_train(self, period: np.ndarray | None) -> None:
        """Repeat ``period`` (a sweep followed by silence) through the room, one response per period at the mic's position."""
        with self._lock:
            if period is None:
                self._pulse = None
            else:
                self._pulse = [np.asarray(period, dtype=np.float64), int(period.size), self._pos]

    def _schedule_pulses(self, n: int) -> None:
        """Called with the lock held: queue the response of every period that starts within the next block."""
        sig, period_n, nxt = self._pulse
        while nxt < self._pos + n + self.fs // 4:
            t_mid = (nxt + 0.5 * period_n) / self.fs
            pos = np.array(self.walker.position(t_mid)) if self.walker is not None else self.mic
            h = self.room.ir(pos, self.fs, 2.0) * self.gain
            resp = fftconvolve(sig, h)[: sig.size + h.size].astype(np.float32)
            self._pending.append((nxt + self.latency, resp))
            nxt += period_n
        self._pulse[2] = nxt

    def set_stimulus(self, spec: MultisineSpec | None) -> None:
        """Play (or stop) the mapping multisine through the room: steady-state at the current mic position."""
        with self._lock:
            self._stim = None if spec is None else line_params(spec)
            self._H_key = None

    def _line_gains(self, f: np.ndarray) -> np.ndarray:
        key = tuple(np.round(self.mic, 2))
        if key != self._H_key:
            self._H = self.room.frequency_response(self.mic, f) * self.gain
            self._H_key = key
        return self._H

    def set_mic_position(self, pos) -> None:
        with self._lock:
            self.mic = np.asarray(pos, dtype=np.float64)
            self._demo = None                          # rebuilt lazily for the new position

    # ------------------------------------------------------------------ SampleSource
    def _demo_signal(self) -> DemoRoomSignal:
        if self._demo is None:
            self._demo = DemoRoomSignal(self.fs, ir=self.ir(), level_dbfs=self.demo_level_dbfs, noise_dbfs=-200.0)
        return self._demo

    def render(self, n: int) -> np.ndarray:
        with self._lock:
            out = (self.noise_amp * self._rng.standard_normal(n)).astype(np.float32)
            if self.walker is not None:
                x, y, z = self.walker.position((self._pos + 0.5 * n) / self.fs)
                self.mic = np.array([x, y, z])
            if self._pulse is not None:
                self._schedule_pulses(n)
            if self._stim is not None:
                f, a, th = self._stim
                H = self._line_gains(f)
                t = (self._pos + np.arange(n)) / self.fs
                out += (np.cos(np.outer(t, 2.0 * np.pi * f) + (th + np.angle(H))[None, :]) @ (a * np.abs(H))).astype(np.float32)
            elif self.demo_on and not self._pending:
                out += self._demo_signal().render(n)
            keep = []
            for start, resp in self._pending:
                lo = max(self._pos, start)
                hi = min(self._pos + n, start + resp.size)
                if hi > lo:
                    out[lo - self._pos: hi - self._pos] += resp[lo - start: hi - start]
                if start + resp.size > self._pos + n:
                    keep.append((start, resp))
            self._pending = keep
            self._pos += n
            return out

    # ------------------------------------------------------------------ playback (measurements)
    def play(self, signal: np.ndarray, fs: int) -> None:
        if fs != self.fs:
            raise ValueError(f"simulated output runs at {self.fs} Hz, got {fs}")
        h = self.ir(length_s=3.0)
        resp = fftconvolve(np.asarray(signal, dtype=np.float64), h)[: signal.size + h.size].astype(np.float32)
        with self._lock:
            self._pending.append((self._pos + self.latency, resp))

    def stop(self) -> None:
        with self._lock:
            self._pending.clear()

    @property
    def playing(self) -> bool:
        with self._lock:
            return bool(self._pending)


class SimPlayer:
    """``Player`` adapter so measurement code can treat the simulated speaker like a real output."""

    def __init__(self, world: SimWorld):
        self.world = world

    def play(self, signal: np.ndarray, fs: int) -> None:
        self.world.play(signal, fs)

    def stop(self) -> None:
        self.world.stop()

    @property
    def playing(self) -> bool:
        return self.world.playing


class SimStimulusPlayer:
    """Plays the mapping multisine into the simulated room (counterpart of ``HardwareStimulusPlayer``)."""

    def __init__(self, world: SimWorld):
        self.world = world
        self.playing = False

    def start(self, spec: MultisineSpec) -> None:
        self.world.set_stimulus(spec)
        self.playing = True

    def stop(self) -> None:
        self.world.set_stimulus(None)
        self.playing = False


class SimPulsePlayer:
    """Plays the decay-camera pulse train into the simulated room (counterpart of ``HardwarePulsePlayer``)."""

    def __init__(self, world: SimWorld):
        self.world = world
        self.playing = False

    def start(self, spec) -> None:
        from roomscape.core.decaymap import generate  # noqa: PLC0415

        self.world.set_pulse_train(generate(spec))
        self.playing = True

    def stop(self) -> None:
        self.world.set_pulse_train(None)
        self.playing = False
