"""Simulated acoustic world: a synthetic room with a speaker and a (movable) microphone.

Implements the ``SampleSource`` protocol for ``SimulatedInput`` (steady demo signal + ambient noise),
and a ``play`` entry point so measurement sweeps played by the app come back through the room's
impulse response at the microphone, with latency and noise - exactly like a real measurement.
"""

from __future__ import annotations

import threading

import numpy as np
from scipy.signal import fftconvolve

from roomscope.core.modes import RoomDims
from roomscope.core.synth import SyntheticRoom, T60Profile
from roomscope.sim.demo import DemoRoomSignal

DEFAULT_DIMS = RoomDims(5.0, 4.0, 2.5)
DEFAULT_SOURCE = (0.4, 1.3, 1.1)
DEFAULT_MIC = (3.5, 2.0, 1.2)


def default_room(seed: int = 0) -> SyntheticRoom:
    return SyntheticRoom(DEFAULT_DIMS, DEFAULT_SOURCE, T60Profile(), seed=seed)


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
            if self.demo_on and not self._pending:
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
