"""Demo signal for the simulated mic: pink-noise bursts played through a synthetic room, looped.

The loop is rendered once with *circular* convolution, which is exactly the steady state of a
periodic signal in a linear room, so the seam is inaudible/invisible and block rendering is free.
"""

from __future__ import annotations

import numpy as np


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std()


def quick_room_ir(fs: int, t60: float = 0.45, modes=((38.0, 1.0), (57.0, 0.8), (76.0, 0.6), (92.0, 0.5), (118.0, 0.45), (141.0, 0.35)),
                  length_s: float = 1.6, seed: int = 1) -> np.ndarray:
    """A plausible small-room impulse response: direct sound, a few bass modes, diffuse tail."""
    rng = np.random.default_rng(seed)
    n = int(length_s * fs)
    t = np.arange(n) / fs
    delta = 6.91 / t60
    tail = rng.standard_normal(n) * np.exp(-delta * t)
    tail *= 0.12 * (1.0 - np.exp(-t / 0.004))                 # reflections build up over a few ms
    ir = tail
    for f0, a in modes:
        ir = ir + 0.006 * a * np.exp(-6.91 / (t60 * 1.35) * t) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
    ir[int(0.003 * fs)] += 1.0                               # direct sound ~1 m away
    return ir


class DemoRoomSignal:
    """Block generator implementing the ``SampleSource`` protocol."""

    def __init__(self, fs: int = 48000, ir: np.ndarray | None = None, loop_s: float = 7.0,
                 level_dbfs: float = -24.0, noise_dbfs: float = -78.0, seed: int = 0):
        self.fs = int(fs)
        rng = np.random.default_rng(seed)
        n = int(loop_s * fs)
        # 0.9 s bursts of pink noise, 0.6 s silence, so decays are visible after each burst
        src = np.zeros(n)
        pos, period, burst = 0, int(1.5 * fs), int(0.9 * fs)
        ramp = int(0.02 * fs)
        env = np.ones(burst)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        noise = pink_noise(n, rng)
        while pos + burst <= n:
            src[pos:pos + burst] = noise[pos:pos + burst] * env
            pos += period
        ir = quick_room_ir(self.fs) if ir is None else np.asarray(ir, dtype=np.float64)
        spec = np.fft.rfft(src) * np.fft.rfft(ir[:n], n)
        y = np.fft.irfft(spec, n)
        y *= 10 ** (level_dbfs / 20) / y.std()
        self._loop = y.astype(np.float32)
        self._pos = 0
        self._rng = rng
        self._noise_amp = 10 ** (noise_dbfs / 20)

    def render(self, n: int) -> np.ndarray:
        idx = (self._pos + np.arange(n)) % self._loop.size
        self._pos = (self._pos + n) % self._loop.size
        return self._loop[idx] + (self._noise_amp * self._rng.standard_normal(n)).astype(np.float32)
