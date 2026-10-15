"""Audio input sources feeding a shared ring buffer: real hardware and a simulated room."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from .devices import DeviceInfo, _sd, wasapi_settings
from .ringbuffer import RingBuffer

RING_SECONDS = 60


class AudioError(RuntimeError):
    pass


class SampleSource(Protocol):
    """Anything that can synthesise mono samples (used by the simulated input)."""

    def render(self, n: int) -> np.ndarray: ...


class AudioSource:
    """Common interface for the GUI: ring buffer + lifecycle + health counters."""

    is_simulated = False

    def __init__(self, fs: int, name: str):
        self.fs = int(fs)
        self.name = name
        self.ring = RingBuffer(RING_SECONDS * self.fs)
        self.xruns = 0
        self.running = False
        self.error: str | None = None

    def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def seconds_recorded(self) -> float:
        return self.ring.total / self.fs


class HardwareInput(AudioSource):
    """Records one channel of a real input device through PortAudio.

    Uses PortAudio's *blocking* read API in a dedicated thread with a generous internal buffer
    (``latency='high'``) instead of a Python callback: a callback needs the GIL inside the audio
    thread, so a GUI repaint holding the GIL for ~10 ms could overflow the device buffer.  With a
    blocking reader PortAudio fills its own C buffer regardless of what the GUI thread is doing.
    """

    READ_FRAMES = 1024

    def __init__(self, device: DeviceInfo, fs: int, channel: int = 0, latency="high"):
        super().__init__(fs, device.name)
        self.device = device
        self.channel = int(channel)
        self.latency = latency
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        sd = _sd()
        if sd is None:
            raise AudioError("Audio backend (PortAudio) is not available. Install the 'sounddevice' package.")
        if self.running:
            return
        n_ch = min(max(self.channel + 1, 1), max(self.device.max_input, 1))
        try:
            self._stream = sd.InputStream(
                device=self.device.index, channels=n_ch, samplerate=self.fs, dtype="float32",
                latency=self.latency, extra_settings=wasapi_settings(self.device),
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioError(f"Could not open '{self.device.name}' at {self.fs} Hz: {e}") from e
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._reader, name="audio-in", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort()
            except Exception:
                pass
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    def _reader(self) -> None:
        stream = self._stream
        while not self._stop.is_set() and stream is not None:
            try:
                data, overflowed = stream.read(self.READ_FRAMES)
            except Exception as e:
                if not self._stop.is_set():
                    self.error = f"{type(e).__name__}: {e}"
                    self.running = False
                break
            if overflowed:
                self.xruns += 1
            self.ring.write(data[:, min(self.channel, data.shape[1] - 1)])


class SimulatedInput(AudioSource):
    """Real-time paced synthetic microphone.  ``generator.render(n)`` returns the next ``n`` samples."""

    is_simulated = True

    def __init__(self, generator: SampleSource, fs: int = 48000, name: str = "Simulated room", blocksize: int = 1024):
        super().__init__(fs, name)
        self.generator = generator
        self.blocksize = int(blocksize)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="sim-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    def _run(self) -> None:
        period = self.blocksize / self.fs
        t_next = time.monotonic()
        while not self._stop.is_set():
            self.ring.write(self.generator.render(self.blocksize))
            t_next += period
            delay = t_next - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -1.0:                # fell badly behind (GUI stall): resync instead of bursting
                t_next = time.monotonic()
                self.xruns += 1

    def pump(self, seconds: float) -> None:
        """Synchronously generate ``seconds`` of audio without real-time pacing (tests, offline use)."""
        n = int(seconds * self.fs)
        while n > 0:
            m = min(n, self.blocksize)
            self.ring.write(self.generator.render(m))
            n -= m
