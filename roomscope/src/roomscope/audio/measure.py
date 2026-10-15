"""Playing a sweep while recording the room: hardware player and the capture/playback runner."""

from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

import numpy as np

from roomscope.core.measurement import MeasurementError
from roomscope.core.sweep import SweepSpec, recording_length, sweep_signal

from .devices import DeviceInfo, _sd, wasapi_settings
from .engine import AudioError, AudioSource

CHANNEL_MODES = ("Both", "Left", "Right")


class Player(Protocol):
    def play(self, signal: np.ndarray, fs: int) -> None: ...
    def stop(self) -> None: ...
    @property
    def playing(self) -> bool: ...


class HardwarePlayer:
    """Plays a mono signal on the left / right / both channels of an output device (non-blocking).

    A writer thread feeds PortAudio's own buffer with blocking writes, so a busy GUI thread cannot
    cause an underflow (an audible glitch inside the sweep would corrupt the measurement).
    """

    CHUNK = 2048

    def __init__(self, device: DeviceInfo, channels: str = "Both"):
        self.device = device
        self.channels = channels
        self._thread: threading.Thread | None = None
        self._abort = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self.underflows = 0

    @property
    def playing(self) -> bool:
        return not self._done.is_set()

    def play(self, signal: np.ndarray, fs: int) -> None:
        sd = _sd()
        if sd is None:
            raise AudioError("Audio backend (PortAudio) is not available.")
        self.stop()
        n_ch = 1 if self.device.max_output < 2 else 2
        data = np.zeros((signal.size, n_ch), dtype=np.float32)
        if n_ch == 1 or self.channels == "Both":
            data[:] = signal[:, None]
        elif self.channels == "Left":
            data[:, 0] = signal
        else:
            data[:, 1] = signal
        try:
            stream = sd.OutputStream(device=self.device.index, samplerate=fs, channels=n_ch, dtype="float32",
                                     latency="high", extra_settings=wasapi_settings(self.device))
            stream.start()
        except Exception as e:
            raise AudioError(f"Could not open output '{self.device.name}' at {fs} Hz: {e}") from e
        self._abort.clear()
        self._done.clear()
        self.underflows = 0
        self._thread = threading.Thread(target=self._writer, args=(stream, data), name="audio-out", daemon=True)
        self._thread.start()

    def _writer(self, stream, data: np.ndarray) -> None:
        try:
            pos = 0
            while pos < len(data) and not self._abort.is_set():
                if stream.write(data[pos: pos + self.CHUNK]):
                    self.underflows += 1
                pos += self.CHUNK
            if self._abort.is_set():
                stream.abort()
            else:
                stream.stop()                      # lets PortAudio finish playing what is buffered
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass
            self._done.set()

    def stop(self) -> None:
        self._abort.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3.0)
        self._done.set()


class MeasurementCancelled(RuntimeError):
    pass


def record_sweep(
    source: AudioSource, player: Player, spec: SweepSpec, lead_s: float = 0.5,
    progress: Callable[[float, str], None] | None = None, cancel: threading.Event | None = None,
) -> np.ndarray:
    """Play the sweep and return the matching stretch of the live input (including ``lead_s`` of pre-roll)."""
    if not source.running:
        raise MeasurementError("Start the microphone input first.")
    if source.fs != spec.fs:
        raise MeasurementError(f"Input runs at {source.fs} Hz but the sweep is {spec.fs} Hz.")
    needed = recording_length(spec, lead_s)
    start = max(source.ring.total - int(lead_s * spec.fs), 0)
    player.play(sweep_signal(spec), spec.fs)
    t_start = time.monotonic()
    deadline = t_start + needed / spec.fs + 10.0
    try:
        while source.ring.total - start < needed:
            if cancel is not None and cancel.is_set():
                raise MeasurementCancelled()
            if time.monotonic() > deadline or not source.running:
                raise MeasurementError("The microphone stopped delivering audio during the measurement.")
            if progress:
                progress(min((source.ring.total - start) / needed, 0.999), "Recording sweep...")
            time.sleep(0.04)
    finally:
        player.stop()
    rec = source.ring.read_range(start, start + needed)
    if rec is None:
        raise MeasurementError("Recording buffer overrun; try again.")
    if progress:
        progress(1.0, "Analyzing...")
    return rec
