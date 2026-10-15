"""Thread-safe mono ring buffer: written from the audio callback, read from the GUI/analysis side."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int):
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._total = 0                       # samples written since creation (monotonic)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def total(self) -> int:
        return self._total

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._total = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = x.size
        if n == 0:
            return
        with self._lock:
            if n >= self._cap:                # keep only the newest `capacity` samples
                self._total += n
                # sample j of the kept block has absolute index total-cap+j -> slot (total-cap+j) % cap
                self._buf[:] = np.roll(x[-self._cap:], (self._total - self._cap) % self._cap)
                return
            i = self._total % self._cap
            first = min(n, self._cap - i)
            self._buf[i:i + first] = x[:first]
            if first < n:
                self._buf[:n - first] = x[first:]
            self._total += n

    def read_latest(self, n: int) -> np.ndarray:
        """Newest ``n`` samples (zero-padded at the front if fewer have been written)."""
        n = int(min(n, self._cap))
        with self._lock:
            end = self._total
            avail = min(end, n)
            out = np.zeros(n, dtype=np.float32)
            if avail:
                out[n - avail:] = self._slice(end - avail, end)
        return out

    def read_range(self, start: int, stop: int) -> np.ndarray | None:
        """Samples with absolute indices [start, stop); None if any were overwritten or not yet written."""
        with self._lock:
            if start < 0 or stop > self._total or start < self._total - self._cap or stop < start:
                return None
            return self._slice(start, stop)

    def read_since(self, cursor: int) -> tuple[np.ndarray, int, bool]:
        """New samples after ``cursor``.  Returns (samples, new_cursor, overrun)."""
        with self._lock:
            end = self._total
            overrun = cursor < end - self._cap
            start = max(cursor, end - self._cap)
            return self._slice(start, end), end, overrun

    def _slice(self, start: int, stop: int) -> np.ndarray:
        n = stop - start
        i = start % self._cap
        first = min(n, self._cap - i)
        if first == n:
            return self._buf[i:i + n].copy()
        return np.concatenate((self._buf[i:], self._buf[:n - first]))
