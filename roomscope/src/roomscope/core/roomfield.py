"""Room configuration, modal pressure fields, modal excitation and measured-peak matching."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import find_peaks

from .modes import (
    Mode, RoomDims, damping_from_t60, forced_response, mode_arrays, mode_norm, mode_shape, room_modes,
    schroeder_frequency,
)
from .units import speed_of_sound


@dataclass(frozen=True)
class RoomConfig:
    """Everything that defines the modelled room (metres, degrees C)."""

    dims: RoomDims = RoomDims(5.0, 4.0, 2.5)
    temp_c: float = 20.0
    source: tuple[float, float, float] = (0.4, 1.3, 1.1)        # one loudspeaker (or sub)
    listener: tuple[float, float, float] = (3.5, 2.0, 1.2)      # listening position = mic height
    t60: float = 0.5                                            # damping for the modal model (s)
    f_max: float = 300.0

    @property
    def c(self) -> float:
        return speed_of_sound(self.temp_c)

    @property
    def volume(self) -> float:
        return self.dims.volume

    @property
    def schroeder_hz(self) -> float:
        return float(schroeder_frequency(self.t60, self.volume))

    def modes(self, f_max: float | None = None) -> list[Mode]:
        return room_modes(self.dims, self.f_max if f_max is None else f_max, self.c)

    def clamp(self, p) -> tuple[float, float, float]:
        L = self.dims.as_array()
        return tuple(float(v) for v in np.clip(np.asarray(p, dtype=float), 0.0, L))

    def with_(self, **kw) -> "RoomConfig":
        return replace(self, **kw)


def excitation_db(cfg: RoomConfig, modes: list[Mode] | None = None) -> np.ndarray:
    """Relative strength (dB, strongest = 0) with which each mode shows up for this source and listener.

    A resonance peaks at |psi_s psi_r| / (Lambda * 2 delta); modes with a node at the speaker or at the
    listening position are barely excited.  Uses the configured damping for every mode.
    """
    modes = cfg.modes() if modes is None else modes
    if not modes:
        return np.zeros(0)
    n_idx, f_n = mode_arrays(modes)
    src = np.asarray(cfg.source, dtype=float)[None, :]
    lis = np.asarray(cfg.listener, dtype=float)[None, :]
    w = np.abs(mode_shape(cfg.dims, n_idx, src)[0] * mode_shape(cfg.dims, n_idx, lis)[0]) / mode_norm(n_idx)
    strength = w / damping_from_t60(cfg.t60)
    return 20.0 * np.log10(np.maximum(strength / max(strength.max(), 1e-30), 1e-6))


def predicted_response_db(cfg: RoomConfig, freqs: np.ndarray, t60=None) -> np.ndarray:
    """Modal frequency response (dB, arbitrary reference) at the listening position."""
    modes = cfg.modes(max(float(np.max(freqs)) * 1.25, cfg.f_max))
    p = forced_response(cfg.dims, modes, np.asarray(cfg.source), np.asarray(cfg.listener)[None, :], freqs,
                        cfg.t60 if t60 is None else t60, cfg.c)[0]
    return 20.0 * np.log10(np.maximum(np.abs(p), 1e-30))


def modal_field_db(cfg: RoomConfig, freq: float, z: float | None = None, n: tuple[int, int] = (96, 72),
                   modes: list[Mode] | None = None, t60=None):
    """Pressure level over a horizontal plane at height ``z`` for a tone at ``freq``.

    Returns (xs, ys, level_db) with ``level_db[iy, ix]`` relative to the plane's median level, plus the
    complex field itself is not needed by callers.  Modes up to 1.3x the frequency are included.
    """
    z = cfg.listener[2] if z is None else z
    nx, ny = n
    xs = (np.arange(nx) + 0.5) / nx * cfg.dims.lx
    ys = (np.arange(ny) + 0.5) / ny * cfg.dims.ly
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack((gx.ravel(), gy.ravel(), np.full(gx.size, z)), axis=1)
    modes = cfg.modes(max(freq * 1.3, 60.0)) if modes is None else modes
    p = forced_response(cfg.dims, modes, np.asarray(cfg.source), pts, np.array([freq]), cfg.t60 if t60 is None else t60, cfg.c)
    mag = np.abs(p[:, 0]).reshape(ny, nx)
    db = 20.0 * np.log10(np.maximum(mag, 1e-30))
    return xs, ys, db - np.median(db)


# --------------------------------------------------------------------------- #
# Measured peaks vs predicted modes
# --------------------------------------------------------------------------- #
@dataclass
class Peak:
    freq: float
    level_db: float
    prominence_db: float


@dataclass
class PeakMatch:
    peak: Peak
    candidates: list[Mode] = field(default_factory=list)
    best: Mode | None = None
    error_hz: float = float("nan")

    @property
    def matched(self) -> bool:
        return self.best is not None


def find_response_peaks(freqs: np.ndarray, level_db: np.ndarray, f_lo: float = 20.0, f_hi: float = 300.0,
                        prominence_db: float = 3.0, step_hz: float = 0.5) -> list[Peak]:
    """Peaks of a bass frequency response on a uniform grid (room modes are linearly spaced)."""
    grid = np.arange(f_lo, f_hi + step_hz, step_hz)
    y = np.interp(grid, freqs, level_db)
    k = max(int(round(2.0 / step_hz)) | 1, 3)                      # ~2 Hz moving average removes ripple, keeps modes
    ys = np.convolve(np.pad(y, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    idx, props = find_peaks(ys, prominence=prominence_db)
    return [Peak(float(grid[i]), float(ys[i]), float(p)) for i, p in zip(idx, props["prominences"])]


def match_peaks(peaks: list[Peak], modes: list[Mode], cfg: RoomConfig | None = None,
                tol_frac: float = 0.04, tol_min_hz: float = 2.0) -> list[PeakMatch]:
    """Assign each measured peak to the most strongly excited predicted mode within tolerance."""
    exc = excitation_db(cfg, modes) if cfg is not None else np.zeros(len(modes))
    bw = float(damping_from_t60(cfg.t60)) / np.pi if cfg is not None else 3.0        # -3 dB width of a mode (Hz)
    f_modes = np.array([m.freq for m in modes])
    out = []
    for pk in peaks:
        tol = max(tol_min_hz, tol_frac * pk.freq)
        near = np.nonzero(np.abs(f_modes - pk.freq) <= tol)[0]
        if near.size == 0:
            out.append(PeakMatch(pk))
            continue
        # strongly excited modes win, but only if they are within about a mode-width of the peak
        best = near[np.argmax(exc[near] - 10.0 * ((f_modes[near] - pk.freq) / bw) ** 2)]
        out.append(PeakMatch(pk, [modes[i] for i in near], modes[best], float(pk.freq - f_modes[best])))
    return out


def room_to_json(cfg: RoomConfig) -> str:
    return json.dumps({"dims": list(cfg.dims.as_array()), "temp_c": cfg.temp_c, "source": list(cfg.source),
                       "listener": list(cfg.listener), "t60": cfg.t60, "f_max": cfg.f_max})


def room_from_json(text: str) -> RoomConfig:
    d = json.loads(text)
    dims = RoomDims(*[float(v) for v in d["dims"]])
    cfg = RoomConfig(dims, float(d["temp_c"]), tuple(float(v) for v in d["source"]),
                     tuple(float(v) for v in d["listener"]), float(d["t60"]), float(d["f_max"]))
    return cfg.with_(source=cfg.clamp(cfg.source), listener=cfg.clamp(cfg.listener))
