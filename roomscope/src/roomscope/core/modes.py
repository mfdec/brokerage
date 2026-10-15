"""Rectangular-room acoustics: eigenmodes, modal sound field, Schroeder frequency, mode statistics.

Walls are treated as rigid with a uniform modal damping derived from a reverberation time, which
is the standard first-order model for the bass region of a small listening room.

Conventions: x along length ``lx``, y along width ``ly``, z along height ``lz`` (metres); pressure
phasors use exp(+j*omega*t).  Mode shape psi_n(r) = cos(nx*pi*x/lx) cos(ny*pi*y/ly) cos(nz*pi*z/lz).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .units import speed_of_sound

KIND_AXIAL, KIND_TANGENTIAL, KIND_OBLIQUE = "axial", "tangential", "oblique"
_AXES = "LWH"


@dataclass(frozen=True)
class RoomDims:
    lx: float
    ly: float
    lz: float

    def __post_init__(self):
        if min(self.lx, self.ly, self.lz) <= 0:
            raise ValueError("room dimensions must be positive")

    @property
    def volume(self) -> float:
        return self.lx * self.ly * self.lz

    @property
    def surface(self) -> float:
        return 2.0 * (self.lx * self.ly + self.ly * self.lz + self.lx * self.lz)

    def as_array(self) -> np.ndarray:
        return np.array([self.lx, self.ly, self.lz])


@dataclass(frozen=True)
class Mode:
    n: tuple[int, int, int]
    freq: float

    @property
    def kind(self) -> str:
        k = sum(1 for v in self.n if v)
        return (KIND_AXIAL, KIND_TANGENTIAL, KIND_OBLIQUE)[k - 1]

    @property
    def label(self) -> str:
        """e.g. '1L', '2W', '1L1W', '1L1W1H' (axes: L = length x, W = width y, H = height z)."""
        return "".join(f"{v}{_AXES[i]}" for i, v in enumerate(self.n) if v)

    @property
    def index_str(self) -> str:
        return ",".join(str(v) for v in self.n)


def room_modes(dims: RoomDims, f_max: float, c: float | None = None, f_min: float = 0.0) -> list[Mode]:
    """All eigenmodes with ``f_min <= f <= f_max`` sorted by frequency (the (0,0,0) mode is excluded)."""
    c = speed_of_sound() if c is None else c
    m = [int(np.floor(2.0 * L * f_max / c)) for L in (dims.lx, dims.ly, dims.lz)]
    nx, ny, nz = np.meshgrid(np.arange(m[0] + 1), np.arange(m[1] + 1), np.arange(m[2] + 1), indexing="ij")
    f = 0.5 * c * np.sqrt((nx / dims.lx) ** 2 + (ny / dims.ly) ** 2 + (nz / dims.lz) ** 2)
    keep = (f <= f_max) & (f >= f_min) & ((nx + ny + nz) > 0)
    modes = [Mode((int(a), int(b), int(d)), float(fr)) for a, b, d, fr in zip(nx[keep], ny[keep], nz[keep], f[keep])]
    modes.sort(key=lambda mo: (mo.freq, mo.n))
    return modes


def mode_arrays(modes: list[Mode]) -> tuple[np.ndarray, np.ndarray]:
    """(index array shape (M, 3), frequency array shape (M,))."""
    n = np.array([m.n for m in modes], dtype=np.float64).reshape(-1, 3)
    f = np.array([m.freq for m in modes], dtype=np.float64)
    return n, f


def mode_shape(dims: RoomDims, n_idx: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """psi for every mode at every position: ``pos`` (P, 3) -> (P, M)."""
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    L = dims.as_array()
    arg = np.pi * pos[:, None, :] * n_idx[None, :, :] / L[None, None, :]
    return np.prod(np.cos(arg), axis=2)


def mode_norm(n_idx: np.ndarray) -> np.ndarray:
    """Lambda_n = prod(1/2 if n_i > 0 else 1): mean-square of psi over the room volume."""
    return np.prod(np.where(n_idx > 0, 0.5, 1.0), axis=1)


def schroeder_frequency(t60: float, volume: float) -> float:
    """Transition between the modal (bass) and statistical regimes: 2000*sqrt(T60/V)."""
    return 2000.0 * np.sqrt(t60 / volume)


def critical_distance(volume: float, t60: float) -> float:
    """Distance at which direct and reverberant energy are equal for an omnidirectional source."""
    return 0.057 * np.sqrt(volume / t60)


def modal_density(freq: float, dims: RoomDims, c: float | None = None) -> float:
    """Modes per Hz (Weyl asymptotics: volume, surface and edge terms)."""
    c = speed_of_sound() if c is None else c
    edges = 4.0 * (dims.lx + dims.ly + dims.lz)
    return 4 * np.pi * dims.volume * freq ** 2 / c ** 3 + np.pi * dims.surface * freq / (2 * c ** 2) + edges / (8 * c)


def count_modes_below(freq: float, dims: RoomDims, c: float | None = None) -> float:
    """Smooth estimate of the number of modes below ``freq`` (Weyl's law)."""
    c = speed_of_sound() if c is None else c
    edges = 4.0 * (dims.lx + dims.ly + dims.lz)
    return (4 * np.pi * dims.volume * freq ** 3 / (3 * c ** 3) + np.pi * dims.surface * freq ** 2 / (4 * c ** 2)
            + edges * freq / (8 * c))


# --------------------------------------------------------------------------- #
# Forced response of a point source
# --------------------------------------------------------------------------- #
T60Fn = Callable[[np.ndarray], np.ndarray]


def damping_from_t60(t60) -> np.ndarray:
    """Amplitude decay rate delta (1/s) of a mode that decays 60 dB in ``t60`` seconds."""
    return 6.9078 / np.asarray(t60, dtype=np.float64)


def forced_response(
    dims: RoomDims,
    modes: list[Mode],
    src: np.ndarray,
    rcv: np.ndarray,
    freqs: np.ndarray,
    t60: float | T60Fn = 0.5,
    c: float | None = None,
) -> np.ndarray:
    """Complex pressure (relative units) at receivers ``rcv`` (P, 3) for a point source at ``src`` (3,).

    Returns shape (P, F).  Each mode is a damped resonator; damping comes from ``t60`` (a number or a
    function of frequency evaluated at each mode's frequency).
    """
    c = speed_of_sound() if c is None else c
    n_idx, f_n = mode_arrays(modes)
    rcv = np.atleast_2d(np.asarray(rcv, dtype=np.float64))
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    if n_idx.size == 0:
        return np.zeros((rcv.shape[0], freqs.size), dtype=np.complex128)
    psi_s = mode_shape(dims, n_idx, np.asarray(src, dtype=np.float64)[None, :])[0]       # (M,)
    psi_r = mode_shape(dims, n_idx, rcv)                                                  # (P, M)
    weight = psi_r * (psi_s / mode_norm(n_idx))[None, :]                                  # (P, M)
    t60_n = t60(f_n) if callable(t60) else np.full(f_n.shape, float(t60))
    delta = damping_from_t60(t60_n)
    w_n = 2 * np.pi * f_n
    w = 2 * np.pi * freqs
    denom = w_n[None, :] ** 2 - w[:, None] ** 2 + 2j * delta[None, :] * w[:, None]       # (F, M)
    resp = (weight[:, None, :] / denom[None, :, :]).sum(axis=2)                           # (P, F)
    return 1j * w[None, :] * resp * (c ** 2) / dims.volume


# --------------------------------------------------------------------------- #
# Mode-set quality statistics
# --------------------------------------------------------------------------- #
def coincident_modes(modes: list[Mode], tol_frac: float = 0.02) -> list[tuple[Mode, Mode]]:
    """Pairs of different modes within ``tol_frac`` of each other (they reinforce into one big peak)."""
    out = []
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            if b.freq > a.freq * (1 + tol_frac):
                break
            out.append((a, b))
    return out


def largest_gaps(modes: list[Mode], f_lo: float = 20.0, f_hi: float = 200.0, top: int = 3) -> list[tuple[float, float]]:
    """Widest empty frequency stretches between consecutive modes: [(gap_hz, start_hz), ...]."""
    f = np.array([m.freq for m in modes if f_lo <= m.freq <= f_hi])
    if f.size < 2:
        return []
    gaps = np.diff(f)
    order = np.argsort(gaps)[::-1][:top]
    return [(float(gaps[i]), float(f[i])) for i in order]


def third_octave_counts(modes: list[Mode], f_lo: float = 20.0, f_hi: float = 300.0):
    """(band centres, number of modes in each 1/3-octave band) - the basis of the Bonello criterion."""
    from .units import band_centers, band_edge_factor

    fc = band_centers(3, f_lo, f_hi)
    k = band_edge_factor(3)
    f = np.array([m.freq for m in modes])
    counts = np.array([int(np.sum((f >= c / k) & (f < c * k))) for c in fc])
    return fc, counts


def bonello_check(modes: list[Mode], f_lo: float = 25.0, f_hi: float = 300.0) -> tuple[bool, list[float]]:
    """Bonello criterion: mode count per 1/3-octave band must not decrease with frequency.

    Returns (passes, [centre frequencies where the count drops]).
    """
    fc, n = third_octave_counts(modes, f_lo, f_hi)
    bad = [float(fc[i]) for i in range(1, n.size) if n[i] < n[i - 1]]
    return (len(bad) == 0), bad
