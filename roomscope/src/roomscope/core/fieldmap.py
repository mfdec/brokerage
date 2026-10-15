"""Spatial accumulator: line levels measured at positions -> a map of the room, per frequency line."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes import RoomDims


@dataclass
class FieldStats:
    n_added: int = 0
    total_weight: float = 0.0


class FieldMap:
    """Weighted power averages on a regular grid.

    ``add`` spreads one measurement over the cells around its position with a Gaussian footprint whose *peak*
    weight is ``weight`` (so a cell's accumulated weight reads as "how many measurements effectively landed
    here"); the map value of a cell is the weighted mean *power* of everything that landed in it (normalised
    convolution), and its confidence grows with the accumulated weight.
    """

    def __init__(self, dims: RoomDims, freqs: np.ndarray, cell: float = 0.1, sigma: float = 0.25):
        self.dims = dims
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.cell = float(cell)
        self.sigma = float(sigma)
        self.nx = max(int(np.ceil(dims.lx / cell)), 2)
        self.ny = max(int(np.ceil(dims.ly / cell)), 2)
        self.xs = (np.arange(self.nx) + 0.5) * dims.lx / self.nx
        self.ys = (np.arange(self.ny) + 0.5) * dims.ly / self.ny
        self.w = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.p = np.zeros((self.ny, self.nx, self.freqs.size), dtype=np.float32)
        self.stats = FieldStats()

    def clear(self) -> None:
        self.w[:] = 0
        self.p[:] = 0
        self.stats = FieldStats()

    # ------------------------------------------------------------------ accumulate
    def add(self, x: float, y: float, power: np.ndarray, weight: float = 1.0, sigma: float | None = None) -> None:
        """``power``: linear mean-square per line (shape (K,)), NaN lines are skipped."""
        s = float(self.sigma if sigma is None else max(sigma, 0.5 * self.cell))
        r = 3.0 * s
        ix0, ix1 = int(max((x - r) / self.dims.lx * self.nx, 0)), int(min((x + r) / self.dims.lx * self.nx + 1, self.nx))
        iy0, iy1 = int(max((y - r) / self.dims.ly * self.ny, 0)), int(min((y + r) / self.dims.ly * self.ny + 1, self.ny))
        if ix1 <= ix0 or iy1 <= iy0:
            return
        gx = np.exp(-0.5 * ((self.xs[ix0:ix1] - x) / s) ** 2)
        gy = np.exp(-0.5 * ((self.ys[iy0:iy1] - y) / s) ** 2)
        k = np.outer(gy, gx) * weight                          # peak weight = ``weight`` at the measurement position
        pw = np.asarray(power, dtype=np.float64)
        ok = np.isfinite(pw)
        self.w[iy0:iy1, ix0:ix1] += k.astype(np.float32)
        add = k[:, :, None] * np.where(ok, pw, 0.0)[None, None, :]
        self.p[iy0:iy1, ix0:ix1] += add.astype(np.float32)
        self.stats.n_added += 1
        self.stats.total_weight += weight

    # ------------------------------------------------------------------ read out
    def band_mask(self, f_lo: float, f_hi: float) -> np.ndarray:
        m = (self.freqs >= f_lo) & (self.freqs <= f_hi)
        if not m.any():                                       # narrower than the line spacing: nearest line
            m = np.zeros(self.freqs.size, dtype=bool)
            m[int(np.argmin(np.abs(self.freqs - 0.5 * (f_lo + f_hi))))] = True
        return m

    def level_db(self, f_lo: float, f_hi: float | None = None, min_weight: float = 0.25) -> np.ndarray:
        """Level (dB, power average over the band) per cell; NaN where too little data has landed."""
        f_hi = f_lo if f_hi is None else f_hi
        m = self.band_mask(f_lo, f_hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_p = self.p[:, :, m].astype(np.float64).mean(axis=2) / self.w
        db = 10.0 * np.log10(np.maximum(mean_p, 1e-30))
        db[self.w < min_weight] = np.nan
        return db

    def relative_db(self, f_lo: float, f_hi: float | None = None, min_weight: float = 0.25) -> np.ndarray:
        """Like ``level_db`` but relative to the median level of the mapped area (shows the standing-wave pattern)."""
        db = self.level_db(f_lo, f_hi, min_weight)
        if np.all(np.isnan(db)):
            return db
        return db - np.nanmedian(db)

    def confidence(self, w_ref: float = 2.0) -> np.ndarray:
        return (1.0 - np.exp(-self.w / w_ref)).astype(np.float64)

    def coverage(self, min_weight: float = 0.5) -> float:
        return float(np.mean(self.w >= min_weight))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        np.savez_compressed(path, w=self.w, p=self.p, freqs=self.freqs, dims=self.dims.as_array(),
                            cell=self.cell, sigma=self.sigma)

    @classmethod
    def load(cls, path: str) -> "FieldMap":
        d = np.load(path)
        fm = cls(RoomDims(*[float(v) for v in d["dims"]]), d["freqs"], float(d["cell"]), float(d["sigma"]))
        fm.w, fm.p = d["w"].astype(np.float32), d["p"].astype(np.float32)
        return fm

    def sample(self, x: float, y: float) -> np.ndarray:
        """Mean power per line at the cell containing (x, y) (NaN if empty)."""
        ix = int(np.clip(x / self.dims.lx * self.nx, 0, self.nx - 1))
        iy = int(np.clip(y / self.dims.ly * self.ny, 0, self.ny - 1))
        w = float(self.w[iy, ix])
        return self.p[iy, ix].astype(np.float64) / w if w > 1e-6 else np.full(self.freqs.size, np.nan)
