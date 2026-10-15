"""IEC 61672 A / C frequency weightings (analytic magnitude and streaming IIR filters)."""

from __future__ import annotations

import numpy as np
from scipy.signal import bilinear_zpk, sosfreqz, zpk2sos

WEIGHTINGS = ("Z", "A", "C")


def weighting_db(f, kind: str = "A") -> np.ndarray:
    """Weighting in dB at frequency ``f`` (Hz).  ``Z`` is flat."""
    f = np.asarray(f, dtype=np.float64)
    f2 = f * f
    if kind == "A":
        ra = (12194.0 ** 2 * f2 * f2) / (
            (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)
        )
        return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0
    if kind == "C":
        rc = (12194.0 ** 2 * f2) / ((f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2))
        return 20.0 * np.log10(np.maximum(rc, 1e-30)) + 0.06
    return np.zeros_like(f)


def weighting_sos(fs: float, kind: str) -> np.ndarray | None:
    """Second-order sections of the weighting filter at sample rate ``fs`` (None for Z)."""
    if kind == "Z":
        return None
    w1, w2, w3, w4 = (2 * np.pi * v for v in (20.598997, 107.65265, 737.86223, 12194.217))
    if kind == "A":
        z = np.zeros(4)
        p = -np.array([w1, w1, w2, w3, w4, w4])
    elif kind == "C":
        z = np.zeros(2)
        p = -np.array([w1, w1, w4, w4])
    else:
        raise ValueError(f"unknown weighting {kind!r}")
    zd, pd, kd = bilinear_zpk(z, p, 1.0, fs)
    sos = zpk2sos(zd, pd, kd)
    _w, h = sosfreqz(sos, worN=np.array([2 * np.pi * 1000.0 / fs]))
    sos[0, :3] /= abs(h[0])                  # 0 dB at 1 kHz
    return sos
