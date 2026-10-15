"""Microphone calibration: frequency-response files and absolute SPL reference.

Supported file layouts (REW / Dayton / miniDSP style and CSV):

    "Dayton Audio iMM-6C  Serial 12345"         <- quoted or '*'/'#' header lines are skipped
    Sens Factor =-1.378dB, SERNO: 7000001      <- recognised and stored (informational only)
    20.0   -0.45   12.3                        <- freq  dB  [phase]
    25.0,  -0.40                               <- comma/semicolon/tab/space separated
    Frequency (Hz),Magnitude (dB)              <- non-numeric header rows are skipped

The file holds the microphone's deviation from flat in dB; the correction applied
to a measurement is therefore ``measured_dB - cal_dB(f)``.

Absolute level is a separate, explicit ``spl_offset_db`` (dBFS -> dB SPL).  It can
be entered by hand or derived from a 94 dB / 1 kHz acoustic calibrator.  Until one
is set the app reports dBFS and says so; relative results (decay times, spatial
maps, response shape) do not depend on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SENS_RE = re.compile(r"sens\s*factor\s*=\s*([-+]?\d+(?:\.\d+)?)\s*db", re.IGNORECASE)
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# A data row is 2+ numbers separated by whitespace / commas / semicolons and nothing else.
_ROW_RE = re.compile(rf"^[\s,;]*({_NUM})(?:[\s,;]+{_NUM})+[\s,;]*$")
_NUM_RE = re.compile(_NUM)


class CalibrationError(ValueError):
    pass


@dataclass
class MicCalibration:
    freqs: np.ndarray                      # Hz, strictly ascending
    mag_db: np.ndarray                     # mic deviation from flat, dB
    phase_deg: np.ndarray | None = None
    name: str = ""
    header: str = ""
    sens_factor_db: float | None = None    # informational ("Sens Factor" header), not applied
    source_path: str = ""

    def correction_db(self, f) -> np.ndarray:
        """Mic deviation (dB) at ``f``; interpolated on a log-frequency axis, held flat outside the file."""
        f = np.asarray(f, dtype=np.float64)
        lf = np.log10(np.maximum(f, 1e-3))
        return np.interp(lf, np.log10(self.freqs), self.mag_db)

    def apply_to_power(self, freqs, power) -> np.ndarray:
        """Remove the mic's response from a per-bin power spectrum."""
        return np.asarray(power) * 10.0 ** (-self.correction_db(freqs) / 10.0)

    @property
    def span(self) -> tuple[float, float]:
        return float(self.freqs[0]), float(self.freqs[-1])


def parse_calibration_text(text: str, name: str = "") -> MicCalibration:
    freqs: list[float] = []
    mags: list[float] = []
    phases: list[float] = []
    header_lines: list[str] = []
    sens: float | None = None
    have_phase = True

    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line:
            continue
        m = _SENS_RE.search(line)
        if m:
            sens = float(m.group(1))
        if line[0] in "*#" or line.startswith(("//", ";")):
            header_lines.append(line.lstrip("*#;/ "))
            continue
        work = line.replace('"', " ").replace("'", " ")
        if ";" in work:                       # European decimal commas: "20,5;0,3"
            work = work.replace(",", ".")
        if not _ROW_RE.match(work):
            if not freqs:
                header_lines.append(line.strip("\"' "))
            continue
        nums = _NUM_RE.findall(work)
        freqs.append(float(nums[0]))
        mags.append(float(nums[1]))
        if len(nums) >= 3:
            phases.append(float(nums[2]))
        else:
            have_phase = False

    if len(freqs) < 2:
        raise CalibrationError("No frequency/magnitude data found in calibration file.")
    f_arr = np.asarray(freqs)
    m_arr = np.asarray(mags)
    p_arr = np.asarray(phases) if (have_phase and len(phases) == len(freqs)) else None
    order = np.argsort(f_arr, kind="stable")
    f_arr, m_arr = f_arr[order], m_arr[order]
    if p_arr is not None:
        p_arr = p_arr[order]
    keep = np.concatenate(([True], np.diff(f_arr) > 0))     # drop duplicate frequencies
    f_arr, m_arr = f_arr[keep], m_arr[keep]
    if p_arr is not None:
        p_arr = p_arr[keep]
    if f_arr[0] <= 0:
        pos = f_arr > 0
        f_arr, m_arr = f_arr[pos], m_arr[pos]
        if p_arr is not None:
            p_arr = p_arr[pos]
    if f_arr.size < 2:
        raise CalibrationError("Calibration file needs at least two positive frequencies.")
    if np.max(np.abs(m_arr)) > 40:
        raise CalibrationError(
            f"Calibration deviations of up to {np.max(np.abs(m_arr)):.0f} dB look wrong "
            "(is this an SPL curve rather than a microphone correction?)."
        )
    return MicCalibration(
        freqs=f_arr, mag_db=m_arr, phase_deg=p_arr, name=name,
        header=" | ".join(h for h in header_lines if h)[:200], sens_factor_db=sens,
    )


def load_calibration(path: str | Path) -> MicCalibration:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise CalibrationError(f"Cannot read {p}: {e}") from e
    cal = parse_calibration_text(text, name=p.stem)
    cal.source_path = str(p)
    return cal


@dataclass
class Calibration:
    """Everything needed to turn raw dBFS into calibrated levels."""

    mic: MicCalibration | None = None
    spl_offset_db: float | None = None     # dB SPL = dBFS(RMS) + offset

    @property
    def unit(self) -> str:
        return "dB SPL" if self.spl_offset_db is not None else "dBFS"

    @property
    def offset(self) -> float:
        return 0.0 if self.spl_offset_db is None else float(self.spl_offset_db)

    def correct_power(self, freqs, power) -> np.ndarray:
        return self.mic.apply_to_power(freqs, power) if self.mic is not None else np.asarray(power)

    def describe(self) -> str:
        parts = [self.mic.name or "mic cal file" if self.mic else "no mic cal file"]
        parts.append(f"SPL offset {self.spl_offset_db:+.1f} dB" if self.spl_offset_db is not None else "uncalibrated level (dBFS)")
        return ", ".join(parts)


def spl_offset_from_calibrator(measured_dbfs_rms: float, calibrator_spl: float = 94.0) -> float:
    """Offset so that a calibrator tone reading ``measured_dbfs_rms`` shows ``calibrator_spl``."""
    return float(calibrator_spl - measured_dbfs_rms)
