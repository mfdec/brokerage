"""Getting measurements out of RoomScape: impulse response as WAV, response and decay tables as CSV, details as JSON."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .measurement import FrequencyResponse, MeasurementResult, frequency_response


def _stem(res: MeasurementResult) -> str:
    label = re.sub(r"[^A-Za-z0-9._@(),+-]+", "_", res.label or "measurement").strip("_") or "measurement"
    return f"roomscape_{time.strftime('%Y%m%d_%H%M%S', time.localtime(res.created))}_{label}"


def _num(x: float, fmt: str = "{:.4g}") -> str:
    return "" if x is None or not np.isfinite(x) else fmt.format(x)


def decay_rows(res: MeasurementResult) -> list[list[str]]:
    """RT60 table: one row per band, plus the broadband row."""
    head = ["band_hz", "f_lo_hz", "f_hi_hz", "edt_s", "t20_s", "t30_s", "rt60_s", "basis", "dynamic_range_db",
            "c50_db", "c80_db", "d50_pct", "bandwidth_x_t60", "reliable", "notes"]
    rows = [head]
    for b in res.decay.bands + [res.decay.broadband]:
        broadband = b is res.decay.broadband
        rows.append(["broadband" if broadband else _num(b.fc), _num(b.f_lo), _num(b.f_hi), _num(b.edt, "{:.3f}"), _num(b.t20, "{:.3f}"),
                     _num(b.t30, "{:.3f}"), _num(b.t60, "{:.3f}"), b.basis, _num(b.dynamic_range_db, "{:.1f}"), _num(b.c50_db, "{:.2f}"),
                     _num(b.c80_db, "{:.2f}"), _num(b.d50_pct, "{:.1f}"), _num(b.bt, "{:.1f}"), str(bool(b.reliable)).lower(),
                     "; ".join(b.notes)])
    return rows


def export_measurement(res: MeasurementResult, folder: str | Path, response: FrequencyResponse | None = None) -> list[Path]:
    """Write ``<stem>_impulse.wav`` (float32, the system gain output/input, direct sound at ``t0``), ``<stem>_response.csv``,
    ``<stem>_rt60.csv`` and ``<stem>_info.json``.  Returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _stem(res)
    out: list[Path] = []

    wav = folder / f"{stem}_impulse.wav"
    wavfile.write(str(wav), int(res.ir.fs), res.ir.h.astype(np.float32))
    out.append(wav)

    fr = response or frequency_response(res.ir, res.spec, res.calibration)
    path = folder / f"{stem}_response.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", f"level_{fr.unit.replace(' ', '_')}"])
        for f, v in zip(fr.freqs, fr.level_db):
            w.writerow([f"{f:.3f}", f"{v:.3f}"])
    out.append(path)

    path = folder / f"{stem}_rt60.csv"
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(decay_rows(res))
    out.append(path)

    path = folder / f"{stem}_info.json"
    info = {
        "label": res.label, "position_m": list(res.position) if res.position else None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(res.created)),
        "sample_rate_hz": int(res.ir.fs), "direct_sound_index": int(res.ir.t0), "latency_samples": int(res.ir.latency_samples),
        "sweep": {"duration_s": res.spec.duration, "f1_hz": res.spec.f1, "f2_hz": res.spec.f2, "level_dbfs": res.spec.level_dbfs,
                  "tail_s": res.spec.tail_s},
        "snr_db": res.snr_db, "level_unit": fr.unit, "response_window_s": fr.window_s, "response_smoothing_oct": fr.smoothing_oct,
        "mic_calibration": res.calibration.mic.name if res.calibration.mic is not None else None,
        "spl_offset_db": res.calibration.spl_offset_db,
        "t_mid_s": None if not np.isfinite(res.decay.t_mid()) else res.decay.t_mid(),
        "warnings": res.warnings,
        "impulse_wav_note": "float32 WAV of the system gain (output / input); the direct sound is at sample 'direct_sound_index'.",
    }
    path.write_text(json.dumps(info, indent=2))
    out.append(path)
    return out
