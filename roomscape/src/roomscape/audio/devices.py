"""Audio device discovery (sounddevice/PortAudio), tolerant of a missing or broken backend."""

from __future__ import annotations

import sys
from dataclasses import dataclass

SIM_INDEX = -1
COMMON_RATES = (44100, 48000, 88200, 96000)

# Windows host-API preference: WASAPI is low-latency and honours the device's own sample rate;
# WDM-KS is hidden because it frequently fails to open and duplicates every other entry.
_HOSTAPI_RANK = {"Windows WASAPI": 0, "MME": 1, "Windows DirectSound": 2, "ALSA": 0, "Core Audio": 0, "JACK Audio Connection Kit": 3}
_HIDDEN_HOSTAPIS = {"Windows WDM-KS"}


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input: int
    max_output: int
    default_fs: float
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def is_simulated(self) -> bool:
        return self.index == SIM_INDEX

    @property
    def label(self) -> str:
        if self.is_simulated:
            return self.name
        return f"{self.name}  [{self.hostapi.replace('Windows ', '')}]"


SIM_DEVICE = DeviceInfo(SIM_INDEX, "Simulated room (no hardware)", "sim", 1, 1, 48000.0)

_backend_error: str | None = None


def _sd():
    """Import sounddevice lazily; remember why it failed so the UI can explain."""
    global _backend_error
    try:
        import sounddevice as sd  # noqa: PLC0415
        _backend_error = None
        return sd
    except Exception as e:  # PortAudio missing, no audio subsystem, etc.
        _backend_error = f"{type(e).__name__}: {e}"
        return None


def backend_error() -> str | None:
    _sd()
    return _backend_error


def _all_devices() -> list[DeviceInfo]:
    sd = _sd()
    if sd is None:
        return []
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in, default_out = sd.default.device
    except Exception as e:
        global _backend_error
        _backend_error = f"{type(e).__name__}: {e}"
        return []
    out: list[DeviceInfo] = []
    for i, d in enumerate(devices):
        api = hostapis[d["hostapi"]]["name"]
        if api in _HIDDEN_HOSTAPIS:
            continue
        out.append(DeviceInfo(
            index=i, name=d["name"], hostapi=api,
            max_input=int(d["max_input_channels"]), max_output=int(d["max_output_channels"]),
            default_fs=float(d["default_samplerate"]),
            is_default_input=(i == default_in), is_default_output=(i == default_out),
        ))
    return out


def _sorted(devs: list[DeviceInfo]) -> list[DeviceInfo]:
    return sorted(devs, key=lambda d: (_HOSTAPI_RANK.get(d.hostapi, 9), not (d.is_default_input or d.is_default_output), d.name.lower()))


def list_input_devices(include_sim: bool = True) -> list[DeviceInfo]:
    devs = _sorted([d for d in _all_devices() if d.max_input > 0])
    return ([SIM_DEVICE] if include_sim else []) + devs


def list_output_devices(include_sim: bool = True) -> list[DeviceInfo]:
    devs = _sorted([d for d in _all_devices() if d.max_output > 0])
    return ([SIM_DEVICE] if include_sim else []) + devs


def find_device(name_fragment: str, devices: list[DeviceInfo]) -> DeviceInfo | None:
    frag = name_fragment.lower()
    for d in devices:
        if frag in d.name.lower():
            return d
    return None


def likely_measurement_mic(devices: list[DeviceInfo]) -> DeviceInfo | None:
    """Pick a plausible measurement mic (e.g. Dayton iMM-6C shows up as a USB audio device)."""
    for key in ("imm-6", "imm6", "dayton", "umik", "usb audio", "usb mic", "measurement"):
        d = find_device(key, [x for x in devices if not x.is_simulated])
        if d:
            return d
    return None


def rates_for(device: DeviceInfo) -> list[int]:
    """Sample rates worth offering: the device default first, then the common ones."""
    rates = [int(device.default_fs)] if device.default_fs else []
    rates += [r for r in COMMON_RATES if r not in rates]
    return rates


def check_input(device: DeviceInfo, fs: int) -> str | None:
    """None if the device can record at ``fs``; otherwise a human-readable reason."""
    if device.is_simulated:
        return None
    sd = _sd()
    if sd is None:
        return f"Audio backend unavailable ({_backend_error})"
    try:
        sd.check_input_settings(device=device.index, channels=1, samplerate=fs, dtype="float32",
                                extra_settings=wasapi_settings(device))
    except Exception as e:
        return str(e)
    return None


def wasapi_settings(device: DeviceInfo):
    """Let Windows shared-mode WASAPI convert sample rate/format so any rate opens."""
    if "WASAPI" not in device.hostapi:
        return None
    sd = _sd()
    try:
        return sd.WasapiSettings(auto_convert=True)
    except Exception:
        return None


def platform_note() -> str:
    if sys.platform == "win32":
        return ("Windows: turn off 'Audio enhancements' for the mic (Sound settings > device Properties) and keep its "
                "input level fixed, otherwise levels will drift.")
    return ""
