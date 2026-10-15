"""``roomscape-doctor``: a plain-text health check of the software, audio and camera setup.

Run it when something does not work (no mic, no camera, odd levels) and paste the output into a bug report::

    roomscape-doctor                 # software + audio devices + a DSP self-test (a few seconds)
    roomscape-doctor --mic           # also record 3 s from the measurement mic and report its noise floor
    roomscape-doctor --cameras       # also look for webcams / the DroidCam client's virtual camera
    roomscape-doctor --url http://PHONE-IP:4747/video     # test a network camera
    roomscape-doctor --all --play    # everything, plus a quiet test tone through the output device
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np

from roomscape import __version__

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"
_TAGS = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]", INFO: "[info]"}      # plain ASCII: safe in any Windows console


@dataclass
class Check:
    status: str
    title: str
    detail: str = ""

    def format(self) -> str:
        head = f"{_TAGS[self.status]} {self.title}"
        if not self.detail:
            return head
        pad = " " * 8
        return head + "\n" + "\n".join(pad + line for line in self.detail.splitlines())


# ============================================================================================ software
def check_software() -> list[Check]:
    out = [Check(INFO, f"RoomScape {__version__}", f"Python {platform.python_version()} ({platform.architecture()[0]}) on {_os_name()}")]
    if sys.version_info < (3, 10):
        out.append(Check(FAIL, "Python is too old", "RoomScape needs Python 3.10 or newer."))
    for mod, label in (("numpy", "NumPy"), ("scipy", "SciPy"), ("cv2", "OpenCV"), ("PySide6", "PySide6 (Qt)"),
                       ("pyqtgraph", "pyqtgraph"), ("sounddevice", "sounddevice")):
        try:
            m = importlib.import_module(mod)
            out.append(Check(OK, f"{label} {getattr(m, '__version__', '')}".strip()))
        except Exception as e:                                          # missing wheel, broken DLL, ...
            out.append(Check(FAIL, f"{label} could not be imported", f"{type(e).__name__}: {e}\nTry: pip install -r requirements.txt"))
    return out


def _os_name() -> str:
    if sys.platform == "win32":
        try:
            build = sys.getwindowsversion().build
            return f"Windows {'11' if build >= 22000 else '10'} (build {build})"
        except Exception:
            pass
    return f"{platform.system()} {platform.release()}"


# ============================================================================================ audio
def check_audio_devices() -> list[Check]:
    from roomscape.audio import devices  # noqa: PLC0415

    sd = devices._sd()
    if sd is None:
        return [Check(FAIL, "Audio backend (PortAudio) is not available", f"{devices.backend_error()}\n"
                                                                            "Reinstall with: pip install --force-reinstall sounddevice")]
    out: list[Check] = []
    try:
        out.append(Check(INFO, "PortAudio", str(getattr(sd, "get_portaudio_version", lambda: ("?", ""))()[1])))
    except Exception:
        pass
    ins = devices.list_input_devices(include_sim=False)
    outs = devices.list_output_devices(include_sim=False)
    if not ins:
        out.append(Check(FAIL, "No audio input devices found",
                         "Plug in the USB mic. On Windows 11 also check Settings > Privacy & security > Microphone: "
                         "'Microphone access' and 'Let desktop apps access your microphone' must be on."))
    else:
        mic = devices.likely_measurement_mic(ins)
        lines = [f"{'*' if d.is_default_input else ' '} {d.label}   ({d.max_input} ch, default {d.default_fs:.0f} Hz)" for d in ins]
        out.append(Check(OK, f"{len(ins)} input device(s)" + (f"; measurement mic looks like '{mic.name}'" if mic else
                                                                  "; none looks like a measurement mic (that is fine, pick yours in the app)"),
                         "\n".join(lines) + "\n(* = Windows default)"))
    if not outs:
        out.append(Check(WARN, "No audio output devices found", "Sweeps and the mapping signal need speakers or an audio interface."))
    else:
        lines = [f"{'*' if d.is_default_output else ' '} {d.label}   ({d.max_output} ch, default {d.default_fs:.0f} Hz)" for d in outs]
        out.append(Check(OK, f"{len(outs)} output device(s)", "\n".join(lines)))
    note = devices.platform_note()
    if note:
        out.append(Check(INFO, "Windows microphone settings", note))
    return out


def _pick_input(spec: str | None):
    from roomscape.audio import devices  # noqa: PLC0415

    ins = devices.list_input_devices(include_sim=False)
    if spec:
        if spec.isdigit():
            hit = next((d for d in ins if d.index == int(spec)), None)
        else:
            hit = devices.find_device(spec, ins)
        return hit
    return devices.likely_measurement_mic(ins) or next((d for d in ins if d.is_default_input), None) or (ins[0] if ins else None)


def check_microphone(spec: str | None = None, fs: int | None = None, seconds: float = 3.0) -> list[Check]:
    """Record a few seconds and report noise floor, clipping, silence, dropouts and the real sample rate."""
    from roomscape.audio import devices  # noqa: PLC0415
    from roomscape.audio.engine import AudioError, HardwareInput  # noqa: PLC0415

    dev = _pick_input(spec)
    if dev is None:
        return [Check(FAIL, "Microphone test", f"No input device matches '{spec}'." if spec else "No input device found.")]
    fs = fs or int(dev.default_fs) or 48000
    reason = devices.check_input(dev, fs)
    if reason:
        return [Check(FAIL, f"'{dev.name}' cannot record at {fs} Hz", reason + "\nTry --rate 44100 (or 48000).")]
    src = HardwareInput(dev, fs, 0)
    try:
        src.start()
    except AudioError as e:
        return [Check(FAIL, f"Could not open '{dev.name}' at {fs} Hz", f"{e}\nIs another program using the mic exclusively?")]
    try:
        warm = min(0.7, 0.3 * seconds)                                  # streams take a moment to start delivering
        time.sleep(warm)
        t_a, n_a = time.monotonic(), src.ring.total
        time.sleep(max(seconds - warm, 0.2))
        t_b, total = time.monotonic(), src.ring.total
        xruns, err = src.xruns, src.error
        x = src.ring.read_latest(max(int((seconds - warm) * fs), 1024)).astype(np.float64)
    finally:
        src.stop()
    rate = (total - n_a) / max(t_b - t_a, 1e-3)                         # measured after warm-up, so start latency does not count
    title = f"Microphone '{dev.name}' @ {fs} Hz"
    if err:
        return [Check(FAIL, title, f"The stream stopped: {err}\nWas the USB mic unplugged?")]
    if total == 0:
        return [Check(FAIL, title, "No audio arrived at all. Check the device is not muted or disabled in Windows Sound settings.")]
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    db = lambda v: 20.0 * np.log10(max(v, 1e-12))          # noqa: E731
    facts = (f"RMS {db(rms):.1f} dBFS, peak {db(peak):.1f} dBFS, DC offset {np.mean(x):+.5f}, "
             f"{rate:.0f} samples/s received, {xruns} dropout(s)")
    checks: list[Check] = []
    if peak == 0.0:
        checks.append(Check(FAIL, title, facts + "\nDigital silence: the mic is muted, or Windows blocks desktop apps from using it "
                                                  "(Settings > Privacy & security > Microphone)."))
    elif peak >= 0.99:
        checks.append(Check(WARN, title, facts + "\nThe input is clipping. Lower the Windows input level for this mic."))
    elif db(rms) > -35.0:
        checks.append(Check(WARN, title, facts + "\nVery loud for a quiet room: lower the Windows input level, or something is making noise."))
    elif db(rms) < -95.0:
        checks.append(Check(WARN, title, facts + "\nExtremely quiet: the Windows input level may be near zero, or the mic is not the real source."))
    else:
        checks.append(Check(OK, title, facts + "\nThis is the room's noise floor as the app sees it; it limits how deep a decay can be measured."))
    if xruns:
        checks.append(Check(WARN, "Input dropouts", "Close other audio programs, or try another host API / sample rate."))
    if abs(rate - fs) / fs > 0.03:
        checks.append(Check(WARN, "Sample rate looks wrong", f"Expected about {fs} samples/s but got {rate:.0f}. Try another sample rate."))
    return checks


def check_output(spec: str | None = None, fs: int = 48000) -> list[Check]:
    """Play a short, quiet 1 kHz tone (only when asked for with --play)."""
    from roomscape.audio import devices  # noqa: PLC0415
    from roomscape.audio.engine import AudioError  # noqa: PLC0415
    from roomscape.audio.measure import HardwarePlayer  # noqa: PLC0415

    outs = devices.list_output_devices(include_sim=False)
    dev = None
    if spec:
        dev = next((d for d in outs if d.index == int(spec)), None) if spec.isdigit() else devices.find_device(spec, outs)
    else:
        dev = next((d for d in outs if d.is_default_output), outs[0] if outs else None)
    if dev is None:
        return [Check(FAIL, "Output test", "No matching output device.")]
    t = np.arange(int(0.8 * fs)) / fs
    tone = (10 ** (-30 / 20) * np.sin(2 * np.pi * 1000 * t) * np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)).astype(np.float32)
    player = HardwarePlayer(dev, "Both")
    try:
        player.play(tone, fs)
        end = time.monotonic() + 3.0
        while player.playing and time.monotonic() < end:
            time.sleep(0.02)
    except AudioError as e:
        return [Check(FAIL, f"Output '{dev.name}'", str(e))]
    finally:
        player.stop()
    return [Check(OK, f"Output '{dev.name}'", "A quiet 1 kHz tone was sent. If you heard nothing, check the Windows output device / volume.")]


# ============================================================================================ cameras
def _quiet_opencv():
    """Silence OpenCV's per-index 'can't open camera' warnings while probing; returns a function that restores the level."""
    try:
        import cv2  # noqa: PLC0415

        log = cv2.utils.logging
        old = log.getLogLevel()
        log.setLogLevel(log.LOG_LEVEL_ERROR)
        return lambda: log.setLogLevel(old)
    except Exception:
        return lambda: None


def check_cameras(max_index: int = 6) -> list[Check]:
    from roomscape.video.camera import CameraError, CameraSpec, OpenCvCamera, camera_names, probe_cameras  # noqa: PLC0415

    restore = _quiet_opencv()
    try:
        found = probe_cameras(max_index)
    finally:
        restore()
    if not found:
        return [Check(WARN, "No local camera found",
                      "USB webcams and the DroidCam Windows client (which adds a virtual webcam) appear here. Start the DroidCam client "
                      "and press Start first. Close other programs that hold the camera (Teams, Zoom, browser tabs). "
                      "Or use a network URL: roomscape-doctor --url http://PHONE-IP:4747/video")]
    names = camera_names()
    out = []
    for i in found:
        label = names[i] if i < len(names) else f"Camera {i}"
        cam = OpenCvCamera(CameraSpec.device(i, 1280, 720))
        try:
            cam.start()
            time.sleep(1.2)
            fr = cam.latest()
            size = f"{fr.image.shape[1]}x{fr.image.shape[0]}" if fr is not None else "no frame"
            out.append(Check(OK, f"{label} [index {i}]", f"{size} at about {cam.fps:.0f} fps"))
        except CameraError as e:
            out.append(Check(WARN, f"{label} [index {i}]", str(e)))
        finally:
            cam.stop()
    if sys.platform == "win32" and not names:
        out.append(Check(INFO, "Camera names", "Install 'pygrabber' (pip install pygrabber) to see friendly camera names."))
    return out


def check_url(url: str) -> list[Check]:
    from roomscape.video.camera import CameraError, CameraSpec, OpenCvCamera  # noqa: PLC0415

    cam = OpenCvCamera(CameraSpec.stream(url))
    try:
        cam.start()
        time.sleep(2.0)
        fr = cam.latest()
        if fr is None:
            return [Check(FAIL, f"Stream {url}", "Connected but no picture arrived.")]
        return [Check(OK, f"Stream {url}", f"{fr.image.shape[1]}x{fr.image.shape[0]} at about {cam.fps:.0f} fps")]
    except CameraError as e:
        return [Check(FAIL, f"Stream {url}", f"{e}\nPhone and PC must be on the same Wi-Fi; check the IP address and port shown in the phone app, "
                                              "and that the PC firewall allows Python.")]
    finally:
        cam.stop()


# ============================================================================================ DSP self-test
def check_dsp(seed: int = 0) -> list[Check]:
    """Run the full sweep -> impulse response -> RT60 chain on a synthetic room whose decay time is known."""
    from scipy.signal import fftconvolve  # noqa: PLC0415

    from roomscape.core.measurement import analyze_recording  # noqa: PLC0415
    from roomscape.core.modes import RoomDims  # noqa: PLC0415
    from roomscape.core.sweep import SweepSpec, recording_length, sweep_signal  # noqa: PLC0415
    from roomscape.core.synth import SyntheticRoom, T60Profile  # noqa: PLC0415

    fs, truth = 48000, 0.45
    t0 = time.perf_counter()
    try:
        room = SyntheticRoom(RoomDims(5.0, 4.0, 2.5), (0.4, 1.3, 1.1), T60Profile(truth), seed=seed)
        h = room.ir((3.5, 2.0, 1.2), fs, 2.0)
        spec = SweepSpec(fs=fs, duration=3.0, tail_s=1.5, level_dbfs=-15)
        n = recording_length(spec)
        y = fftconvolve(sweep_signal(spec).astype(np.float64), h)
        rec = np.zeros(n)
        m = min(y.size, n - 4000)
        rec[4000:4000 + m] = y[:m]
        rec += 10 ** (-90 / 20) * np.random.default_rng(seed).standard_normal(n)
        res = analyze_recording(rec, spec)
        t_mid = res.decay.t_mid()
    except Exception as e:
        return [Check(FAIL, "Signal-processing self-test", f"{type(e).__name__}: {e}")]
    dt = time.perf_counter() - t0
    err = abs(t_mid - truth) / truth
    if not np.isfinite(t_mid) or err > 0.12:
        return [Check(FAIL, "Signal-processing self-test", f"Expected T_mid {truth:.2f} s, got {t_mid:.2f} s. The numeric libraries misbehave on this machine.")]
    return [Check(OK, "Signal-processing self-test", f"Simulated room with RT60 {truth:.2f} s measured as {t_mid:.2f} s "
                                                       f"({err * 100:.1f} % off) in {dt:.1f} s")]


# ============================================================================================ CLI
def run(args: argparse.Namespace) -> list[Check]:
    checks = check_software()
    checks += check_audio_devices()
    if args.mic or args.all:
        checks += check_microphone(args.device, args.rate, args.seconds)
    if args.play:
        checks += check_output(args.output, args.rate or 48000)
    if args.cameras or args.all:
        checks += check_cameras(args.max_cameras)
    if args.url:
        checks += check_url(args.url)
    if not args.skip_dsp:
        checks += check_dsp()
    return checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="roomscape-doctor", description="Check RoomScape's audio, camera and software setup.")
    p.add_argument("--mic", action="store_true", help="record a few seconds from the mic and report its noise floor")
    p.add_argument("--device", help="input device: index or part of its name (default: the measurement mic, else the Windows default)")
    p.add_argument("--rate", type=int, help="sample rate for the mic test (default: the device's own)")
    p.add_argument("--seconds", type=float, default=3.0, help="mic test length (default 3)")
    p.add_argument("--play", action="store_true", help="play a quiet 1 kHz test tone through the output device")
    p.add_argument("--output", help="output device: index or part of its name (default: the Windows default)")
    p.add_argument("--cameras", action="store_true", help="look for local cameras")
    p.add_argument("--max-cameras", type=int, default=6, help="how many camera indices to try (default 6)")
    p.add_argument("--url", help="test a network camera stream, e.g. http://PHONE-IP:4747/video")
    p.add_argument("--all", action="store_true", help="mic + camera tests as well")
    p.add_argument("--skip-dsp", action="store_true", help="skip the signal-processing self-test")
    args = p.parse_args(argv)
    checks = run(args)
    print("RoomScape doctor\n")
    for c in checks:
        print(c.format())
    bad = sum(c.status == FAIL for c in checks)
    warn = sum(c.status == WARN for c in checks)
    print(f"\n{bad} problem(s), {warn} warning(s).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
