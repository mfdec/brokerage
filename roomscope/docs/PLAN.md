# RoomScope roadmap

A simplified REW-style room analyzer for the Dayton Audio iMM-6C USB mic, Windows 11 first.
Stack: Python, PySide6 (Qt), pyqtgraph, sounddevice (PortAudio), NumPy / SciPy, OpenCV.

Guiding priorities (from the original brief): **room decay** and **walk-the-room mapping with a camera** matter most;
the live views must stay smooth (30+ fps, no audio dropouts); everything must be usable with a plain webcam, DroidCam
or an IP camera.

## Versions

| Version | Scope | Status |
|---|---|---|
| **v0.1** | Mic input, live RTA, level meter, mic calibration file, SPL reference, simulated room | done |
| **v0.2** | Live spectrogram and 3D waterfall at 30+ fps, no dropouts | done |
| **v0.3** | Exponential sweep, impulse response, **room decay** (EDT / T20 / T30 / RT60 per octave or 1/3 octave, C50 / C80, Schroeder curves), CSD waterfall, frequency response | done |
| **v0.4** | Room-mode calculator (axial / tangential / oblique), predicted bass response, matching against measured peaks, standing-wave field viewer | done |
| **v0.5** | Camera overlay: webcam / DroidCam / IP camera, pin floor points, pose + FOV tools, modelled sound field drawn on the live picture | done |
| **v0.6** | **Walk-the-room mapping**: coloured marker tracked by the camera, multisine mapping signal, measured field painted on the picture and plan views, model-vs-measured difference, spot decay sweeps pinned on the picture, `roomscope-doctor`, Windows launchers | **done (simulation-verified)** |
| v0.7 | Real-hardware hardening (see below), whatever the first Windows 11 run turns up | next |
| v1.0 | Polish: sessions, exports, REW alignment (see below) | planned |

## Next: v0.7 hardware hardening

Nothing below needs new features, only contact with reality. The checklist is in [VERIFICATION.md](VERIFICATION.md).

1. First run on Windows 11 with the iMM-6C: device list, WASAPI open at 44.1 / 48 kHz, noise floor, clipping indicator,
   `roomscope-doctor --all`.
2. Sweep on real speakers: auto-detected latency, S/N, RT60 against REW for the same room and position.
3. Camera: DroidCam client, IP Webcam URL, a plain USB webcam; marker colour picking under real light.
4. A real walk-the-room map checked against a handful of spot measurements (one fixed mic position per spot).
5. Fix whatever breaks; tune the defaults (signal level, tolerances, walking speed gate) from real data.

## v1.0 candidates (polish)

* **Distribution:** an installer (Inno Setup / MSIX) and code signing, so SmartScreen stops warning; a GitHub release per tag built on a Windows runner.
* **Sessions:** save / load everything (room, positions, calibration, measurements, maps) in one file; autosave on exit.
* **Exports:** CSV / WAV for a measurement and CSV / NPZ for a walk map already ship (v0.6); still to do: PNG of every plot and of the
  camera overlay, and one-click export of everything in a session.
* **REW alignment:** import REW `.txt` response exports and overlay them; match REW's RT60 conventions and smoothing so numbers
  can be compared one to one; optionally import REW impulse-response WAVs.
* **Lens distortion:** a barrel/pincushion slider (cheap webcams, wide-angle phone modes) applied before pinning and tracking.
  The pose model today is a pure pinhole camera.
* **Decay map:** interpolate RT60 pins into a plan view once several spot sweeps exist.
* **Multi-speaker / sub-pair modelling** in the modal model (the model has one source today).
* **Mapping above ~130 Hz:** a slower, denser walk or a stationary-grid mode with longer frames.
* **Mic stand height tracking:** estimate marker height from two views, or a tall-marker mode, instead of the fixed ear height.

## Decisions worth remembering

* **Why a multisine for mapping, and sweeps for decay?** The sweep gives the impulse response (needed for decay) but takes
  seconds per position. A periodic multisine has a *fixed* spectrum, so each 1 s frame while walking yields the level of 100+
  frequency lines at once, needs no sync between playback and capture, and its empty FFT bins measure the noise in the same
  frame. Decay at a spot is done by stopping and sweeping.
* **Marker mapped at mic height.** A floor homography alone would put a raised marker in the wrong place; the tracker
  back-projects the marker ray onto the plane at mic height using the full camera pose, so the lens FOV matters
  (hence the FOV tools).
* **Levels are dBFS until an SPL reference is set.** Relative results do not need absolute calibration; the app says so
  rather than guessing a sensitivity. The `Sens Factor` line in Dayton calibration files is read but deliberately not applied.
* **Python threads and the GIL.** PortAudio is read and written from plain threads with blocking calls and a generous latency;
  Python audio callbacks would need the GIL, which GUI painting holds for ~10 ms at a time.
