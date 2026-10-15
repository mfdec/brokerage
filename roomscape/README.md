# RoomScape

A simplified, REW-style **room analyzer for Windows 11** built around a Dayton Audio **iMM-6C** USB measurement
microphone. It does the things you actually need when treating a room, in one window:

| Tab | What it does |
|---|---|
| **Live** | Real-time analyzer (fractional-octave RTA or FFT), level meter, scrolling **spectrogram** and a rotatable **3D waterfall** at 30+ fps. |
| **Measure / Decay** | Plays a sweep through your speakers, records it, and gives you the **room decay**: RT60 / EDT / T20 / T30 per octave (or 1/3 octave), clarity C50 / C80, Schroeder decay curves, impulse response, frequency response and a **cumulative-spectral-decay waterfall**. |
| **Room modes** | Enter the room size, speaker and listening position: axial / tangential / oblique modes, predicted bass response, match against what you measured, and the standing-wave pattern in plan and section. |
| **Camera map** | Point a webcam, **DroidCam** or any **IP camera** at the room, pin four floor points, and the predicted standing-wave field is drawn **on the live picture**. |
| **Camera map → Walk-the-room** | Stick a coloured marker on the mic, play the mapping signal, and **walk the mic around the room**. The camera tracks the marker, the mic records the bass level, and the app *paints the measured sound field* onto the picture and onto three plan views (measured / model / difference). Stand still and press a button to run a **decay sweep at that spot**; each result is pinned on the picture. |
| **Camera map → Decay camera** | A separate mode next to the level map: a short **bass sweep repeats every 1.6 s** while you walk slowly with the marked mic. Every burst is turned into an impulse response, so the picture shows **how long the room keeps ringing at each spot** (EDT / T20 of a chosen band) and an **animation of the energy still to come** after the burst: where the sound lingers, and where it is gone first. |
| **Camera map → AI analyst** *(optional)* | With your own Anthropic (Claude) API key, Claude reads the measured numbers, looks up more detail with tools (modes, decay at a point, other bands) and explains **where and why** the room rings, with suggestions and **pins on the camera picture**. Off unless you add a key; nothing is sent until you press *Analyse*. |

![Live spectrogram](docs/screenshots/01_live_spectrogram.png)

![Room decay](docs/screenshots/03_decay_rt60.png)

![Walk-the-room map](docs/screenshots/09_walk_map_result.png)

![Decay camera: decay time (EDT) at 63 Hz on the picture and from above](docs/screenshots/12_decay_camera_edt.png)

![Decay camera: energy still to come 0.45 s after the burst](docs/screenshots/13_decay_camera_energy_left.png)

> **Status: v0.7.** Everything is implemented and verified against a physically-motivated *simulation* (a synthetic
> room with known decay times and a known standing-wave field, a virtual camera and a virtual person walking
> through it). It has **not yet been run against real hardware or on Windows**; see
> [docs/VERIFICATION.md](docs/VERIFICATION.md) for exactly what that means and a checklist to close the gap.

---

## Install (Windows 11)

### Easiest: the packaged program (no Python needed)

Download **`RoomScape-0.7.0-win64.zip`**, unzip the *whole folder* anywhere (for example `C:\RoomScape`) and double-click
**`RoomScape.exe`**. The exe is not code-signed, so Windows SmartScreen may say *"Windows protected your PC"*: click
*More info → Run anyway*. Keep the `_internal` folder next to the exe.

Two built-in checks run from a Command Prompt / PowerShell opened in that folder:
`RoomScape.exe --selftest` (about 20 s; exercises the program end to end with simulated hardware and writes a report and a
screenshot) and `RoomScape.exe --doctor --all` (audio devices, mic noise floor, cameras).

*Build it yourself:* on Windows run `scripts\setup.bat` then `scripts\build_exe.bat` (PyInstaller; result in `dist\RoomScape`).
On Linux, `packaging/build_exe_wine.sh` builds the same program under Wine.

### Or run from source

1. Install **Python 3.11 or 3.12** (64-bit) from <https://www.python.org/downloads/windows/>; tick *"Add python.exe to PATH"*.
2. Get this folder onto your PC (clone the repository or download the ZIP) and double-click **`scripts\setup.bat`**.
   It creates a private `.venv` and installs everything (NumPy, SciPy, OpenCV, Qt, pyqtgraph, sounddevice).
3. Double-click **`scripts\run.bat`** to start RoomScape.

Manual install, any OS: `python -m venv .venv`, activate it, then `pip install -e ".[camnames]"` and run `roomscape`.

**No hardware yet?** Start it with `scripts\run.bat --sim` (or choose *"Simulated room (no hardware)"* as the input).
Everything below works against the simulated room, simulated camera and a virtual walker, including the
**"Demo: virtual walk"** button on the Camera tab.

**Something not working?** Run `scripts\doctor.bat --all`. It lists your audio devices and cameras, records a few
seconds from the mic and reports its noise floor, and says in plain words what looks wrong.

## One-time hardware setup

**Microphone.** Plug in the iMM-6C. In *Windows Settings → System → Sound → Input → (the mic) → Properties*:
turn **Audio enhancements off**, and set the input **level to a fixed value** (e.g. 50 %) and leave it there; otherwise
Windows' automatic processing and gain changes will distort every measurement. In *Settings → Privacy & security →
Microphone* allow desktop apps to use the mic.

**Mic calibration file.** Dayton serves a per-unit calibration file for your serial number (REW / miniDSP text format).
Press **Mic cal file…** in the top bar and load it. RoomScape subtracts the mic's deviation from flat from the
response and band levels (decay times do not depend on it). Without it the app tells you the levels are uncorrected.

**Absolute level (optional).** Relative results (decay times, maps, response shape) do not need it. For dB SPL, press
**SPL reference** and either enter an offset or fit a 94 dB / 1 kHz calibrator on the mic and press *Measure calibrator*.
Do not touch the Windows input level afterwards.

**Speakers.** Sweeps and the mapping signal play through whatever output you pick on the *Measure* tab (WASAPI,
shared mode). Use the speaker or subwoofer you care about; the mic hears the room as that speaker excites it.
For the bass maps use the speaker/sub position you entered on the *Room modes* tab.

**Camera.** Any of these (pick the source on the *Camera map* tab, **Find cameras** scans the local ones):

| Camera | How |
|---|---|
| **USB webcam / laptop camera** | Appears in the list. 1280 x 720 is plenty. |
| **DroidCam (phone as webcam), PC client** | Install the DroidCam Client on the PC and the app on the phone, connect by USB or Wi-Fi in the client, press *Start*. The client adds a virtual webcam called *"DroidCam Source 3"*; **Find cameras** lists it. |
| **DroidCam without the PC client** | Choose *Network stream* and use `http://PHONE-IP:4747/video` (preset available). |
| **Android "IP Webcam" app** | `http://PHONE-IP:8080/video` (preset available). |
| **RTSP camera** | `rtsp://USER:PASSWORD@CAMERA-IP:554/stream1` (preset available). |

Phone and PC must be on the same Wi-Fi network for the network options. Close other programs (Teams, Zoom, browser tabs)
that might hold a local camera. For network cameras expect 100-400 ms of delay; the mapping panel has a **Camera delay**
setting to compensate.

## How to use it

### 1. Check the mic (Live tab)
Choose the iMM-6C in the top bar and press **Start listening**. The level meter, RTA and spectrogram move. Clap once:
you should see a broadband flash and, in a reverberant room, the decay tail on the spectrogram. Aim for a peak well below
full scale (a red clip LED latches for two seconds if the input hits full scale).

### 2. Measure the room decay (Measure / Decay tab)
Pick the output device (your speakers), keep the defaults (6 s sweep, -15 dBFS) and press **Measure**. Raise the amplifier
volume until the result reports a *signal-to-noise ratio* of 40 dB or more, and keep the room quiet. You get:

* **Room decay (RT60):** EDT, T20, T30 and RT60 per octave band, a broadband value, T_mid and bass ratio. Values that cannot
  be trusted (too little dynamic range, low B·T at bass, curved decays) are greyed with a note saying why. Click a band to
  see its Schroeder decay curve with the fitted RT60 line and where the noise floor was cut off. Set your target RT60 range to see it shaded on the chart.
* **Waterfall:** cumulative spectral decay in 3D or as a 2D decay map, bass presets with long windows for modal ringing.
* **Frequency response** with selectable smoothing and gating window, and the **impulse response**.

*Decay tail* controls how long it records after the sweep: use about 1.5 x the expected RT60.

**Export CSV / WAV...** writes the shown measurement to a folder: the impulse response as a float32 WAV (direct sound at the
index given in the JSON), the frequency response and the RT60 / EDT / T20 / T30 / C50 / C80 table as CSV, and a JSON with the sweep
settings, calibration, position and S/N. The walk map exports as CSV and as a compressed `.npz` (`np.load`: `w` weights (ny, nx),
`p` summed power (ny, nx, lines), `freqs`, `dims`, `cell`, `sigma`; level in dB = `10*log10(p/w)`).

### 3. Room modes
Enter length/width/height, the speaker and the listening position (0,0,0 = front-left floor corner) and the room's bass
decay time (button: *take from last measurement*). The tab lists the modes, shows the predicted bass response at the
listening position and, after a measurement, marks which measured peaks match which modes, plus the standing-wave
pattern at any frequency in plan and section views. The Schroeder frequency tells you up to where modal theory applies.

### 4. Camera overlay (model)
On the *Camera map* tab: **(1)** connect the camera. **(2)** Put four tape marks on the floor at known positions
(measured from the front-left corner; spread out, not in a line; room corners work if visible), click **Click points in
picture** and click each mark in the order listed. **(3)** Set the lens field of view: *Estimate from perspective*,
*From camera height* (if you know how high the camera is) or *Marker above pin* (hold the marker at mic height above a
pin and click it) pin it down. The white room outline must hug the real walls and ceiling. **(4)** The predicted standing
wave at the chosen frequency appears on the picture (warm = louder than the room average, cool = quieter);
*Snap to nearest mode* jumps to a resonance. The map is drawn at ear height by default; set *Map height* to 0 for the floor.

### 5. Walk-the-room mapping
1. Fix a **bright, matte, saturated ball or sticker** (at least 5-8 cm across: a ball, or a paper disc taped to a stand) to the mic
   stand or the mic itself, in a colour nothing else in the view has (hot pink, lime green and orange work well; avoid glossy
   surfaces and anything near the skin tone).
2. **Pick marker colour**, then click the marker in the picture. It is remembered for this camera.
3. Press **Play mapping signal** (a 20-300 Hz multisine; the level setting is its peak digital level). Raise the amplifier
   until the status line says the signal-to-noise ratio is good (10 dB or more).
4. Press **Start mapping** and walk slowly (about 0.5 m/s) over the room, holding the mic at ear height and keeping the
   marker visible to the camera. Each second of audio paints the spot where the marker was. The status line tells you when
   you are too fast, the marker is lost, the input clips, or the signal is too weak.
5. The picture overlay and the three plan views fill in. Choose what the picture shows (**Model / Measured / Measured minus
   model**), pick the frequency, and use *Snap to nearest mode* to see the resonances. Walk the gaps until coverage is high.
6. **Measure sweep here:** stand still at a spot and press it (or tick the automatic option, which fires after 4 s of
   standing still). The mapping signal pauses, a decay sweep runs, the result is added to the *Measure* tab tagged with its
   position, pinned on the picture with its RT60, and the mapping signal resumes.
7. **Save / Load** keeps a map for later, **CSV** exports it as a table (one row per cell: `x_m`, `y_m`, `weight`, then the level in dB of every
   2 Hz line, ready for pandas or a spreadsheet), **Clear map** starts over. Your mapping settings and each camera's marker colour and delay are remembered.

![Walk mapping in progress](docs/screenshots/08_walk_mapping_in_progress.png)

Tips: walk in rows like mowing a lawn; stay a little away from the walls; keep your body out of the line between the speaker and
the mic where you can (hold the mic out on a pole or stand); if the painted pattern looks shifted along your path with a network
camera, raise **Camera delay**.

### 6. Decay camera (how long the room rings, spot by spot)
Uses the same marker, camera calibration and walking technique as step 5, but a different signal: a **20-300 Hz sweep burst (0.5 s)
followed by silence, repeated every 1.6 s**. Each burst is recorded, deconvolved into an impulse response and analysed in the
band you picked in step 4 (1/3, 1/6 or 1 octave), so you get a real decay measurement at the marker's position every 1.6 s.

1. Do steps 1-3 of the walk-the-room mapping (marker colour, camera calibration). Set the *Frequency* in step 4 to the band you care about.
2. In **6 Decay camera** press **Start decay mapping** and walk **slowly (about 0.3 m/s; the status line warns if you are faster)**.
   Bursts recorded while you move too fast, with a low signal-to-noise ratio or without a marker fix are skipped and counted.
3. Choose what the picture shows: **EDT** or **T20** (warm = the room rings longer here, cool = shorter; colours are relative to the median),
   or **Energy left after the burst** and press **Play decay animation** (or drag *Time after burst*): the colours show how much of the
   energy is still arriving *t* seconds after the burst, so you can watch the ringing fade from the dead spots first and linger in the
   slow ones. A fourth plan view shows the EDT map from above. **Save / Load / CSV** work as for the level map.
4. There is a **Demo: virtual slow walk** for the simulated room.

Decay times in a small room below the Schroeder frequency are properties of individual modes, so expect them to *vary across the room*; that
variation is the point of the map. Absolute values from a walking burst measurement are good to roughly +-15 %; the pattern is what to trust.

### 7. AI analyst (optional, needs your own Claude API key)
In the *7 AI analyst* group press **Save key** with a key from <https://console.anthropic.com/> (kept in the **Windows Credential Manager**,
never in a file; `ANTHROPIC_API_KEY` in the environment also works). **Preview what is sent...** shows the exact data: the room model,
the mode list, the sweep measurements' summary and the decay map's statistics and coarse grid (plus one camera picture if you tick
that box). **Analyse with Claude** sends it with your question; Claude may ask the app for more (decay at a point, another band, mode
pressure at a point). The answer is a summary, findings with the numbers they rest on, things to try, the limits of the data, and
**pins on the camera picture** for the spots it talks about.

The physics stays in the app: every number comes from the analysis above and Claude only interprets it, so the report is labelled
*AI estimate*. It is far better than a fixed rule at connecting what you measured (a slow corner at 63 Hz, a mode whose antinode
sits there, a source position that excites it) but it can still be wrong; use it to decide what to try and re-measure to confirm.
Costs are billed by Anthropic to your key (a typical run is a few thousand tokens). Without a key the program never goes online.

## What to expect (accuracy and limits)

* **The frequency axes start at 0 Hz.** The live RTA / spectrogram / 3D waterfall have *linear* axes from 0 Hz (and logarithmic ones
  down to 2 Hz), the sweep can start at 5 Hz, and the room-mode plots and response start at 0 Hz. Be aware of what is physical: the iMM-6C
  is a small electret capsule whose response rolls off in the very low bass, speakers rarely reproduce anything below 20 Hz and the
  Windows audio path may high-pass at a few Hz, so the 0-20 Hz region mostly shows noise, room rumble and the mic's own roll-off.

* **Decay times** are measured with the standard method (sweep, inverse filter, band filtering, Lundeby noise truncation,
  Schroeder integration). Against a synthetic room with known decay times the full chain recovered every octave band from 125 Hz
  to 8 kHz within about 4 %. Below ~125 Hz decay is dominated by individual modes and the statistical flags (B·T) tell you
  when a band value is only indicative.
* **The measured map has a spatial resolution of roughly 0.4 m** (each 1 s of audio is smeared over the area you walked
  in that second). Standing waves are metres long, so this is plenty; it does mean the map cannot show the sharp null of a
  perfect mode as deep as the model does, and the *measured minus model* view will show a stripe there. That is expected.
* **Mapping works up to about 130 Hz** with the default settings (line spacing 2 Hz, walking speed 0.5 m/s). Above that, wavelengths
  get shorter than the map resolution.
* The camera estimates the mic's room position from a **single camera** and a **floor calibration**, so the lens field
  of view matters (a wrong value is invisible on the floor but shifts the mic mapping); use the FOV tools in step 3.
  In simulation the marker was located to about 6 cm.
* Levels are **dBFS** until you set an SPL reference. Comparing between rooms needs the same Windows input level.

## Troubleshooting

| Symptom | What to do |
|---|---|
| No microphone in the list | Run `roomscape-doctor`; check *Privacy & security → Microphone*, unplug/replug, press *Rescan*. |
| Levels drift or look compressed | Turn **Audio enhancements** off and fix the input level (see above). |
| "No clear sweep found" | The speaker is off, muted or not the selected output, the volume is too low, or the wrong mic is selected. |
| Camera not listed | Start the DroidCam client first and press *Start*; close other camera apps; use *Network stream* instead. |
| Network camera "could not open" | Same Wi-Fi? Correct IP/port? Windows Firewall may ask to allow Python: allow it on private networks. |
| Marker not found | Marker must be in view, saturated, unique in colour; raise *Colour tolerance* a little or enable *Follow slow colour drift*; avoid glare. |
| Map looks shifted or mirrored | Check *Mirror image* for selfie-style cameras; recheck the pin order and the FOV; raise *Camera delay*. |
| Fit error is high after pinning | A pin is on the wrong spot: compare the numbers in the table with where you clicked. |

## For developers

```text
src/roomscape/
  core/    signal processing: spectrum, sweep/deconvolution, decay, modes, geometry, multisine, field map, mapping, decay map (no Qt)
  ai/      optional Claude analyst: Messages-API client + tool loop, evidence/tools/report parsing, key storage (no Qt)
  audio/   sounddevice input/output threads, ring buffer, device discovery
  video/   camera capture, overlay drawing, marker tracking, simulated camera
  sim/     simulated room, virtual walker (used by --sim, the tests and the screenshots)
  ui/      Qt: main window, tabs, widgets
tests/     ~320 tests; GUI tests run headless (Qt offscreen)
docs/      PLAN.md (roadmap), VERIFICATION.md (what is verified and how), screenshots/
scripts/   Windows launchers, make_screenshots.py
```

```bash
pip install -e ".[dev]"
pytest                          # full suite, takes a few minutes (the simulation runs in real time)
python scripts/make_screenshots.py
```

See [CLAUDE.md](CLAUDE.md) for architecture notes and the gotchas that cost time (level conventions, the GIL and `QTest.qWait`,
OpenCV kernel quirks).

## Licence

MIT. Dayton Audio and iMM-6C are trademarks of their owners; RoomScape is an independent hobby project and is not affiliated
with them or with REW.
