# What has been verified, and what has not

**Short version:** every algorithm and every screen has been exercised against a *simulation with known ground truth*
(~320 automated tests, run headless on Linux and, for the packaged program, on a Windows runner). **Nothing has been run
with the iMM-6C, with a real speaker, or with a real camera yet.** This file says exactly what the simulation can and cannot tell you, and gives the checklist
for closing the gap.

## 1. Verified by simulation

The simulation is deliberately more than a mock: a synthetic room built from the exact modal Green's function of a
rectangular room (bass), a direct path with 1/r level and fractional delay, and a diffuse tail with a per-octave
exponential decay of **known** RT60. A virtual camera renders that room (floor texture, walls, a coloured marker and
distractor objects) with a known pose and lens; a virtual person walks the marker through it in rows.

| Claim | How it was checked | Result |
|---|---|---|
| Level conventions (RMS dBFS, band power, tone levels, A/C/Z weighting) | analytic signals; IEC weighting tables | RMS / peak exact to 0.01 dB; A and C weighting within 0.06 dB of the IEC tables (0.7 dB at 8 kHz with fs = 44.1 kHz, a known bilinear-transform effect) |
| Mic calibration files (REW / Dayton / miniDSP / CSV) | parsing + interpolation tests, odd headers | correct; `Sens Factor` is read but not applied (by design) |
| Sweep, inverse filter, latency, gain | known linear systems with delay | gain ripple < 0.3 dB (0.19 dB observed), latency exact, harmonic distortion lands at negative time and is rejected |
| **RT60 / EDT / T20 / T30** | ideal exponential decays; full simulated chain (sweep → room → recording → analysis) | ideal decays within ~1 % even at 25 dB S/N; through the full chain every octave band 125 Hz – 8 kHz within about 4 % of the known value |
| Noise handling | Lundeby truncation, tail-corrected Schroeder integral | monotonic curves; T30 only reported with ≥ 45 dB dynamic range, T20 with ≥ 35 dB, else EDT, and the table says which |
| Speaker and mic on separate sound cards | sweep played with a clock offset of ±100 ppm and 1000 ppm | decay times unchanged (≤ 3 % broadband, ≤ 5 % per band) |
| Room modes and bass response | modal model vs. the synthetic room; peak matching | measured bass peaks land on predicted modes (≥ 80 %); coincident modes only counted below Schroeder frequency |
| Camera geometry | synthetic scenes, noisy clicks | pose from 4 pins; overlay geometry exact; a wrong FOV shows up as a higher fit error; FOV tools recover the true lens |
| Marker tracking | sim frames with noise, dimming, occlusion, distractors | marker found within about 1 px; room position within **~6 cm** at 0.5 m/s; re-acquires after occlusion; wrong FOV shows as a position offset |
| **Walk-the-room map** | virtual walk (32 s) vs. the true standing-wave field of the synthetic room | correlation **r = 0.98 / 0.98 / 0.91 / 0.96** at 34 / 52 / 69 / 100 Hz, RMSE 1.0 – 1.7 dB against a true pattern spanning ~25 dB; every frame accepted in the demo walk; coverage follows the walked area |
| **Decay camera** (burst sweeps while walking) | scripted burst chain through the simulated room; an exactly known damped-mode impulse response; a slow virtual walk in the UI | a perfect recording through the mapper's synchronisation, gating and windowing gives the same band EDT / T20 as analysing that impulse response directly; for a known 63 Hz mode (T60 0.70 s) T20 / T60 come out within 3 %; bursts recorded too fast, without a marker fix or with low S/N are rejected and counted; the demo walk accepted 30 of 30 bursts. **Not** claimed: that the decay map equals the broadband RT60 of the synthetic room (the synthetic room's bass is a damped modal sum plus a diffuse tail with its own acausal band-limiting ringing, so a band EDT at 63 Hz differs from the nominal T60 by up to ~15 %) |
| **0 Hz axes** | linear spectrogram / RTA tests, sweep from 5 Hz, response and CSD from the lower limits | DC bin never plotted, nothing breaks down to 0 Hz; what is physically measurable down there is limited by the mic, speaker and Windows (see README) |
| **AI analyst** | key store, the whole Messages-API tool loop against a scripted fake server (requests, tool results, retries, 401/404/429/529, refusal, cancel, step limit), evidence size and content, tools, report parsing (pins outside the room dropped), the panel end to end in the UI | the request/response handling is tested; **a real call to the Claude API has not been made from here** (no key, and no direct network route in the build sandbox). The first real run with your key is the test of that last hop |
| Wrong FOV in mapping | same walk with an 8° FOV error | positions shift, yet the standing-wave pattern is still recognisable (r > 0.6) |
| Spot sweeps while mapping | UI test | position tagged and pinned; sweep audio never enters the map; mapping signal resumes |
| Live performance | offscreen Qt, simulated mic | ~31 – 33 fps with spectrogram + 3D waterfall, ~3.5 ms per tick, **zero** audio dropouts; camera frame with overlay ~20 ms |
| Failure handling | fake PortAudio / fake cameras | missing backend, open failures, yanked mic, no camera, bad URL all end in a plain-language message, not a crash |

Everything above is in `tests/` (`pytest`). `scripts/make_screenshots.py` regenerates the pictures in the README from the
same simulation.

### The packaged Windows program

`RoomScape-<version>-win64.zip` is built with PyInstaller (`packaging/roomscape.spec`) from a pinned set of libraries
(`packaging/requirements-windows.txt`; the full test suite is run with exactly that set) by the GitHub Actions workflow
`.github/workflows/roomscape-windows.yml` on a **real Windows runner** (a commit message containing `[build exe]` triggers it). The
workflow then starts the built `RoomScape.exe --selftest` (libraries, DSP, live analyzer, sweep + decay, export, camera overlay,
walk-the-room mapping, decay camera, AI analyst offline checks incl. a real Windows Credential Manager round trip, screenshot) and
runs the full test suite on Windows. That shows the bundle is complete and the program runs on Windows; the runner has no sound
card, camera or screen, so the real audio stack, cameras and display scaling are still on the checklist below. Windows SmartScreen
will warn about the unsigned exe. (`packaging/build_exe_wine.sh` builds under Wine on Linux as a fallback; Wine cannot run the newest
libraries, so it is not used for releases.)

### Known limits of the simulation

* The synthetic room is rectangular and empty. Real rooms have furniture, non-rigid walls, bass traps, doors; their modes
  are less clean than the model's. The *measured* map is the point of the walk, not the model.
* The virtual camera is a perfect pinhole (no lens distortion, no motion blur, no rolling shutter, no auto-exposure,
  no auto white balance). Real marker colours drift with light.
* Audio is perfectly synchronised in the simulation. The real paths (PortAudio capture thread, separate playback stream,
  Windows' audio engine) add latency and jitter that were only reasoned about and tested with fakes.
* The simulated walker is a point on a perfect line. A real person disturbs the field a little and moves the mic around.

## 2. Needs real hardware (not yet done)

Tick these off on a Windows 11 machine. `roomscape-doctor --all` covers the first block.

**Software and audio**
- [ ] `scripts\setup.bat` completes; `scripts\run.bat` opens the window with the dark theme at your display scaling
- [ ] The packaged `RoomScape.exe` starts on your Windows 11 PC (it passed `--selftest` on a Windows runner), `RoomScape.exe --selftest` reports ALL CHECKS PASSED
- [ ] `roomscape-doctor --all` lists the iMM-6C, reports a sensible noise floor (quiet room: roughly -80 … -60 dBFS), no dropouts, sample rate within 1 %
- [ ] The mic opens at 44.1 and 48 kHz (WASAPI shared mode with auto-convert); *Rescan* finds a mic plugged in later
- [ ] Windows *Audio enhancements* off and a fixed input level: levels no longer drift when you clap / play music
- [ ] Live tab: 30+ fps with spectrogram and waterfall on your machine, xruns stay at 0 for 10 minutes
- [ ] Mic cal file loads (check against REW's cal import for the same file)

**Decay**
- [ ] A sweep through your speakers measures with S/N ≥ 40 dB; the detected latency looks plausible (output + input device latency)
- [ ] RT60 per octave within ~10 % of REW for the same room, mic position and a similar sweep (see section 3)
- [ ] A very dead room or a very live one both give sensible notes instead of nonsense numbers

**Decay camera and AI**
- [ ] The decay burst is audible and not clipping at the default -18 dBFS; the status line reports S/N of 25 dB or more
- [ ] Walking at ~0.3 m/s accepts most bursts; the EDT map at a mode frequency is higher near that mode's antinodes than at its nodes
- [ ] Decay camera EDT at a spot agrees with a stand-still sweep (Measure tab) at that spot to about 15 %
- [ ] AI analyst: save a real key, press Analyse, get a report; the pins land where the text says; note anything the report claims that the numbers do not support

**Camera and mapping**
- [ ] USB webcam opens (DirectShow, MJPG) and shows its name with `pygrabber` installed
- [ ] DroidCam client virtual webcam appears under *Find cameras*; the `http://PHONE-IP:4747/video` URL connects over Wi-Fi
- [ ] Pin four floor points; the fit error is below ~6 px; the outline hugs the real walls once the FOV is right
- [ ] The marker colour can be picked and tracked in your real lighting; no lock-on to other things in the room
- [ ] With a network camera, a *Camera delay* of 100 – 400 ms removes the along-the-path shift in the map
- [ ] **Walk-the-room accuracy check:** map the room, then place the mic at 4 – 6 marked spots and read the level at one mode
  frequency from a steady tone (Live tab, FFT mode, flat-top window) or a sweep; the walk map's relative levels at those
  spots should agree to within a few dB
- [ ] Spot sweep at a position tags the right place on the picture

## 3. Cross-checking against REW (recommended, 15 minutes)

REW is the reference for decay and response numbers. Do both measurements in the same session, same speaker, same mic
position, same Windows input level, quiet room.

1. **Mic cal:** load the same calibration file in both programs.
2. **Sweeps:** REW: 6 s sweep, 20 Hz – 20 kHz, level so peaks are about -15 dBFS, use the same output device.
   RoomScape: 6 s sweep, -15 dBFS (defaults).
3. **RT60:** in REW use *RT60* in the *Filtered IR* / *RT60* view with octave bands; compare T20 and T30 by band with
   RoomScape's table (RoomScape reports the fit it used per band; below 125 Hz expect more scatter in both programs).
   Agreement of ~10 % is good; larger gaps are usually the noise-floor truncation, the fit range (T20 vs T30), or a different
   sweep length.
4. **Frequency response:** compare RoomScape's *Frequency response* page (1/6 oct smoothing, 500 ms window) with REW's
   SPL graph at the same smoothing. Shapes should overlay; absolute level matches only if both are SPL-calibrated.
5. **Room modes:** the peaks RoomScape matches to modes should sit where REW's response shows its bass peaks.

Please send back (or commit) any difference above ~15 % in RT60 or ~2 dB in response shape together with the doctor report;
those are the places the processing needs tuning.

## 4. Honest limits of the method (not bugs)

* **Map resolution ≈ 0.4 m** and **useful up to ~130 Hz** with default settings. Narrow nulls look shallower than the model.
* **Single camera:** the marker height is assumed (ear height from the *Room modes* tab), the camera FOV must be known, and
  the lens is assumed distortion-free; wide-angle webcams need a narrower mode or careful pinning in the middle of the picture.
* **Decay at low frequencies** below ~100 Hz is statistically uncertain in small rooms (few modes): the B·T flag shows when.
* **Absolute SPL** needs the calibrator step; a typical measurement mic is accurate to ±1 dB with its cal file only if the
  Windows input level is fixed.
