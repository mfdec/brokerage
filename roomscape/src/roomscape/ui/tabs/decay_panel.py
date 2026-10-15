"""Decay camera (Camera tab, step 6): walk the room with the marked mic while bursts play; see how long the room rings, spot by spot.

Reuses the marker, tracking and calibration of the walk-the-room panel.  Overlays: decay time (EDT or T20) of the chosen band, and
the energy that is still to come *t* seconds after a burst - played as an animation that shows where the sound lingers.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from roomscape.audio.engine import AudioError
from roomscape.audio.measure import HardwarePulsePlayer
from roomscape.core.decaymap import (
    T_GRID_MAX, DecayFields, DecaySettings, DecayWalkMapper, PulseSpec, build_fields, load_frames, save_frames,
)
from roomscape.sim.world import LawnmowerPath, SimPulsePlayer
from roomscape.ui import theme
from roomscape.video import overlay
from roomscape.video.sim import MARKER_BGR

SOURCES = ["Off (show the level map / model)", "Decay time, EDT (early decay)", "Decay time, T20", "Energy left after the burst (animated)"]
WIDTHS = [("1/3 octave", 1.0 / 3.0), ("1/6 octave", 1.0 / 6.0), ("1 octave", 1.0)]
SPEEDS = [("Real time", 1.0), ("Half speed", 0.5), ("Quarter speed", 0.25)]


def _spin(lo, hi, val, step, suffix="", dec=1) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setValue(val)
    return s


class DecayPanel(QtCore.QObject):
    def __init__(self, tab, mapping):
        super().__init__(tab)
        self.tab, self.mapping, self.state = tab, mapping, tab.state
        self.mapper: DecayWalkMapper | None = None
        self.stim = None
        self.running = False
        self._cursor = 0
        self._metric_cache: dict = {}
        self._fields: DecayFields | None = None
        self._fields_key = None
        self._tex_key = None
        self._tex = None
        self._last_label_t = 0.0
        self._note_shown = None
        self.anim_t = 0.0
        self.anim_playing = False
        self._anim_wall = time.monotonic()
        self._build()
        tab.pre_frame_hooks.append(self.on_frame)
        tab.overlay_providers.append(self.texture)
        tab.overlay_busy = lambda: self.source.currentIndex() != 0
        tab.extra_draw.append(self.draw)
        tab.shutdown_hooks.append(self.shutdown)
        self.state.sourceChanged.connect(self._source_changed)
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.setInterval(33)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.anim_timer.start()
        self.plan_timer = QtCore.QTimer(self)
        self.plan_timer.setInterval(500)
        self.plan_timer.timeout.connect(self._update_plan)
        self.plan_timer.start()

    # ================================================================== UI
    def _build(self) -> None:
        box = QtWidgets.QGroupBox("6  Decay camera")
        f = QtWidgets.QFormLayout(box)
        how = QtWidgets.QLabel(
            "Shows how long the room keeps ringing at each spot. A short bass sweep repeats every 1.6 s while you walk slowly (about "
            "0.3 m/s, slower than for the level map) with the marked mic; each burst measures the decay where the marker is. Uses the "
            "marker colour and the camera calibration from step 5. The frequency band is the one chosen in step 4.")
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(how)
        self.level = _spin(-45, -6, -18, 1, " dBFS", 0)
        self.f1 = _spin(4, 100, 20, 2, " Hz", 0)
        self.f2 = _spin(80, 600, 300, 10, " Hz", 0)
        self.f1.setToolTip("Lowest frequency of the burst. Below about 20 Hz most speakers cannot follow and large excursions can damage them.")
        self.vmax = _spin(0.1, 1.5, 0.7, 0.1, " m/s", 1)
        self.snr = _spin(10, 60, 25, 1, " dB", 0)
        self.run_btn = QtWidgets.QPushButton("Start decay mapping")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._toggle)
        f.addRow("Burst level", self.level)
        f.addRow("Sweep from", self.f1)
        f.addRow("Sweep to", self.f2)
        f.addRow("Max walking speed", self.vmax)
        f.addRow("Min signal-to-noise", self.snr)
        f.addRow(self.run_btn)
        self.status = QtWidgets.QLabel("Not mapping.")
        self.status.setWordWrap(True)
        f.addRow(self.status)

        self.source = QtWidgets.QComboBox()
        self.source.addItems(SOURCES)
        self.width = QtWidgets.QComboBox()
        for lbl, v in WIDTHS:
            self.width.addItem(lbl, v)
        self.width.setToolTip("Width of the band around the frequency from step 4 that the decay is measured in.")
        self.speed = QtWidgets.QComboBox()
        for lbl, v in SPEEDS:
            self.speed.addItem(lbl, v)
        self.speed.setCurrentIndex(1)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, int(T_GRID_MAX * 100))
        self.play_btn = QtWidgets.QPushButton("Play decay animation")
        self.play_btn.clicked.connect(self._toggle_anim)
        self.time_slider.valueChanged.connect(self._slider_moved)
        self.source.currentIndexChanged.connect(self._source_picked)
        f.addRow("Overlay shows", self.source)
        f.addRow("Band width", self.width)
        f.addRow("Animation speed", self.speed)
        f.addRow("Time after burst", self.time_slider)
        f.addRow(self.play_btn)

        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Clear", self._clear), ("Save...", self._save), ("Load...", self._load), ("CSV...", self._export_csv)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow(row)
        self.demo_btn = QtWidgets.QPushButton("Demo: virtual slow walk (simulated room)")
        self.demo_btn.clicked.connect(self._demo_walk)
        f.addRow(self.demo_btn)
        self.tab.extra_box.addWidget(box)

    # ================================================================== run
    def _spec(self) -> PulseSpec:
        src = self.state.source
        fs = src.fs if src is not None else 48000
        return PulseSpec(fs=fs, f1=self.f1.value(), f2=max(self.f2.value(), self.f1.value() * 2.0), level_dbfs=self.level.value())

    def _stop_level_map(self) -> None:
        mp = self.mapping
        if mp.mapping:
            mp._toggle_mapping()
        if mp.stim is not None and mp.stim.playing:
            mp._toggle_stim()

    def _toggle(self) -> None:
        if self.running:
            self.stop()
            return
        src = self.state.source
        if src is None or not src.running:
            self.status.setText("Start the microphone input first (top bar).")
            return
        if self.tab.cam_model is None:
            self.status.setText("Calibrate the camera first (steps 1-3).")
            return
        if self.mapping.tracker.target is None:
            self.status.setText("Pick the marker colour first (step 5).")
            return
        self._stop_level_map()
        spec = self._spec()
        try:
            if self.state.world is not None:
                self.stim = SimPulsePlayer(self.state.world)
            else:
                dev, ch = self.tab.output_provider() if self.tab.output_provider else (None, "Both")
                if dev is None or dev.is_simulated:
                    self.status.setText("Choose the output device (your speakers) on the Measure tab first.")
                    return
                self.stim = HardwarePulsePlayer(dev, ch)
            self.stim.start(spec)
        except AudioError as e:
            self.stim = None
            QtWidgets.QMessageBox.warning(self.tab, "Decay bursts", str(e))
            return
        old = self.mapper
        self.mapper = DecayWalkMapper(self.state.room.dims, DecaySettings(spec=spec, v_max=self.vmax.value(), snr_min_db=self.snr.value(),
                                                                          camera_delay_s=self.mapping.delay.value() / 1000.0))
        if old is not None and old.frames and old.s.spec == spec and old.dims == self.mapper.dims:
            self.mapper.frames = old.frames                     # keep what has been mapped so far
        self._cursor = src.ring.total
        self.mapper.reset_audio_clock()
        self.running = True
        self._repolish(True)
        self.status.setText("Bursts playing - raise the amplifier volume until the signal-to-noise below is good, then walk slowly.")

    def stop(self) -> None:
        if self.stim is not None:
            self.stim.stop()
            self.stim = None
        self.running = False
        self._repolish(False)

    def _repolish(self, on: bool) -> None:
        self.run_btn.setText("Stop decay mapping" if on else "Start decay mapping")
        self.run_btn.setObjectName("danger" if on else "primary")
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)

    # ================================================================== per frame
    def on_frame(self, img: np.ndarray, fr) -> None:
        if not self.running or self.mapper is None or fr is None:
            return
        mp = self.mapping
        if mp.mapping or (mp.stim is not None and mp.stim.playing):
            self.stop()
            self.status.setText("Stopped: the level map took over the speakers.")
            return
        st = mp.track
        s = self.mapper.s
        s.v_max, s.snr_min_db, s.camera_delay_s = self.vmax.value(), self.snr.value(), mp.delay.value() / 1000.0
        self.mapper.feed_position(fr.t, st.xy, st.valid)
        src = self.state.source
        if src is not None and src.running:
            new, self._cursor, over = src.ring.read_since(self._cursor)
            if over:
                self.mapper.reset_audio_clock()
            if new.size:
                self.mapper.feed_audio(new, time.monotonic())
        now = time.monotonic()
        if now - self._last_label_t >= 0.25:
            self._last_label_t = now
            self._update_status()

    def _update_status(self) -> None:
        m, st = self.mapper, self.mapping.track
        c = m.counts
        hint, color = "", theme.MUTED
        if not m.locked:
            hint, color = "Listening for the bursts - is the speaker on and the volume up?", theme.AMBER
        elif not st.valid:
            hint, color = "Marker lost - keep it visible to the camera.", theme.AMBER
        elif st.speed > self.vmax.value():
            hint, color = "Too fast - slow down.", theme.AMBER
        elif np.isfinite(m.last_snr) and m.last_snr < self.snr.value():
            hint, color = "Bursts too weak - raise the volume.", theme.AMBER
        snr = f"{m.last_snr:.0f} dB" if np.isfinite(m.last_snr) else "-"
        n = sum(c.values())
        self.status.setText(f"<span style='color:{color}'>{hint}</span><br>Bursts used: {c['accepted']}/{n} · S/N {snr} · "
                            f"{st.speed:.2f} m/s")

    # ================================================================== fields and overlay
    def _band(self) -> tuple[float, float]:
        return float(self.tab.freq.value()), float(self.width.currentData())

    def fields(self) -> DecayFields | None:
        m = self.mapper
        if m is None or not m.frames:
            return None
        fc, bw = self._band()
        key = (len(m.frames), round(fc, 3), round(bw, 4), self.state.room.dims)
        if key != self._fields_key:
            self._fields = build_fields(m.frames, self.state.room.dims, fc, bw, cache=self._metric_cache)
            self._fields_key = key
            self._tex_key = None
        return self._fields

    def _note(self, text: str | None) -> None:
        text = text or self.mapping._NOTE_DEFAULT
        if text != self._note_shown:
            self._note_shown = text
            self.tab.overlay_note.setText(text)

    def texture(self):
        mode = self.source.currentIndex()
        if mode == 0:
            return None
        fields = self.fields()
        if fields is None or fields.n_used == 0:
            self._note("Decay camera: nothing measured yet - start decay mapping and walk slowly.")
            return None
        c = self.tab
        alpha = np.clip(fields.confidence() * 1.3, 0.0, 1.0)
        key = (mode, self._fields_key, round(self.anim_t, 2) if mode == 3 else 0.0, c.contrast.currentData())
        if key == self._tex_key and self._tex is not None:
            return self._tex
        if mode in (1, 2):
            arr = fields.edt_s() if mode == 1 else fields.t20_s()
            med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")
            if not np.isfinite(med):
                return None
            self._tex = overlay.colorize(arr, 0.5 * med, 1.5 * med, alpha=alpha)
            self._note(f"Decay camera: {'EDT' if mode == 1 else 'T20'} at {fields.fc:g} Hz. Median {med:.2f} s; warm = the room rings longer "
                       f"here, cool = shorter.")
        else:
            self._tex = overlay.colorize(fields.energy_left_db(self.anim_t), -30.0, 0.0, alpha=alpha)
            self._note(f"Decay camera: energy still to come {self.anim_t:.2f} s after the burst at {fields.fc:g} Hz (warm = still ringing, "
                       f"cool = already gone).")
        self._tex_key = key
        return self._tex

    def draw(self, out: np.ndarray, cm, tab) -> None:
        mode = self.source.currentIndex()
        fields = self._fields if mode else None
        st = self.mapping.track
        if fields is None or not st.valid or not st.xy:
            return
        e, t20 = fields.value_at(*st.xy)
        if np.isfinite(e):
            txt = f"EDT {e:.2f} s" + (f"  T20 {t20:.2f} s" if np.isfinite(t20) else "")
            overlay.draw_marker(out, cm, (st.xy[0], st.xy[1], self.state.room.listener[2]), (255, 255, 255), txt, 3)

    # ================================================================== animation
    def _source_picked(self) -> None:
        self._tex_key = None
        if self.source.currentIndex() != 0:
            self.mapping.source.setCurrentIndex(0)             # one overlay at a time: the level map steps aside
        self.tab._invalidate()

    def _toggle_anim(self) -> None:
        self.anim_playing = not self.anim_playing
        self.play_btn.setText("Pause decay animation" if self.anim_playing else "Play decay animation")
        if self.anim_playing:
            if self.source.currentIndex() != 3:
                self.source.setCurrentIndex(3)
            if self.anim_t >= T_GRID_MAX - 1e-6:
                self.anim_t = 0.0
        self._anim_wall = time.monotonic()

    def _anim_tick(self) -> None:
        now = time.monotonic()
        dt, self._anim_wall = now - self._anim_wall, now
        if not self.anim_playing:
            return
        self.anim_t += dt * float(self.speed.currentData())
        if self.anim_t > T_GRID_MAX + 0.4:                      # linger a moment on the last frame, then start again
            self.anim_t = 0.0
        self.anim_t = min(self.anim_t, T_GRID_MAX)
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(self.anim_t * 100))
        self.time_slider.blockSignals(False)

    def _slider_moved(self, v: int) -> None:
        self.anim_t = v / 100.0
        if self.source.currentIndex() != 3:
            self.source.setCurrentIndex(3)

    # ================================================================== plan view
    def _update_plan(self) -> None:
        view = self.mapping.strip.views["decay"]
        fields = self.fields() if self.tab.isVisible() else None
        view.setVisible(fields is not None and fields.n_used > 0)
        if fields is None or fields.n_used == 0:
            return
        cfg = self.state.room
        alpha = np.clip(fields.confidence() * 1.3, 0.0, 1.0)
        arr = fields.edt_s()
        med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
        if med > 0:
            view.set_room(cfg.dims.lx, cfg.dims.ly, cfg.source[:2])
            view.set_image(overlay.colorize(arr, 0.5 * med, 1.5 * med, alpha=alpha)[..., [2, 1, 0, 3]], cfg.dims.lx, cfg.dims.ly)
            view.pi.setTitle(f"EDT at {fields.fc:g} Hz (median {med:.2f} s)", size="9pt", color=theme.MUTED)

    # ================================================================== data
    def _clear(self) -> None:
        if self.mapper is not None:
            self.mapper.clear()
        self._metric_cache.clear()
        self._fields_key = None
        self._fields = None
        self._tex_key = None

    def _save(self) -> None:
        if self.mapper is None or not self.mapper.frames:
            self.status.setText("Nothing to save yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.tab, "Save decay map", "roomscape-decay.npz", "RoomScape decay map (*.npz)")
        if path:
            save_frames(self.mapper.frames, path, self.mapper.s.spec, self.mapper.dims)

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.tab, "Load decay map", "", "RoomScape decay map (*.npz)")
        if not path:
            return
        frames, spec, dims = load_frames(path)
        self.mapper = DecayWalkMapper(dims, DecaySettings(spec=spec))
        self.mapper.frames = frames
        self._metric_cache.clear()
        self._fields_key = None

    def _export_csv(self) -> None:
        fields = self.fields()
        if fields is None or fields.n_used == 0:
            self.status.setText("Nothing to export yet: map some of the room first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.tab, "Export decay map", "roomscape-decay.csv", "CSV (*.csv)")
        if path:
            n = fields.export_csv(path)
            self.status.setText(f"Exported {n} map cells ({fields.fc:g} Hz band) to {path}")

    # ================================================================== demo and housekeeping
    def _demo_walk(self) -> None:
        world = self.state.world
        if world is None or self.tab.camera is None or not self.tab.camera.is_simulated:
            self.status.setText("The demo needs the simulated input and the simulated camera (Start input, Connect).")
            return
        mp = self.mapping
        mp.tracker.set_target_bgr(MARKER_BGR)
        d = self.state.room.dims
        path = LawnmowerPath(0.16 * d.lx, 0.64 * d.lx, 0.22 * d.ly, 0.78 * d.ly, z=self.state.room.listener[2], speed=0.3,
                             row_spacing=0.55, pause_s=0.0, start_delay_s=world.time_s + 1.0)
        world.set_mic_position(path.position(0.0))
        world.walker = path
        mp.pipe.filter.reset()
        mp.pipe.trail.clear()
        self.status.setText("Demo: a virtual person starts a slow walk in 1 s. Press 'Start decay mapping' now.")

    def _source_changed(self) -> None:
        if self.state.source is None:
            self.stop()
        else:
            self._cursor = self.state.source.ring.total

    def shutdown(self) -> None:
        self.anim_timer.stop()
        self.plan_timer.stop()
        self.stop()
