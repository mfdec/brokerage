"""Walk-the-room mapping controls for the Camera tab: marker, stimulus, mapping run, measured overlays, spot sweeps."""

from __future__ import annotations

import json
import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from roomscope.audio.devices import DeviceInfo
from roomscope.audio.engine import AudioError
from roomscope.audio.measure import HardwareStimulusPlayer
from roomscope.core.fieldmap import FieldMap
from roomscope.core.mapping import MapperSettings, WalkMapper
from roomscope.core.multisine import MultisineSpec
from roomscope.core.roomfield import modal_field_db
from roomscope.sim.world import LawnmowerPath, SimStimulusPlayer
from roomscope.ui import theme
from roomscope.ui.widgets.map_strip import MapStrip
from roomscope.video import overlay
from roomscope.video.sim import MARKER_BGR
from roomscope.video.tracking import ColorMarkerTracker, MarkerPipeline

SOURCES = ["Model (modal theory)", "Measured (walk map)", "Measured minus model"]


def _spin(lo, hi, val, step, suffix="", dec=1) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setValue(val)
    return s


class MappingPanel(QtCore.QObject):
    def __init__(self, tab):
        super().__init__(tab)
        self.tab = tab
        self.state = tab.state
        self.tracker = ColorMarkerTracker()
        self.pipe = MarkerPipeline(self.tracker, self.state.room.dims, self.state.room.listener[2])
        self.mapper: WalkMapper | None = None
        self.stim = None
        self.mapping = False
        self._cursor = 0
        self._tex_key = None
        self._tex = None
        self._frames = 0
        self._last_pin_t = 0.0
        self._swept = False                 # a spot sweep ran: the mapper's audio clock must restart
        self._resume_stim = False
        self.track = self.pipe.state
        self._build()
        tab.pre_frame_hooks.append(self.on_frame)
        tab.texture_provider = self.texture
        tab.extra_draw.append(self.draw)
        tab.modelChanged.connect(lambda: self.pipe.set_camera(tab.cam_model))
        tab.pictureSizeKnown.connect(self.load_marker)
        tab.shutdown_hooks.append(self.shutdown)
        self.state.roomChanged.connect(self._room_changed)
        self.state.sourceChanged.connect(self._source_changed)
        self.strip = MapStrip()
        tab.add_bottom_widget(self.strip)
        self.plan_timer = QtCore.QTimer(self)
        self.plan_timer.setInterval(250)
        self.plan_timer.timeout.connect(self._update_plan)
        self.plan_timer.start()
        self._room_changed()

    # ================================================================== UI
    def _build(self) -> None:
        box = QtWidgets.QGroupBox("5  Walk-the-room map")
        f = QtWidgets.QFormLayout(box)
        how = QtWidgets.QLabel(
            "Put a bright, matte coloured ball or sticker on the mic (a colour nothing else in the room has). Click it "
            "in the picture, start the signal, then start mapping and walk slowly over the room holding the mic at ear "
            "height. Each second of audio paints the spot where the marker was.")
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(how)

        self.pick_btn = QtWidgets.QPushButton("Pick marker colour (click it in the picture)")
        self.pick_btn.clicked.connect(self._pick_marker)
        self.tol = _spin(4, 30, 12, 1, " hue", 0)
        self.tol.valueChanged.connect(lambda v: setattr(self.tracker, "hue_tol", int(v)))
        self.adapt = QtWidgets.QCheckBox("Follow slow colour drift (changing light)")
        self.adapt.toggled.connect(lambda on: setattr(self.tracker, "adapt", bool(on)))
        self.track_label = QtWidgets.QLabel("Marker: not picked")
        self.track_label.setWordWrap(True)
        f.addRow(self.pick_btn)
        f.addRow("Colour tolerance", self.tol)
        f.addRow(self.adapt)
        f.addRow(self.track_label)

        self.level = _spin(-45, -6, -20, 1, " dBFS", 0)
        self.fl = _spin(20, 250, 20, 5, " Hz", 0)
        self.fh = _spin(60, 600, 300, 10, " Hz", 0)
        self.tilt = QtWidgets.QComboBox()
        self.tilt.addItems(["Pink (more bass energy)", "Flat"])
        self.play_btn = QtWidgets.QPushButton("Play mapping signal")
        self.play_btn.setObjectName("primary")
        self.play_btn.clicked.connect(self._toggle_stim)
        f.addRow("Signal level", self.level)
        f.addRow("From", self.fl)
        f.addRow("To", self.fh)
        f.addRow("Tilt", self.tilt)
        f.addRow(self.play_btn)

        self.vmax = _spin(0.2, 2.0, 0.9, 0.1, " m/s", 1)
        self.snr = _spin(3, 40, 10, 1, " dB", 0)
        self.delay = _spin(0, 600, 0, 20, " ms", 0)
        self.delay.setToolTip("How late the camera picture arrives (network cameras lag). Raise it if the map looks shifted along the path.")
        self.map_btn = QtWidgets.QPushButton("Start mapping")
        self.map_btn.setObjectName("primary")
        self.map_btn.clicked.connect(self._toggle_mapping)
        row = QtWidgets.QHBoxLayout()
        clear = QtWidgets.QPushButton("Clear map")
        clear.clicked.connect(self._clear)
        save = QtWidgets.QPushButton("Save...")
        save.clicked.connect(self._save)
        load = QtWidgets.QPushButton("Load...")
        load.clicked.connect(self._load)
        for b in (clear, save, load):
            row.addWidget(b)
        f.addRow("Max walking speed", self.vmax)
        f.addRow("Min signal-to-noise", self.snr)
        f.addRow("Camera delay", self.delay)
        f.addRow(self.map_btn)
        f.addRow(row)
        self.map_status = QtWidgets.QLabel("Not mapping.")
        self.map_status.setWordWrap(True)
        f.addRow(self.map_status)

        self.source = QtWidgets.QComboBox()
        self.source.addItems(SOURCES)
        self.source.setCurrentIndex(0)
        self.source.currentIndexChanged.connect(lambda _=0: self.tab._invalidate())
        self.bw = _spin(1, 20, 2, 1, " Hz", 0)
        self.bw.setToolTip("Half-width of the frequency band averaged for the measured map (lines are 2 Hz apart).")
        f.addRow("Overlay shows", self.source)
        f.addRow("Band +-", self.bw)

        self.pin_btn = QtWidgets.QPushButton("Measure sweep here (decay at this spot)")
        self.pin_btn.clicked.connect(self._sweep_here)
        self.auto_pin = QtWidgets.QCheckBox("Do it automatically after standing still 4 s")
        f.addRow(self.pin_btn)
        f.addRow(self.auto_pin)

        self.demo_btn = QtWidgets.QPushButton("Demo: virtual walk (simulated room)")
        self.demo_btn.setToolTip("Simulated camera + input only: sets the marker colour and starts a virtual person sweeping the room.")
        self.demo_btn.clicked.connect(self._demo_walk)
        f.addRow(self.demo_btn)
        self.tab.extra_box.addWidget(box)

    # ================================================================== marker
    def _pick_marker(self) -> None:
        if self.tab.camera is None:
            self.track_label.setText("Connect a camera first.")
            return
        self.track_label.setText("Click the marker in the picture...")
        self.tab.request_pick(self._marker_clicked)

    def _marker_clicked(self, u: float, v: float) -> None:
        fr = self.tab.camera.latest() if self.tab.camera else None
        if fr is None:
            return
        img = self.tab._work_frame(fr.image)
        hsv = self.tracker.set_target_from_click(img, u, v)
        self.pipe.filter.reset()
        self.track_label.setText(f"Marker colour picked (hue {hsv[0]:.0f}, saturation {hsv[1]:.0f}, brightness {hsv[2]:.0f}).")
        self._save_marker()

    def _save_marker(self) -> None:
        if self.tracker.target is not None:
            self.state.settings.setValue(self.tab._calib_key() + "/marker", json.dumps(list(map(float, self.tracker.target))))

    def load_marker(self) -> None:
        raw = self.state.settings.value(self.tab._calib_key() + "/marker", "", str)
        if raw:
            try:
                self.tracker.target = np.array(json.loads(raw), dtype=float)
            except (ValueError, TypeError):
                pass

    # ================================================================== stimulus
    def _spec(self) -> MultisineSpec:
        src = self.state.source
        fs = src.fs if src is not None else 48000
        return MultisineSpec(fs, 0.5, self.fl.value(), max(self.fh.value(), self.fl.value() + 20), self.tilt.currentIndex() == 0, self.level.value())

    def _toggle_stim(self) -> None:
        if self.stim is not None and self.stim.playing:
            self.stim.stop()
            self.stim = None
            self.play_btn.setText("Play mapping signal")
            self.play_btn.setObjectName("primary")
            self._repolish(self.play_btn)
            return
        src = self.state.source
        if src is None or not src.running:
            self.map_status.setText("Start the microphone input first (top bar).")
            return
        try:
            if self.state.world is not None:
                self.stim = SimStimulusPlayer(self.state.world)
            else:
                dev: DeviceInfo | None
                dev, ch = self.tab.output_provider() if self.tab.output_provider else (None, "Both")
                if dev is None or dev.is_simulated:
                    self.map_status.setText("Choose the output device (your speakers) on the Measure tab first.")
                    return
                self.stim = HardwareStimulusPlayer(dev, ch)
            self.stim.start(self._spec())
        except AudioError as e:
            self.stim = None
            QtWidgets.QMessageBox.warning(self.tab, "Mapping signal", str(e))
            return
        self.play_btn.setText("Stop mapping signal")
        self.play_btn.setObjectName("danger")
        self._repolish(self.play_btn)
        self.map_status.setText("Signal playing at the level you set - raise your amplifier volume until the SNR below is good.")

    @staticmethod
    def _repolish(w: QtWidgets.QWidget) -> None:
        w.style().unpolish(w)
        w.style().polish(w)

    # ================================================================== mapping run
    def _new_mapper(self) -> None:
        cfg = self.state.room
        s = MapperSettings(spec=self._spec(), v_max=self.vmax.value(), snr_min_db=self.snr.value(),
                           camera_delay_s=self.delay.value() / 1000.0)
        old = self.mapper
        self.mapper = WalkMapper(cfg.dims, s, self.state.calibration)
        if old is not None and old.field.w.shape == self.mapper.field.w.shape and np.array_equal(old.field.freqs, self.mapper.field.freqs):
            self.mapper.field = old.field                         # keep what has been mapped so far

    def _toggle_mapping(self) -> None:
        if self.mapping:
            self.mapping = False
            self.map_btn.setText("Start mapping")
            self.map_btn.setObjectName("primary")
            self._repolish(self.map_btn)
            return
        src = self.state.source
        if src is None or not src.running:
            self.map_status.setText("Start the microphone input first (top bar).")
            return
        if self.tab.cam_model is None:
            self.map_status.setText("Calibrate the camera first (steps 1-3).")
            return
        if self.tracker.target is None:
            self.map_status.setText("Pick the marker colour first.")
            return
        self._new_mapper()
        self._cursor = src.ring.total
        self.mapper.reset_audio_clock()
        self.mapping = True
        self.map_btn.setText("Stop mapping")
        self.map_btn.setObjectName("danger")
        self._repolish(self.map_btn)

    def _clear(self) -> None:
        if self.mapper is not None:
            self.mapper.clear()
        self._tex_key = None

    def _save(self) -> None:
        if self.mapper is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.tab, "Save walk map", "roomscope-map.rsmap.npz", "RoomScope map (*.npz)")
        if path:
            self.mapper.field.save(path)

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.tab, "Load walk map", "", "RoomScope map (*.npz)")
        if not path:
            return
        fm = FieldMap.load(path)
        self._new_mapper()
        self.mapper.field = fm
        self._tex_key = None

    # ================================================================== per frame
    def on_frame(self, img: np.ndarray, fr) -> None:
        if self.tracker.target is None or fr is None:
            return
        st = self.pipe.process(img, fr.t)
        self.track = st
        self._frames += 1
        if self.mapping and self.mapper is not None:
            self.mapper.feed_position(fr.t, st.xy, st.valid)
            src = self.state.source
            if self._sweep_running():
                if src is not None:
                    self._cursor = src.ring.total            # the sweep is not multisine: keep it out of the map
                self._swept = True
            elif src is not None and src.running:
                if self._swept:
                    self.mapper.reset_audio_clock()
                    self._swept = False
                new, self._cursor, over = src.ring.read_since(self._cursor)
                if over:
                    self.mapper.reset_audio_clock()
                if new.size:
                    self.mapper.feed_audio(new, time.monotonic())
            if self._frames % 6 == 0:
                self._update_status()
            if self.auto_pin.isChecked() and not self._sweep_running() and self.mapper.stationary_for(4.0) \
                    and time.monotonic() - self._last_pin_t > 30.0:
                self._sweep_here()
        elif self._frames % 10 == 0:
            self._update_track_label()

    def _update_track_label(self) -> None:
        st = self.track
        if st.valid and st.xy:
            self.track_label.setText(f"Tracking: x {st.xy[0]:.2f} m, y {st.xy[1]:.2f} m, {st.speed:.2f} m/s")
        elif self.tracker.target is not None:
            self.track_label.setText("Marker not found - is it in view, and is the colour unique in the room?")

    def _update_status(self) -> None:
        m = self.mapper
        st = self.track
        cov = m.field.coverage() * 100
        hint, color = "", theme.MUTED
        if self._sweep_running():
            hint, color = "Measuring the decay at this spot - stand still and keep quiet.", theme.ACCENT
        elif not st.valid:
            hint, color = "Marker lost - keep it visible to the camera.", theme.AMBER
        elif st.speed > self.vmax.value():
            hint, color = "Too fast - slow down.", theme.AMBER
        elif m.last is not None and m.last.clipped:
            hint, color = "Input is clipping - lower the signal level or volume.", theme.BAD
        elif m.last is not None and np.isfinite(m.last.snr_db) and m.last.snr_db < self.snr.value():
            hint, color = "Signal too weak - raise the volume (or reduce noise).", theme.AMBER
        snr = f"{m.last.snr_db:.0f} dB" if m.last is not None and np.isfinite(m.last.snr_db) else "-"
        c = m.counts
        n = sum(c.values())
        self.map_status.setText(
            f"<span style='color:{color}'>{hint}</span><br>Mapping: {c['accepted']}/{n} frames used · S/N {snr} · "
            f"{st.speed:.2f} m/s · coverage {cov:.0f} %")
        if not hint:
            self._update_track_label()

    # ================================================================== overlay
    def _model_rel(self, nx: int, ny: int) -> np.ndarray:
        cfg = self.state.room
        _, _, db = modal_field_db(cfg, self.tab.freq.value(), self.tab.plane.value(), n=(nx, ny))
        return db

    def _measured_rel(self) -> np.ndarray | None:
        if self.mapper is None:
            return None
        f, bw = self.tab.freq.value(), self.bw.value()
        return self.mapper.field.relative_db(f - bw, f + bw)

    def texture(self):
        mode = self.source.currentIndex()
        if mode == 0 or self.mapper is None:
            return None
        key = (mode, round(self.tab.freq.value(), 2), self.bw.value(), self.tab.contrast.currentData(),
               self.mapper.field.stats.n_added, round(self.tab.plane.value(), 3))
        if key == self._tex_key and self._tex is not None:
            return self._tex
        c = float(self.tab.contrast.currentData())
        fm = self.mapper.field
        rel = self._measured_rel()
        alpha = np.clip(fm.confidence() * 1.3, 0.0, 1.0)
        if mode == 2 and rel is not None and np.isfinite(rel).any():
            model = self._model_rel(fm.nx, fm.ny)
            ok = np.isfinite(rel)
            rel = rel - model - np.nanmedian((rel - model)[ok])
        self._tex = overlay.colorize(rel if rel is not None else np.full((fm.ny, fm.nx), np.nan), -c, c, alpha=alpha)
        self._tex_key = key
        return self._tex

    def draw(self, out: np.ndarray, cm, tab) -> None:
        st = self.track
        if st.px is not None:
            import cv2  # noqa: PLC0415

            cv2.circle(out, (int(st.px[0]), int(st.px[1])), 14, (0, 255, 0), 2, cv2.LINE_AA)
        if st.valid and st.xy:
            overlay.draw_marker(out, cm, (st.xy[0], st.xy[1], self.state.room.listener[2]), (0, 255, 0), "mic", 6)
        if len(self.pipe.trail) > 1:
            overlay.draw_trail(out, cm, np.array([(x, y) for _t, x, y in self.pipe.trail]), self.state.room.listener[2], (255, 255, 255))
        for m in self.state.measurements:
            if m.position is not None and np.isfinite(m.decay.broadband.t60):
                overlay.draw_marker(out, cm, (m.position[0], m.position[1], 0.0), (255, 200, 60), f"RT60 {m.decay.broadband.t60:.2f}s", 7)

    def _update_plan(self) -> None:
        if not self.tab.isVisible():
            return
        cfg = self.state.room
        lx, ly = cfg.dims.lx, cfg.dims.ly
        views = self.strip.views
        for v in views.values():
            v.set_room(lx, ly, cfg.source[:2])
        c = float(self.tab.contrast.currentData())
        trail = np.array([(x, y) for _t, x, y in self.pipe.trail]) if len(self.pipe.trail) > 1 else None
        mic = self.track.xy if self.track.valid else None
        if self.mapper is not None:
            fm = self.mapper.field
            rel = self._measured_rel()
            alpha = np.clip(fm.confidence() * 1.3, 0.0, 1.0)
            model = self._model_rel(fm.nx, fm.ny)
            views["measured"].set_image(overlay.colorize(rel, -c, c, alpha=alpha)[..., [2, 1, 0, 3]], lx, ly)
            ok = np.isfinite(rel)
            diff = rel - model
            diff = diff - (np.nanmedian(diff[ok]) if ok.any() else 0.0)
            views["diff"].set_image(overlay.colorize(diff, -c, c, alpha=alpha)[..., [2, 1, 0, 3]], lx, ly)
        else:
            model = self._model_rel(48, 38)
            for k in ("measured", "diff"):
                views[k].set_image(None, lx, ly)
        views["model"].set_image(overlay.colorize(model, -c, c)[..., [2, 1, 0, 3]], lx, ly)
        for v in views.values():
            v.set_path(trail, mic)

    # ================================================================== sweeps at a position
    def _sweep_running(self) -> bool:
        mt = getattr(self.tab, "measure_tab", None)
        return mt is not None and mt.worker is not None and mt.worker.isRunning()

    def _sweep_here(self) -> None:
        pos = self.track.xy if self.track.valid else None
        if pos is None:
            self.map_status.setText("Need a tracked marker to know where to put the sweep.")
            return
        mt = getattr(self.tab, "measure_tab", None)
        if mt is None or self._sweep_running():
            return
        self._last_pin_t = time.monotonic()
        src = self.state.source
        if self.mapping and src is not None:
            self._cursor, self._swept = src.ring.total, True          # nothing from here on belongs in the map
        was_playing = self.stim is not None and self.stim.playing
        if was_playing:
            self._toggle_stim()
        if mt.start_with_position((float(pos[0]), float(pos[1]))):
            self._resume_stim = was_playing
            mt.worker.finished.connect(self._sweep_finished)
        elif was_playing:
            self._toggle_stim()

    def _sweep_finished(self) -> None:
        if self._resume_stim and self.stim is None and self.state.running:
            self._toggle_stim()
        self._resume_stim = False

    # ================================================================== demo, housekeeping
    def _demo_walk(self) -> None:
        world = self.state.world
        if world is None or self.tab.camera is None or not self.tab.camera.is_simulated:
            self.map_status.setText("The demo needs the simulated input and the simulated camera (steps: Start input, Connect).")
            return
        self.tracker.set_target_bgr(MARKER_BGR)
        d = self.state.room.dims
        path = LawnmowerPath(0.16 * d.lx, 0.64 * d.lx, 0.22 * d.ly, 0.78 * d.ly, z=self.state.room.listener[2], speed=0.5,
                             row_spacing=0.55, pause_s=0.3, start_delay_s=world.time_s + 1.0)
        world.set_mic_position(path.position(0.0))                 # the virtual person is already standing at the start
        world.walker = path
        self.pipe.filter.reset()
        self.pipe.trail.clear()
        self.track_label.setText("Demo marker colour set; a virtual person starts walking in 1 s.")

    def _room_changed(self) -> None:
        self.pipe.dims = self.state.room.dims
        self.pipe.z = self.state.room.listener[2]
        if self.mapper is not None and (self.mapper.dims != self.state.room.dims):
            self.mapper = None
        self._tex_key = None

    def _source_changed(self) -> None:
        src = self.state.source
        if src is None:
            self.mapping = False
            self.map_btn.setText("Start mapping")
            self.map_btn.setObjectName("primary")
            self._repolish(self.map_btn)
            if self.stim is not None:
                self.stim.stop()
                self.stim = None
                self.play_btn.setText("Play mapping signal")
        else:
            self._cursor = src.ring.total

    def shutdown(self) -> None:
        self.plan_timer.stop()
        if self.stim is not None:
            self.stim.stop()
