"""Camera tab: webcam / DroidCam / IP camera, floor-point calibration, and the sound-field overlay."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from roomscape.core.geometry import (
    GeometryError, Intrinsics, RoomCamera, estimate_focal, fit_camera, fov_from_camera_height, fov_from_raised_points,
)
from roomscape.core.roomfield import modal_field_db
from roomscape.ui import theme
from roomscape.ui.state import AppState
from roomscape.ui.widgets.camera_view import CameraView
from roomscape.video import overlay
from roomscape.video.camera import (
    RESOLUTIONS, URL_PRESETS, CameraError, CameraSource, CameraSpec, OpenCvCamera, camera_names, probe_cameras,
)
from roomscape.video.sim import SimCamera, SimScene, default_camera

WORK_MAX_WIDTH = 1280
CONTRASTS = [6, 9, 12, 18, 24]


@dataclass
class Pin:
    x: float
    y: float
    px: tuple[float, float] | None = None


class _Worker(QtCore.QThread):
    ok = QtCore.Signal(object)
    fail = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self) -> None:
        try:
            self.ok.emit(self.fn())
        except CameraError as e:
            self.fail.emit(str(e))
        except Exception as e:  # pragma: no cover
            self.fail.emit(f"{type(e).__name__}: {e}")


def _spin(lo, hi, val, step=0.1, suffix="", dec=2) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setValue(val)
    return s


class CameraTab(QtWidgets.QWidget):
    modelChanged = QtCore.Signal()                      # camera pose / calibration changed (or lost)
    pictureSizeKnown = QtCore.Signal()                  # first frame of a connection arrived (saved settings can be restored)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.camera: CameraSource | None = None
        self.pins: list[Pin] = []
        self.cam_model: RoomCamera | None = None
        self.rms = float("nan")
        self.work_size = (0, 0)
        self._last_index = -1
        self._tex: np.ndarray | None = None
        self._tex_key = None
        self._mask_cache: dict = {}
        self._mode = "idle"                     # idle | pin | raised | external
        self._external_pick = None
        self._raised: list[tuple[np.ndarray, np.ndarray]] = []
        self._worker: _Worker | None = None
        self.last_frame_ms = 0.0
        self.extra_draw = []                    # callables(frame, cam_model, tab) used by the mapping extension
        self.pre_frame_hooks = []               # callables(work_frame, Frame) run before drawing (tracking, mapping)
        self.texture_provider = None            # callable() -> BGRA texture or None (None = modal model)
        self.overlay_providers = []             # more callables(); the first to return a texture wins, then the modal model
        self.overlay_busy = None                # callable() -> True while one of those providers owns the overlay note text
        self.output_provider = None             # callable() -> (DeviceInfo | None, channels)
        self.measure_tab = None                 # set by the main window: spot sweeps at the marker position
        self.shutdown_hooks = []
        self._loading = False

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(372)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(self._build_controls())
        root.addWidget(scroll, 0)
        self.view = CameraView()
        self.view.picked.connect(self._on_picked)
        self.view.pointMoved.connect(self._on_point_moved)
        self.view.pointReleased.connect(lambda _i: self._refit())
        self.right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right.addWidget(self.view)
        self.right.setStretchFactor(0, 1)
        root.addWidget(self.right, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        state.roomChanged.connect(self._room_changed)
        self._fill_pins("inset")
        self._refresh_sources()

    # ================================================================== controls
    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 4, 0)

        cam = QtWidgets.QGroupBox("1  Camera")
        f = QtWidgets.QFormLayout(cam)
        self.source = QtWidgets.QComboBox()
        self.find_btn = QtWidgets.QPushButton("Find cameras")
        self.find_btn.setToolTip("Scan for webcams and the DroidCam client's virtual webcam (takes a few seconds).")
        self.find_btn.clicked.connect(self._find_cameras)
        self.url = QtWidgets.QLineEdit()
        self.url.setPlaceholderText("http://PHONE-IP:4747/video")
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("URL presets...")
        self.preset.addItems(list(URL_PRESETS))
        self.preset.currentTextChanged.connect(lambda t: self.url.setText(URL_PRESETS[t]) if t in URL_PRESETS else None)
        self.res = QtWidgets.QComboBox()
        for rw, rh in RESOLUTIONS:
            self.res.addItem(f"{rw} x {rh}", (rw, rh))
        self.res.setCurrentIndex(1)
        self.flip = QtWidgets.QCheckBox("Mirror image (undo selfie flip)")
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setObjectName("primary")
        self.connect_btn.clicked.connect(self._toggle_connect)
        self.cam_status = QtWidgets.QLabel("Not connected")
        self.cam_status.setWordWrap(True)
        self.cam_status.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow("Source", self.source)
        f.addRow(self.find_btn)
        f.addRow("Stream URL", self.url)
        f.addRow(self.preset)
        f.addRow("Resolution", self.res)
        f.addRow(self.flip)
        f.addRow(self.connect_btn)
        f.addRow(self.cam_status)
        v.addWidget(cam)

        cal = QtWidgets.QGroupBox("2  Pin known floor points")
        c = QtWidgets.QVBoxLayout(cal)
        how = QtWidgets.QLabel(
            "Mark at least 4 spots on the floor whose position you know (tape-measure from the front-left corner), "
            "spread out - not in a row. Then click each one in the picture. Room corners work if you can see them.")
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.MUTED};")
        c.addWidget(how)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "x (m)", "y (m)", "clicked"])
        self.table.verticalHeader().setVisible(False)
        self.table.setFixedHeight(168)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._table_edited)
        c.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Corners", lambda: self._fill_pins("corners")), ("Inset", lambda: self._fill_pins("inset")),
                         ("Front half", lambda: self._fill_pins("front")), ("+", self._add_pin), ("-", self._del_pin)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        c.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.pick_btn = QtWidgets.QPushButton("Click points in picture")
        self.pick_btn.setObjectName("primary")
        self.pick_btn.clicked.connect(self._start_pick)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_clicks)
        row.addWidget(self.pick_btn, 2)
        row.addWidget(self.clear_btn, 1)
        c.addLayout(row)
        self.demo_btn = QtWidgets.QPushButton("Demo: place points automatically")
        self.demo_btn.setToolTip("Simulated camera only: drops the pins exactly where they really are.")
        self.demo_btn.clicked.connect(self._demo_pins)
        c.addWidget(self.demo_btn)
        self.pose_status = QtWidgets.QLabel("Camera pose: not calibrated")
        self.pose_status.setWordWrap(True)
        c.addWidget(self.pose_status)
        v.addWidget(cal)

        fov = QtWidgets.QGroupBox("3  Lens field of view")
        g = QtWidgets.QGridLayout(fov)
        self.fov_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fov_slider.setRange(300, 1300)
        self.fov_slider.setValue(650)
        self.fov = _spin(30, 130, 65.0, 0.5, " deg", 1)
        g.addWidget(QtWidgets.QLabel("Horizontal FOV"), 0, 0)
        g.addWidget(self.fov, 0, 1)
        g.addWidget(self.fov_slider, 1, 0, 1, 2)
        hint = QtWidgets.QLabel("A wrong FOV is invisible on the floor but shifts the mic marker's mapped position. "
                                "Use any one of these to pin it down; the room outline should hug the real walls and ceiling.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        g.addWidget(hint, 2, 0, 1, 2)
        self.auto_btn = QtWidgets.QPushButton("Estimate from perspective")
        self.auto_btn.clicked.connect(self._auto_fov)
        g.addWidget(self.auto_btn, 3, 0, 1, 2)
        self.cam_h = _spin(0.3, 6.0, 2.0, 0.05, " m")
        self.cam_h_btn = QtWidgets.QPushButton("From camera height")
        self.cam_h_btn.clicked.connect(self._fov_from_height)
        g.addWidget(QtWidgets.QLabel("Camera height"), 4, 0)
        g.addWidget(self.cam_h, 4, 1)
        g.addWidget(self.cam_h_btn, 5, 0, 1, 2)
        self.raised_pin = QtWidgets.QComboBox()
        self.raised_btn = QtWidgets.QPushButton("Marker above pin: click it")
        self.raised_btn.setToolTip("Hold the mic marker at mic height directly above the chosen floor point, then click the marker.")
        self.raised_btn.clicked.connect(self._start_raised)
        g.addWidget(QtWidgets.QLabel("Marker over pin"), 6, 0)
        g.addWidget(self.raised_pin, 6, 1)
        g.addWidget(self.raised_btn, 7, 0, 1, 2)
        self.fov_status = QtWidgets.QLabel("")
        self.fov_status.setWordWrap(True)
        g.addWidget(self.fov_status, 8, 0, 1, 2)
        v.addWidget(fov)

        ov = QtWidgets.QGroupBox("4  Sound-field overlay")
        o = QtWidgets.QFormLayout(ov)
        self.show_map = QtWidgets.QCheckBox("Show map")
        self.show_map.setChecked(True)
        self.show_wire = QtWidgets.QCheckBox("Room outline")
        self.show_wire.setChecked(True)
        self.show_icons = QtWidgets.QCheckBox("Speaker / listener")
        self.show_icons.setChecked(True)
        o.addRow(self.show_map)
        o.addRow(self.show_wire)
        o.addRow(self.show_icons)
        self.freq = _spin(1, 600, 34.3, 0.5, " Hz", 1)
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.freq_slider.setRange(2, 1200)
        self.freq_slider.setValue(69)
        self.snap_btn = QtWidgets.QPushButton("Snap to nearest mode")
        self.snap_btn.clicked.connect(self._snap)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(62)
        self.contrast = QtWidgets.QComboBox()
        for db in CONTRASTS:
            self.contrast.addItem(f"+-{db} dB", db)
        self.contrast.setCurrentIndex(2)
        self.plane = _spin(0, 5, 1.2, 0.05, " m")
        self.plane.setToolTip("Height of the map above the floor; normally ear / microphone height.")
        o.addRow("Frequency", self.freq)
        o.addRow(self.freq_slider)
        o.addRow(self.snap_btn)
        o.addRow("Opacity", self.opacity)
        o.addRow("Contrast", self.contrast)
        o.addRow("Map height", self.plane)
        self.overlay_note = QtWidgets.QLabel("Warm = louder than the room average at this frequency, cool = quieter (standing-wave pattern).")
        self.overlay_note.setWordWrap(True)
        self.overlay_note.setStyleSheet(f"color: {theme.MUTED};")
        o.addRow(self.overlay_note)
        v.addWidget(ov)
        self.extra_box = QtWidgets.QVBoxLayout()          # the walk-the-room mapping controls plug in here
        v.addLayout(self.extra_box)
        v.addStretch(1)

        self.fov.valueChanged.connect(self._fov_changed)
        self.fov_slider.valueChanged.connect(lambda x: self.fov.setValue(x / 10.0))
        self.freq.valueChanged.connect(lambda x: self._freq_changed(x, self.freq))
        self.freq_slider.valueChanged.connect(lambda x: self._freq_changed(x / 2.0, self.freq_slider))
        self.plane.valueChanged.connect(lambda _=0: self._invalidate())
        self.contrast.currentIndexChanged.connect(lambda _=0: self._invalidate())
        return w

    # ================================================================== sources
    def _refresh_sources(self, found: list[int] | None = None, names: list[str] | None = None) -> None:
        keep = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItem("Simulated camera (demo)", ("sim", None))
        for i in found or []:
            label = names[i] if names and i < len(names) else f"Camera {i}"
            self.source.addItem(f"{label}  [index {i}]", ("dev", i))
        self.source.addItem("Network stream (URL below)", ("url", None))
        if keep is not None:
            i = self.source.findData(keep)
            if i >= 0:
                self.source.setCurrentIndex(i)
        self.source.blockSignals(False)

    def _find_cameras(self) -> None:
        self.find_btn.setEnabled(False)
        self.cam_status.setText("Scanning for cameras...")
        self._probe = _Worker(lambda: (probe_cameras(6), camera_names()), self)
        self._probe.ok.connect(self._found)
        self._probe.fail.connect(lambda m: (self.cam_status.setText(m), self.find_btn.setEnabled(True)))
        self._probe.start()

    def _found(self, result) -> None:
        found, names = result
        self._refresh_sources(found, names)
        self.find_btn.setEnabled(True)
        self.cam_status.setText(f"Found {len(found)} camera(s)." if found else
                                "No local camera found. If you use DroidCam, start its Windows client first, or pick 'Network stream'.")

    def _toggle_connect(self) -> None:
        if self.camera is not None:
            self.disconnect_camera()
            return
        kind, idx = self.source.currentData()
        if kind == "sim":
            self._connect_sim()
            return
        w, h = self.res.currentData()
        spec = CameraSpec.device(idx, w, h) if kind == "dev" else CameraSpec.stream(self.url.text())
        self.connect_btn.setEnabled(False)
        self.cam_status.setText(f"Connecting to {spec.label} ...")
        cam = OpenCvCamera(spec)
        self._worker = _Worker(lambda: (cam.start(), cam)[1], self)
        self._worker.ok.connect(self._on_connected)
        self._worker.fail.connect(self._on_connect_failed)
        self._worker.start()

    def _connect_sim(self) -> None:
        room = self.state.room
        world = self.state.world
        pos = (lambda: (*world.mic[:2], world.mic[2])) if world is not None else (lambda: tuple(room.listener))
        scene = SimScene(room.dims, default_camera(960, 540), (960, 540), speaker=room.source)
        cam = SimCamera(scene, pos)
        cam.start()
        self._fill_pins("front")                     # a rectangle this simulated camera can actually see
        self._on_connected(cam)

    def _on_connected(self, cam: CameraSource) -> None:
        self.camera = cam
        self._last_index = -1
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setObjectName("danger")
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)
        self.connect_btn.setEnabled(True)
        self.source.setEnabled(False)
        self.cam_status.setText(f"Connected: {cam.name}")
        self.demo_btn.setEnabled(cam.is_simulated)

    def _on_connect_failed(self, msg: str) -> None:
        self.connect_btn.setEnabled(True)
        self.cam_status.setText(msg)
        QtWidgets.QMessageBox.warning(self, "Camera", msg)

    def disconnect_camera(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.view.clear()
        self.view.hint = "No camera connected"
        self.connect_btn.setText("Connect")
        self.connect_btn.setObjectName("primary")
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)
        self.source.setEnabled(True)
        self.cam_status.setText("Not connected")
        self.cam_model = None
        self.work_size = (0, 0)

    def shutdown(self) -> None:
        self.timer.stop()
        for fn in self.shutdown_hooks:
            fn()
        self.disconnect_camera()

    # ================================================================== pins
    def _fill_pins(self, kind: str) -> None:
        d = self.state.room.dims
        if kind == "corners":
            pts = [(0.0, 0.0), (d.lx, 0.0), (d.lx, d.ly), (0.0, d.ly)]
        elif kind == "front":                       # a rectangle in the far half: visible from a camera at the rear wall
            pts = [(0.16 * d.lx, 0.22 * d.ly), (0.64 * d.lx, 0.22 * d.ly), (0.64 * d.lx, 0.78 * d.ly), (0.16 * d.lx, 0.78 * d.ly)]
        else:
            m = min(0.6, 0.15 * min(d.lx, d.ly))
            pts = [(m, m), (d.lx - m, m), (d.lx - m, d.ly - m), (m, d.ly - m)]
        self.pins = [Pin(round(x, 2), round(y, 2)) for x, y in pts]
        self._pins_changed()

    def _add_pin(self) -> None:
        d = self.state.room.dims
        self.pins.append(Pin(round(d.lx / 2, 2), round(d.ly / 2, 2)))
        self._pins_changed()

    def _del_pin(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        i = rows[0].row() if rows else len(self.pins) - 1
        if len(self.pins) > 4 and 0 <= i < len(self.pins):
            del self.pins[i]
            self._pins_changed()

    def _clear_clicks(self) -> None:
        for p in self.pins:
            p.px = None
        self._pins_changed()

    def _pins_changed(self) -> None:
        self._loading = True
        self.table.setRowCount(len(self.pins))
        for r, p in enumerate(self.pins):
            cells = [str(r + 1), f"{p.x:.2f}", f"{p.y:.2f}", "yes" if p.px else "-"]
            for c, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                if c in (0, 3):
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if c == 3 and p.px:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(theme.GOOD)))
                self.table.setItem(r, c, it)
        self.raised_pin.clear()
        for r in range(len(self.pins)):
            self.raised_pin.addItem(f"pin {r + 1}", r)
        self._loading = False
        self.view.points = [p.px for p in self.pins]
        self.view.update()
        self._refit()

    def _table_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._loading or item.column() not in (1, 2):
            return
        try:
            val = float(item.text().replace(",", "."))
        except ValueError:
            return
        p = self.pins[item.row()]
        if item.column() == 1:
            p.x = val
        else:
            p.y = val
        self._refit()

    def _start_pick(self) -> None:
        if self.camera is None:
            self.cam_status.setText("Connect a camera first.")
            return
        self._mode = "pin"
        self.view.pick_mode = True
        self.pick_btn.setText("Picking... (click picture)")

    def _stop_pick(self) -> None:
        self._mode = "idle"
        self.view.pick_mode = False
        self.pick_btn.setText("Click points in picture")

    def _next_unplaced(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        if rows and self.pins[rows[0].row()].px is None:
            return rows[0].row()
        return next((i for i, p in enumerate(self.pins) if p.px is None), None)

    def _on_picked(self, u: float, v: float) -> None:
        if self._mode == "external":
            cb, self._external_pick = self._external_pick, None
            self._stop_pick()
            if cb is not None:
                cb(u, v)
        elif self._mode == "pin":
            i = self._next_unplaced()
            if i is None:
                self._stop_pick()
                return
            self.pins[i].px = (u, v)
            self._pins_changed()
            if self._next_unplaced() is None:
                self._stop_pick()
                self.table.selectRow(i)
            else:
                self.table.selectRow(self._next_unplaced())
        elif self._mode == "raised":
            self._stop_pick()
            idx = int(self.raised_pin.currentData() or 0)
            p = self.pins[idx]
            self._raised.append((np.array([u, v]), np.array([p.x, p.y, self.state.room.listener[2]])))
            self._fov_from_raised()

    def _on_point_moved(self, i: int, u: float, v: float) -> None:
        if 0 <= i < len(self.pins):
            self.pins[i].px = (u, v)
            self.view.points = [p.px for p in self.pins]
            self.view.update()

    # ================================================================== pose / FOV
    def _intrinsics(self) -> Intrinsics | None:
        w, h = self.work_size
        return Intrinsics(w, h, self.fov.value()) if w else None

    def _placed(self):
        placed = [p for p in self.pins if p.px is not None]
        return (np.array([p.px for p in placed], float), np.array([[p.x, p.y] for p in placed], float)) if placed else (None, None)

    def _refit(self) -> None:
        self.cam_model = None
        intr = self._intrinsics()
        img, wxy = self._placed()
        n = 0 if img is None else len(img)
        if intr is None or n < 4:
            self.pose_status.setText(f"Camera pose: {n} of {max(len(self.pins), 4)} points clicked (need 4+).")
            self._invalidate()
            self.modelChanged.emit()
            return
        try:
            cam, rms = fit_camera(img, wxy, intr)
        except GeometryError as e:
            self.pose_status.setText(f"<span style='color:{theme.BAD}'>{e}</span>")
            self._invalidate()
            self.modelChanged.emit()
            return
        self.cam_model, self.rms = cam, rms
        c = cam.center
        warn = "" if rms < 6 else f"  <span style='color:{theme.AMBER}'>High error: re-check which point is which.</span>"
        self.pose_status.setText(f"<span style='color:{theme.GOOD}'>Calibrated.</span> Fit error {rms:.1f} px. "
                                 f"Camera at x {c[0]:.1f}, y {c[1]:.1f}, height {c[2]:.1f} m.{warn}")
        self.cam_h.blockSignals(True)
        self.cam_h.setValue(float(np.clip(c[2], 0.3, 6.0)))
        self.cam_h.blockSignals(False)
        self._invalidate()
        self._save_calibration()
        self.modelChanged.emit()

    def _fov_changed(self, v: float) -> None:
        self.fov_slider.blockSignals(True)
        self.fov_slider.setValue(int(round(v * 10)))
        self.fov_slider.blockSignals(False)
        self._refit()

    def _set_fov(self, v: float) -> None:
        self.fov.setValue(float(np.clip(v, 30, 130)))

    def _auto_fov(self) -> None:
        img, wxy = self._placed()
        w, h = self.work_size
        f = estimate_focal(img, wxy, w, h) if img is not None and len(img) >= 4 and w else None
        if f is None:
            self.fov_status.setText("Not enough perspective in this view to estimate (camera too head-on, or fewer than 4 points). "
                                    "Use the camera height or the marker method instead.")
            return
        fov = Intrinsics.fov_from_focal(w, f)
        self._set_fov(fov)
        self.fov_status.setText(f"Estimated from perspective: {fov:.0f} deg. Confirm with the room outline or the marker method.")

    def _fov_from_height(self) -> None:
        img, wxy = self._placed()
        w, h = self.work_size
        est = fov_from_camera_height(img, wxy, w, h, self.cam_h.value()) if img is not None and len(img) >= 4 and w else None
        if est is None:
            self.fov_status.setText("No lens FOV reproduces that camera height; re-check the height and the pinned points.")
            return
        self._set_fov(est)
        self.fov_status.setText(f"FOV {est:.0f} deg gives a camera height of {self.cam_h.value():.2f} m.")

    def _start_raised(self) -> None:
        if self.camera is None or self.cam_model is None:
            self.fov_status.setText("Connect a camera and calibrate the floor points first.")
            return
        self._mode = "raised"
        self.view.pick_mode = True
        self.fov_status.setText("Click the marker in the picture (held at mic height right above the chosen pin).")

    def _fov_from_raised(self) -> None:
        img, wxy = self._placed()
        w, h = self.work_size
        res = fov_from_raised_points(img, wxy, w, h, self._raised)
        if res is None:
            self.fov_status.setText("Could not fit a FOV to that marker position.")
            return
        fov, err = res
        self._set_fov(fov)
        self.fov_status.setText(f"FOV {fov:.0f} deg from {len(self._raised)} marker sample(s); mapping error {err * 100:.0f} cm. "
                                "Repeat over another pin for a tighter result.")

    def _demo_pins(self) -> None:
        cam = getattr(getattr(self.camera, "scene", None), "cam", None)
        if cam is None:
            return
        for p in self.pins:
            uv, _ = cam.project(np.array([[p.x, p.y, 0.0]]))
            p.px = (float(uv[0, 0]), float(uv[0, 1]))
        self._pins_changed()

    # ================================================================== overlay
    def _room_changed(self) -> None:
        self.plane.blockSignals(True)
        self.plane.setValue(self.state.room.listener[2])
        self.plane.blockSignals(False)
        self._invalidate()

    def _freq_changed(self, f: float, source: QtWidgets.QWidget) -> None:
        for w, val in ((self.freq, f), (self.freq_slider, int(round(f * 2)))):
            if w is not source:
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)
        self._invalidate()

    def _snap(self) -> None:
        modes = self.state.room.modes()
        if modes:
            self._freq_changed(min(modes, key=lambda m: abs(m.freq - self.freq.value())).freq, QtWidgets.QWidget())

    def _invalidate(self) -> None:
        self._tex_key = None

    def _texture(self) -> np.ndarray:
        if self.texture_provider is not None:
            tex = self.texture_provider()
            if tex is not None:
                return tex
        for provider in self.overlay_providers:
            tex = provider()
            if tex is not None:
                return tex
        cfg = self.state.room
        c = float(self.contrast.currentData())
        key = (round(self.freq.value(), 2), round(self.plane.value(), 3), c, cfg)
        if key != self._tex_key or self._tex is None:
            _, _, db = modal_field_db(cfg, self.freq.value(), self.plane.value(), n=(90, 72))
            self._tex = overlay.colorize(db, -c, c)
            self._tex_key = key
        return self._tex

    # ================================================================== frame loop
    def _work_frame(self, img: np.ndarray) -> np.ndarray:
        if self.flip.isChecked():
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        if w > WORK_MAX_WIDTH:
            img = cv2.resize(img, (WORK_MAX_WIDTH, int(round(h * WORK_MAX_WIDTH / w))), interpolation=cv2.INTER_AREA)
        return img

    def _tick(self) -> None:
        cam = self.camera
        if cam is None or not self.isVisible():
            return
        fr = cam.latest()
        if fr is None or fr.index == self._last_index:
            if cam is not None and getattr(cam, "error", None) and fr is None:
                self.cam_status.setText(str(cam.error))
            return
        t0 = time.perf_counter()
        self._last_index = fr.index
        img = self._work_frame(fr.image)
        size = (img.shape[1], img.shape[0])
        if size != self.work_size:
            self._on_size_change(size)
        self.process_frame(img, fr)
        self.last_frame_ms = 0.9 * self.last_frame_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
        if getattr(cam, "fps", 0) and fr.index % 15 == 0:
            self.cam_status.setText(f"Connected: {cam.name}  {size[0]}x{size[1]}  {cam.fps:.0f} fps  ({self.last_frame_ms:.0f} ms/frame)")

    def process_frame(self, img: np.ndarray, fr=None) -> np.ndarray:
        """Draw overlays on a working-size frame and show it (also used directly by tests / the mapping layer)."""
        out = img
        for hook in self.pre_frame_hooks:
            hook(img, fr)
        cm = self.cam_model
        if cm is not None:
            cfg = self.state.room
            if self.show_map.isChecked():
                out = overlay.draw_map(out, cm, cfg.dims, self._texture(), self.plane.value(),
                                       self.opacity.value() / 100.0, self._mask_cache)
            if out is img:
                out = img.copy()
            if self.show_wire.isChecked():
                overlay.draw_wireframe(out, cm, cfg.dims, (210, 210, 210))
            if self.show_icons.isChecked():
                overlay.draw_marker(out, cm, cfg.source, (60, 180, 255), "speaker")
                overlay.draw_marker(out, cm, cfg.listener, (120, 220, 120), "listener")
            for fn in self.extra_draw:
                fn(out, cm, self)
        self.view.set_frame(out)
        return out

    def add_bottom_widget(self, w: QtWidgets.QWidget) -> None:
        self.right.addWidget(w)
        self.right.setStretchFactor(1, 0)
        self.right.setSizes([620, 230])

    def request_pick(self, callback) -> None:
        """Let another component grab the next click in the picture (image coordinates)."""
        self._external_pick = callback
        self._mode = "external"
        self.view.pick_mode = True

    def _on_size_change(self, size: tuple[int, int]) -> None:
        old = self.work_size
        self.work_size = size
        if old[0]:                                          # rescale existing clicks to the new picture size
            sx, sy = size[0] / old[0], size[1] / old[1]
            for p in self.pins:
                if p.px:
                    p.px = (p.px[0] * sx, p.px[1] * sy)
        else:
            self._load_calibration()
            self.pictureSizeKnown.emit()
        self.view.points = [p.px for p in self.pins]
        self._pins_changed()

    # ================================================================== persistence
    def _calib_key(self) -> str:
        w, h = self.work_size
        name = (self.camera.name if self.camera else "none").replace("/", "_").replace("\\", "_")
        return f"camera/{name}/{w}x{h}"

    def _save_calibration(self) -> None:
        if self.camera is None or self._loading or not self.work_size[0]:
            return
        data = {"fov": self.fov.value(), "pins": [[p.x, p.y, *(p.px or (None, None))] for p in self.pins]}
        self.state.settings.setValue(self._calib_key(), json.dumps(data))

    def _load_calibration(self) -> None:
        raw = self.state.settings.value(self._calib_key(), "", str)
        if not raw:
            return
        try:
            d = json.loads(raw)
            self.pins = [Pin(float(x), float(y), (float(u), float(v)) if u is not None else None) for x, y, u, v in d["pins"]]
            self.fov.blockSignals(True)
            self.fov.setValue(float(d["fov"]))
            self.fov.blockSignals(False)
            self.fov_slider.setValue(int(round(float(d["fov"]) * 10)))
            self.cam_status.setText(self.cam_status.text() + "  (restored your saved calibration)")
        except (ValueError, KeyError, TypeError):
            pass
