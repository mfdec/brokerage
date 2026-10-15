"""Room-modes tab: dimensions + positions -> predicted modes, their excitation, modal response and pressure field."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from roomscope.core.modes import bonello_check, coincident_modes, largest_gaps
from roomscope.core.roomfield import (
    PeakMatch, RoomConfig, excitation_db, find_response_peaks, match_peaks, predicted_response_db,
)
from roomscope.ui import theme
from roomscope.ui.state import AppState
from roomscope.ui.widgets.mode_plots import KIND_COLORS, ModeFieldView, ModeMapPlot


def _spin(lo: float, hi: float, val: float, step: float = 0.05, suffix: str = " m", dec: int = 2) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setValue(val)
    return s


class ModesTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.modes = []
        self.matches: list[PeakMatch] = []
        self._loading = False

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self._build_controls(), 0)
        root.addWidget(self._build_views(), 1)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self.recompute)
        state.roomChanged.connect(self._load_from_state)
        state.measurementsChanged.connect(lambda: self._debounce.start())
        self._load_from_state()

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(292)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        room = QtWidgets.QGroupBox("Room (metres)")
        f = QtWidgets.QFormLayout(room)
        self.lx, self.ly, self.lz = _spin(1, 30, 5.0), _spin(1, 30, 4.0), _spin(1.5, 10, 2.5)
        self.temp = _spin(-10, 40, 20.0, 1.0, " °C", 0)
        self.t60 = _spin(0.1, 3.0, 0.5, 0.05, " s")
        self.t60.setToolTip("Damping used for the modal model (how long the bass takes to decay 60 dB).")
        self.t60_btn = QtWidgets.QPushButton("Take from last measurement")
        self.t60_btn.clicked.connect(self._t60_from_measurement)
        f.addRow("Length (x)", self.lx)
        f.addRow("Width (y)", self.ly)
        f.addRow("Height (z)", self.lz)
        f.addRow("Temperature", self.temp)
        f.addRow("Bass decay T60", self.t60)
        f.addRow(self.t60_btn)
        v.addWidget(room)

        pos = QtWidgets.QGroupBox("Positions  (0,0,0 = front-left floor corner)")
        p = QtWidgets.QGridLayout(pos)
        self.src = [_spin(0, 30, 0.4), _spin(0, 30, 1.3), _spin(0, 10, 1.1)]
        self.lis = [_spin(0, 30, 3.5), _spin(0, 30, 2.0), _spin(0, 10, 1.2)]
        for r, (name, spins) in enumerate((("Speaker / sub", self.src), ("Listener (mic)", self.lis))):
            p.addWidget(QtWidgets.QLabel(name), r * 2, 0, 1, 3)
            for c, sp in enumerate(spins):
                sp.setSuffix("")
                p.addWidget(sp, r * 2 + 1, c)
        p.addWidget(QtWidgets.QLabel("x            y            z   (m)"), 4, 0, 1, 3)
        v.addWidget(pos)

        an = QtWidgets.QGroupBox("Analysis")
        a = QtWidgets.QFormLayout(an)
        self.fmax = _spin(100, 600, 300.0, 10.0, " Hz", 0)
        self.prom = _spin(1, 15, 4.0, 0.5, " dB", 1)
        self.prom.setToolTip("Minimum height of a measured response peak to be matched against the predicted modes.")
        a.addRow("Up to", self.fmax)
        a.addRow("Peak prominence", self.prom)
        self.kind_checks = {}
        row = QtWidgets.QHBoxLayout()
        for kind in ("axial", "tangential", "oblique"):
            cb = QtWidgets.QCheckBox(kind)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {KIND_COLORS[kind]};")
            cb.toggled.connect(lambda _=False: self._debounce.start())
            self.kind_checks[kind] = cb
            row.addWidget(cb)
        a.addRow(row)
        v.addWidget(an)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.summary)
        v.addStretch(1)

        for sp in (self.lx, self.ly, self.lz, self.temp, self.t60, self.fmax, *self.src, *self.lis):
            sp.valueChanged.connect(self._controls_changed)
        self.prom.valueChanged.connect(lambda _=0: self._debounce.start())
        return panel

    # ------------------------------------------------------------------ views
    def _build_views(self) -> QtWidgets.QWidget:
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.map = ModeMapPlot()
        split.addWidget(self.map)
        lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Freq Hz", "n (x,y,z)", "Type", "Mode", "Excites dB", "Measured peak"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._row_selected)
        lower.addWidget(self.table)

        fieldbox = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(fieldbox)
        fv.setContentsMargins(0, 0, 0, 0)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Frequency"))
        self.field_f = _spin(20, 600, 34.3, 0.5, " Hz", 1)
        self.field_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.field_slider.setRange(40, 1200)
        self.field_slider.setValue(69)
        snap = QtWidgets.QPushButton("Snap to mode")
        snap.clicked.connect(self._snap)
        self.field_z = _spin(0, 10, 1.2, 0.1, " m")
        ctl.addWidget(self.field_slider, 1)
        ctl.addWidget(self.field_f)
        ctl.addWidget(snap)
        ctl.addWidget(QtWidgets.QLabel("Height"))
        ctl.addWidget(self.field_z)
        fv.addLayout(ctl)
        self.field = ModeFieldView()
        self.field.positionsChanged.connect(self._dragged)
        fv.addWidget(self.field, 1)
        lower.addWidget(fieldbox)
        lower.setSizes([420, 560])
        split.addWidget(lower)
        split.setSizes([330, 330])
        self.field_slider.valueChanged.connect(lambda v: self._field_freq(v / 2.0, self.field_slider))
        self.field_f.valueChanged.connect(lambda v: self._field_freq(v, self.field_f))
        self.field_z.valueChanged.connect(lambda _=0: self._update_field())
        return split

    # ------------------------------------------------------------------ state <-> widgets
    def _controls_changed(self) -> None:
        if self._loading:
            return
        from roomscope.core.modes import RoomDims  # noqa: PLC0415

        cfg = RoomConfig(
            RoomDims(self.lx.value(), self.ly.value(), self.lz.value()), self.temp.value(),
            tuple(s.value() for s in self.src), tuple(s.value() for s in self.lis), self.t60.value(), self.fmax.value(),
        )
        cfg = cfg.with_(source=cfg.clamp(cfg.source), listener=cfg.clamp(cfg.listener))
        self.state.set_room(cfg)

    def _load_from_state(self) -> None:
        c = self.state.room
        self._loading = True
        for sp, v in ((self.lx, c.dims.lx), (self.ly, c.dims.ly), (self.lz, c.dims.lz), (self.temp, c.temp_c),
                      (self.t60, c.t60), (self.fmax, c.f_max), *zip(self.src, c.source), *zip(self.lis, c.listener)):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
        self.field_z.blockSignals(True)
        self.field_z.setValue(c.listener[2])
        self.field_z.blockSignals(False)
        self._loading = False
        self.field.set_config(c)
        self._debounce.start()

    def _dragged(self, src: tuple, lis: tuple) -> None:
        c = self.state.room
        self.state.set_room(c.with_(source=c.clamp(src), listener=c.clamp(lis)))

    def _t60_from_measurement(self) -> None:
        if not self.state.measurements:
            QtWidgets.QMessageBox.information(self, "Bass decay", "Run a measurement on the Measure tab first.")
            return
        d = self.state.measurements[-1].decay
        vals = [b.t60 for b in d.bands if 50 <= b.fc <= 300 and np.isfinite(b.t60)]
        t = float(np.mean(vals)) if vals else d.t_mid()
        if np.isfinite(t):
            self.t60.setValue(float(np.clip(t, 0.1, 3.0)))

    # ------------------------------------------------------------------ analysis
    def recompute(self) -> None:
        cfg = self.state.room
        self.modes = cfg.modes()
        exc = excitation_db(cfg, self.modes)
        self.map.visible_kinds = {k for k, cb in self.kind_checks.items() if cb.isChecked()}
        self.map.set_modes(self.modes, exc, cfg.f_max)
        f = np.arange(20.0, cfg.f_max + 0.5, 0.5)
        y = predicted_response_db(cfg, f)
        f_m = y_m = None
        self.matches = []
        if self.state.measurements:
            f_m, y_m = self.state.measurements[-1].bass(f_hi=cfg.f_max * 1.05)
            keep = (f_m >= 20) & (f_m <= cfg.f_max)
            f_m, y_m = f_m[keep], y_m[keep]
            peaks = find_response_peaks(f_m, y_m, 20.0, cfg.f_max, self.prom.value())
            self.matches = match_peaks(peaks, self.modes, cfg)
        self.map.set_curves(f, y, f_m, y_m)
        self.map.set_matches(self.matches, f_m, y_m)
        self._fill_table(cfg, exc)
        self._fill_summary(cfg)
        self._update_field()

    def _low_modes(self, cfg: RoomConfig) -> list:
        """Individual modes only matter below the Schroeder frequency; above it they overlap by nature."""
        return [m for m in self.modes if m.freq <= max(cfg.schroeder_hz, 80.0)]

    def _fill_table(self, cfg: RoomConfig, exc: np.ndarray) -> None:
        by_mode = {m.best.n: m for m in self.matches if m.matched}
        pairs = coincident_modes(self._low_modes(cfg), 0.02)
        coinc = {a.n for a, _b in pairs} | {b.n for _a, b in pairs}
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.modes))
        for r, (m, e) in enumerate(zip(self.modes, exc)):
            hit = by_mode.get(m.n)
            cells = [f"{m.freq:.1f}", m.index_str, m.kind, m.label + ("  *" if m.n in coinc else ""), f"{e:.0f}",
                     f"{hit.peak.freq:.1f} ({hit.error_hz:+.1f})" if hit else ""]
            for c, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setForeground(QtGui.QBrush(QtGui.QColor(KIND_COLORS[m.kind] if c == 2 else (theme.GOOD if (c == 5 and hit) else theme.TEXT))))
                if c in (0, 4):
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _fill_summary(self, cfg: RoomConfig) -> None:
        kinds = {k: sum(1 for m in self.modes if m.kind == k) for k in ("axial", "tangential", "oblique")}
        ok, bad = bonello_check(self.modes, 25.0, cfg.f_max)
        pairs = coincident_modes(self._low_modes(cfg), 0.02)
        gaps = largest_gaps(self.modes, 20.0, min(cfg.f_max, 200.0), 1)
        lines = [
            f"Volume {cfg.volume:.1f} m³ · speed of sound {cfg.c:.0f} m/s",
            f"Schroeder frequency ≈ {cfg.schroeder_hz:.0f} Hz (modes overlap above this)",
            f"{len(self.modes)} modes below {cfg.f_max:.0f} Hz: {kinds['axial']} axial, {kinds['tangential']} tangential, {kinds['oblique']} oblique",
            f"Lowest mode {self.modes[0].freq:.1f} Hz ({self.modes[0].label})" if self.modes else "",
            f"Bonello criterion: {'passes' if ok else 'fails near ' + ', '.join(f'{b:.0f}' for b in bad[:4]) + ' Hz'}",
            f"{len(pairs)} mode pairs within 2 % below the Schroeder frequency (marked *) merge into single big peaks",
        ]
        if gaps:
            lines.append(f"Widest gap: {gaps[0][0]:.1f} Hz above {gaps[0][1]:.0f} Hz")
        if self.state.measurements:
            n = len(self.matches)
            hit = sum(m.matched for m in self.matches)
            lines.append(f"Measured response: {n} peaks, {hit} on predicted modes" if n else "Measured response: no clear peaks")
        self.summary.setText("\n".join(x for x in lines if x))

    # ------------------------------------------------------------------ field
    def _field_freq(self, f: float, source: QtWidgets.QWidget | None = None) -> None:
        """Set the field frequency; keeps the spin box and slider in step (``source`` is the widget that was edited)."""
        for w, value in ((self.field_f, f), (self.field_slider, int(round(f * 2)))):
            if w is not source:
                w.blockSignals(True)
                w.setValue(value)
                w.blockSignals(False)
        self._update_field()

    def _snap(self) -> None:
        if self.modes:
            f = self.field_f.value()
            self._field_freq(min(self.modes, key=lambda m: abs(m.freq - f)).freq)

    def _update_field(self) -> None:
        self.field.set_field(self.state.room, self.field_f.value(), self.field_z.value())

    def _row_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self.modes):
            self._field_freq(self.modes[rows[0].row()].freq)
