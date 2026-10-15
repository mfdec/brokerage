"""Room-mode plots: frequency-axis mode map with measured/predicted responses, and a draggable top-view field."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from roomscope.core.modes import Mode
from roomscope.core.roomfield import PeakMatch, RoomConfig, modal_field_db
from roomscope.ui import theme
from roomscope.ui.widgets.plots import style_plot

KIND_COLORS = {"axial": theme.ACCENT, "tangential": theme.AMBER, "oblique": "#7d8794"}
BASELINE = -46.0


class ModeMapPlot(pg.PlotWidget):
    """Modes as stems on a linear frequency axis (height = how strongly this speaker/listener pair excites them),
    with the predicted modal response and, if available, the measured bass response."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="dB (relative)", bottom="Frequency (Hz)")
        self.pi.setMenuEnabled(False)
        self.pi.setYRange(BASELINE - 4, 16, padding=0)
        self.pi.getAxis("left").setWidth(54)
        self.stems = {k: pg.PlotCurveItem(pen=pg.mkPen(c, width=2), connect="pairs") for k, c in KIND_COLORS.items()}
        for it in self.stems.values():
            self.pi.addItem(it)
        self.pred = self.pi.plot(pen=pg.mkPen(theme.GOOD, width=1.6, style=QtCore.Qt.PenStyle.DashLine), name="model")
        self.meas = self.pi.plot(pen=pg.mkPen("#ffffff", width=1.8), name="measured")
        self.labels: list[pg.TextItem] = []
        self.modes: list[Mode] = []
        self.visible_kinds = {"axial", "tangential", "oblique"}
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._mouse)
        leg = self.pi.addLegend(offset=(-10, 8))
        leg.setLabelTextColor(theme.MUTED)

    def set_modes(self, modes: list[Mode], excitation_db: np.ndarray, f_max: float) -> None:
        self.modes = modes
        for kind, item in self.stems.items():
            sel = [i for i, m in enumerate(modes) if m.kind == kind and kind in self.visible_kinds]
            if not sel:
                item.setData([], [])
                continue
            f = np.array([modes[i].freq for i in sel])
            top = np.maximum(excitation_db[sel], BASELINE + 2.0)
            x = np.repeat(f, 2)
            y = np.empty(f.size * 2)
            y[0::2], y[1::2] = BASELINE, top
            item.setData(x, y)
        self.pi.setXRange(20, f_max, padding=0.0)
        self.pi.setLimits(xMin=10, xMax=f_max * 1.2)

    def set_curves(self, f_pred: np.ndarray | None, y_pred: np.ndarray | None,
                   f_meas: np.ndarray | None, y_meas: np.ndarray | None) -> None:
        """Both curves are shown relative to their own median so shapes can be compared."""
        if f_pred is None:
            self.pred.setData([], [])
        else:
            self.pred.setData(f_pred, y_pred - np.median(y_pred))
        if f_meas is None:
            self.meas.setData([], [])
        else:
            self.meas.setData(f_meas, y_meas - np.median(y_meas))

    def set_matches(self, matches: list[PeakMatch], f_meas, y_meas) -> None:
        for t in self.labels:
            self.pi.removeItem(t)
        self.labels.clear()
        if not matches or f_meas is None:
            return
        med = float(np.median(y_meas))
        for m in sorted(matches, key=lambda m: -m.peak.prominence_db)[:14]:
            txt = f"{m.peak.freq:.0f} {m.best.label}" if m.matched else f"{m.peak.freq:.0f} ?"
            t = pg.TextItem(txt, color=theme.TEXT if m.matched else theme.BAD, anchor=(0.5, 1.1))
            t.setFont(QtGui.QFont(self.font().family(), 8))
            t.setPos(m.peak.freq, m.peak.level_db - med)
            self.pi.addItem(t)
            self.labels.append(t)

    def _mouse(self, evt) -> None:
        if not self.modes or not self.sceneBoundingRect().contains(evt[0]):
            return
        f = self.pi.vb.mapSceneToView(evt[0]).x()
        near = min((m for m in self.modes if m.kind in self.visible_kinds), key=lambda m: abs(m.freq - f), default=None)
        if near is not None:
            self.pi.setTitle(f"nearest mode: {near.freq:.1f} Hz  {near.label} ({near.kind})", size="10pt", color=theme.TEXT)


class ModeFieldView(pg.PlotWidget):
    """Top view of the modal pressure field at one frequency; drag the speaker / listener markers."""

    positionsChanged = QtCore.Signal(tuple, tuple)          # (source xyz), (listener xyz)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pi = style_plot(self, left="y (m)", bottom="x (m)")
        self.pi.setMenuEnabled(False)
        self.pi.setAspectLocked(True)
        self.pi.showGrid(x=False, y=False)
        self.pi.vb.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.pi.addItem(self.img)
        self.img.setLookupTable(pg.colormap.get("CET-D1").getLookupTable(0.0, 1.0, 256))
        self.img.setLevels((-15.0, 15.0))
        self.outline = pg.PlotCurveItem(pen=pg.mkPen(theme.TEXT, width=2))
        self.pi.addItem(self.outline)
        self.src_t = pg.TargetItem(pos=(0.5, 0.5), size=16, pen=pg.mkPen(theme.AMBER, width=2), label="speaker",
                                   labelOpts=dict(color=theme.AMBER, offset=(10, -8)))
        self.lis_t = pg.TargetItem(pos=(2.5, 2.0), size=16, pen=pg.mkPen(theme.GOOD, width=2), label="listener",
                                   labelOpts=dict(color=theme.GOOD, offset=(10, -8)))
        for t in (self.src_t, self.lis_t):
            t.setZValue(20)
            self.pi.addItem(t)
            t.sigPositionChangeFinished.connect(self._emit_positions)
        self._cfg: RoomConfig | None = None
        self._xs = self._ys = None
        self._db: np.ndarray | None = None
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._mouse)
        self._silent = False

    def set_config(self, cfg: RoomConfig) -> None:
        self._cfg = cfg
        L, W = cfg.dims.lx, cfg.dims.ly
        self.outline.setData([0, L, L, 0, 0], [0, 0, W, W, 0])
        self.pi.setRange(xRange=(-0.15, L + 0.15), yRange=(-0.15, W + 0.15), padding=0.02)
        self._silent = True
        self.src_t.setPos(cfg.source[0], cfg.source[1])
        self.lis_t.setPos(cfg.listener[0], cfg.listener[1])
        self._silent = False

    def set_field(self, cfg: RoomConfig, freq: float, z: float) -> None:
        xs, ys, db = modal_field_db(cfg, freq, z)
        self._xs, self._ys, self._db = xs, ys, db
        self.img.setImage(db, autoLevels=False, levels=(-15.0, 15.0))
        self.img.setRect(QtCore.QRectF(0, 0, cfg.dims.lx, cfg.dims.ly))
        self.pi.setTitle(f"{freq:.1f} Hz at height {z:.2f} m  (dB re room median)", size="10pt", color=theme.MUTED)

    def _emit_positions(self) -> None:
        if self._silent or self._cfg is None:
            return
        s, l = self.src_t.pos(), self.lis_t.pos()
        c = self._cfg
        self.positionsChanged.emit((s.x(), s.y(), c.source[2]), (l.x(), l.y(), c.listener[2]))

    def _mouse(self, evt) -> None:
        if self._db is None or not self.sceneBoundingRect().contains(evt[0]):
            return
        p = self.pi.vb.mapSceneToView(evt[0])
        ix = int(np.clip(p.x() / self._cfg.dims.lx * self._db.shape[1], 0, self._db.shape[1] - 1))
        iy = int(np.clip(p.y() / self._cfg.dims.ly * self._db.shape[0], 0, self._db.shape[0] - 1))
        self.pi.setTitle(f"x {p.x():.2f} m  y {p.y():.2f} m   {self._db[iy, ix]:+.1f} dB", size="10pt", color=theme.TEXT)
