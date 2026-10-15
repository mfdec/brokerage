"""Three small plan-view maps (measured / modelled / difference) shown under the camera picture."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from roomscape.ui import theme


class PlanView(pg.PlotWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setBackground(theme.PLOT_BG)
        self.pi = self.getPlotItem()
        self.pi.setMenuEnabled(False)
        self.pi.setAspectLocked(True)
        self.pi.hideButtons()
        self.pi.vb.setMouseEnabled(x=False, y=False)
        for ax in ("left", "bottom"):
            self.pi.getAxis(ax).setPen(pg.mkPen(theme.GRID))
            self.pi.getAxis(ax).setTextPen(pg.mkPen(theme.MUTED))
        self.pi.setTitle(title, size="9pt", color=theme.MUTED)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.pi.addItem(self.img)
        self.outline = pg.PlotCurveItem(pen=pg.mkPen(theme.TEXT, width=1.5))
        self.pi.addItem(self.outline)
        self.trail = pg.PlotCurveItem(pen=pg.mkPen(255, 255, 255, 140, width=1.5))
        self.pi.addItem(self.trail)
        self.mic = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(theme.GOOD), pen=pg.mkPen("#102010"))
        self.spk = pg.ScatterPlotItem(size=9, symbol="s", brush=pg.mkBrush(theme.AMBER), pen=pg.mkPen("#201808"))
        self.pi.addItem(self.mic)
        self.pi.addItem(self.spk)
        self._dims = None

    def set_room(self, lx: float, ly: float, source_xy) -> None:
        if self._dims != (lx, ly):
            self._dims = (lx, ly)
            self.outline.setData([0, lx, lx, 0, 0], [0, 0, ly, ly, 0])
            self.pi.setRange(xRange=(-0.1, lx + 0.1), yRange=(-0.1, ly + 0.1), padding=0.0)
        self.spk.setData([source_xy[0]], [source_xy[1]])

    def set_image(self, rgba: np.ndarray | None, lx: float, ly: float) -> None:
        if rgba is None:
            self.img.clear()
            return
        self.img.setImage(rgba, autoLevels=False)
        self.img.setRect(QtCore.QRectF(0, 0, lx, ly))

    def set_path(self, trail_xy: np.ndarray | None, mic_xy) -> None:
        if trail_xy is not None and len(trail_xy) > 1:
            self.trail.setData(trail_xy[:, 0], trail_xy[:, 1], connect="finite")
        else:
            self.trail.setData([], [])
        if mic_xy is None:
            self.mic.setData([], [])
        else:
            self.mic.setData([mic_xy[0]], [mic_xy[1]])


class MapStrip(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.views = {
            "measured": PlanView("Measured (walk map)"),
            "model": PlanView("Model (modal theory)"),
            "diff": PlanView("Measured minus model"),
            "decay": PlanView("Decay map"),
        }
        for v in self.views.values():
            lay.addWidget(v, 1)
        self.views["decay"].setVisible(False)                # shown once the decay camera has data
        self.setMinimumHeight(170)
