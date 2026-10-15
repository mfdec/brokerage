"""Small helpers for log-frequency pyqtgraph plots."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from roomscope.core.units import format_freq
from roomscope.ui import theme

_MAJOR = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
_MINOR = [15, 30, 40, 60, 70, 80, 90, 150, 300, 400, 600, 700, 800, 900, 1500, 3000, 4000, 6000, 7000, 8000, 9000, 15000]


def freq_ticks(f_min: float, f_max: float):
    major = [(np.log10(f), format_freq(f)) for f in _MAJOR if f_min * 0.999 <= f <= f_max * 1.001]
    minor = [(np.log10(f), "") for f in _MINOR if f_min <= f <= f_max]
    return [major, minor]


def apply_log_freq_axis(plot: pg.PlotItem, f_min: float, f_max: float, axis: str = "bottom") -> None:
    plot.getAxis(axis).setTicks(freq_ticks(f_min, f_max))
    rng = (np.log10(f_min / 1.12), np.log10(f_max * 1.1))      # a little room so end labels are not clipped
    if axis == "bottom":
        plot.setXRange(*rng, padding=0)
        plot.setLimits(xMin=np.log10(f_min * 0.6), xMax=np.log10(f_max * 1.5))
    else:
        plot.setYRange(*rng, padding=0)


def style_plot(widget: pg.PlotWidget, left: str = "", bottom: str = "") -> pg.PlotItem:
    pi = widget.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.22)
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(theme.GRID))
        ax.setTextPen(pg.mkPen(theme.MUTED))
    if left:
        pi.setLabel("left", left)
    if bottom:
        pi.setLabel("bottom", bottom)
    widget.setBackground(theme.PLOT_BG)
    return pi
