"""3D waterfall (slices of spectrum stacked in time) drawn with QPainter - no OpenGL needed.

Used for the live waterfall (slices = recent spectra, newest in front) and, in v0.3, for the
cumulative-spectral-decay waterfall of a measured room (slice 0 = direct sound, later slices = decay).
Drag to rotate, wheel to zoom, double-click to reset.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from roomscope.core.units import format_freq
from roomscope.ui import theme
from roomscope.ui.widgets.spectrogram_view import colormap_lut

_FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
_DEFAULT_VIEW = (-34.0, 28.0, 1.0)


def nice_ticks(span: float, target: int = 5) -> np.ndarray:
    """Round tick values in [0, span]."""
    if span <= 0:
        return np.array([0.0])
    raw = span / max(target, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    return np.arange(0.0, span + 1e-9, step)


class Waterfall3D(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 220)
        self.setMouseTracking(True)
        self.freqs: np.ndarray | None = None
        self.slices: np.ndarray | None = None      # (n_slices, n_freq) dB, slice 0 in front
        self.times: np.ndarray | None = None       # (n_slices,) in `time_unit`
        self.span = 1.0
        self.lo, self.hi = -100.0, 0.0
        self.az, self.el, self.zoom = _DEFAULT_VIEW
        self.time_unit = "s"
        self.time_axis_label = "seconds ago"
        self.level_unit = "dB"
        self.depth_ratio, self.height_ratio, self.persp = 0.55, 0.42, 0.30
        self.x_ratio = 1.0              # frequency axis stretches to the widget's aspect ratio
        self.set_colormap("inferno")
        self._last_pos: QtCore.QPoint | None = None
        self.last_paint_ms = 0.0

    # ------------------------------------------------------------------ data
    def set_colormap(self, name: str) -> None:
        self._lut = colormap_lut(name, 256)
        self.update()

    def set_data(self, freqs: np.ndarray, slices: np.ndarray, times: np.ndarray, lo: float, hi: float,
                 span: float | None = None) -> None:
        self.freqs, self.slices, self.times = freqs, slices, np.asarray(times, dtype=np.float64)
        self.lo, self.hi = float(lo), float(hi)
        self.span = float(span if span else max(self.times.max(), 1e-9))
        self.update()

    def set_view(self, az: float, el: float, zoom: float | None = None) -> None:
        self.az = float(np.clip(az, -80, 80))
        self.el = float(np.clip(el, 4, 85))
        if zoom is not None:
            self.zoom = float(np.clip(zoom, 0.5, 3.0))
        self.update()

    # ------------------------------------------------------------------ geometry
    def _project(self, u, v, w):
        x = (np.asarray(u) - 0.5) * self.x_ratio
        y = (np.asarray(v) - 0.5) * self.depth_ratio
        z = (np.asarray(w) - 0.5) * self.height_ratio
        az, el = np.radians(self.az), np.radians(self.el)
        x1 = x * np.cos(az) - y * np.sin(az)
        y1 = x * np.sin(az) + y * np.cos(az)
        y2 = y1 * np.cos(el) - z * np.sin(el)
        z2 = y1 * np.sin(el) + z * np.cos(el)
        p = 1.0 / (1.0 + self.persp * y2)
        return x1 * p, z2 * p, y2

    def _fit(self):
        ml, mr, mt, mb = 54, 54, 30, 44
        self.x_ratio = float(np.clip((self.width() - ml - mr) / max(self.height() - mt - mb, 1) / 1.7, 1.0, 2.8))
        c = np.array([[u, v, w] for u in (0, 1) for v in (0, 1) for w in (0, 1)], dtype=float)
        sx, sy, _ = self._project(c[:, 0], c[:, 1], c[:, 2])
        bw, bh = sx.max() - sx.min(), sy.max() - sy.min()
        scale = min((self.width() - ml - mr) / bw, (self.height() - mt - mb) / bh) * self.zoom
        cx =ml + (self.width() - ml - mr) / 2 - 0.5 * (sx.max() + sx.min()) * scale
        cy = mt + (self.height() - mt - mb) / 2 + 0.5 * (sy.max() + sy.min()) * scale
        return cx, cy, scale

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(theme.PLOT_BG))
        if self.slices is None or self.freqs is None or self.slices.shape[0] < 2:
            p.setPen(QtGui.QColor(theme.MUTED))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for data...")
            p.end()
            return
        t0 = time.perf_counter()
        cx, cy, k = self._fit()

        def px(sx, sy):
            return cx + sx * k, cy - sy * k

        ns, nf = self.slices.shape
        lf = np.log10(self.freqs)
        u = (lf - lf[0]) / (lf[-1] - lf[0])
        v = np.clip(self.times / self.span, 0.0, 1.0)
        w = np.clip((self.slices - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        U = np.broadcast_to(u[None, :], (ns, nf))
        V = np.broadcast_to(v[:, None], (ns, nf))
        sx, sy, depth = self._project(U, V, w)
        sx0, sy0, _ = self._project(U, V, np.zeros_like(w))
        X, Y = px(sx, sy)
        X0, Y0 = px(sx0, sy0)

        font = QtGui.QFont(self.font())
        font.setPointSize(8)
        p.setFont(font)
        self._draw_back_grid(p, px, u)

        # --- horizon (hidden-surface) pass: each slice only paints/strokes the part nobody nearer covers
        order = np.argsort(depth.mean(axis=1))                       # nearest first
        xs = np.arange(np.floor(X.min()), np.ceil(X.max()) + 3.0, 3.0)
        curve = np.empty((ns, xs.size))
        floor = np.empty_like(curve)
        for r, j in enumerate(order):
            curve[r] = np.interp(xs, X[j], Y[j], left=np.inf, right=np.inf)
            floor[r] = np.interp(xs, X0[j], Y0[j], left=np.inf, right=np.inf)
        horizon = np.minimum.accumulate(curve, axis=0)
        prev = np.vstack((np.full((1, xs.size), np.inf), horizon[:-1]))
        bottom = np.minimum(floor, prev)
        visible = curve < bottom

        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        n_stops = 12
        base = self._lut[np.linspace(0, 255, n_stops).astype(int), :3].astype(float)
        bg = np.array([11, 14, 18], dtype=float)
        vs = v[order]
        _, yf, _ = self._project(0.5, vs, 0.0)
        _, yc, _ = self._project(0.5, vs, 1.0)
        yf_px, yc_px = px(0.0, yf)[1], px(0.0, yc)[1]
        stroke_x = [[] for _ in range(4)]
        stroke_y = [[] for _ in range(4)]
        nan = np.array([np.nan])
        front = None
        for r in range(ns):
            if not visible[r].any():
                continue
            cols = np.isfinite(curve[r]) & np.isfinite(floor[r])
            x = xs[cols]
            bot = bottom[r][cols]
            top = np.minimum(curve[r][cols], bot)                    # hidden columns collapse onto the horizon
            fade = 0.62 * float(vs[r])
            rgb = base * (1.0 - fade) + bg * fade
            grad = QtGui.QLinearGradient(0.0, yf_px[r], 0.0, yc_px[r])
            grad.setStops([(j / (n_stops - 1), QtGui.QColor(int(c[0]), int(c[1]), int(c[2]))) for j, c in enumerate(rgb)])
            poly = pg.arrayToQPath(np.concatenate((x, x[::-1])), np.concatenate((top, bot[::-1])), connect="all")
            poly.closeSubpath()
            p.fillPath(poly, QtGui.QBrush(grad))
            bucket = min(3, int(4 * vs[r]))
            stroke_x[bucket] += [x, nan]
            stroke_y[bucket] += [np.where(visible[r][cols], curve[r][cols], np.nan), nan]
            if r == 0:
                front = (x, np.where(visible[r][cols], curve[r][cols], np.nan))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for b_i in range(4):
            if stroke_x[b_i]:
                path = pg.arrayToQPath(np.concatenate(stroke_x[b_i]), np.concatenate(stroke_y[b_i]), connect="finite")
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 215 - 45 * b_i), 1.0))
                p.drawPath(path)
        if front is not None:                                        # front slice gets a heavier line
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 245), 1.8))
            p.drawPath(pg.arrayToQPath(np.concatenate((front[0], nan)), np.concatenate((front[1], nan)), connect="finite"))

        self._draw_axes(p, px, u, v)
        p.end()
        self.last_paint_ms = (time.perf_counter() - t0) * 1000.0

    def _draw_back_grid(self, p: QtGui.QPainter, px, u: np.ndarray) -> None:
        pen = QtGui.QPen(QtGui.QColor(theme.GRID), 1.0)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        p.setPen(pen)
        # back wall: dB lines + frequency lines; side wall: dB lines
        step = 20.0 if (self.hi - self.lo) > 50 else 10.0
        levels = np.arange(np.ceil(self.lo / step) * step, self.hi + 1e-6, step)
        for lv in levels:
            wv = (lv - self.lo) / (self.hi - self.lo)
            a = px(*self._project(0.0, 1.0, wv)[:2])
            b = px(*self._project(1.0, 1.0, wv)[:2])
            p.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
            p.setPen(QtGui.QColor(theme.MUTED))
            p.drawText(QtCore.QRectF(a[0] - 46, a[1] - 7, 42, 14), QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, f"{lv:g}")
            p.setPen(pen)
        lf0, lf1 = np.log10(self.freqs[0]), np.log10(self.freqs[-1])
        for f in _FREQ_TICKS:
            if self.freqs[0] <= f <= self.freqs[-1]:
                uu = (np.log10(f) - lf0) / (lf1 - lf0)
                a = px(*self._project(uu, 1.0, 0.0)[:2])
                b = px(*self._project(uu, 1.0, 1.0)[:2])
                p.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))

    def _draw_axes(self, p: QtGui.QPainter, px, u, v) -> None:
        p.setPen(QtGui.QPen(QtGui.QColor(theme.MUTED), 1.0))
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        pts = [QtCore.QPointF(*px(*self._project(a, b, 0.0)[:2])) for a, b in corners]
        p.drawPolyline(pts + [pts[0]])
        lf0, lf1 = np.log10(self.freqs[0]), np.log10(self.freqs[-1])
        # frequency ticks along the front edge
        for f in _FREQ_TICKS:
            if self.freqs[0] <= f <= self.freqs[-1]:
                uu = (np.log10(f) - lf0) / (lf1 - lf0)
                x, y = px(*self._project(uu, 0.0, 0.0)[:2])
                p.setPen(QtGui.QPen(QtGui.QColor(theme.MUTED), 1.0))
                p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + 4))
                p.drawText(QtCore.QRectF(x - 24, y + 5, 48, 14), QtCore.Qt.AlignmentFlag.AlignCenter, format_freq(f))
        # time ticks along the right (or left) edge, whichever is on the open side
        edge_u = 1.0 if self.az <= 0 else 0.0
        sign = 1 if edge_u == 1.0 else -1
        for t in nice_ticks(self.span, 4):
            vv = t / self.span
            x, y = px(*self._project(edge_u, vv, 0.0)[:2])
            p.setPen(QtGui.QPen(QtGui.QColor(theme.MUTED), 1.0))
            p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + 4 * sign, y))
            rect = QtCore.QRectF(x + 6 if sign > 0 else x - 56, y - 7, 50, 14)
            p.drawText(rect, (QtCore.Qt.AlignmentFlag.AlignLeft if sign > 0 else QtCore.Qt.AlignmentFlag.AlignRight)
                       | QtCore.Qt.AlignmentFlag.AlignVCenter, f"{t:g}{self.time_unit}")
        p.setPen(QtGui.QColor(theme.MUTED))
        p.drawText(QtCore.QRectF(8, 4, 300, 16), QtCore.Qt.AlignmentFlag.AlignLeft, f"{self.level_unit}  |  {self.time_axis_label} (depth)  |  drag: rotate, wheel: zoom")

    # ------------------------------------------------------------------ interaction
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._last_pos = ev.position().toPoint()

    def mouseReleaseEvent(self, _ev) -> None:
        self._last_pos = None

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._last_pos is None or not (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            return
        pos = ev.position().toPoint()
        d = pos - self._last_pos
        self._last_pos = pos
        self.set_view(self.az + d.x() * 0.4, self.el + d.y() * 0.4)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        self.set_view(self.az, self.el, self.zoom * (1.0015 ** ev.angleDelta().y()))

    def mouseDoubleClickEvent(self, _ev) -> None:
        self.az, self.el, self.zoom = _DEFAULT_VIEW
        self.update()
