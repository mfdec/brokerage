"""Video display with clickable / draggable calibration points."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from roomscape.ui import theme


class CameraView(QtWidgets.QWidget):
    picked = QtCore.Signal(float, float)               # click in pick mode (image coordinates)
    pointMoved = QtCore.Signal(int, float, float)      # a pin was dragged
    pointReleased = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setMouseTracking(True)
        self._image: QtGui.QImage | None = None
        self._buf: np.ndarray | None = None
        self.points: list[tuple[float, float] | None] = []   # pixel positions of the pins (image coords)
        self.pick_mode = False
        self.hint = "No camera connected"
        self._drag: int | None = None
        self._hover: tuple[float, float] | None = None

    # ------------------------------------------------------------------ frame
    def set_frame(self, bgr: np.ndarray) -> None:
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        self._buf = bgr                                      # keep the pixels alive while the QImage uses them
        self._image = QtGui.QImage(bgr.data, w, h, bgr.strides[0], QtGui.QImage.Format.Format_BGR888)
        self.update()

    def clear(self) -> None:
        self._image = self._buf = None
        self.update()

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._image.width(), self._image.height()) if self._image is not None else (0, 0)

    # ------------------------------------------------------------------ geometry
    def _transform(self):
        if self._image is None:
            return 1.0, 0.0, 0.0
        iw, ih = self._image.width(), self._image.height()
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2.0, (self.height() - ih * s) / 2.0

    def to_image(self, pos: QtCore.QPointF) -> tuple[float, float]:
        s, ox, oy = self._transform()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    def to_widget(self, u: float, v: float) -> QtCore.QPointF:
        s, ox, oy = self._transform()
        return QtCore.QPointF(ox + u * s, oy + v * s)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(theme.PLOT_BG))
        if self._image is None:
            p.setPen(QtGui.QColor(theme.MUTED))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.hint)
            p.end()
            return
        s, ox, oy = self._transform()
        p.drawImage(QtCore.QRectF(ox, oy, self._image.width() * s, self._image.height() * s), self._image)
        font = QtGui.QFont(self.font())
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        for i, pt in enumerate(self.points):
            if pt is None:
                continue
            c = self.to_widget(*pt)
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 4))
            p.drawEllipse(c, 9, 9)
            p.setPen(QtGui.QPen(QtGui.QColor(theme.AMBER), 2))
            p.drawEllipse(c, 9, 9)
            p.drawLine(c + QtCore.QPointF(-14, 0), c + QtCore.QPointF(14, 0))
            p.drawLine(c + QtCore.QPointF(0, -14), c + QtCore.QPointF(0, 14))
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawText(c + QtCore.QPointF(13, -9), str(i + 1))
            p.setPen(QtGui.QColor(theme.AMBER))
            p.drawText(c + QtCore.QPointF(12, -10), str(i + 1))
        if self.pick_mode:
            p.setPen(QtGui.QColor(theme.AMBER))
            p.drawText(QtCore.QRectF(10, 8, self.width() - 20, 22), QtCore.Qt.AlignmentFlag.AlignLeft, "Click in the picture to place the point  (drag points to fine-tune)")
        p.end()

    # ------------------------------------------------------------------ mouse
    def _nearest(self, pos: QtCore.QPointF, radius: float = 14.0) -> int | None:
        best, bd = None, radius
        for i, pt in enumerate(self.points):
            if pt is None:
                continue
            d = QtCore.QLineF(self.to_widget(*pt), pos).length()
            if d < bd:
                best, bd = i, d
        return best

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._image is None or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = ev.position()
        i = self._nearest(pos)
        if i is not None:
            self._drag = i
            return
        if self.pick_mode:
            self.picked.emit(*self.to_image(pos))

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._drag is not None:
            u, v = self.to_image(ev.position())
            iw, ih = self.image_size
            self.pointMoved.emit(self._drag, float(np.clip(u, 0, iw - 1)), float(np.clip(v, 0, ih - 1)))
        else:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor if (self.pick_mode or self._nearest(ev.position()) is not None)
                           else QtCore.Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, _ev) -> None:
        if self._drag is not None:
            i, self._drag = self._drag, None
            self.pointReleased.emit(i)
