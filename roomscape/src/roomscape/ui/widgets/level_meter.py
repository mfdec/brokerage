"""Level read-out: big number, dBFS bar with peak tick, latching clip LED."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from roomscape.ui import theme

BAR_MIN, BAR_MAX = -100.0, 0.0


class LevelMeterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(230, 112)
        self._level = -200.0       # numeric read-out in `unit`
        self._bar_dbfs = -200.0    # digital level used for the bar
        self._peak_bar = -200.0
        self._peak_text = -200.0
        self._leq = -200.0
        self._unit = "dBFS"
        self._weighting = "Z"
        self._clip_until = 0.0
        self._peak_hold = -200.0
        self._peak_hold_t = 0.0

    def set_values(self, level: float, bar_dbfs: float, peak_dbfs: float, leq: float, clipped: bool,
                   unit: str, weighting: str, offset: float) -> None:
        now = time.monotonic()
        self._level, self._bar_dbfs, self._leq = level, bar_dbfs, leq
        self._unit, self._weighting = unit, weighting
        if peak_dbfs >= self._peak_hold or now - self._peak_hold_t > 1.5:
            self._peak_hold, self._peak_hold_t = peak_dbfs, now
        self._peak_bar = self._peak_hold
        self._peak_text = self._peak_hold + offset
        if clipped:
            self._clip_until = now + 2.0
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor(theme.PANEL))
        p.setPen(QtGui.QColor(theme.TEXT))
        big = QtGui.QFont(self.font())
        big.setPointSize(26)
        big.setBold(True)
        p.setFont(big)
        txt = "--" if self._level < -150 else f"{self._level:.1f}"
        p.drawText(QtCore.QRectF(10, 4, w - 60, 44), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, txt)
        small = QtGui.QFont(self.font())
        small.setPointSize(9)
        p.setFont(small)
        p.setPen(QtGui.QColor(theme.MUTED))
        w_tag = "" if self._weighting == "Z" else f" ({self._weighting})"
        p.drawText(QtCore.QRectF(10, 46, w - 20, 16), QtCore.Qt.AlignmentFlag.AlignLeft, f"{self._unit}{w_tag}   Peak {self._peak_text:.1f}   Leq {self._leq:.1f}"
                   if self._level > -150 else f"{self._unit}{w_tag}")
        # bar
        bx, by, bw, bh = 10, h - 30, w - 20, 12
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#0b0e12"))
        p.drawRoundedRect(QtCore.QRectF(bx, by, bw, bh), 4, 4)
        frac = max(0.0, min(1.0, (self._bar_dbfs - BAR_MIN) / (BAR_MAX - BAR_MIN)))
        grad = QtGui.QLinearGradient(bx, 0, bx + bw, 0)
        grad.setColorAt(0.0, QtGui.QColor("#2c8a5a"))
        grad.setColorAt(0.75, QtGui.QColor("#d6c35a"))
        grad.setColorAt(1.0, QtGui.QColor(theme.BAD))
        p.setBrush(grad)
        p.drawRoundedRect(QtCore.QRectF(bx, by, bw * frac, bh), 4, 4)
        pf = max(0.0, min(1.0, (self._peak_bar - BAR_MIN) / (BAR_MAX - BAR_MIN)))
        p.setPen(QtGui.QPen(QtGui.QColor(theme.TEXT), 2))
        p.drawLine(QtCore.QPointF(bx + bw * pf, by - 2), QtCore.QPointF(bx + bw * pf, by + bh + 2))
        p.setPen(QtGui.QColor(theme.MUTED))
        for db in (-80, -60, -40, -20, 0):
            x = bx + bw * (db - BAR_MIN) / (BAR_MAX - BAR_MIN)
            p.drawText(QtCore.QRectF(x - 20, by + bh + 1, 40, 12), QtCore.Qt.AlignmentFlag.AlignCenter, str(db))
        # clip LED
        on = time.monotonic() < self._clip_until
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(theme.BAD if on else "#3a2226"))
        p.drawEllipse(QtCore.QPointF(w - 22, 20), 8, 8)
        p.setPen(QtGui.QColor(theme.MUTED))
        p.drawText(QtCore.QRectF(w - 50, 30, 56, 14), QtCore.Qt.AlignmentFlag.AlignCenter, "CLIP")
        p.end()
