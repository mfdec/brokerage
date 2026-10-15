"""The application icon, painted in code (a decaying wave with its decay envelope), so no image files need shipping."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui

SIZES = (16, 24, 32, 48, 64, 128, 256)


def make_pixmap(size: int = 256) -> QtGui.QPixmap:
    s = float(size)
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    bg = QtGui.QLinearGradient(0, 0, 0, s)
    bg.setColorAt(0.0, QtGui.QColor("#1d2634"))
    bg.setColorAt(1.0, QtGui.QColor("#0b0e12"))
    p.setBrush(bg)
    p.setPen(QtGui.QPen(QtGui.QColor("#35425a"), max(s * 0.015, 1.0)))
    p.drawRoundedRect(QtCore.QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92), s * 0.2, s * 0.2)

    n, tau = 200, 3.0
    wave, env = QtGui.QPainterPath(), QtGui.QPainterPath()
    for i in range(n + 1):
        t = i / n
        x = s * (0.15 + 0.70 * t)
        decay = math.exp(-tau * t)
        yw = s * 0.5 - s * 0.30 * decay * math.sin(2.0 * math.pi * 5.5 * t)
        ye = s * 0.5 - s * 0.30 * decay
        wave.moveTo(x, yw) if i == 0 else wave.lineTo(x, yw)
        env.moveTo(x, ye) if i == 0 else env.lineTo(x, ye)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    dash = QtGui.QPen(QtGui.QColor("#ffb454"), max(s * 0.018, 1.0), QtCore.Qt.PenStyle.DashLine)
    p.setPen(dash)
    p.drawPath(env)
    p.setPen(QtGui.QPen(QtGui.QColor("#4cc2ff"), max(s * 0.04, 1.2), QtCore.Qt.PenStyle.SolidLine,
                        QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
    p.drawPath(wave)
    p.end()
    return pm


def make_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for size in SIZES:
        icon.addPixmap(make_pixmap(size))
    return icon
