"""Dark theme shared by Qt widgets and pyqtgraph plots."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

BG = "#0f1318"
PANEL = "#161b22"
PLOT_BG = "#0b0e12"
BORDER = "#2a313c"
TEXT = "#d6dde6"
MUTED = "#8b95a3"
ACCENT = "#4cc2ff"
AMBER = "#ffb454"
GOOD = "#5fd38d"
BAD = "#ff6b6b"
GRID = "#3a4350"

QSS = f"""
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 12px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QPushButton {{ background: #1e2530; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #555e6b; }}
QPushButton#primary {{ background: #12394f; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton#danger {{ background: #4a1f25; border-color: {BAD}; color: white; font-weight: 600; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: #11161d; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{ padding: 7px 16px; background: {BG}; border: 1px solid {BORDER}; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; }}
QStatusBar {{ color: {MUTED}; }}
QHeaderView::section {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 3px; }}
QTableView {{ gridline-color: {BORDER}; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    roles = {
        QtGui.QPalette.ColorRole.Window: BG, QtGui.QPalette.ColorRole.WindowText: TEXT,
        QtGui.QPalette.ColorRole.Base: "#11161d", QtGui.QPalette.ColorRole.AlternateBase: PANEL,
        QtGui.QPalette.ColorRole.Text: TEXT, QtGui.QPalette.ColorRole.Button: "#1e2530",
        QtGui.QPalette.ColorRole.ButtonText: TEXT, QtGui.QPalette.ColorRole.ToolTipBase: PANEL,
        QtGui.QPalette.ColorRole.ToolTipText: TEXT, QtGui.QPalette.ColorRole.Highlight: "#1d5f86",
        QtGui.QPalette.ColorRole.HighlightedText: "white", QtGui.QPalette.ColorRole.PlaceholderText: MUTED,
        QtGui.QPalette.ColorRole.Link: ACCENT,
    }
    for role, col in roles.items():
        pal.setColor(role, c(col))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    f = app.font()
    f.setPointSize(10)
    app.setFont(f)
    pg.setConfigOptions(antialias=True, background=PLOT_BG, foreground=MUTED)
