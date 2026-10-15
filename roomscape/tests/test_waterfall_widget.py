import numpy as np
import pytest
from PySide6 import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    from roomscape.ui.theme import apply_theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app)
    return app


def make_data(ns=40, nf=200, seed=0):
    rng = np.random.default_rng(seed)
    f = np.geomspace(20, 20000, nf)
    base = -30 - 10 * np.log10(f / 20)[None, :] - np.linspace(0, 40, ns)[:, None]
    return f, base + rng.normal(0, 2, (ns, nf)), np.linspace(0, 6, ns)


def nonbackground_fraction(img):
    a = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width(), :3]
    return float(np.mean(np.abs(a.astype(int) - np.array([14, 11, 11])).sum(axis=2) > 40))   # BGRA, PLOT_BG


def test_waterfall_renders_and_rotates(qapp):
    from roomscape.ui.widgets.waterfall3d import Waterfall3D

    w = Waterfall3D()
    w.resize(900, 400)
    f, sl, t = make_data()
    w.set_data(f, sl, t, -100, -20)
    img0 = w.grab().toImage()
    assert nonbackground_fraction(img0) > 0.06
    for az, el in ((-70, 10), (0, 85), (70, 40), (-34, 28)):
        w.set_view(az, el)
        assert nonbackground_fraction(w.grab().toImage()) > 0.05, (az, el)
    assert w.last_paint_ms < 200


def test_nearest_slice_occludes_those_behind(qapp):
    """A tall front wall must hide a loud slice directly behind it (hidden-surface pass)."""
    from roomscape.ui.widgets.waterfall3d import Waterfall3D

    f = np.geomspace(20, 20000, 120)
    ns = 12
    sl = np.full((ns, f.size), -90.0)
    sl[0] = -25.0                       # front slice: tall
    sl[1:] = -85.0
    sl[6] = -50.0                       # a mid-height slice behind the wall
    w = Waterfall3D()
    w.resize(700, 420)
    w.set_view(0, 30, 1.0)
    w.set_data(f, sl, np.linspace(0, 6, ns), -100, -20)
    img = w.grab().toImage()
    assert nonbackground_fraction(img) > 0.05
    # drawing must not crash with an all-floor surface or constant data either
    w.set_data(f, np.full((ns, f.size), -200.0), np.linspace(0, 6, ns), -100, -20)
    w.grab()
    w.set_data(f, np.full((ns, f.size), 0.0), np.linspace(0, 6, ns), -100, -20)
    w.grab()


def test_mouse_rotation_changes_view(qapp):
    from PySide6 import QtCore, QtGui, QtTest

    from roomscape.ui.widgets.waterfall3d import Waterfall3D

    w = Waterfall3D()
    w.resize(600, 400)
    w.show()
    az0, el0 = w.az, w.el
    QtTest.QTest.mousePress(w, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(100, 100))
    QtTest.QTest.mouseMove(w, QtCore.QPoint(160, 130))
    QtTest.QTest.mouseRelease(w, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(160, 130))
    assert (w.az, w.el) != (az0, el0)
    w.mouseDoubleClickEvent(None)
    assert (w.az, w.el) == (az0, el0)
