"""Shared test helpers."""

import time

from PySide6 import QtWidgets


def spin(ms: int) -> None:
    """Process Qt events for ``ms`` milliseconds.

    Deliberately not ``QTest.qWait``: that call holds the Python GIL while it waits, which starves the
    audio/worker threads the tests are exercising (a 0.1 s job took 35 s).
    """
    end = time.monotonic() + ms / 1000.0
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.004)


def spin_until(pred, timeout_s: float = 30.0) -> bool:
    end = time.monotonic() + timeout_s
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.004)
    return False
