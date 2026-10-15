"""Application entry point."""

from __future__ import annotations

import argparse
import sys

from roomscope import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roomscope", description="RoomScope room acoustics analyzer")
    parser.add_argument("--version", action="version", version=f"RoomScope {__version__}")
    parser.add_argument("--sim", action="store_true", help="start listening to the simulated room immediately")
    args = parser.parse_args(argv)

    from PySide6 import QtWidgets  # noqa: PLC0415  (keep --version fast and Qt-free)

    from roomscope.ui.main_window import MainWindow  # noqa: PLC0415
    from roomscope.ui.theme import apply_theme  # noqa: PLC0415

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("RoomScope")
    app.setOrganizationName("RoomScope")
    apply_theme(app)
    win = MainWindow()
    win.show()
    if args.sim:
        from roomscope.audio.devices import SIM_DEVICE  # noqa: PLC0415

        win.state.open_source(SIM_DEVICE, 48000)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
