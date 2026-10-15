"""Application entry point."""

from __future__ import annotations

import argparse
import sys

from roomscope import __version__

_CONSOLE_FLAGS = ("--version", "-h", "--help", "--doctor", "--selftest")


def _ensure_console() -> bool:
    """A windowed Windows executable (the packaged build) has no console: join the parent's (cmd, PowerShell) or open one.

    Returns True when a *new* console window had to be opened, so the caller can keep it up until the user has read it.
    """
    if sys.platform != "win32" or (sys.stdout is not None and sys.stderr is not None):
        return False
    import ctypes  # noqa: PLC0415

    kernel32 = ctypes.windll.kernel32
    attached = bool(kernel32.AttachConsole(-1))                          # ATTACH_PARENT_PROCESS
    if not attached and not kernel32.AllocConsole():
        return False
    for name, mode, device in (("stdout", "w", "CONOUT$"), ("stderr", "w", "CONOUT$"), ("stdin", "r", "CONIN$")):
        try:
            setattr(sys, name, open(device, mode, encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return not attached


def _pause(opened_console: bool) -> None:
    if opened_console:
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opened_console = _ensure_console() if any(a in _CONSOLE_FLAGS for a in argv) else False

    parser = argparse.ArgumentParser(prog="roomscope", description="RoomScope room acoustics analyzer")
    parser.add_argument("--version", action="version", version=f"RoomScope {__version__}")
    parser.add_argument("--sim", action="store_true", help="start listening to the simulated room immediately")
    parser.add_argument("--selftest", action="store_true", help="check the installation end to end with simulated hardware, then exit")
    parser.add_argument("--out", metavar="FOLDER", help="where --selftest writes its report and screenshot (default: a temporary folder)")
    parser.add_argument("--no-pause", action="store_true", help="never wait for Enter before closing a console window opened by --doctor / --selftest")
    parser.add_argument("--doctor", nargs=argparse.REMAINDER, metavar="OPTIONS",
                        help="run the audio / camera / software health check (same options as roomscope-doctor, e.g. --doctor --all)")
    args = parser.parse_args(argv)

    if args.doctor is not None:
        from roomscope import diagnostics  # noqa: PLC0415

        rc = diagnostics.main(args.doctor)
        _pause(opened_console and not args.no_pause)
        return rc
    if args.selftest:
        from roomscope import selftest  # noqa: PLC0415

        rc = selftest.run(args.out)
        _pause(opened_console and not args.no_pause)
        return rc

    from PySide6 import QtWidgets  # noqa: PLC0415  (keep --version fast and Qt-free)

    from roomscope.ui.icon import make_icon  # noqa: PLC0415
    from roomscope.ui.main_window import MainWindow  # noqa: PLC0415
    from roomscope.ui.theme import apply_theme  # noqa: PLC0415

    if sys.platform == "win32":                                          # let the taskbar show RoomScope's own icon and group
        try:
            import ctypes  # noqa: PLC0415

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RoomScope.RoomScope")
        except Exception:  # noqa: BLE001
            pass
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("RoomScope")
    app.setOrganizationName("RoomScope")
    app.setWindowIcon(make_icon())
    apply_theme(app)
    win = MainWindow()
    win.show()
    if args.sim:
        from roomscope.audio.devices import SIM_DEVICE  # noqa: PLC0415

        win.state.open_source(SIM_DEVICE, 48000)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
