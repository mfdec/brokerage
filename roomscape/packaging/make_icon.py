"""Write packaging/roomscape.ico (PNG-compressed frames) from the icon painted in roomscape.ui.icon.

    python packaging/make_icon.py
"""

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from roomscape.ui.icon import SIZES, make_pixmap  # noqa: E402


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frames = []
    for size in SIZES:
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        make_pixmap(size).save(buf, "PNG")
        frames.append((size, bytes(buf.data())))
    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    entries, blobs = b"", b""
    for size, png in frames:
        dim = 0 if size >= 256 else size                                   # 0 means 256 in the ICO directory
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(blobs))
        blobs += png
    out = Path(__file__).with_name("roomscape.ico")
    out.write_bytes(header + entries + blobs)
    print(f"wrote {out} ({out.stat().st_size} bytes, sizes {[s for s, _ in frames]})")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
