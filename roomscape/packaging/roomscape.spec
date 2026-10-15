# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RoomScape: a Windows one-folder build.

On Windows:  scripts\\build_exe.bat             (or: python -m PyInstaller --noconfirm --clean packaging\\roomscape.spec)
Result:      dist\\RoomScape\\RoomScape.exe      (keep the _internal folder next to it)

The program has no console; run it with --selftest or --doctor from a terminal to see text output
(RoomScape.exe --selftest  /  RoomScape.exe --doctor --all).
"""
import os
import sys

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from roomscape import __version__  # noqa: E402

ver = tuple(int(x) for x in __version__.split(".")[:3]) + (0,)
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=ver, prodvers=ver, mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", "RoomScape"),
            StringStruct("FileDescription", "RoomScape room acoustics analyzer"),
            StringStruct("FileVersion", __version__),
            StringStruct("InternalName", "RoomScape"),
            StringStruct("LegalCopyright", "MIT licence"),
            StringStruct("OriginalFilename", "RoomScape.exe"),
            StringStruct("ProductName", "RoomScape"),
            StringStruct("ProductVersion", __version__),
        ])]),
        VarFileInfo([VarStruct("Translation", [1033, 1200])]),
    ],
)
os.makedirs(CONF["workpath"], exist_ok=True)
version_file = os.path.join(CONF["workpath"], "version_info.txt")
with open(version_file, "w", encoding="utf-8") as fh:
    fh.write(str(version_info))

a = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=collect_data_files("pyqtgraph", excludes=["examples/**", "**/tests/**"]),      # colour-map tables, icons
    hiddenimports=["pygrabber.dshow_graph"],                                            # optional: friendly camera names
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "pyqtgraph.examples", "pyqtgraph.opengl", "OpenGL"],
    noarchive=False,
)

# Trim what a raster-painted Qt Widgets program never loads (software OpenGL, Direct3D shader compiler, translations).
DROP = ("opengl32sw.dll", "d3dcompiler_47.dll")
a.binaries = [b for b in a.binaries if os.path.basename(b[0]).lower() not in DROP]
a.datas = [d for d in a.datas if "/translations/" not in d[0].replace("\\", "/")]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RoomScape",
    debug=False,
    strip=False,
    upx=False,                                   # UPX-packed executables trip antivirus scanners
    console=False,
    icon=os.path.join(SPECPATH, "roomscape.ico"),
    version=version_file,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="RoomScape")
