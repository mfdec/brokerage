#!/usr/bin/env bash
# Build the Windows program on Linux, under Wine (no Windows PC needed), then check it with its own --selftest.
#
#   sudo apt-get install -y wine64 libwine fonts-wine xvfb curl unzip zip python3-pip
#   packaging/build_exe_wine.sh                  # result: $WORK/RoomScape-<version>-win64.zip
#
# What it does: fetches a Windows CPython (NuGet package) and the Windows wheels from PyPI, installs them into a throw-away
# Wine prefix, runs PyInstaller there on packaging/roomscape.spec, runs `RoomScape.exe --selftest` under Wine and zips the folder.
# The exe is unsigned: Windows SmartScreen will ask "Run anyway" on first start.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"                 # the roomscape/ folder
WORK="${WORK:-${TMPDIR:-/tmp}/roomscape-winbuild}"
PYVER="${PYVER:-3.12.10}"
PYMM="${PYVER%.*}"                                                    # 3.12
ABI="cp${PYMM//./}"                                                   # cp312
WINE="${WINE:-/usr/lib/wine/wine64}"
export WINEPREFIX="$WORK/prefix" WINEDEBUG="-all" WINEDLLOVERRIDES="mscoree,mshtml="
mkdir -p "$WORK/dl" "$WORK/wheels"
winpath() { echo "Z:${1//\//\\}"; }
# Wine's Python cannot use a plain file as stdout (WinError 6 at start-up), so its output always goes through a pipe.
run() { xvfb-run -a "$WINE" "$@" 2>&1 | cat; }
PY="$WORK/py/tools/python.exe"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$HERE/src/roomscape/__init__.py")"

echo "== Windows Python $PYVER"
if [ ! -f "$PY" ]; then
    curl -fsSL -o "$WORK/dl/python.nupkg" "https://api.nuget.org/v3-flatcontainer/python/$PYVER/python.$PYVER.nupkg"
    rm -rf "$WORK/py" && mkdir -p "$WORK/py" && unzip -q "$WORK/dl/python.nupkg" -d "$WORK/py"
fi

echo "== Windows wheels"
python3 -m pip download -q -d "$WORK/wheels" --platform win_amd64 --python-version "$PYMM" --implementation cp \
    --abi "$ABI" --only-binary=:all: -r "$HERE/packaging/requirements-windows.txt"
python3 -m pip download -q -d "$WORK/wheels" --platform win_amd64 --python-version "$PYMM" --implementation cp \
    --abi "$ABI" --only-binary=:all: pefile pywin32-ctypes              # Windows-only dependencies that pip skips on Linux

echo "== Wine prefix and packages"
run wineboot --init >/dev/null 2>&1 || true
run "$PY" -m pip install -q --no-index --find-links "$(winpath "$WORK/wheels")" -r "$(winpath "$HERE/packaging/requirements-windows.txt")" \
    --no-warn-script-location

echo "== PyInstaller"
rm -rf "$WORK/dist" "$WORK/build"
run "$PY" -m PyInstaller --noconfirm --clean --log-level WARN --distpath "$(winpath "$WORK/dist")" --workpath "$(winpath "$WORK/build")" \
    "$(winpath "$HERE/packaging/roomscape.spec")"

echo "== Self-test of the built program (under Wine, offscreen)"
rm -rf "$WORK/selftest"
run "$WORK/dist/RoomScape/RoomScape.exe" --selftest --out "$(winpath "$WORK/selftest")" || true
cat "$WORK/selftest/roomscape-selftest.txt"
grep -q "ALL CHECKS PASSED" "$WORK/selftest/roomscape-selftest.txt"

echo "== Zip"
cp "$HERE/packaging/WINDOWS-README.txt" "$WORK/dist/RoomScape/README.txt" 2>/dev/null || true
( cd "$WORK/dist" && rm -f "../RoomScape-$VERSION-win64.zip" && zip -q -r -9 "../RoomScape-$VERSION-win64.zip" RoomScape )
ls -la "$WORK/RoomScape-$VERSION-win64.zip"
