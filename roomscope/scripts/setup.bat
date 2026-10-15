@echo off
rem RoomScope - one-time setup on Windows 10/11.
rem Creates a private virtual environment in .venv and installs RoomScope with everything it needs.
setlocal
cd /d "%~dp0.."

set "PY="
rem Prefer a Python that has binary wheels for everything RoomScope needs.
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python was not found.
    echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
    echo and tick "Add python.exe to PATH" in the installer, then run this file again.
    pause
    exit /b 1
)

echo Using: %PY%
%PY% -m venv .venv || goto :fail
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail
python -m pip install -e ".[camnames]" || goto :fail

echo.
echo Setup finished.
echo   Start RoomScope:   scripts\run.bat
echo   Check your setup:  scripts\doctor.bat --all
pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
