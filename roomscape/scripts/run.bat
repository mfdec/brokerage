@echo off
rem Start RoomScape (run scripts\setup.bat once first).
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\pythonw.exe" (
    echo RoomScape is not set up yet - run scripts\setup.bat first.
    pause
    exit /b 1
)
rem pythonw: no console window.  Add --sim to try the built-in simulated room without any hardware.
start "" ".venv\Scripts\pythonw.exe" -m roomscape %*
