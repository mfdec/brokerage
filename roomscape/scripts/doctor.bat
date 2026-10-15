@echo off
rem Health check of audio, camera and software.  Examples:
rem   scripts\doctor.bat --all
rem   scripts\doctor.bat --url http://192.168.1.20:4747/video
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo RoomScape is not set up yet - run scripts\setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m roomscape.diagnostics %*
echo.
pause
