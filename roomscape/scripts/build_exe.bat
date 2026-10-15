@echo off
rem Build the Windows program (dist\RoomScape\RoomScape.exe) with PyInstaller.  Run scripts\setup.bat first.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo RoomScape is not set up yet - run scripts\setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pyinstaller || goto :fail
python -m PyInstaller --noconfirm --clean packaging\roomscape.spec || goto :fail

echo.
echo Done: dist\RoomScape\RoomScape.exe
echo Check it:  dist\RoomScape\RoomScape.exe --selftest
echo To share it, zip the whole dist\RoomScape folder (the exe needs the _internal folder next to it).
pause
exit /b 0

:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
