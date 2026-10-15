@echo off
rem Build the Windows program (dist\RoomScope\RoomScope.exe) with PyInstaller.  Run scripts\setup.bat first.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo RoomScope is not set up yet - run scripts\setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pyinstaller || goto :fail
python -m PyInstaller --noconfirm --clean packaging\roomscope.spec || goto :fail

echo.
echo Done: dist\RoomScope\RoomScope.exe
echo Check it:  dist\RoomScope\RoomScope.exe --selftest
echo To share it, zip the whole dist\RoomScope folder (the exe needs the _internal folder next to it).
pause
exit /b 0

:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
