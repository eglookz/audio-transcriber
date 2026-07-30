@echo off
chcp 65001 >nul
rem Transcribe everything in .\input - drag-and-drop friendly launcher.
rem You can also drop files directly on this .bat: they are passed through as args.
setlocal
cd /d "%~dp0"
python transcribe.py %*
echo.
pause
