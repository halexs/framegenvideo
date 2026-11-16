@echo off
REM Start the StreamerFrames web UI in the background and open the browser (double-click friendly).
REM   start_server.bat               background, opens the browser
REM   start_server.bat -NoBrowser    background only
REM   start_server.bat -Foreground   run in this window with live output (Ctrl+C to stop)
REM   start_server.bat -Status       show whether it is running
REM Stop it with stop_server.bat.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1" %*
if errorlevel 1 pause
