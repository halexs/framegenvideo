@echo off
REM Stop the StreamerFrames web UI started by start_server.bat.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1" -Stop
pause
