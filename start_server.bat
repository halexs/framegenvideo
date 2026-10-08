@echo off
REM Start the StreamerFrames web UI and job manager using the project's virtualenv Python.
SETLOCAL
cd /d "%~dp0"
set VENV_PY=%~dp0..\.framegen\Scripts\python.exe
if not exist "%VENV_PY%" (
  echo Virtualenv Python not found at %VENV_PY%
  echo Please create or point .framegen to your Python venv.
  pause
  exit /b 1
)

"%VENV_PY%" -m streamerframes serve %*

echo Server exited. Press any key to close.
pause
