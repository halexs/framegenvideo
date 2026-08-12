@echo off
REM Start the StreamerFrames FastAPI server using the project's virtualenv Python
SETLOCAL ENABLEDELAYEDEXPANSION

set VENV_PY=%~dp0\..\.framegen\Scripts\python.exe
if not exist "%VENV_PY%" (
  echo Virtualenv Python not found at %VENV_PY%
  echo Please create or point .framegen to your Python venv.
  pause
  exit /b 1
)

n"%VENV_PY%" -m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info

necho Server exited. Press any key to close.
pause
