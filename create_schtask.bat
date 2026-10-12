@echo off
REM Create a Windows scheduled task that starts the server in the background on logon (no browser)
setlocal
set TASK_NAME=StreamerFramesServer
set SCRIPT=%~dp0start_server.ps1
if not exist "%SCRIPT%" (
  echo start_server.ps1 not found at %SCRIPT%
  exit /b 1
)
schtasks /Create /SC ONLOGON /RL LIMITED /F /TN "%TASK_NAME%" /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"%SCRIPT%\" -NoBrowser"
if %ERRORLEVEL%==0 (
  echo Task "%TASK_NAME%" created. To run now: schtasks /Run /TN "%TASK_NAME%"
) else (
  echo Failed to create scheduled task. Errorlevel %ERRORLEVEL%
)
