@echo off
REM Create a Windows scheduled task that runs start_server.bat on logon
setlocal
set TASK_NAME=StreamerFramesServer
set SCRIPT=%~dp0start_server.bat
if not exist "%SCRIPT%" (
  echo start_server.bat not found at %SCRIPT%
  exit /b 1
)
schtasks /Create /SC ONLOGON /RL HIGHEST /F /TN "%TASK_NAME%" /TR "\"%SCRIPT%\""
if %ERRORLEVEL%==0 (
  echo Task "%TASK_NAME%" created. To run now: schtasks /Run /TN "%TASK_NAME%"
) else (
  echo Failed to create scheduled task. Errorlevel %ERRORLEVEL%
)
