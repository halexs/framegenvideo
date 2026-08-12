# PowerShell helper to start the StreamerFrames FastAPI server using the virtualenv Python
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $scriptDir "..\.framegen\Scripts\python.exe"

if (!(Test-Path $python)) {
    Write-Error "Python not found at: $python"
    exit 1
}

# Start uvicorn in a new window so logs are visible
$arg = "-m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level info"
Start-Process -FilePath $python -ArgumentList $arg -WindowStyle Normal

# Open the default browser to the web UI
Start-Sleep -Seconds 2
Start-Process "http://localhost:8000"
