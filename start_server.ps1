# Start the StreamerFrames web UI and job manager using the virtualenv Python, then open the browser.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $scriptDir "..\.framegen\Scripts\python.exe"

if (!(Test-Path $python)) {
    Write-Error "Python not found at: $python"
    exit 1
}

# New window so logs are visible; host/port come from streamerframes.toml (default 127.0.0.1:8000).
Start-Process -FilePath $python -ArgumentList "-m streamerframes serve" -WorkingDirectory $scriptDir -WindowStyle Normal

Start-Sleep -Seconds 3
Start-Process "http://localhost:8000"
