# Start, stop or check the StreamerFrames web UI and job manager.
#   .\start_server.ps1               start in the background (no window) and open the browser
#   .\start_server.ps1 -NoBrowser    start in the background without opening the browser
#   .\start_server.ps1 -Foreground   run in this console with live output (Ctrl+C to stop)
#   .\start_server.ps1 -Status       show whether it is running and on which addresses
#   .\start_server.ps1 -Stop         stop the server (running GPU jobs finish and are adopted on next start)
# Host and port come from streamerframes.toml (default 127.0.0.1:8000).
param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Foreground,
    [switch]$NoBrowser
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $scriptDir "..\.framegen\Scripts\python.exe"
$consoleLog = Join-Path $scriptDir "server-console.log"   # startup errors; request logs go to <cache>\server.log

if (!(Test-Path $python)) {
    Write-Error "Python not found at: $python"
    exit 1
}

$port = 8000
$config = Join-Path $scriptDir "streamerframes.toml"
if ($env:PORT) { $port = [int]$env:PORT }
elseif (Test-Path $config) {
    $m = Select-String -Path $config -Pattern '^\s*port\s*=\s*(\d+)' | Select-Object -First 1
    if ($m) { $port = [int]$m.Matches[0].Groups[1].Value }
}

# The venv launcher and the real interpreter both carry this command line, so both are matched.
function Get-ServerProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -match '-m streamerframes serve' }
}

function Get-WorkerProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -match 'streamerframes\.pipeline\.worker' }
}

function Show-Status {
    $procs = @(Get-ServerProcesses)
    if ($procs.Count -eq 0) {
        Write-Host "StreamerFrames is not running."
    } else {
        $listen = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        Write-Host "StreamerFrames is running (pid $(($procs.ProcessId) -join ', '))."
        if ($listen -and $listen.LocalAddress -in @('0.0.0.0', '::')) {
            $ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' }
            Write-Host "  http://localhost:$port/"
            foreach ($ip in $ips) { Write-Host "  http://$($ip.IPAddress):$port/  ($($ip.InterfaceAlias))" }
        } elseif ($listen) {
            Write-Host "  http://localhost:$port/  (this PC only; set host = `"0.0.0.0`" for LAN access)"
        } else {
            Write-Host "  not listening on port $port yet; see $consoleLog"
        }
    }
    $workers = @(Get-WorkerProcesses)
    if ($workers.Count -gt 0) { Write-Host "GPU jobs running: $($workers.Count) worker process(es)." }
}

if ($Status) { Show-Status; exit 0 }

if ($Stop) {
    $procs = @(Get-ServerProcesses)
    if ($procs.Count -eq 0) { Write-Host "StreamerFrames is not running."; exit 0 }
    foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "StreamerFrames stopped."
    $workers = @(Get-WorkerProcesses)
    if ($workers.Count -gt 0) {
        Write-Host "Note: $($workers.Count) GPU job worker(s) are still running; they finish on their own (the PC stays"
        Write-Host "awake until then) and the next server start picks them up. Cancel them from the web UI to stop them."
    }
    exit 0
}

if (@(Get-ServerProcesses).Count -gt 0) {
    Write-Host "StreamerFrames is already running."
    Show-Status
    if (!$NoBrowser -and !$Foreground) { Start-Process "http://localhost:$port" }
    exit 0
}

if ($Foreground) {
    & $python -m streamerframes serve
    exit $LASTEXITCODE
}

# Background: hidden window; console output goes to server-console.log so startup errors aren't lost.
Start-Process -FilePath $python -ArgumentList "-m streamerframes serve" -WorkingDirectory $scriptDir `
    -WindowStyle Hidden -RedirectStandardOutput ($consoleLog -replace '\.log$', '.out.log') -RedirectStandardError $consoleLog

$up = $false
for ($i = 0; $i -lt 30 -and !$up; $i++) {
    Start-Sleep -Seconds 1
    $up = [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    if (!$up -and @(Get-ServerProcesses).Count -eq 0) { break }
}
if (!$up) {
    Write-Host "StreamerFrames failed to start. Last lines of $($consoleLog):"
    Get-Content $consoleLog -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}
Show-Status
if (!$NoBrowser) { Start-Process "http://localhost:$port" }
