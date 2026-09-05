# ============================================================================
# SINUS — train until you stop it.
#
#   .\start.ps1              preflight, then run
#   .\start.ps1 -SkipCheck   run immediately (you already checked this session)
#   .\start.ps1 -Quick       preflight in quick mode (skips the pipeline build)
#
# Ctrl-C once asks for a clean stop. Ctrl-C twice kills it. Either is safe: the
# leaderboard is rewritten atomically after every trial and the champion is
# written on promotion, so you lose at most the trial in flight.
#
# For the two-year backfill first, run .\overnight.ps1 instead.
# ============================================================================
param(
    [switch]$SkipCheck,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
Set-Location C:\sinus-daemon
. .\.venv\Scripts\Activate.ps1
. .\_sinus_env.ps1

# --- hold the machine awake while this window lives -------------------------
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED. This does NOT defeat closing the lid —
# for that, set Control Panel > Power Options > "When I close the lid" to
# "Do nothing" on AC. It does stop the idle sleep timer, which is what kills
# most overnight runs.
# PowerShell parses hex literals above 0x7FFFFFFF as a negative Int32, so these are
# decimal: 2147483649 = ES_CONTINUOUS|ES_SYSTEM_REQUIRED, 2147483648 = ES_CONTINUOUS.
$script:KeepAwake = $false
try {
    # The type survives for the life of the PowerShell session, so re-running this
    # script in the same window would fail with TYPE_ALREADY_EXISTS. Only add it once.
    if (-not ('Win32.Power' -as [type])) {
        Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
    }
    [void][Win32.Power]::SetThreadExecutionState([uint32]2147483649)
    $script:KeepAwake = $true
    Write-Host "sleep suppressed for the life of this window" -ForegroundColor DarkGray
} catch {
    # Never let this stop a run. Worst case Windows sleeps on its own idle timer.
    Write-Host "could not suppress sleep ($($_.Exception.Message)) - continuing" -ForegroundColor Yellow
}

function Release-KeepAwake {
    if ($script:KeepAwake) {
        [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648)
        Write-Host "sleep suppression released"
    }
}

# --- preflight --------------------------------------------------------------
if (-not $SkipCheck) {
    $args = @()
    if ($Quick) { $args += "--quick" }
    python preflight.py @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`npreflight failed - not starting. Fix the FAIL lines above." -ForegroundColor Red
        Release-KeepAwake
        exit 1
    }
}

# --- run, with supervision --------------------------------------------------
# railway_daemon already retries internally with backoff and gives up after five
# identical failures. This outer loop covers the case where the process itself
# dies (OOM, driver reset, an unhandled exit) rather than raising.
$log = "C:\sinus\data\run_$(Get-Date -Format 'yyyy-MM-dd_HHmm').log"
Write-Host "logging to $log`n" -ForegroundColor DarkGray

$restarts = 0
while ($true) {
    $started = Get-Date
    python railway_daemon.py 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $ran = (Get-Date) - $started

    if ($code -eq 0) {
        Write-Host "`ndaemon exited cleanly after $([int]$ran.TotalMinutes) min." -ForegroundColor Green
        break
    }
    # A non-zero exit that came back fast is a configuration problem, not a
    # transient one. Restarting into it just hides the message.
    if ($ran.TotalMinutes -lt 2) {
        Write-Host "`ndaemon exited ($code) after $([int]$ran.TotalSeconds)s - too fast to be transient." -ForegroundColor Red
        Write-Host "Read the last lines above, or $log" -ForegroundColor Red
        break
    }
    $restarts++
    if ($restarts -ge 5) {
        Write-Host "`nrestarted $restarts times - stopping. Something is wrong that a restart will not fix." -ForegroundColor Red
        break
    }
    Write-Host "`ndaemon exited ($code) after $([int]$ran.TotalMinutes) min - restarting ($restarts/5) in 30s" -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}

Release-KeepAwake
