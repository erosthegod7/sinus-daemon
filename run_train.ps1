# ============================================================================
# run_train.ps1 — the trainer, shepherd edition.
#
#   .\run_train.ps1            build history if missing, then evolve forever
#   .\run_train.ps1 -Build     build history only, then stop
#
# Differences from run_train.bat, which it supersedes for unattended runs:
#   * sleep is suppressed for the life of this window (same call start.ps1 makes),
#     so the idle timer cannot kill an overnight run. Closing the lid still can —
#     set "When I close the lid" to "Do nothing" on AC.
#   * everything the trainer prints is also written to C:\sinus\data\train_<stamp>.log,
#     which is what a check-in reads to know whether the run is healthy.
#
# NO SECRETS IN THIS FILE — _sinus_env.ps1 pulls them from the user environment.
# ============================================================================
param([switch]$Build)

Set-Location C:\sinus-daemon
. .\.venv\Scripts\Activate.ps1
. .\_sinus_env.ps1

# trainer knobs — same values as run_train.bat, stated here so they are not inherited by accident
$env:SINUS_CSV              = "C:\sinus\data\spy_1min_ohlcv.csv"
$env:SINUS_SYMBOL           = "SPY"
$env:SINUS_MAX_SESSIONS     = "500"
$env:SINUS_BAND             = "15"
$env:SINUS_API_SLEEP        = "0"
$env:SINUS_PRUNE            = "1"
$env:SINUS_PRUNE_PERCENTILE = "60"
$env:SINUS_SCREEN_ROUNDS    = "2"
$env:SINUS_TFT_EPOCHS       = "12"
$env:PYTHONUNBUFFERED       = "1"
$env:PYTHONIOENCODING       = "utf-8"

# --- hold the machine awake while this window lives ---------------------------
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED, as decimal because PowerShell reads hex above
# 0x7FFFFFFF as a negative Int32.
$script:KeepAwake = $false
try {
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
    Write-Host "could not suppress sleep ($($_.Exception.Message)) - continuing" -ForegroundColor Yellow
}

$log = "C:\sinus\data\train_$(Get-Date -Format 'yyyyMMdd_HHmm').log"
Write-Host "log -> $log" -ForegroundColor DarkGray
$extra = ""
if ($Build) { $extra = " --build" }

# cmd does the stderr merge: PowerShell 5.1 would wrap every stderr line in an ErrorRecord.
cmd /c "python -u sinus_train.py$extra 2>&1" | Tee-Object -FilePath $log

if ($script:KeepAwake) { [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648) }
Write-Host "trainer exited - log is $log"
