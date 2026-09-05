# Shared environment for start.ps1 / overnight.ps1. Dot-source it; don't run it alone.
#
# NO SECRETS IN THIS FILE. Keys live in your Windows user environment, set once:
#   [Environment]::SetEnvironmentVariable("POLYGON_KEY",     "<key>", "User")
#   [Environment]::SetEnvironmentVariable("SINUS_GIT_REPO",  "erosthegod7/sinus-champion", "User")
#   [Environment]::SetEnvironmentVariable("SINUS_GIT_TOKEN", "<github PAT>", "User")

# Read from the User scope (the registry, always current) and copy into this process, so this
# works even in a window that was already open when the variables were set.
$missing = @()
foreach ($name in "POLYGON_KEY","SINUS_GIT_REPO","SINUS_GIT_TOKEN") {
    $val = [Environment]::GetEnvironmentVariable($name, "User")
    if ([string]::IsNullOrWhiteSpace($val)) { $missing += $name } else { Set-Item "env:$name" $val }
}
if ($missing) {
    Write-Host "Missing user env vars: $($missing -join ', ')" -ForegroundColor Red
    Write-Host 'Set them with [Environment]::SetEnvironmentVariable("NAME","value","User"), then rerun.'
    exit 1
}
$env:MASSIVE_API_KEY = $env:POLYGON_KEY      # same key, two names in the codebase

$env:SINUS_VOLUME           = "C:\sinus\data"
$env:SINUS_DATA             = "C:\sinus\data"
# Real candles. spy_1min_parity.csv is ts,spot only — training on it leaves the whole
# candle/volume feature block NaN. Parity is the INPUT to run_ohlcv.bat, not a training file.
$env:SINUS_CSV              = "C:\sinus\data\spy_1min_ohlcv.csv"
$env:SINUS_NODE             = "laptop"
$env:SINUS_MIN_SESSIONS     = "7"    # HARD FLOOR: below 7 the val split is EMPTY and every
                                     # trial scores inf forever. Do not lower this.
$env:SINUS_MAX_SESSIONS     = "500"  # drop to 300 if the TFT tensor OOMs
$env:SINUS_TFT_EPOCHS       = "12"
$env:SINUS_PRUNE            = "1"
$env:SINUS_PRUNE_PERCENTILE = "60"  # raised from 30 on 2026-09-05: pruning has not killed a
                                    # likely winner, so screen harder
$env:SINUS_SCREEN_ROUNDS    = "200"
$env:PYTHONIOENCODING       = "utf-8"

Write-Host "env ready - key ...$($env:POLYGON_KEY.Substring([Math]::Max(0,$env:POLYGON_KEY.Length-4))) | repo $env:SINUS_GIT_REPO" -ForegroundColor DarkGray
