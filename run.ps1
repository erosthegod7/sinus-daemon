# ============================================================================
# SINUS — run the serving/search daemon directly, logging to run.log.
#
# NO SECRETS IN THIS FILE. _sinus_env.ps1 pulls POLYGON_KEY / SINUS_GIT_REPO /
# SINUS_GIT_TOKEN from your Windows user environment.
#
# For the guarded launch with a preflight, use .\start.ps1 instead.
# ============================================================================
Set-Location C:\sinus-daemon
. .\.venv\Scripts\Activate.ps1
. .\_sinus_env.ps1

$env:SINUS_NODE = "laptop"
$env:SINUS_VOLUME = "C:\sinus\data"
$env:SINUS_MAX_SESSIONS = "500"
$env:SINUS_TFT_EPOCHS = "12"
$env:SINUS_PRUNE = "1"
$env:SINUS_PRUNE_PERCENTILE = "60"
$env:SINUS_SCREEN_ROUNDS = "200"

python railway_daemon.py 2>&1 | Tee-Object -FilePath C:\sinus\data\run.log
