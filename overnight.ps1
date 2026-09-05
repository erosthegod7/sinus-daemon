# ============================================================================
# SINUS overnight: backfill 2 years of chain data, then train until morning.
# Run it and walk away. Ctrl-C is safe — the backfill resumes where it stopped.
# ============================================================================
Set-Location C:\sinus-daemon
. .\.venv\Scripts\Activate.ps1
. .\_sinus_env.ps1

Write-Host "`n=== [1/3] backfill $(Get-Date -Format 'HH:mm') ===" -ForegroundColor Cyan
python polygon_chain_history.py --start 2024-09-03
if ($LASTEXITCODE -ne 0) {
    Write-Host "backfill failed (exit $LASTEXITCODE) - stopping. Nothing downloaded is lost; rerun to resume." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== [2/3] rebuilding parity CSV ===" -ForegroundColor Cyan
python -c "from sinus_chain_loader import load_history; s,c,f = load_history(r'C:\sinus\data'); s.to_csv(r'C:\sinus\data\spy_1min_parity.csv', index=False); print(len(s),'bars ->', s.ts.min(), s.ts.max())"
if ($LASTEXITCODE -ne 0) { Write-Host "CSV rebuild failed - stopping." -ForegroundColor Red; exit 1 }

Write-Host "`n=== [3/3] training $(Get-Date -Format 'HH:mm') ===" -ForegroundColor Cyan
python railway_daemon.py 2>&1 | Tee-Object -FilePath C:\sinus\data\run.log
