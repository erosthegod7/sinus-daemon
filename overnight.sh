#!/bin/sh
cd /c/sinus-daemon
PY=./.venv/Scripts/python.exe

# Output here is redirected, so Python would otherwise pick cp1252 and die on the
# first non-ASCII line the engine prints.
export PYTHONIOENCODING=utf-8

echo "=== [1/3] backfill starting $(date) ==="
$PY -u polygon_chain_history.py --start 2024-09-03
echo "=== [1/3] backfill exited $? at $(date) ==="

echo "=== [2/3] rebuilding parity CSV ==="
$PY -u -c "from sinus_chain_loader import load_history; s,c,f = load_history(r'C:\sinus\data'); s.to_csv(r'C:\sinus\data\spy_1min_parity.csv', index=False); print(len(s),'bars ->', s.ts.min(), s.ts.max())"

echo "=== [3/3] training starting $(date) ==="
exec $PY -u railway_daemon.py
