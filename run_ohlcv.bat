@echo off
REM ============================================================================
REM  run_ohlcv.bat  -  rebuild the 1-minute underlying file with real candles
REM  Safe to run while the chain build is going: different endpoint, no collision.
REM
REM  NO SECRETS IN THIS FILE. POLYGON_KEY comes from your Windows user
REM  environment — see the header of run_train.bat to set it.
REM ============================================================================
cd /d C:\sinus-daemon
call .venv\Scripts\activate.bat

set SINUS_SYMBOL=SPY
set SINUS_CSV=C:\sinus\data\spy_1min_parity.csv
set SINUS_OUT=C:\sinus\data\spy_1min_ohlcv.csv
set SINUS_YEARS=2
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

if not defined POLYGON_KEY (
  echo. ^& echo  POLYGON_KEY is not set in your user environment. See run_train.bat's header. ^& pause ^& exit /b 2
)
if not exist polygon_ohlcv.py (
  echo. ^& echo  MISSING: polygon_ohlcv.py - copy it into C:\sinus-daemon ^& pause ^& exit /b 2
)

python polygon_ohlcv.py
pause
