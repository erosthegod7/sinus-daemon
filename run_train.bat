@echo off
REM ============================================================================
REM  run_train.bat  -  SINUS fresh-start trainer, laptop launcher
REM
REM  Usage:   run_train.bat build     build option history only, then stop
REM           run_train.bat           build if missing, then evolve forever
REM
REM  NO SECRETS IN THIS FILE. POLYGON_KEY and SINUS_GIT_TOKEN are read from your
REM  Windows user environment, same as _sinus_env.ps1 does. Set them once:
REM    [Environment]::SetEnvironmentVariable("POLYGON_KEY",     "<key>", "User")
REM    [Environment]::SetEnvironmentVariable("SINUS_GIT_TOKEN", "<PAT>", "User")
REM  then open a NEW window — an existing one keeps its old copy.
REM ============================================================================
cd /d C:\sinus-daemon
call .venv\Scripts\activate.bat

set SINUS_CSV=C:\sinus\data\spy_1min_ohlcv.csv
set SINUS_NODE=laptop
set SINUS_VOLUME=C:\sinus\data
set SINUS_SYMBOL=SPY
set SINUS_MAX_SESSIONS=500
set SINUS_BAND=15
set SINUS_API_SLEEP=0
set SINUS_PRUNE=1
set SINUS_PRUNE_PERCENTILE=60
set SINUS_SCREEN_ROUNDS=2
set SINUS_TFT_EPOCHS=12
set PYTHONUNBUFFERED=1
REM The engine prints non-ASCII (arrows, bullets). Without this, Python falls back to
REM cp1252 whenever stdout is a pipe or a file and dies on the first such line.
set PYTHONIOENCODING=utf-8

if not defined SINUS_GIT_REPO set SINUS_GIT_REPO=erosthegod7/sinus-champion

if not defined POLYGON_KEY (
  echo.
  echo  POLYGON_KEY is not set in your user environment.
  echo  See the header of this file, then open a new window.
  pause ^& exit /b 2
)
if not exist "%SINUS_CSV%" (
  echo.
  echo  SINUS_CSV not found: %SINUS_CSV%
  echo  Edit the path at the top of this file.
  pause ^& exit /b 2
)
if not defined SINUS_GIT_TOKEN (
  echo  SINUS_GIT_TOKEN not set - champions will stay local, training still runs.
)

for %%F in (sinus.py sinus_daemon.py sinus_search.py sinus_gitstore.py sinus_train.py) do (
  if not exist "%%F" ( echo  MISSING: %%F  - copy it into C:\sinus-daemon ^& pause ^& exit /b 2 )
)

if /i "%1"=="build" (
  echo  ---- building option history only ----
  python sinus_train.py --build
) else (
  echo  ---- build if missing, then evolve forever.  Ctrl+C to stop. ----
  python sinus_train.py
)
pause
