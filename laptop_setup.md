# SINUS on the laptop (i9-14900HX · RTX 5070 · 32 GB)

The laptop is the PRIMARY searcher — 24 cores for the trees, a GPU for the TFT, and enough RAM
to skip the session cap. Railway is the always-on fallback. Both share one champion via GitHub.

## One-time setup (PowerShell)

```powershell
# 1. clone the code repo
git clone https://github.com/erosthegod7/sinus-daemon.git
cd sinus-daemon

# 2. python env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn pyarrow requests lightgbm catboost

# 3. torch WITH CUDA — this is the line that turns the GPU on (do NOT use the CPU build in requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line must print `CUDA: True NVIDIA GeForce RTX 5070`. If it says False, the CPU
build is installed — uninstall torch and redo step 3.

## Environment (set once — System Properties → Environment Variables, or per-session)

```powershell
$env:POLYGON_KEY        = "your key"
$env:SINUS_GIT_REPO     = "erosthegod7/sinus-champion"     # a SEPARATE repo from the code
$env:SINUS_GIT_TOKEN    = "github_pat_..."                  # fine-grained, Contents: read/write on that repo
$env:SINUS_NODE         = "laptop"
$env:SINUS_VOLUME       = "C:\sinus\data"
$env:SINUS_MAX_SESSIONS = "500"        # 32 GB handles it; Railway is capped at 300
$env:SINUS_TFT_EPOCHS   = "12"
```

## Run

```powershell
python railway_daemon.py
```

Despite the name it's the same entrypoint everywhere. First boot pulls ~2 years of 1-min SPY
from Polygon (a few minutes), caches it, pulls the current champion from GitHub, and starts
searching. Ctrl-C stops it cleanly; the leaderboard is flushed on exit.

## The champion repo

Create `sinus-champion` on GitHub (private, empty). The daemon initialises it on first push.
It will contain:

    champion/                current best weights + champion.json
    lineage/                 one JSON per promotion, dated, per node — the rollback trail
    leaderboard_laptop.csv   every trial this machine ran
    leaderboard_railway.csv  every trial Railway ran

Both machines push to it. A slower node cannot overwrite a better champion — the push
re-checks the remote first.

## Handoff

When the laptop stops (closed, off, busy), Railway keeps going from the last champion on
GitHub. When the laptop starts again it pulls whatever Railway found and continues. Neither
needs to know about the other.

## Expected speed

Roughly 20–45 s per trial on this hardware versus several minutes on Railway. Overnight ≈
800–1,400 trials. The daemon prints a plateau notice after 50 trials without a promotion —
when you see it, more trials will not help; more sessions (the daily ladders) will.
