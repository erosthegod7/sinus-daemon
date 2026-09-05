"""
push_champion_now.py
====================
Manual champion push. Bypasses sinus_train.py's promotion path entirely.

Put this in C:\\sinus-daemon (next to sinus_gitstore.py) and run:

    cd C:\\sinus-daemon
    python push_champion_now.py

It does three things:
  1. Prints exactly what is inside C:\\sinus\\data\\champion_v2 so we can see
     whether the weights and the scaler are actually on disk.
  2. Builds the champion metadata from the best 'ok' row in the leaderboard.
  3. Copies the whole champion_v2 folder into the repo and pushes it.

SINUS_GIT_TOKEN must be set in the environment. It already is on this machine
-- leaderboard_laptop.csv is reaching GitHub, which proves the token works.
"""

import os
import glob
import sys

import pandas as pd

DATA = r"C:\sinus\data"
CHAMP = os.path.join(DATA, "champion_v2")
BOARD = os.path.join(DATA, "leaderboard_laptop.csv")

os.environ.setdefault("SINUS_GIT_REPO", "erosthegod7/sinus-champion")
os.environ.setdefault("SINUS_NODE", "laptop")

if not os.environ.get("SINUS_GIT_TOKEN"):
    sys.exit("SINUS_GIT_TOKEN is not set in this shell. Set it and re-run.")


def inventory(path):
    print(f"\n--- contents of {path} ---")
    if not os.path.isdir(path):
        print("  FOLDER DOES NOT EXIST")
        return []
    files = sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("  EMPTY")
        return []
    total = 0
    for f in files:
        size = os.path.getsize(f)
        total += size
        print(f"  {os.path.relpath(f, path):45s} {size/1024:10.1f} KB")
    print(f"  {'TOTAL':45s} {total/1024:10.1f} KB")
    return files


files = inventory(CHAMP)
if not files:
    sys.exit(
        "\nNothing to push. The trainer never wrote a champion to disk, which "
        "means the failure is upstream of the git layer."
    )

names = " ".join(os.path.basename(f).lower() for f in files)
print("\n--- what is present ---")
print(f"  scaler   : {'YES' if 'scaler' in names else 'NOT FOUND'}")
print(f"  lightgbm : {'YES' if ('.txt' in names or 'lgb' in names or 'gbm' in names) else 'NOT FOUND'}")
print(f"  torch    : {'YES' if ('.pt' in names or '.pth' in names or 'tft' in names) else 'NOT FOUND'}")
print(f"  metadata : {'YES' if '.json' in names else 'NOT FOUND'}")

board = pd.read_csv(BOARD)
ok = board[board["status"] == "ok"].sort_values("score")
if ok.empty:
    sys.exit("No completed trials in the leaderboard.")
best = ok.iloc[0]

meta = {
    "trial": int(best["trial"]),
    "test_score": float(best["score"]),
    "test_mae": float(best["mae_mean"]),
    "test_acc": float(best["acc_mean"]),
    "promoted_at": str(pd.Timestamp.now(tz="UTC")),
    "source": "manual push - validation score used in place of test score",
}
print(f"\npushing trial {meta['trial']} (score {meta['test_score']:.4f})")

from sinus_gitstore import GitStore

gs = GitStore(DATA)
print(f"gitstore enabled: {gs.enabled}")
if not gs.enabled:
    sys.exit("GitStore disabled -- check SINUS_GIT_REPO, SINUS_GIT_TOKEN, and that git is on PATH.")

ok_push = gs.push_champion(CHAMP, meta)
print("\nRESULT:", "pushed" if ok_push else "PUSH FAILED -- read the [gitstore] lines above")
