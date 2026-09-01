"""
sinus_gitstore.py
=================
GitHub as the durable, shared store for the SINUS search.

Why: the champion lived only on a Railway volume — one container away from gone. GitHub
gives version history, off-site durability, and lets two machines (laptop + Railway)
search the same space without duplicating work or clobbering each other.

What gets committed
    leaderboard_<node>.csv   every SAVE_EVERY trials — one file per machine, merged on read
    champion/                on every promotion, plus champion.json
    (weights are small: a few MB of LightGBM/CatBoost/torch state)

Concurrency model
    Each node owns its own leaderboard file, so two nodes never write the same file.
    The champion is shared; the node that promotes pulls first, re-checks that its candidate
    still beats whatever the OTHER node may have promoted meanwhile, then pushes. A stale
    push is rejected by git, retried once after a rebase, then given up — losing one
    promotion is far better than a corrupted repo.

Setup
    Env:  SINUS_GIT_REPO   e.g. erosthegod7/sinus-champion   (separate from the code repo)
          SINUS_GIT_TOKEN  a GitHub fine-grained token with Contents: read/write on that repo
          SINUS_NODE       a short name for this machine — 'laptop', 'railway'
    The repo can start empty; the first push creates the layout.

Everything degrades to local-only if git is unavailable or the env is unset — the search
must never stop because the backup failed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

import pandas as pd

SAVE_EVERY = int(os.environ.get("SINUS_GIT_SAVE_EVERY", "10"))


def _log(msg: str) -> None:
    print(f"[gitstore] {msg}", flush=True)


class GitStore:
    """Thin wrapper around a clone of the champion repo. All methods are safe to call when
    disabled — they return quietly, so callers don't need to branch on availability."""

    def __init__(self, work_dir: str, repo: Optional[str] = None, token: Optional[str] = None,
                 node: Optional[str] = None):
        self.repo = repo or os.environ.get("SINUS_GIT_REPO")
        self.token = token or os.environ.get("SINUS_GIT_TOKEN")
        self.node = (node or os.environ.get("SINUS_NODE") or "node").strip().replace(" ", "_")
        self.work_dir = work_dir
        self.clone_dir = os.path.join(work_dir, "_gitstore")
        self.enabled = bool(self.repo and self.token and shutil.which("git"))
        if not self.enabled:
            missing = [k for k, v in (("SINUS_GIT_REPO", self.repo), ("SINUS_GIT_TOKEN", self.token),
                                      ("git binary", shutil.which("git"))) if not v]
            _log(f"disabled — missing {missing}. Champion stays local only.")
            return
        self.url = f"https://x-access-token:{self.token}@github.com/{self.repo}.git"
        self._ensure_clone()

    # ------------------------------------------------------------------ #
    def _git(self, *args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        r = subprocess.run(["git", *args], cwd=self.clone_dir, capture_output=True, text=True)
        if check and r.returncode != 0 and not quiet:
            _log(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
        return r

    def _sync(self) -> None:
        """Bring the clone to the remote's main. Uses FETCH_HEAD so it works even on a clone
        created via `git init` (empty remote) where no remote-tracking refspec exists yet."""
        self._git("config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*", quiet=True)
        r = self._git("fetch", "origin", "main", quiet=True)
        if r.returncode == 0:
            self._git("reset", "--hard", "FETCH_HEAD", quiet=True)

    def _ensure_clone(self) -> None:
        if os.path.isdir(os.path.join(self.clone_dir, ".git")):
            self._sync()
            return
        os.makedirs(self.work_dir, exist_ok=True)
        r = subprocess.run(["git", "clone", "--depth", "50", self.url, self.clone_dir],
                           capture_output=True, text=True)
        if r.returncode != 0:
            if "empty repository" in r.stderr or "not found" in r.stderr.lower():
                # brand-new repo: init locally and create main on first push
                os.makedirs(self.clone_dir, exist_ok=True)
                subprocess.run(["git", "init", "-b", "main"], cwd=self.clone_dir, capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", self.url], cwd=self.clone_dir, capture_output=True)
                _log("initialised empty champion repo")
            else:
                _log(f"clone failed: {r.stderr.strip()[:300]} — disabling")
                self.enabled = False
                return
        self._git("config", "user.email", f"sinus-{self.node}@local", quiet=True)
        self._git("config", "user.name", f"sinus-{self.node}", quiet=True)
        _log(f"clone ready at {self.clone_dir} (node={self.node})")

    def _commit_push(self, message: str) -> bool:
        """Commit everything staged and push. On rejection, rebase once and retry."""
        self._git("add", "-A", quiet=True)
        st = self._git("status", "--porcelain", quiet=True)
        if not st.stdout.strip():
            return True                                   # nothing changed
        self._git("commit", "-m", message, quiet=True)
        for attempt in range(2):
            r = self._git("push", "-u", "origin", "main", quiet=True)
            if r.returncode == 0:
                return True
            if attempt == 0:
                self._git("pull", "--rebase", "origin", "main", quiet=True)
        _log(f"push failed after retry: {r.stderr.strip()[:200]}")
        return False

    # ------------------------------------------------------------------ #
    # Champion
    # ------------------------------------------------------------------ #
    def pull_champion(self) -> Optional[Dict[str, Any]]:
        """Fetch the shared champion into the local work_dir. Returns its metadata, or None."""
        if not self.enabled:
            return None
        self._sync()
        src = os.path.join(self.clone_dir, "champion")
        meta_p = os.path.join(src, "champion.json")
        if not os.path.exists(meta_p):
            return None
        dst = os.path.join(self.work_dir, "champion")
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        meta = json.load(open(meta_p))
        _log(f"pulled champion trial {meta.get('trial')} (test {meta.get('test_score', float('nan')):.4f}) "
             f"promoted by {meta.get('node', '?')}")
        return meta

    def remote_champion_score(self) -> Optional[float]:
        """Current champion's test score on GitHub, after a fresh fetch. Used to re-check a
        candidate right before pushing, in case the other node promoted meanwhile."""
        if not self.enabled:
            return None
        self._sync()
        p = os.path.join(self.clone_dir, "champion", "champion.json")
        if not os.path.exists(p):
            return None
        try:
            return float(json.load(open(p))["test_score"])
        except Exception:
            return None

    def push_champion(self, local_champion_dir: str, meta: Dict[str, Any], min_improvement: float = 0.0) -> bool:
        """Publish a newly promoted champion. Re-checks against the remote first so a slower
        node can't overwrite a better champion the other node just pushed."""
        if not self.enabled:
            return False
        remote = self.remote_champion_score()
        mine = float(meta["test_score"])
        if remote is not None and mine >= remote - min_improvement:
            _log(f"not pushing: remote champion {remote:.4f} is already at least as good as {mine:.4f}")
            return False
        dst = os.path.join(self.clone_dir, "champion")
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(local_champion_dir, dst)
        meta = {**meta, "node": self.node, "pushed_at": str(pd.Timestamp.now(tz="UTC"))}
        json.dump(meta, open(os.path.join(dst, "champion.json"), "w"), indent=2, default=float)
        # keep a dated lineage so a bad later champion can be rolled back
        lineage = os.path.join(self.clone_dir, "lineage")
        os.makedirs(lineage, exist_ok=True)
        json.dump(meta, open(os.path.join(lineage, f"champion_trial{meta['trial']}_{self.node}.json"), "w"),
                  indent=2, default=float)
        ok = self._commit_push(f"champion: trial {meta['trial']} on {self.node} · test {mine:.4f}")
        _log(("pushed" if ok else "FAILED to push") + f" champion trial {meta['trial']} (test {mine:.4f})")
        return ok

    # ------------------------------------------------------------------ #
    # Leaderboard
    # ------------------------------------------------------------------ #
    def push_leaderboard(self, board: pd.DataFrame, trial: int) -> bool:
        """Commit this node's leaderboard. Called by the daemon every SAVE_EVERY trials."""
        if not self.enabled:
            return False
        p = os.path.join(self.clone_dir, f"leaderboard_{self.node}.csv")
        board.to_csv(p, index=False)
        ok = self._commit_push(f"leaderboard: {self.node} through trial {trial} ({len(board)} rows)")
        if ok:
            _log(f"saved leaderboard to GitHub at trial {trial}")
        return ok

    def merged_leaderboard(self) -> pd.DataFrame:
        """All nodes' leaderboards concatenated, for a global view of the search."""
        if not self.enabled:
            return pd.DataFrame()
        self._sync()
        frames = []
        for f in os.listdir(self.clone_dir):
            if f.startswith("leaderboard_") and f.endswith(".csv"):
                try:
                    d = pd.read_csv(os.path.join(self.clone_dir, f))
                    d["node"] = f[len("leaderboard_"):-4]
                    frames.append(d)
                except Exception:
                    continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def status(work_dir: str) -> None:
    """Print the global search picture across every node."""
    gs = GitStore(work_dir)
    if not gs.enabled:
        print("gitstore disabled — set SINUS_GIT_REPO / SINUS_GIT_TOKEN")
        return
    b = gs.merged_leaderboard()
    print(f"trials across all nodes : {len(b)}")
    if len(b) and "node" in b:
        print(b.groupby("node").agg(trials=("trial", "size"),
                                    best=("score", "min")).to_string())
    m = gs.pull_champion()
    if m:
        print(f"champion : trial {m['trial']} on {m.get('node')} · test {m['test_score']:.4f} "
              f"· mae {m.get('test_mae', float('nan')):.4f} · acc {m.get('test_acc', float('nan')):.3f}")
    else:
        print("champion : none yet")
