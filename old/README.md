# old/

Nothing here is on the run path. Kept rather than deleted, per standing preference:
superseded files move aside, they don't get destroyed.

Moved 2026-09-05.

## patches/

One-shot migration scripts that rewrite other files in place. **All three have already
been applied** — the `.bak` files in `backups/` are their output, which is how we know.
Re-running them is not needed and `patch_mag.py` would fail its own `assert` if you did,
because the anchor text it looks for is already replaced.

| File | What it did |
|---|---|
| `sinus_fixes.py` | five daemon fixes on 2026-09-03 (Ctrl-C handling among them); wrote `*.prefix.bak` |
| `patch_daemon_chain.py` | wired chain/flow into the daemon and predict_live; wrote `*.bak` |
| `patch_mag.py` | made `fit_magnitude` conditional on `SINUS_FIT_MAG` |

## backups/

Mechanical copies written by the scripts above, not history worth reading. Gitignored.

## superseded/

| File | Why it moved |
|---|---|
| `score_predictions_patched.py` | its pre-11:00 eod rule is now the real implementation in `sinus_search.py`, covered by `tests/test_scoring.py`. It also carried a tz bug — `pd.Series(ts).values` drops the timezone and re-reads UTC wall-clock as Eastern, shifting every bar by the offset so the 11:00 cut masked the wrong rows. Fixed on the way in. |
| `push_champion_now.py` | **do not run this.** It copies the whole of `champion_v2` into the champion repo, and `champion_v2` used to contain `_gitstore/.git/config` with the GitHub PAT in the remote URL — pushing it would have published the token to a public repo. It also attached the leaderboard's best score to whatever weights happened to be on disk, which were anonymous. The normal promotion path in `sinus_daemon.py` does this correctly; nothing needs a manual push. |

## daemon_run.log

Run log from 2026-09-02, kept for reference.
