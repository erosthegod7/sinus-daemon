# SINUS search daemon — Railway deployment

Perpetual hyperparameter search. Runs forever, checkpoints every trial to a persistent
volume, promotes a new champion only when it beats the reigning one on a held-out test split.

## Deploy

1. Create a GitHub repo and push **all files in this folder** to it (root level, not nested).
2. Railway → New Project → Deploy from GitHub repo → pick it.
3. Add a **Volume**, mount path `/data`.
4. Set variables:

   | variable | value |
   |---|---|
   | `POLYGON_KEY` | your key |
   | `SINUS_VOLUME` | `/data` |
   | `SINUS_SYMBOL` | `SPY` |
   | `SINUS_YEARS` | `2` |
   | `SINUS_TFT_EPOCHS` | `12` |
   | `SINUS_MIN_SESSIONS` | `100` |

5. Deploy. Watch the logs.

## Price data

On first boot the daemon tries, in order:

1. a cache on the volume (instant after the first run)
2. a CSV you uploaded to the volume — set `SINUS_CSV` to its filename
3. Polygon stock aggregates

**Polygon bills stocks and options separately.** If your key is options-only, step 3 returns
403 and the daemon says so explicitly. In that case upload `spy_1min_2008_2021_cleaned.csv`
to the volume and set `SINUS_CSV=spy_1min_2008_2021_cleaned.csv`.

## What it produces

On the volume at `/data/champion/`:

* `leaderboard.csv` — every trial, its config and scores. Survives restarts.
* `champion/` — the current best model weights.
* `champion/champion.json` — which trial won, and its test-set score.

## Reading it from Colab

The volume is not Google Drive. To see results, either use `railway logs`, or add a small
HTTP endpoint later. The simplest path for now: watch the logs for the ★ NEW CHAMPION lines.

## Honest expectations

The search explores the CONFIGURATION space, which is effectively unbounded — so the daemon
always has work. It does NOT extract more signal from a fixed dataset than the dataset
contains. Expect fast improvement, then a plateau. The daemon prints a notice after 50 trials
without a promotion; when you see it, more compute is not the answer. More sessions are.

