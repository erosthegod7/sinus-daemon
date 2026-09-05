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

## Training on the laptop

```powershell
.\run_train.ps1          # build option history if missing, then evolve forever
.\run_train.ps1 -Build   # history only
```

`run_train.ps1` suppresses sleep for the life of its window and tees everything to
`C:\sinus\data\train_<stamp>.log`. `run_train.bat` is the same run without either — fine at a
desk, not for overnight. The trainer is `sinus_train.py`; it owns the feature contract
(`FEATURE_SET`, currently `v3`), and every champion it promotes carries weights, `pipeline.pkl`
(the fitted scaler) and `champion.json` to `erosthegod7/sinus-champion`.

**v3 (2026-09-05).** Every earlier champion was scored on a leak: `hist_prior_ret` was the
*current* day's close-to-close return, joined onto every bar of that day. Ablating that one
column dropped eod direction accuracy from 0.94 to 0.33. Two volume features in the trainer were
also full-session aggregates. All three are fixed, `tests/test_causality.py` proves the whole
matrix causal, and nothing tagged `v2` is comparable to anything tagged `v3` or later. v3 also
added Black-Scholes vanna/charm (five features that were NaN) and a lag block — price returns at
nine lags and the change over six windows for sixteen option-book state variables — so the
trees see trajectory, not just the current bar. **v4** keeps that lag block out of the TFT's
window (`sinus.SEQUENCE_EXCLUDE`): the TFT already sees thirty bars, so lagged copies cost it
width, epoch time and RAM for nothing. Tabular matrix 226 columns; TFT window ~121.

A plateau where wildly different hyperparameters all score within a hair of each other, and
one horizon is far better than naive, is a leak until proven otherwise. The check-in prompt in
the shepherd session flags any horizon with direction accuracy above 0.80 or MAE below 0.35.

## Credentials

No launcher stores a key. `POLYGON_KEY`, `SINUS_GIT_REPO` and `SINUS_GIT_TOKEN` are read from
the Windows user environment on the laptop, and from Railway variables in the cloud. Set them
once:

```powershell
[Environment]::SetEnvironmentVariable("POLYGON_KEY",     "<key>", "User")
[Environment]::SetEnvironmentVariable("SINUS_GIT_TOKEN", "<PAT>", "User")
```

then open a new window — an existing one keeps its old copy. `_sinus_env.ps1` pulls them into
the PowerShell launchers; the `.bat` launchers inherit them directly.

## Tests

```bash
python tests/run_all.py           # everything, ~2 min (loads real bars and the chain cache)
python tests/run_all.py --fast    # unit tests only, no data load
```

| suite | covers |
|---|---|
| `test_scoring.py` | the pre-11:00 eod rule — mask, drop ratio, tz handling, fallback, degenerate inputs |
| `test_promotion_gate.py` | how the search seeds the bar it must beat; regression for the 2026-09-04 stall |
| `test_gitstore_guard.py` | `push_champion` refuses a work directory, so a clone's token can't be published |
| `test_serve_gate.py` | `serve()` refuses a champion from another feature set and says so; never silently physics-only |
| `test_pipeline_smoke.py` | end-to-end on real SPY bars: RTH filtering, bundle/timestamp alignment, scoring |
| `test_causality.py` | no feature knows the future: the trainer's full v3 matrix on its own feed, cut at three points; regression for the `hist_prior_ret` leak |

`preflight.py` is the other half of this — it checks the machine and the data rather than the
code. Run `python preflight.py --quick` before a long session.

## Honest expectations

The search explores the CONFIGURATION space, which is effectively unbounded — so the daemon
always has work. It does NOT extract more signal from a fixed dataset than the dataset
contains. Expect fast improvement, then a plateau. The daemon prints a notice after 50 trials
without a promotion; when you see it, more compute is not the answer. More sessions are.

