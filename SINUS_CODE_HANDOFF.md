# SINUS — Pipeline Handoff to Claude Code
**Written 2026-09-05, 10:15 ET. State below was verified live against GitHub and Railway, not assumed.**

You are picking up Project Sinus on Anthony's laptop (MSI Katana 15 HX, Windows 11).
Connect these folders: `C:\sinus-daemon` (code) and `C:\sinus\data` (champions, history cache).

Work the tasks in the order given. Task 0 gates everything else — do not skip it to get to the
code fixes, because it may change what the code fixes should be.

---

## 0. What the 2026-09-05 laptop session found and changed — READ THIS FIRST

Written by Claude Code on the laptop, with file access. Everything below was measured, not
inferred. Sections 1–5 are the original brief and are kept for context; where they conflict
with this section, this section is right.

**Task 0 answer: the options block is fully populated.** GEX, premium and IV features are 0% NaN
on the trainer's own feed (`C:\sinus\data\history`, 500 sessions, 2024-09-05 → 2026-09-02).
The "60 dead columns" hypothesis is dead. The plateau had a different cause.

**The plateau was a leak.** In `sinus.py::price_context_features`, `daily["prior_ret"]` was
missing its `.shift(1)`, so `hist_prior_ret` was the *current* day's close-to-close return
joined onto every bar of that day. With `hist_prior_close` and spot also in the matrix, the eod
target was arithmetic. Measured on champion trial 18: eod direction accuracy 0.94 → **0.33**
with that one column zeroed, MAE 0.12 → 0.99. On 5m/10m/20m the champion was *worse than
predicting zero*. Since the leak was identical for every config, all 66 completed trials
scored within 0.006 of each other — hyperparameters could not matter. It was also a TFT
static covariate. Two more, in `sinus_train.py::_candle_block`: `vol_ratio_session` and
`vol_cum_frac` divided by the day's mean/total volume. The engine's own `verify_causality`
missed all of this because it isolated a single day, which left every cross-day feature NaN
on both sides and therefore "equal".

**Every champion this system ever produced is invalid**, including the trial-36 one Railway
is still holding and the trial-18 v2 one in `sinus-champion/champion/`.

**Fixed, and proven:**
- `prior_ret` shifted; the two volume features made causal (expanding mean; cumulative volume
  against the *prior* session's total). `verify_causality` now keeps prior days.
- `tests/test_causality.py`: per-column, three cut points, on the trainer's full matrix and
  its own feed. Zero columns differ. Part of `tests/run_all.py`.
- `FEATURE_SET` bumped to `v3`. Board and champion archived to
  `C:\sinus\data\Superseded\2026-09-05_v2-leaky-features\`. Nothing v2 is comparable to v3.
- Black-Scholes vanna/charm computed at load (`sinus_train._fill_bs_greeks`) — five features
  that were NaN are live. `max_pain` stays dead: it needs open interest and no equation
  recovers that.
- Lag block (`sinus_train._lag_block`): price returns at 1/5/10/15/20/25/30/40/50m and the
  change over 5/10/15/20/30/50m for sixteen option-book state variables. 105 columns, causal.
- Flow `side` relabelled ask/bid (the builder wrote buy/sell, the engine keyed on ask/bid, so
  `block_signed_bias` was zero on all 500 sessions).
- v3 matrix: 202 raw + 24 `__isna` = 226 columns, 10 dead, all of them constants or
  OI-dependent. Train `X_seq` ≈ 3 GB at 500 sessions, lookback 30.
- `champion.json` now carries `scoring_version` and a real `n_features` (Task 4 both parts);
  `push_champion` refuses to compare across scoring versions.

**Railway (Task 2).** `sinus_daemon.serve()` now installs the trainer's feature contract, refuses
any champion whose `feature_set` ≠ this build's, and runs `sinus.predict_live` — live OHLCV
from Polygon for the context window, today's option book through the SAME builder the champion
was fitted on (`build_session_chain` + BS greeks), the champion's own scaler, all three experts,
physics fusion. Falls back to physics-only and states the reason. `tests/test_serve_gate.py`
covers the gate. Dockerfile ships `sinus_train.py` and friends. `SINUS_LIVE_CHAIN=build` is the
default; `cache` restores the old snapshot-logger path; `SINUS_USE_CHAIN=0` disables.

**Task 1 (git).** `C:\sinus-daemon` *was* a repo with `origin` set and zero local commits. Local
`main` is now based on `origin/main` (ref move only). Secrets are out of every launcher and
in the Windows user environment; `.gitignore` and `.gitattributes` added.

**Unattended runs.** `run_train.ps1` (sleep suppressed, everything logged to
`C:\sinus\data\train_<stamp>.log`). A 2-hour check-in runs in the Code session; its prompt
stops the trainer if any horizon's direction accuracy exceeds 0.80 or MAE drops below 0.35.

**Corrections to the brief:** prune percentile is 60, not 50; the trainer reads
`C:\sinus\data\history` (its own cache), not `chain/`; `n_features: 0` was `_promote` reading
attributes the scaler never had.

---

## 1. What this system is

A perpetual evolutionary search that trains SPY prediction champions on the laptop, publishes them
to GitHub, and serves predictions from a Railway service.

    laptop trainer (sinus_train.py)  ->  GitHub erosthegod7/sinus-champion  ->  Railway sinus-inbox  ->  chat

Horizons: `5m / 10m / 20m / 40m / 60m / eod`.
EOD is scored ONLY on samples timestamped before 11:00 ET. It is still predicted and displayed at
every window; it just stops counting toward a trial's rank. Scoring version string is `eod_pre11`.

Repos (both public):
- `erosthegod7/sinus-daemon` — engine code. `sinus.py` (184 KB) is the real engine.
- `erosthegod7/sinus-champion` — data store. `champion/`, `lineage/`, `inbox/`, `old/`, `leaderboard_laptop.csv`

---

## 2. Verified current state

**Trainer: running, healthy mechanically.**
- 110 trials in 8.4 hours, run started 2026-09-05 00:54 ET
- ~5.5 min median per trial, 53 of 110 pruned
- Last leaderboard push 09:17 ET (trial 109)
- Leaderboard restarted from zero as intended; `scoring_version` = `eod_pre11` on all rows
- EOD cut confirmed active: mean 6,375 EOD rows scored vs 22,575 dropped per trial

**The 9/4 "weights never reach GitHub" bug is FIXED.**
`champion/` now contains `champion.json`, `engine.json`, `pipeline.pkl`, `trees/`, `tft/`.
`lineage/` is populating.

**Current champion** (`champion/champion.json`):
```
trial 18 · val 0.48488 · test 0.49063 · test_mae 0.53895 · test_acc 0.69329
n_sessions 502 · feature_set "v2" · node laptop · promoted 2026-09-05 03:12:47 -04:00
n_features 0        <-- wrong, see Task 4
```

**THE ACTUAL PROBLEM — the search is flat.**
Best score 0.48488 was reached at trial 18 (03:12 ET). Trials 19–109 produced nothing better.
That is 91 trials and roughly six hours with zero improvement.

    trials 0-9     best 0.48602
    trials 10-19   best 0.48488   <-- last improvement
    trials 20-109  best 0.48488   (unchanged, nine straight blocks)

Trial 0 was already 0.4879. Total improvement across the entire run is ~0.6%. Direction accuracy is
pinned near 0.70 and will not move. Hyperparameters are being explored (the leaderboard shows real
spread in learning_rate, num_leaves, d_model, objective) and it still cannot find anything better.
That pattern points at the data, not the search.

**Railway: up, but serving a dead champion.**
- Project `sinus-inbox` id `ed580fa6-899d-4908-a8f7-67462b7b8999`
- Service `sinus-inbox` id `09b46267-580b-4df3-b5d3-226a5349eb53`, status SUCCESS
- Deployment `ed5ba618-9b74-425d-ad5e-822f2c4a3b36`, deployed 2026-09-04 02:36 UTC, never redeployed
- Domain `sinus-inbox-production.up.railway.app`, `POST /submit?name=<file>`
- `SINUS_INBOX_POLL=0` — it must stay this way. It must never run on its own and drain CPU.
- Startup log: `[gitstore] pulled champion trial 36 (test 0.4911) promoted by laptop`

Trial 36 is from the **old, pre-reset, price-only lineage**. The container has not restarted since,
so it still holds a model that was deliberately discarded. Both 9/4 extractions ran and archived
cleanly to `old/2026-09-04`, but they were scored by the wrong model.

**`inbox/` is clean** — only `.gitkeep`. Hygiene is correct.

**`sinus_train.py` and `run_train.bat` are in NO repo.** Confirmed: `sinus-daemon` contains only
Dockerfile, README.md, laptop_setup.md, railway.json, railway_daemon.py, requirements.txt, sinus.py,
sinus_daemon.py, sinus_gitstore.py, sinus_inbox.py, sinus_search.py. The trainer exists only on this
laptop. Nothing outside the machine can read the champion format it writes.

---

## 3. Tasks

### Task 0 — Diagnose whether the features are actually populated (DO THIS FIRST)

`champion/engine.json` lists ~150 feature names including a full options block: `gex_net_total`,
`gex_zero_level`, `gex_wall_above/below`, `gex_pos_centroid`, `gex_hhi`, `vanna_net_total`,
`charm_net_total`, `max_pain`, `net_prem_1m/5m/15m`, `otm_skew_*`, `block_notional_15m`,
`sweep_notional_15m`, `dark_pool_share`, `iv_call_atm`, `iv_put_atm`, `iv_skew_atm`, `straddle_pct`.

A feature being named is not proof there is data behind it. Two signals say there may not be:
- every one of those columns has an `__isna` companion column
- there are `has_gex_snapshot` and `has_flow_prints` presence switches
- Polygon (Options Starter, $30) provides **no historical open interest and no historical greeks**

Hypothesis to confirm or kill: the model is training on price and volume with ~60 permanently-NaN
columns attached, which would fully explain a plateau this hard and this early.

Do this:
1. Read `sinus_train.py` and find how the training matrix is built and cached (look under
   `C:\sinus\data\history`).
2. Build or load one training matrix and report **non-null percentage per feature**, grouped:
   time/price, history, GEX/greeks, flow/premium, block/sweep/dark pool, IV/straddle, volume.
3. Report how many of the 502 sessions have `has_gex_snapshot=1` and `has_flow_prints=1`.
4. Report the actual date range of the sessions.

**Report those numbers before writing any fix.** If the options block is mostly NaN, the correct
next move is repairing the data feed, not tuning the search — and more trials are wasted compute
until then. The `sinus_chain` package already exists for this (`polygon_chain_history.py`,
`chain_snapshot_logger.py`, `sinus_chain_loader.py`) and lives in `C:\sinus-daemon`.

### Task 1 — Get the trainer into version control

`C:\sinus-daemon` is **not currently a git clone**. Anthony has always pushed by browser upload.

Add `sinus_train.py` and `run_train.bat` to `erosthegod7/sinus-daemon`. Set up a proper remote so
future changes are one commit, not a browser upload. Do not force-push and do not remove existing
files — `sinus.py` is the engine both the trainer and the Railway service import.

Acceptance: both files visible in the GitHub repo, `git status` clean, `git log` shows the commit.

### Task 2 — Make Railway able to load the v2 champion

Two separate defects:

**2a. Stale champion.** The service cached trial 36 from the old lineage at startup on 9/4 and has
not pulled since. Redeploying forces a fresh `pull_champion()`.

**2b. Format mismatch, the more dangerous one.** In `sinus_gitstore.py`, `pull_champion()` copies
`champion/` wholesale and returns the metadata with **no feature-set check and no schema check**.
The current champion is `feature_set: "v2"`, written by `sinus_train.py` — code the Railway image
has never seen. So after it pulls, it may fail to load `pipeline.pkl` / `trees/` / `tft/` and fall
back to physics-only with every learned component at 0.00. That is the exact 9/4 symptom, and it
would fail quietly.

Do this:
1. Read `sinus_inbox.py` and establish exactly what champion format it expects to load.
2. Make it load the v2 artifacts, or write an explicit adapter.
3. Add a hard check in `pull_champion()`: if `feature_set` or `scoring_version` is not one this
   build understands, **log loudly and refuse** rather than silently degrading.
4. Redeploy. The service **must build from the Dockerfile, not Railpack** — Railpack's image has no
   git binary, which silently disables GitStore.
5. Verify in the deploy logs that it pulls trial 18 (or later) and that learned components are
   non-zero on a test prediction.

Acceptance: deploy log shows a v2 champion pulled, and a probe submission returns a prediction that
is demonstrably not physics-only.

### Task 3 — Wire `serve()` to read the extraction file

Known open gap, untouched since 9/3. `serve()` does not read the submitted extraction into the
feature window. Anthony's screenshot-derived GEX ladder, ATM straddle, IV, greeks and net/strike
premium are archived to `old/<date>/` and then ignored at prediction time.

This is the one path that gets real open interest and real greeks into the model, since Polygon
provides neither historically. Wire the extraction's fields into the same feature names the champion
was trained on. Mismatched names here will fail silently — verify by name, not by position.

Acceptance: a submitted extraction visibly changes the prediction versus the same call without it.

### Task 4 — Two smaller correctness items

- **Push gate.** `push_champion()` gates purely on raw `test_score`. It does not compare
  `scoring_version`. Scores under `eod_pre11` are not comparable to the old scale. If
  `railway_daemon.py` ever wakes as a second node under old scoring, it can hijack the champion with
  a meaningless score. Add a `scoring_version` equality check before any comparison.
- **`n_features: 0`** in `champion.json` is wrong — `engine.json` lists ~150. The new trainer isn't
  populating the field. Cosmetic, but it makes the metadata untrustworthy for exactly the diagnosis
  in Task 0.

---

## 4. Guardrails

- **Do not delete anything without a backup.** Standing instruction from Anthony: "let's be real
  careful with this deleting stuff. We worked really hard on building this." Superseded material
  gets moved, not removed.
- **Do not change scoring without wiping the leaderboard.** Any change to the scoring function makes
  existing rows incomparable. If you change it, wipe `C:\sinus\data\leaderboard_laptop.csv` and the
  copy in the champion repo, and bump `scoring_version`.
- **Leave `SINUS_INBOX_POLL=0`.** The Railway service runs on demand only, by explicit instruction.
- **Do not start a second searching node.** The laptop is the only trainer right now. See the push
  gate defect above.
- Existing local config: `SINUS_MAX_SESSIONS=500`, `SINUS_PRUNE_PERCENTILE=50`.
  Champions write to `C:\sinus\data\champion_v2`, option history caches under `C:\sinus\data\history`.
- Secrets (`SINUS_GIT_TOKEN`, `POLYGON_KEY`) are already set on Railway and on the laptop. Do not
  print them and do not commit them.
- The trainer is currently running. If you need to stop it, say so first — trials in flight are lost.

---

## 5. Report back with

1. Task 0 feature-population numbers, per group, with the session count and date range
2. Your read: is the plateau a data problem or a search problem?
3. What you changed, per task, with the commits
4. Railway deploy log confirming which champion it now holds
5. Anything you found that this brief got wrong
