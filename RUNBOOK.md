# SINUS laptop runbook

Everything below assumes `C:\sinus-daemon` with the venv at `.\.venv`.

## Keys

As of 2026-09-05 no launcher carries a secret: `run.ps1`, `run_train.bat`, `run_ohlcv.bat`
and `run_train.ps1` all read `POLYGON_KEY` / `SINUS_GIT_TOKEN` from the Windows user
environment (`_sinus_env.ps1` does the lookup for the PowerShell ones). Both values had
previously been pasted into launcher files and read by tooling, so rotating them is still
the right call — deferred by choice, not forgotten:

- GitHub PAT: https://github.com/settings/personal-access-tokens
- Polygon key: https://polygon.io/dashboard/keys

```powershell
[Environment]::SetEnvironmentVariable("POLYGON_KEY",     "<new key>",  "User")
[Environment]::SetEnvironmentVariable("SINUS_GIT_REPO",  "erosthegod7/sinus-champion", "User")
[Environment]::SetEnvironmentVariable("SINUS_GIT_TOKEN", "<new PAT>",  "User")
```

Open a new window afterwards; an existing one keeps its old copy.

The PAT needs **Contents: read and write** on `erosthegod7/sinus-champion` (the champion
store) and on `erosthegod7/sinus-daemon` (the code — pushes from the laptop need it; extended
2026-09-05).

## Install

Already done. `sinus_fixes.py` was applied on 2026-09-03 and now lives in
`old/patches/` — its five fixes are in the files on disk, and the `*.prefix.bak`
copies it wrote are in `old/backups/`. There is nothing to install; go straight
to Run.

## Run

```powershell
.\start.ps1
```

Preflight runs first and refuses to start on a FAIL. `-Quick` skips the feature
pipeline build; `-SkipCheck` skips preflight entirely.

## Stopping

One Ctrl-C asks for a clean stop — it now lands inside the TFT epoch loop and
inside LightGBM boosting, so it takes seconds, not two full model fits. A second
Ctrl-C kills the process. Both are safe.

## What was wrong

**The daemon was not running.** It crash-looped every 60 seconds on
`KeyError: 'spot'` and completed zero trials. The cause was in the old
`sinus.py`: `premium_flow_features` merged the grid's spot onto a chain frame
that already had a `spot` column, so pandas emitted `spot_x`/`spot_y` and the
next line asked for `spot`. You already fixed that at 23:12 on 9/2 — the fix is
in `sinus.py` on disk, the crashing process was just holding the old module in
memory. It needs nothing but a restart.

Five things were fixed by `sinus_fixes.py` (applied 2026-09-03, now in `old/patches/`):

1. **Ctrl-C did nothing during training.** The handler replaced Python's default
   with one that only set a flag read between trials, so `KeyboardInterrupt` was
   never raised and repeat presses did nothing either.

2. **One margin gated two decisions.** `min_improvement=0.002` decided both
   "is this worth a test refit" and "does this test win count". Scores cluster
   inside 0.005, so the second use threw away real winners. Trial 13 beat the
   champion on test by 0.00118 and needed 0.002. Now `screen_margin` (0.002)
   and `promote_margin` (0.0005) are separate, overridable via
   `SINUS_SCREEN_MARGIN` / `SINUS_PROMOTE_MARGIN`.

3. **The screen bar came from the whole board.** A candidate that failed
   promotion still lowered the bar it had failed to clear, locking its own genes
   out. Crossover re-found trial 13's config at trials 23 and 29; both were
   rejected without ever being evaluated. The bar now seeds from
   `champion.json`'s `val_score` and moves only on promotion.

4. **The session floor was checked too early.** `main()` counted sessions on the
   Polygon/CSV frame, then `SINUS_SPOT_FROM_CHAIN` replaced `spot_df` with the
   chain's parity spot. On 9/2 that was 5 sessions against a floor of 7, with no
   complaint. Now re-checked after substitution.

5. **Crash-looping looked like running.** Fixed 60s retry, forever. Now
   exponential backoff, and the daemon exits after five identical consecutive
   failures so you find out in the morning.

## After the first night

Trial 13's genes will come back — breeding found them three times in one night —
and under the new margin they promote. Expect the champion to change early.

Then watch the turnover rate. A 0.0005 promote margin on 28,500 test rows is
defensible, but every promotion spends a little of the test set. If the champion
starts changing every few trials on sub-0.001 gains, that is not the model
improving, it is the held-out split being used up. Raise
`SINUS_PROMOTE_MARGIN` back toward 0.002 when you see it.

The real ceiling is data. The plateau notice is honest: past it, more trials buy
nothing and more sessions buy everything.
