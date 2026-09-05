# SINUS champion training v2 — options book + evolution + magnitude meter (2026-09-02)

## What the $29 Massive/Polygon Options Starter plan gives us
Verified against https://massive.com/docs/rest/options/aggregates/custom-bars.md and
https://massive.com/docs/rest/options/llms.txt

| Data | Starter | Notes |
|---|---|---|
| 1-min OHLCV + VWAP + trade count per contract | **2 years** | every strike, every expiry, expired included |
| Contract index (strike/type/expiry, expired too) | yes | `/v3/reference/options/contracts` |
| Live chain snapshot: OI, greeks, IV, day volume | yes, **15-min delayed** | the only place OI exists |
| Historical open interest | **no plan has it** | must be logged by us going forward |
| Tick trades / quotes (bid-ask side) | no ($79 / $199 tiers) | side is approximated by tick rule |
| SPY stock bars | no (options-only key) | spot rebuilt by put-call parity |

So the training feed is rebuilt from minute bars: parity spot, local Black-Scholes IV and
greeks, premium = vwap × volume × 100, side by tick rule, and **volume-gamma** as the dealer
position proxy (for 0DTE the day's OI is mostly the day's volume). Real OI days come from
the snapshot logger and are marked `gex_source='oi'`.

## Files (drop all six into C:\sinus-daemon)
| file | job |
|---|---|
| `polygon_chain_history.py` | builds `chain_/flow_/spot_<date>` files, 2 years back, resumable |
| `chain_snapshot_logger.py` | records the real 0DTE chain (OI/greeks/IV) every 5 min from today on |
| `sinus_chain_loader.py` | assembles the files into spot_df / chain_df / flow_df for sinus.py |
| `sinus_evolve.py` | **breed/cull generational search** — cull 70% every 25 min, survivors stay, refill with mutated clones + crossovers; horizon-weighted scoring |
| `sinus_magnitude.py` | **the magnitude meter** — MAG▲ 0–10, MAG▼ 0–10, NET −10..+10, 0.1 steps, fitted on the champion's features |
| `patch_daemon_chain.py` | wired the feed, the evolve loop and the meter into sinus_daemon / railway_daemon / predict_live — **applied 2026-09-02, now in `old/patches/`** |

## Steps (laptop, PowerShell, from C:\sinus-daemon)
```powershell
$env:MASSIVE_API_KEY = "<your key>"        # same as POLYGON_API_KEY
$env:SINUS_DATA      = "C:\sinus\data"

# 1. backfill — start with 60 sessions to prove it, then go 2 years
python polygon_chain_history.py --days 60
python polygon_chain_history.py --start 2024-09-03      # ~45 calls/session, ~2-3 h total

# 2. wiring — ALREADY APPLIED 2026-09-02, script is in old/patches/. Skip this.

# 3. restart the search — it prints "[chain] feed loaded: ... rows" on boot
python railway_daemon.py

# 4. in a SECOND window, during market hours, keep this running
python chain_snapshot_logger.py --every 5
```
Ctrl-C the builder any time; it resumes. Set `SINUS_USE_CHAIN=0` to get the old price-only
daemon back. Set `SINUS_SPOT_FROM_CHAIN=0` to keep using the Polygon stock crawl for spot.

## Evolution knobs (env vars)
| var | default | meaning |
|---|---|---|
| `SINUS_MODE` | `evolve` | `random` = old search loop |
| `SINUS_POP` | 20 | generation size |
| `SINUS_CULL` | 0.70 | fraction cut each generation |
| `SINUS_GEN_MINUTES` | 25 | a generation ends when everyone is scored OR this many minutes pass — unscored members carry over, nobody is cut unscored |

Refill mix: ⅓ small mutations (one knob nudged), ⅓ large (3–5 knobs, wide), ⅓ crossovers
(each knob from one of two survivor parents). The current champion is always seeded as
member zero. The test-split promotion gate is unchanged — breeding never skips it.

## Scoring weights
`5m 0.30 · 10m 0.30 · 15m 0.25 · 30m 0.25 · 1h 0.08 · eod 0.02`, renormalised over what
exists. Score per horizon = MAE × (1.5 − direction accuracy), so a horizon that calls the
direction right is rewarded on top of being close. **10m is not a horizon in sinus.py yet**
(HORIZONS = 5m/15m/30m/1h/eod) — adding it is a one-line HORIZONS edit plus a Phase 1 target;
the weights pick it up automatically when it appears.

## The magnitude meter
Two dials, printed on every predict_live and stored in `res['_magnitude']`:
```
MAG ▲7.8 [████████··]   ▼2.1 [██········]   NET +5.7   ≈ +$1.90 / -$0.35 in 30m   [SITTING ON A MOVE]
```
Scale is seismic: R = 30-min excursion ÷ baseline (median excursion at this time of day,
last 20 sessions). M = 10·(1 − 2^−R). R=1 → 5.0 (normal), R=2 → 7.5, R=3 → 8.75, R=4 → 9.4.
The heads are quantile regressors (q=0.75) — they answer *how far it could go*, not the
average. Scored on held-out days: MAE vs the naive "always 5.0", plus big-move recall and
precision (true ≥7.0 called ≥6.0), plus net-direction accuracy on decisive windows.
Metrics land in `champion/magnitude.json`. Tag thresholds: ≥6.0 elevated, ≥7.5 SITTING ON A MOVE.

## What changes in the champion
The leaderboard restarts from zero because the feature space changed (152 price features →
price + book). The first champion with the book will be scored on the same held-out test
split rule as before, so the comparison is honest: **old champion test score vs new champion
test score** is the number that says whether the book helps. Expect the first 20–30 trials
to be worse (more features, same data), then better if the book carries signal.

## The move-predictor target
`flow_df` and `chain_df` now exist minute-by-minute back two years. The "big move in the
next 20 minutes" label (`|20-min forward range| > 2 × prior 20-min range`) can be built
from `spot_*` and joined to `flow_*` — that is the next file, once the feed is on disk.

## Known limits, stated plainly
* Starter snapshots are 15 minutes delayed. Training uses `ts_feed` so nothing leaks;
  the live call is a 15-minute-old book. Advanced ($199) is the real-time fix, not UW.
* Tick-rule side is a guess. Ask/bid premium columns are labelled as such.
* Volume-gamma ≠ OI-gamma on non-0DTE expiries; only the 0DTE chain is pulled, on purpose.
* Flow at 1-min × contract is big (~8k prints/day). `SINUS_MAX_SESSIONS` bounds it; if the
  TFT tensor OOMs again, drop it to 300 first.
