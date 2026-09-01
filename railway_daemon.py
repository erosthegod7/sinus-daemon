"""
railway_daemon.py
=================
Entrypoint for the always-on SINUS search daemon on Railway.

Responsibilities, in order:
  1. Resolve price history — from the persistent volume if a CSV was uploaded, otherwise
     pulled from Polygon's stock aggregates and cached to the volume so later restarts
     are instant.
  2. Run `search_forever` against a work_dir on the volume, so the leaderboard and champion
     survive redeploys, crashes and container recycling.
  3. Log heartbeats so `railway logs` shows progress without attaching to the process.

Environment variables (set these in the Railway dashboard):
    POLYGON_KEY        required
    SINUS_VOLUME       volume mount path            default /data
    SINUS_CSV          filename of a price CSV on the volume, if you uploaded one
    SINUS_SYMBOL       default SPY
    SINUS_YEARS        years of history to pull from Polygon   default 2
    SINUS_TFT_EPOCHS   epochs per search trial       default 12
    SINUS_MIN_SESSIONS refuse to start below this    default 100

Why it refuses to start on thin data: a search daemon left running on 60 sessions will
happily produce a champion, and that champion will be luck. Failing loudly at boot is
better than burning a month of compute on a number that means nothing.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd

VOL = os.environ.get("SINUS_VOLUME", "/data")
SYMBOL = os.environ.get("SINUS_SYMBOL", "SPY")
YEARS = float(os.environ.get("SINUS_YEARS", "2"))
EPOCHS = int(os.environ.get("SINUS_TFT_EPOCHS", "12"))
MIN_SESSIONS = int(os.environ.get("SINUS_MIN_SESSIONS", "100"))
CACHE_PQ = os.path.join(VOL, f"{SYMBOL.lower()}_1min_cache.parquet")
CACHE_CSV = os.path.join(VOL, f"{SYMBOL.lower()}_1min_cache.csv.gz")


def _cache_write(df: pd.DataFrame) -> str:
    """Parquet if pyarrow is available, gzipped CSV otherwise. The cache is an optimisation,
    never a hard dependency — a missing parquet engine must not stop the daemon booting."""
    try:
        df.to_parquet(CACHE_PQ, index=False)
        return CACHE_PQ
    except Exception as e:
        _log(f"parquet unavailable ({type(e).__name__}); caching as gzipped CSV")
        df.to_csv(CACHE_CSV, index=False, compression="gzip")
        return CACHE_CSV


def _cache_read() -> Optional[pd.DataFrame]:
    for p, fn in ((CACHE_PQ, pd.read_parquet), (CACHE_CSV, pd.read_csv)):
        if os.path.exists(p):
            try:
                df = fn(p)
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
                return df
            except Exception as e:
                _log(f"cache {os.path.basename(p)} unreadable ({e}) — ignoring it")
    return None


def _log(msg: str) -> None:
    print(f"[{pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_polygon_minutes(symbol: str, years: float, api_key: str) -> pd.DataFrame:
    """Pull 1-minute bars from Polygon in monthly chunks.

    Requires the STOCKS asset class on the key. Options and stocks are billed separately at
    Polygon, so an options-only key returns 403 here — the error says so plainly rather than
    letting the daemon start on an empty frame.
    """
    from sinus import _get

    end = pd.Timestamp.now(tz="America/New_York").normalize()
    start = end - pd.Timedelta(days=int(365 * years))
    frames, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=30), end)
        try:
            j = _get(f"/v2/aggs/ticker/{symbol}/range/1/minute/{cur.date()}/{nxt.date()}",
                     {"adjusted": "true", "sort": "asc", "limit": 50000}, api_key)
        except Exception as e:
            if "403" in str(e) or "NOT_AUTHORIZED" in str(e).upper():
                raise RuntimeError(
                    "Polygon returned 403 for stock aggregates. Your key covers OPTIONS but not "
                    "STOCKS (they are separate subscriptions). Either add the stocks plan, or "
                    "upload your 1-min CSV to the Railway volume and set SINUS_CSV to its filename."
                ) from e
            _log(f"chunk {cur.date()} failed: {e}")
            cur = nxt
            continue
        res = j.get("results") or []
        if res:
            frames.append(pd.DataFrame(res)[["t", "c"]])
        _log(f"  {cur.date()} → {nxt.date()}: {len(res):,} bars")
        cur = nxt
        time.sleep(13)      # free stocks tier is 5 calls/min — stay under it
                                # be polite even on unlimited plans
    if not frames:
        raise RuntimeError("Polygon returned no bars at all — check the key and the symbol")
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return pd.DataFrame({"ts": ts, "spot": pd.to_numeric(df["c"], errors="coerce")})


def resolve_price_history() -> pd.DataFrame:
    """Volume cache → uploaded CSV → Polygon. Caches whatever it resolves for next boot."""
    import sinus

    cached = _cache_read()
    if cached is not None:
        _log(f"price history from cache: {len(cached):,} bars")
        return sinus._finalise(cached, verbose=True)

    csv = os.environ.get("SINUS_CSV")
    if csv:
        path = csv if os.path.isabs(csv) else os.path.join(VOL, csv)
        if os.path.exists(path):
            _log(f"loading uploaded CSV {path}")
            df = sinus.load_csv(path)
            _log(f"cached to {_cache_write(df)}")
            return df
        _log(f"SINUS_CSV={csv} set but {path} not found — falling through to Polygon")

    key = os.environ.get("POLYGON_KEY")
    if not key:
        raise RuntimeError("POLYGON_KEY is not set and no CSV was provided")
    _log(f"pulling {YEARS}y of {SYMBOL} 1-min from Polygon (first boot only)")
    raw = fetch_polygon_minutes(SYMBOL, YEARS, key)
    df = sinus._finalise(raw, verbose=True)
    _log(f"cached {len(df):,} bars to {_cache_write(df)}")
    return df


def main() -> int:
    _log("SINUS daemon starting")
    _log(f"volume={VOL} symbol={SYMBOL} epochs/trial={EPOCHS}")
    os.makedirs(VOL, exist_ok=True)

    try:
        spot_df = resolve_price_history()
    except Exception as e:
        _log(f"FATAL could not resolve price history: {e}")
        traceback.print_exc()
        return 1

    n_sess = spot_df["ts"].dt.normalize().nunique()
    _log(f"{len(spot_df):,} bars across {n_sess} sessions")
    if n_sess < MIN_SESSIONS:
        _log(f"FATAL only {n_sess} sessions, minimum is {MIN_SESSIONS}. A search on this little "
             f"data finds luck, not edge. Lower SINUS_MIN_SESSIONS only if you know why.")
        return 1

    import sinus_daemon as sd
    work = os.path.join(VOL, "champion")
    _log(f"work_dir {work} — leaderboard and champion persist here across restarts")

    while True:                                        # outer loop: survive an unexpected crash
        try:
            sd.search_forever(spot_df, work_dir=work, tft_epochs=EPOCHS)
            _log("search_forever returned (stop requested) — exiting")
            return 0
        except KeyboardInterrupt:
            _log("interrupted — exiting cleanly")
            return 0
        except Exception as e:
            _log(f"daemon crashed: {e}")
            traceback.print_exc()
            _log("restarting in 60s; the leaderboard is intact so at most one trial is lost")
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
