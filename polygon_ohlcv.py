#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# polygon_ohlcv.py
# ================
# Rebuild the 1-minute underlying file with REAL candles.
#
# spy_1min_parity.csv is timestamp + close only. That leaves eight features in the
# trainer as constant zeros: bar range, body, both wicks, volume average, volume
# z-score, session volume ratio, cumulative volume fraction. A constant column
# teaches a tree nothing, so those features were dead weight.
#
# Polygon's stock aggregates endpoint carries o/h/l/c/v plus vw (VWAP) and n (trade
# count). This pulls it month by month, pages through next_url, and merges onto the
# existing file.
#
# CRITICAL DESIGN CHOICE: the existing close is KEPT as `spot`.
# The option chain cache under SINUS_VOLUME/history was built by solving IV against
# those exact spot values. Replacing close with Polygon's would invalidate every
# cached IV and gamma. So Polygon supplies open/high/low/volume/vwap/trades, and the
# parity file keeps supplying close. Rows Polygon has that parity does not are added
# with Polygon's close, since there is nothing cached against them anyway.
#
# adjusted=false on purpose: adjusted=true splits volume across historical split
# events and returns fractional share counts (25669.549219). Raw as-traded is what
# a volume feature should see.
#
# Env
#   POLYGON_KEY     required
#   SINUS_CSV       existing parity file, used as the close reference
#   SINUS_SYMBOL    default SPY
#   SINUS_OUT       output path, default alongside SINUS_CSV as *_ohlcv.csv
#   SINUS_YEARS     how far back to attempt, default 2
#
# Run:  python polygon_ohlcv.py

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional

import pandas as pd
import requests

SYMBOL = os.environ.get("SINUS_SYMBOL", "SPY")
KEY = os.environ.get("POLYGON_KEY", "")
PARITY = os.environ.get("SINUS_CSV", r"C:\sinus\data\spy_1min_parity.csv")
YEARS = float(os.environ.get("SINUS_YEARS", "2"))
OUT = os.environ.get("SINUS_OUT") or os.path.splitext(PARITY)[0].replace("_parity", "") + "_ohlcv.csv"
TZ = "America/New_York"
BASE = "https://api.polygon.io"


def _log(m: str) -> None:
    print(f"[ohlcv] {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} {m}", flush=True)


def _fetch_range(start: str, end: str) -> List[dict]:
    """One date range, following next_url until exhausted. 50k bars per page."""
    url = (f"{BASE}/v2/aggs/ticker/{SYMBOL}/range/1/minute/{start}/{end}"
           f"?adjusted=false&sort=asc&limit=50000&apiKey={KEY}")
    rows: List[dict] = []
    while url:
        for attempt in range(5):
            r = requests.get(url, timeout=60)
            if r.status_code == 429:                       # rate limited, back off
                time.sleep(2 ** attempt * 5)
                continue
            break
        if r.status_code == 403:
            raise PermissionError(f"403 on {start}..{end} - plan does not cover this range")
        r.raise_for_status()
        j = r.json()
        rows.extend(j.get("results") or [])
        nxt = j.get("next_url")
        url = f"{nxt}&apiKey={KEY}" if nxt else None
    return rows


def pull(months_back: int) -> pd.DataFrame:
    """Month-sized windows so one failure loses a month, not the whole pull."""
    end = pd.Timestamp.now(tz=TZ).normalize()
    start = end - pd.DateOffset(months=months_back)
    edges = pd.date_range(start, end, freq="MS").tolist()
    edges = [start] + edges + [end]
    edges = sorted(set(pd.Timestamp(e).normalize() for e in edges))

    frames, earliest_ok = [], None
    for a, b in zip(edges[:-1], edges[1:]):
        s, e = a.strftime("%Y-%m-%d"), (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            rows = _fetch_range(s, e)
        except PermissionError as ex:
            _log(f"{s}: {ex} - skipping, plan window starts later")
            continue
        except Exception as ex:
            _log(f"{s}: FAILED {ex} - skipping")
            continue
        if not rows:
            _log(f"{s}..{e}: empty")
            continue
        earliest_ok = earliest_ok or s
        frames.append(pd.DataFrame(rows))
        _log(f"{s}..{e}: {len(rows):,} bars")

    if not frames:
        raise SystemExit("no bars returned at all - check POLYGON_KEY")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ)
    df = (df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                             "v": "volume", "vw": "vwap", "n": "trades"})
            .drop(columns=[c for c in ("t", "otc") if c in df.columns])
            .drop_duplicates("ts", keep="last")
            .sort_values("ts")
            .reset_index(drop=True))
    _log(f"pulled {len(df):,} bars, {df['ts'].dt.normalize().nunique()} sessions, "
         f"{df['ts'].min().date()} -> {df['ts'].max().date()}")
    return df


def merge(poly: pd.DataFrame, parity_path: str) -> pd.DataFrame:
    """Parity close wins where both exist, so the cached chain history stays valid."""
    if not os.path.exists(parity_path):
        _log(f"no parity file at {parity_path} - using Polygon close as spot")
        poly["spot"] = poly["close"]
        return poly

    p = pd.read_csv(parity_path)
    tsc = next((c for c in p.columns if c.lower() in ("ts", "timestamp", "time", "datetime", "date")), None)
    pxc = next((c for c in p.columns if c.lower() in ("close", "c", "price", "spot")), None)
    if not tsc or not pxc:
        raise SystemExit(f"cannot find timestamp/close in {parity_path}: {list(p.columns)}")
    ts = pd.to_datetime(p[tsc], errors="coerce")
    ts = ts.dt.tz_convert(TZ) if isinstance(ts.dtype, pd.DatetimeTZDtype) else \
        ts.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(TZ)
    par = pd.DataFrame({"ts": ts, "spot": pd.to_numeric(p[pxc], errors="coerce")}).dropna()
    par = par.drop_duplicates("ts", keep="last")

    out = poly.merge(par, on="ts", how="outer").sort_values("ts").reset_index(drop=True)
    matched = out["spot"].notna() & out["close"].notna()
    if matched.any():
        d = (out.loc[matched, "spot"] - out.loc[matched, "close"]).abs()
        _log(f"overlap {matched.sum():,} bars - close agreement: median {d.median():.4f}, "
             f"p99 {d.quantile(0.99):.4f}, max {d.max():.4f}")
    out["spot"] = out["spot"].fillna(out["close"])          # parity wins, Polygon fills gaps
    out = out.dropna(subset=["spot"]).reset_index(drop=True)
    return out


def main() -> int:
    if not KEY:
        _log("POLYGON_KEY is not set")
        return 2
    poly = pull(int(round(YEARS * 12)))
    out = merge(poly, PARITY)

    cols = ["ts", "spot", "open", "high", "low", "close", "volume", "vwap", "trades"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT, index=False)

    cov = {c: f"{out[c].notna().mean():.0%}" for c in ("open", "high", "low", "volume", "vwap", "trades")
           if c in out}
    _log(f"wrote {OUT}")
    _log(f"{len(out):,} rows, {out['ts'].dt.normalize().nunique()} sessions, coverage {cov}")
    _log("now set  SINUS_CSV=" + OUT + "  and rerun the trainer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
