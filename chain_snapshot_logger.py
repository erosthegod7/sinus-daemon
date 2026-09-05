r"""
chain_snapshot_logger.py
========================
Polygon/Massive has NO historical open-interest endpoint. The only place OI, dealer greeks
and IV exist is the live chain snapshot. So from today forward we record it ourselves.

Every `--every` minutes during regular hours this pulls the 0DTE chain snapshot for the
underlying and appends one row per strike to DATA_DIR/chain/snap_YYYY-MM-DD.<parquet|pkl>.
At the close it folds the day into chain_YYYY-MM-DD with gex_source='oi', which the
loader prefers over the volume-proxy build for that date.

Starter plan note: the snapshot is 15-minute DELAYED. The logger stamps both the wall
clock (`ts`) and the delayed feed time (`ts_feed` = ts - 15 min) so training uses the
feed time and nothing leaks.

Run on the laptop (leave it in its own PowerShell window):
  python chain_snapshot_logger.py --data C:\sinus\data --every 5
Or once, ad hoc:
  python chain_snapshot_logger.py --data C:\sinus\data --once
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from polygon_chain_history import (BASE, MULT, TZ, Client, _path, load_frame, save_frame)

DELAY_MIN = int(os.environ.get("SINUS_FEED_DELAY_MIN", "15"))


def _snap_path(data_dir: str, d: date) -> str:
    return _path(data_dir, "snap", d)


def pull_snapshot(c: Client, underlying: str, d: date, window: float = 12.0) -> pd.DataFrame:
    rows = list(c.paged(f"{BASE}/v3/snapshot/options/{underlying}",
                        {"expiration_date": f"{d:%Y-%m-%d}", "limit": 250}))
    if not rows:
        return pd.DataFrame()
    recs = []
    spot = np.nan
    for r in rows:
        det, g, day = r.get("details", {}), r.get("greeks", {}) or {}, r.get("day", {}) or {}
        ua = r.get("underlying_asset", {}) or {}
        spot = ua.get("price", spot)
        recs.append({
            "strike": float(det.get("strike_price", np.nan)),
            "type": str(det.get("contract_type", ""))[:1].upper(),
            "oi": r.get("open_interest", np.nan),
            "iv": r.get("implied_volatility", np.nan),
            "delta": g.get("delta", np.nan), "gamma": g.get("gamma", np.nan),
            "vega": g.get("vega", np.nan), "theta": g.get("theta", np.nan),
            "day_vol": day.get("volume", np.nan), "day_vwap": day.get("vwap", np.nan),
            "day_close": day.get("close", np.nan),
            "last_px": (r.get("last_trade") or {}).get("price", np.nan),
            "bid": (r.get("last_quote") or {}).get("bid", np.nan),
            "ask": (r.get("last_quote") or {}).get("ask", np.nan),
        })
    df = pd.DataFrame(recs).dropna(subset=["strike"])
    if not np.isfinite(spot):
        # options-only key: the underlying block may be empty — parity on the ATM pair
        piv = df.pivot_table(index="strike", columns="type", values="last_px")
        if {"C", "P"} <= set(piv.columns):
            piv["par"] = piv["C"] - piv["P"] + piv.index
            piv["atm"] = (piv["C"] - piv["P"]).abs()
            spot = float(piv.sort_values("atm").head(3)["par"].median())
    df = df[(df["strike"] >= spot - window) & (df["strike"] <= spot + window)]
    now = pd.Timestamp.now(tz=TZ).floor("min")
    wide = df.pivot_table(index="strike", columns="type",
                          values=["oi", "iv", "delta", "gamma", "vega", "theta", "day_vol", "day_vwap", "day_close", "bid", "ask"])
    out = pd.DataFrame(index=wide.index)
    for f_, name in (("oi", "oi"), ("iv", "iv"), ("delta", "delta"), ("gamma", "gamma"), ("vega", "vega"),
                     ("theta", "theta"), ("day_vol", "dayvol"), ("day_vwap", "dayvwap"), ("day_close", "close"),
                     ("bid", "bid"), ("ask", "ask")):
        for t, side in (("C", "call"), ("P", "put")):
            out[f"{side}_{name}"] = wide[f_][t] if (f_, t) in wide.columns else np.nan
    out["spot"] = spot
    out["ts"] = now
    out["ts_feed"] = now - pd.Timedelta(minutes=DELAY_MIN)
    S2 = spot ** 2 * 0.01 * MULT
    out["gex"] = (out["call_gamma"].fillna(0) * out["call_oi"].fillna(0)
                  - out["put_gamma"].fillna(0) * out["put_oi"].fillna(0)) * S2
    out["gex_source"] = "oi"
    return out.reset_index()


def fold_day(data_dir: str, d: date) -> None:
    """Turn the day's snapshots into a chain_<date> file in the same shape as the history builder."""
    p = _snap_path(data_dir, d)
    if not os.path.exists(p):
        return
    s = load_frame(p).sort_values(["ts_feed", "strike"])
    chain = pd.DataFrame({
        "ts": s["ts_feed"], "strike": s["strike"],
        "call_oi": s["call_oi"], "put_oi": s["put_oi"],
        "call_iv": s["call_iv"], "put_iv": s["put_iv"],
        "call_gamma": s["call_gamma"], "put_gamma": s["put_gamma"],
        "call_delta": s["call_delta"], "put_delta": s["put_delta"],
        "call_close": s["call_close"], "put_close": s["put_close"],
        "spot": s["spot"], "gex": s["gex"], "gex_source": "oi",
    })
    # cumulative day volume -> per-snapshot volume and premium
    for side in ("call", "put"):
        cum = s.groupby("strike")[f"{side}_dayvol"].diff().fillna(s[f"{side}_dayvol"]).clip(lower=0)
        chain[f"{side}_vol"] = cum.to_numpy()
        chain[f"{side}_prem"] = (cum * s[f"{side}_dayvwap"].fillna(s[f"{side}_close"]).fillna(0) * MULT).to_numpy()
        chain[f"{side}_cumvol"] = s[f"{side}_dayvol"].to_numpy()
        for extra in ("ask_prem", "bid_prem", "signed_v", "cum_signed"):
            chain[f"{side}_{extra}"] = np.nan
    chain["gex_signed"] = np.nan
    chain["vanna"] = np.nan
    chain["charm"] = np.nan
    chain["volume"] = chain["call_vol"] + chain["put_vol"]
    save_frame(chain.reset_index(drop=True), _path(data_dir, "chain", d))
    print(f"folded {d}: {chain['ts'].nunique()} snaps x {chain['strike'].nunique()} strikes -> chain file (gex_source=oi)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="SPY")
    ap.add_argument("--data", default=os.environ.get("SINUS_DATA", "/data"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--window", type=float, default=12.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    c = Client(sleep=0.0)
    while True:
        now = pd.Timestamp.now(tz=TZ)
        d = now.date()
        hm = now.hour * 60 + now.minute
        if now.weekday() < 5 and 9 * 60 + 30 <= hm <= 16 * 60 + 5:
            try:
                snap = pull_snapshot(c, a.underlying, d, a.window)
                if not snap.empty:
                    p = _snap_path(a.data, d)
                    prev = load_frame(p) if os.path.exists(p) else pd.DataFrame()
                    save_frame(pd.concat([prev, snap], ignore_index=True), p)
                    atm = snap.iloc[(snap["strike"] - snap["spot"]).abs().argsort()[:1]]
                    print(f"[{now:%H:%M}] spot {snap['spot'].iloc[0]:.2f} · {len(snap)} strikes · "
                          f"ATM {float(atm['strike'].iloc[0]):.0f} OI c/p {float(atm['call_oi'].iloc[0]):.0f}/{float(atm['put_oi'].iloc[0]):.0f} · "
                          f"net gex {snap['gex'].sum() / 1e9:+.2f}B", flush=True)
                if hm >= 16 * 60:
                    fold_day(a.data, d)
            except Exception as e:
                print(f"[{now:%H:%M}] snapshot failed: {type(e).__name__}: {e}", flush=True)
        else:
            if a.once:
                print("market closed - nothing to snapshot")
        if a.once:
            break
        # sleep to the next multiple of --every minutes
        nxt = (now + pd.Timedelta(minutes=a.every)).floor(f"{a.every}min")
        time.sleep(max(5.0, (nxt - pd.Timestamp.now(tz=TZ)).total_seconds()))


if __name__ == "__main__":
    main()
