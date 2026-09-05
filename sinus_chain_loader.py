r"""
sinus_chain_loader.py
=====================
Glue between the per-day files in DATA_DIR/chain/ and sinus.py.

    from sinus_chain_loader import load_history
    spot_df, chain_df, flow_df = load_history(r'C:\sinus\data', max_sessions=500)
    pipeline.fit(spot_df, chain_df, flow_df)          # instead of (spot_df, None, None)

* spot_df  ts, spot            -- parity spot, 1 row per minute (no stocks plan needed)
* chain_df ts, strike, ...     -- every column the pipeline's chain contract names
                                  (gex, call_oi, put_oi, call_prem, put_prem, call_ask_prem,
                                  call_bid_prem, put_ask_prem, put_bid_prem, call_vol,
                                  put_vol, vanna, charm, volume) plus the extras
                                  (iv, gamma, delta, cumvol, signed premium, gex_signed,
                                  gex_source). Missing values are NaN: the pipeline flags
                                  them, it never drops them.
* flow_df  ts, premium, option_type, strike, side, trade_type, size, underlying_price
"""
from __future__ import annotations

import glob
import os
import re
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from polygon_chain_history import TZ, load_frame

CHAIN_COLS = ["ts", "strike", "gex", "call_oi", "put_oi", "call_prem", "put_prem",
              "call_ask_prem", "call_bid_prem", "put_ask_prem", "put_bid_prem",
              "call_vol", "put_vol", "vanna", "charm", "volume",
              "call_iv", "put_iv", "call_gamma", "put_gamma", "call_delta", "put_delta",
              "call_close", "put_close", "call_cumvol", "put_cumvol",
              "call_signed_v", "put_signed_v", "call_cum_signed", "put_cum_signed",
              "gex_signed", "spot", "gex_source"]
FLOW_COLS = ["ts", "premium", "option_type", "strike", "side", "trade_type", "size", "underlying_price",
             "n_trades", "price", "vwap", "iv", "delta", "gamma"]

_DATE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.")


def _days(data_dir: str, kind: str) -> List[date]:
    out = []
    for p in glob.glob(os.path.join(data_dir, "chain", f"{kind}_*.*")):
        if p.endswith(".tmp"):
            continue
        m = _DATE.search(os.path.basename(p))
        if m:
            out.append(date.fromisoformat(m.group(1)))
    return sorted(set(out))


def _file(data_dir: str, kind: str, d: date) -> Optional[str]:
    hits = [p for p in glob.glob(os.path.join(data_dir, "chain", f"{kind}_{d:%Y-%m-%d}.*")) if not p.endswith(".tmp")]
    return hits[0] if hits else None


def _tz(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(TZ) if s.dt.tz is not None else s.dt.tz_localize(TZ)


def load_history(data_dir: str, max_sessions: Optional[int] = None,
                 start: Optional[str] = None, end: Optional[str] = None,
                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = _days(data_dir, "chain")
    if start:
        days = [d for d in days if d >= date.fromisoformat(start)]
    if end:
        days = [d for d in days if d <= date.fromisoformat(end)]
    if max_sessions:
        days = days[-max_sessions:]
    if not days:
        raise FileNotFoundError(f"no chain_*.parquet/pkl under {data_dir}/chain — run polygon_chain_history.py first")

    spots, chains, flows = [], [], []
    n_oi = 0
    for d in days:
        cp, fp, sp = _file(data_dir, "chain", d), _file(data_dir, "flow", d), _file(data_dir, "spot", d)
        ch = load_frame(cp)
        ch["ts"] = _tz(ch["ts"])
        for col in CHAIN_COLS:
            if col not in ch.columns:
                ch[col] = np.nan if col != "gex_source" else "unknown"
        chains.append(ch[CHAIN_COLS])
        n_oi += int((ch["gex_source"] == "oi").any())
        if fp:
            fl = load_frame(fp)
            fl["ts"] = _tz(fl["ts"])
            for col in FLOW_COLS:
                if col not in fl.columns:
                    fl[col] = np.nan
            flows.append(fl[FLOW_COLS])
        if sp:
            s = load_frame(sp)
            s["ts"] = _tz(s["ts"])
            spots.append(s[["ts", "spot"]])
        else:  # snapshot-only day: spot from the chain rows
            s = ch.groupby("ts")["spot"].last().reset_index()
            spots.append(s)

    spot_df = pd.concat(spots, ignore_index=True).dropna().drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    chain_df = pd.concat(chains, ignore_index=True).sort_values(["ts", "strike"]).reset_index(drop=True)
    flow_df = (pd.concat(flows, ignore_index=True).sort_values("ts").reset_index(drop=True)
               if flows else pd.DataFrame(columns=FLOW_COLS))
    if verbose:
        print(f"[chain] {len(days)} sessions {days[0]} -> {days[-1]} · spot {len(spot_df):,} bars · "
              f"chain {len(chain_df):,} rows · flow {len(flow_df):,} prints · real-OI days: {n_oi}")
    return spot_df, chain_df, flow_df


def load_recent_for_live(data_dir: str, context_sessions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The last few sessions plus whatever the snapshot logger has written today."""
    spot_df, chain_df, flow_df = load_history(data_dir, max_sessions=context_sessions, verbose=False)
    today = pd.Timestamp.now(tz=TZ).date()
    sp = _file(data_dir, "snap", today)
    if sp and today not in _days(data_dir, "chain"):
        s = load_frame(sp)
        s["ts"] = _tz(s["ts_feed"])
        today_chain = pd.DataFrame({c: s[c] if c in s.columns else np.nan for c in CHAIN_COLS})
        today_chain["gex_source"] = "oi"
        chain_df = pd.concat([chain_df, today_chain], ignore_index=True)
        spot_df = pd.concat([spot_df, s.groupby("ts")["spot"].last().reset_index()], ignore_index=True)
    return spot_df, chain_df, flow_df


if __name__ == "__main__":
    import sys
    s, c, f = load_history(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SINUS_DATA", "/data"))
    print(c.tail(3).T)
