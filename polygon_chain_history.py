r"""
polygon_chain_history.py
========================
Build the SINUS options-book training feed from the Massive (Polygon) Options API.

What the $29 Options Starter plan can give us historically (verified 2026-09-02):
  * 1-minute OHLCV + VWAP + trade count for EVERY expired/active contract, 2 years back
  * the contract index (strike, type, expiry) for every expiry, including expired ones
  * NOT: historical open interest (only in the live snapshot), NOT: trades/quotes ticks
  * live snapshot (OI, greeks, IV) is 15-minute DELAYED on Starter

So for history we rebuild the book from minute bars:
  spot        put-call parity on the ATM 0DTE pair, minute by minute  (S = C - P + K)
  iv / greeks Black-Scholes solved locally from each bar's close and the parity spot
  premium     vwap x volume x 100 per contract per minute  (= UW "strike premium" flow)
  side        tick rule (bar close vs previous close) -> ask / bid / mid
  gex         VOLUME-gamma, cumulative since the open, as the dealer-position proxy.
              For 0DTE this is close to the real thing: the day's OI IS mostly the day's
              volume. Column gex_source='volume' marks it. Days logged by
              chain_snapshot_logger.py carry real OI and gex_source='oi'.

Output per session (written to DATA_DIR/chain/):
  chain_YYYY-MM-DD.parquet   one row per (snapshot ts, strike)  -- the pipeline's chain_df
  flow_YYYY-MM-DD.parquet    one row per (minute, contract) with a print  -- flow_df
  spot_YYYY-MM-DD.parquet    parity spot per minute -- replaces the free stocks-tier pull

Usage (laptop, PowerShell):
  $env:MASSIVE_API_KEY="..."            # or POLYGON_API_KEY
  python polygon_chain_history.py --start 2024-09-03 --end 2026-09-02 --data C:\sinus\data
  python polygon_chain_history.py --days 30              # last 30 sessions only
  python polygon_chain_history.py --date 2026-09-02      # one day

Resumable: a session whose three files already exist is skipped. Ctrl-C any time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE = "https://api.polygon.io"          # api.massive.com is the same service
TZ = "America/New_York"
CLOSE_MIN = 16 * 60                       # 4:00 pm in minutes-from-midnight
OPEN_MIN = 9 * 60 + 30
MULT = 100.0                              # shares per contract

_HAS_PARQUET = False
try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except Exception:
    try:
        import fastparquet  # noqa: F401
        _HAS_PARQUET = True
    except Exception:
        _HAS_PARQUET = False


# --------------------------------------------------------------------------- #
# storage helpers (parquet if an engine exists, pickle otherwise)
# --------------------------------------------------------------------------- #
def _path(data_dir: str, kind: str, d: date) -> str:
    ext = "parquet" if _HAS_PARQUET else "pkl"
    return os.path.join(data_dir, "chain", f"{kind}_{d:%Y-%m-%d}.{ext}")


def save_frame(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    if path.endswith(".parquet"):
        df.to_parquet(tmp, index=False)
    else:
        df.to_pickle(tmp)
    os.replace(tmp, path)


def load_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def session_done(data_dir: str, d: date) -> bool:
    return all(os.path.exists(_path(data_dir, k, d)) for k in ("chain", "flow", "spot"))


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, api_key: Optional[str] = None, sleep: float = 0.05, verbose: bool = True):
        self.key = api_key or os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
        if not self.key:
            raise RuntimeError("set MASSIVE_API_KEY (or POLYGON_API_KEY)")
        if requests is None:
            raise RuntimeError("pip install requests")
        self.s = requests.Session()
        self.sleep = sleep
        self.verbose = verbose
        self.calls = 0

    def get(self, url: str, params: Optional[dict] = None, tries: int = 6) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.key
        for i in range(tries):
            self.calls += 1
            r = self.s.get(url, params=params, timeout=30)
            if r.status_code == 200:
                time.sleep(self.sleep)
                return r.json()
            if r.status_code == 429:               # rate limited: back off hard
                wait = 15 * (i + 1)
                if self.verbose:
                    print(f"  429 rate limit - sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (403, 401):
                raise PermissionError(f"{r.status_code} on {url}: {r.text[:200]}")
            if r.status_code >= 500:
                time.sleep(2 * (i + 1))
                continue
            raise RuntimeError(f"{r.status_code} on {url}: {r.text[:200]}")
        raise RuntimeError(f"gave up on {url}")

    def paged(self, url: str, params: dict) -> Iterable[dict]:
        j = self.get(url, params)
        while True:
            for row in j.get("results", []) or []:
                yield row
            nxt = j.get("next_url")
            if not nxt:
                break
            j = self.get(nxt)


# --------------------------------------------------------------------------- #
# Polygon pulls
# --------------------------------------------------------------------------- #
def list_contracts(c: Client, underlying: str, expiry: date) -> pd.DataFrame:
    """Every contract expiring on `expiry` (expired or live)."""
    rows = []
    for expired in ("true", "false"):
        rows += list(c.paged(f"{BASE}/v3/reference/options/contracts",
                             {"underlying_ticker": underlying, "expiration_date": f"{expiry:%Y-%m-%d}",
                              "expired": expired, "limit": 1000}))
    if not rows:
        return pd.DataFrame(columns=["ticker", "strike", "type"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"ticker": df["ticker"],
                        "strike": pd.to_numeric(df["strike_price"], errors="coerce"),
                        "type": df["contract_type"].str[0].str.upper()})
    return out.dropna().drop_duplicates("ticker").reset_index(drop=True)


def minute_bars(c: Client, ticker: str, d: date) -> pd.DataFrame:
    j = c.get(f"{BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{d:%Y-%m-%d}/{d:%Y-%m-%d}",
              {"adjusted": "true", "sort": "asc", "limit": 50000})
    res = j.get("results") or []
    if not res:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v", "vw", "n"])
    df = pd.DataFrame(res)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ)
    return df[["ts", "o", "h", "l", "c", "v", "vw", "n"]]


def daily_spot_range(c: Client, underlying: str, start: date, end: date) -> pd.DataFrame:
    """One call on the free STOCKS tier: daily OHLC, used only to pick the strike window.
    If the key has no stocks access at all, fall back to the previous session's parity spot."""
    try:
        j = c.get(f"{BASE}/v2/aggs/ticker/{underlying}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                  {"adjusted": "true", "sort": "asc", "limit": 50000})
    except PermissionError:
        return pd.DataFrame(columns=["date", "low", "high"])
    res = j.get("results") or []
    if not res:
        return pd.DataFrame(columns=["date", "low", "high"])
    df = pd.DataFrame(res)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ).dt.date
    return df[["date", "l", "h"]].rename(columns={"l": "low", "h": "high"})


# --------------------------------------------------------------------------- #
# math: parity spot, Black-Scholes IV and greeks
# --------------------------------------------------------------------------- #
def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, is_call):
    """Vectorised Black-Scholes, r=q=0 (fine for 0DTE)."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / sq
    d2 = d1 - sq
    call = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    put = K * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return np.where(is_call, call, put)


def bs_gamma(S, K, T, sigma):
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return _norm_pdf(d1) / (S * sq)


def bs_delta(S, K, T, sigma, is_call):
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)


def implied_vol(price, S, K, T, is_call, lo=0.01, hi=5.0, iters=40):
    """Vectorised bisection. Prices at or below intrinsic return NaN."""
    price, S, K, T = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (price, S, K, T))
    price, S, K, T = np.broadcast_arrays(price, S, K, T)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    intrinsic = np.where(is_call, np.maximum(S - K, 0), np.maximum(K - S, 0))
    ok = (price > intrinsic + 1e-4) & (T > 0) & (S > 0)
    lo_a = np.full(price.shape, lo)
    hi_a = np.full(price.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        pm = bs_price(S, K, T, mid, is_call)
        up = pm < price
        lo_a = np.where(up, mid, lo_a)
        hi_a = np.where(up, hi_a, mid)
    iv = 0.5 * (lo_a + hi_a)
    iv = np.array(iv, dtype=float)
    iv[~ok] = np.nan
    iv[iv >= hi - 1e-3] = np.nan            # hit the ceiling: garbage print
    return iv


def minutes_to_close(ts: pd.Series) -> np.ndarray:
    m = ts.dt.hour * 60 + ts.dt.minute
    return np.maximum(CLOSE_MIN - m.to_numpy(), 0.0)


def parity_spot(bars: pd.DataFrame, n_strikes: int = 3) -> pd.DataFrame:
    """bars: columns ts, strike, type, c. Returns ts, spot (one row per minute with a pair)."""
    calls = bars[bars["type"] == "C"][["ts", "strike", "c"]].rename(columns={"c": "call"})
    puts = bars[bars["type"] == "P"][["ts", "strike", "c"]].rename(columns={"c": "put"})
    pair = calls.merge(puts, on=["ts", "strike"], how="inner")
    if pair.empty:
        return pd.DataFrame(columns=["ts", "spot"])
    pair["parity"] = pair["call"] - pair["put"] + pair["strike"]
    pair["atm"] = (pair["call"] - pair["put"]).abs()
    pair = pair.sort_values(["ts", "atm"])
    top = pair.groupby("ts", sort=True).head(n_strikes)
    spot = top.groupby("ts")["parity"].median().rename("spot").reset_index()
    return spot


# --------------------------------------------------------------------------- #
# one session
# --------------------------------------------------------------------------- #
def build_session(c: Client, underlying: str, d: date, data_dir: str,
                  strike_window: float = 8.0, snap_minutes: int = 5,
                  spot_hint: Optional[Tuple[float, float]] = None, verbose: bool = True) -> Optional[dict]:
    contracts = list_contracts(c, underlying, d)
    if contracts.empty:
        if verbose:
            print(f"{d}: no contracts expiring (holiday / no 0DTE listing)")
        return None

    # strike window ------------------------------------------------------------------
    if spot_hint is None:
        # no stocks access and no prior session: probe a wide band first (cheap: one call
        # per contract, so widen only as far as needed)
        lo, hi = contracts["strike"].quantile(0.35), contracts["strike"].quantile(0.65)
    else:
        lo, hi = spot_hint
    lo, hi = lo - strike_window, hi + strike_window
    sel = contracts[(contracts["strike"] >= lo) & (contracts["strike"] <= hi)]
    sel = sel[(sel["strike"] % 1 == 0)]                    # whole-dollar strikes only
    if verbose:
        print(f"{d}: {len(sel)} contracts in [{lo:.0f}, {hi:.0f}] "
              f"({len(contracts)} listed)", flush=True)

    # minute bars per contract -----------------------------------------------------------
    frames = []
    for _, row in sel.iterrows():
        b = minute_bars(c, row["ticker"], d)
        if b.empty:
            continue
        b["strike"] = row["strike"]
        b["type"] = row["type"]
        b["ticker"] = row["ticker"]
        frames.append(b)
    if not frames:
        if verbose:
            print(f"{d}: no bars in window - skipped")
        return None
    bars = pd.concat(frames, ignore_index=True)
    bars = bars[(bars["ts"].dt.hour * 60 + bars["ts"].dt.minute >= OPEN_MIN) &
                (bars["ts"].dt.hour * 60 + bars["ts"].dt.minute < CLOSE_MIN)]
    bars = bars.sort_values(["ticker", "ts"]).reset_index(drop=True)

    # spot ---------------------------------------------------------------------------------
    spot = parity_spot(bars)
    if spot.empty:
        if verbose:
            print(f"{d}: no call/put pair traded in the same minute - skipped")
        return None
    grid = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min", tz=TZ)
    spot_min = (spot.set_index("ts")["spot"].reindex(grid).ffill().bfill()
                .rename("spot").rename_axis("ts").reset_index())

    # per-bar enrichment: iv, greeks, premium, side ------------------------------------------
    bars = bars.merge(spot_min, on="ts", how="left")
    T = minutes_to_close(bars["ts"]) / (365.0 * 24 * 60)
    is_call = (bars["type"] == "C").to_numpy()
    bars["iv"] = implied_vol(bars["c"].to_numpy(), bars["spot"].to_numpy(),
                             bars["strike"].to_numpy(), T, is_call)
    iv_f = bars.groupby("ticker")["iv"].transform(lambda s: s.ffill().bfill())
    bars["gamma"] = bs_gamma(bars["spot"], bars["strike"], T, iv_f.fillna(0.2))
    bars["delta"] = bs_delta(bars["spot"], bars["strike"], T, iv_f.fillna(0.2), is_call)
    bars["premium"] = bars["vw"] * bars["v"] * MULT
    prev = bars.groupby("ticker")["c"].shift(1)
    bars["side"] = np.select([bars["c"] > prev, bars["c"] < prev], ["ask", "bid"], default="mid")
    bars["side"] = np.where(prev.isna(), "mid", bars["side"])
    # dealer sign: customer buys at the ask -> dealer short -> dealer gamma negative for both
    # calls and puts. We keep the UW convention instead (calls +, puts -) so the numbers
    # line up with the heatmaps he reads; side-signed variants are separate columns.
    bars["signed_v"] = np.where(bars["side"] == "ask", bars["v"], np.where(bars["side"] == "bid", -bars["v"], 0.0))

    # flow_df ------------------------------------------------------------------------------
    flow = pd.DataFrame({
        "ts": bars["ts"], "strike": bars["strike"], "option_type": bars["type"],
        "premium": bars["premium"], "side": bars["side"], "size": bars["v"],
        "n_trades": bars["n"], "price": bars["c"], "vwap": bars["vw"],
        "iv": bars["iv"], "delta": bars["delta"], "gamma": bars["gamma"],
        "underlying_price": bars["spot"],
        "trade_type": np.where(bars["v"] >= 500, "block", "regular"),
    })

    # chain_df: snapshot grid (every snap_minutes) x strike ------------------------------------
    bars["snap"] = bars["ts"].dt.floor(f"{snap_minutes}min")
    strikes = np.sort(sel["strike"].unique())
    snaps = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq=f"{snap_minutes}min", tz=TZ)
    idx = pd.MultiIndex.from_product([snaps, strikes], names=["ts", "strike"])

    def agg(kind: str) -> pd.DataFrame:
        k = bars[bars["type"] == kind]
        g = k.groupby(["snap", "strike"])
        out = pd.DataFrame({
            f"{kind}_vol": g["v"].sum(),
            f"{kind}_prem": g["premium"].sum(),
            f"{kind}_ask_prem": g.apply(lambda x: x.loc[x["side"] == "ask", "premium"].sum(), include_groups=False),
            f"{kind}_bid_prem": g.apply(lambda x: x.loc[x["side"] == "bid", "premium"].sum(), include_groups=False),
            f"{kind}_signed_v": g["signed_v"].sum(),
            f"{kind}_iv": g["iv"].mean(),
            f"{kind}_gamma": g["gamma"].mean(),
            f"{kind}_delta": g["delta"].mean(),
            f"{kind}_close": g["c"].last(),
        })
        out.index.names = ["ts", "strike"]
        return out

    chain = pd.concat([agg("C"), agg("P")], axis=1).reindex(idx)
    for col in chain.columns:
        if col.endswith(("_vol", "_prem", "_ask_prem", "_bid_prem", "_signed_v")):
            chain[col] = chain[col].fillna(0.0)
    # carry the last known iv/gamma/close forward within the day so a quiet strike still has a book
    for col in [c_ for c_ in chain.columns if c_.endswith(("_iv", "_gamma", "_delta", "_close"))]:
        chain[col] = chain.groupby(level="strike")[col].ffill()
    chain = chain.rename(columns={"C_vol": "call_vol", "P_vol": "put_vol", "C_prem": "call_prem", "P_prem": "put_prem",
                                  "C_ask_prem": "call_ask_prem", "P_ask_prem": "put_ask_prem",
                                  "C_bid_prem": "call_bid_prem", "P_bid_prem": "put_bid_prem",
                                  "C_signed_v": "call_signed_v", "P_signed_v": "put_signed_v",
                                  "C_iv": "call_iv", "P_iv": "put_iv", "C_gamma": "call_gamma", "P_gamma": "put_gamma",
                                  "C_delta": "call_delta", "P_delta": "put_delta",
                                  "C_close": "call_close", "P_close": "put_close"}).reset_index()

    # cumulative-since-open volume as the OI stand-in, then gex in the UW convention
    chain = chain.sort_values(["strike", "ts"])
    chain["call_cumvol"] = chain.groupby("strike")["call_vol"].cumsum()
    chain["put_cumvol"] = chain.groupby("strike")["put_vol"].cumsum()
    chain["call_cum_signed"] = chain.groupby("strike")["call_signed_v"].cumsum()
    chain["put_cum_signed"] = chain.groupby("strike")["put_signed_v"].cumsum()
    chain = chain.merge(spot_min.assign(ts=spot_min["ts"].dt.floor(f"{snap_minutes}min"))
                        .groupby("ts")["spot"].last().reset_index(), on="ts", how="left")
    S2 = chain["spot"] ** 2 * 0.01 * MULT
    chain["gex"] = (chain["call_gamma"].fillna(0) * chain["call_cumvol"]
                    - chain["put_gamma"].fillna(0) * chain["put_cumvol"]) * S2
    chain["gex_signed"] = -(chain["call_gamma"].fillna(0) * chain["call_cum_signed"]
                            + chain["put_gamma"].fillna(0) * chain["put_cum_signed"]) * S2
    chain["call_oi"] = np.nan          # unknown historically: flagged, never faked
    chain["put_oi"] = np.nan
    chain["vanna"] = np.nan
    chain["charm"] = np.nan
    chain["volume"] = chain["call_vol"] + chain["put_vol"]
    chain["gex_source"] = "volume"
    chain = chain.sort_values(["ts", "strike"]).reset_index(drop=True)

    save_frame(chain, _path(data_dir, "chain", d))
    save_frame(flow, _path(data_dir, "flow", d))
    save_frame(spot_min, _path(data_dir, "spot", d))
    lo_s, hi_s = float(spot_min["spot"].min()), float(spot_min["spot"].max())
    if verbose:
        print(f"{d}: spot {lo_s:.2f}-{hi_s:.2f} · {len(flow):,} prints · "
              f"{chain['ts'].nunique()} snaps x {len(strikes)} strikes · {c.calls} calls so far", flush=True)
    return {"date": str(d), "spot_lo": lo_s, "spot_hi": hi_s, "prints": int(len(flow))}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def sessions_between(start: date, end: date) -> List[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="SPY")
    ap.add_argument("--data", default=os.environ.get("SINUS_DATA", "/data"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--days", type=int, help="last N sessions (ignored if --start given)")
    ap.add_argument("--date", help="single session YYYY-MM-DD")
    ap.add_argument("--window", type=float, default=8.0, help="strikes each side of the day's range")
    ap.add_argument("--snap", type=int, default=5, help="chain snapshot spacing in minutes")
    ap.add_argument("--sleep", type=float, default=0.05, help="seconds between calls")
    ap.add_argument("--force", action="store_true", help="rebuild sessions already on disk")
    a = ap.parse_args(argv)

    today = datetime.now().date()
    if a.date:
        days = [date.fromisoformat(a.date)]
    elif a.start:
        days = sessions_between(date.fromisoformat(a.start), date.fromisoformat(a.end) if a.end else today)
    else:
        n = a.days or 30
        days = sessions_between(today - timedelta(days=int(n * 1.6) + 5), today)[-n:]
    days = [d for d in days if d <= today]

    c = Client(sleep=a.sleep)
    rng = daily_spot_range(c, a.underlying, days[0] - timedelta(days=7), days[-1])
    rng = rng.set_index("date") if not rng.empty else rng
    log_path = os.path.join(a.data, "chain", "build_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    last_hint = None
    t0 = time.time()
    for i, d in enumerate(days):
        if session_done(a.data, d) and not a.force:
            sp = load_frame(_path(a.data, "spot", d))
            last_hint = (float(sp["spot"].min()), float(sp["spot"].max()))
            continue
        hint = None
        if not rng.empty and d in rng.index:
            hint = (float(rng.loc[d, "low"]), float(rng.loc[d, "high"]))
        elif last_hint is not None:
            hint = (last_hint[0] - 5, last_hint[1] + 5)
        try:
            info = build_session(c, a.underlying, d, a.data, a.window, a.snap, hint)
        except KeyboardInterrupt:
            print("\nstopped - rerun to resume")
            return
        except Exception as e:  # keep going; one bad day must not kill the run
            print(f"{d}: FAILED {type(e).__name__}: {e}", flush=True)
            info = {"date": str(d), "error": f"{type(e).__name__}: {e}"}
        if info:
            with open(log_path, "a") as f:
                f.write(json.dumps(info) + "\n")
            if "spot_lo" in info:
                last_hint = (info["spot_lo"], info["spot_hi"])
        done = i + 1
        rate = (time.time() - t0) / max(done, 1)
        print(f"   [{done}/{len(days)}] ~{rate * (len(days) - done) / 60:.0f} min left", flush=True)
    print(f"done · {c.calls} API calls · {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
