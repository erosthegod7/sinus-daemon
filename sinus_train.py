#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sinus_train.py
# ==============
# Fresh training program for SINUS. Builds the FULL feature history the pipeline was
# designed for and then evolves champions on it indefinitely.
#
# What the old daemon trained on:  spot_df = (ts, spot). Nothing else.
# What this trains on:
#   spot_df   1-min OHLCV                                   -> candles, volume, realized vol
#   chain_df  per-minute per-strike option ladder           -> GEX (spot/dir/vol), skew, charm,
#             call/put premium, call/put volume, gamma         OTM accel, strike premiums
#   flow_df   per-minute premium prints with side/type      -> net premium, block/sweep bias
#   extras    ATM IV call/put, IV skew, straddle %, avg vol -> appended to the feature matrix
#
# Horizons: 5m 10m 20m 40m 60m eod   (replaces 5/15/30/1h/eod)
#
# HONEST LIMITS, stated once here so nobody rediscovers them the hard way:
#   * Polygon does not publish historical open interest or greeks. Historical GEX here is
#     VOLUME-weighted with Black-Scholes gamma computed from solved IV. call_oi/put_oi are
#     NaN for history and fill in only from live captures going forward. The docstring on
#     fetch_historical_ladder in sinus.py says the same thing.
#   * Historical flow is approximated from minute bars: side = uptick/downtick of the
#     contract, type = sweep if volume spikes vs its own rolling median else block. This is
#     a proxy for the tape, not the tape.
#   * Everything is cached per session under SINUS_VOLUME/history so Polygon is hit once.
#
# Env
#   POLYGON_KEY          required for chain/flow history
#   SINUS_CSV            1-min SPY OHLCV csv (stocks plan not needed if this is set)
#   SINUS_VOLUME         default /data
#   SINUS_BAND           strikes either side of open/close to pull, default 15
#   SINUS_MAX_SESSIONS   cap sessions used, default 500
#   SINUS_API_SLEEP      seconds between Polygon calls, default 0.0
#   SINUS_GIT_REPO / SINUS_GIT_TOKEN   champion store, same as before
#
# Run:   python sinus_train.py            (build history if missing, then evolve forever)
#        python sinus_train.py --build    (history only, then exit)

from __future__ import annotations

import argparse
import gzip
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sinus
from sinus import _bs_gamma, _get, _paged, _key, _finalise, TZ

# --------------------------------------------------------------------------- #
# Horizons. Patch the module constant AND the config so every consumer agrees.
# --------------------------------------------------------------------------- #
HORIZONS_NEW: Tuple[str, ...] = ("5m", "10m", "20m", "40m", "60m", "eod")
HORIZON_MINUTES: Dict[str, Optional[int]] = {"5m": 5, "10m": 10, "20m": 20, "40m": 40, "60m": 60, "eod": None}


def _override_init(cls, **overrides) -> None:
    """Wrap a dataclass __init__ so instances get new defaults for the named fields unless
    the caller passed them explicitly. Patching the class (not a name binding) means every
    construction path picks it up: EngineConfig() in sinus_search, GateConfig via
    FusionConfig's default_factory, from_json, all of it."""
    if getattr(cls, "_sinus_train_patched", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        for k, v in overrides.items():
            if k not in kw:
                setattr(self, k, v() if callable(v) else v)

    cls.__init__ = __init__
    cls._sinus_train_patched = True


# Fusion priors are (lgb, cat, tft, physics), interpolated from the original five so the
# shape of the schedule - trees dominate short, TFT mid, physics at the close - is kept.
GATE_PRIORS = {
    "5m":  (0.40, 0.35, 0.15, 0.10),
    "10m": (0.39, 0.34, 0.17, 0.10),
    "20m": (0.30, 0.25, 0.33, 0.12),
    "40m": (0.18, 0.14, 0.51, 0.17),
    "60m": (0.15, 0.12, 0.53, 0.20),
    "eod": (0.15, 0.10, 0.30, 0.45),
}
NO_ENTRY_LAST = {"5m": 6.0, "10m": 11.0, "20m": 21.0, "40m": 41.0, "60m": 61.0, "eod": 12.0}


def install_horizons() -> None:
    """Rebind every horizon-keyed default in the engine to the new set. Idempotent."""
    import sinus_daemon, sinus_search
    for mod in (sinus, sinus_daemon, sinus_search):
        mod.HORIZONS = HORIZONS_NEW
    sinus.HORIZON_MINUTES = {k: (float(v) if v is not None else None) for k, v in HORIZON_MINUTES.items()}

    _override_init(sinus.PipelineConfig, horizons=lambda: dict(HORIZON_MINUTES))
    _override_init(sinus.EngineConfig,
                   horizons=HORIZONS_NEW,
                   tree_primary_horizons=("5m", "10m", "20m"),
                   deep_primary_horizons=("40m", "60m", "eod"))
    _override_init(sinus.GateConfig, priors=lambda: dict(GATE_PRIORS))
    _override_init(sinus.RiskConfig, no_entry_last_minutes=lambda: dict(NO_ENTRY_LAST))

    # plain function with a default arg bound at def time
    sp = sinus_search.score_predictions
    sp.__defaults__ = tuple(HORIZONS_NEW if d is sinus_search.HORIZONS or isinstance(d, tuple) else d
                            for d in (sp.__defaults__ or ()))


VOL = os.environ.get("SINUS_VOLUME", "/data")
HIST = os.path.join(VOL, "history")
BAND = float(os.environ.get("SINUS_BAND", "15"))
MAX_SESSIONS = int(os.environ.get("SINUS_MAX_SESSIONS", "500"))
API_SLEEP = float(os.environ.get("SINUS_API_SLEEP", "0"))
MIN_PER_YEAR = 252 * 390
RISK_FREE = 0.045


def _log(msg: str) -> None:
    print(f"[train] {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. SPOT: keep the whole candle, not just the close
# --------------------------------------------------------------------------- #
_TS = ("ts", "timestamp", "time", "datetime", "date", "t")
_O, _H, _L, _C, _V = ("open", "o"), ("high", "h"), ("low", "l"), ("close", "c", "price", "spot"), ("volume", "v", "vol")


def _pick(cols, names) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _parse_ts(col: pd.Series, tz: Optional[str] = None) -> pd.Series:
    """Timestamps -> tz-aware in TZ, whatever form they arrive in.

    A file spanning a DST boundary carries BOTH -04:00 and -05:00 offsets, and
    pd.to_datetime refuses mixed offsets unless told to normalise. Parsing with
    utc=True handles that; naive strings still need localising to Eastern instead,
    because assuming UTC on an Eastern file shifts every session by four hours and
    silently ruins minutes-to-close.
    """
    raw = col.astype(str)
    aware = raw.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True, na=False).any()
    if aware:
        return pd.to_datetime(raw, errors="coerce", utc=True).dt.tz_convert(TZ)
    ts = pd.to_datetime(raw, errors="coerce")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        return ts.dt.tz_convert(TZ)
    return ts.dt.tz_localize(tz or TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(TZ)


def load_ohlcv(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    """1-min OHLCV -> (ts, spot, open, high, low, volume). spot == close so the existing
    pipeline keeps working unchanged; the extra columns feed the candle/volume block."""
    df = pd.read_csv(path)
    tsc, cc = _pick(df.columns, _TS), _pick(df.columns, _C)
    if not tsc or not cc:
        raise ValueError(f"{path}: need a timestamp and a close column, got {list(df.columns)}")
    ts = _parse_ts(df[tsc], tz)
    out = pd.DataFrame({"ts": ts, "spot": pd.to_numeric(df[cc], errors="coerce")})
    for name, cands in (("open", _O), ("high", _H), ("low", _L), ("volume", _V)):
        c = _pick(df.columns, cands)
        out[name] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
    out = _finalise(out, verbose=True)
    _log(f"spot: {len(out):,} bars, volume present={out['volume'].notna().mean():.0%}")
    return out


# --------------------------------------------------------------------------- #
# 2. CHAIN + FLOW history from Polygon option minute bars
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S: float, K: float, T: float, sigma: float, call: bool, r: float = RISK_FREE) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def solve_iv(price: float, S: float, K: float, T: float, call: bool) -> float:
    """Bisection on BS. Returns NaN when the price is below intrinsic or T is degenerate.
    Bounded 1%..500% - a 0DTE ATM at the close can legitimately print several hundred percent."""
    if not (price > 0 and S > 0 and K > 0 and T > 0):
        return float("nan")
    intrinsic = max(0.0, (S - K) if call else (K - S))
    if price <= intrinsic + 1e-6:
        return float("nan")
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _bs_price(S, K, T, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _otick(underlying: str, exp: str, kind: str, strike: float) -> str:
    return f"O:{underlying}{pd.Timestamp(exp).strftime('%y%m%d')}{kind}{int(round(strike * 1000)):08d}"


def _contracts_for(underlying: str, session: str, key: str) -> Tuple[str, List[dict]]:
    """Contracts expiring ON the session (0DTE). Falls back to the nearest later expiry."""
    res = _paged("/v3/reference/options/contracts",
                 {"underlying_ticker": underlying, "expiration_date": session, "as_of": session,
                  "limit": 1000}, key)
    if res:
        return session, res
    res = _paged("/v3/reference/options/contracts",
                 {"underlying_ticker": underlying, "expiration_date.gte": session, "as_of": session,
                  "limit": 1000, "sort": "expiration_date", "order": "asc"}, key)
    if not res:
        return session, []
    exp = min(r["expiration_date"] for r in res)
    return exp, [r for r in res if r["expiration_date"] == exp]


def _minute_bars(ticker: str, session: str, key: str, tries: int = 5) -> Optional[pd.DataFrame]:
    """One contract's minute bars, with its own retry.

    sinus._get uses timeout=30 and does not retry, so a single slow response from
    Polygon aborted the entire history build. A run that walks 500 sessions x 60
    contracts makes ~30,000 requests; at that volume a transient timeout is a
    certainty, not an edge case. Connect timeout stays short, read timeout is
    generous, and 429 backs off rather than hammering.
    """
    import requests
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{session}/{session}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    j = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=(10, 120))
            if r.status_code == 429:
                time.sleep(min(60, 2 ** attempt * 5))
                continue
            if r.status_code == 404:
                return None                      # contract never traded, not an error
            r.raise_for_status()
            j = r.json()
            break
        except Exception as e:
            if attempt == tries - 1:
                _log(f"  {ticker}: giving up after {tries} tries ({type(e).__name__})")
                return None
            time.sleep(min(30, 2 ** attempt * 2))
    rows = (j or {}).get("results") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ)
    return df[["ts", "o", "h", "l", "c", "v"]]


def build_session_chain(spot_day: pd.DataFrame, session: str, underlying: str, key: str,
                        band: float = BAND) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One session -> (chain_df rows, flow_df rows) at 1-minute resolution.

    chain_df columns: ts strike call_prem put_prem call_vol put_vol call_oi put_oi gex iv_call iv_put
    flow_df  columns: ts premium option_type strike side trade_type
    """
    exp, contracts = _contracts_for(underlying, session, key)
    if not contracts:
        return pd.DataFrame(), pd.DataFrame()
    lo = min(spot_day["spot"].iloc[0], spot_day["spot"].iloc[-1]) - band
    hi = max(spot_day["spot"].iloc[0], spot_day["spot"].iloc[-1]) + band
    strikes = sorted({float(c["strike_price"]) for c in contracts if lo <= float(c["strike_price"]) <= hi})

    close_ts = spot_day["ts"].iloc[-1].normalize() + pd.Timedelta(hours=16)
    spot_by_ts = spot_day.set_index("ts")["spot"]

    chain_rows, flow_rows = [], []
    for K in strikes:
        per_type = {}
        for kind in ("C", "P"):
            bars = _minute_bars(_otick(underlying, exp, kind, K), session, key)
            if API_SLEEP:
                time.sleep(API_SLEEP)
            if bars is None:
                continue
            bars = bars.set_index("ts").reindex(spot_by_ts.index)
            per_type[kind] = bars
            # flow proxy: premium = close * volume * 100; side from the contract's own tick
            c, v = bars["c"], bars["v"].fillna(0)
            # ask/bid, not buy/sell: the engine's flow features map side with {"ask": +1, "bid": -1}
            # and anything else to 0, which left block_signed_bias identically zero for 500 sessions.
            side = np.where(c.diff().fillna(0) >= 0, "ask", "bid")
            med = v.rolling(30, min_periods=5).median()
            ttype = np.where(v > 3 * med.fillna(v), "sweep", "block")
            prem = (c * v * 100).fillna(0)
            for ts, p, s, t in zip(bars.index, prem, side, ttype):
                if p > 0:
                    flow_rows.append((ts, float(p), "call" if kind == "C" else "put", K, s, t))

        if not per_type:
            continue
        cb, pb = per_type.get("C"), per_type.get("P")
        for ts, S in spot_by_ts.items():
            T = max((close_ts - ts).total_seconds() / 60.0, 1.0) / MIN_PER_YEAR
            cp = float(cb["c"].get(ts, np.nan)) if cb is not None else np.nan
            pp = float(pb["c"].get(ts, np.nan)) if pb is not None else np.nan
            cv = float(cb["v"].get(ts, 0) or 0) if cb is not None else 0.0
            pv = float(pb["v"].get(ts, 0) or 0) if pb is not None else 0.0
            ivc = solve_iv(cp, S, K, T, True) if np.isfinite(cp) else np.nan
            ivp = solve_iv(pp, S, K, T, False) if np.isfinite(pp) else np.nan
            sig = np.nanmean([ivc, ivp]) if (np.isfinite(ivc) or np.isfinite(ivp)) else np.nan
            g = _bs_gamma(S, K, T, sig) if np.isfinite(sig) else np.nan
            # VOLUME-weighted GEX (no historical OI). Dealer convention: long calls +, long puts -.
            gex = (g * (cv - pv) * 100 * S * S * 0.01) if np.isfinite(g) else np.nan
            chain_rows.append((ts, K, cp, pp, cv, pv, np.nan, np.nan, gex, ivc, ivp))

    chain = pd.DataFrame(chain_rows, columns=["ts", "strike", "call_prem", "put_prem", "call_vol", "put_vol",
                                              "call_oi", "put_oi", "gex", "iv_call", "iv_put"])
    flow = pd.DataFrame(flow_rows, columns=["ts", "premium", "option_type", "strike", "side", "trade_type"])
    return chain, flow


def build_history(spot_df: pd.DataFrame, underlying: str = "SPY") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every session in spot_df -> cached chain + flow. Resumable; re-run picks up where it stopped."""
    os.makedirs(HIST, exist_ok=True)
    key = _key(os.environ.get("POLYGON_KEY"))
    sessions = sorted(spot_df["ts"].dt.normalize().unique())[-MAX_SESSIONS:]
    chains, flows = [], []
    for i, day in enumerate(sessions, 1):
        d = pd.Timestamp(day).strftime("%Y-%m-%d")
        cpath, fpath = os.path.join(HIST, f"chain_{d}.csv.gz"), os.path.join(HIST, f"flow_{d}.csv.gz")
        if os.path.exists(cpath) and os.path.exists(fpath):
            chains.append(pd.read_csv(cpath, parse_dates=["ts"]))
            flows.append(pd.read_csv(fpath, parse_dates=["ts"]))
            continue
        spot_day = spot_df[spot_df["ts"].dt.normalize() == day]
        if len(spot_day) < 60:
            continue
        t0 = time.time()
        try:
            chain, flow = build_session_chain(spot_day, d, underlying, key)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # One bad session must not end a multi-hour walk. Nothing is cached for it,
            # so a later rerun retries this date and skips everything already done.
            _log(f"[{i}/{len(sessions)}] {d}: FAILED ({type(e).__name__}: {e}) - skipping, "
                 f"rerun to retry this date")
            continue
        with gzip.open(cpath, "wt") as fh:
            chain.to_csv(fh, index=False)
        with gzip.open(fpath, "wt") as fh:
            flow.to_csv(fh, index=False)
        chains.append(chain)
        flows.append(flow)
        _log(f"[{i}/{len(sessions)}] {d}: {len(chain):,} chain rows, {len(flow):,} flow rows, "
             f"{chain['gex'].notna().mean() if len(chain) else 0:.0%} gex, {time.time() - t0:.0f}s")

    chain_df = pd.concat(chains, ignore_index=True) if chains else pd.DataFrame()
    flow_df = pd.concat(flows, ignore_index=True) if flows else pd.DataFrame()
    for df in (chain_df, flow_df):
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
    if len(flow_df) and "side" in flow_df.columns:
        # Sessions cached before 2026-09-05 carry buy/sell labels; see build_session_chain.
        flow_df["side"] = flow_df["side"].replace({"buy": "ask", "sell": "bid"})
    chain_df = _fill_bs_greeks(chain_df, spot_df)
    return chain_df, flow_df


def _fill_bs_greeks(chain_df: pd.DataFrame, spot_df: pd.DataFrame, r: float = RISK_FREE) -> pd.DataFrame:
    """Black-Scholes vanna and charm for every chain row, volume-weighted and dealer-signed the
    same way the gex column is.

    Polygon publishes no historical greeks, so these arrived as NaN and five features
    (vanna_net_total, vanna_band_net, charm_net_total, charm_band_net, charm_drift_to_close)
    trained on nothing. They are closed forms of the inputs gamma already uses — spot, strike,
    time to the close, the solved IV — so there was never a reason for them to be empty.
    Computed at load, so the 500-session cache is not rebuilt. Fills only where missing, so a
    feed that carries real greeks is left alone.

      vanna = dDelta/dSigma          ->  $ delta change per 1 vol point, net of calls minus puts
      charm = -dDelta/dt (per year)  ->  $ delta decay over one 390-minute session
    q = 0 throughout, which makes charm identical for calls and puts.
    """
    if not len(chain_df):
        return chain_df
    need = [c for c in ("vanna", "charm") if c not in chain_df.columns or chain_df[c].isna().all()]
    if not need:
        return chain_df
    ch = chain_df
    S = ch[["ts"]].merge(spot_df[["ts", "spot"]].drop_duplicates("ts"), on="ts", how="left")["spot"].to_numpy(float)
    K = ch["strike"].to_numpy(float)
    ts = pd.DatetimeIndex(ch["ts"])
    T = np.maximum(((ts.normalize() + pd.Timedelta(hours=16)) - ts).total_seconds() / 60.0, 1.0) / MIN_PER_YEAR
    ivc, ivp = ch["iv_call"].to_numpy(float), ch["iv_put"].to_numpy(float)
    sig = np.where(np.isfinite(ivc) & np.isfinite(ivp), 0.5 * (ivc + ivp), np.where(np.isfinite(ivc), ivc, ivp))
    ok = np.isfinite(S) & np.isfinite(sig) & (S > 0) & (K > 0) & (sig > 0) & (T > 0)
    vanna, charm = np.full(len(ch), np.nan), np.full(len(ch), np.nan)
    s, k, t, v = S[ok], K[ok], T[ok], sig[ok]
    sqt = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / (v * sqt)
    d2 = d1 - v * sqt
    phi = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    vanna_u = -phi * d2 / v
    charm_u = -phi * (2.0 * r * t - d2 * v * sqt) / (2.0 * t * v * sqt)
    net = (ch["call_vol"].fillna(0).to_numpy(float) - ch["put_vol"].fillna(0).to_numpy(float))[ok] * 100.0 * s
    vanna[ok] = vanna_u * net * 0.01
    charm[ok] = charm_u * net * (390.0 / MIN_PER_YEAR)
    if "vanna" in need:
        ch["vanna"] = vanna
    if "charm" in need:
        ch["charm"] = charm
    _log(f"greeks: vanna/charm from Black-Scholes on {ok.mean():.0%} of {len(ch):,} chain rows")
    return ch


# --------------------------------------------------------------------------- #
# 3. EXTRA FEATURES the pipeline does not build: candles, volume, ATM IV block
#    Installed by wrapping FeaturePipeline._engineer so BOTH fit and live transform get them.
# --------------------------------------------------------------------------- #
def _candle_block(spot_df: pd.DataFrame) -> pd.DataFrame:
    s = spot_df.set_index("ts")
    by = s.index.normalize()
    rng = (s["high"] - s["low"]).replace(0, np.nan)
    f = pd.DataFrame(index=s.index)
    f["bar_range_bp"] = rng / s["spot"] * 1e4
    f["bar_body_frac"] = ((s["spot"] - s["open"]) / rng).clip(-1, 1)
    f["upper_wick_frac"] = ((s["high"] - s[["open", "spot"]].max(axis=1)) / rng).clip(0, 1)
    f["lower_wick_frac"] = ((s[["open", "spot"]].min(axis=1) - s["low"]) / rng).clip(0, 1)
    v = s["volume"]
    f["vol_avg_30"] = v.groupby(by).transform(lambda x: x.rolling(30, min_periods=5).mean())
    f["vol_z_30"] = v.groupby(by).transform(lambda x: (x - x.rolling(30, 5).mean()) / x.rolling(30, 5).std())
    # Both must be causal. The originals divided by the day's MEAN and TOTAL volume — numbers
    # that exist only at the close. Expanding mean of the session so far instead, and cumulative
    # volume against the PRIOR session's total, since today's total is the one thing a live bar
    # can never know. Caught by tests/test_causality.py.
    n_so_far = v.groupby(by).cumcount() + 1.0
    f["vol_ratio_session"] = v / (v.groupby(by).cumsum() / n_so_far)
    prior_tot = pd.Series(by, index=s.index).map(v.groupby(by).sum().shift(1))
    f["vol_cum_frac"] = v.groupby(by).cumsum() / prior_tot
    return f.replace([np.inf, -np.inf], np.nan)


def _atm_block(spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    f = pd.DataFrame(index=pd.DatetimeIndex(spot_df["ts"]))
    if chain_df is None or not len(chain_df) or "iv_call" not in chain_df:
        for c in ("iv_call_atm", "iv_put_atm", "iv_skew_atm", "straddle_pct", "iv_atm_chg_5"):
            f[c] = np.nan
        return f
    spot = spot_df.set_index("ts")["spot"]
    ch = chain_df.merge(spot.rename("S"), left_on="ts", right_index=True, how="inner")
    ch["dist"] = (ch["strike"] - ch["S"]).abs()
    atm = ch.sort_values("dist").groupby("ts").head(1).set_index("ts")
    f["iv_call_atm"] = atm["iv_call"]
    f["iv_put_atm"] = atm["iv_put"]
    f["iv_skew_atm"] = atm["iv_put"] - atm["iv_call"]
    f["straddle_pct"] = (atm["call_prem"] + atm["put_prem"]) / atm["S"] * 100
    f["iv_atm_chg_5"] = f[["iv_call_atm", "iv_put_atm"]].mean(axis=1).diff(5)
    return f


# Where each variable WAS versus where it is now. Every base feature is a snapshot of the
# current bar; the TFT sees a window, but the trees see one row, and the only history they
# had was four price returns. Price gets the full ladder; the option-book STATE variables —
# the ones whose trajectory says something the level does not — get their change over six
# windows. Lags never cross a session. The first k bars of a day are NaN and are filled with 0
# ("no change yet") by the caller, so they add no __isna columns.
LAG_MINUTES = (1, 5, 10, 15, 20, 25, 30, 40, 50)
LAG_STATE_MINUTES = (5, 10, 15, 20, 30, 50)
LAG_STATE_FEATURES = (
    "gex_net_total", "gex_abs_total", "gex_band_net", "gex_dist_to_zero",
    "gex_wall_above_dist", "gex_wall_below_dist", "gex_pos_centroid_dist",
    "net_prem_cum_session", "otm_skew_level",
    "iv_call_atm", "iv_put_atm", "iv_skew_atm", "straddle_pct",
    "vanna_band_net", "charm_band_net", "rv_1m",
)


def _lag_block(feats: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    by = meta["session_date"].to_numpy()
    lp = pd.Series(np.log(meta["spot"].to_numpy(float)), index=feats.index)
    g = lp.groupby(by, sort=False)
    cols: Dict[str, pd.Series] = {}
    for k in LAG_MINUTES:
        if f"ret_{k}m" not in feats.columns:                 # 1/5/15/30 already exist in the base engine
            cols[f"ret_{k}m"] = lp - g.shift(k)
    for c in LAG_STATE_FEATURES:
        if c not in feats.columns:
            continue
        s = feats[c]
        gs = s.groupby(by, sort=False)
        for k in LAG_STATE_MINUTES:
            cols[f"{c}_d{k}m"] = s - gs.shift(k)
    # one frame from a dict, not a hundred column inserts — the latter fragments the frame
    return pd.DataFrame(cols, index=feats.index)


def install_extra_features() -> None:
    """Wrap FeaturePipeline._engineer to append candle + ATM + lag blocks. Idempotent.
    Wrapping _engineer means BOTH fit_transform (training) and transform (live) get them,
    so the scaler never meets a column it was not fitted on."""
    FP = sinus.FeaturePipeline
    if getattr(FP, "_sinus_train_patched", False):
        return
    orig = FP._engineer

    def _engineer(self, spot_df, chain_df=None, flow_df=None):
        meta, feats, targets = orig(self, spot_df, chain_df, flow_df)
        extra = pd.concat([_candle_block(spot_df), _atm_block(spot_df, chain_df)], axis=1)
        extra = extra.reindex(pd.DatetimeIndex(meta["ts"])).astype(np.float32)
        extra = extra.fillna(0.0)
        if isinstance(feats, pd.DataFrame):
            feats = pd.concat([feats.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
            lags = _lag_block(feats, meta.reset_index(drop=True)).astype(np.float32).fillna(0.0)
            feats = pd.concat([feats, lags], axis=1)
        else:
            feats = np.hstack([np.asarray(feats, dtype=np.float32), extra.to_numpy()])
        return meta, feats, targets

    FP._engineer = _engineer
    # The lag block is for the trees. Keep it out of the TFT window, which already sees the
    # bars the lags were computed from: 226 -> ~121 columns per timestep, and the epoch time
    # and RAM that go with it.
    sinus.SEQUENCE_EXCLUDE = r"_d\d+m$|^ret_(10|20|25|40|50)m$"
    FP._sinus_train_patched = True



# --------------------------------------------------------------------------- #
# 3b. FRESH START: make the champion store ignore anything not from this feature set.
#     Champions written by this trainer carry feature_set="v2" and their horizon list.
#     Anything on GitHub without that tag (the old price-only champion) is treated as
#     absent: never pulled in as a starting point, never used as the bar to beat.
#     Multi-node coordination still works between v2 nodes.
# --------------------------------------------------------------------------- #
# v3 (2026-09-05): hist_prior_ret, vol_ratio_session and vol_cum_frac were lookahead; fixed.
# Nothing tagged v2 is comparable to anything tagged v3 — its scores were measured against
# a matrix that contained the answer.
# v4 (2026-09-05): the lag block no longer enters the TFT window (trees only), so a v3
# champion's tft.pt has the wrong input width for this code. Tabular matrix unchanged.
FEATURE_SET = "v4"


def install_fresh_start() -> None:
    from sinus_gitstore import GitStore
    if getattr(GitStore, "_sinus_train_patched", False):
        return
    import json as _json

    def _remote_is_v2(self) -> bool:
        p = os.path.join(self.clone_dir, "champion", "champion.json")
        if not os.path.exists(p):
            return False
        try:
            m = _json.load(open(p))
            return m.get("feature_set") == FEATURE_SET and tuple(m.get("horizons", ())) == HORIZONS_NEW
        except Exception:
            return False

    orig_pull, orig_remote, orig_push = GitStore.pull_champion, GitStore.remote_champion_score, GitStore.push_champion

    def pull_champion(self):
        if not self.enabled:
            return None
        self._sync()
        if not _remote_is_v2(self):
            _log("fresh start: remote champion is not feature_set v2 - ignoring it")
            return None
        return orig_pull(self)

    def remote_champion_score(self):
        if not self.enabled:
            return None
        self._sync()
        return orig_remote(self) if _remote_is_v2(self) else None

    def push_champion(self, local_champion_dir, meta, min_improvement=0.0):
        meta = dict(meta)
        meta["feature_set"] = FEATURE_SET
        meta["horizons"] = list(HORIZONS_NEW)
        # stamp the local champion.json too so pull_champion on another node recognises it
        try:
            cp = os.path.join(local_champion_dir, "champion.json")
            if os.path.exists(cp):
                m = _json.load(open(cp)); m.update(feature_set=FEATURE_SET, horizons=list(HORIZONS_NEW))
                _json.dump(m, open(cp, "w"), indent=2)
        except Exception:
            pass
        return orig_push(self, local_champion_dir, meta, min_improvement)

    GitStore.pull_champion = pull_champion
    GitStore.remote_champion_score = remote_champion_score
    GitStore.push_champion = push_champion
    GitStore._sinus_train_patched = True


# --------------------------------------------------------------------------- #
# 3c. SHIP THE SCALER WITH THE CHAMPION.
#     CoreModelingEngine.save writes engine.json + trees/ + tft/ and nothing else.
#     The scaler lives on FeaturePipeline, and the daemon never calls its save().
#     So a promoted champion was weights with no way to reproduce the input
#     transform - the trees and TFT were fit on features scaled by TRAIN-split
#     RobustScaler statistics, and anything else feeds them a different
#     distribution. FeaturePipeline.save's own docstring says exactly this.
#
#     Fix: stash the fitted pipeline when fit_transform runs, then pickle it into
#     the champion directory on promotion. _promote copytree's the candidate first,
#     so writing after it means the file lands in champion/ and rides along to
#     GitHub with everything else push_champion sends.
# --------------------------------------------------------------------------- #
_LAST_PIPELINE: list = []


def install_champion_scaler() -> None:
    import sinus_daemon as sd
    if getattr(sd, "_sinus_train_scaler_patched", False):
        return

    FP = sinus.FeaturePipeline
    orig_ft = FP.fit_transform

    def fit_transform(self, spot_df, chain_df=None, flow_df=None):
        out = orig_ft(self, spot_df, chain_df, flow_df)
        _LAST_PIPELINE[:] = [self]
        return out

    FP.fit_transform = fit_transform

    orig_promote = sd._promote

    def _promote(work_dir, model_dir, params, val, test, trial,
                 pipeline=None, n_sessions=None):
        # sinus_daemon._promote grew pipeline= and n_sessions= keywords. This wrapper
        # did not, so every promotion raised TypeError inside the search's try block and
        # was swallowed as a failed trial. 35 trials, zero champions, champ nan.
        orig_promote(work_dir, model_dir, params, val, test, trial,
                     pipeline=pipeline, n_sessions=n_sessions)
        dest = os.path.join(work_dir, sd.CHAMPION)
        if not _LAST_PIPELINE:
            _log("WARNING: no fitted pipeline in scope - champion saved WITHOUT its scaler")
            return
        pipe = _LAST_PIPELINE[0]
        try:
            pipe.save(os.path.join(dest, "pipeline.pkl"))
        except Exception as e:
            _log(f"WARNING: could not save scaler with champion: {e}")
            return
        # Record the feature contract next to the weights. A silent column-order or
        # count mismatch at load time predicts confidently and wrongly, which is the
        # worst failure mode available, so make it checkable.
        try:
            import json as _json
            mp = os.path.join(dest, "champion.json")
            m = _json.load(open(mp))
            fn = pipe.scaler.feature_names_out_
            names = list(fn() if callable(fn) else fn)
            m["n_features"] = len(names)
            m["feature_names_sha"] = __import__("hashlib").sha1("|".join(names).encode()).hexdigest()[:12]
            m["feature_set"] = FEATURE_SET
            m["horizons"] = list(HORIZONS_NEW)
            m["has_scaler"] = True
            _json.dump(m, open(mp, "w"), indent=2, default=float)
        except Exception:
            pass
        _log(f"champion trial {trial}: saved pipeline.pkl alongside the weights")

    sd._promote = _promote
    sd._sinus_train_scaler_patched = True

# --------------------------------------------------------------------------- #
# 4. EVOLVE: hand the full history to the existing champion machinery
# --------------------------------------------------------------------------- #
def evolve_forever(spot_df: pd.DataFrame, chain_df: pd.DataFrame, flow_df: pd.DataFrame, work_dir: str) -> None:
    import sinus_daemon as sd
    _log(f"evolving on {len(spot_df):,} bars, {len(chain_df):,} chain rows, {len(flow_df):,} flow rows, "
         f"horizons {HORIZONS_NEW}")
    # flow_df: search_forever hardcodes None as the third arg. Bind it via a wrapper on the pipeline.
    if len(flow_df):
        FP = sinus.FeaturePipeline
        orig_ft = FP.fit_transform

        def fit_transform(self, s, c=None, f=None):
            return orig_ft(self, s, c, flow_df if f is None else f)
        FP.fit_transform = fit_transform

    prune = os.environ.get("SINUS_PRUNE", "1") == "1"
    sd.search_forever(spot_df, work_dir=work_dir,
                      chain_df=chain_df if len(chain_df) else None,
                      tft_epochs=int(os.environ.get("SINUS_TFT_EPOCHS", "12")),
                      prune=prune,
                      prune_percentile=float(os.environ.get("SINUS_PRUNE_PERCENTILE", "60")),
                      screen_rounds=int(os.environ.get("SINUS_SCREEN_ROUNDS", "2")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build history only, then exit")
    ap.add_argument("--csv", default=os.environ.get("SINUS_CSV"), help="1-min OHLCV csv")
    ap.add_argument("--symbol", default=os.environ.get("SINUS_SYMBOL", "SPY"))
    args = ap.parse_args()

    if not args.csv or not os.path.exists(args.csv):
        _log("SINUS_CSV is required: a 1-min OHLCV file for the underlying. Polygon stocks plan not needed.")
        return 2

    install_horizons()
    install_extra_features()
    install_fresh_start()
    install_champion_scaler()
    os.makedirs(VOL, exist_ok=True)

    spot_df = load_ohlcv(args.csv)
    chain_df, flow_df = build_history(spot_df, args.symbol)
    if len(chain_df):
        _log(f"history: {chain_df['ts'].dt.normalize().nunique()} sessions with chain, "
             f"gex coverage {chain_df['gex'].notna().mean():.0%}, "
             f"iv coverage {chain_df['iv_call'].notna().mean():.0%}")
    else:
        _log("WARNING: no chain history built - training would be price-only again. Check POLYGON_KEY.")
    if args.build:
        return 0
    evolve_forever(spot_df, chain_df, flow_df, work_dir=os.path.join(VOL, "champion_v2"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
