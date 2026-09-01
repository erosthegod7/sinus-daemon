"""
sinus.py
========
SINUS — Multi-Horizon Stacked Heterogeneous Ensemble Network for SPY 0DTE.
Single-file production build (unified Phases 1–4, audited 2026-09-01).

    Phase 1  FeaturePipeline        raw ticks → causal features + multi-horizon targets
    Phase 2  CoreModelingEngine     LightGBM + CatBoost + native-PyTorch TFT
    Phase 3  EnsembleFusionEngine   options-physics overlay + adaptive gate
    Phase 4  LiveTradingEngine / Backtester / Orchestrator

Audit changes vs. the four-file build
-------------------------------------
1. Shape alignment. Every tensor hand-off is asserted and logged (structured
   JSON) at build time: tabular X[n,F] ↔ TFT X_seq[N,L,F] share one feature
   list; static/known covariate widths are frozen at fit and re-checked at
   predict; dtypes are cast once (float32 features, int64 groups) — never
   implicitly.
2. Vectorisation. The per-snapshot GEX loop is gone: the chain is pivoted to
   (T × K) matrices and every physics metric — zero-gamma crossing, max pain
   (as two matrix products), clusters, walls, centroids — is computed for all
   snapshots at once. Rolling statistics use Cython groupby-rolling, not
   Python lambdas. Phase 3 clusters are computed with bincount; the five
   horizon mean-shifts run as one vector iteration. The live tick path
   (`RollingBuffer.push`) is O(1) with an incrementally-maintained latest
   ladder, so ingest cost is microseconds per tick.
3. Memory. All live buffers are `deque(maxlen=…)`; the session rollover clears
   them; the latest-ladder cache is a bounded dict; order/trade logs are
   append-only lists bounded by the number of decisions in a session.
4. Sanitisation. Every division carries a guarded denominator, every ratio is
   clipped, `inf` is mapped to NaN before imputation, empty-mask reductions
   return 0/NaN explicitly, and the fusion layer tolerates any expert being
   NaN (weights renormalise over what exists). Missing screens are flagged,
   never dropped.
5. Leakage. Features at bar t use prints with timestamp ≤ t (snapshots are
   ceiled to the bar they complete in); targets are t+h within the session
   only; the split is session-level with an embargo; the scaler and target
   winsorisation are fit on TRAIN rows only; rolling-origin validation drives
   early stopping; the backtester sees only the chain snapshot available at
   or before each bar. `verify_causality()` proves it numerically by
   truncating the tape and checking that features at t do not change.

Heavy optional dependencies (lightgbm, catboost, torch, websockets) are imported
lazily; the tree/deep layers degrade to NaN experts when absent and the gate
re-normalises over the experts that exist.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

# ----------------------------------------------------------------------------- #
# Optional heavy dependencies
# ----------------------------------------------------------------------------- #
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:                                   # pragma: no cover
    lgb, _HAS_LGB = None, False
try:
    from catboost import CatBoostRegressor, Pool
    _HAS_CAT = True
except Exception:                                   # pragma: no cover
    CatBoostRegressor, Pool, _HAS_CAT = None, None, False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    _HAS_TORCH = True
except Exception:                                   # pragma: no cover
    torch, nn, Fn, _HAS_TORCH = None, None, None, False
try:
    import websockets
    _HAS_WS = True
except Exception:                                   # pragma: no cover
    websockets, _HAS_WS = None, False

# ----------------------------------------------------------------------------- #
# Structured logging
# ----------------------------------------------------------------------------- #
class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, plus any ``extra`` fields."""
    _SKIP = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info",
             "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread",
             "threadName", "processName", "process", "message", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        for k, v in record.__dict__.items():
            if k not in self._SKIP and not k.startswith("_"):
                d[k] = v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def get_logger(name: str = "sinus", json_lines: Optional[bool] = None) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        use_json = json_lines if json_lines is not None else os.environ.get("SINUS_LOG_JSON", "1") == "1"
        h.setFormatter(JsonFormatter() if use_json else logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        lg.addHandler(h)
        lg.setLevel(os.environ.get("SINUS_LOG_LEVEL", "INFO"))
        lg.propagate = False
    return lg


log = get_logger("sinus")


def shape_of(x: Any) -> Any:
    if isinstance(x, (np.ndarray, pd.DataFrame)):
        return list(x.shape)
    if isinstance(x, pd.Series):
        return [len(x)]
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items() if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series))}
    return None


def log_shapes(event: str, **arrays: Any) -> None:
    """Structured shape trace at every data transition."""
    log.info(event, extra={"event": "shape", **{k: shape_of(v) for k, v in arrays.items()}})


EPS = 1e-9
TZ = "America/New_York"
HORIZONS: Tuple[str, ...] = ("5m", "15m", "30m", "1h", "eod")
HORIZON_MINUTES: Dict[str, Optional[float]] = {"5m": 5.0, "15m": 15.0, "30m": 30.0, "1h": 60.0, "eod": None}
EXPERTS: Tuple[str, ...] = ("lgb", "cat", "tft", "physics")


def safe_div(a: Union[np.ndarray, pd.Series, float], b: Union[np.ndarray, pd.Series, float],
             fill: float = 0.0) -> Union[np.ndarray, pd.Series, float]:
    """a / b with |b| < EPS → fill, and inf/nan → fill. Works on scalars, arrays and Series."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a = pd.Series(a, index=b.index) if not isinstance(a, pd.Series) else a
        b = pd.Series(b, index=a.index) if not isinstance(b, pd.Series) else b
        out = a / b.where(b.abs() >= EPS)
        return out.replace([np.inf, -np.inf], np.nan).fillna(fill)
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) >= EPS, a / np.where(np.abs(b) >= EPS, b, 1.0), fill)
    out = np.where(np.isfinite(out), out, fill)
    return float(out) if out.ndim == 0 else out


# ============================================================================= #
# PHASE 1 — FEATURE PIPELINE
# ============================================================================= #
@dataclass
class PipelineConfig:
    tz: str = TZ
    session_open: str = "09:30"
    session_close: str = "16:00"
    grid_freq: str = "1min"
    horizons: Dict[str, Optional[int]] = field(default_factory=lambda: {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "eod": None})
    spot_band_points: float = 3.0
    gex_decay_windows: Tuple[int, ...] = (5, 15)
    top_k_concentration: int = 3
    flow_windows: Tuple[int, ...] = (5, 15)
    zscore_lookback: int = 60
    zscore_min_periods: int = 20
    otm_accel_window: int = 5
    block_window: int = 15
    rv_lookback: int = 30
    rv_floor: float = 1e-4
    train_frac: float = 0.70
    val_frac: float = 0.15
    embargo_sessions: int = 1
    clip_sigma: float = 8.0
    ratio_clip: float = 50.0
    dtype: type = np.float32

    def horizon_names(self) -> List[str]:
        return list(self.horizons.keys())


CHAIN_REQUIRED = ("ts", "strike")
CHAIN_OPTIONAL = ("call_prem", "put_prem", "call_ask_prem", "call_bid_prem", "put_ask_prem", "put_bid_prem",
                  "call_oi", "put_oi", "call_vol", "put_vol", "gex", "vanna", "charm")
FLOW_REQUIRED = ("ts", "premium", "option_type", "strike", "side", "trade_type")
FLOW_OPTIONAL = ("size",)
TEXT_COLS = ("ts", "option_type", "side", "trade_type")


def to_session_tz(ts: pd.Series, tz: str = TZ) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce")
    return out.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") if out.dt.tz is None else out.dt.tz_convert(tz)


def coerce_frame(df: Optional[pd.DataFrame], required: Sequence[str], optional: Sequence[str],
                 cfg: PipelineConfig, name: str) -> Optional[pd.DataFrame]:
    """Validate, add absent optional columns as NaN, coerce numerics, map inf→NaN, sort. Never raises on dirty cells."""
    if df is None or len(df) == 0:
        log.warning(f"{name} absent — dependent features NaN + flagged", extra={"event": "missing_input", "frame": name})
        return None
    df = df.copy()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    for c in optional:
        if c not in df.columns:
            df[c] = np.nan
    df["ts"] = to_session_tz(df["ts"], cfg.tz)
    df = df.dropna(subset=["ts"])
    num = [c for c in df.columns if c not in TEXT_COLS]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
    return df.sort_values("ts", kind="mergesort").reset_index(drop=True)


def _session_bounds(dates: Iterable[pd.Timestamp], cfg: PipelineConfig) -> Dict[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]]:
    out = {}
    for d in dates:
        d = pd.Timestamp(d).normalize()
        out[d] = (pd.Timestamp(f"{d.date()} {cfg.session_open}", tz=cfg.tz), pd.Timestamp(f"{d.date()} {cfg.session_close}", tz=cfg.tz))
    return out


def build_spot_grid(spot_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Regular RTH 1-min grid per session; each bar carries the last print at or before the bar time."""
    s = spot_df.set_index("ts")["spot"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s > 0]
    frames = []
    for d, (o, c) in _session_bounds(s.index.normalize().unique(), cfg).items():
        day = s[(s.index >= o) & (s.index <= c)]
        if day.empty:
            continue
        grid = pd.date_range(o, c, freq=cfg.grid_freq, tz=cfg.tz)
        aligned = day.reindex(day.index.union(grid)).sort_index().ffill().reindex(grid)
        f = pd.DataFrame({"ts": grid, "spot": aligned.to_numpy()})
        f["session_date"] = d
        tot = (c - o).total_seconds() / 60.0
        f["minutes_since_open"] = (f["ts"] - o).dt.total_seconds() / 60.0
        f["minutes_to_close"] = tot - f["minutes_since_open"]
        f["session_frac"] = f["minutes_since_open"] / tot
        frames.append(f)
    if not frames:
        raise ValueError("No spot data inside regular trading hours.")
    g = pd.concat(frames, ignore_index=True)
    g["session_sin"], g["session_cos"] = np.sin(2 * np.pi * g["session_frac"]), np.cos(2 * np.pi * g["session_frac"])
    g["dow"] = g["ts"].dt.dayofweek.astype(float)
    return g


def snap_to_grid(ts: pd.Series, cfg: PipelineConfig) -> pd.Series:
    """Ceil to the bar the print completes in — a 10:14:37 snapshot is first usable at the 10:15 decision."""
    return ts.dt.ceil(cfg.grid_freq)


def groll(x: pd.Series, by: pd.Series, window: int, fn: str, min_periods: int = 1) -> pd.Series:
    """Causal per-session rolling statistic (Cython groupby-rolling, no Python lambdas)."""
    r = getattr(x.groupby(by, sort=False).rolling(window, min_periods=min_periods), fn)()
    return r.reset_index(level=0, drop=True).reindex(x.index)


def rolling_z(x: pd.Series, by: pd.Series, lookback: int, min_periods: int, clip: float = 8.0) -> pd.Series:
    mu = groll(x, by, lookback, "mean", min_periods)
    sd = groll(x, by, lookback, "std", min_periods)
    return safe_div(x - mu, sd, 0.0).clip(-clip, clip)


# ----------------------------------------------------------------------------- #
# Vectorised snapshot physics (all snapshots at once)
# ----------------------------------------------------------------------------- #
GEX_COLS = ["gex_net_total", "gex_abs_total", "gex_zero_level", "gex_dist_to_zero", "gex_dist_to_zero_pct", "gex_above_zero",
            "gex_top3_concentration", "gex_hhi", "gex_band_net", "gex_band_abs", "gex_band_share", "gex_pos_centroid",
            "gex_pos_centroid_dist", "gex_abs_centroid_dist", "gex_max_node_strike", "gex_max_node_value", "gex_max_node_dist",
            "gex_wall_above", "gex_wall_above_dist", "gex_wall_below", "gex_wall_below_dist", "gex_neg_pocket_above",
            "gex_neg_pocket_below", "vanna_net_total", "vanna_band_net", "charm_net_total", "charm_band_net",
            "max_pain", "max_pain_dist"]


def _crossings(K: np.ndarray, Y: np.ndarray, spot: np.ndarray) -> np.ndarray:
    """Spot-nearest linear zero crossing of profile Y (T×K) over strikes K. NaN where Y never changes sign."""
    s = np.sign(Y)
    cross = (s[:, :-1] * s[:, 1:]) < 0
    y0, y1 = Y[:, :-1], Y[:, 1:]
    den = np.where(np.abs(y1 - y0) < EPS, EPS, y1 - y0)
    lvl = K[None, :-1] - y0 * (K[1:] - K[:-1])[None, :] / den
    dist = np.where(cross, np.abs(lvl - spot[:, None]), np.inf)
    out = np.take_along_axis(lvl, np.argmin(dist, axis=1)[:, None], 1)[:, 0]
    return np.where(cross.any(axis=1), out, np.nan)


def zero_gamma_matrix(K: np.ndarray, G: np.ndarray, spot: np.ndarray) -> np.ndarray:
    """Gamma flip level: the spot-nearest strike where DEALER GAMMA CHANGES SIGN.

    Audit fix (2026-09-01). The previous version scanned the *cumulative* profile, which never crosses
    zero when one node dominates the ladder — it returned NaN on the real 8/31 SPY book where UW printed
    a flip at 766.10. Vendors (UW, SpotGamma) report the crossing of the per-strike profile, so that is
    what is computed first; the cumulative crossing is kept only as a fallback for ladders whose
    per-strike profile is single-signed.
    """
    per = _crossings(K, G, spot)
    cum = _crossings(K, np.cumsum(G, axis=1), spot)
    return np.where(np.isfinite(per), per, cum)


def max_pain_matrix(K: np.ndarray, C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Max pain per row via two matrix products: pain[t,i] = Σ_j C[t,j]·max(K_i−K_j,0) + P[t,j]·max(K_j−K_i,0)."""
    diff = K[None, :] - K[:, None]                                        # [j, i] = K_i − K_j
    pain = np.nan_to_num(C) @ np.maximum(diff, 0) + np.nan_to_num(P) @ np.maximum(-diff, 0)
    ok = np.isfinite(C).sum(1) + np.isfinite(P).sum(1) >= 2
    return np.where(ok, K[np.argmin(pain, axis=1)], np.nan)


def snapshot_physics(ch: pd.DataFrame, spot_map: pd.Series, cfg: PipelineConfig) -> pd.DataFrame:
    """All per-snapshot GEX/vanna/charm/max-pain metrics, vectorised over snapshots. Index = ts_grid."""
    piv = ch.pivot_table(index="ts_grid", columns="strike", values=["gex", "vanna", "charm", "call_oi", "put_oi"],
                         aggfunc="last", observed=True)
    piv = piv.sort_index(axis=1, level=1)
    K = piv["gex"].columns.to_numpy(float)
    spot = spot_map.reindex(piv.index).to_numpy(float)
    keep = np.isfinite(spot)
    piv, spot = piv.iloc[keep], spot[keep]
    T = len(piv)
    if T == 0:
        return pd.DataFrame(columns=GEX_COLS)
    Graw = piv["gex"].to_numpy(float)
    valid = np.isfinite(Graw)
    G = np.nan_to_num(Graw)
    absg = np.abs(G)
    tot_abs = absg.sum(1)
    has = valid.any(1)

    flip = zero_gamma_matrix(K, G, spot)
    top = np.sort(absg, axis=1)[:, -cfg.top_k_concentration:].sum(1)
    band = np.abs(K[None, :] - spot[:, None]) <= cfg.spot_band_points
    above, below = K[None, :] > spot[:, None], K[None, :] < spot[:, None]
    pos = G > 0
    band_net, band_abs = (G * band).sum(1), (absg * band).sum(1)
    pos_mass = (G * pos).sum(1)
    pos_cent = safe_div((K * G * pos).sum(1), pos_mass, np.nan)
    imax = np.argmax(absg, axis=1)
    kmax, gmax = K[imax], np.take_along_axis(G, imax[:, None], 1)[:, 0]

    def _wall(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = mask & pos
        gm = np.where(m, G, -np.inf)
        j = np.argmax(gm, axis=1)
        val = np.where(m.any(1), np.take_along_axis(gm, j[:, None], 1)[:, 0], 0.0)
        strike = np.where(mask.any(1), K[np.argmax(np.where(mask, G, -np.inf), axis=1)], np.nan)
        return val, strike

    wa, wa_k = _wall(above)
    wb, wb_k = _wall(below)
    neg_above = np.where(above.any(1), np.min(np.where(above, G, np.inf), axis=1), 0.0)
    neg_below = np.where(below.any(1), np.min(np.where(below, G, np.inf), axis=1), 0.0)

    def _sum(name: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        A = piv[name].to_numpy(float)
        fin = np.isfinite(A) if mask is None else (np.isfinite(A) & mask)
        return np.where(fin.any(1), np.nansum(np.where(fin, A, 0.0), axis=1), np.nan)

    mp = max_pain_matrix(K, piv["call_oi"].to_numpy(float), piv["put_oi"].to_numpy(float))
    out = pd.DataFrame({
        "gex_net_total": G.sum(1), "gex_abs_total": tot_abs, "gex_zero_level": flip,
        "gex_dist_to_zero": spot - flip, "gex_dist_to_zero_pct": safe_div(spot - flip, spot, np.nan),
        "gex_above_zero": np.where(np.isfinite(flip), (spot > flip).astype(float), np.nan),
        "gex_top3_concentration": safe_div(top, tot_abs, 0.0), "gex_hhi": (safe_div(absg, tot_abs[:, None], 0.0) ** 2).sum(1),
        "gex_band_net": band_net, "gex_band_abs": band_abs, "gex_band_share": safe_div(band_abs, tot_abs, 0.0),
        "gex_pos_centroid": pos_cent, "gex_pos_centroid_dist": pos_cent - spot,
        "gex_abs_centroid_dist": safe_div((K * absg).sum(1), tot_abs, np.nan) - spot,
        "gex_max_node_strike": kmax, "gex_max_node_value": gmax, "gex_max_node_dist": kmax - spot,
        "gex_wall_above": wa, "gex_wall_above_dist": wa_k - spot, "gex_wall_below": wb, "gex_wall_below_dist": spot - wb_k,
        "gex_neg_pocket_above": neg_above, "gex_neg_pocket_below": neg_below,
        "vanna_net_total": _sum("vanna"), "vanna_band_net": _sum("vanna", band),
        "charm_net_total": _sum("charm"), "charm_band_net": _sum("charm", band),
        "max_pain": mp, "max_pain_dist": mp - spot,
    }, index=piv.index)
    out.loc[~has, GEX_COLS] = np.nan                                       # snapshot with no readable GEX at all
    return out[GEX_COLS]


# ----------------------------------------------------------------------------- #
# Feature blocks
# ----------------------------------------------------------------------------- #
def gex_profile_features(chain: Optional[pd.DataFrame], grid: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Snapshot physics aligned to the grid, carried forward within the session, plus causal decay velocities."""
    spine = grid[["ts", "session_date", "spot", "minutes_to_close"]]
    by = spine["session_date"]
    if chain is None or chain["gex"].notna().sum() == 0:
        f = pd.DataFrame(np.nan, index=spine.index, columns=GEX_COLS)
        f["gex_snapshot_age_min"] = np.nan
    else:
        ch = chain.assign(ts_grid=snap_to_grid(chain["ts"], cfg))
        snap = snapshot_physics(ch, spine.set_index("ts")["spot"], cfg)
        f = spine[["ts"]].join(snap, on="ts").drop(columns=["ts"])
        f[GEX_COLS] = f[GEX_COLS].groupby(by, sort=False).ffill()
        seen = spine["ts"].where(spine["ts"].isin(snap.index)).groupby(by, sort=False).ffill()
        f["gex_snapshot_age_min"] = (spine["ts"] - seen).dt.total_seconds() / 60.0
    for w in cfg.gex_decay_windows:
        f[f"gex_band_decay_{w}m"] = f["gex_band_net"].groupby(by, sort=False).diff(w)
        f[f"gex_band_decay_pct_{w}m"] = safe_div(f[f"gex_band_decay_{w}m"], f["gex_band_abs"].abs(), 0.0).clip(-cfg.ratio_clip, cfg.ratio_clip)
        f[f"gex_dist_zero_chg_{w}m"] = f["gex_dist_to_zero"].groupby(by, sort=False).diff(w)
    f["charm_drift_to_close"] = f["charm_band_net"] * spine["minutes_to_close"] / 390.0
    return f


def premium_flow_features(chain: Optional[pd.DataFrame], flow: Optional[pd.DataFrame], grid: pd.DataFrame,
                          cfg: PipelineConfig) -> pd.DataFrame:
    """Net premium velocity (rolling sums, z-scores, 1st differences), OTM skew momentum, institutional block weight.
    Every ratio is guarded and clipped; bars with no prints carry zero flow (nothing traded), not NaN."""
    spine = grid[["ts", "session_date", "spot"]]
    by = spine["session_date"]
    f = pd.DataFrame(index=spine.index)
    base = ["net_prem_1m", "call_prem_1m", "put_prem_1m", "otm_call_prem_1m", "otm_put_prem_1m"]
    if chain is None:
        for c in base:
            f[c] = np.nan
        f["flow_side_weighted"] = np.nan
    else:
        ch = chain.assign(ts_grid=snap_to_grid(chain["ts"], cfg)).merge(
            spine[["ts", "spot"]].rename(columns={"ts": "ts_grid"}), on="ts_grid", how="inner")
        side_ok = ch[["call_ask_prem", "call_bid_prem", "put_ask_prem", "put_bid_prem"]].notna().all(axis=1)
        side_weighted = float(side_ok.mean() > 0.5) if len(ch) else 0.0
        if side_weighted:
            ch["sc"] = ch["call_ask_prem"].fillna(0.0) - ch["call_bid_prem"].fillna(0.0)
            ch["sp"] = ch["put_ask_prem"].fillna(0.0) - ch["put_bid_prem"].fillna(0.0)
        else:
            ch["sc"], ch["sp"] = ch["call_prem"].fillna(0.0), ch["put_prem"].fillna(0.0)
        ch["oc"] = ch["sc"] * (ch["strike"] > ch["spot"])
        ch["op"] = ch["sp"] * (ch["strike"] < ch["spot"])
        agg = ch.groupby("ts_grid", sort=False)[["sc", "sp", "oc", "op"]].sum()
        agg.columns = ["call_prem_1m", "put_prem_1m", "otm_call_prem_1m", "otm_put_prem_1m"]
        agg["net_prem_1m"] = agg["call_prem_1m"] - agg["put_prem_1m"]
        f = spine[["ts"]].join(agg, on="ts").drop(columns=["ts"]).fillna(0.0)
        f["flow_side_weighted"] = side_weighted
    for w in cfg.flow_windows:
        roll = groll(f["net_prem_1m"], by, w, "sum")
        f[f"net_prem_{w}m"] = roll
        f[f"net_prem_z_{w}m"] = rolling_z(roll, by, cfg.zscore_lookback, cfg.zscore_min_periods)
        f[f"net_prem_vel_{w}m"] = roll.groupby(by, sort=False).diff(1)
        f[f"net_prem_vel_z_{w}m"] = rolling_z(f[f"net_prem_vel_{w}m"], by, cfg.zscore_lookback, cfg.zscore_min_periods)
    f["net_prem_cum_session"] = f["net_prem_1m"].groupby(by, sort=False).cumsum()
    f["call_put_prem_ratio_15m"] = safe_div(groll(f["call_prem_1m"], by, 15, "sum"),
                                            groll(f["put_prem_1m"], by, 15, "sum").abs(), 0.0).clip(-cfg.ratio_clip, cfg.ratio_clip)
    w = cfg.otm_accel_window
    oc, op = groll(f["otm_call_prem_1m"], by, w, "sum"), groll(f["otm_put_prem_1m"], by, w, "sum")
    acc_c = oc.groupby(by, sort=False).diff(1).groupby(by, sort=False).diff(1)
    acc_p = op.groupby(by, sort=False).diff(1).groupby(by, sort=False).diff(1)
    f["otm_call_accel"], f["otm_put_accel"] = acc_c, acc_p
    f["otm_skew_momentum"] = safe_div(acc_c - acc_p, acc_c.abs() + acc_p.abs(), 0.0).clip(-1, 1)
    f["otm_skew_log_ratio"] = np.log((acc_c.abs().fillna(0.0) + 1.0) / (acc_p.abs().fillna(0.0) + 1.0)) * np.sign((acc_c - acc_p).fillna(0.0))
    f["otm_skew_level"] = safe_div(oc - op, oc.abs() + op.abs(), 0.0).clip(-1, 1)
    blk = ("block_notional_15m", "sweep_notional_15m", "inst_block_weight", "dark_pool_share", "block_vs_sweep_z", "block_signed_bias")
    if flow is None:
        for c in blk:
            f[c] = np.nan
    else:
        fl = flow.assign(ts_grid=snap_to_grid(flow["ts"], cfg))
        tt = fl["trade_type"].astype(str).str.lower()
        prem = fl["premium"].fillna(0.0)
        fl["blk"] = prem * tt.isin(["block", "dark_pool"])
        fl["dark"] = prem * (tt == "dark_pool")
        fl["swp"] = prem * (tt == "sweep")
        fl["all"] = prem
        sd = fl["side"].astype(str).str.lower().map({"ask": 1.0, "bid": -1.0}).fillna(0.0)
        ot = fl["option_type"].astype(str).str.upper().map({"C": 1.0, "P": -1.0}).fillna(0.0)
        fl["sblk"] = fl["blk"] * sd * ot
        b = fl.groupby("ts_grid", sort=False)[["blk", "dark", "swp", "all", "sblk"]].sum()
        b = spine[["ts"]].join(b, on="ts").drop(columns=["ts"]).fillna(0.0)
        wb = cfg.block_window
        rs = {c: groll(b[c], by, wb, "sum") for c in b.columns}
        f["block_notional_15m"], f["sweep_notional_15m"] = rs["blk"], rs["swp"]
        f["inst_block_weight"] = safe_div(rs["blk"], rs["blk"] + rs["swp"], 0.0).clip(0, 1)
        f["dark_pool_share"] = safe_div(rs["dark"], rs["all"], 0.0).clip(0, 1)
        f["block_vs_sweep_z"] = rolling_z(rs["blk"] - rs["swp"], by, cfg.zscore_lookback, cfg.zscore_min_periods)
        f["block_signed_bias"] = safe_div(rs["sblk"], rs["blk"], 0.0).clip(-1, 1)
    return f


def price_context_features(grid: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Causal spot features + lagged-session history (prior close / return / range, today's gap)."""
    by, p = grid["session_date"], grid["spot"]
    f = pd.DataFrame(index=grid.index)
    logp = np.log(p.where(p > 0))
    r1 = logp.groupby(by, sort=False).diff(1)
    f["ret_1m"] = r1
    for w in (5, 15, 30):
        f[f"ret_{w}m"] = logp.groupby(by, sort=False).diff(w)
    f["rv_1m"] = groll(r1, by, cfg.rv_lookback, "std", 5).clip(lower=cfg.rv_floor)
    f["ret_from_open"] = logp - logp.groupby(by, sort=False).transform("first")
    hi, lo = p.groupby(by, sort=False).cummax(), p.groupby(by, sort=False).cummin()
    f["session_range"] = hi - lo
    f["pos_in_session_range"] = safe_div(p - lo, hi - lo, 0.5).clip(0, 1)
    f["dist_to_session_high"], f["dist_to_session_low"] = hi - p, p - lo
    f["dist_to_twap"] = p - p.groupby(by, sort=False).cumsum() / (grid["minutes_since_open"] + 1.0)
    daily = grid.groupby("session_date", sort=True)["spot"].agg(["first", "last", "max", "min"])
    daily["prior_close"] = daily["last"].shift(1)
    daily["prior_ret"] = np.log(daily["last"] / daily["prior_close"])
    daily["prior_range_pct"] = ((daily["max"] - daily["min"]) / daily["last"]).shift(1)
    daily["gap_pct"] = np.log(daily["first"] / daily["prior_close"])
    h = grid[["session_date"]].join(daily[["prior_close", "prior_ret", "prior_range_pct", "gap_pct"]], on="session_date")
    for c in ("prior_close", "prior_ret", "prior_range_pct", "gap_pct"):
        f[f"hist_{c}"] = h[c].to_numpy()
    f["dist_to_prior_close"] = p - f["hist_prior_close"]
    return f.replace([np.inf, -np.inf], np.nan)


def build_targets(grid: pd.DataFrame, rv_1m: pd.Series, cfg: PipelineConfig) -> pd.DataFrame:
    """y_price / y_ret / y_retn / y_dprice per horizon. Look-aheads crossing the close are NaN — never filled across days."""
    by, p = grid["session_date"], grid["spot"]
    logp = np.log(p.where(p > 0))
    t = pd.DataFrame(index=grid.index)
    eod = p.groupby(by, sort=False).transform("last")
    for name, h in cfg.horizons.items():
        fut = eod if h is None else p.groupby(by, sort=False).shift(-h)
        hmin = grid["minutes_to_close"].clip(lower=1.0) if h is None else float(h)
        ret = np.log(fut.where(fut > 0)) - logp
        t[f"y_price_{name}"], t[f"y_ret_{name}"], t[f"y_dprice_{name}"] = fut, ret, fut - p
        t[f"y_retn_{name}"] = safe_div(ret, rv_1m * np.sqrt(hmin), np.nan)
    t["y_eod_minutes_ahead"] = grid["minutes_to_close"]
    return t.replace([np.inf, -np.inf], np.nan)


def chronological_split(session_dates: pd.Series, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    """Whole-session chronological split with an embargo gap between blocks."""
    days = np.array(sorted(session_dates.unique()))
    n, sd = len(days), session_dates.to_numpy()
    z = np.zeros(len(sd), bool)
    if n < 3:
        log.warning("fewer than 3 sessions — everything assigned to train", extra={"event": "split", "sessions": n})
        return {"train": ~z, "val": z.copy(), "test": z.copy(), "embargo": z.copy()}
    n_tr, n_va, e = max(1, int(cfg.train_frac * n)), max(1, int(cfg.val_frac * n)), cfg.embargo_sessions
    tr, va, te = days[:n_tr], days[n_tr + e: n_tr + e + n_va], days[n_tr + e + n_va + e:]
    if len(te) == 0:
        te, va = days[-1:], np.setdiff1d(va, days[-1:])
    used = set(tr) | set(va) | set(te)
    emb = np.array([d for d in days if d not in used], dtype=days.dtype)
    return {"train": np.isin(sd, tr), "val": np.isin(sd, va), "test": np.isin(sd, te), "embargo": np.isin(sd, emb)}


class FeatureScaler:
    """inf→NaN, missing indicators, median impute, RobustScaler — all statistics from TRAIN rows only.
    ``transform`` never drops or reorders a column, so the live feature space always matches training."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.columns_: List[str] = []
        self.indicator_cols_: List[str] = []
        self.binary_cols_: List[str] = []
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = RobustScaler(quantile_range=(5.0, 95.0))
        self._cont_idx: List[int] = []

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        X = X.replace([np.inf, -np.inf], np.nan)
        self.columns_ = list(X.columns)
        nun = X.nunique(dropna=True)
        self.binary_cols_ = [c for c in X.columns if nun[c] <= 2]
        self.indicator_cols_ = [c for c in X.columns if X[c].isna().any()]
        Xi = self.imputer_.fit_transform(X)
        self._cont_idx = [i for i, c in enumerate(self.columns_) if c not in self.binary_cols_]
        self.scaler_.fit(Xi[:, self._cont_idx])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.columns_).replace([np.inf, -np.inf], np.nan)
        ind = X[self.indicator_cols_].isna().astype(self.cfg.dtype)
        ind.columns = [f"{c}__isna" for c in self.indicator_cols_]
        Xi = self.imputer_.transform(X)
        Xi[:, self._cont_idx] = np.clip(self.scaler_.transform(Xi[:, self._cont_idx]), -self.cfg.clip_sigma, self.cfg.clip_sigma)
        out = pd.DataFrame(np.nan_to_num(Xi), columns=self.columns_, index=X.index).astype(self.cfg.dtype)
        return pd.concat([out, ind], axis=1)

    @property
    def feature_names_out_(self) -> List[str]:
        return self.columns_ + [f"{c}__isna" for c in self.indicator_cols_]


STATIC_COVARIATES = ["dow", "hist_prior_ret", "hist_prior_range_pct", "hist_gap_pct"]
KNOWN_FUTURE = ["minutes_to_close", "session_frac", "session_sin", "session_cos"]


@dataclass
class PipelineOutput:
    """Row i of meta / features_raw / features / targets is the same bar."""
    meta: pd.DataFrame
    features_raw: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    masks: Dict[str, np.ndarray]
    scaler: FeatureScaler
    cfg: PipelineConfig

    def tabular_multi(self, split: str, target_kind: str = "retn") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.masks[split]
        cols = [f"y_{target_kind}_{h}" for h in self.cfg.horizon_names()]
        return (self.features.to_numpy(dtype=self.cfg.dtype)[m], self.targets[cols].to_numpy(dtype=self.cfg.dtype)[m],
                np.where(m)[0])

    def sequences(self, split: str, lookback: int = 60, target_kind: str = "retn") -> Dict[str, np.ndarray]:
        """Sliding windows that never cross a session. X_seq (N,L,F), X_static (N,S), X_known (N,H,K), Y (N,H), row_idx (N,)."""
        m = self.masks[split]
        F = self.features.to_numpy(dtype=self.cfg.dtype)
        cols = [f"y_{target_kind}_{h}" for h in self.cfg.horizon_names()]
        Yall = self.targets[cols].to_numpy(dtype=self.cfg.dtype)
        static_all = np.nan_to_num(self.features_raw.reindex(columns=STATIC_COVARIATES).to_numpy(dtype=self.cfg.dtype))
        mtc = self.meta["minutes_to_close"].to_numpy(float)
        tot = 390.0
        hs = [HORIZON_MINUTES[h] for h in self.cfg.horizon_names()]
        Xs, Ss, Ks, Ys, idx = [], [], [], [], []
        for blk in self.meta.groupby("session_date", sort=True).indices.values():
            blk = np.asarray(blk)
            if len(blk) < lookback:
                continue
            ends = blk[lookback - 1:]
            sel = m[ends]
            if not sel.any():
                continue
            win = np.lib.stride_tricks.sliding_window_view(F[blk], (lookback, F.shape[1]))[:, 0][sel]
            e = ends[sel]
            ahead = np.stack([np.zeros(len(e)) if h is None else np.clip(mtc[e] - h, 0, None) for h in hs], 1)
            frac = 1.0 - ahead / tot
            Ks.append(np.stack([ahead, frac, np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], 2))
            Xs.append(win); Ys.append(Yall[e]); Ss.append(static_all[e]); idx.append(e)
        H, S, Kn = len(cols), len(STATIC_COVARIATES), len(KNOWN_FUTURE)
        if not Xs:
            return {"X_seq": np.zeros((0, lookback, F.shape[1]), self.cfg.dtype), "X_static": np.zeros((0, S), self.cfg.dtype),
                    "X_known": np.zeros((0, H, Kn), self.cfg.dtype), "Y": np.zeros((0, H), self.cfg.dtype), "row_idx": np.zeros(0, int)}
        out = {"X_seq": np.concatenate(Xs).astype(self.cfg.dtype), "X_static": np.concatenate(Ss).astype(self.cfg.dtype),
               "X_known": np.concatenate(Ks).astype(self.cfg.dtype), "Y": np.concatenate(Ys).astype(self.cfg.dtype),
               "row_idx": np.concatenate(idx)}
        assert out["X_seq"].shape[2] == self.features.shape[1], "TFT feature axis must equal tabular feature count"
        assert out["X_known"].shape[1:] == (H, Kn) and out["Y"].shape[1] == H, "horizon axis mismatch"
        return out


class FeaturePipeline:
    """Phase 1 front door: fit_transform on history, transform on the live buffer with the fitted scaler."""

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.scaler: Optional[FeatureScaler] = None

    def _engineer(self, spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame], flow_df: Optional[pd.DataFrame]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        spot = coerce_frame(spot_df, ("ts", "spot"), (), cfg, "spot_df")
        if spot is None:
            raise ValueError("spot_df is required")
        chain = coerce_frame(chain_df, CHAIN_REQUIRED, CHAIN_OPTIONAL, cfg, "chain_df")
        flow = coerce_frame(flow_df, FLOW_REQUIRED, FLOW_OPTIONAL, cfg, "flow_df")
        grid = build_spot_grid(spot, cfg)
        gex, prem, px = gex_profile_features(chain, grid, cfg), premium_flow_features(chain, flow, grid, cfg), price_context_features(grid, cfg)
        feats = pd.concat([grid[["minutes_since_open", "minutes_to_close", "session_frac", "session_sin", "session_cos", "dow"]],
                           px, gex, prem], axis=1).replace([np.inf, -np.inf], np.nan)
        feats["has_gex_snapshot"] = gex["gex_net_total"].notna().astype(float)
        feats["has_flow_prints"] = prem["inst_block_weight"].notna().astype(float)
        targets = build_targets(grid, px["rv_1m"], cfg)
        meta = grid[["ts", "session_date", "spot", "minutes_since_open", "minutes_to_close"]]
        log_shapes("phase1.engineered", meta=meta, features_raw=feats, targets=targets)
        return meta, feats, targets

    def fit_transform(self, spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame] = None,
                      flow_df: Optional[pd.DataFrame] = None) -> PipelineOutput:
        meta, feats, targets = self._engineer(spot_df, chain_df, flow_df)
        masks = chronological_split(meta["session_date"], self.cfg)
        self.scaler = FeatureScaler(self.cfg).fit(feats[masks["train"]])
        scaled = self.scaler.transform(feats)
        log.info("phase1.split", extra={"event": "split", **{k: int(meta.loc[v, "session_date"].nunique()) for k, v in masks.items()},
                                        "n_features": int(scaled.shape[1])})
        return PipelineOutput(*(x.reset_index(drop=True) for x in (meta, feats, scaled, targets)), masks, self.scaler, self.cfg)

    def transform(self, spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame] = None,
                  flow_df: Optional[pd.DataFrame] = None) -> PipelineOutput:
        if self.scaler is None:
            raise RuntimeError("fit_transform before transform")
        meta, feats, targets = self._engineer(spot_df, chain_df, flow_df)
        n = len(meta)
        masks = {"train": np.zeros(n, bool), "val": np.zeros(n, bool), "test": np.ones(n, bool), "embargo": np.zeros(n, bool)}
        return PipelineOutput(*(x.reset_index(drop=True) for x in (meta, feats, self.scaler.transform(feats), targets)), masks, self.scaler, self.cfg)


def verify_causality(pipe: FeaturePipeline, spot: pd.DataFrame, chain: pd.DataFrame, flow: pd.DataFrame,
                     cut_minutes: int = 200, atol: float = 1e-6) -> bool:
    """Leakage test: features at bar t computed from the FULL day must equal features computed from a tape truncated at t."""
    last_day = pd.to_datetime(spot["ts"]).max().normalize()
    def _day(df: pd.DataFrame) -> pd.DataFrame:
        ts = to_session_tz(df["ts"], pipe.cfg.tz)
        return df[ts.dt.normalize() == last_day]
    s, c, f = _day(spot), _day(chain), _day(flow)
    full = pipe.transform(s, c, f)
    cut_ts = full.meta["ts"].iloc[cut_minutes]
    trunc = pipe.transform(s[to_session_tz(s["ts"]) <= cut_ts], c[to_session_tz(c["ts"]) <= cut_ts], f[to_session_tz(f["ts"]) <= cut_ts])
    a = full.features_raw.iloc[cut_minutes].to_numpy(float)
    b = trunc.features_raw.iloc[cut_minutes].to_numpy(float)
    ok = np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=atol, rtol=1e-5)
    log.info("phase1.causality_check", extra={"event": "causality", "bar": str(cut_ts), "passed": bool(ok),
                                             "max_abs_diff": float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b))))})
    return bool(ok)


# ============================================================================= #
# PHASE 2 — CORE MODELING ENGINE
# ============================================================================= #
@dataclass
class TreeConfig:
    objective: str = "pseudo_huber"          # pseudo_huber | cauchy | huber | l2
    robust_delta: float = 1.0
    winsor_q: Tuple[float, float] = (0.005, 0.995)
    n_rounds_max: int = 4000
    early_stopping_rounds: int = 150
    learning_rate: float = 0.02
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 60
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.5
    lambda_l2: float = 5.0
    path_smooth: float = 1.0
    cat_depth: int = 6
    cat_l2_leaf_reg: float = 6.0
    cat_random_strength: float = 1.5
    cat_bagging_temperature: float = 0.5
    cat_custom_objective: bool = False
    cat_task_type: str = "CPU"
    n_folds: int = 3
    embargo_groups: int = 1
    min_train_groups: int = 3
    fallback_val_frac: float = 0.2
    fallback_embargo_rows: int = 30


@dataclass
class TFTConfig:
    d_model: int = 32
    n_heads: int = 4
    lstm_layers: int = 1
    dropout: float = 0.15
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 120
    patience: int = 12
    grad_clip: float = 1.0
    val_frac_groups: float = 0.2
    embargo_groups: int = 1
    seed: int = 7
    device: str = "auto"


@dataclass
class EngineConfig:
    horizons: Tuple[str, ...] = HORIZONS
    tree_primary_horizons: Tuple[str, ...] = ("5m", "15m", "30m")
    deep_primary_horizons: Tuple[str, ...] = ("30m", "1h", "eod")
    fit_trees_on_all_horizons: bool = True
    tree: TreeConfig = field(default_factory=TreeConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    seed: int = 7
    n_threads: int = -1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "EngineConfig":
        d = json.loads(s)
        d["tree"], d["tft"] = TreeConfig(**d["tree"]), TFTConfig(**d["tft"])
        for k in ("horizons", "tree_primary_horizons", "deep_primary_horizons"):
            d[k] = tuple(d[k])
        d["tree"].winsor_q, d["tft"].quantiles = tuple(d["tree"].winsor_q), tuple(d["tft"].quantiles)
        return cls(**d)


def _is_third_friday(d: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((d.weekday == 4) & (d.day >= 15) & (d.day <= 21))


@dataclass
class TrainingBundle:
    """Aligned tabular + sequence inputs for one row set. Static tensor = Phase 1 statics + [is_monthly_opex, is_quad_witching]."""
    X: np.ndarray
    Y: np.ndarray
    groups: np.ndarray
    row_idx: np.ndarray
    feature_names: List[str]
    seq: Optional[Dict[str, np.ndarray]] = None
    static_names: List[str] = field(default_factory=lambda: list(STATIC_COVARIATES) + ["is_monthly_opex", "is_quad_witching"])
    known_names: List[str] = field(default_factory=lambda: list(KNOWN_FUTURE))

    @classmethod
    def from_phase1(cls, out: PipelineOutput, split: str, lookback: int = 60, target_kind: str = "retn",
                    with_sequences: bool = True) -> "TrainingBundle":
        X, Y, row_idx = out.tabular_multi(split, target_kind)
        sess = pd.DatetimeIndex(out.meta["session_date"].iloc[row_idx])
        groups = sess.tz_convert("UTC").tz_localize(None).asi8 if sess.tz is not None else sess.asi8
        seq = None
        if with_sequences:
            seq = out.sequences(split, lookback=lookback, target_kind=target_kind)
            sd = pd.DatetimeIndex(out.meta["session_date"].iloc[seq["row_idx"]]) if len(seq["row_idx"]) else pd.DatetimeIndex([])
            tf = _is_third_friday(sd).astype(np.float32)
            qw = (tf.astype(bool) & np.asarray(sd.month.isin([3, 6, 9, 12]))).astype(np.float32)
            seq["X_static"] = np.concatenate([seq["X_static"], np.stack([tf, qw], 1).reshape(len(sd), 2)], 1).astype(np.float32)
            seq["groups"] = (sd.tz_convert("UTC").tz_localize(None).asi8 if sd.tz is not None else sd.asi8) if len(sd) else np.zeros(0, np.int64)
        b = cls(X=X.astype(np.float32), Y=Y.astype(np.float32), groups=np.asarray(groups, np.int64), row_idx=np.asarray(row_idx, int),
                feature_names=list(out.features.columns), seq=seq)
        log_shapes(f"phase2.bundle[{split}]", X=b.X, Y=b.Y, **{f"seq_{k}": v for k, v in (seq or {}).items()})
        return b


class RollingOriginSplitter:
    """Expanding-window CV over sessions with an embargo; falls back to a single embargoed row split when history is thin."""

    def __init__(self, tc: TreeConfig):
        self.tc = tc

    def split(self, groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        tc = self.tc
        uniq = np.unique(groups)
        G = len(uniq)
        v = max(1, G // (tc.n_folds + 2))
        folds = []
        for k in range(tc.n_folds):
            vs = G - (tc.n_folds - k) * v
            te = vs - tc.embargo_groups
            if te >= tc.min_train_groups:
                folds.append((np.where(np.isin(groups, uniq[:te]))[0], np.where(np.isin(groups, uniq[vs: vs + v]))[0]))
        if folds:
            yield from folds
            return
        order = np.argsort(groups, kind="mergesort")
        n = len(groups)
        cut = n - max(1, int(n * tc.fallback_val_frac))
        log.warning("rolling-origin fallback to row split", extra={"event": "cv_fallback", "sessions": int(G)})
        yield order[: max(1, cut - tc.fallback_embargo_rows)], order[cut:]


def robust_grad_hess(r: np.ndarray, kind: str, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient/Hessian of pseudo-Huber, Cauchy or L2 w.r.t. prediction (r = pred − y)."""
    r = np.asarray(r, np.float64)
    u = (r / delta) ** 2
    if kind == "pseudo_huber":
        return r / np.sqrt(1.0 + u), (1.0 + u) ** -1.5
    if kind == "cauchy":
        return r / (1.0 + u), np.maximum((1.0 - u) / (1.0 + u) ** 2, 1e-3)
    if kind == "l2":
        return r, np.ones_like(r)
    raise ValueError(kind)


def make_lgb_objective(kind: str, delta: float) -> Callable:
    def _obj(preds: np.ndarray, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        g, h = robust_grad_hess(preds - data.get_label(), kind, delta)
        w = data.get_weight()
        return (g * w, h * w) if w is not None else (g, h)
    return _obj


def lgb_tmae(preds: np.ndarray, data: Any) -> Tuple[str, float, bool]:
    e = np.abs(preds - data.get_label())
    lo, hi = np.quantile(e, [0.05, 0.95])
    return "tmae", float(e[(e >= lo) & (e <= hi)].mean()), False


def lgb_dir_acc(preds: np.ndarray, data: Any) -> Tuple[str, float, bool]:
    y = data.get_label()
    m = np.abs(y) > 0.1
    return "dir_acc", float((np.sign(preds[m]) == np.sign(y[m])).mean()) if m.any() else 0.5, True


class CatCauchyObjective:
    """CatBoost custom objective — returns (−∂L/∂f, −∂²L/∂f²) as CatBoost maximises."""

    def __init__(self, delta: float):
        self.delta = delta

    def calc_ders_range(self, approxes, targets, weights):
        g, h = robust_grad_hess(np.asarray(approxes, float) - np.asarray(targets, float), "cauchy", self.delta)
        if weights is not None:
            w = np.asarray(weights, float)
            g, h = g * w, h * w
        return list(zip((-g).tolist(), (-h).tolist()))


class TabularTreeLayer:
    """LightGBM + CatBoost per horizon; robust objective; TRAIN-only winsorisation; rolling-origin early stopping;
    final refit on all rows at the median best iteration."""

    def __init__(self, cfg: EngineConfig):
        self.cfg, self.tc = cfg, cfg.tree
        self.horizons = list(cfg.horizons) if cfg.fit_trees_on_all_horizons else list(cfg.tree_primary_horizons)
        self.lgb_models: Dict[str, Any] = {}
        self.cat_models: Dict[str, Any] = {}
        self.offsets: Dict[str, float] = {}
        self.winsor: Dict[str, Tuple[float, float]] = {}
        self.history: Dict[str, Any] = {"lgb": {}, "cat": {}}
        self.feature_names: List[str] = []

    def _lgb_params(self) -> Dict[str, Any]:
        tc = self.tc
        p = dict(learning_rate=tc.learning_rate, num_leaves=tc.num_leaves, max_depth=tc.max_depth, min_data_in_leaf=tc.min_data_in_leaf,
                 feature_fraction=tc.feature_fraction, bagging_fraction=tc.bagging_fraction, bagging_freq=tc.bagging_freq,
                 lambda_l1=tc.lambda_l1, lambda_l2=tc.lambda_l2, path_smooth=tc.path_smooth, verbosity=-1, seed=self.cfg.seed,
                 num_threads=self.cfg.n_threads, metric="None", boost_from_average=False)
        if tc.objective == "huber":
            p["objective"], p["alpha"] = "huber", tc.robust_delta
        else:
            p["objective"] = make_lgb_objective(tc.objective, tc.robust_delta)
        return p

    def _cat_model(self, iterations: int, od_wait: Optional[int]) -> Any:
        tc = self.tc
        loss: Any = CatCauchyObjective(tc.robust_delta) if tc.cat_custom_objective else (
            "RMSE" if tc.objective == "l2" else f"Huber:delta={tc.robust_delta}")
        kw = dict(iterations=iterations, learning_rate=tc.learning_rate * 1.5, depth=tc.cat_depth, l2_leaf_reg=tc.cat_l2_leaf_reg,
                  random_strength=tc.cat_random_strength, bootstrap_type="Bayesian", bagging_temperature=tc.cat_bagging_temperature,
                  loss_function=loss, eval_metric="MAE", random_seed=self.cfg.seed, thread_count=self.cfg.n_threads, verbose=False,
                  allow_writing_files=False, task_type=tc.cat_task_type)
        if od_wait is not None:
            kw.update(od_type="Iter", od_wait=od_wait, use_best_model=True)
        return CatBoostRegressor(**kw)

    def fit(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, feature_names: Sequence[str]) -> "TabularTreeLayer":
        self.feature_names = list(feature_names)
        X = np.asarray(X, np.float32)
        assert X.shape[1] == len(self.feature_names), f"X has {X.shape[1]} columns, {len(self.feature_names)} names"
        splitter = RollingOriginSplitter(self.tc)
        for j, h in enumerate(self.cfg.horizons):
            if h not in self.horizons:
                continue
            valid = np.isfinite(Y[:, j]) & np.isfinite(X).all(1)
            Xh, gh, y = X[valid], groups[valid], Y[valid, j].astype(np.float64)
            if len(y) < 50:
                log.warning("too few rows for trees", extra={"event": "trees_skip", "horizon": h, "rows": int(len(y))})
                continue
            lo, hi = np.quantile(y, self.tc.winsor_q)
            self.winsor[h] = (float(lo), float(hi))
            yc = np.clip(y, lo, hi)
            self.offsets[h] = float(np.median(yc))
            yh = yc - self.offsets[h]
            folds = list(splitter.split(gh))
            log.info("trees.fit", extra={"event": "trees_fit", "horizon": h, "rows": int(len(yh)), "folds": len(folds)})
            if _HAS_LGB:
                self.history["lgb"][h], best = {}, []
                for k, (tr, va) in enumerate(folds):
                    dtr = lgb.Dataset(Xh[tr], label=yh[tr], feature_name=self.feature_names, free_raw_data=False)
                    dva = lgb.Dataset(Xh[va], label=yh[va], reference=dtr, free_raw_data=False)
                    ev: Dict[str, Any] = {}
                    bst = lgb.train(self._lgb_params(), dtr, num_boost_round=self.tc.n_rounds_max, valid_sets=[dva], valid_names=["val"],
                                    feval=[lgb_tmae, lgb_dir_acc],
                                    callbacks=[lgb.early_stopping(self.tc.early_stopping_rounds, first_metric_only=True, verbose=False),
                                               lgb.record_evaluation(ev)])
                    bi = int(bst.best_iteration or self.tc.n_rounds_max)
                    best.append(bi)
                    self.history["lgb"][h][f"fold{k}"] = {"best_iteration": bi, "val_tmae": float(ev["val"]["tmae"][bi - 1]),
                                                          "val_dir_acc": float(ev["val"]["dir_acc"][bi - 1])}
                n_final = max(50, int(np.median(best))) if best else 500
                self.lgb_models[h] = lgb.train(self._lgb_params(), lgb.Dataset(Xh, label=yh, feature_name=self.feature_names), num_boost_round=n_final)
                self.history["lgb"][h]["final_rounds"] = n_final
            if _HAS_CAT:
                self.history["cat"][h], best = {}, []
                for k, (tr, va) in enumerate(folds):
                    m = self._cat_model(self.tc.n_rounds_max, self.tc.early_stopping_rounds)
                    m.fit(Pool(Xh[tr], yh[tr], feature_names=self.feature_names), eval_set=Pool(Xh[va], yh[va], feature_names=self.feature_names))
                    bi = int(m.get_best_iteration() or 0) + 1
                    best.append(bi)
                    curve = m.get_evals_result()["validation"]["MAE"]
                    self.history["cat"][h][f"fold{k}"] = {"best_iteration": bi, "val_mae": float(curve[min(bi, len(curve)) - 1])}
                n_final = max(50, int(np.median(best))) if best else 500
                m = self._cat_model(n_final, None)
                m.fit(Pool(Xh, yh, feature_names=self.feature_names))
                self.cat_models[h] = m
                self.history["cat"][h]["final_rounds"] = n_final
        if not (_HAS_LGB or _HAS_CAT):
            log.warning("neither lightgbm nor catboost installed — tree experts will be NaN", extra={"event": "trees_unavailable"})
        return self

    def predict(self, X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        X = np.asarray(X, np.float32)
        n = X.shape[0]
        nan = np.full(n, np.nan, np.float32)
        out = {}
        for h in self.cfg.horizons:
            d = {"lgb": (self.lgb_models[h].predict(X) + self.offsets[h]).astype(np.float32) if h in self.lgb_models else nan,
                 "cat": (self.cat_models[h].predict(X) + self.offsets[h]).astype(np.float32) if h in self.cat_models else nan}
            st = np.stack([d["lgb"], d["cat"]])
            fin = np.isfinite(st)
            cnt = fin.sum(0)
            d["tree_mean"] = np.where(cnt > 0, np.where(fin, st, 0.0).sum(0) / np.maximum(cnt, 1), np.nan).astype(np.float32)
            out[h] = d
        return out

    def feature_importance(self) -> pd.DataFrame:
        rows = {}
        for h in self.cfg.horizons:
            if h in self.lgb_models:
                g = self.lgb_models[h].feature_importance(importance_type="gain")
                rows[f"lgb_{h}"] = safe_div(g, g.sum(), 0.0)
            if h in self.cat_models:
                g = np.asarray(self.cat_models[h].get_feature_importance())
                rows[f"cat_{h}"] = safe_div(g, g.sum(), 0.0)
        return pd.DataFrame(rows, index=self.feature_names)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for h, m in self.lgb_models.items():
            m.save_model(str(d / f"lgb_{h}.txt"))
        for h, m in self.cat_models.items():
            m.save_model(str(d / f"cat_{h}.cbm"))
        (d / "trees.json").write_text(json.dumps({"offsets": self.offsets, "winsor": self.winsor, "horizons": self.horizons,
                                                  "feature_names": self.feature_names, "history": self.history}, default=float))

    def load(self, d: Path) -> "TabularTreeLayer":
        if not (d / "trees.json").exists():
            return self
        meta = json.loads((d / "trees.json").read_text())
        self.offsets = {k: float(v) for k, v in meta["offsets"].items()}
        self.winsor = {k: tuple(v) for k, v in meta["winsor"].items()}
        self.horizons, self.feature_names, self.history = meta["horizons"], meta["feature_names"], meta["history"]
        for h in self.horizons:
            if _HAS_LGB and (d / f"lgb_{h}.txt").exists():
                self.lgb_models[h] = lgb.Booster(model_file=str(d / f"lgb_{h}.txt"))
            if _HAS_CAT and (d / f"cat_{h}.cbm").exists():
                m = CatBoostRegressor()
                m.load_model(str(d / f"cat_{h}.cbm"))
                self.cat_models[h] = m
        return self


# ----------------------------------------------------------------------------- #
# Temporal Fusion Transformer (native PyTorch)
# ----------------------------------------------------------------------------- #
if _HAS_TORCH:

    class GatedLinearUnit(nn.Module):
        def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
            super().__init__()
            self.fc, self.drop = nn.Linear(d_in, 2 * d_out), nn.Dropout(dropout)

        def forward(self, x):
            a, b = self.fc(self.drop(x)).chunk(2, dim=-1)
            return a * torch.sigmoid(b)

    class GatedResidualNetwork(nn.Module):
        def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float, d_ctx: Optional[int] = None):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.ctx = nn.Linear(d_ctx, d_hidden, bias=False) if d_ctx else None
            self.fc2 = nn.Linear(d_hidden, d_hidden)
            self.glu = GatedLinearUnit(d_hidden, d_out, dropout)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.norm = nn.LayerNorm(d_out)

        def forward(self, x, c=None):
            h = self.fc1(x)
            if c is not None and self.ctx is not None:
                while c.dim() < h.dim():
                    c = c.unsqueeze(1)
                h = h + self.ctx(c)
            return self.norm(self.skip(x) + self.glu(self.fc2(Fn.elu(h))))

    class BatchedGRN(nn.Module):
        """One GRN per variable in a single einsum: (..., V, d) → (..., V, d)."""

        def __init__(self, n_vars: int, d: int, dropout: float):
            super().__init__()
            k = 1.0 / math.sqrt(d)
            self.W1 = nn.Parameter(torch.empty(n_vars, d, d).uniform_(-k, k)); self.b1 = nn.Parameter(torch.zeros(n_vars, d))
            self.W2 = nn.Parameter(torch.empty(n_vars, d, d).uniform_(-k, k)); self.b2 = nn.Parameter(torch.zeros(n_vars, d))
            self.Wg = nn.Parameter(torch.empty(n_vars, d, 2 * d).uniform_(-k, k)); self.bg = nn.Parameter(torch.zeros(n_vars, 2 * d))
            self.drop, self.norm = nn.Dropout(dropout), nn.LayerNorm(d)

        def forward(self, x):
            h = Fn.elu(torch.einsum("...vd,vdh->...vh", x, self.W1) + self.b1)
            h = torch.einsum("...vd,vdh->...vh", h, self.W2) + self.b2
            a, b = (torch.einsum("...vd,vdh->...vh", self.drop(h), self.Wg) + self.bg).chunk(2, dim=-1)
            return self.norm(x + a * torch.sigmoid(b))

    class VariableSelectionNetwork(nn.Module):
        def __init__(self, n_vars: int, d: int, dropout: float, d_ctx: Optional[int] = None):
            super().__init__()
            self.flat_grn = GatedResidualNetwork(n_vars * d, d, n_vars, dropout, d_ctx)
            self.var_grn = BatchedGRN(n_vars, d, dropout)

        def forward(self, emb, c=None):
            w = torch.softmax(self.flat_grn(emb.flatten(start_dim=-2), c), dim=-1)
            return (self.var_grn(emb) * w.unsqueeze(-1)).sum(dim=-2), w

    class VariableEmbedding(nn.Module):
        def __init__(self, n_vars: int, d: int):
            super().__init__()
            self.W, self.b = nn.Parameter(torch.randn(n_vars, d) * 0.1), nn.Parameter(torch.zeros(n_vars, d))

        def forward(self, x):
            return x.unsqueeze(-1) * self.W + self.b

    class InterpretableMultiHeadAttention(nn.Module):
        def __init__(self, d: int, n_heads: int, dropout: float):
            super().__init__()
            self.h, self.dk = n_heads, d // n_heads
            self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(n_heads)])
            self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(n_heads)])
            self.v, self.out, self.drop = nn.Linear(d, self.dk), nn.Linear(self.dk, d), nn.Dropout(dropout)

        def forward(self, q, kv, mask):
            v = self.v(kv)
            heads, attns = [], []
            for i in range(self.h):
                s = torch.einsum("nqd,nkd->nqk", self.q[i](q), self.k[i](kv)) / math.sqrt(self.dk)
                a = self.drop(torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1))
                heads.append(torch.einsum("nqk,nkd->nqd", a, v)); attns.append(a)
            return self.out(torch.stack(heads).mean(0)), torch.stack(attns).mean(0)

    class TemporalFusionTransformer(nn.Module):
        """forward(x_past (N,L,F), x_static (N,S), x_known (N,H,K)) → quantiles (N,H,Q) + VSN weights + attention.
        Quantiles are non-crossing: median direct, other quantiles cumulative softplus offsets."""

        def __init__(self, n_past: int, n_static: int, n_known: int, n_horizons: int, cfg: TFTConfig):
            super().__init__()
            d, p = cfg.d_model, cfg.dropout
            self.cfg, self.dims = cfg, (n_past, n_static, n_known, n_horizons)
            self.quantiles = sorted(cfg.quantiles)
            self.med_idx = self.quantiles.index(min(self.quantiles, key=lambda q: abs(q - 0.5)))
            self.emb_past, self.emb_static, self.emb_known = VariableEmbedding(n_past, d), VariableEmbedding(n_static, d), VariableEmbedding(n_known, d)
            self.horizon_emb = nn.Embedding(n_horizons, d)
            self.vsn_static = VariableSelectionNetwork(n_static, d, p)
            self.vsn_past = VariableSelectionNetwork(n_past, d, p, d_ctx=d)
            self.vsn_known = VariableSelectionNetwork(n_known, d, p, d_ctx=d)
            self.ctx_selection, self.ctx_enrich = GatedResidualNetwork(d, d, d, p), GatedResidualNetwork(d, d, d, p)
            self.ctx_h, self.ctx_c = GatedResidualNetwork(d, d, d, p), GatedResidualNetwork(d, d, d, p)
            dl = p if cfg.lstm_layers > 1 else 0.0
            self.lstm_enc = nn.LSTM(d, d, num_layers=cfg.lstm_layers, batch_first=True, dropout=dl)
            self.lstm_dec = nn.LSTM(d, d, num_layers=cfg.lstm_layers, batch_first=True, dropout=dl)
            self.post_lstm_gate, self.post_lstm_norm = GatedLinearUnit(d, d, p), nn.LayerNorm(d)
            self.enrich = GatedResidualNetwork(d, d, d, p, d_ctx=d)
            self.attn = InterpretableMultiHeadAttention(d, cfg.n_heads, p)
            self.post_attn_gate, self.post_attn_norm = GatedLinearUnit(d, d, p), nn.LayerNorm(d)
            self.pos_ff = GatedResidualNetwork(d, d, d, p)
            self.out_gate, self.out_norm = GatedLinearUnit(d, d, p), nn.LayerNorm(d)
            self.head = nn.Linear(d, len(self.quantiles))

        def _assemble(self, raw):
            med = raw[..., self.med_idx: self.med_idx + 1]
            outs, Q = [None] * raw.shape[-1], raw.shape[-1]
            outs[self.med_idx] = med
            acc = med
            for i in range(self.med_idx + 1, Q):
                acc = acc + Fn.softplus(raw[..., i: i + 1]); outs[i] = acc
            acc = med
            for i in range(self.med_idx - 1, -1, -1):
                acc = acc - Fn.softplus(raw[..., i: i + 1]); outs[i] = acc
            return torch.cat(outs, dim=-1)

        def forward(self, x_past, x_static, x_known):
            N, L, F = x_past.shape
            H = x_known.shape[1]
            assert F == self.dims[0] and x_static.shape[1] == self.dims[1] and x_known.shape[2] == self.dims[2], \
                f"TFT input dims {(F, x_static.shape[1], x_known.shape[2])} != model dims {self.dims[:3]}"
            s_vec, w_static = self.vsn_static(self.emb_static(x_static))
            c_sel, c_enr, c_h, c_c = self.ctx_selection(s_vec), self.ctx_enrich(s_vec), self.ctx_h(s_vec), self.ctx_c(s_vec)
            p_vec, w_past = self.vsn_past(self.emb_past(x_past), c_sel)
            k_vec, w_known = self.vsn_known(self.emb_known(x_known), c_sel)
            k_vec = k_vec + self.horizon_emb(torch.arange(H, device=x_known.device)).unsqueeze(0)
            nl = self.cfg.lstm_layers
            enc, state = self.lstm_enc(p_vec, (c_h.unsqueeze(0).repeat(nl, 1, 1).contiguous(), c_c.unsqueeze(0).repeat(nl, 1, 1).contiguous()))
            dec, _ = self.lstm_dec(k_vec, state)
            temporal = self.post_lstm_norm(torch.cat([p_vec, k_vec], 1) + self.post_lstm_gate(torch.cat([enc, dec], 1)))
            enriched = self.enrich(temporal, c_enr)
            q = enriched[:, L:]
            mask = torch.ones(H, L + H, dtype=torch.bool, device=x_past.device).tril(diagonal=L)
            attn_out, attn_w = self.attn(q, enriched, mask.unsqueeze(0).expand(N, -1, -1))
            x = self.post_attn_norm(q + self.post_attn_gate(attn_out))
            x = self.out_norm(temporal[:, L:] + self.out_gate(self.pos_ff(x)))
            return {"quantiles": self._assemble(self.head(x)), "w_past": w_past, "w_static": w_static, "w_known": w_known, "attention": attn_w}

    def masked_pinball_loss(pred, target, quantiles: Sequence[float]):
        valid = torch.isfinite(target)
        t = torch.where(valid, target, torch.zeros_like(target)).unsqueeze(-1)
        q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
        e = t - pred
        m = valid.unsqueeze(-1).to(pred.dtype)
        return (torch.maximum(q * e, (q - 1) * e) * m).sum() / (m.sum() * len(quantiles) + EPS)


class DeepSequenceLayer:
    """Trains / serves the TFT with TRAIN-only robust target standardisation and embargoed session validation."""

    def __init__(self, cfg: EngineConfig):
        self.cfg, self.tc = cfg, cfg.tft
        self.model = None
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.dims: Dict[str, int] = {}
        self.history: Dict[str, List[float]] = {"train": [], "val": [], "lr": []}
        self.best_epoch = -1
        self.past_names: List[str] = []
        self.static_names: List[str] = []
        self.known_names: List[str] = []

    def _device(self):
        return torch.device("cuda" if self.tc.device == "auto" and torch.cuda.is_available() else ("cpu" if self.tc.device == "auto" else self.tc.device))

    def _split(self, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uniq = np.unique(groups)
        G = len(uniq)
        n_val = max(1, int(round(G * self.tc.val_frac_groups)))
        if G - n_val - self.tc.embargo_groups < 1:
            order = np.argsort(groups, kind="mergesort")
            cut = int(len(groups) * 0.8)
            return order[: max(1, cut - 30)], order[cut:]
        return np.where(np.isin(groups, uniq[: G - n_val - self.tc.embargo_groups]))[0], np.where(np.isin(groups, uniq[G - n_val:]))[0]

    def _build(self):
        return TemporalFusionTransformer(self.dims["n_past"], self.dims["n_static"], self.dims["n_known"], self.dims["n_horizons"], self.tc)

    def fit(self, seq: Dict[str, np.ndarray], groups: np.ndarray, past_names: Sequence[str], static_names: Sequence[str],
            known_names: Sequence[str]) -> "DeepSequenceLayer":
        if not _HAS_TORCH:
            log.warning("torch unavailable — deep layer skipped", extra={"event": "tft_unavailable"})
            return self
        torch.manual_seed(self.tc.seed); np.random.seed(self.tc.seed)
        dev = self._device()
        Xp, Xs, Xk, Y = (np.asarray(seq["X_seq"], np.float32), np.nan_to_num(np.asarray(seq["X_static"], np.float32)),
                         np.asarray(seq["X_known"], np.float32), np.asarray(seq["Y"], np.float32))
        assert Xp.shape[2] == len(past_names) and Xs.shape[1] == len(static_names) and Xk.shape[2] == len(known_names), "TFT name/tensor width mismatch"
        self.past_names, self.static_names, self.known_names = list(past_names), list(static_names), list(known_names)
        self.dims = {"n_past": Xp.shape[2], "n_static": Xs.shape[1], "n_known": Xk.shape[2], "n_horizons": Y.shape[1], "lookback": Xp.shape[1]}
        tr, va = self._split(groups)
        self.center = np.nanmedian(Y[tr], axis=0).astype(np.float32)
        mad = np.nanmedian(np.abs(Y[tr] - self.center), axis=0) * 1.4826
        self.scale = np.where(mad > 1e-6, mad, 1.0).astype(np.float32)
        Yz = ((Y - self.center) / self.scale).astype(np.float32)
        self.model = self._build().to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.tc.lr, weight_decay=self.tc.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

        def batches(idx: np.ndarray, shuffle: bool):
            idx = np.random.permutation(idx) if shuffle else idx
            for i in range(0, len(idx), self.tc.batch_size):
                b = idx[i: i + self.tc.batch_size]
                yield tuple(torch.from_numpy(a[b]).to(dev) for a in (Xp, Xs, Xk, Yz))

        best, bad, best_state = float("inf"), 0, None
        for epoch in range(self.tc.max_epochs):
            self.model.train()
            tot = cnt = 0
            for xp, xs, xk, y in batches(tr, True):
                opt.zero_grad(set_to_none=True)
                loss = masked_pinball_loss(self.model(xp, xs, xk)["quantiles"], y, self.model.quantiles)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
                opt.step()
                tot += float(loss) * len(y); cnt += len(y)
            val_loss = self._evaluate(va, batches)
            sched.step(val_loss)
            self.history["train"].append(tot / max(cnt, 1)); self.history["val"].append(val_loss); self.history["lr"].append(opt.param_groups[0]["lr"])
            log.info("tft.epoch", extra={"event": "tft_epoch", "epoch": epoch, "train": tot / max(cnt, 1), "val": val_loss})
            if val_loss < best - 1e-5:
                best, bad, self.best_epoch = val_loss, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.tc.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _evaluate(self, idx: np.ndarray, batches: Callable) -> float:
        self.model.eval()
        tot = cnt = 0
        with torch.no_grad():
            for xp, xs, xk, y in batches(idx, False):
                tot += float(masked_pinball_loss(self.model(xp, xs, xk)["quantiles"], y, self.model.quantiles)) * len(y); cnt += len(y)
        return tot / max(cnt, 1)

    def predict(self, seq: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.model is None or not len(seq["X_seq"]):
            return {"quantiles": None, "importance": None}
        dev = self._device()
        Xp, Xs, Xk = (np.asarray(seq["X_seq"], np.float32), np.nan_to_num(np.asarray(seq["X_static"], np.float32)), np.asarray(seq["X_known"], np.float32))
        if (Xp.shape[2], Xs.shape[1], Xk.shape[2]) != (self.dims["n_past"], self.dims["n_static"], self.dims["n_known"]):
            log.error("tft.shape_mismatch", extra={"event": "shape_error", "got": [Xp.shape[2], Xs.shape[1], Xk.shape[2]], "expected": self.dims})
            return {"quantiles": None, "importance": None}
        qs, wp, ws, wk, at = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(Xp), self.tc.batch_size):
                o = self.model(*(torch.from_numpy(a[i: i + self.tc.batch_size]).to(dev) for a in (Xp, Xs, Xk)))
                qs.append(o["quantiles"].cpu().numpy()); wp.append(o["w_past"].mean(0).cpu().numpy()); ws.append(o["w_static"].mean(0).cpu().numpy())
                wk.append(o["w_known"].mean(0).cpu().numpy()); at.append(o["attention"].mean(0).cpu().numpy())
        q = np.concatenate(qs) * self.scale[None, :, None] + self.center[None, :, None]
        wpt = np.mean(wp, 0)
        imp = {"past": pd.Series(wpt.mean(0), index=self.past_names).sort_values(ascending=False),
               "past_over_time": pd.DataFrame(wpt, columns=self.past_names), "static": pd.Series(np.mean(ws, 0), index=self.static_names),
               "known": pd.DataFrame(np.mean(wk, 0), columns=self.known_names, index=list(self.cfg.horizons)[: self.dims["n_horizons"]]),
               "attention": np.mean(at, 0)}
        return {"quantiles": q.astype(np.float32), "importance": imp}

    def save(self, d: Path) -> None:
        if self.model is None:
            return
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / "tft.pt")
        (d / "tft.json").write_text(json.dumps({"dims": self.dims, "center": self.center.tolist(), "scale": self.scale.tolist(),
                                                "history": self.history, "best_epoch": self.best_epoch, "past_names": self.past_names,
                                                "static_names": self.static_names, "known_names": self.known_names}))

    def load(self, d: Path) -> "DeepSequenceLayer":
        if not _HAS_TORCH or not (d / "tft.pt").exists():
            return self
        m = json.loads((d / "tft.json").read_text())
        self.dims, self.history, self.best_epoch = m["dims"], m["history"], m["best_epoch"]
        self.center, self.scale = np.array(m["center"], np.float32), np.array(m["scale"], np.float32)
        self.past_names, self.static_names, self.known_names = m["past_names"], m["static_names"], m["known_names"]
        self.model = self._build()
        self.model.load_state_dict(torch.load(d / "tft.pt", map_location="cpu"))
        self.model.to(self._device()).eval()
        return self


class CoreModelingEngine:
    """fit(bundle) / predict(bundle) → {horizon: {lgb, cat, tree_mean, tft_q10, tft_q50, tft_q90, primary}, _meta}, all
    arrays aligned to the bundle's tabular row order (TFT NaN where no full look-back window exists)."""

    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()
        self.trees, self.deep = TabularTreeLayer(self.cfg), DeepSequenceLayer(self.cfg)
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, b: TrainingBundle) -> "CoreModelingEngine":
        self.feature_names = list(b.feature_names)
        log_shapes("phase2.fit", X=b.X, Y=b.Y, **{f"seq_{k}": v for k, v in (b.seq or {}).items()})
        self.trees.fit(b.X, b.Y, b.groups, self.feature_names)
        if b.seq is not None and _HAS_TORCH and len(b.seq["X_seq"]):
            self.deep.fit(b.seq, np.asarray(b.seq["groups"]), self.feature_names, b.static_names, b.known_names)
        else:
            log.warning("TFT not trained (no sequences or no torch)", extra={"event": "tft_skipped"})
        self.fitted = True
        return self

    def predict(self, b: TrainingBundle) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("engine not fitted")
        if list(b.feature_names) != self.feature_names:
            raise ValueError("feature list differs from training — refuse to predict on a misaligned matrix")
        n = b.X.shape[0]
        tree = self.trees.predict(b.X)
        qn = [f"tft_q{int(round(q * 100)):02d}" for q in sorted(self.cfg.tft.quantiles)]
        deep = self.deep.predict(b.seq) if b.seq is not None else {"quantiles": None, "importance": None}
        pos = np.full(0, -1, int)
        if deep["quantiles"] is not None:
            lookup = pd.Series(np.arange(n), index=b.row_idx)
            pos = lookup.reindex(b.seq["row_idx"]).fillna(-1).to_numpy(int)
        out: Dict[str, Any] = {}
        for j, h in enumerate(self.cfg.horizons):
            d: Dict[str, Any] = dict(tree[h])
            for qi, name in enumerate(qn):
                arr = np.full(n, np.nan, np.float32)
                if deep["quantiles"] is not None:
                    ok = pos >= 0
                    arr[pos[ok]] = deep["quantiles"][ok, j, qi]
                d[name] = arr
            d["primary"] = "deep" if h in self.cfg.deep_primary_horizons and h not in self.cfg.tree_primary_horizons else (
                "tree" if h in self.cfg.tree_primary_horizons and h not in self.cfg.deep_primary_horizons else "both")
            out[h] = d
        out["_meta"] = {"row_idx": b.row_idx, "horizons": list(self.cfg.horizons), "importance": deep["importance"],
                        "quantiles": tuple(sorted(self.cfg.tft.quantiles))}
        return out

    @property
    def history(self) -> Dict[str, Any]:
        return {"trees": self.trees.history, "tft": self.deep.history, "tft_best_epoch": self.deep.best_epoch}

    def save(self, path: Union[str, Path]) -> Path:
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        (d / "engine.json").write_text(json.dumps({"cfg": json.loads(self.cfg.to_json()), "feature_names": self.feature_names}))
        self.trees.save(d / "trees"); self.deep.save(d / "tft")
        log.info("engine.saved", extra={"event": "save", "path": str(d)})
        return d

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CoreModelingEngine":
        d = Path(path)
        m = json.loads((d / "engine.json").read_text())
        eng = cls(EngineConfig.from_json(json.dumps(m["cfg"])))
        eng.feature_names = m["feature_names"]
        eng.trees.load(d / "trees"); eng.deep.load(d / "tft")
        eng.fitted = True
        return eng


# ============================================================================= #
# PHASE 3 — OPTIONS PHYSICS + ADAPTIVE GATE
# ============================================================================= #
@dataclass
class MarketState:
    """Live ladder snapshot. Arrays are aligned over strikes; NaNs are zero-masked; implied_move / rv_1m derive from each other."""
    spot: float
    strikes: np.ndarray
    call_oi: np.ndarray
    put_oi: np.ndarray
    gex: np.ndarray
    minutes_to_close: float
    volume: Optional[np.ndarray] = None
    implied_move: Optional[float] = None
    net_prem_z: float = 0.0
    rv_1m: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        self.spot = float(self.spot)
        if not np.isfinite(self.spot) or self.spot <= 0:
            raise ValueError("spot must be a positive finite price")
        self.strikes = np.asarray(self.strikes, float)
        ok = np.isfinite(self.strikes)
        self.strikes = self.strikes[ok]
        for nm in ("call_oi", "put_oi", "gex"):
            a = np.nan_to_num(np.asarray(getattr(self, nm), float))
            if len(a) != len(ok):
                raise ValueError(f"{nm} length {len(a)} != strikes length {len(ok)}")
            setattr(self, nm, a[ok])
        if self.volume is not None:
            v = np.nan_to_num(np.asarray(self.volume, float))
            self.volume = v[ok] if len(v) == len(ok) else None
        self.minutes_to_close = float(max(self.minutes_to_close, 0.0))
        self.net_prem_z = float(self.net_prem_z) if np.isfinite(self.net_prem_z) else 0.0
        mtc = max(self.minutes_to_close, 1.0)
        if self.implied_move is None or not np.isfinite(self.implied_move) or self.implied_move <= 0:
            self.implied_move = float(self.spot * self.rv_1m * math.sqrt(mtc)) if (self.rv_1m and np.isfinite(self.rv_1m) and self.rv_1m > 0) else 0.01 * self.spot
        if self.rv_1m is None or not np.isfinite(self.rv_1m) or self.rv_1m <= 0:
            self.rv_1m = float(self.implied_move / (self.spot * math.sqrt(mtc)))


@dataclass
class PhysicsConfig:
    top_k_clusters: int = 5
    cluster_merge_points: float = 1.0
    valley_ratio: float = 0.6
    mass_w_gex: float = 0.60
    mass_w_oi: float = 0.25
    mass_w_vol: float = 0.15
    band_points: float = 3.0
    sigma_floor_points: float = 0.35
    sigma_im_mult: float = 0.60
    spot_anchor_mass: float = 0.35
    mean_shift_iters: int = 40
    mean_shift_tol: float = 1e-4
    snap_enabled: bool = False               # audit 2026-09-01: OFF by default — on real 8/31 SPY the snap
                                             # dragged the EOD call $0.35 off a close the unrounded blend
                                             # had within $0.05. Pin strike/strength are still REPORTED;
                                             # set True only after the snap beats the blend out of sample.
    snap_grid: float = 0.5
    snap_radius_im_mult: float = 0.35
    snap_min_mass_share: float = 0.22
    snap_full_minutes: float = 30.0
    snap_ramp_minutes: float = 10.0
    negative_gamma_damping: float = 0.55
    negative_gamma_sigma_mult: float = 1.6
    short_horizon_mass_decay: float = 0.55


class OptionsPhysicsEngine:
    """Deterministic overlay. Everything is NumPy over the ladder; the five horizon mean-shifts run as one vector."""

    def __init__(self, cfg: Optional[PhysicsConfig] = None):
        self.cfg = cfg or PhysicsConfig()

    @staticmethod
    def zero_gamma_level(strikes: np.ndarray, gex: np.ndarray, spot: float) -> float:
        o = np.argsort(strikes)
        return float(zero_gamma_matrix(strikes[o], gex[o][None, :], np.array([spot]))[0])

    @staticmethod
    def max_pain(strikes: np.ndarray, call_oi: np.ndarray, put_oi: np.ndarray) -> float:
        if len(strikes) < 2:
            return float("nan")
        return float(max_pain_matrix(strikes, call_oi[None, :], put_oi[None, :])[0])

    def attractor_masses(self, st: MarketState) -> np.ndarray:
        c = self.cfg
        absg, oi = np.abs(st.gex), st.call_oi + st.put_oi
        m = c.mass_w_gex * safe_div(absg, absg.sum(), 0.0) + c.mass_w_oi * safe_div(oi, oi.sum(), 0.0)
        if st.volume is not None and st.volume.sum() > 0:
            m = m + c.mass_w_vol * safe_div(st.volume, st.volume.sum(), 0.0)
        tot = m.sum()
        return m / tot if tot > EPS else np.full(len(m), 1.0 / max(len(m), 1))

    def clusters(self, st: MarketState, masses: np.ndarray) -> pd.DataFrame:
        """|GEX| clusters, fully vectorised: boundaries at ladder gaps or local valleys, aggregates via bincount."""
        c = self.cfg
        o = np.argsort(st.strikes)
        k, g, m = st.strikes[o], st.gex[o], masses[o]
        a = np.abs(g)
        n = len(k)
        if n == 0:
            return pd.DataFrame(columns=["center", "peak_strike", "mass", "gex_sum", "abs_gex_sum", "n_strikes", "sign", "dist"])
        gap = np.r_[False, np.diff(k) > c.cluster_merge_points]
        prev, nxt = np.r_[np.inf, a[:-1]], np.r_[a[1:], np.inf]
        valley_at = (a < prev) & (a <= nxt) & (a < c.valley_ratio * np.minimum(prev, nxt))   # strike i is a valley floor
        start = gap | np.r_[False, valley_at[:-1]]                                             # new cluster begins after a floor
        ids = np.cumsum(start)
        nc = ids.max() + 1
        w = m + EPS
        mass, gsum, asum, cnt = np.bincount(ids, m, nc), np.bincount(ids, g, nc), np.bincount(ids, a, nc), np.bincount(ids, None, nc)
        center = np.bincount(ids, k * w, nc) / np.bincount(ids, w, nc)
        order = np.lexsort((a, ids))                                                           # sort by id then |gex|
        last_per_cluster = order[np.r_[np.diff(ids[order]) != 0, True]]
        peak = k[last_per_cluster]
        df = pd.DataFrame({"center": center, "peak_strike": peak, "mass": mass, "gex_sum": gsum, "abs_gex_sum": asum,
                           "n_strikes": cnt.astype(int), "sign": np.sign(gsum).astype(int)})
        df["dist"] = df["center"] - st.spot
        return df.sort_values("abs_gex_sum", ascending=False).head(c.top_k_clusters).reset_index(drop=True)

    def profile(self, st: MarketState) -> Dict[str, Any]:
        c = self.cfg
        masses = self.attractor_masses(st)
        zg, mp = self.zero_gamma_level(st.strikes, st.gex, st.spot), self.max_pain(st.strikes, st.call_oi, st.put_oi)
        band = np.abs(st.strikes - st.spot) <= c.band_points
        above, below, pos = st.strikes > st.spot, st.strikes < st.spot, st.gex > 0

        def _wall(mask):
            sel = mask & pos
            if not sel.any():
                return float("nan"), 0.0, 0.0
            i = np.argmax(np.where(sel, st.gex, -np.inf))
            return float(st.strikes[i]), float(st.gex[i]), float(masses[i])

        wa, wa_g, wa_m = _wall(above)
        wb, wb_g, wb_m = _wall(below)
        neg = bool(np.isfinite(zg) and st.spot < zg)
        return {"zero_gamma_level": zg, "dist_to_zero_gamma": st.spot - zg if np.isfinite(zg) else float("nan"),
                "regime": "negative_gamma" if neg else "positive_gamma", "max_pain": mp,
                "dist_to_max_pain": mp - st.spot if np.isfinite(mp) else float("nan"),
                "gex_net_total": float(st.gex.sum()), "gex_band_net": float(st.gex[band].sum()) if band.any() else 0.0,
                "band_mass_share": float(masses[band].sum()) if band.any() else 0.0,
                "wall_above": wa, "wall_above_gex": wa_g, "wall_above_mass": wa_m, "wall_below": wb, "wall_below_gex": wb_g, "wall_below_mass": wb_m,
                "peak_gex_strike": float(st.strikes[np.argmax(np.abs(st.gex))]) if len(st.strikes) else float("nan"),
                "clusters": self.clusters(st, masses), "masses": masses}

    def time_ramp(self, minutes_to_close: float) -> float:
        c = self.cfg
        return float(1.0 / (1.0 + math.exp(np.clip((minutes_to_close - c.snap_full_minutes) / max(c.snap_ramp_minutes, 1e-3), -60, 60))))

    def targets(self, st: MarketState, ml_consensus: Dict[str, float], prof: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Physics target per horizon (vector mean-shift from the ML consensus) plus the EOD pin correction."""
        c = self.cfg
        prof = prof or self.profile(st)
        masses, neg = prof["masses"], prof["regime"] == "negative_gamma"
        mtc = max(st.minutes_to_close, 1.0)
        hm = np.array([mtc if HORIZON_MINUTES[h] is None else min(HORIZON_MINUTES[h], mtc) for h in HORIZONS])
        tf = np.sqrt(hm / mtc)
        sigma = np.maximum(c.sigma_im_mult * st.implied_move * np.minimum(tf, 1.0), c.sigma_floor_points) * (c.negative_gamma_sigma_mult if neg else 1.0)
        frac = np.where([HORIZON_MINUTES[h] is None for h in HORIZONS], 1.0, (hm / mtc) ** c.short_horizon_mass_decay)
        ms = (c.negative_gamma_damping if neg else 1.0) * frac                               # (H,)
        x0 = np.array([ml_consensus.get(h, np.nan) for h in HORIZONS], float)
        x0 = np.where(np.isfinite(x0), x0, st.spot)
        K = st.strikes
        x = x0.copy()
        its = 0
        if len(K):
            for its in range(1, c.mean_shift_iters + 1):                                       # (H,K) vector iteration
                d = K[None, :] - x[:, None]
                w = (masses[None, :] * ms[:, None]) * np.exp(-0.5 * (d / sigma[:, None]) ** 2)
                w0 = c.spot_anchor_mass * np.exp(-0.5 * ((x0 - x) / (2.0 * sigma)) ** 2)
                x_new = ((w * K[None, :]).sum(1) + w0 * x0) / (w.sum(1) + w0 + EPS)
                done = np.abs(x_new - x).max() < c.mean_shift_tol
                x = x_new
                if done:
                    break
            d0 = K[None, :] - x0[:, None]
            grav = ((masses[None, :] * ms[:, None]) * np.exp(-0.5 * (d0 / sigma[:, None]) ** 2) * d0).sum(1) / sigma ** 2
        else:
            grav = np.zeros(len(HORIZONS))
        per = {h: {"physics_target": float(x[i]), "gravity": float(grav[i]), "sigma": float(sigma[i]), "mass_scale": float(ms[i]), "start": float(x0[i])}
               for i, h in enumerate(HORIZONS)}
        # ---- EOD pin snap toward the most massive positive-GEX wall inside the capture radius
        xe = per["eod"]["physics_target"]
        radius = max(c.snap_radius_im_mult * st.implied_move, c.snap_grid)
        pin_strike, pin_mass, strength, applied = float("nan"), 0.0, 0.0, False
        pos = st.gex > 0
        if pos.any() and len(K):
            ks, mk = K[pos], masses[pos]
            d = np.abs(ks - xe)
            near = d <= radius
            if near.any():
                j = int(np.argmax(np.where(near, mk, -1.0)))
                band_mass = float(masses[np.abs(K - st.spot) <= c.band_points].sum()) + EPS
                share = float(mk[j] / band_mass)
                if share >= c.snap_min_mass_share:
                    pin_strike = float(round(ks[j] / c.snap_grid) * c.snap_grid)
                    pin_mass = float(mk[j])
                    ramp, prox = self.time_ramp(st.minutes_to_close), 1.0 - min(d[j] / radius, 1.0)
                    strength = float(min(1.0, share / (2.0 * c.snap_min_mass_share)) * (0.35 + 0.65 * ramp) * (0.5 + 0.5 * prox))
                    if neg:
                        strength *= c.negative_gamma_damping
                    if c.snap_enabled:                       # snap is opt-in; the wall is always reported
                        xe = (1.0 - strength) * xe + strength * pin_strike
                        applied = True
        return {"per_horizon": {**per, "eod": {**per["eod"], "physics_target": float(xe)}}, "eod_unsnapped": per["eod"]["physics_target"],
                "eod_target": float(xe), "pin_strike": pin_strike, "pin_mass": pin_mass, "pin_strength": strength, "snap_applied": applied,
                "snap_radius": radius, "time_ramp": self.time_ramp(st.minutes_to_close), "iterations": its}


@dataclass
class GateConfig:
    priors: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        "5m": (0.40, 0.35, 0.15, 0.10), "15m": (0.38, 0.32, 0.20, 0.10), "30m": (0.20, 0.15, 0.50, 0.15),
        "1h": (0.15, 0.12, 0.53, 0.20), "eod": (0.15, 0.10, 0.30, 0.45)})
    eod_physics_max: float = 0.85
    eod_ramp_center_minutes: float = 30.0
    eod_ramp_steepness_minutes: float = 8.0
    pin_strength_boost: float = 0.5
    high_vol_z: float = 3.0
    high_vol_tft_weight: float = 0.0
    high_vol_physics_mult: float = 0.5
    high_vol_eod_physics_floor: float = 0.30
    ridge_alpha: float = 5.0
    shrink_n0: float = 60.0
    min_history_rows: int = 30
    disagreement_scale_im: float = 0.35

    def prior_vector(self, h: str) -> np.ndarray:
        p = np.asarray(self.priors[h], float)
        return p / p.sum()


class MetaGatingNetwork:
    """MoE gate over (lgb, cat, tft, physics): horizon priors × EOD time ramp × regime fallback, blended with a
    non-negative Ridge meta-regressor shrunk by n/(n+n0). Weights are non-negative and sum to one over available experts."""

    def __init__(self, cfg: Optional[GateConfig] = None):
        self.cfg = cfg or GateConfig()
        self.learned_w: Dict[str, np.ndarray] = {}
        self.n_fit: Dict[str, int] = {}
        self.fit_report: Dict[str, Any] = {}

    def prior_weights(self, h: str, minutes_to_close: float, pin_strength: float = 0.0) -> np.ndarray:
        c = self.cfg
        w = c.prior_vector(h).copy()
        if h == "eod":
            ramp = 1.0 / (1.0 + math.exp(np.clip((minutes_to_close - c.eod_ramp_center_minutes) / c.eod_ramp_steepness_minutes, -60, 60)))
            phys = min(1.0, w[3] + (c.eod_physics_max - w[3]) * ramp + c.pin_strength_boost * pin_strength * ramp)
            others = w[:3] / (w[:3].sum() + EPS)
            w = np.concatenate([others * (1.0 - phys), [phys]])
        return w

    def regime_adjust(self, w: np.ndarray, h: str, net_prem_z: float) -> Tuple[np.ndarray, bool]:
        c = self.cfg
        if not np.isfinite(net_prem_z) or abs(net_prem_z) < c.high_vol_z:
            return w, False
        w = w.copy()
        freed = w[2] - c.high_vol_tft_weight
        w[2] = c.high_vol_tft_weight
        new_phys = max(w[3] * c.high_vol_physics_mult, c.high_vol_eod_physics_floor if h == "eod" else 0.0)
        freed += w[3] - new_phys
        w[3] = new_phys
        w[:2] = w[:2] + freed * (w[:2] / (w[:2].sum() + EPS))
        return w / (w.sum() + EPS), True

    def fit(self, history: pd.DataFrame) -> "MetaGatingNetwork":
        """history: horizon, spot, realized, lgb, cat, tft, physics (prices). Learns non-negative simplex weights on deltas."""
        c = self.cfg
        for h in HORIZONS:
            d = history[history["horizon"] == h].dropna(subset=["realized", "spot"])
            if len(d) < c.min_history_rows:
                self.fit_report[h] = {"n": int(len(d)), "status": "priors only"}
                continue
            X = np.column_stack([(d[e] - d["spot"]).fillna(0.0).to_numpy(float) for e in EXPERTS])
            y = (d["realized"] - d["spot"]).to_numpy(float)
            X, y = np.nan_to_num(X), np.nan_to_num(y)
            w = np.clip(Ridge(alpha=c.ridge_alpha, fit_intercept=False, positive=True).fit(X, y).coef_, 0.0, None)
            w = w / w.sum() if w.sum() > EPS else c.prior_vector(h)
            self.learned_w[h], self.n_fit[h] = w, int(len(d))
            self.fit_report[h] = {"n": int(len(d)), "weights": {e: round(float(v), 3) for e, v in zip(EXPERTS, w)},
                                  "mae": float(np.abs(y - X @ w).mean()), "status": "fit"}
        log.info("gate.fit", extra={"event": "gate_fit", "report": self.fit_report})
        return self

    def weights(self, h: str, minutes_to_close: float, net_prem_z: float, available: np.ndarray, pin_strength: float = 0.0) -> Dict[str, Any]:
        prior = self.prior_weights(h, minutes_to_close, pin_strength)
        alpha = 0.0
        w = prior
        if h in self.learned_w:
            alpha = self.n_fit[h] / (self.n_fit[h] + self.cfg.shrink_n0)
            w = (1.0 - alpha) * prior + alpha * self.learned_w[h]
        w, hv = self.regime_adjust(w, h, net_prem_z)
        w = np.where(available, w, 0.0)
        if w.sum() <= EPS:
            w = available.astype(float)
        return {"weights": w / (w.sum() + EPS), "prior": prior, "alpha_learned": alpha, "high_vol_fallback": hv}


@dataclass
class FusionConfig:
    ml_target_kind: str = "retn"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    report_grid: float = 0.01
    direction_deadband_im_frac: float = 0.05


class EnsembleFusionEngine:
    """predict_final(ml_predictions, state, row) → {h: {target, delta, direction, confidence, band_lo, band_hi, weights, experts,
    gravity, high_vol_fallback, alpha_learned, [target_unsnapped]}, _physics, _meta}. Tolerates any expert being NaN."""

    def __init__(self, cfg: Optional[FusionConfig] = None):
        self.cfg = cfg or FusionConfig()
        self.physics, self.gate = OptionsPhysicsEngine(self.cfg.physics), MetaGatingNetwork(self.cfg.gate)

    def fit_gate(self, history: pd.DataFrame) -> "EnsembleFusionEngine":
        self.gate.fit(history)
        return self

    @staticmethod
    def _scalar(v: Any, row: int) -> float:
        if v is None:
            return float("nan")
        a = np.asarray(v, float).ravel()
        if a.size == 0 or row >= a.size or row < -a.size:
            return float("nan")
        return float(a[row])

    def to_price(self, val: float, h: str, st: MarketState) -> float:
        if not np.isfinite(val):
            return float("nan")
        k = self.cfg.ml_target_kind
        if k == "price":
            return float(val)
        if k == "dprice":
            return float(st.spot + val)
        if k == "ret":
            return float(st.spot * math.exp(np.clip(val, -0.2, 0.2)))
        hm = HORIZON_MINUTES[h]
        h_min = max(st.minutes_to_close, 1.0) if hm is None else hm
        return float(st.spot * math.exp(np.clip(val * st.rv_1m * math.sqrt(h_min), -0.2, 0.2)))

    def predict_final(self, ml: Dict[str, Any], st: MarketState, row: int = -1) -> Dict[str, Any]:
        cfg = self.cfg
        ex: Dict[str, Dict[str, float]] = {}
        for h in HORIZONS:
            d = ml.get(h, {}) or {}
            q50 = self._scalar(d.get("tft_q50", d.get("tft")), row)
            lo, hi = self.to_price(self._scalar(d.get("tft_q10"), row), h, st), self.to_price(self._scalar(d.get("tft_q90"), row), h, st)
            tft = self.to_price(q50, h, st)
            if not np.isfinite(tft) and np.isfinite(lo) and np.isfinite(hi):
                tft = 0.5 * (lo + hi)
            ex[h] = {"lgb": self.to_price(self._scalar(d.get("lgb"), row), h, st), "cat": self.to_price(self._scalar(d.get("cat"), row), h, st),
                     "tft": tft, "tft_lo": lo, "tft_hi": hi}
        consensus = {}
        for h in HORIZONS:
            v = np.array([ex[h]["lgb"], ex[h]["cat"], ex[h]["tft"]])
            consensus[h] = float(np.nanmean(v)) if np.isfinite(v).any() else st.spot
        prof = self.physics.profile(st)
        phys = self.physics.targets(st, consensus, prof)
        res: Dict[str, Any] = {}
        for h in HORIZONS:
            e, p_t = ex[h], phys["per_horizon"][h]["physics_target"]
            vec = np.array([e["lgb"], e["cat"], e["tft"], p_t], float)
            avail = np.isfinite(vec)
            g = self.gate.weights(h, st.minutes_to_close, st.net_prem_z, avail, phys["pin_strength"] if h == "eod" else 0.0)
            w = g["weights"]
            target = float(np.where(avail, vec * w, 0.0).sum()) if avail.any() else st.spot
            unsnapped = target
            if h == "eod" and cfg.physics.snap_enabled and phys["snap_applied"] and abs(target - phys["pin_strike"]) <= phys["snap_radius"]:
                s = phys["pin_strength"] * w[3]
                target = (1.0 - s) * target + s * phys["pin_strike"]
            fin = vec[avail]
            spread = float(fin.max() - fin.min()) if fin.size > 1 else 0.0
            agree = math.exp(-spread / (cfg.gate.disagreement_scale_im * st.implied_move + EPS))
            band_w = (e["tft_hi"] - e["tft_lo"]) if np.isfinite(e["tft_hi"]) and np.isfinite(e["tft_lo"]) else float("nan")
            band_pen = math.exp(-band_w / (2.0 * st.implied_move + EPS)) if np.isfinite(band_w) else 0.8
            conf = float(np.clip(0.5 * agree + 0.25 * float((w ** 2).sum()) + 0.25 * band_pen, 0.0, 1.0))
            delta = target - st.spot
            direction = int(np.sign(delta)) if abs(delta) > cfg.direction_deadband_im_frac * st.implied_move else 0
            if np.isfinite(band_w):
                half = 0.5 * band_w
            else:
                hm = HORIZON_MINUTES[h]
                half = 1.2816 * st.spot * st.rv_1m * math.sqrt(max(st.minutes_to_close, 1.0) if hm is None else hm)
            res[h] = {"target": round(target / cfg.report_grid) * cfg.report_grid, "delta": float(delta), "direction": direction,
                      "confidence": conf, "band_lo": float(target - half), "band_hi": float(target + half),
                      "weights": dict(zip(EXPERTS, np.round(w, 4).tolist())),
                      "experts": {"lgb": e["lgb"], "cat": e["cat"], "tft": e["tft"], "physics": p_t},
                      "gravity": phys["per_horizon"][h]["gravity"], "high_vol_fallback": g["high_vol_fallback"], "alpha_learned": g["alpha_learned"]}
            if h == "eod":
                res[h]["target_unsnapped"] = float(unsnapped)
        res["_physics"] = {k: v for k, v in prof.items() if k not in ("clusters", "masses")}
        res["_physics"]["clusters"] = prof["clusters"].to_dict("records")
        res["_physics"].update({k: phys[k] for k in ("eod_unsnapped", "pin_strike", "pin_strength", "pin_mass", "snap_applied", "snap_radius", "time_ramp")})
        res["_meta"] = {"spot": st.spot, "minutes_to_close": st.minutes_to_close, "implied_move": st.implied_move, "net_prem_z": st.net_prem_z,
                        "rv_1m": st.rv_1m, "ml_target_kind": cfg.ml_target_kind, "timestamp": str(st.timestamp) if st.timestamp is not None else None,
                        "gate_fit": self.gate.fit_report}
        return res

    @staticmethod
    def format_call(r: Dict[str, Any]) -> str:
        p, m = r["_physics"], r["_meta"]
        L = [f"SPY {m['spot']:.2f} · {m['minutes_to_close']:.0f}m to close · {p['regime']} · zero-γ {p['zero_gamma_level']:.2f} · max pain {p['max_pain']:.2f}"]
        for h in HORIZONS:
            d, w = r[h], r[h]["weights"]
            arrow = "▲" if d["direction"] > 0 else ("▼" if d["direction"] < 0 else "■")
            L.append(f"{h:>3}: {d['target']:.2f} {arrow} ({d['delta']:+.2f}) conf {d['confidence']:.2f} | lgb {w['lgb']:.2f} cat {w['cat']:.2f} "
                     f"tft {w['tft']:.2f} phys {w['physics']:.2f}" + (" | HIGH-VOL" if d["high_vol_fallback"] else ""))
        wall = f" · wall {p['pin_strike']:.2f} (pull {p['pin_strength']:.2f}{'' if p['snap_applied'] else ', not applied'})" \
            if np.isfinite(p["pin_strike"]) else f" (no wall inside {p['snap_radius']:.2f} pts)"
        L.append(f"📌 EOD Pin Call: {r['eod']['target']:.2f}{wall}")
        return "\n".join(L)


# ============================================================================= #
# PHASE 4 — FEEDS, BUFFER, EXECUTION, LIVE ENGINE, BACKTESTER
# ============================================================================= #
@dataclass
class Tick:
    """kind ∈ {spot, chain, flow, heartbeat}; payload keys follow the Phase 1 column names."""
    ts: pd.Timestamp
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


class BaseFeed:
    async def __aiter__(self) -> AsyncIterator[Tick]:      # pragma: no cover
        raise NotImplementedError
        yield  # noqa

    async def close(self) -> None:
        return None


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


class UnusualWhalesWebSocketFeed(BaseFeed):
    """wss://api.unusualwhales.com/socket?token=<KEY> · Authorization: Bearer · join frames {"channel": "gex:SPY", "msg_type": "join"}
    · messages [channel, payload] · ping every 20s · stale socket after 45s → reconnect with 1s→60s jittered backoff."""

    URL = "wss://api.unusualwhales.com/socket"

    def __init__(self, api_key: str, symbol: str = "SPY", channels: Sequence[str] = ("option_trades", "gex", "price"),
                 ping_interval: float = 20.0, stale_seconds: float = 45.0, max_backoff: float = 60.0):
        if not _HAS_WS:
            raise RuntimeError("pip install websockets for the live Unusual Whales feed")
        if not api_key:
            raise ValueError("api_key required (UW_API_KEY)")
        self.api_key, self.symbol = api_key, symbol.upper()
        self.channels = [f"{c}:{self.symbol}" for c in channels]
        self.ping_interval, self.stale_seconds, self.max_backoff = ping_interval, stale_seconds, max_backoff
        self._closed, self._unknown = False, set()
        self.stats = {"frames": 0, "ticks": 0, "reconnects": 0, "errors": 0}

    async def close(self) -> None:
        self._closed = True

    @staticmethod
    def _ts(p: Dict[str, Any]) -> pd.Timestamp:
        raw = p.get("executed_at") or p.get("timestamp") or p.get("time")
        try:
            ts = pd.Timestamp(raw, unit="ms") if isinstance(raw, (int, float)) else (pd.Timestamp(raw) if raw else pd.Timestamp.utcnow())
        except (ValueError, TypeError):
            ts = pd.Timestamp.utcnow()
        return ts.tz_localize("UTC").tz_convert(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)

    def _normalise(self, channel: str, p: Any) -> List[Tick]:
        out: List[Tick] = []
        if not isinstance(p, (dict, list)):
            return out
        if channel.startswith("option_trades") and isinstance(p, dict):
            ts = self._ts(p)
            tags = [str(t).lower() for t in (p.get("tags") or [])]
            side = str(p.get("side") or "").lower() or ("ask" if any("ask" in t for t in tags) else ("bid" if any("bid" in t for t in tags) else "mid"))
            ttype = "dark_pool" if any("dark" in t for t in tags) else ("sweep" if any("sweep" in t for t in tags) else ("block" if any("block" in t for t in tags) else "regular"))
            out.append(Tick(ts, "flow", {"premium": _f(p.get("premium"), 0.0), "option_type": "C" if str(p.get("option_type", "")).lower().startswith("c") else "P",
                                         "strike": _f(p.get("strike")), "side": side, "trade_type": ttype, "size": _f(p.get("size", p.get("volume")), 0.0)}))
            if p.get("underlying_price") is not None:
                out.append(Tick(ts, "spot", {"spot": _f(p["underlying_price"])}))
        elif channel.startswith("gex"):
            rows = p if isinstance(p, list) else [p]
            for r in rows:
                if not isinstance(r, dict):
                    continue
                out.append(Tick(self._ts(r), "chain", {"strike": _f(r.get("strike")), "gex": _f(r.get("gamma_per_one_percent_move_dir", r.get("gex")), 0.0),
                                                       "call_oi": _f(r.get("call_oi", r.get("call_open_interest")), 0.0), "put_oi": _f(r.get("put_oi", r.get("put_open_interest")), 0.0),
                                                       "call_vol": _f(r.get("call_volume"), 0.0), "put_vol": _f(r.get("put_volume"), 0.0),
                                                       "vanna": _f(r.get("vanna_per_one_percent_move_dir")), "charm": _f(r.get("charm_per_one_percent_move_dir"))}))
        elif channel.startswith("price") and isinstance(p, dict):
            px = p.get("price") or p.get("close") or p.get("last")
            if px is not None:
                out.append(Tick(self._ts(p), "spot", {"spot": _f(px)}))
        elif channel not in self._unknown:
            self._unknown.add(channel)
            log.warning("uw.unknown_channel", extra={"event": "feed", "channel": channel})
        return out

    async def _pinger(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            try:
                await ws.send(json.dumps({"msg_type": "ping"}))
            except Exception:
                return

    async def __aiter__(self) -> AsyncIterator[Tick]:
        backoff = 1.0
        while not self._closed:
            try:
                async with websockets.connect(f"{self.URL}?token={self.api_key}", extra_headers={"Authorization": f"Bearer {self.api_key}"},
                                              ping_interval=None, max_size=8 * 1024 * 1024) as ws:
                    log.info("uw.connected", extra={"event": "feed", "channels": self.channels})
                    for ch in self.channels:
                        await ws.send(json.dumps({"channel": ch, "msg_type": "join"}))
                    backoff = 1.0
                    ping = asyncio.create_task(self._pinger(ws))
                    try:
                        while not self._closed:
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=self.stale_seconds)
                            except asyncio.TimeoutError:
                                raise ConnectionError(f"stale socket ({self.stale_seconds}s silent)")
                            self.stats["frames"] += 1
                            try:
                                msg = json.loads(raw)
                            except (json.JSONDecodeError, TypeError):
                                self.stats["errors"] += 1
                                continue
                            if isinstance(msg, dict) and msg.get("msg_type") in ("pong", "ping", "joined"):
                                yield Tick(pd.Timestamp.now(tz=TZ), "heartbeat", {"type": msg.get("msg_type")})
                            elif isinstance(msg, list) and len(msg) >= 2:
                                for t in self._normalise(str(msg[0]), msg[1]):
                                    self.stats["ticks"] += 1
                                    yield t
                    finally:
                        ping.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.stats["reconnects"] += 1
                wait = min(backoff * (1.0 + 0.2 * (2 * np.random.rand() - 1)), self.max_backoff)
                log.warning("uw.reconnect", extra={"event": "feed", "error": str(e), "wait_s": round(wait, 1), "reconnects": self.stats["reconnects"]})
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, self.max_backoff)


class ReplayFeed(BaseFeed):
    """Replays Phase 1 style frames as ticks at ``speed``× real time (0 = as fast as consumed)."""

    def __init__(self, spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame] = None, flow_df: Optional[pd.DataFrame] = None,
                 speed: float = 0.0, session_date: Optional[Any] = None):
        def _prep(df: Optional[pd.DataFrame]) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=["ts"])
            d = df.copy()
            d["ts"] = to_session_tz(d["ts"])
            if session_date is not None:
                sd = pd.Timestamp(session_date)
                sd = sd.tz_localize(TZ) if sd.tzinfo is None else sd.tz_convert(TZ)
                d = d[d["ts"].dt.normalize() == sd.normalize()]
            return d.sort_values("ts")
        self.spot, self.chain, self.flow, self.speed, self._closed = _prep(spot_df), _prep(chain_df), _prep(flow_df), speed, False

    async def close(self) -> None:
        self._closed = True

    async def __aiter__(self) -> AsyncIterator[Tick]:
        ev: List[Tuple[pd.Timestamp, str, Dict[str, Any]]] = [(r.ts, "spot", {"spot": float(r.spot)}) for r in self.spot.itertuples(index=False)]
        for df, kind in ((self.chain, "chain"), (self.flow, "flow")):
            cols = [c for c in df.columns if c != "ts"]
            ev.extend((rec["ts"], kind, {c: rec[c] for c in cols}) for rec in df.to_dict("records"))
        ev.sort(key=lambda e: e[0])
        prev = None
        for ts, kind, payload in ev:
            if self._closed:
                return
            if self.speed > 0 and prev is not None:
                await asyncio.sleep(max(0.0, (ts - prev).total_seconds() / self.speed))
            prev = ts
            yield Tick(ts, kind, payload)
            if self.speed == 0:
                await asyncio.sleep(0)


class RollingBuffer:
    """Bounded, session-scoped tick store. push() is O(1); the latest complete ladder is maintained incrementally
    so no per-cycle scan of the chain deque is needed."""

    def __init__(self, max_spot: int = 200_000, max_chain: int = 400_000, max_flow: int = 400_000):
        self.spot: Deque[Dict[str, Any]] = deque(maxlen=max_spot)
        self.chain: Deque[Dict[str, Any]] = deque(maxlen=max_chain)
        self.flow: Deque[Dict[str, Any]] = deque(maxlen=max_flow)
        self.session_date: Optional[pd.Timestamp] = None
        self.last_tick_at: Optional[float] = None
        self.last_ts: Optional[pd.Timestamp] = None
        self.counts = {"spot": 0, "chain": 0, "flow": 0, "dropped": 0}
        self._ladder_ts: Optional[pd.Timestamp] = None
        self._ladder: Dict[float, Dict[str, Any]] = {}
        self._ladder_done: Optional[pd.DataFrame] = None
        self._min_ns: int = -1                       # cached minute bucket → day/bar keys computed once per minute, not per tick
        self._day_key: Optional[pd.Timestamp] = None
        self._bar_key: Optional[pd.Timestamp] = None

    def _reset(self, d: pd.Timestamp) -> None:
        if self.session_date is not None:
            log.info("buffer.rollover", extra={"event": "buffer", "new_session": str(d.date()), "counts": dict(self.counts)})
        self.spot.clear(); self.chain.clear(); self.flow.clear()
        self.session_date, self.counts = d, {"spot": 0, "chain": 0, "flow": 0, "dropped": 0}
        self._ladder_ts, self._ladder, self._ladder_done = None, {}, None

    def push(self, t: Tick) -> None:
        self.last_tick_at = time.monotonic()
        if t.kind == "heartbeat":
            return
        mn = t.ts.value // 60_000_000_000
        if mn != self._min_ns:                       # one Timestamp.floor/normalize per minute bucket
            self._min_ns, self._bar_key = mn, t.ts.floor("1min")
            self._day_key = self._bar_key.normalize()
        d = self._day_key
        if self.session_date is None or d != self.session_date:
            self._reset(d)
        row = {"ts": t.ts, **t.payload}
        try:
            if t.kind == "spot" and _f(row.get("spot")) > 0:
                self.spot.append(row)
            elif t.kind == "chain" and np.isfinite(_f(row.get("strike"))):
                self.chain.append(row)
                key = self._bar_key                                             # a snapshot = all strikes printed within one bar
                if self._ladder_ts is None or key > self._ladder_ts:            # new bar → freeze the previous snapshot
                    if len(self._ladder) >= 3:
                        self._ladder_done = pd.DataFrame(list(self._ladder.values()))
                    self._ladder_ts, self._ladder = key, {}
                self._ladder[float(row["strike"])] = row                        # latest print per strike within the bar
            elif t.kind == "flow" and _f(row.get("premium"), 0.0) >= 0:
                self.flow.append(row)
            else:
                self.counts["dropped"] += 1
                return
        except (KeyError, TypeError, ValueError):
            self.counts["dropped"] += 1
            return
        self.counts[t.kind] += 1
        self.last_ts = t.ts if self.last_ts is None or t.ts > self.last_ts else self.last_ts

    def frames(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return (pd.DataFrame(list(self.spot)), pd.DataFrame(list(self.chain)) if self.chain else None,
                pd.DataFrame(list(self.flow)) if self.flow else None)

    def latest_ladder(self) -> Optional[pd.DataFrame]:
        """The bar being filled once it holds ≥3 strikes, else the last frozen snapshot. O(K) — never scans the chain deque."""
        if len(self._ladder) >= 3:
            return pd.DataFrame(list(self._ladder.values()))
        return self._ladder_done


@dataclass
class OptionCostModel:
    """ATM 0DTE friction: premium ≈ 0.4·S·σ₁ₘ·√T; full spread per round trip; √size slippage; commission+fees per side; √T theta bleed."""
    premium_mult: float = 0.40
    spread_pct: float = 0.04
    min_spread: float = 0.02
    slip_per_side: float = 0.01
    commission_per_contract: float = 0.65
    fees_per_contract: float = 0.05
    atm_delta: float = 0.50
    contract_multiplier: float = 100.0

    def premium(self, spot: float, rv_1m: float, minutes_to_expiry: float) -> float:
        return max(0.05, self.premium_mult * spot * rv_1m * math.sqrt(max(minutes_to_expiry, 1.0)))

    def round_trip_costs(self, premium: float, contracts: int) -> float:
        spread = max(self.min_spread, self.spread_pct * premium)
        return contracts * ((spread + 2 * self.slip_per_side * math.sqrt(max(contracts, 1))) * self.contract_multiplier
                            + 2 * (self.commission_per_contract + self.fees_per_contract))

    def pnl(self, side: int, entry_spot: float, exit_spot: float, premium: float, hold_min: float, remaining: float, contracts: int) -> Tuple[float, float, float]:
        du = (exit_spot - entry_spot) * side
        theta = premium * (1.0 - math.sqrt(max(0.0, 1.0 - min(hold_min, remaining) / max(remaining, 1.0))))
        gross = contracts * self.contract_multiplier * (self.atm_delta * du - theta)
        costs = self.round_trip_costs(premium, contracts)
        return gross, costs, gross - costs


@dataclass
class RiskConfig:
    min_confidence: float = 0.55
    min_delta_im_frac: float = 0.08
    contracts_per_trade: int = 2
    max_open_positions: int = 3
    no_entry_last_minutes: Dict[str, float] = field(default_factory=lambda: {"5m": 6.0, "15m": 16.0, "30m": 31.0, "1h": 61.0, "eod": 12.0})
    eod_flat_minutes: float = 2.0
    sl_band_frac: float = 1.0
    tp_target_frac: float = 1.0
    min_sl_im_frac: float = 0.12
    max_sl_im_frac: float = 1.50
    flip_confidence_margin: float = 0.10


@dataclass
class Position:
    horizon: str
    side: int
    entry_ts: pd.Timestamp
    entry_spot: float
    target: float
    stop: float
    take_profit: float
    expiry_ts: pd.Timestamp
    contracts: int
    premium: float
    remaining_at_entry: float
    confidence: float

    @property
    def command(self) -> str:
        return "BUY_CALL" if self.side > 0 else "BUY_PUT"


@dataclass
class Order:
    ts: pd.Timestamp
    command: str
    horizon: str
    spot: float
    contracts: int
    stop: Optional[float] = None
    take_profit: Optional[float] = None
    target: Optional[float] = None
    reason: str = ""
    pnl_net: Optional[float] = None

    def line(self) -> str:
        s = f"{self.ts.strftime('%H:%M:%S')} {self.command:<14} {self.horizon:>3} @ {self.spot:.2f} x{self.contracts}"
        return s + (f" | TP {self.take_profit:.2f} SL {self.stop:.2f} tgt {self.target:.2f}" if self.command != "CLOSE_POSITION"
                    else f" | {self.reason} pnl {self.pnl_net:+.2f}")


class PositionManager:
    """Signal → BUY_CALL / BUY_PUT / CLOSE_POSITION with TFT-band stops and horizon expiry. Shared by live and backtest."""

    def __init__(self, risk: Optional[RiskConfig] = None, costs: Optional[OptionCostModel] = None):
        self.risk, self.costs = risk or RiskConfig(), costs or OptionCostModel()
        self.open: Dict[str, Position] = {}
        self.trades: List[Dict[str, Any]] = []
        self.orders: List[Order] = []

    @staticmethod
    def _close_ts(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.normalize() + pd.Timedelta(hours=16)

    def _close(self, pos: Position, ts: pd.Timestamp, spot: float, reason: str) -> Order:
        hold = (ts - pos.entry_ts).total_seconds() / 60.0
        gross, costs, net = self.costs.pnl(pos.side, pos.entry_spot, spot, pos.premium, hold, pos.remaining_at_entry, pos.contracts)
        self.trades.append({"horizon": pos.horizon, "side": pos.command, "entry_ts": pos.entry_ts, "exit_ts": ts, "entry_spot": pos.entry_spot,
                            "exit_spot": spot, "target": pos.target, "stop": pos.stop, "take_profit": pos.take_profit, "contracts": pos.contracts,
                            "premium": pos.premium, "hold_min": hold, "reason": reason, "confidence": pos.confidence, "pnl_gross": gross,
                            "costs": costs, "pnl_net": net,
                            "hit_target": bool((spot - pos.entry_spot) * pos.side >= (pos.target - pos.entry_spot) * pos.side)})
        o = Order(ts, "CLOSE_POSITION", pos.horizon, spot, pos.contracts, reason=reason, pnl_net=net)
        self.orders.append(o)
        del self.open[pos.horizon]
        return o

    def on_bar(self, ts: pd.Timestamp, spot: float) -> List[Order]:
        out = []
        mtc = (self._close_ts(ts) - ts).total_seconds() / 60.0
        for pos in list(self.open.values()):
            if mtc <= self.risk.eod_flat_minutes:
                out.append(self._close(pos, ts, spot, "EOD_FLAT"))
            elif ts >= pos.expiry_ts:
                out.append(self._close(pos, ts, spot, "HORIZON_EXPIRY"))
            elif (spot - pos.stop) * pos.side <= 0:
                out.append(self._close(pos, ts, spot, "STOP_LOSS"))
            elif (spot - pos.take_profit) * pos.side >= 0:
                out.append(self._close(pos, ts, spot, "TAKE_PROFIT"))
        return out

    def on_signal(self, res: Dict[str, Any], ts: pd.Timestamp, spot: float, rv_1m: float) -> List[Order]:
        out, r, m = [], self.risk, res["_meta"]
        im, mtc = float(m["implied_move"]), float(m["minutes_to_close"])
        for h in HORIZONS:
            d = res[h]
            direction, conf, delta = int(d["direction"]), float(d["confidence"]), float(d["delta"])
            if direction == 0 or conf < r.min_confidence or abs(delta) < r.min_delta_im_frac * im or mtc <= r.no_entry_last_minutes.get(h, 5.0):
                continue
            if h in self.open:
                pos = self.open[h]
                if pos.side == direction or conf < pos.confidence + r.flip_confidence_margin:
                    continue
                out.append(self._close(pos, ts, spot, "FLIP"))
            if len(self.open) >= r.max_open_positions:
                continue
            hm = HORIZON_MINUTES[h]
            horizon_min = mtc if hm is None else min(hm, mtc)
            adverse = (spot - float(d["band_lo"])) if direction > 0 else (float(d["band_hi"]) - spot)
            adverse = float(np.clip(adverse * r.sl_band_frac, r.min_sl_im_frac * im, r.max_sl_im_frac * im))
            stop, take = spot - direction * adverse, spot + r.tp_target_frac * delta
            expiry = min(ts + pd.Timedelta(minutes=horizon_min), self._close_ts(ts) - pd.Timedelta(minutes=r.eod_flat_minutes))
            pos = Position(h, direction, ts, spot, float(d["target"]), stop, take, expiry, r.contracts_per_trade, self.costs.premium(spot, rv_1m, mtc), mtc, conf)
            self.open[h] = pos
            o = Order(ts, pos.command, h, spot, pos.contracts, stop=stop, take_profit=take, target=pos.target, reason=f"conf {conf:.2f} delta {delta:+.2f}")
            self.orders.append(o); out.append(o)
        return out

    def flatten(self, ts: pd.Timestamp, spot: float, reason: str = "SHUTDOWN") -> List[Order]:
        return [self._close(p, ts, spot, reason) for p in list(self.open.values())]

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


def market_state_from_ladder(ladder: pd.DataFrame, spot: float, ts: pd.Timestamp, minutes_to_close: float, rv_1m: Optional[float],
                             net_prem_z: float, implied_move: Optional[float] = None) -> MarketState:
    g = lambda c: pd.to_numeric(ladder[c], errors="coerce").to_numpy(float) if c in ladder else np.zeros(len(ladder))  # noqa: E731
    vol = (g("call_vol") + g("put_vol")) if ("call_vol" in ladder or "put_vol" in ladder) else (g("volume") if "volume" in ladder else None)
    return MarketState(spot=spot, strikes=g("strike"), call_oi=g("call_oi"), put_oi=g("put_oi"), gex=g("gex"), volume=vol,
                       minutes_to_close=minutes_to_close, implied_move=implied_move, net_prem_z=net_prem_z, rv_1m=rv_1m, timestamp=ts)


def bundle_for_rows(out: PipelineOutput, rows: np.ndarray, lookback: int) -> TrainingBundle:
    m = np.zeros(len(out.meta), bool)
    m[rows] = True
    out.masks["_rows"] = m
    try:
        return TrainingBundle.from_phase1(out, "_rows", lookback=lookback)
    finally:
        out.masks.pop("_rows", None)


class CycleRunner:
    """One decision: frames → Phase 1 transform → Phase 2 predict → Phase 3 fuse. Uses only bars ≤ last observed print."""

    def __init__(self, pipeline: FeaturePipeline, core: CoreModelingEngine, fusion: EnsembleFusionEngine, lookback: int = 60):
        self.pipeline, self.core, self.fusion, self.lookback = pipeline, core, fusion, lookback

    def run(self, spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame], flow_df: Optional[pd.DataFrame],
            ladder: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if spot_df is None or len(spot_df) < 5:
            return None
        out = self.pipeline.transform(spot_df, chain_df, flow_df)
        obs = to_session_tz(spot_df["ts"]).max()
        idx = int(np.clip(int((out.meta["ts"] <= obs).sum()) - 1, 0, len(out.meta) - 1))
        bundle = bundle_for_rows(out, np.arange(idx + 1), self.lookback)
        pred = self.core.predict(bundle)
        meta = out.meta.iloc[idx]
        fr = out.features_raw.iloc[idx]
        rv = float(fr["rv_1m"]) if np.isfinite(fr["rv_1m"]) else None
        z = float(fr.get("net_prem_vel_z_5m", 0.0)) if np.isfinite(fr.get("net_prem_vel_z_5m", np.nan)) else 0.0
        spot = float(meta["spot"])
        if ladder is None or len(ladder) < 3:
            ladder = pd.DataFrame({"strike": [math.floor(spot), math.floor(spot) + 1.0, math.floor(spot) + 2.0], "gex": [0.0] * 3, "call_oi": [0.0] * 3, "put_oi": [0.0] * 3})
        st = market_state_from_ladder(ladder, spot, meta["ts"], float(meta["minutes_to_close"]), rv, z)
        res = self.fusion.predict_final(pred, st, row=len(bundle.row_idx) - 1)
        res["_meta"]["bar_ts"] = meta["ts"]
        return res


@dataclass
class LiveConfig:
    cycle_seconds: float = 60.0
    align_to_wall_clock: bool = True
    stale_seconds: float = 90.0
    min_spot_rows: int = 30
    orders_path: Optional[str] = None
    max_runtime_seconds: Optional[float] = None


class LiveTradingEngine:
    """Ingest task + minute scheduler + execution. One bad cycle never stops the next; a dead feed triggers a final cycle then stop."""

    def __init__(self, feed: BaseFeed, runner: CycleRunner, pm: Optional[PositionManager] = None, cfg: Optional[LiveConfig] = None,
                 on_order: Optional[Callable[[Order], None]] = None):
        self.feed, self.runner, self.pm, self.cfg, self.on_order = feed, runner, pm or PositionManager(), cfg or LiveConfig(), on_order
        self.buffer = RollingBuffer()
        self._stop, self._feed_done = asyncio.Event(), False
        self.cycles = self.failures = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self.latencies_ms: Deque[float] = deque(maxlen=1000)

    async def _ingest(self) -> None:
        try:
            async for t in self.feed:
                self.buffer.push(t)
                if self._stop.is_set():
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("ingest.terminated", extra={"event": "feed_error", "error": str(e)}, exc_info=True)
        finally:
            self._feed_done = True

    async def _sleep_to_boundary(self) -> None:
        if self.cfg.align_to_wall_clock and self.cfg.cycle_seconds >= 1:
            await asyncio.sleep(self.cfg.cycle_seconds - (time.time() % self.cfg.cycle_seconds))
        else:
            await asyncio.sleep(self.cfg.cycle_seconds)

    def _emit(self, orders: List[Order]) -> None:
        for o in orders:
            log.info(o.line(), extra={"event": "order", **{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(o).items()}})
            if self.on_order:
                try:
                    self.on_order(o)
                except Exception as e:
                    log.error("on_order failed", extra={"event": "callback_error", "error": str(e)})
            if self.cfg.orders_path:
                with open(self.cfg.orders_path, "a") as fh:
                    fh.write(json.dumps({**asdict(o), "ts": str(o.ts)}, default=float) + "\n")

    async def _cycle(self) -> None:
        t0 = time.perf_counter()
        spot_df, chain_df, flow_df = self.buffer.frames()
        if len(spot_df) < self.cfg.min_spot_rows:
            log.info("cycle.warmup", extra={"event": "cycle", "spot_rows": int(len(spot_df))})
            return
        if self.buffer.last_tick_at is not None and time.monotonic() - self.buffer.last_tick_at > self.cfg.stale_seconds and not self._feed_done:
            log.warning("cycle.stale_data", extra={"event": "cycle", "silent_s": round(time.monotonic() - self.buffer.last_tick_at)})
            self._emit(self.pm.on_bar(self.buffer.last_ts, float(spot_df["spot"].iloc[-1])))
            return
        res = await asyncio.to_thread(self.runner.run, spot_df, chain_df, flow_df, self.buffer.latest_ladder())
        if res is None:
            return
        self.last_result = res
        ts, spot, rv = res["_meta"]["bar_ts"], float(res["_meta"]["spot"]), float(res["_meta"]["rv_1m"])
        self._emit(self.pm.on_bar(ts, spot) + self.pm.on_signal(res, ts, spot, rv))
        self.cycles += 1
        ms = (time.perf_counter() - t0) * 1e3
        self.latencies_ms.append(ms)
        log.info(self.runner.fusion.format_call(res), extra={"event": "cycle", "n": self.cycles, "bar": str(ts), "latency_ms": round(ms, 1),
                                                             "buffer": dict(self.buffer.counts), "open_positions": list(self.pm.open)})

    async def _scheduler(self) -> None:
        start = time.monotonic()
        while not self._stop.is_set():
            await self._sleep_to_boundary()
            if self._stop.is_set():
                break
            try:
                await self._cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.failures += 1
                log.error("cycle.failed", extra={"event": "cycle_error", "failures": self.failures, "error": str(e)}, exc_info=True)
            if self._feed_done or (self.cfg.max_runtime_seconds and time.monotonic() - start > self.cfg.max_runtime_seconds):
                self._stop.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        tasks = [asyncio.create_task(self._ingest(), name="ingest"), asyncio.create_task(self._scheduler(), name="scheduler")]
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.feed.close()
            spot_df, _, _ = self.buffer.frames()
            if self.pm.open and len(spot_df):
                self._emit(self.pm.flatten(self.buffer.last_ts, float(spot_df["spot"].iloc[-1])))
            log.info("engine.stopped", extra={"event": "engine", "cycles": self.cycles, "failures": self.failures, "trades": len(self.pm.trades),
                                              "p50_latency_ms": float(np.median(self.latencies_ms)) if self.latencies_ms else None})


@dataclass
class BacktestConfig:
    decision_every_minutes: int = 1
    lookback: int = 60
    initial_capital: float = 25_000.0
    direction_deadband_im_frac: float = 0.05
    print_dashboard: bool = True


class Backtester:
    """Session-by-session event-driven replay: Phase 2 predicted once per session (vectorised), physics/gate/execution
    walked bar by bar using only the chain snapshot available at or before each bar; realised path used for exits/scoring only."""

    def __init__(self, out: PipelineOutput, chain_df: Optional[pd.DataFrame], core: CoreModelingEngine, fusion: EnsembleFusionEngine,
                 cfg: Optional[BacktestConfig] = None, risk: Optional[RiskConfig] = None, costs: Optional[OptionCostModel] = None):
        self.out, self.core, self.fusion, self.cfg = out, core, fusion, cfg or BacktestConfig()
        self.pm = PositionManager(risk, costs)
        self._chain_by_day: Dict[Any, pd.DataFrame] = {}
        if chain_df is not None and len(chain_df):
            ch = chain_df.copy()
            ch["ts"] = to_session_tz(ch["ts"])
            ch["ts_grid"] = ch["ts"].dt.ceil("1min")
            ch["_ns"] = ch["ts_grid"].dt.tz_convert("UTC").astype("int64")
            ch = ch.sort_values("_ns", kind="mergesort")
            self._chain_by_day = {d: g for d, g in ch.groupby(ch["ts_grid"].dt.normalize())}
        self.signals: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

    def _ladder_at(self, ts: pd.Timestamp, session: pd.Timestamp) -> Optional[pd.DataFrame]:
        day = self._chain_by_day.get(session)
        if day is None:
            return None
        n = int(np.searchsorted(day["_ns"].to_numpy(), ts.tz_convert("UTC").value, side="right"))
        if n == 0:
            return None
        snap = day.iloc[:n]
        snap = snap[snap["_ns"] == snap["_ns"].iloc[-1]].drop_duplicates("strike", keep="last")
        return snap if len(snap) >= 3 else None

    def run(self, sessions: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        meta, fr, cfg = self.out.meta, self.out.features_raw, self.cfg
        all_days = sorted(meta["session_date"].unique())
        days = all_days if sessions is None else [d for d in all_days if d in set(pd.to_datetime(list(sessions)))]
        log.info("backtest.start", extra={"event": "backtest", "sessions": len(days), "cadence_min": cfg.decision_every_minutes})
        spot_all, sess_all, mtc_all = meta["spot"].to_numpy(float), meta["session_date"].to_numpy(), meta["minutes_to_close"].to_numpy(float)
        rv_all = fr["rv_1m"].to_numpy(float)
        z_all = fr["net_prem_vel_z_5m"].to_numpy(float) if "net_prem_vel_z_5m" in fr else np.zeros(len(fr))
        ts_all = meta["ts"]
        for day in days:
            rows = np.where(sess_all == day)[0]
            if len(rows) < cfg.lookback + 5:
                continue
            pred = self.core.predict(bundle_for_rows(self.out, rows, cfg.lookback))
            eod_close = float(spot_all[rows[-1]])
            for k, ridx in enumerate(rows):
                ts, spot = ts_all.iloc[ridx], float(spot_all[ridx])
                if not np.isfinite(spot):
                    continue
                self.pm.on_bar(ts, spot)
                if k % cfg.decision_every_minutes or k < cfg.lookback - 1:
                    continue
                ladder = self._ladder_at(ts, pd.Timestamp(day))
                if ladder is None:
                    continue
                st = market_state_from_ladder(ladder, spot, ts, float(mtc_all[ridx]), float(rv_all[ridx]) if np.isfinite(rv_all[ridx]) else None,
                                              float(z_all[ridx]) if np.isfinite(z_all[ridx]) else 0.0)
                try:
                    res = self.fusion.predict_final(pred, st, row=k)
                except Exception as e:
                    log.error("backtest.fusion_failed", extra={"event": "backtest_error", "ts": str(ts), "error": str(e)})
                    continue
                self.pm.on_signal(res, ts, spot, float(res["_meta"]["rv_1m"]))
                im = float(res["_meta"]["implied_move"])
                for h in HORIZONS:
                    hm = HORIZON_MINUTES[h]
                    j = rows[-1] if hm is None else ridx + int(hm)
                    realized = float(spot_all[j]) if j <= rows[-1] else float("nan")
                    rdir = 0
                    if np.isfinite(realized):
                        rd = realized - spot
                        rdir = int(np.sign(rd)) if abs(rd) > cfg.direction_deadband_im_frac * im else 0
                    d = res[h]
                    self.signals.append({"session": day, "ts": ts, "horizon": h, "spot": spot, "target": d["target"], "realized": realized,
                                         "pred_dir": d["direction"], "real_dir": rdir, "confidence": d["confidence"], "abs_err": abs(d["target"] - realized),
                                         "eod_close": eod_close, "pin_strike": res["_physics"]["pin_strike"],
                                         **{f"w_{e}": v for e, v in d["weights"].items()}, **{f"x_{e}": v for e, v in d["experts"].items()}})
            self.pm.flatten(ts_all.iloc[rows[-1]], eod_close, "SESSION_END")
        self.results = self.analytics()
        if cfg.print_dashboard:
            print(self.dashboard_text(self.results))
        return self.results

    def analytics(self) -> Dict[str, Any]:
        trades, sig, cap = self.pm.trades_frame(), pd.DataFrame(self.signals), self.cfg.initial_capital
        res: Dict[str, Any] = {"n_trades": int(len(trades)), "n_signals": int(len(sig))}
        if len(trades):
            trades["day"] = pd.to_datetime(trades["exit_ts"]).dt.normalize()
            daily = trades.groupby("day")["pnl_net"].sum()
            equity = cap + daily.cumsum()
            ret = daily / cap
            dn = ret[ret < 0]
            dd = (equity - equity.cummax()) / equity.cummax()
            res.update({"cumulative_return": float(equity.iloc[-1] / cap - 1.0), "total_pnl": float(trades["pnl_net"].sum()),
                        "total_costs": float(trades["costs"].sum()),
                        "sharpe": float(ret.mean() / (ret.std(ddof=1) + EPS) * math.sqrt(252)) if len(ret) > 1 else float("nan"),
                        "sortino": float(ret.mean() / (dn.std(ddof=1) + EPS) * math.sqrt(252)) if len(dn) > 1 else float("nan"),
                        "max_drawdown": float(dd.min()), "daily_pnl": daily, "equity": equity,
                        "win_rate_overall": float((trades["pnl_net"] > 0).mean()), "avg_trade": float(trades["pnl_net"].mean()),
                        "profit_factor": float(safe_div(trades.loc[trades.pnl_net > 0, "pnl_net"].sum(), abs(trades.loc[trades.pnl_net < 0, "pnl_net"].sum()), float("inf")))})
            res["per_horizon"] = trades.groupby("horizon").agg(trades=("pnl_net", "size"), win_rate=("pnl_net", lambda s: (s > 0).mean()),
                                                              pnl=("pnl_net", "sum"), avg=("pnl_net", "mean"), hit_target=("hit_target", "mean")).reindex(list(HORIZONS))
            res["exit_reasons"] = trades.groupby("reason")["pnl_net"].agg(["size", "sum", "mean"])
        if len(sig):
            s = sig.dropna(subset=["realized"])
            labels = [-1, 0, 1]
            cm = pd.crosstab(s["pred_dir"], s["real_dir"]).reindex(index=labels, columns=labels, fill_value=0)
            cm.index, cm.columns = [f"pred {x:+d}" for x in labels], [f"real {x:+d}" for x in labels]
            nf = s[s["pred_dir"] != 0]
            res["confusion"] = cm
            res["direction_accuracy_nonflat"] = float((nf["pred_dir"] == nf["real_dir"]).mean()) if len(nf) else float("nan")
            res["direction_accuracy_by_horizon"] = nf.groupby("horizon").apply(lambda g: (g["pred_dir"] == g["real_dir"]).mean()).reindex(list(HORIZONS))
            res["target_mae_by_horizon"] = s.groupby("horizon")["abs_err"].mean().reindex(list(HORIZONS))
            last = s[s["horizon"] == "eod"].sort_values("ts").groupby("session").tail(1)
            res["eod_final_call_error"] = float((last["target"] - last["eod_close"]).abs().mean()) if len(last) else float("nan")
            res["eod_naive_error"] = float((last["spot"] - last["eod_close"]).abs().mean()) if len(last) else float("nan")
        res["trades"], res["signals"] = trades, sig
        return res

    @staticmethod
    def dashboard_text(r: Dict[str, Any]) -> str:
        L = ["=" * 78, "SINUS BACKTEST DASHBOARD", "=" * 78]
        if "cumulative_return" in r:
            L += [f"Trades {r['n_trades']:>6d} | Signals {r['n_signals']:>7d}",
                  f"Cumulative return {r['cumulative_return']*100:>8.2f}% | Total P&L ${r['total_pnl']:>10,.2f} | Costs ${r['total_costs']:>9,.2f}",
                  f"Sharpe {r['sharpe']:>6.2f} | Sortino {r['sortino']:>6.2f} | Max drawdown {r['max_drawdown']*100:>6.2f}%",
                  f"Win rate {r['win_rate_overall']*100:>5.1f}% | Avg trade ${r['avg_trade']:>8.2f} | Profit factor {r['profit_factor']:.2f}",
                  "", "Per horizon:", r["per_horizon"].round(3).to_string(), "", "Exit reasons:", r["exit_reasons"].round(2).to_string()]
        else:
            L.append("No trades opened (signals below confidence / delta gates).")
        if "confusion" in r:
            L += ["", "Direction confusion matrix (rows = predicted, cols = realised):", r["confusion"].to_string(),
                  f"Direction accuracy on non-flat calls: {r['direction_accuracy_nonflat']*100:.1f}%", "By horizon:",
                  r["direction_accuracy_by_horizon"].round(3).to_string(), "", "Target MAE by horizon (points):",
                  r["target_mae_by_horizon"].round(3).to_string(),
                  f"EOD final call error ${r['eod_final_call_error']:.3f} vs naive (spot) ${r['eod_naive_error']:.3f}"]
        return "\n".join(L + ["=" * 78])




# ============================================================================= #
# DATA ADAPTERS — Polygon options chains + real price history
# (merged into sinus.py on 2026-09-01: three separate modules meant three saves and
#  three chances at a stale copy on the phone. One file, one import, one source of truth.)
# ============================================================================= #
try:
    import requests
except ImportError:                                   # pragma: no cover
    requests = None

POLYGON_BASE = "https://api.polygon.io"
DEALER_SIGN = 1.0                                     # +1: dealers short calls / long puts (UW convention)
CONTRACT_MULT = 100.0
_TS_NAMES = ("ts", "timestamp", "datetime", "date_time", "time", "date", "caldt", "bar_time", "t")
_PX_NAMES = ("spot", "close", "adj close", "adj_close", "closeprice", "close_price", "last", "price", "c", "vw")

def _key(explicit: Optional[str] = None) -> str:
    k = explicit or os.environ.get("POLYGON_KEY") or os.environ.get("POLYGON_API_KEY")
    if not k:
        raise RuntimeError("Set POLYGON_KEY (Colab: Secrets panel, then os.environ['POLYGON_KEY']=...)")
    return k


def _get(path: str, params: Dict[str, Any], api_key: str, retries: int = 4) -> Dict[str, Any]:
    """GET with backoff on 429. Polygon paginates via an absolute next_url."""
    if requests is None:
        raise RuntimeError("pip install requests")
    url = path if path.startswith("http") else f"{POLYGON_BASE}{path}"
    p = dict(params or {})
    p["apiKey"] = api_key
    for attempt in range(retries):
        r = requests.get(url, params=p, timeout=30)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"[polygon] rate limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 403:
            raise RuntimeError("403 from Polygon — the options asset class is not on this key's plan")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited after retries")


def _paged(path: str, params: Dict[str, Any], api_key: str, cap: int = 20) -> List[Dict[str, Any]]:
    out, url, p = [], path, dict(params)
    for _ in range(cap):
        j = _get(url, p, api_key)
        out.extend(j.get("results") or [])
        nxt = j.get("next_url")
        if not nxt:
            break
        url, p = nxt, {}
    return out


# ----------------------------------------------------------------------------- #
# Ladder assembly
# ----------------------------------------------------------------------------- #
def _rows_to_ladder(rows: List[Dict[str, float]], spot: float) -> pd.DataFrame:
    """rows: dicts with strike, type ('call'/'put'), gamma, oi, vol → per-strike ladder."""
    if not rows:
        return pd.DataFrame(columns=["strike", "gex", "call_oi", "put_oi", "call_vol", "put_vol"])
    df = pd.DataFrame(rows)
    for c in ("gamma", "oi", "vol"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"])
    calls = df[df["type"] == "call"].groupby("strike").agg(cg=("gamma", "mean"), coi=("oi", "sum"), cv=("vol", "sum"))
    puts = df[df["type"] == "put"].groupby("strike").agg(pg=("gamma", "mean"), poi=("oi", "sum"), pv=("vol", "sum"))
    lad = calls.join(puts, how="outer").fillna(0.0).reset_index()
    unit = (spot ** 2) * 0.01 * CONTRACT_MULT              # dollar gamma per 1% move, per contract
    lad["gex"] = DEALER_SIGN * unit * (lad["cg"] * lad["coi"] - lad["pg"] * lad["poi"])
    lad = lad.rename(columns={"coi": "call_oi", "poi": "put_oi", "cv": "call_vol", "pv": "put_vol"})
    return lad[["strike", "gex", "call_oi", "put_oi", "call_vol", "put_vol"]].sort_values("strike").reset_index(drop=True)


def _spot_from_records(res: List[Dict[str, Any]]) -> float:
    """Underlying price from the snapshot payload. Polygon puts it on ``underlying_asset``, but the
    field is named differently across plans and is EMPTY outside market hours — so several names are
    tried and the most common non-zero value wins (a single stale record shouldn't decide it)."""
    seen: Dict[float, int] = {}
    for r in res:
        ua = r.get("underlying_asset") or {}
        for f in ("price", "last_updated_price", "value", "last_price", "close", "p"):
            v = ua.get(f)
            if isinstance(v, (int, float)) and v > 0:
                seen[float(v)] = seen.get(float(v), 0) + 1
                break
    return max(seen, key=seen.get) if seen else 0.0


def _spot_from_quote(underlying: str, api_key: str) -> float:
    """Fallback: ask the stock endpoints. ``prev`` close works on every plan including free."""
    for path, pick in (("/v2/snapshot/locale/us/markets/stocks/tickers/" + underlying,
                        lambda j: ((j.get("ticker") or {}).get("lastTrade") or {}).get("p")),
                       ("/v2/aggs/ticker/" + underlying + "/prev",
                        lambda j: (j.get("results") or [{}])[0].get("c"))):
        try:
            v = pick(_get(path, {"adjusted": "true"}, api_key))
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        except Exception:
            continue
    return 0.0


def _spot_from_chain(res: List[Dict[str, Any]]) -> float:
    """Last resort: infer ATM from the chain itself. The strike carrying the most combined
    call+put volume sits at the money on any liquid 0DTE book. Coarse (±1 strike) but never zero."""
    vol: Dict[float, float] = {}
    for r in res:
        d = r.get("details") or {}
        k = d.get("strike_price")
        v = (r.get("day") or {}).get("volume") or 0
        if k is not None:
            vol[float(k)] = vol.get(float(k), 0.0) + float(v)
    return max(vol, key=vol.get) if vol else 0.0


def fetch_snapshot_ladder(underlying: str = "SPY", expiry: Optional[str] = None, band: float = 15.0,
                          api_key: Optional[str] = None, spot: Optional[float] = None, verbose: bool = True
                          ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Live chain snapshot -> (ladder, spot, raw_first_record).

    Uses /v3/snapshot/options/{underlying}: greeks, open_interest and day volume per contract.
    ``expiry`` defaults to today (0DTE). ``band`` keeps strikes within spot +/- band.

    Spot resolution order: explicit ``spot`` argument -> the snapshot payload -> a stock quote
    endpoint -> the max-volume strike in the chain. It RAISES if all four fail, because GEX scales
    with spot squared and a zero spot silently produces a ladder of zeros (that was the 2026-09-01 bug).
    """
    k = _key(api_key)
    exp = expiry or pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    res = _paged(f"/v3/snapshot/options/{underlying}", {"expiration_date": exp, "limit": 250}, k)
    if not res:
        raise RuntimeError(f"no contracts for {underlying} {exp} - holiday, or no expiry that date")

    src = "argument"
    if spot is None or spot <= 0:
        spot, src = _spot_from_records(res), "snapshot"
    if spot <= 0:
        spot, src = _spot_from_quote(underlying, k), "quote endpoint"
    if spot <= 0:
        spot, src = _spot_from_chain(res), "max-volume strike (approx)"
    if spot <= 0:
        raise RuntimeError("could not resolve the underlying price - pass spot=<price> explicitly")
    if verbose:
        print(f"[polygon] spot {spot:.2f} (from {src})")
        if src.startswith("max-volume"):
            print("[polygon] WARNING spot is inferred from chain volume, accurate to about one strike")

    rows, no_gamma = [], 0
    for r in res:
        d = r.get("details") or {}
        strike = d.get("strike_price")
        if strike is None or abs(float(strike) - spot) > band:
            continue
        g = (r.get("greeks") or {}).get("gamma")
        if not g:
            no_gamma += 1
        rows.append({"strike": float(strike), "type": str(d.get("contract_type", "")).lower(),
                     "gamma": g, "oi": r.get("open_interest"), "vol": (r.get("day") or {}).get("volume")})
    lad = _rows_to_ladder(rows, spot)
    if verbose:
        print(f"[polygon] {underlying} {exp}: {len(res)} contracts -> {len(lad)} strikes within +/-{band:.0f}")
        if no_gamma:
            print(f"[polygon] {no_gamma}/{len(rows)} in-band contracts had no gamma")
        nz = int((lad['gex'].abs() > 0).sum()) if len(lad) else 0
        if nz == 0:
            print("[polygon] WARNING every GEX is zero - greeks missing on this plan, or all OI is zero")
        else:
            print(f"[polygon] GEX populated on {nz}/{len(lad)} strikes")
    return lad, spot, (res[0] if res else {})


def fetch_historical_ladder(underlying: str, session: str, band: float = 15.0, api_key: Optional[str] = None,
                            verbose: bool = True) -> Tuple[pd.DataFrame, float]:
    """Reconstruct a past session's ladder → (ladder, close).

    Snapshots are live-only, so history is rebuilt from two endpoints:
      /v3/reference/options/contracts   contracts listed as of that date (as_of)
      /v2/aggs/ticker/{ticker}/range/1/day/{d}/{d}   that day's OHLCV per contract

    LIMITATION worth knowing before you rely on it: the daily aggregate carries volume but NOT
    open interest, and greeks are not published historically. So this path returns a
    VOLUME-weighted ladder with gamma approximated by a Black-Scholes gamma computed from the
    session close and a flat IV estimate. It is good enough to backtest the *shape* of the
    physics layer, and it is NOT equivalent to the OI-based GEX that the snapshot path gives.
    Treat historical and live ladders as two different datasets; do not mix them in one fit.
    """
    k = _key(api_key)
    spot_j = _get(f"/v1/open-close/{underlying}/{session}", {"adjusted": "true"}, k)
    close = float(spot_j.get("close") or 0.0)
    if not close:
        raise RuntimeError(f"no close for {underlying} on {session}")
    contracts = _paged("/v3/reference/options/contracts",
                       {"underlying_ticker": underlying, "expiration_date": session, "as_of": session,
                        "limit": 1000}, k)
    contracts = [c for c in contracts if abs(float(c.get("strike_price", 1e9)) - close) <= band]
    if verbose:
        print(f"[polygon] {session}: {len(contracts)} contracts within ±{band} of {close:.2f}")
    rows = []
    for c in contracts:
        t = c.get("ticker")
        try:
            a = _get(f"/v2/aggs/ticker/{t}/range/1/day/{session}/{session}", {"adjusted": "true"}, k)
            bars = a.get("results") or []
            vol = float(bars[0].get("v", 0.0)) if bars else 0.0
        except Exception as e:
            if verbose:
                print(f"[polygon] {t}: {e}")
            vol = 0.0
        strike = float(c["strike_price"])
        rows.append({"strike": strike, "type": str(c.get("contract_type", "")).lower(),
                     "gamma": _bs_gamma(close, strike, 1.0 / 252.0, 0.18), "oi": vol, "vol": vol})
    return _rows_to_ladder(rows, close), close


def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    """Black-Scholes gamma. Used only on the historical path, where greeks aren't published."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-0.5 * d1 ** 2) / (np.sqrt(2 * np.pi) * S * sigma * np.sqrt(T)))


# ----------------------------------------------------------------------------- #
# Bridge into SINUS
# ----------------------------------------------------------------------------- #
def market_state_from_polygon(ladder: pd.DataFrame, spot: float, minutes_to_close: float,
                              net_prem_z: float = 0.0, rv_1m: Optional[float] = None,
                              implied_move: Optional[float] = None):
    """Build a SINUS MarketState from a Polygon ladder."""
    return MarketState(spot=spot, strikes=ladder["strike"].to_numpy(float),
                       call_oi=ladder["call_oi"].to_numpy(float), put_oi=ladder["put_oi"].to_numpy(float),
                       gex=ladder["gex"].to_numpy(float),
                       volume=(ladder["call_vol"] + ladder["put_vol"]).to_numpy(float),
                       minutes_to_close=minutes_to_close, implied_move=implied_move, rv_1m=rv_1m,
                       net_prem_z=net_prem_z, timestamp=pd.Timestamp.now(tz="America/New_York"))


def run_physics_now(underlying: str = "SPY", api_key: Optional[str] = None) -> Dict[str, Any]:
    """Pull today's chain and print the SINUS physics read. The one-liner for a daily run."""
    lad, spot, _ = fetch_snapshot_ladder(underlying, api_key=api_key)
    now = pd.Timestamp.now(tz="America/New_York")
    mtc = max((now.normalize() + pd.Timedelta(hours=16) - now).total_seconds() / 60.0, 0.0)
    st = market_state_from_polygon(lad, spot, mtc)
    eng = EnsembleFusionEngine()
    res = eng.predict_final({h: {} for h in HORIZONS}, st)   # physics-only until the ML layer is trained
    print(eng.format_call(res))
    return res



def _pick(cols, candidates, kind: str) -> str:
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:                       # substring fallback: "Close (USD)" etc.
        for lc, orig in low.items():
            if cand in lc:
                return orig
    raise ValueError(f"could not find a {kind} column among {list(cols)} — pass it explicitly")


def load_csv(path: str, ts_col: Optional[str] = None, px_col: Optional[str] = None,
             tz: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """Read a 1-min OHLCV file → spot_df(ts, spot).

    ``tz``: timezone of the file's timestamps if they are naive. Default assumes they are
    already US/Eastern, which is what most retail SPY exports use. If your file is UTC,
    pass tz="UTC" — getting this wrong shifts every session and silently ruins the
    minutes-to-close feature, so the function prints the first and last bar for a sanity check.
    """
    df = pd.read_csv(path)
    tsc = ts_col or _pick(df.columns, _TS_NAMES, "timestamp")
    pxc = px_col or _pick(df.columns, _PX_NAMES, "price")
    if verbose:
        print(f"[data] {os.path.basename(path)}: {len(df):,} rows · timestamp='{tsc}' · price='{pxc}'")
    ts = pd.to_datetime(df[tsc], errors="coerce", utc=False)
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert(TZ)
    else:
        ts = ts.dt.tz_localize(tz or TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(TZ)
    out = pd.DataFrame({"ts": ts, "spot": pd.to_numeric(df[pxc], errors="coerce")})
    return _finalise(out, verbose)


def load_yfinance(days: int = 30, interval: str = "1m", symbol: str = "SPY",
                  verbose: bool = True) -> pd.DataFrame:
    """Free intraday history from yfinance. 1m is capped near 30 days by Yahoo; 5m near 60.

    Note the cap is Yahoo's, not ours — asking for 90 days of 1m silently returns ~30.
    The printed session count is the truth; check it.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("pip install yfinance")
    df = yf.download(symbol, period=f"{min(days, 60)}d", interval=interval,
                     progress=False, auto_adjust=False, prepost=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned nothing — symbol, interval, or rate limit")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    tsc = _pick(df.columns, _TS_NAMES, "timestamp")
    ts = pd.to_datetime(df[tsc])
    ts = ts.dt.tz_convert(TZ) if isinstance(ts.dtype, pd.DatetimeTZDtype) else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    if verbose:
        print(f"[data] yfinance {symbol} {interval}: {len(df):,} bars")
    return _finalise(pd.DataFrame({"ts": ts, "spot": pd.to_numeric(df["Close"], errors="coerce")}), verbose)


def _finalise(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Drop bad rows, keep regular hours, dedupe, sort, report."""
    df = df.dropna(subset=["ts", "spot"])
    df = df[df["spot"] > 0]
    t = df["ts"].dt.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("16:00").time())]
    df = df[df["ts"].dt.dayofweek < 5]
    df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise ValueError("no regular-hours rows survived — check the timezone of your timestamps")
    n_sess = df["ts"].dt.normalize().nunique()
    if verbose:
        print(f"[data] → {len(df):,} RTH bars · {n_sess} sessions · {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
        per = len(df) / max(n_sess, 1)
        if per < 200:
            print(f"[data] WARNING only {per:.0f} bars/session — expected ~390 for 1-min. Gaps or wrong interval.")
    return df


def ladders_to_chain(ladders: dict) -> pd.DataFrame:
    """Stack saved Polygon ladders into the chain_df contract.

    ``ladders``: {timestamp_or_date: ladder DataFrame} — e.g. one snapshot per capture window,
    or a directory of saved parquet/csv ladders loaded into a dict. Columns strike, gex,
    call_oi, put_oi, call_vol, put_vol are carried through; ts is added from the key.
    """
    frames = []
    for k, lad in ladders.items():
        f = lad.copy()
        ts = pd.Timestamp(k)
        f["ts"] = ts.tz_localize(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["ts", "strike", "gex", "call_oi", "put_oi"])
    return pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)


def load_ladder_dir(path: str, pattern: str = "*.csv", verbose: bool = True) -> pd.DataFrame:
    """Load a directory of saved ladders (one file per capture, named with its date) → chain_df."""
    files = sorted(glob.glob(os.path.join(path, pattern)))
    lads = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        for token in stem.replace("_", " ").split():
            try:
                key = pd.Timestamp(token)
                lads[key] = pd.read_csv(f)
                break
            except (ValueError, TypeError):
                continue
    if verbose:
        print(f"[data] {len(lads)} ladders loaded from {path}")
    return ladders_to_chain(lads)


def train_on_real(spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame] = None,
                  model_dir: str = "sinus_model", lookback: int = 60, tft_epochs: int = 40,
                  verbose: bool = True) -> Tuple:
    """Fit SINUS on real data. Returns (pipeline, out, core, fusion).

    Guards against the quiet failure mode: too few sessions to split. Phase 1 needs at least
    3 sessions to form train/val/test with an embargo, and the TFT wants far more than that
    to mean anything — the function says so plainly rather than producing a confident model
    fitted on a week of data.
    """
    n_sess = spot_df["ts"].dt.normalize().nunique()
    if n_sess < 3:
        raise ValueError(f"{n_sess} session(s) — need 3+ to split train/val/test")
    if verbose and n_sess < 20:
        print(f"[train] {n_sess} sessions. This will FIT but not GENERALISE — treat results as a smoke test, "
              f"not evidence. 20+ sessions is where the numbers start to mean something.")
    pipeline = FeaturePipeline(PipelineConfig())
    out = pipeline.fit_transform(spot_df, chain_df, None)
    train = TrainingBundle.from_phase1(out, "train", lookback=lookback)
    cfg = EngineConfig()
    cfg.tft.max_epochs = tft_epochs
    core = CoreModelingEngine(cfg).fit(train)
    core.save(model_dir)
    fusion = EnsembleFusionEngine(FusionConfig(ml_target_kind="retn"))
    if verbose:
        print(f"[train] done · {len(out.meta):,} bars · {out.features.shape[1]} features · saved to {model_dir}")
    return pipeline, out, core, fusion


def backtest_real(out, chain_df, core, fusion, sessions: Optional[int] = None, cadence: int = 5):
    """Run the SINUS backtester over the most recent ``sessions`` of a real fit."""
    days = sorted(out.meta["session_date"].unique())
    if sessions:
        days = days[-sessions:]
    bt = Backtester(out, chain_df, core, fusion, BacktestConfig(decision_every_minutes=cadence))
    return bt.run(days)



# ============================================================================= #
# SYNTHETIC TAPE (self-test only) + ORCHESTRATOR
# ============================================================================= #
def make_synthetic_tape(n_sessions: int = 8, seed: int = 7, cfg: Optional[PipelineConfig] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """SPY-like 1-min spot path, 5-min chain snapshots (with a missing-screen hour on day 3) and trade prints."""
    cfg = cfg or PipelineConfig()
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(pd.Timestamp("2026-08-10", tz=cfg.tz), periods=n_sessions)
    spot_rows, chain_rows, flow_rows = [], [], []
    px = 640.0
    for d in days:
        o = pd.Timestamp(f"{d.date()} {cfg.session_open}", tz=cfg.tz)
        grid = pd.date_range(o, pd.Timestamp(f"{d.date()} {cfg.session_close}", tz=cfg.tz), freq="1min")
        px *= math.exp(rng.normal(0, 0.004))
        path = px * np.exp(np.cumsum(rng.normal(0, 0.0004, len(grid))))
        spot_rows.append(pd.DataFrame({"ts": grid, "spot": path}))
        strikes = np.arange(np.floor(px) - 10, np.floor(px) + 11, 1.0)
        n = len(strikes)
        for i in range(0, len(grid), 5):
            ts, s = grid[i], path[i]
            dist = strikes - s
            gex = 1e8 * np.exp(-0.5 * (dist / 3.0) ** 2) * np.where(dist > 0, 1.0, -0.6) + rng.normal(0, 1e7, n)
            decay = np.exp(-np.abs(dist) / 4)
            ca, cb, pa, pb = (rng.gamma(2.0, sc, n) * decay for sc in (2e5, 1.5e5, 2e5, 1.5e5))
            chain_rows.append(pd.DataFrame({"ts": ts, "strike": strikes, "call_prem": ca + cb, "put_prem": pa + pb, "call_ask_prem": ca, "call_bid_prem": cb,
                                            "put_ask_prem": pa, "put_bid_prem": pb, "call_oi": rng.integers(500, 20000, n).astype(float),
                                            "put_oi": rng.integers(500, 20000, n).astype(float), "call_vol": rng.integers(0, 5000, n).astype(float),
                                            "put_vol": rng.integers(0, 5000, n).astype(float), "gex": gex, "vanna": rng.normal(0, 5e6, n), "charm": rng.normal(-2e6, 3e6, n)}))
            m = int(rng.integers(3, 12))
            flow_rows.append(pd.DataFrame({"ts": ts + pd.to_timedelta(rng.integers(0, 300, m), unit="s"), "premium": rng.gamma(1.5, 4e4, m),
                                           "option_type": rng.choice(["C", "P"], m), "strike": rng.choice(strikes, m),
                                           "side": rng.choice(["ask", "bid", "mid"], m, p=[0.45, 0.45, 0.1]),
                                           "trade_type": rng.choice(["sweep", "block", "dark_pool", "regular"], m, p=[0.4, 0.15, 0.1, 0.35]),
                                           "size": rng.integers(1, 500, m).astype(float)}))
        px = path[-1]
    chain = pd.concat(chain_rows, ignore_index=True)
    if n_sessions >= 3:
        d3 = days[2]
        gap = (chain["ts"] >= pd.Timestamp(f"{d3.date()} 12:30", tz=cfg.tz)) & (chain["ts"] < pd.Timestamp(f"{d3.date()} 13:30", tz=cfg.tz))
        chain = chain[~gap]
    # dirty-data injection: a few NaN/inf/zero cells the pipeline must absorb
    chain.loc[chain.sample(frac=0.002, random_state=seed).index, "gex"] = np.nan
    chain.loc[chain.sample(frac=0.001, random_state=seed + 1).index, "call_oi"] = 0.0
    chain.loc[chain.sample(frac=0.001, random_state=seed + 2).index, "put_prem"] = np.inf
    return pd.concat(spot_rows, ignore_index=True), chain, pd.concat(flow_rows, ignore_index=True)


def build_system(n_sessions: int = 45, model_dir: str = "/tmp/sinus_engine", lookback: int = 60
                 ) -> Tuple[FeaturePipeline, PipelineOutput, CoreModelingEngine, EnsembleFusionEngine, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Operator bootstrap: tape → Phase 1 fit → Phase 2 fit/save/reload → Phase 3."""
    spot, chain, flow = make_synthetic_tape(n_sessions=n_sessions)
    pipeline = FeaturePipeline(PipelineConfig())
    t0 = time.perf_counter()
    out = pipeline.fit_transform(spot, chain, flow)
    log.info("phase1.fit_done", extra={"event": "timing", "seconds": round(time.perf_counter() - t0, 2), "rows": len(out.meta), "features": int(out.features.shape[1])})
    verify_causality(pipeline, spot, chain, flow)
    train = TrainingBundle.from_phase1(out, "train", lookback=lookback)
    cfg = EngineConfig()
    cfg.tree.n_rounds_max, cfg.tft.max_epochs = 600, 15
    core = CoreModelingEngine(cfg).fit(train)
    core.save(model_dir)
    core = CoreModelingEngine.load(model_dir)
    return pipeline, out, core, EnsembleFusionEngine(FusionConfig(ml_target_kind="retn")), (spot, chain, flow)


async def boot_live(pipeline: FeaturePipeline, core: CoreModelingEngine, fusion: EnsembleFusionEngine,
                    tape: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], demo_seconds: float = 40.0) -> LiveTradingEngine:
    """UW socket when UW_API_KEY is set (60s wall-clock cadence), otherwise an accelerated replay of the last tape session."""
    spot, chain, flow = tape
    key = os.environ.get("UW_API_KEY", "")
    if key and _HAS_WS:
        feed: BaseFeed = UnusualWhalesWebSocketFeed(key, "SPY")
        cfg = LiveConfig(cycle_seconds=60.0, align_to_wall_clock=True, orders_path="orders_live.jsonl")
    else:
        last_day = to_session_tz(spot["ts"]).max().normalize()
        feed = ReplayFeed(spot, chain, flow, speed=390.0 * 60.0 / max(demo_seconds, 1.0), session_date=last_day)
        cfg = LiveConfig(cycle_seconds=2.0, align_to_wall_clock=False, stale_seconds=30.0, max_runtime_seconds=demo_seconds * 1.5, orders_path="orders_replay.jsonl")
        log.info("live.replay_mode", extra={"event": "live", "session": str(last_day.date()), "reason": "no UW_API_KEY or websockets"})
    engine = LiveTradingEngine(feed, CycleRunner(pipeline, core, fusion), cfg=cfg)
    await engine.run()
    return engine


def self_test_physics() -> None:
    """Unit checks on the physics layer with a known ladder: flip level, max pain, pin snap, high-vol fallback, NaN experts."""
    rng = np.random.default_rng(3)
    spot = 645.37
    strikes = np.arange(635.0, 656.0, 1.0)
    dist = strikes - spot
    gex = 1.4e8 * np.exp(-0.5 * (dist / 2.5) ** 2) * np.where(dist > -1.0, 1.0, -0.7) + rng.normal(0, 8e6, len(strikes))
    gex[np.argmin(np.abs(strikes - 646.0))] *= 2.2
    st = MarketState(spot, strikes, rng.integers(2000, 30000, len(strikes)), rng.integers(2000, 30000, len(strikes)), gex, 22.0,
                     volume=rng.integers(500, 20000, len(strikes)), implied_move=2.9, net_prem_z=0.4, rv_1m=0.00035)
    eng = EnsembleFusionEngine()
    ml = {h: {"lgb": [0.35], "cat": [0.20], "tft_q10": [-0.9], "tft_q50": [0.10], "tft_q90": [1.1]} for h in HORIZONS}
    r = eng.predict_final(ml, st)
    assert abs(r["_physics"]["pin_strike"] - 646.0) < 1e-9, "646 wall should be identified"
    assert not r["_physics"]["snap_applied"], "snap is opt-in and must be off by default"
    assert r["eod"]["weights"]["physics"] > 0.8, "physics must dominate EOD inside 30 minutes"
    st2 = MarketState(spot, strikes, st.call_oi, st.put_oi, gex, 240.0, implied_move=2.9, net_prem_z=3.6, rv_1m=0.00035)
    r2 = eng.predict_final(ml, st2)
    assert r2["30m"]["weights"]["tft"] == 0.0 and r2["30m"]["high_vol_fallback"], "high-vol must zero the TFT"
    r3 = eng.predict_final({h: {"lgb": [np.nan], "cat": [np.nan], "tft_q50": [np.nan]} for h in HORIZONS}, st)
    assert all(np.isfinite(r3[h]["target"]) for h in HORIZONS) and r3["5m"]["weights"]["physics"] == 1.0, "NaN experts → physics only"
    st0 = MarketState(spot, strikes, np.zeros(len(strikes)), np.zeros(len(strikes)), np.zeros(len(strikes)), 100.0, rv_1m=0.0)
    r4 = eng.predict_final(ml, st0)
    assert all(np.isfinite(r4[h]["target"]) for h in HORIZONS), "all-zero ladder must not divide by zero"
    log.info("phase3.self_test_passed", extra={"event": "selftest", "pin": r["_physics"]["pin_strike"], "eod_weights": r["eod"]["weights"]})
    print(eng.format_call(r))


if __name__ == "__main__":
    self_test_physics()
    n_sessions = int(os.environ.get("SINUS_SESSIONS", "45"))
    n_bt = int(os.environ.get("SINUS_BT_SESSIONS", "30"))
    pipeline, out, core, fusion, tape = build_system(n_sessions=n_sessions)
    sessions = sorted(out.meta["session_date"].unique())[-n_bt:]
    bt = Backtester(out, tape[1], core, fusion, BacktestConfig(decision_every_minutes=5))
    t0 = time.perf_counter()
    results = bt.run(sessions)
    log.info("backtest.done", extra={"event": "timing", "seconds": round(time.perf_counter() - t0, 2), "decisions": results["n_signals"] // len(HORIZONS)})
    sig = results["signals"]
    if len(sig):
        hist = sig[["horizon", "spot", "realized"]].copy()
        for e in EXPERTS:
            hist[e] = sig[f"x_{e}"]
        fusion.fit_gate(hist.dropna(subset=["realized"]))
    engine = asyncio.run(boot_live(pipeline, core, fusion, tape, demo_seconds=float(os.environ.get("SINUS_DEMO_SECONDS", "40"))))
    if engine.pm.trades:
        print("\nLive-session closed trades:")
        print(engine.pm.trades_frame()[["horizon", "side", "entry_ts", "exit_ts", "reason", "pnl_net"]].to_string(index=False))
