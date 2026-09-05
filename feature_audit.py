#!/usr/bin/env python3
"""
feature_audit.py — what is actually training, not just what is present.

    python feature_audit.py [n_sessions]        default 60

A column can sit in the feature matrix and teach the model nothing. If a block of
features is entirely NaN it gets imputed to a constant, and a constant column has no
split to offer a tree and no gradient to give the net — it is dead weight that still
shows up in a feature count. This prints, per family: how many columns exist, how much
of each is NaN before imputation, and how many end up with no variance at all.

Reads the same inputs the trainer does: SINUS_CSV for spot, SINUS_VOLUME/chain for the
options book and the flow prints.
"""
import os
import sys

import numpy as np
import pandas as pd

import sinus_train
from sinus import FeaturePipeline, PipelineConfig

VOL = os.environ.get("SINUS_VOLUME", r"C:\sinus\data")
CSV = os.environ.get("SINUS_CSV", os.path.join(VOL, "spy_1min_ohlcv.csv"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# Ordered: first pattern that matches a column wins, so a column is counted exactly once.
FAMILIES = [
    ("Lags / deltas (trees only)", r"_d\d+m$|^ret_(10|20|25|40|50)m$"),
    ("GEX / gamma book", r"gex|gamma"),
    ("Greeks (vanna/charm)", r"vanna|charm"),
    ("Implied vol", r"\biv_|_iv\b|implied|skew"),
    ("Premium / flow $", r"prem"),
    ("Open interest", r"_oi\b|open_int"),
    ("Option volume", r"call_vol|put_vol|opt_vol"),
    ("Tape flow prints", r"flow|aggress|sweep"),
    ("Pin / max-pain", r"pin|max_pain|magnet"),
    ("Candle / volume / VWAP", r"open|high|low|close|volume|vwap|trades|candle|body|wick|range"),
    ("Momentum / trend", r"rsi|macd|ema|sma|mom|slope|trend|ret_|retn"),
    ("Volatility / realised", r"rv_|realis|realiz|atr|sigma|vol_"),
    ("Session clock", r"minutes|session|time_|sin_|cos_|opex|witch"),
]


def classify(name):
    import re
    for label, pat in FAMILIES:
        if re.search(pat, name, re.I):
            return label
    return "Other"


def main():
    # Exactly the trainer's contract: horizons, candle + ATM blocks, lag block, BS greeks,
    # and its own per-session cache under history/. Auditing anything else audits a
    # different model.
    sinus_train.install_horizons()
    sinus_train.install_extra_features()

    print(f"spot : {CSV}")
    spot = sinus_train.load_ohlcv(CSV)
    has_candles = spot["open"].notna().mean()
    print(f"       {len(spot):,} RTH bars · open/high/low present on {has_candles:.0%} of rows")
    if has_candles < 0.5:
        print("       !! this file has no real candles — the whole candle block will be dead")
    keep = sorted(spot["ts"].dt.normalize().unique())[-N:]
    spot = spot[spot["ts"].dt.normalize().isin(keep)].reset_index(drop=True)

    print(f"chain: {VOL}\\history  (last {N} sessions — the trainer's cache; an uncached session would be built from Polygon)")
    chain, flow = sinus_train.build_history(spot, os.environ.get("SINUS_SYMBOL", "SPY"))
    print(f"       {len(chain):,} chain rows · {len(flow):,} flow rows on {spot['ts'].dt.normalize().nunique()} sessions\n")

    out = FeaturePipeline(PipelineConfig()).fit_transform(spot, chain, flow)
    raw, fin = out.features_raw, out.features
    nan_frac = raw.isna().mean()
    std = fin.std(axis=0, numeric_only=True)
    dead = (std.fillna(0) == 0)

    groups = {}
    for c in fin.columns:
        groups.setdefault(classify(c), []).append(c)

    print(f"{'family':40s} {'cols':>5s} {'%NaN raw':>9s} {'dead':>5s}  status")
    print("-" * 78)
    total_dead = 0
    for label, _ in FAMILIES + [("Other", "")]:
        cols = groups.get(label)
        if not cols:
            continue
        cols_r = [c for c in cols if c in nan_frac.index]
        pct = float(nan_frac[cols_r].mean()) if cols_r else 0.0
        d = int(dead.reindex(cols).fillna(False).sum())
        total_dead += d
        if d == len(cols):
            status = "DEAD — nothing to learn from"
        elif d:
            status = f"{d} of {len(cols)} dead"
        elif pct > 0.5:
            status = "sparse but live"
        else:
            status = "ok"
        print(f"{label:40s} {len(cols):5d} {pct:8.0%} {d:5d}  {status}")

    print("-" * 78)
    print(f"{'TOTAL':40s} {len(fin.columns):5d} {float(nan_frac.mean()):8.0%} {total_dead:5d}")
    live = len(fin.columns) - total_dead
    print(f"\n{live} of {len(fin.columns)} features carry variance and can actually train.")
    if total_dead:
        names = list(fin.columns[dead.reindex(fin.columns).fillna(False).to_numpy()])
        print(f"dead columns: {names[:12]}{' ...' if len(names) > 12 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
