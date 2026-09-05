r"""
sinus_magnitude.py
==================
The magnitude meter. Two numbers every window:

    MAG UP   0.0 .. 10.0   how big the move could be if it goes up
    MAG DOWN 0.0 .. 10.0   how big the move could be if it goes down
    NET      -10 .. +10    up minus down — one signed reading

Scale (seismic, not linear). Let R = the excursion divided by BASELINE, where baseline is the
median 30-minute excursion at this time of day over the last 20 sessions:

    M = 10 * (1 - 2 ** (-R))       R=0.5 -> 2.9   R=1 -> 5.0   R=2 -> 7.5   R=3 -> 8.75   R=4 -> 9.4

So 5.0 is "a normal 30 minutes", 7.5 is double normal, 9+ is a three-to-four-sigma day.
A 0.1 step in M above 8 is a lot more dollars than a 0.1 step near 2, which is the point:
the top of the dial is reserved for the moves that pay.

"Potential", not expectation. The heads are quantile regressors at q=0.75: they answer
"how far COULD this go in the next 30 minutes" — the number Anthony's own meter gives him
when strike premium spikes and he knows it's going.

Targets come from the parity spot alone (no book needed): forward 30-minute max excursion
up and down, inside the session. Features come from the champion's own Phase 1 output, so
the meter sees exactly what the champion sees (price + book once the feed is wired).

API
---
    make_targets(spot_df)                        -> DataFrame ts, up, dn, base, mag_up, mag_dn
    fit_magnitude(out, spot_df, save_dir)        -> metrics dict (also writes save_dir/magnitude.json + models)
    predict_magnitude(save_dir, X_row, ts, spot_df) -> dict(mag_up, mag_dn, net, dollars_up, dollars_dn)
    format_meter(res)                            -> one line for the screen
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

TZ = "America/New_York"
HORIZON_MIN = int(os.environ.get("SINUS_MAG_HORIZON", "30"))
BASE_SESSIONS = 20
Q = float(os.environ.get("SINUS_MAG_Q", "0.75"))

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    lgb, _HAS_LGB = None, False
try:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_SK = True
except Exception:  # pragma: no cover
    GradientBoostingRegressor, _HAS_SK = None, False


# --------------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------------- #
def to_mag(excursion, baseline) -> np.ndarray:
    r = np.asarray(excursion, dtype=float) / np.maximum(np.asarray(baseline, dtype=float), 1e-6)
    m = 10.0 * (1.0 - np.power(2.0, -np.maximum(r, 0.0)))
    return np.round(np.clip(m, 0.0, 10.0), 1)


def from_mag(mag, baseline) -> np.ndarray:
    m = np.clip(np.asarray(mag, dtype=float), 0.0, 9.99)
    return -np.log2(1.0 - m / 10.0) * np.asarray(baseline, dtype=float)


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def _tod_bucket(ts: pd.Series) -> np.ndarray:
    m = ts.dt.hour * 60 + ts.dt.minute - (9 * 60 + 30)
    return (m // 15).to_numpy()                      # 26 buckets per session


def make_targets(spot_df: pd.DataFrame, horizon: int = HORIZON_MIN) -> pd.DataFrame:
    df = spot_df[["ts", "spot"]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ)
    df = df.sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.date
    up = np.full(len(df), np.nan)
    dn = np.full(len(df), np.nan)
    for _, g in df.groupby("date", sort=False):
        idx = g.index.to_numpy()
        s = g["spot"].to_numpy()
        n = len(s)
        # forward max/min over (t, t+horizon], within the session
        for i in range(n):
            j = min(n, i + 1 + horizon)
            if j - (i + 1) < max(5, horizon // 3):       # too close to the close to measure
                continue
            w = s[i + 1:j]
            up[idx[i]] = max(w.max() - s[i], 0.0)
            dn[idx[i]] = max(s[i] - w.min(), 0.0)
    df["up"], df["dn"] = up, dn
    df["bucket"] = _tod_bucket(df["ts"])
    # baseline: median of (up+dn)/2 over the previous BASE_SESSIONS sessions, same bucket.
    # shift by one session so today's own excursions never leak into today's baseline.
    daily = df.groupby(["date", "bucket"])[["up", "dn"]].median()
    daily["exc"] = (daily["up"] + daily["dn"]) / 2.0
    daily = daily["exc"].unstack("bucket").sort_index()
    base = daily.shift(1).rolling(BASE_SESSIONS, min_periods=5).median().ffill()
    base = base.stack(future_stack=True).rename("base").reset_index()
    df = df.merge(base, on=["date", "bucket"], how="left")
    glob_med = float(np.nanmedian(df["base"])) if np.isfinite(np.nanmedian(df["base"])) else 0.5
    df["base"] = df["base"].fillna(glob_med).clip(lower=0.05)
    df["mag_up"] = to_mag(df["up"], df["base"])
    df["mag_dn"] = to_mag(df["dn"], df["base"])
    return df[["ts", "spot", "up", "dn", "base", "mag_up", "mag_dn"]]


def baseline_now(spot_df: pd.DataFrame, ts: pd.Timestamp) -> float:
    t = make_targets(spot_df)
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize(TZ)
    b = _tod_bucket(pd.Series([ts]))[0]
    t["bucket"] = _tod_bucket(t["ts"])
    recent = t[(t["bucket"] == b) & (t["ts"] < ts.normalize())].tail(BASE_SESSIONS * 15)
    if len(recent):
        return float(recent["base"].iloc[-1])
    return float(t["base"].median()) if len(t) else 0.5


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _features_from_out(out) -> Tuple[np.ndarray, pd.Series, list]:
    F = getattr(out, "features", None)
    if F is None:
        F = out.features_raw
    if isinstance(F, pd.DataFrame):
        names = list(F.columns)
        X = F.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(F, dtype=np.float32)
        names = list(getattr(out, "feature_names", [f"f{i}" for i in range(X.shape[1])]))
    ts = pd.to_datetime(out.meta["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(TZ)
    return X, ts.reset_index(drop=True), names


class _Head:
    """Quantile regressor with a LightGBM path and a sklearn fallback."""

    def __init__(self, q: float = Q):
        self.q, self.m = q, None

    def fit(self, X, y):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if _HAS_LGB:
            self.m = lgb.LGBMRegressor(objective="quantile", alpha=self.q, n_estimators=400, learning_rate=0.03,
                                       num_leaves=31, min_child_samples=60, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
        elif _HAS_SK:
            self.m = GradientBoostingRegressor(loss="quantile", alpha=self.q, n_estimators=200, max_depth=3,
                                               learning_rate=0.05, subsample=0.8)
        else:
            raise RuntimeError("need lightgbm or scikit-learn")
        self.m.fit(X, y)
        return self

    def predict(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.m.predict(X)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.m, f)

    @classmethod
    def load(cls, path, q=Q):
        import pickle
        h = cls(q)
        with open(path, "rb") as f:
            h.m = pickle.load(f)
        return h


def fit_magnitude(out, spot_df: pd.DataFrame, save_dir: str, test_frac: float = 0.15,
                  verbose: bool = True) -> Dict[str, Any]:
    X, ts, names = _features_from_out(out)
    tg = make_targets(spot_df)
    tg = tg.set_index("ts")
    y = tg.reindex(ts)
    ok = y["mag_up"].notna().to_numpy() & y["mag_dn"].notna().to_numpy()
    X, ts = X[ok], ts[ok].reset_index(drop=True)
    y = y[ok]
    days = ts.dt.date.to_numpy()
    uniq = np.array(sorted(set(days)))
    n_test = max(3, int(len(uniq) * test_frac))
    test_days = set(uniq[-n_test:])
    tr = np.array([d not in test_days for d in days])
    te = ~tr
    heads = {"up": _Head().fit(X[tr], y["mag_up"].to_numpy()[tr]),
             "dn": _Head().fit(X[tr], y["mag_dn"].to_numpy()[tr])}
    os.makedirs(save_dir, exist_ok=True)
    heads["up"].save(os.path.join(save_dir, "mag_up.pkl"))
    heads["dn"].save(os.path.join(save_dir, "mag_dn.pkl"))
    metrics = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "horizon_min": HORIZON_MIN, "q": Q,
               "feature_names": names}
    for k, col in (("up", "mag_up"), ("dn", "mag_dn")):
        p = np.clip(heads[k].predict(X[te]), 0, 10)
        t = y[col].to_numpy()[te]
        metrics[f"mae_{k}"] = float(np.abs(p - t).mean())
        big = t >= 7.0
        metrics[f"big_{k}_n"] = int(big.sum())
        metrics[f"big_{k}_recall"] = float((p[big] >= 6.0).mean()) if big.any() else float("nan")
        called = p >= 6.0
        metrics[f"big_{k}_precision"] = float((t[called] >= 7.0).mean()) if called.any() else float("nan")
        # naive bar: always say 5.0 (a normal window)
        metrics[f"naive_mae_{k}"] = float(np.abs(5.0 - t).mean())
    # net direction accuracy where the realised net is meaningful
    pu, pd_ = np.clip(heads["up"].predict(X[te]), 0, 10), np.clip(heads["dn"].predict(X[te]), 0, 10)
    tu, td = y["mag_up"].to_numpy()[te], y["mag_dn"].to_numpy()[te]
    m = np.abs(tu - td) >= 2.0
    metrics["net_dir_acc"] = float((np.sign(pu - pd_)[m] == np.sign(tu - td)[m]).mean()) if m.any() else float("nan")
    metrics["net_dir_n"] = int(m.sum())
    json.dump(metrics, open(os.path.join(save_dir, "magnitude.json"), "w"), indent=2, default=float)
    if verbose:
        print(f"[mag] up  mae {metrics['mae_up']:.2f} (naive {metrics['naive_mae_up']:.2f}) · big-move recall "
              f"{metrics['big_up_recall']:.2f} precision {metrics['big_up_precision']:.2f} on {metrics['big_up_n']} events")
        print(f"[mag] dn  mae {metrics['mae_dn']:.2f} (naive {metrics['naive_mae_dn']:.2f}) · big-move recall "
              f"{metrics['big_dn_recall']:.2f} precision {metrics['big_dn_precision']:.2f} on {metrics['big_dn_n']} events")
        print(f"[mag] net direction {metrics['net_dir_acc']:.2f} on {metrics['net_dir_n']} decisive windows")
    return metrics


def predict_magnitude(save_dir: str, X_row, ts, spot_df: Optional[pd.DataFrame] = None,
                      baseline: Optional[float] = None) -> Dict[str, Any]:
    up = _Head.load(os.path.join(save_dir, "mag_up.pkl"))
    dn = _Head.load(os.path.join(save_dir, "mag_dn.pkl"))
    X = np.asarray(X_row, dtype=np.float32).reshape(1, -1)
    mu = float(np.clip(up.predict(X)[0], 0, 10))
    md = float(np.clip(dn.predict(X)[0], 0, 10))
    base = baseline if baseline is not None else (baseline_now(spot_df, ts) if spot_df is not None else None)
    res = {"mag_up": round(mu, 1), "mag_dn": round(md, 1), "net": round(mu - md, 1), "horizon_min": HORIZON_MIN}
    if base:
        res["baseline"] = round(float(base), 2)
        res["dollars_up"] = round(float(from_mag(mu, base)), 2)
        res["dollars_dn"] = round(float(from_mag(md, base)), 2)
    return res


def format_meter(r: Dict[str, Any]) -> str:
    bar = lambda m: "█" * int(round(m)) + "·" * (10 - int(round(m)))
    line = f"MAG ▲{r['mag_up']:.1f} [{bar(r['mag_up'])}]   ▼{r['mag_dn']:.1f} [{bar(r['mag_dn'])}]   NET {r['net']:+.1f}"
    if "dollars_up" in r:
        line += f"   ≈ +${r['dollars_up']:.2f} / -${r['dollars_dn']:.2f} in {r['horizon_min']}m"
    tag = ("SITTING ON A MOVE" if max(r["mag_up"], r["mag_dn"]) >= 7.5 else
           "elevated" if max(r["mag_up"], r["mag_dn"]) >= 6.0 else "normal")
    return line + f"   [{tag}]"


if __name__ == "__main__":
    # self-test on a synthetic tape
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.bdate_range("2026-06-01", periods=40):
        g = pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="1min", tz=TZ)
        vol = rng.choice([0.03, 0.05, 0.12])
        rows.append(pd.DataFrame({"ts": g, "spot": 760 + np.cumsum(rng.normal(0, vol, len(g)))}))
    sp = pd.concat(rows, ignore_index=True)
    t = make_targets(sp)
    print(t.describe().round(2))
    print("mag at R=1:", to_mag(1.0, 1.0), " R=2:", to_mag(2.0, 1.0), " back:", from_mag(7.5, 1.0))
