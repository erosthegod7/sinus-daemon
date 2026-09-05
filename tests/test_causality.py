"""
test_causality.py — no feature may know the future.

For the last session of the trainer's own feed, every raw feature at bar t computed from
the full tape must equal the same feature computed from the tape cut at t. Prior sessions
are kept on both sides so cross-day features are exercised — the 2026-09-05 leak lived in
one of those (hist_prior_ret was TODAY's close-to-close return), and a single-day test
cannot see it.

Runs the trainer's complete feature set (base engine + install_extra_features) on the
trainer's cache under SINUS_VOLUME/history, so it tests exactly what the champion sees.

Run:  python tests/test_causality.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus_train  # noqa: E402
from sinus import FeaturePipeline, PipelineConfig, to_session_tz, verify_causality, TZ  # noqa: E402

VOL = os.environ.get("SINUS_VOLUME", r"C:\sinus\data")
HIST = os.path.join(VOL, "history")
CSV = os.environ.get("SINUS_CSV", os.path.join(VOL, "spy_1min_ohlcv.csv"))
N_SESSIONS, CUTS = 12, (45, 120, 240)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def load_feed():
    days = sorted(f[6:16] for f in os.listdir(HIST) if f.startswith("chain_"))[-N_SESSIONS:]
    rd = lambda k, d: pd.read_csv(os.path.join(HIST, f"{k}_{d}.csv.gz"), parse_dates=["ts"])  # noqa: E731
    chain = pd.concat([rd("chain", d) for d in days], ignore_index=True)
    flow = pd.concat([rd("flow", d) for d in days], ignore_index=True)
    for df in (chain, flow):
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
    spot = sinus_train.load_ohlcv(CSV)
    spot = spot[spot["ts"].dt.normalize().isin(pd.DatetimeIndex(days).tz_localize(TZ))].reset_index(drop=True)
    return spot, chain, flow, days


def main():
    if not (os.path.isdir(HIST) and os.path.exists(CSV)):
        print(f"SKIP — need {HIST} and {CSV}")
        return 0
    ok = True
    sinus_train.install_horizons()
    sinus_train.install_extra_features()
    spot, chain, flow, days = load_feed()
    pipe = FeaturePipeline(PipelineConfig())
    pipe.fit_transform(spot, chain, flow)

    print("1. the regression: hist_prior_ret must be YESTERDAY's return")
    out = pipe.transform(spot, chain, flow)
    d = (out.meta.assign(hpr=out.features_raw["hist_prior_ret"].to_numpy())
         .groupby("session_date").agg(close=("spot", "last"), hpr=("hpr", "first")))
    today = np.log(d["close"] / d["close"].shift(1))
    ok &= check("does not equal today's close-to-close",
                not np.allclose(d["hpr"].iloc[2:], today.iloc[2:], atol=1e-9))
    ok &= check("equals yesterday's close-to-close",
                np.allclose(d["hpr"].iloc[2:], today.shift(1).iloc[2:], atol=1e-9))

    print(f"2. per-column causality on {days[-1]}, full trainer feature set ({out.features_raw.shape[1]} raw cols)")
    last = pd.Timestamp(days[-1], tz=TZ)
    upto = lambda df: df[to_session_tz(df["ts"], pipe.cfg.tz).dt.normalize() <= last]  # noqa: E731
    s, c, f = upto(spot), upto(chain), upto(flow)
    full = pipe.transform(s, c, f)
    day_rows = np.where(full.meta["ts"].dt.normalize() == last)[0]
    bad = {}
    for cut in CUTS:
        pos = int(day_rows[cut])
        cut_ts = full.meta["ts"].iloc[pos]
        tr = pipe.transform(s[to_session_tz(s["ts"]) <= cut_ts], c[to_session_tz(c["ts"]) <= cut_ts],
                            f[to_session_tz(f["ts"]) <= cut_ts])
        a, b = full.features_raw.iloc[pos], tr.features_raw.iloc[pos]
        for col in a.index:
            x, y = float(a[col]), float(b[col])
            if (np.isnan(x) != np.isnan(y)) or (not np.isnan(x) and not np.isclose(x, y, atol=1e-6, rtol=1e-5)):
                bad.setdefault(col, []).append((cut, x, y))
    for col, v in sorted(bad.items()):
        print(f"       LEAK {col:24s}", "  ".join(f"@{cc}: full={x:.4g} cut={y:.4g}" for cc, x, y in v))
    ok &= check(f"zero columns differ at cuts {CUTS}", not bad, f"{len(bad)} leak" if bad else "")

    print("3. the engine's own verify_causality agrees (multi-day form)")
    ok &= check("verify_causality passes", verify_causality(pipe, spot, chain, flow, cut_minutes=120))

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
