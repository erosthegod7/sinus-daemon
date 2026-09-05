"""
test_scoring.py — the eod pre-11:00 scoring rule.

Run:  python tests/test_scoring.py

No pytest dependency on purpose: this has to be runnable on the laptop mid-run
with nothing installed but the training venv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinus_search import score_predictions, SCORING_VERSION  # noqa: E402

HZ = ("5m", "10m", "20m", "40m", "60m", "eod")
TZ = "America/New_York"


def _session(n_sessions=3, seed=0):
    """RTH minute bars, 390 per session, plus matching Y and predictions."""
    rng = np.random.default_rng(seed)
    # Build tz-aware ranges and concat as Series. Never round-trip through .values here:
    # that drops the zone and re-reads UTC wall-clock as Eastern, which is the exact bug
    # this suite exists to catch.
    parts = [pd.date_range(pd.Timestamp(d) + pd.Timedelta("9h30m"), periods=390, freq="1min", tz=TZ)
             for d in pd.bdate_range("2026-03-02", periods=n_sessions)]
    ts = pd.concat([pd.Series(p) for p in parts], ignore_index=True)
    n = len(ts)
    Y = rng.normal(0, 1, (n, len(HZ)))
    # eod is deliberately easy after 11:00 and hard before, so masking must move the number
    before = (ts.dt.time < pd.Timestamp("11:00").time()).to_numpy()
    pred_eod = np.where(before, Y[:, 5] + rng.normal(0, 3, n), Y[:, 5] + rng.normal(0, 0.01, n))
    pred = {h: {"lgb": Y[:, j] + rng.normal(0, 0.5, n)} for j, h in enumerate(HZ)}
    pred["eod"] = {"lgb": pred_eod}
    return ts, Y, pred, before


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main():
    ok = True
    ts, Y, pred, before = _session()

    print("1. mask bites and the drop ratio is right")
    s = score_predictions(pred, Y, horizons=HZ, ts=ts)
    scored, dropped = s["eod_scored_rows"], s["eod_dropped_rows"]
    ok &= check("eod_scored_rows reported", scored > 0, f"{scored}")
    ok &= check("eod_dropped_rows reported", dropped > 0, f"{dropped}")
    # 09:30-10:59 is 90 bars of 390; 300 dropped / 90 scored = 3.33
    ratio = dropped / max(scored, 1)
    ok &= check("dropped is roughly 3x scored", 2.5 < ratio < 4.0, f"ratio {ratio:.2f}")
    ok &= check("scored+dropped == all rows", scored + dropped == len(Y), f"{scored}+{dropped} vs {len(Y)}")
    ok &= check("scoring_version stamped", s["scoring_version"] == SCORING_VERSION, s["scoring_version"])

    print("2. masking actually changes the eod number")
    s_full = score_predictions(pred, Y, horizons=HZ)          # no ts -> old behaviour
    ok &= check("eod mae is worse when masked to the hard morning rows",
                s["mae_eod"] > s_full["mae_eod"], f"masked {s['mae_eod']:.3f} vs full {s_full['mae_eod']:.3f}")
    ok &= check("fallback stamps a different version",
                s_full["scoring_version"] == "eod_full_session", s_full["scoring_version"])
    ok &= check("non-eod horizons are untouched by the mask",
                np.isclose(s["mae_5m"], s_full["mae_5m"]), f"{s['mae_5m']:.6f} vs {s_full['mae_5m']:.6f}")

    print("3. misaligned ts is refused, not silently tolerated")
    try:
        score_predictions(pred, Y, horizons=HZ, ts=ts.iloc[:-5])
        ok &= check("raises on length mismatch", False, "no exception")
    except ValueError as e:
        ok &= check("raises on length mismatch", True, str(e)[:48])

    print("4. tz-naive timestamps are localised, not misread")
    s_naive = score_predictions(pred, Y, horizons=HZ, ts=ts.dt.tz_localize(None))
    ok &= check("naive ts gives the same mask as aware",
                s_naive["eod_scored_rows"] == scored, f"{s_naive['eod_scored_rows']} vs {scored}")

    print("5. too few eod rows degrades to NaN, not a crash or a lie")
    # 10:45 -> 12:24. 100 rows in total, but only the 15 before 11:00 are eod-scorable,
    # so eod must go NaN while every other horizon still has plenty to score on.
    sl = slice(75, 175)
    s_small = score_predictions({h: {"lgb": pred[h]["lgb"][sl]} for h in HZ},
                                Y[sl], horizons=HZ, ts=ts.iloc[sl])
    ok &= check("eod mae is NaN when under 30 rows survive the mask",
                np.isnan(s_small["mae_eod"]), str(s_small["mae_eod"]))
    ok &= check("other horizons still scored", np.isfinite(s_small["mae_5m"]), f"{s_small['mae_5m']:.3f}")
    ok &= check("mae_mean ignores the NaN horizon", np.isfinite(s_small["mae_mean"]), f"{s_small['mae_mean']:.3f}")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
