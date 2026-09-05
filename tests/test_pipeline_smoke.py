"""
test_pipeline_smoke.py — end-to-end on real SPY bars.

Proves the thing unit tests cannot: that a TrainingBundle built from the real
pipeline carries per-row timestamps that actually line up with its Y, and that
scoring against them masks the rows we think it masks.

Run:  python tests/test_pipeline_smoke.py [n_sessions]

Uses SINUS_CSV if set, else C:\\sinus\\data\\spy_1min_ohlcv.csv. Spot-only —
no chain feed needed, since what is under test is row alignment, not features.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus_train  # noqa: E402
from sinus import FeaturePipeline, PipelineConfig, TrainingBundle  # noqa: E402
from sinus_search import score_predictions  # noqa: E402

CSV = os.environ.get("SINUS_CSV", r"C:\sinus\data\spy_1min_ohlcv.csv")
N_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main():
    if not os.path.exists(CSV):
        print(f"SKIP — {CSV} not found")
        return 0

    ok = True
    sinus_train.install_horizons()

    print(f"0. loading {CSV} (last {N_SESSIONS} sessions)")
    spot = sinus_train.load_ohlcv(CSV)
    keep = sorted(spot["ts"].dt.normalize().unique())[-N_SESSIONS:]
    spot = spot[spot["ts"].dt.normalize().isin(keep)].reset_index(drop=True)
    print(f"   {len(spot):,} bars over {spot['ts'].dt.normalize().nunique()} sessions")

    print("1. loader keeps regular hours only")
    t = spot["ts"].dt.time
    ok &= check("no bar before 09:30", t.min() >= pd.Timestamp("09:30").time(), str(t.min()))
    ok &= check("no bar after 16:00", t.max() <= pd.Timestamp("16:00").time(), str(t.max()))
    ok &= check("no weekend bars", bool((spot["ts"].dt.dayofweek < 5).all()))
    per = len(spot) / spot["ts"].dt.normalize().nunique()
    ok &= check("~390 bars per session", 380 <= per <= 400, f"{per:.0f}")

    print("2. pipeline builds and bundles carry timestamps")
    out = FeaturePipeline(PipelineConfig()).fit_transform(spot, None, None)
    print(f"   {len(out.meta):,} rows · {out.features.shape[1]} features (spot-only)")
    for split in ("train", "val", "test"):
        b = TrainingBundle.from_phase1(out, split, lookback=60)
        ok &= check(f"{split}: ts present", b.ts is not None)
        ok &= check(f"{split}: len(ts) == len(Y)", len(b.ts) == len(b.Y), f"{len(b.ts)} vs {len(b.Y)}")
        ok &= check(f"{split}: ts is tz-aware", b.ts.dt.tz is not None, str(b.ts.dt.tz))
        # the real alignment test: bundle timestamps must equal meta's at those row indices
        expect = out.meta["ts"].iloc[b.row_idx].reset_index(drop=True)
        ok &= check(f"{split}: ts matches meta at row_idx", bool(b.ts.equals(expect)))

    print("3. scoring on a real bundle masks the right rows")
    b = TrainingBundle.from_phase1(out, "val", lookback=60)
    rng = np.random.default_rng(0)
    hz = tuple(sinus_train.HORIZONS_NEW)
    pred = {h: {"lgb": b.Y[:, j] + rng.normal(0, 0.5, len(b.Y))} for j, h in enumerate(hz)}
    s = score_predictions(pred, b.Y, horizons=hz, ts=b.ts)
    scored, dropped = s.get("eod_scored_rows", 0), s.get("eod_dropped_rows", 0)
    ok &= check("eod row counts reported", scored > 0 and dropped > 0, f"{scored} scored / {dropped} dropped")
    ok &= check("dropped is roughly 3x scored", 2.0 < dropped / max(scored, 1) < 4.5,
                f"ratio {dropped / max(scored, 1):.2f}")
    # Independent recount straight off the bundle's own timestamps. Must apply the same
    # finiteness filter the scorer does: eod targets are NaN at the tail of each session,
    # where there is no close far enough ahead to label against.
    eod_j = hz.index("eod")
    before = (b.ts.dt.time < pd.Timestamp("11:00").time()).to_numpy()
    finite = np.isfinite(b.Y[:, eod_j]) & np.isfinite(pred["eod"]["lgb"])
    manual = int((before & finite).sum())
    ok &= check("scored count matches an independent recount", scored == manual, f"{scored} vs {manual}")
    ok &= check("dropped count matches an independent recount",
                dropped == int((~before & finite).sum()), f"{dropped} vs {int((~before & finite).sum())}")
    ok &= check("scoring_version stamped", s["scoring_version"] == "eod_pre11", s["scoring_version"])
    ok &= check("score is finite", np.isfinite(s["score"]), f"{s['score']:.4f}")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
