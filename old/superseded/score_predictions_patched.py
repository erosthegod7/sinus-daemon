"""
score_predictions_patched.py
============================
Drop-in replacement for score_predictions().

Change: the 'eod' horizon is only scored on samples timestamped BEFORE 11:00 ET.
Anything from 11:00 onward is still PREDICTED and still shown live -- it is just
not counted when ranking a trial. From 11:00 on, the close is close enough to
read off the tape, and letting those samples into the average was inflating eod
accuracy to 95%+ and flattening the difference between good and bad configs.

Every other horizon is scored on every sample, unchanged.

Two things to wire up in sinus_train.py:

  1. Pass timestamps in. The scorer now takes a `ts` argument -- the per-row
     bar timestamp for the same rows that make up Y, tz-aware America/New_York,
     same length as Y. Your meta frame already carries this; if it only has
     session_date, you need the intraday time too.

  2. If ts is None the function falls back to the old behaviour and scores eod
     on everything, so nothing breaks while you wire it -- but it will print a
     warning, because a silent fallback here is exactly the kind of thing that
     costs a week.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

EOD_SCORING_CUTOFF = "11:00"


def score_predictions(pred: Dict[str, Any],
                      Y: np.ndarray,
                      horizons=("5m", "10m", "20m", "40m", "60m", "eod"),
                      ts: Optional[pd.Series] = None,
                      eod_cutoff: str = EOD_SCORING_CUTOFF) -> Dict[str, float]:
    """Trimmed MAE + directional accuracy per horizon, on the ensemble mean of whatever
    experts exist. Trimmed at the 95th percentile because a handful of gap bars would
    otherwise decide the whole ranking.

    'eod' is scored only on samples before `eod_cutoff` local market time.
    """
    out, maes, accs = {}, [], []

    eod_mask = None
    if ts is not None:
        t = pd.to_datetime(pd.Series(ts).values)
        if getattr(t, "tz", None) is None:
            t = pd.DatetimeIndex(t).tz_localize("America/New_York")
        else:
            t = pd.DatetimeIndex(t).tz_convert("America/New_York")
        if len(t) != len(Y):
            raise ValueError(f"ts has {len(t)} rows but Y has {len(Y)} -- they must line up")
        eod_mask = (t.time < pd.Timestamp(eod_cutoff).time())
    else:
        print("[score] WARNING no ts passed -- eod scored on the full session (old behaviour)",
              flush=True)

    for j, h in enumerate(horizons):
        y = Y[:, j]
        stack = [pred[h][k] for k in ("lgb", "cat", "tft_q50") if k in pred[h]]
        stack = [a for a in stack if np.isfinite(a).any()]
        if not stack:
            continue
        arr = np.stack(stack)
        cnt = np.isfinite(arr).sum(0)
        p = np.where(cnt > 0, np.where(np.isfinite(arr), arr, 0.0).sum(0) / np.maximum(cnt, 1), np.nan)

        m = np.isfinite(y) & np.isfinite(p)
        if h == "eod" and eod_mask is not None:
            m = m & eod_mask
        if m.sum() < 30:
            out[f"mae_{h}"], out[f"acc_{h}"] = float("nan"), float("nan")
            continue

        e = np.abs(p[m] - y[m])
        cut = np.quantile(e, 0.95)
        mae = float(e[e <= cut].mean())
        nz = np.abs(y[m]) > 0.1
        acc = float((np.sign(p[m][nz]) == np.sign(y[m][nz])).mean()) if nz.any() else 0.5

        out[f"mae_{h}"], out[f"acc_{h}"] = mae, acc
        maes.append(mae)
        accs.append(acc)

        if h == "eod" and eod_mask is not None:
            out["eod_scored_rows"] = int(m.sum())
            out["eod_dropped_rows"] = int((np.isfinite(y) & np.isfinite(p) & ~eod_mask).sum())

    if not maes:
        out["note"] = "no expert produced finite predictions -- lightgbm/catboost/torch missing?"
    out["mae_mean"] = float(np.mean(maes)) if maes else float("inf")
    out["acc_mean"] = float(np.mean(accs)) if accs else 0.0
    out["score"] = out["mae_mean"] - 0.25 * (out["acc_mean"] - 0.5)
    out["scoring_version"] = "eod_pre11"
    return out
