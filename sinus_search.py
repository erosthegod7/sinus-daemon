"""
sinus_search.py
===============
Hyperparameter search for SINUS. This is the legitimate "run it non-stop" workload: each
trial is a DIFFERENT model, not the same model relearning the same data.

Why random search and not grid: with 6+ knobs a grid explodes combinatorially and spends most
of its budget on dimensions that don't matter. Random search finds a good region far faster
for the same number of trials — and it can be stopped at any point, which matters when the
runtime can vanish mid-run.

Leakage discipline (the whole point of the exercise):
    * Phase 1 is fit ONCE, outside the loop, on the training sessions only. Refitting the
      scaler per trial would leak validation statistics into every candidate.
    * Trials are scored on the VALIDATION split. The TEST split is never touched here —
      it exists so the winner can be measured once, at the end, on data no search decision
      ever saw. Peeking at test during a search is how you build a model that backtests
      beautifully and loses money.
    * Score is trimmed MAE on vol-normalised returns, averaged across horizons, plus a
      directional-accuracy tiebreak.

Checkpointing: every trial appends to a CSV as it completes. Colab dying at trial 37 of 50
costs you trial 37, not the previous 36.

Usage in Colab
--------------
    import sinus, sinus_search
    spot_df = sinus.load_csv(f'{CODE}/spy_1min_2008_2021_cleaned.csv')
    res = sinus_search.run_search(spot_df, n_trials=40, out_csv=f'{DATA}/search_results.csv')
    print(res.head(10))
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sinus import (FeaturePipeline, PipelineConfig, TrainingBundle, CoreModelingEngine,
                   EngineConfig, HORIZONS, log)


# ----------------------------------------------------------------------------- #
# Search space
# ----------------------------------------------------------------------------- #
def sample_config(rng: np.random.Generator) -> Dict[str, Any]:
    """One random point in the search space. Ranges are deliberately wide — the point of a
    search is to discover where the good region is, not to confirm a guess."""
    return {
        # trees
        "learning_rate": float(10 ** rng.uniform(-2.3, -1.0)),        # 0.005 .. 0.10
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "min_data_in_leaf": int(rng.choice([20, 40, 60, 120, 250])),
        "feature_fraction": float(rng.uniform(0.4, 0.95)),
        "lambda_l2": float(10 ** rng.uniform(-0.5, 1.7)),             # 0.3 .. 50
        "robust_delta": float(rng.uniform(0.5, 2.5)),
        "objective": str(rng.choice(["pseudo_huber", "cauchy", "huber"])),
        # tft
        "d_model": int(rng.choice([16, 32, 64])),
        "n_heads": int(rng.choice([2, 4])),
        "dropout": float(rng.uniform(0.05, 0.35)),
        "tft_lr": float(10 ** rng.uniform(-3.5, -2.3)),               # 0.0003 .. 0.005
        "lookback": int(rng.choice([30, 60, 90])),
    }


def _apply(cfg: EngineConfig, p: Dict[str, Any]) -> EngineConfig:
    cfg.tree.learning_rate = p["learning_rate"]
    cfg.tree.num_leaves = p["num_leaves"]
    cfg.tree.min_data_in_leaf = p["min_data_in_leaf"]
    cfg.tree.feature_fraction = p["feature_fraction"]
    cfg.tree.lambda_l2 = p["lambda_l2"]
    cfg.tree.robust_delta = p["robust_delta"]
    cfg.tree.objective = p["objective"]
    cfg.tft.d_model = p["d_model"]
    cfg.tft.n_heads = p["n_heads"]
    cfg.tft.dropout = p["dropout"]
    cfg.tft.lr = p["tft_lr"]
    return cfg


# ----------------------------------------------------------------------------- #
# Scoring
# ----------------------------------------------------------------------------- #
def score_predictions(pred: Dict[str, Any], Y: np.ndarray, horizons=HORIZONS) -> Dict[str, float]:
    """Trimmed MAE + directional accuracy per horizon, on the ensemble mean of whatever
    experts exist. Trimmed at the 95th percentile because a handful of gap bars would
    otherwise decide the whole ranking."""
    out, maes, accs = {}, [], []
    for j, h in enumerate(horizons):
        y = Y[:, j]
        stack = [pred[h][k] for k in ("lgb", "cat", "tft_q50") if k in pred[h]]
        stack = [a for a in stack if np.isfinite(a).any()]        # a family that never fitted is absent, not NaN
        if not stack:
            continue
        arr = np.stack(stack)
        cnt = np.isfinite(arr).sum(0)
        p = np.where(cnt > 0, np.where(np.isfinite(arr), arr, 0.0).sum(0) / np.maximum(cnt, 1), np.nan)
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 30:
            continue
        e = np.abs(p[m] - y[m])
        cut = np.quantile(e, 0.95)
        mae = float(e[e <= cut].mean())
        nz = np.abs(y[m]) > 0.1
        acc = float((np.sign(p[m][nz]) == np.sign(y[m][nz])).mean()) if nz.any() else 0.5
        out[f"mae_{h}"], out[f"acc_{h}"] = mae, acc
        maes.append(mae); accs.append(acc)
    if not maes:
        out["note"] = "no expert produced finite predictions — lightgbm/catboost/torch missing?"
    out["mae_mean"] = float(np.mean(maes)) if maes else float("inf")
    out["acc_mean"] = float(np.mean(accs)) if accs else 0.0
    # lower is better; direction is a mild tiebreak, not a co-equal objective
    out["score"] = out["mae_mean"] - 0.25 * (out["acc_mean"] - 0.5)
    return out


# ----------------------------------------------------------------------------- #
# Search
# ----------------------------------------------------------------------------- #
def run_search(spot_df: pd.DataFrame, chain_df: Optional[pd.DataFrame] = None, n_trials: int = 40,
               tft_epochs: int = 12, out_csv: Optional[str] = None, seed: int = 0,
               time_budget_min: Optional[float] = None, verbose: bool = True) -> pd.DataFrame:
    """Random search over model configs. Returns a DataFrame of trials sorted best-first.

    ``time_budget_min`` stops cleanly when the budget is spent — set it below your Colab
    session limit so results are always written even if the runtime later dies.
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()

    if verbose:
        print("[search] engineering features once (fit on train split only)")
    pipeline = FeaturePipeline(PipelineConfig())
    out = pipeline.fit_transform(spot_df, chain_df, None)
    n_sess = out.meta["session_date"].nunique()
    print(f"[search] {len(out.meta):,} bars · {n_sess} sessions · {out.features.shape[1]} features")
    if n_sess < 30:
        print("[search] WARNING under 30 sessions — the winner of this search will be noise")

    rows: List[Dict[str, Any]] = []
    for i in range(n_trials):
        if time_budget_min and (time.time() - t_start) / 60.0 > time_budget_min:
            print(f"[search] time budget reached after {i} trials")
            break
        p = sample_config(rng)
        t0 = time.time()
        rec: Dict[str, Any] = {"trial": i, **p}
        try:
            train = TrainingBundle.from_phase1(out, "train", lookback=p["lookback"])
            val = TrainingBundle.from_phase1(out, "val", lookback=p["lookback"])
            cfg = _apply(EngineConfig(), p)
            cfg.tft.max_epochs = tft_epochs
            eng = CoreModelingEngine(cfg).fit(train)
            rec.update(score_predictions(eng.predict(val), val.Y))
            rec["status"] = "ok"
        except Exception as e:                       # one bad config must not end the search
            rec.update({"status": f"failed: {type(e).__name__}", "score": float("inf"),
                        "mae_mean": float("inf"), "acc_mean": 0.0})
            if verbose:
                print(f"[search] trial {i} failed: {e}")
                traceback.print_exc(limit=1)
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        if verbose:
            print(f"[search] trial {i:>3}  score {rec.get('score', float('nan')):.4f}  "
                  f"mae {rec.get('mae_mean', float('nan')):.4f}  acc {rec.get('acc_mean', 0):.3f}  "
                  f"{rec['seconds']:.0f}s  lr={p['learning_rate']:.4f} leaves={p['num_leaves']} d={p['d_model']}")
        if out_csv:                                   # checkpoint EVERY trial — disconnects are expected
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[search] wrote {out_csv}")
    if len(df) and np.isfinite(df.iloc[0]["score"]):
        b = df.iloc[0]
        print(f"\n[search] BEST trial {int(b['trial'])}: score {b['score']:.4f} · "
              f"mae {b['mae_mean']:.4f} · dir acc {b['acc_mean']:.3f}")
        print("[search] NOTE this is the best on VALIDATION. Confirm it on test with "
              "confirm_on_test() before believing it.")
    return df


def best_config(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract the winning hyper-parameters from a search result frame."""
    keys = list(sample_config(np.random.default_rng(0)).keys())
    b = df[df["status"] == "ok"].sort_values("score").iloc[0]
    return {k: (b[k].item() if hasattr(b[k], "item") else b[k]) for k in keys}


def confirm_on_test(spot_df: pd.DataFrame, params: Dict[str, Any], chain_df=None,
                    tft_epochs: int = 40, model_dir: str = "sinus_model_best") -> Dict[str, float]:
    """Refit the winning config and score it ONCE on the held-out test split.

    This is the number that matters. Search scores are optimistic by construction — you
    picked the best of N tries on validation, so some of that edge is luck. Test is the
    only split no decision was made against. If test is much worse than validation, the
    search overfit and the honest move is a simpler config, not a better story.
    """
    pipeline = FeaturePipeline(PipelineConfig())
    out = pipeline.fit_transform(spot_df, chain_df, None)
    train = TrainingBundle.from_phase1(out, "train", lookback=params["lookback"])
    test = TrainingBundle.from_phase1(out, "test", lookback=params["lookback"])
    cfg = _apply(EngineConfig(), params)
    cfg.tft.max_epochs = tft_epochs
    eng = CoreModelingEngine(cfg).fit(train)
    eng.save(model_dir)
    s = score_predictions(eng.predict(test), test.Y)
    print(f"[test] mae {s['mae_mean']:.4f} · dir acc {s['acc_mean']:.3f} · saved to {model_dir}")
    for h in HORIZONS:
        if f"mae_{h}" in s:
            print(f"       {h:>3}: mae {s[f'mae_{h}']:.3f}  acc {s[f'acc_{h}']:.3f}")
    return s


def resume(out_csv: str) -> pd.DataFrame:
    """Reload a checkpointed search after a disconnect."""
    df = pd.read_csv(out_csv).sort_values("score").reset_index(drop=True)
    print(f"[search] {len(df)} trials recovered · best score {df.iloc[0]['score']:.4f}")
    return df
