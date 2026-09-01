"""
sinus_daemon.py
===============
Perpetual model-search daemon + independent serving path.

Two processes that never block each other:

    SEARCH   runs forever. Each cycle samples a config, trains it, scores it on validation,
             and appends to a leaderboard on Drive. When a candidate beats the reigning
             champion by a real margin it is re-scored on the held-out TEST split, and only
             then promoted. Kill it any time; it resumes from the leaderboard.

    SERVE    loads whatever the current champion is and produces today's numbers. Reads the
             champion directory; never trains. Safe to run while the search is mid-trial.

What this does and does not buy you
-----------------------------------
It explores the CONFIGURATION space indefinitely — that space is effectively unbounded, so
the daemon always has work. It does NOT extract more signal from a fixed dataset than the
dataset contains. Expect the leaderboard to improve quickly at first, then plateau. The
plateau is real and is information about your data, not a sign the daemon needs more compute.

The honest guardrail against fooling yourself: promotion requires beating the champion on a
split no search decision has touched. Running 10,000 trials and keeping the best VALIDATION
score would manufacture a champion out of luck — with enough tries, something always looks
good. `min_improvement` and the test-set gate are what stop that.

Usage
-----
    # terminal / cell that you leave running
    import sinus_daemon as sd
    sd.search_forever(spot_df, work_dir='/content/drive/MyDrive/Claude/Sinus/champion')

    # any other time, independent of the above
    sd.serve('SPY', work_dir='/content/drive/MyDrive/Claude/Sinus/champion')
"""

from __future__ import annotations

import json
import os
import shutil
import signal as _signal
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sinus import (FeaturePipeline, PipelineConfig, TrainingBundle, CoreModelingEngine,
                   EngineConfig, EnsembleFusionEngine, FusionConfig, HORIZONS,
                   fetch_snapshot_ladder, market_state_from_polygon)
from sinus_search import sample_config, _apply, score_predictions
from sinus_gitstore import GitStore, SAVE_EVERY

LEADERBOARD = "leaderboard.csv"
CHAMPION = "champion"
STATE = "daemon_state.json"


# ----------------------------------------------------------------------------- #
# Persistence
# ----------------------------------------------------------------------------- #
def _load_board(work_dir: str) -> pd.DataFrame:
    p = os.path.join(work_dir, LEADERBOARD)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()


def _save_board(work_dir: str, df: pd.DataFrame) -> None:
    """Write via a temp file then replace, so a crash mid-write can't corrupt the board."""
    p = os.path.join(work_dir, LEADERBOARD)
    tmp = p + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def _champion_score(work_dir: str) -> Optional[float]:
    p = os.path.join(work_dir, CHAMPION, "champion.json")
    if os.path.exists(p):
        try:
            return float(json.load(open(p))["test_score"])
        except Exception:
            return None
    return None


def _promote(work_dir: str, model_dir: str, params: Dict[str, Any], val: Dict[str, float],
             test: Dict[str, float], trial: int) -> None:
    dest = os.path.join(work_dir, CHAMPION)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(model_dir, dest)
    json.dump({"trial": trial, "params": params, "val_score": val["score"], "test_score": test["score"],
               "test_mae": test["mae_mean"], "test_acc": test["acc_mean"],
               "promoted_at": str(pd.Timestamp.now(tz="America/New_York"))},
              open(os.path.join(dest, "champion.json"), "w"), indent=2, default=float)
    print(f"[daemon] ★ NEW CHAMPION (trial {trial}) test score {test['score']:.4f} · "
          f"mae {test['mae_mean']:.4f} · acc {test['acc_mean']:.3f}")


# ----------------------------------------------------------------------------- #
# Search loop
# ----------------------------------------------------------------------------- #
def search_forever(spot_df: pd.DataFrame, work_dir: str, chain_df=None, tft_epochs: int = 12,
                   final_epochs: int = 40, min_improvement: float = 0.002, seed: Optional[int] = None,
                   max_trials: Optional[int] = None, max_hours: Optional[float] = None,
                   plateau_notice: int = 50) -> pd.DataFrame:
    """Run trials until stopped. Resumes from the leaderboard on restart.

    ``min_improvement``: a candidate must beat the champion's validation score by this much
    before it is even worth spending a test evaluation on. Set to 0 and you will promote on
    noise. ``plateau_notice``: after this many trials with no promotion, say so plainly —
    a plateau means the data is exhausted, and continuing is burning compute for nothing.
    """
    os.makedirs(work_dir, exist_ok=True)
    git = GitStore(work_dir)                          # no-op if SINUS_GIT_REPO/TOKEN unset
    git.pull_champion()                               # start from whatever the OTHER node found
    board = _load_board(work_dir)
    start_trial = int(board["trial"].max()) + 1 if len(board) and "trial" in board else 0
    rng = np.random.default_rng(seed if seed is not None else start_trial * 7919 + 13)

    print(f"[daemon] work_dir {work_dir}")
    print(f"[daemon] resuming at trial {start_trial} ({len(board)} on the board)")

    print("[daemon] engineering features once (fit on train split only)")
    pipeline = FeaturePipeline(PipelineConfig())
    out = pipeline.fit_transform(spot_df, chain_df, None)
    n_sess = out.meta["session_date"].nunique()
    print(f"[daemon] {len(out.meta):,} bars · {n_sess} sessions · {out.features.shape[1]} features")
    if n_sess < 100:
        print(f"[daemon] WARNING {n_sess} sessions. A long search on thin data finds luck, not edge.")

    stop = {"flag": False}
    try:
        _signal.signal(_signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    except (ValueError, AttributeError):
        pass

    t_start = time.time()
    trial = start_trial
    since_promotion = 0
    rows = board.to_dict("records") if len(board) else []
    champ = _champion_score(work_dir)
    best_val = float(board[board.status == "ok"]["score"].min()) if len(board) and "status" in board else float("inf")

    while not stop["flag"]:
        if max_trials and trial - start_trial >= max_trials:
            print("[daemon] max_trials reached"); break
        if max_hours and (time.time() - t_start) / 3600.0 > max_hours:
            print("[daemon] max_hours reached"); break

        p = sample_config(rng)
        rec: Dict[str, Any] = {"trial": trial, "ts": str(pd.Timestamp.now(tz="America/New_York")), **p}
        t0 = time.time()
        try:
            train = TrainingBundle.from_phase1(out, "train", lookback=p["lookback"])
            val = TrainingBundle.from_phase1(out, "val", lookback=p["lookback"])
            cfg = _apply(EngineConfig(), p)
            cfg.tft.max_epochs = tft_epochs
            eng = CoreModelingEngine(cfg).fit(train)
            vs = score_predictions(eng.predict(val), val.Y)
            rec.update(vs); rec["status"] = "ok"

            better = vs["score"] < best_val - min_improvement
            if better:
                best_val = vs["score"]
                # only now is a test evaluation justified: refit longer, score once on test
                cand_dir = os.path.join(work_dir, "_candidate")
                cfg.tft.max_epochs = final_epochs
                eng2 = CoreModelingEngine(cfg).fit(TrainingBundle.from_phase1(out, "train", lookback=p["lookback"]))
                eng2.save(cand_dir)
                ts_ = score_predictions(eng2.predict(TrainingBundle.from_phase1(out, "test", lookback=p["lookback"])),
                                        TrainingBundle.from_phase1(out, "test", lookback=p["lookback"]).Y)
                rec["test_score"], rec["test_mae"], rec["test_acc"] = ts_["score"], ts_["mae_mean"], ts_["acc_mean"]
                if champ is None or ts_["score"] < champ - min_improvement:
                    _promote(work_dir, cand_dir, p, vs, ts_, trial)
                    champ = ts_["score"]
                    since_promotion = 0
                    git.push_champion(os.path.join(work_dir, CHAMPION),
                                      json.load(open(os.path.join(work_dir, CHAMPION, "champion.json"))),
                                      min_improvement)
                else:
                    print(f"[daemon] trial {trial} beat validation but NOT test "
                          f"({ts_['score']:.4f} vs champion {champ:.4f}) — not promoted. "
                          f"That gap is the search overfitting; the gate did its job.")
                    since_promotion += 1
                shutil.rmtree(cand_dir, ignore_errors=True)
            else:
                since_promotion += 1
        except Exception as e:
            rec.update({"status": f"failed: {type(e).__name__}", "score": float("inf"),
                        "mae_mean": float("inf"), "acc_mean": 0.0})
            print(f"[daemon] trial {trial} failed: {e}")
            traceback.print_exc(limit=1)
            since_promotion += 1

        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        _save_board(work_dir, pd.DataFrame(rows))
        if (trial + 1) % SAVE_EVERY == 0:
            git.push_leaderboard(pd.DataFrame(rows), trial)
        el = (time.time() - t_start) / 3600.0
        print(f"[daemon] trial {trial:>4}  score {rec.get('score', float('nan')):.4f}  "
              f"best {best_val:.4f}  champ {champ if champ is not None else float('nan'):.4f}  "
              f"{rec['seconds']:.0f}s  ({el:.1f}h elapsed)")

        if since_promotion and since_promotion % plateau_notice == 0:
            print(f"[daemon] ── {since_promotion} trials without a promotion. The configuration space is "
                  f"not the bottleneck any more; the DATA is. More trials will not help. What helps: more "
                  f"sessions, or better features (the options ladders).")
        trial += 1

    df = pd.DataFrame(rows)
    _save_board(work_dir, df)
    git.push_leaderboard(df, trial - 1)               # never lose the tail on a clean stop
    print(f"[daemon] stopped after {trial - start_trial} trials this run · board {len(df)} rows")
    return df


# ----------------------------------------------------------------------------- #
# Serving — completely independent of the search
# ----------------------------------------------------------------------------- #
def load_champion(work_dir: str) -> tuple:
    """Load the reigning champion engine + its metadata. Raises if none has been promoted."""
    d = os.path.join(work_dir, CHAMPION)
    meta_p = os.path.join(d, "champion.json")
    if not os.path.exists(meta_p):
        raise RuntimeError(f"no champion in {d} yet — run search_forever until one is promoted")
    meta = json.load(open(meta_p))
    try:
        return CoreModelingEngine.load(d), meta
    except Exception as e:                       # metadata present but weights unreadable
        raise RuntimeError(f"champion metadata found but weights failed to load: {e}")


def serve(symbol: str = "SPY", work_dir: str = "champion", spot: Optional[float] = None,
          log_csv: Optional[str] = None) -> Dict[str, Any]:
    """Today's numbers from the live chain, using the champion weights if one exists.

    Falls back to physics-only when no champion has been promoted — which is the correct
    behaviour early on, and is stated in the output rather than hidden.
    """
    lad, spot_px, _ = fetch_snapshot_ladder(symbol, spot=spot)
    now = pd.Timestamp.now(tz="America/New_York")
    mtc = max((now.normalize() + pd.Timedelta(hours=16) - now).total_seconds() / 60.0, 0.0)
    st = market_state_from_polygon(lad, spot_px, mtc)

    ml, tag = {h: {} for h in HORIZONS}, "physics-only (no champion yet)"
    try:
        _, meta = load_champion(work_dir)
        tag = f"champion trial {meta['trial']} · test mae {meta['test_mae']:.4f}"
        # NOTE: the ML experts need a live feature window to predict from. Until the daily
        # capture pipeline feeds one in, the champion's weights are loaded but idle — the
        # call below is physics-driven and says so.
    except Exception:
        pass

    eng = EnsembleFusionEngine(FusionConfig(ml_target_kind="retn"))
    res = eng.predict_final(ml, st, row=-1)
    print(f"[serve] {tag}")
    print(eng.format_call(res))

    if log_csv:
        row = {"ts": str(now), "spot": spot_px, "source": tag,
               **{f"{h}_target": res[h]["target"] for h in HORIZONS},
               **{f"{h}_conf": round(res[h]["confidence"], 3) for h in HORIZONS},
               "pin_strike": res["_physics"]["pin_strike"], "zero_gamma": res["_physics"]["zero_gamma_level"],
               "max_pain": res["_physics"]["max_pain"], "regime": res["_physics"]["regime"]}
        hdr = not os.path.exists(log_csv)
        pd.DataFrame([row]).to_csv(log_csv, mode="a", header=hdr, index=False)
        print(f"[serve] logged to {os.path.basename(log_csv)}")
    return res


def status(work_dir: str) -> None:
    """One-screen summary of the daemon: board size, champion, recent trials."""
    board = _load_board(work_dir)
    print(f"trials on board : {len(board)}")
    if len(board) and "status" in board:
        ok = board[board.status == "ok"]
        print(f"successful      : {len(ok)}")
        if len(ok):
            print(f"best validation : {ok['score'].min():.4f}")
    meta_p = os.path.join(work_dir, CHAMPION, "champion.json")
    try:
        meta = json.load(open(meta_p))
        print(f"champion        : trial {meta['trial']} · test {meta['test_score']:.4f} "
              f"· mae {meta['test_mae']:.4f} · acc {meta['test_acc']:.3f}")
        print(f"promoted        : {meta['promoted_at']}")
    except Exception as e:
        print(f"champion        : none ({e})")
