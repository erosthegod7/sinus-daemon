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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sinus import (_C, _colour_enabled, FeaturePipeline, PipelineConfig, TrainingBundle, CoreModelingEngine,
                   EngineConfig, EnsembleFusionEngine, FusionConfig, HORIZONS,
                   fetch_snapshot_ladder, market_state_from_polygon)
from sinus_search import sample_config, _apply, score_predictions, SCORING_VERSION
from sinus_gitstore import GitStore, SAVE_EVERY

LEADERBOARD = "leaderboard.csv"


def _say(msg: str) -> None:
    """print() with the same two-colour rule the logger uses, so the daemon's own lines match
    the library's. Red for champion/prune decisions, green for the rest."""
    if not _colour_enabled():
        print(msg, flush=True)
        return
    low = msg.lower()
    c = (_C.BOLD + _C.RED) if ("champion" in low or "★" in msg) else (
        _C.RED if "prune" in low else (_C.BOLD + _C.RED if ("fatal" in low or "failed" in low) else _C.GREEN))
    print(f"{c}{msg}{_C.RESET}", flush=True)
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


def _seed_bar(board: pd.DataFrame, champ: Optional[float]) -> Tuple[float, str]:
    """Starting value for best_val, and a line explaining it.

    best_val gates whether a trial earns a test evaluation; champ gates promotion. They
    have to agree about what is being defended, and they are seeded from different places
    — the board persists on disk, the champion directory may not. Two ways that goes wrong:

      * No champion, but a board with scores. The bar is set to a number no saved model
        backs, so the search declines to test anything until it beats a score whose weights
        were never kept — it can never promote and never recover. This stalled trials 19-78
        of the 2026-09-04 run: trial 18 set 0.5004, lost its weights to the promotion
        TypeError, and 51 later trials were measured against a ghost.
      * A board mixing scoring versions. A stale low score from before the eod_pre11 cutover
        jams the gate exactly the same way. A board with no scoring_version column predates
        versioning entirely, so every row on it is old-scale.

    Returns +inf in both cases, which lets the next good trial establish a real champion.
    """
    if champ is None:
        return float("inf"), ("no champion on disk — bar starts at +inf, ignoring the board's "
                              "best score so the first good trial can establish one")
    if not len(board) or "status" not in board:
        return float("inf"), "empty board — bar starts at +inf"

    ok = board[board["status"] == "ok"]
    ok = ok[ok["scoring_version"] == SCORING_VERSION] if "scoring_version" in ok.columns else ok.iloc[:0]
    scores = pd.to_numeric(ok["score"], errors="coerce").dropna()
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        return float("inf"), f"no rows on the board scored at {SCORING_VERSION} — bar starts at +inf"
    return float(scores.min()), f"bar seeded from the board at {scores.min():.4f} ({len(scores)} comparable rows)"


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
             test: Dict[str, float], trial: int, pipeline=None, n_sessions: Optional[int] = None) -> None:
    dest = os.path.join(work_dir, CHAMPION)
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(model_dir, dest)
    if pipeline is not None:                          # the scaler is what makes these weights usable live
        pipeline.save(os.path.join(dest, "pipeline.pkl"))
    json.dump({"trial": trial, "params": params, "val_score": val["score"], "test_score": test["score"],
               "test_mae": test["mae_mean"], "test_acc": test["acc_mean"], "n_sessions": n_sessions,
               "scoring_version": SCORING_VERSION,
               "promoted_at": str(pd.Timestamp.now(tz="America/New_York"))},
              open(os.path.join(dest, "champion.json"), "w"), indent=2, default=float)
    _say(f"[daemon] ★ NEW CHAMPION (trial {trial}) test score {test['score']:.4f} · "
          f"mae {test['mae_mean']:.4f} · acc {test['acc_mean']:.3f}")



# ----------------------------------------------------------------------------- #
# Successive-halving screen
# ----------------------------------------------------------------------------- #
def _screen_trial(out, p: Dict[str, Any], cfg: EngineConfig, screen_rounds: int,
                  screen_frac: float) -> float:
    """Cheap first look at a config: TREES ONLY, few rounds, on a slice of the training data.

    The expensive half of a trial is the TFT. Most hopeless configs are already obviously
    hopeless to a shallow forest on a fraction of the data, so paying for the transformer to
    confirm it is waste. This trains on the most RECENT screen_frac of training sessions —
    recent, not random, so the screen never sees data the full fit would treat as future.

    Returns the validation score of the screen. Higher is worse (same convention as the
    full score).
    """
    scr = EngineConfig()
    scr.tree = cfg.tree
    scr.tree.n_rounds_max = screen_rounds
    scr.tree.early_stopping_rounds = max(20, screen_rounds // 4)
    scr.fit_trees_on_all_horizons = True

    train = TrainingBundle.from_phase1(out, "train", lookback=p["lookback"], with_sequences=False)
    val = TrainingBundle.from_phase1(out, "val", lookback=p["lookback"], with_sequences=False)
    if screen_frac < 1.0 and len(train.groups):
        days = np.unique(train.groups)
        keep = days[-max(3, int(len(days) * screen_frac)):]
        m = np.isin(train.groups, keep)
        train = TrainingBundle(X=train.X[m], Y=train.Y[m], groups=train.groups[m],
                               row_idx=train.row_idx[m], feature_names=train.feature_names, seq=None)
    eng = CoreModelingEngine(scr).fit(train)
    return score_predictions(eng.predict(val), val.Y, ts=val.ts)["score"]


# ----------------------------------------------------------------------------- #
# Search loop
# ----------------------------------------------------------------------------- #
def search_forever(spot_df: pd.DataFrame, work_dir: str, chain_df=None, flow_df=None, tft_epochs: int = 12,
                   final_epochs: int = 40, min_improvement: float = 0.002, seed: Optional[int] = None,
                   max_trials: Optional[int] = None, max_hours: Optional[float] = None,
                   plateau_notice: int = 50, prune: bool = True, prune_percentile: float = 60.0,
                   prune_warmup: int = 8, screen_rounds: int = 150, screen_frac: float = 0.5) -> pd.DataFrame:
    """Run trials until stopped. Resumes from the leaderboard on restart.

    ``prune``: two-stage trials. A cheap trees-only screen runs first; if its score is worse
    than ``prune_percentile`` of previous screens, the trial is abandoned before the TFT is
    ever trained. Typically 50-70%% of trials die at the screen for ~15%% of their cost, which
    is where the throughput comes from. The first ``prune_warmup`` trials always run in full,
    because a percentile over two samples is meaningless.

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

    _say(f"[daemon] work_dir {work_dir}")
    _say(f"[daemon] resuming at trial {start_trial} ({len(board)} on the board)")

    print("[daemon] engineering features once (fit on train split only)")
    pipeline = FeaturePipeline(PipelineConfig())
    out = pipeline.fit_transform(spot_df, chain_df, flow_df)
    n_sess = out.meta["session_date"].nunique()
    _say(f"[daemon] {len(out.meta):,} bars · {n_sess} sessions · {out.features.shape[1]} features")
    if n_sess < 100:
        _say(f"[daemon] WARNING {n_sess} sessions. A long search on thin data finds luck, not edge.")

    stop = {"flag": False}
    try:
        _signal.signal(_signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    except (ValueError, AttributeError):
        pass

    t_start = time.time()
    trial = start_trial
    since_promotion = 0
    pruned = 0
    # screen history seeds from the board so pruning stays calibrated across restarts
    screens: List[float] = ([float(x) for x in board["screen_score"].dropna()]
                            if len(board) and "screen_score" in board else [])
    rows = board.to_dict("records") if len(board) else []
    champ = _champion_score(work_dir)
    if champ is not None and not np.isfinite(champ):
        champ = None                                  # a NaN champion would block every promotion
    # NaN guard lives in _seed_bar: an empty "ok" selection makes .min() return NaN, and EVERY
    # comparison against NaN is False — which silently disables promotion forever. This bit on
    # 2026-09-01: ten good trials ran and none was ever considered, because the resumed board
    # held only failures.
    best_val, why = _seed_bar(board, champ)
    _say(f"[daemon] {why}")

    while not stop["flag"]:
        if max_trials and trial - start_trial >= max_trials:
            print("[daemon] max_trials reached"); break
        if max_hours and (time.time() - t_start) / 3600.0 > max_hours:
            print("[daemon] max_hours reached"); break

        p = sample_config(rng)
        rec: Dict[str, Any] = {"trial": trial, "ts": str(pd.Timestamp.now(tz="America/New_York")), **p}
        t0 = time.time()
        try:
            cfg = _apply(EngineConfig(), p)
            cfg.tft.max_epochs = tft_epochs
            # Val turns at epoch 0-1 and never comes back. patience=12 burned a dozen
            # epochs per trial proving that again each time.
            cfg.tft.patience = int(os.environ.get("SINUS_TFT_PATIENCE", "3"))

            if prune:
                sc = _screen_trial(out, p, _apply(EngineConfig(), p), screen_rounds, screen_frac)
                rec["screen_score"] = sc
                if len(screens) >= prune_warmup and np.isfinite(sc):
                    bar = float(np.percentile([s for s in screens if np.isfinite(s)], prune_percentile))
                    if sc > bar:
                        rec.update({"status": "pruned", "score": float("inf"),
                                    "mae_mean": float("inf"), "acc_mean": 0.0, "prune_bar": bar})
                        rec["seconds"] = round(time.time() - t0, 1)
                        rows.append(rec); screens.append(sc)
                        _save_board(work_dir, pd.DataFrame(rows))
                        pruned += 1
                        if (trial + 1) % SAVE_EVERY == 0:
                            git.push_leaderboard(pd.DataFrame(rows), trial)
                        _say(f"[daemon] trial {trial:>4}  PRUNED at screen {sc:.4f} > p{prune_percentile:.0f} "
                              f"bar {bar:.4f}  {rec['seconds']:.0f}s  ({pruned}/{trial - start_trial + 1} pruned)")
                        trial += 1
                        since_promotion += 1
                        continue
                screens.append(sc)

            train = TrainingBundle.from_phase1(out, "train", lookback=p["lookback"])
            val = TrainingBundle.from_phase1(out, "val", lookback=p["lookback"])
            eng = CoreModelingEngine(cfg).fit(train)
            vs = score_predictions(eng.predict(val), val.Y, ts=val.ts)
            rec.update(vs); rec["status"] = "ok"

            better = np.isfinite(vs["score"]) and vs["score"] < best_val - min_improvement
            if better:
                # only now is a test evaluation justified: refit longer, score once on test
                cand_dir = os.path.join(work_dir, "_candidate")
                cfg.tft.max_epochs = final_epochs
                # `train` is already the bundle for (out, "train", this lookback) and building
                # one is not cheap — sliding windows over every session. Only the epoch budget
                # changed, and that lives on cfg, not on the data.
                eng2 = CoreModelingEngine(cfg).fit(train)
                eng2.save(cand_dir)
                test_b = TrainingBundle.from_phase1(out, "test", lookback=p["lookback"])
                test_s = score_predictions(eng2.predict(test_b), test_b.Y, ts=test_b.ts)
                rec["test_score"], rec["test_mae"], rec["test_acc"] = test_s["score"], test_s["mae_mean"], test_s["acc_mean"]
                if champ is None or test_s["score"] < champ - min_improvement:
                    _promote(work_dir, cand_dir, p, vs, test_s, trial, pipeline=pipeline, n_sessions=int(n_sess))
                    champ = test_s["score"]
                    since_promotion = 0
                    git.push_champion(os.path.join(work_dir, CHAMPION),
                                      json.load(open(os.path.join(work_dir, CHAMPION, "champion.json"))),
                                      min_improvement)
                else:
                    _say(f"[daemon] trial {trial} beat validation but NOT test "
                          f"({test_s['score']:.4f} vs champion {champ:.4f}) — not promoted. "
                          f"That gap is the search overfitting; the gate did its job.")
                    since_promotion += 1
                # Advance the bar ONLY after the test+promote path survived. Moving it
                # earlier meant one exception raised best_val permanently while champ
                # stayed None, so nothing could ever promote again.
                best_val = vs["score"]
                shutil.rmtree(cand_dir, ignore_errors=True)
            else:
                since_promotion += 1
        except Exception as e:
            rec.update({"status": f"failed: {type(e).__name__}", "score": float("inf"),
                        "mae_mean": float("inf"), "acc_mean": 0.0})
            _say(f"[daemon] trial {trial} failed: {e}")
            traceback.print_exc(limit=1)
            since_promotion += 1

        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        _save_board(work_dir, pd.DataFrame(rows))
        if (trial + 1) % SAVE_EVERY == 0:
            git.push_leaderboard(pd.DataFrame(rows), trial)
        el = (time.time() - t_start) / 3600.0
        _say(f"[daemon] trial {trial:>4}  score {rec.get('score', float('nan')):.4f}  "
              f"best {best_val:.4f}  champ {champ if champ is not None else float('nan'):.4f}  "
              f"{rec['seconds']:.0f}s  ({el:.1f}h elapsed)")

        if since_promotion and since_promotion % plateau_notice == 0:
            _say(f"[daemon] ── {since_promotion} trials without a promotion. The configuration space is "
                  f"not the bottleneck any more; the DATA is. More trials will not help. What helps: more "
                  f"sessions, or better features (the options ladders).")
        trial += 1

    df = pd.DataFrame(rows)
    _save_board(work_dir, df)
    git.push_leaderboard(df, trial - 1)               # never lose the tail on a clean stop
    _say(f"[daemon] stopped after {trial - start_trial} trials this run "
          f"({pruned} pruned at screen) · board {len(df)} rows")
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
    """Today's numbers: the champion's ML experts on a live feature window from Polygon, fused
    with the physics layer.

    Falls back to physics-only — and says so in the output — when there is no champion, when
    the champion was fitted on a different feature set than this build produces, or when the
    live window cannot be built. Stated, never silent: a physics-only call that reads like a
    champion call is the 9/4 failure.
    """
    # The trainer owns the feature contract: six horizons, the candle + ATM + lag blocks, its
    # chain builder. Without these installed, transform() yields the base matrix and the scaler
    # pads the rest with NaN — the model runs, and is quietly wrong.
    want = None
    try:
        import sinus_train
        sinus_train.install_horizons()
        sinus_train.install_extra_features()
        want = sinus_train.FEATURE_SET
    except Exception as e:
        print(f"[serve] trainer feature installs unavailable ({type(e).__name__}: {e})")

    champ_dir = os.path.join(work_dir, CHAMPION)
    meta_p = os.path.join(champ_dir, "champion.json")
    why = None
    if want is None:
        why = "feature installs missing"
    elif not os.path.exists(meta_p):
        why = "no champion yet"
    else:
        have = json.load(open(meta_p)).get("feature_set")
        if have != want:
            why = f"champion is feature_set {have!r}, this build serves {want!r} — refusing it"
    if why is None:
        try:
            from sinus import predict_live
            return predict_live(symbol, champion_dir=champ_dir, log_csv=log_csv)
        except Exception as e:
            why = f"live window failed: {type(e).__name__}: {e}"
            traceback.print_exc(limit=2)

    # ---- physics-only, and the output says exactly why ------------------------------------
    # Say it BEFORE touching the ladder: on a weekend or holiday the ladder fetch itself raises
    # (no 0DTE expiry), and a run that dies there must still have recorded why it was not
    # a champion call in the first place.
    print(f"[serve] physics-only ({why})")
    lad, spot_px, _ = fetch_snapshot_ladder(symbol, spot=spot)
    now = pd.Timestamp.now(tz="America/New_York")
    mtc = max((now.normalize() + pd.Timedelta(hours=16) - now).total_seconds() / 60.0, 0.0)
    st = market_state_from_polygon(lad, spot_px, mtc)
    ml, tag = {h: {} for h in HORIZONS}, f"physics-only ({why})"
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
