r"""
sinus_evolve.py
===============
Generational champion search. Replaces the random search loop in sinus_daemon with the
thing Anthony described (it's a (mu, lambda) evolution strategy with elitism):

    generation = POP configs
    every GEN_MINUTES (or when the whole generation has been scored, whichever first):
        rank by horizon-weighted score
        CULL the worst CULL_FRAC (default 70%)
        the survivors STAY, untouched
        REFILL the empty slots with children of the survivors:
            ~1/3 clones with small mutations  (one or two knobs nudged)
            ~1/3 clones with large mutations  (several knobs, wider jumps)
            ~1/3 crossovers                    (two survivors, each knob from one parent)
    a child that beats the reigning champion on VALIDATION by min_improvement is refit longer
    and scored ONCE on the held-out TEST split; only a test win promotes. Same gate as before —
    breeding does not get to skip it.

Horizon weights (the scoring Anthony asked for; 10m is picked up automatically if HORIZONS
ever includes it):
    5m 0.30 · 10m 0.30 · 15m 0.25 · 30m 0.25 · 1h 0.08 · eod 0.02   (renormalised over what exists)

On every promotion the magnitude heads (sinus_magnitude.py) are fitted on the same Phase 1
features and saved into the champion folder, so predict_live can print the meter.

Drop-in: railway_daemon calls evolve_forever(spot_df, work_dir=..., chain_df=..., flow_df=...,
tft_epochs=...) with the same keyword set it used for search_forever; unknown keywords are
ignored. Resumes from generation state on disk (evolve_state.json) and the leaderboard.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal as _signal
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sinus import (FeaturePipeline, PipelineConfig, TrainingBundle, CoreModelingEngine, EngineConfig, HORIZONS, STOP, SinusStopped)
from sinus_search import sample_config, _apply, score_predictions
from sinus_daemon import (_say, _load_board, _save_board, _champion_score, _promote, _screen_trial, CHAMPION)
try:
    from sinus_gitstore import GitStore, SAVE_EVERY
except Exception:  # pragma: no cover
    GitStore, SAVE_EVERY = None, 10

STATE = "evolve_state.json"

HORIZON_WEIGHTS = {"5m": 0.30, "10m": 0.30, "15m": 0.25, "30m": 0.25, "1h": 0.08, "60m": 0.08, "eod": 0.02}

# the search space, mirrored from sinus_search.sample_config so mutation stays inside it
SPACE = {
    "learning_rate":    ("log", -2.3, -1.0),
    "num_leaves":       ("choice", [15, 31, 63, 127]),
    "min_data_in_leaf": ("choice", [20, 40, 60, 120, 250]),
    "feature_fraction": ("lin", 0.4, 0.95),
    "lambda_l2":        ("log", -0.5, 1.7),
    "robust_delta":     ("lin", 0.5, 2.5),
    "objective":        ("choice", ["pseudo_huber", "cauchy", "huber"]),
    "d_model":          ("choice", [16, 32, 64]),
    "n_heads":          ("choice", [2, 4]),
    "dropout":          ("lin", 0.05, 0.35),
    "tft_lr":           ("log", -3.5, -2.3),
    "lookback":         ("choice", [30, 60, 90]),
}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def weighted_score(per_h: Dict[str, float], horizons=HORIZONS) -> float:
    """Horizon-weighted score from score_predictions output (lower is better).
    score = Σ w_h · mae_h · (1.5 − acc_h): a horizon that gets direction right is rewarded
    on top of being close. Renormalised over the horizons actually present."""
    num, den = 0.0, 0.0
    for h in horizons:
        mae, acc = per_h.get(f"mae_{h}"), per_h.get(f"acc_{h}")
        if mae is None or not np.isfinite(mae):
            continue
        w = HORIZON_WEIGHTS.get(str(h), 0.05)
        acc = acc if (acc is not None and np.isfinite(acc)) else 0.5
        num += w * mae * (1.5 - acc)
        den += w
    return float(num / den) if den > 0 else float("inf")


def _score(engine, bundle) -> Dict[str, float]:
    s = score_predictions(engine.predict(bundle), bundle.Y, ts=bundle.ts)
    s["raw_score"] = s.get("score", float("nan"))
    s["score"] = weighted_score(s)
    return s


# --------------------------------------------------------------------------- #
# genetics
# --------------------------------------------------------------------------- #
def _clip(key: str, val):
    kind = SPACE[key]
    if kind[0] == "log":
        return float(10 ** np.clip(math.log10(max(val, 1e-9)), kind[1], kind[2]))
    if kind[0] == "lin":
        return float(np.clip(val, kind[1], kind[2]))
    return val


def mutate(p: Dict[str, Any], rng: np.random.Generator, strength: str = "small") -> Dict[str, Any]:
    child = dict(p)
    n_knobs = 1 if strength == "small" else int(rng.integers(3, 6))
    sigma = 0.15 if strength == "small" else 0.5
    keys = list(SPACE)
    for key in rng.choice(keys, size=min(n_knobs, len(keys)), replace=False):
        kind = SPACE[key]
        if kind[0] == "log":
            child[key] = _clip(key, 10 ** (math.log10(p[key]) + rng.normal(0, sigma * (kind[2] - kind[1]))))
        elif kind[0] == "lin":
            child[key] = _clip(key, p[key] + rng.normal(0, sigma * (kind[2] - kind[1])))
        else:
            opts = kind[1]
            if strength == "small" and p[key] in opts and len(opts) > 2:
                i = opts.index(p[key])
                j = int(np.clip(i + rng.choice([-1, 1]), 0, len(opts) - 1))
                child[key] = opts[j]
            else:
                child[key] = opts[int(rng.integers(len(opts)))]
    # keep types honest
    for key in ("num_leaves", "min_data_in_leaf", "d_model", "n_heads", "lookback"):
        child[key] = int(child[key])
    return child


def crossover(a: Dict[str, Any], b: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    child = {k: (a[k] if rng.random() < 0.5 else b[k]) for k in SPACE}
    return mutate(child, rng, "small") if rng.random() < 0.3 else child


def make_children(parents: List[Dict[str, Any]], n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    kids = []
    while len(kids) < n:
        r = rng.random()
        if r < 0.34 or len(parents) < 2:
            kids.append(("mut_s", mutate(parents[int(rng.integers(len(parents)))], rng, "small")))
        elif r < 0.67:
            kids.append(("mut_l", mutate(parents[int(rng.integers(len(parents)))], rng, "large")))
        else:
            i, j = rng.choice(len(parents), size=2, replace=False)
            kids.append(("cross", crossover(parents[i], parents[j], rng)))
    return kids[:n]


def _genes(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: p[k] for k in SPACE if k in p}


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def _load_state(work_dir: str) -> Dict[str, Any]:
    p = os.path.join(work_dir, STATE)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"generation": 0, "population": [], "trial": 0}


def _save_state(work_dir: str, st: Dict[str, Any]) -> None:
    p = os.path.join(work_dir, STATE)
    json.dump(st, open(p + ".tmp", "w"), indent=1, default=float)
    os.replace(p + ".tmp", p)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def evolve_forever(spot_df: pd.DataFrame, work_dir: str, chain_df=None, flow_df=None,
                   tft_epochs: int = 12, final_epochs: int = 40,
                   screen_margin: float = float(os.environ.get("SINUS_SCREEN_MARGIN", "0.002")),
                   promote_margin: float = float(os.environ.get("SINUS_PROMOTE_MARGIN", "0.0005")),
                   min_improvement: Optional[float] = None,
                   pop: int = int(os.environ.get("SINUS_POP", "20")),
                   cull_frac: float = float(os.environ.get("SINUS_CULL", "0.70")),
                   gen_minutes: float = float(os.environ.get("SINUS_GEN_MINUTES", "25")),
                   screen_rounds: int = 150, screen_frac: float = 0.5, seed: Optional[int] = None,
                   max_generations: Optional[int] = None, **_ignored) -> pd.DataFrame:
    os.makedirs(work_dir, exist_ok=True)
    git = GitStore(work_dir) if GitStore else None
    if git:
        git.pull_champion()
    board = _load_board(work_dir)
    rows = board.to_dict("records") if len(board) else []
    st = _load_state(work_dir)
    trial = max(int(st.get("trial", 0)), int(board["trial"].max()) + 1 if len(board) and "trial" in board else 0)
    rng = np.random.default_rng(seed if seed is not None else trial * 7919 + 17)

    _say(f"[evolve] work_dir {work_dir} · pop {pop} · cull {cull_frac:.0%} every {gen_minutes:.0f} min · "
         f"weights {dict((h, HORIZON_WEIGHTS.get(str(h), 0.05)) for h in HORIZONS)}")

    print("[evolve] engineering features once (fit on train split only)")
    pipeline = FeaturePipeline(PipelineConfig())
    out = pipeline.fit_transform(spot_df, chain_df, flow_df)
    n_sess = out.meta["session_date"].nunique()
    _say(f"[evolve] {len(out.meta):,} bars · {n_sess} sessions · {out.features.shape[1]} features · "
         f"book={'yes' if chain_df is not None else 'no'}")

    stop = STOP                                   # shared with the training loops in sinus.py
    stop["flag"] = False

    def _on_sigint(*_):
        """First press asks for a clean stop and restores the default handler, so a
        second press raises KeyboardInterrupt and kills the process outright. The old
        handler swallowed every press forever."""
        stop["flag"] = True
        _say("[evolve] stop requested — finishing the current step. Press Ctrl-C again to force.")
        try:
            _signal.signal(_signal.SIGINT, _signal.default_int_handler)
        except (ValueError, AttributeError, TypeError):
            pass

    try:
        _signal.signal(_signal.SIGINT, _on_sigint)
    except (ValueError, AttributeError):
        pass

    if min_improvement is not None:              # old callers keep working
        screen_margin = promote_margin = min_improvement
    champ = _champion_score(work_dir)
    if champ is not None and not np.isfinite(champ):
        champ = None
    # The screen bar is the CHAMPION's validation score. It used to be the minimum
    # over every ok trial, which meant a candidate that failed promotion still raised
    # the bar it had just failed to clear — locking its own genes out of ever being
    # evaluated again, however many times breeding rediscovered them.
    best_val = float("inf")
    try:
        _cv = float(json.load(open(os.path.join(work_dir, CHAMPION, "champion.json")))["val_score"])
        if np.isfinite(_cv):
            best_val = _cv
    except Exception:
        pass
    _say(f"[evolve] screen bar {best_val:.5f} (-{screen_margin}) · promote bar "
         f"{'none' if champ is None else format(champ, '.5f')} (-{promote_margin})")

    # population: resume or seed
    population: List[Dict[str, Any]] = st.get("population") or []
    if not population:
        population = [{"genes": sample_config(rng), "origin": "seed", "score": None} for _ in range(pop)]
        # if a champion exists, it is always member zero
        cj = os.path.join(work_dir, CHAMPION, "champion.json")
        if os.path.exists(cj):
            try:
                population[0] = {"genes": _genes(json.load(open(cj))["params"]), "origin": "champion", "score": None}
            except Exception:
                pass
    generation = int(st.get("generation", 0))

    while not stop["flag"]:
        if max_generations is not None and generation >= max_generations:
            break
        t_gen = time.time()
        _say(f"[evolve] ── generation {generation} · {sum(1 for m in population if m['score'] is None)} unscored of {len(population)}")

        # ---- score everyone who needs it, within the time budget ------------------------------
        for member in population:
            if stop["flag"]:
                break
            if member["score"] is not None:
                continue
            if (time.time() - t_gen) / 60.0 > gen_minutes:
                break
            p = member["genes"]
            rec: Dict[str, Any] = {"trial": trial, "generation": generation, "origin": member["origin"],
                                   "ts": str(pd.Timestamp.now(tz="America/New_York")), **p}
            t0 = time.time()
            try:
                cfg = _apply(EngineConfig(), p)
                cfg.tft.max_epochs = tft_epochs
                sc = _screen_trial(out, p, _apply(EngineConfig(), p), screen_rounds, screen_frac)
                rec["screen_score"] = sc
                train = TrainingBundle.from_phase1(out, "train", lookback=p["lookback"])
                val = TrainingBundle.from_phase1(out, "val", lookback=p["lookback"])
                eng = CoreModelingEngine(cfg).fit(train)
                vs = _score(eng, val)
                rec.update(vs)
                rec["status"] = "ok"
                member["score"] = float(vs["score"]) if np.isfinite(vs["score"]) else float("inf")

                if (np.isfinite(vs["score"]) and vs["score"] < best_val - screen_margin
                        and not stop["flag"]):     # never start a long refit on the way out
                    # best_val is NOT advanced here — only a promotion moves the bar.
                    cand = os.path.join(work_dir, "_candidate")
                    cfg.tft.max_epochs = final_epochs
                    eng2 = CoreModelingEngine(cfg).fit(TrainingBundle.from_phase1(out, "train", lookback=p["lookback"]))
                    eng2.save(cand)
                    test = TrainingBundle.from_phase1(out, "test", lookback=p["lookback"])
                    ts_ = _score(eng2, test)
                    rec["test_score"], rec["test_mae"], rec["test_acc"] = ts_["score"], ts_["mae_mean"], ts_["acc_mean"]
                    if champ is None or ts_["score"] < champ - promote_margin:
                        _promote(work_dir, cand, p, vs, ts_, trial, pipeline=pipeline, n_sessions=int(n_sess))
                        champ = ts_["score"]
                        best_val = vs["score"]     # the bar moves only when the champion does
                        try:
                            from sinus_magnitude import fit_magnitude
                            (fit_magnitude(out, spot_df, os.path.join(work_dir, CHAMPION)) if os.environ.get('SINUS_FIT_MAG','1')=='1' else None)
                        except Exception as e:
                            _say(f"[evolve] magnitude heads failed: {type(e).__name__}: {e}")
                        if git:
                            git.push_champion(os.path.join(work_dir, CHAMPION),
                                              json.load(open(os.path.join(work_dir, CHAMPION, "champion.json"))),
                                              promote_margin)
                    else:
                        _say(f"[evolve] trial {trial} beat validation but NOT test "
                             f"({ts_['score']:.4f} vs champion {champ:.4f}, short of the "
                             f"{promote_margin} margin by {promote_margin - (champ - ts_['score']):.5f})"
                             f" — not promoted")
                    shutil.rmtree(cand, ignore_errors=True)
            except SinusStopped:
                _say("[evolve] stopped mid-training — this trial is discarded, the board is intact")
                stop["flag"] = True
                break
            except Exception as e:
                rec.update({"status": f"failed: {type(e).__name__}", "score": float("inf"),
                            "mae_mean": float("inf"), "acc_mean": 0.0})
                member["score"] = float("inf")
                _say(f"[evolve] trial {trial} failed: {e}")
                traceback.print_exc(limit=1)
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
            _save_board(work_dir, pd.DataFrame(rows))
            if git and (trial + 1) % SAVE_EVERY == 0:
                git.push_leaderboard(pd.DataFrame(rows), trial)
            _say(f"[evolve] g{generation} trial {trial:>4} {member['origin']:<8} score {member['score']:.4f}  "
                 f"best {best_val:.4f}  champ {champ if champ is not None else float('nan'):.4f}  {rec['seconds']:.0f}s")
            trial += 1
            st.update({"generation": generation, "population": population, "trial": trial})
            _save_state(work_dir, st)

        if stop["flag"]:
            break

        # ---- cull and refill --------------------------------------------------------------------
        scored = [m for m in population if m["score"] is not None]
        unscored = [m for m in population if m["score"] is None]
        scored.sort(key=lambda m: m["score"])
        n_keep = max(2, int(round(len(population) * (1.0 - cull_frac))))
        survivors = scored[:n_keep]
        culled = len(scored) - len(survivors)
        # unscored members (time ran out) carry over so nobody is judged before being scored
        kids = make_children([m["genes"] for m in survivors], len(population) - len(survivors) - len(unscored), rng)
        population = ([{"genes": m["genes"], "origin": "elite", "score": m["score"]} for m in survivors]
                      + unscored
                      + [{"genes": g, "origin": o, "score": None} for o, g in kids])
        _say(f"[evolve] ── cut {culled} · kept {len(survivors)} (best {survivors[0]['score']:.4f}) · "
             f"bred {len(kids)} ({sum(1 for o, _ in kids if o == 'mut_s')} small / "
             f"{sum(1 for o, _ in kids if o == 'mut_l')} large / {sum(1 for o, _ in kids if o == 'cross')} cross) · "
             f"{(time.time() - t_gen) / 60:.1f} min")
        generation += 1
        st.update({"generation": generation, "population": population, "trial": trial})
        _save_state(work_dir, st)

    df = pd.DataFrame(rows)
    _save_board(work_dir, df)
    if git:
        git.push_leaderboard(df, trial - 1)
    _say(f"[evolve] stopped at generation {generation} · board {len(df)} rows")
    return df
