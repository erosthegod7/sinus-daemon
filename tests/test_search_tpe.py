"""
test_search_tpe.py — the Bayesian search and the recency weights.

  * install_optuna_search warm-starts from the board (ok rows as completed trials, pruned
    rows as pruned), replaces the daemon's sampler, and records every finished trial back.
  * _time_decay_weights halves per half-life and is uniform when off.

Run:  python tests/test_search_tpe.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus_daemon as sd  # noqa: E402
import sinus_train  # noqa: E402
from sinus import _time_decay_weights  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def board_rows(n_ok=3, n_pruned=1):
    rows = []
    for i in range(n_ok + n_pruned):
        rows.append({"trial": i, "status": "ok" if i < n_ok else "pruned", "score": 0.70 - 0.01 * i if i < n_ok else float("inf"),
                     "screen_score": 0.75 + 0.01 * i, "learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 60,
                     "feature_fraction": 0.8, "lambda_l2": 5.0, "robust_delta": 1.0, "objective": "huber", "d_model": 32,
                     "n_heads": 4, "dropout": 0.2, "tft_lr": 0.001, "lookback": 60})
    return pd.DataFrame(rows)                  # note: no time_decay_halflife column — an older board


def main():
    ok = True
    print("1. recency weights")
    g = np.repeat(np.arange(5), 3)             # five sessions, three rows each
    ok &= check("off -> uniform", np.allclose(_time_decay_weights(g, 0.0), 1.0))
    w = _time_decay_weights(g, 1.0)
    ok &= check("newest session weighs 1, one half-life older weighs 0.5", np.isclose(w[-1], 1.0) and np.isclose(w[9], 0.5), f"{w[-1]:.2f} {w[9]:.2f}")
    ok &= check("oldest of five at half-life 1 is 1/16", np.isclose(w[0], 0.0625), f"{w[0]:.4f}")
    ok &= check("half-life 2 decays half as fast", np.isclose(_time_decay_weights(g, 2.0)[0], 0.25))

    print("2. TPE search: warm start, sampling, recording")
    root = tempfile.mkdtemp(prefix="sinus_tpe_")
    orig_sample, orig_save = sd.sample_config, sd._save_board
    try:
        board = board_rows()
        board.to_csv(os.path.join(root, "leaderboard.csv"), index=False)
        study = sinus_train.install_optuna_search(root, storage="memory")
        ok &= check("study created", study is not None)
        states = [t.state.name for t in study.trials]
        ok &= check("warm-started 3 complete + 1 pruned from the board", sorted(states) == ["COMPLETE", "COMPLETE", "COMPLETE", "PRUNED"], str(states))
        ok &= check("old rows got the 'off' half-life", all(t.params.get("time_decay_halflife") == 0.0 for t in study.trials))
        ok &= check("sampler replaced", sd.sample_config is not orig_sample)

        p = sd.sample_config(np.random.default_rng(0))
        ok &= check("proposal has every parameter", set(p) == set(sinus_train.SEARCH_SPACE), str(sorted(set(sinus_train.SEARCH_SPACE) - set(p))))
        ok &= check("proposal inside the space", 0.6 <= p["feature_fraction"] <= 0.95 and p["objective"] in ("pseudo_huber", "cauchy", "huber")
                    and p["time_decay_halflife"] in (0.0, 60.0, 120.0, 250.0))

        new = pd.concat([board, pd.DataFrame([{**p, "trial": 4, "status": "ok", "score": 0.66, "screen_score": 0.7}])], ignore_index=True)
        sd._save_board(root, new)
        ok &= check("finished trial recorded as complete with its score",
                    study.trials[-1].state.name == "COMPLETE" and np.isclose(study.trials[-1].value, 0.66), f"{study.trials[-1].state.name} {study.trials[-1].value}")
        ok &= check("board file still written by the original saver", os.path.exists(os.path.join(root, "leaderboard.csv")))

        p2 = sd.sample_config(np.random.default_rng(1))
        pruned = pd.concat([new, pd.DataFrame([{**p2, "trial": 5, "status": "pruned", "score": float("inf"), "screen_score": 0.8}])], ignore_index=True)
        sd._save_board(root, pruned)
        ok &= check("pruned trial recorded as pruned with its screen score",
                    study.trials[-1].state.name == "PRUNED" and study.trials[-1].intermediate_values.get(0) == 0.8, study.trials[-1].state.name)
        ok &= check("study now holds 6 trials", len(study.trials) == 6, str(len(study.trials)))
    finally:
        sd.sample_config, sd._save_board = orig_sample, orig_save
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
