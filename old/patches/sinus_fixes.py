"""
sinus_fixes.py — apply every outstanding fix to the SINUS daemon.

Run from C:\\sinus-daemon with the venv active:

    python sinus_fixes.py            # apply
    python sinus_fixes.py --check    # report only, change nothing

Backs each file up to <name>.prefix.bak before its first edit. Idempotent: a
second run reports "already applied" instead of double-patching. If any anchor
does not match, that file is left completely untouched and the reason is printed.

WHAT IT FIXES
  1. Ctrl-C is a no-op during training.  The SIGINT handler only set a flag that
     is read at the top of the trial loop, and it replaced the default handler,
     so KeyboardInterrupt was never raised. First Ctrl-C now sets the flag AND
     restores the default handler, so a second press kills for real. The TFT
     epoch loop and both LightGBM train calls now check the flag too, so a stop
     lands in seconds instead of after two full model fits.

  2. One margin gating two different decisions.  min_improvement=0.002 gated
     both "is this worth a test refit" and "does this test win count". Scores
     cluster inside 0.005, so the second use rejected real winners — trial 13
     beat the champion on test by 0.00118 and was thrown away. Now
     screen_margin (0.002) and promote_margin (0.0005) are separate.

  3. best_val came from the whole board, not the champion.  A candidate that
     failed promotion still lowered the bar it had just failed to clear, so its
     config could never be screened again. Crossover re-found trial 13's genes
     twice more (trials 23, 29) and both were rejected without evaluation.
     best_val now seeds from champion.json's val_score and advances only on
     promotion.

  4. The minimum-session guard ran before the chain substitution.  main()
     checked session count on the Polygon/CSV frame, then SINUS_SPOT_FROM_CHAIN
     replaced spot_df with the chain's parity spot — 5 sessions on 2026-09-02,
     under the floor of 7, with no complaint. Now re-checked after substitution.

  5. Crash-looping forever counted as "running".  A failure repeated every 60
     seconds all night with no trials completed. Now: exponential backoff, and
     after 5 identical consecutive failures the daemon exits so you find out.
"""

from __future__ import annotations

import os
import shutil
import sys

CHECK = "--check" in sys.argv

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class Skip(Exception):
    """An anchor did not match — abandon this file without writing."""


def read(path):
    """Return (normalised_text, original_newline). Files here are mixed CRLF/LF."""
    raw = open(path, encoding="utf-8", newline="").read()
    return raw.replace("\r\n", "\n"), ("\r\n" if "\r\n" in raw else "\n")


def write(path, text, nl):
    open(path, "w", encoding="utf-8", newline="").write(
        text.replace("\n", nl) if nl == "\r\n" else text)


def sub(text, old, new, label):
    n = text.count(old)
    if n == 0:
        raise Skip(f"anchor not found: {label}")
    if n > 1:
        raise Skip(f"anchor matches {n} times, expected 1: {label}")
    print(f"      + {label}")
    return text.replace(old, new)


def apply_to(path, marker, fn):
    """Run fn(text)->text on path unless marker is already present."""
    if not os.path.exists(path):
        print(f"  {path}: NOT FOUND — skipped")
        return False
    text, nl = read(path)
    if marker in text:
        print(f"  {path}: already applied")
        return False
    print(f"  {path}:")
    try:
        out = fn(text)
    except Skip as e:
        print(f"      ! {e}")
        print(f"      ! {path} left unchanged")
        return False
    if CHECK:
        print("      (check mode — not written)")
        return True
    bak = path + ".prefix.bak"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    write(path, out, nl)
    return True


# --------------------------------------------------------------------------- #
# 1. sinus.py — cooperative stop inside the training loops
# --------------------------------------------------------------------------- #
STOP_BLOCK = '''

# --------------------------------------------------------------------------- #
# Cooperative stop (sinus_fixes)
# --------------------------------------------------------------------------- #
# Ctrl-C used to be swallowed: the daemon's SIGINT handler set a flag that was
# only read between trials, so a press during a TFT fit or a 4000-round LightGBM
# train did nothing for many minutes. The search loop sets STOP["flag"]; the
# training loops below check it and bail out promptly.
STOP = {"flag": False}


class SinusStopped(Exception):
    """Raised inside a training loop when a stop has been requested. The search
    loop catches this specifically and exits cleanly rather than logging a
    failed trial."""


def _lgb_stop():
    """LightGBM callback that aborts boosting when a stop is requested. There is
    no graceful early exit in lgb.train, so raising is the only way out."""
    def _cb(env):
        if STOP["flag"]:
            raise SinusStopped("stop requested during LightGBM training")
    _cb.order = 0
    _cb.before_iteration = False
    return _cb
'''


def patch_sinus(t):
    t = sub(t,
            'try:\n    import websockets\n    _HAS_WS = True\n'
            'except Exception:                                   # pragma: no cover\n'
            '    websockets, _HAS_WS = None, False\n',
            'try:\n    import websockets\n    _HAS_WS = True\n'
            'except Exception:                                   # pragma: no cover\n'
            '    websockets, _HAS_WS = None, False\n' + STOP_BLOCK,
            "STOP flag + SinusStopped + LightGBM stop callback")

    t = sub(t,
            '        best, bad, best_state = float("inf"), 0, None\n'
            '        for epoch in range(self.tc.max_epochs):\n'
            '            self.model.train()\n',
            '        best, bad, best_state = float("inf"), 0, None\n'
            '        for epoch in range(self.tc.max_epochs):\n'
            '            if STOP["flag"]:            # Ctrl-C lands between epochs, not after the fit\n'
            '                break\n'
            '            self.model.train()\n',
            "TFT epoch loop checks the stop flag")

    t = sub(t,
            'callbacks=[lgb.early_stopping(self.tc.early_stopping_rounds, first_metric_only=True, verbose=False),\n'
            '                                               lgb.record_evaluation(ev)])\n',
            'callbacks=[lgb.early_stopping(self.tc.early_stopping_rounds, first_metric_only=True, verbose=False),\n'
            '                                               lgb.record_evaluation(ev), _lgb_stop()])\n',
            "LightGBM CV folds honour the stop flag")

    t = sub(t,
            '                self.lgb_models[h] = lgb.train(self._lgb_params(), '
            'lgb.Dataset(Xh, label=yh, feature_name=self.feature_names), num_boost_round=n_final)\n',
            '                self.lgb_models[h] = lgb.train(self._lgb_params(), '
            'lgb.Dataset(Xh, label=yh, feature_name=self.feature_names), num_boost_round=n_final,\n'
            '                                              callbacks=[_lgb_stop()])\n',
            "LightGBM final full-data fit honours the stop flag")
    return t


# --------------------------------------------------------------------------- #
# 2. sinus_evolve.py — the gate, best_val, and a Ctrl-C that works
# --------------------------------------------------------------------------- #
def patch_evolve(t):
    t = sub(t,
            'from sinus import (FeaturePipeline, PipelineConfig, TrainingBundle, '
            'CoreModelingEngine, EngineConfig, HORIZONS)\n',
            'from sinus import (FeaturePipeline, PipelineConfig, TrainingBundle, '
            'CoreModelingEngine, EngineConfig, HORIZONS, STOP, SinusStopped)\n',
            "import the stop flag")

    t = sub(t,
            '                   tft_epochs: int = 12, final_epochs: int = 40, min_improvement: float = 0.002,\n',
            '                   tft_epochs: int = 12, final_epochs: int = 40,\n'
            '                   screen_margin: float = float(os.environ.get("SINUS_SCREEN_MARGIN", "0.002")),\n'
            '                   promote_margin: float = float(os.environ.get("SINUS_PROMOTE_MARGIN", "0.0005")),\n'
            '                   min_improvement: Optional[float] = None,\n',
            "split screen_margin from promote_margin")

    # --- best_val: seed from the champion, not from the whole board -----------
    t = sub(t,
            '    champ = _champion_score(work_dir)\n'
            '    if champ is not None and not np.isfinite(champ):\n'
            '        champ = None\n'
            '    best_val = float("inf")\n'
            '    if rows:\n'
            '        ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("score", np.inf))]\n'
            '        if ok:\n'
            '            best_val = float(min(r["score"] for r in ok))\n',

            '    if min_improvement is not None:              # old callers keep working\n'
            '        screen_margin = promote_margin = min_improvement\n'
            '    champ = _champion_score(work_dir)\n'
            '    if champ is not None and not np.isfinite(champ):\n'
            '        champ = None\n'
            '    # The screen bar is the CHAMPION\'s validation score. It used to be the minimum\n'
            '    # over every ok trial, which meant a candidate that failed promotion still raised\n'
            '    # the bar it had just failed to clear — locking its own genes out of ever being\n'
            '    # evaluated again, however many times breeding rediscovered them.\n'
            '    best_val = float("inf")\n'
            '    try:\n'
            '        _cv = float(json.load(open(os.path.join(work_dir, CHAMPION, "champion.json")))["val_score"])\n'
            '        if np.isfinite(_cv):\n'
            '            best_val = _cv\n'
            '    except Exception:\n'
            '        pass\n'
            '    _say(f"[evolve] screen bar {best_val:.5f} (-{screen_margin}) · promote bar "\n'
            '         f"{\'none\' if champ is None else format(champ, \'.5f\')} (-{promote_margin})")\n',
            "screen bar comes from the champion, not the board minimum")

    # --- SIGINT: flag first, then hand back to Python -------------------------
    t = sub(t,
            '    stop = {"flag": False}\n'
            '    try:\n'
            '        _signal.signal(_signal.SIGINT, lambda *_: stop.__setitem__("flag", True))\n'
            '    except (ValueError, AttributeError):\n'
            '        pass\n',

            '    stop = STOP                                   # shared with the training loops in sinus.py\n'
            '    stop["flag"] = False\n'
            '\n'
            '    def _on_sigint(*_):\n'
            '        """First press asks for a clean stop and restores the default handler, so a\n'
            '        second press raises KeyboardInterrupt and kills the process outright. The old\n'
            '        handler swallowed every press forever."""\n'
            '        stop["flag"] = True\n'
            '        _say("[evolve] stop requested — finishing the current step. Press Ctrl-C again to force.")\n'
            '        try:\n'
            '            _signal.signal(_signal.SIGINT, _signal.default_int_handler)\n'
            '        except (ValueError, AttributeError, TypeError):\n'
            '            pass\n'
            '\n'
            '    try:\n'
            '        _signal.signal(_signal.SIGINT, _on_sigint)\n'
            '    except (ValueError, AttributeError):\n'
            '        pass\n',
            "Ctrl-C sets the flag, then restores the default handler")

    # --- the gate -------------------------------------------------------------
    t = sub(t,
            '                if np.isfinite(vs["score"]) and vs["score"] < best_val - min_improvement:\n'
            '                    best_val = vs["score"]\n'
            '                    cand = os.path.join(work_dir, "_candidate")\n',

            '                if (np.isfinite(vs["score"]) and vs["score"] < best_val - screen_margin\n'
            '                        and not stop["flag"]):     # never start a long refit on the way out\n'
            '                    # best_val is NOT advanced here — only a promotion moves the bar.\n'
            '                    cand = os.path.join(work_dir, "_candidate")\n',
            "screen uses screen_margin; no bar move; no refit while stopping")

    t = sub(t,
            '                    if champ is None or ts_["score"] < champ - min_improvement:\n'
            '                        _promote(work_dir, cand, p, vs, ts_, trial, pipeline=pipeline, n_sessions=int(n_sess))\n'
            '                        champ = ts_["score"]\n',

            '                    if champ is None or ts_["score"] < champ - promote_margin:\n'
            '                        _promote(work_dir, cand, p, vs, ts_, trial, pipeline=pipeline, n_sessions=int(n_sess))\n'
            '                        champ = ts_["score"]\n'
            '                        best_val = vs["score"]     # the bar moves only when the champion does\n',
            "promotion uses promote_margin and advances the bar")

    t = sub(t,
            '                            git.push_champion(os.path.join(work_dir, CHAMPION),\n'
            '                                              json.load(open(os.path.join(work_dir, CHAMPION, "champion.json"))),\n'
            '                                              min_improvement)\n',
            '                            git.push_champion(os.path.join(work_dir, CHAMPION),\n'
            '                                              json.load(open(os.path.join(work_dir, CHAMPION, "champion.json"))),\n'
            '                                              promote_margin)\n',
            "cross-node push check uses the same promote margin")

    t = sub(t,
            '                             f"({ts_[\'score\']:.4f} vs champion {champ:.4f}) — not promoted")\n',
            '                             f"({ts_[\'score\']:.4f} vs champion {champ:.4f}, short of the "\n'
            '                             f"{promote_margin} margin by {promote_margin - (champ - ts_[\'score\']):.5f})"\n'
            '                             f" — not promoted")\n',
            "rejection message reports the actual shortfall")

    # --- a stop raised from inside training is not a failed trial -------------
    t = sub(t,
            '            except Exception as e:\n'
            '                rec.update({"status": f"failed: {type(e).__name__}", "score": float("inf"),\n',

            '            except SinusStopped:\n'
            '                _say("[evolve] stopped mid-training — this trial is discarded, the board is intact")\n'
            '                stop["flag"] = True\n'
            '                break\n'
            '            except Exception as e:\n'
            '                rec.update({"status": f"failed: {type(e).__name__}", "score": float("inf"),\n',
            "a requested stop is not logged as a failed trial")
    return t


# --------------------------------------------------------------------------- #
# 3. railway_daemon.py — session floor after substitution, and crash backoff
# --------------------------------------------------------------------------- #
def patch_railway(t):
    t = sub(t,
            "                    if os.environ.get('SINUS_SPOT_FROM_CHAIN', '1') == '1' and len(_spot2) > 0:\n"
            "                        spot_df = _spot2      # parity spot from the chain: no stocks plan, no 5-calls/min crawl\n",

            "                    if os.environ.get('SINUS_SPOT_FROM_CHAIN', '1') == '1' and len(_spot2) > 0:\n"
            "                        spot_df = _spot2      # parity spot from the chain: no stocks plan, no 5-calls/min crawl\n"
            "                        # The MIN_SESSIONS check above ran on the Polygon/CSV frame. Substituting the\n"
            "                        # chain's parity spot can drop the count far below the floor without tripping\n"
            "                        # it — on 2026-09-02 that started a search on 5 sessions against a floor of 7.\n"
            "                        _n = spot_df['ts'].dt.normalize().nunique()\n"
            "                        if _n < MIN_SESSIONS:\n"
            "                            _log(f\"FATAL chain parity spot covers {_n} sessions, floor is {MIN_SESSIONS}. \"\n"
            "                                 f\"Back-fill more days (polygon_chain_history.py) or set \"\n"
            "                                 f\"SINUS_SPOT_FROM_CHAIN=0 to keep the longer price-only series.\")\n"
            "                            return 1\n",
            "re-check the session floor after chain substitution")

    t = sub(t,
            '        except Exception as e:\n'
            '            _log(f"daemon crashed: {e}")\n'
            '            traceback.print_exc()\n'
            '            _log("restarting in 60s; the leaderboard is intact so at most one trial is lost")\n'
            '            time.sleep(60)\n',

            '        except Exception as e:\n'
            '            _log(f"daemon crashed: {e}")\n'
            '            traceback.print_exc()\n'
            '            # A bug that fails on every boot used to retry every 60s all night and still\n'
            '            # look like a running daemon. Back off, and give up on a fault that is clearly\n'
            '            # not transient so it is visible in the morning instead of buried in a log.\n'
            '            sig = f"{type(e).__name__}: {e}"\n'
            '            _repeat = _repeat + 1 if sig == _last_err else 1\n'
            '            _last_err = sig\n'
            '            if _repeat >= 5:\n'
            '                _log(f"FATAL same failure {_repeat}x in a row — this is not transient. Exiting so "\n'
            '                     f"you see it. The leaderboard and champion on disk are intact.")\n'
            '                return 1\n'
            '            wait = min(60 * 2 ** (_repeat - 1), 900)\n'
            '            _log(f"restarting in {wait}s (failure {_repeat}/5); the leaderboard is intact "\n'
            '                 f"so at most one trial is lost")\n'
            '            time.sleep(wait)\n',
            "exponential backoff and abort on a repeating failure")

    t = sub(t,
            '    while True:                                        # outer loop: survive an unexpected crash\n',
            '    _last_err, _repeat = None, 0\n'
            '    while True:                                        # outer loop: survive an unexpected crash\n',
            "track the repeating-failure counter")
    return t


# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print(f"sinus_fixes — {here}")
    print("check mode: nothing will be written\n" if CHECK else "")

    n = 0
    n += apply_to("sinus.py", "SinusStopped", patch_sinus)
    n += apply_to("sinus_evolve.py", "promote_margin", patch_evolve)
    n += apply_to("railway_daemon.py", "_repeat >= 5", patch_railway)

    print()
    if CHECK:
        print(f"{n} file(s) would change.")
        return
    print(f"{n} file(s) changed. Backups: *.prefix.bak")
    if n:
        for d in ("__pycache__",):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                print("cleared __pycache__")
        print("\nNow run:  python preflight.py")


if __name__ == "__main__":
    main()
