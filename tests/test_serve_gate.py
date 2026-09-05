"""
test_serve_gate.py — serve() must never quietly run the wrong champion.

Three cases, no network (the ladder, the physics layer and predict_live are stubbed):
  * no champion            -> physics-only, says "no champion yet", ML path not entered
  * champion from another  -> physics-only, says it refused, ML path not entered
    feature set
  * champion from this     -> predict_live IS called with the champion directory
    feature set

Run:  python tests/test_serve_gate.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus  # noqa: E402
import sinus_daemon as sd  # noqa: E402
import sinus_train  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


class _Fusion:
    def __init__(self, *a, **k):
        pass

    def predict_final(self, ml, st, row=-1):
        return {"_stub": True}

    def format_call(self, res):
        return "(stub call)"


def main():
    ok = True
    calls = []
    sd.fetch_snapshot_ladder = lambda *a, **k: (None, 765.0, {})
    sd.market_state_from_polygon = lambda *a, **k: None
    sd.EnsembleFusionEngine = _Fusion
    sinus.predict_live = lambda symbol, champion_dir, log_csv=None, **k: calls.append(champion_dir) or {"_live": True}

    root = tempfile.mkdtemp(prefix="sinus_serve_")
    try:
        champ = os.path.join(root, sd.CHAMPION)

        def run():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = sd.serve("SPY", work_dir=root, spot=765.0)
            return res, buf.getvalue()

        print("1. no champion on disk")
        res, out = run()
        ok &= check("physics-only, reason stated", "physics-only (no champion yet)" in out, out.strip().splitlines()[0][:70])
        ok &= check("predict_live not entered", not calls)

        print("2. champion from a different feature set")
        os.makedirs(champ, exist_ok=True)
        json.dump({"trial": 18, "feature_set": "v2"}, open(os.path.join(champ, "champion.json"), "w"))
        res, out = run()
        ok &= check("refused, reason names both sets", "refusing" in out and "'v2'" in out and repr(sinus_train.FEATURE_SET) in out,
                    [l for l in out.splitlines() if "serve" in l][0][:90])
        ok &= check("predict_live not entered", not calls)

        print("3. champion from this feature set")
        json.dump({"trial": 7, "feature_set": sinus_train.FEATURE_SET}, open(os.path.join(champ, "champion.json"), "w"))
        res, out = run()
        ok &= check("predict_live entered with the champion dir", calls == [champ], str(calls))
        ok &= check("live result returned", res.get("_live") is True)

        print("4. market closed: no options ladder exists")
        os.remove(os.path.join(champ, "champion.json"))

        def _no_ladder(*a, **k):
            raise RuntimeError("no contracts for SPY 2026-09-05 - holiday, or no expiry that date")
        sd.fetch_snapshot_ladder = _no_ladder
        try:
            res, out = run()
            ok &= check("returns instead of raising", isinstance(res, dict) and res.get("_meta", {}).get("served") is False)
            ok &= check("says so in one line", "no options ladder" in out and "Traceback" not in out,
                        [l for l in out.splitlines() if "ladder" in l][0][:80])
        except Exception as e:
            ok &= check("returns instead of raising", False, f"raised {type(e).__name__}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
