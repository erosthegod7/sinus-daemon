"""
test_promotion_gate.py — how the search seeds the bar it has to beat.

The failure this guards against is silent: the search keeps running, keeps logging
"ok" trials, and simply never promotes again. Run:  python tests/test_promotion_gate.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinus_daemon import _seed_bar  # noqa: E402
from sinus_search import SCORING_VERSION  # noqa: E402

INF = float("inf")


def board(scores, status="ok", version=SCORING_VERSION, with_version_col=True):
    rows = [{"trial": i, "status": status, "score": s} for i, s in enumerate(scores)]
    df = pd.DataFrame(rows)
    if with_version_col and len(df):
        df["scoring_version"] = version
    return df


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main():
    ok = True

    print("1. the regression: board with scores, no champion on disk")
    # This is the 2026-09-04 stall. Trial 18 scored 0.5004, its weights were lost, and
    # every later trial was measured against a number nothing backed.
    bar, why = _seed_bar(board([0.5004, 0.5058, 0.5071]), champ=None)
    ok &= check("bar is +inf so the next good trial can promote", bar == INF, f"{bar}  [{why}]")

    print("2. normal resume: champion exists and the board is comparable")
    bar, why = _seed_bar(board([0.5004, 0.5058]), champ=0.51)
    ok &= check("bar seeds from the board's best", np.isclose(bar, 0.5004), f"{bar}")

    print("3. scoring-version cutover")
    bar, _ = _seed_bar(board([0.30, 0.31], version="eod_full_session"), champ=0.51)
    ok &= check("old-scale rows are ignored", bar == INF, f"{bar}")
    bar, _ = _seed_bar(board([0.30], with_version_col=False), champ=0.51)
    ok &= check("board predating versioning is ignored", bar == INF, f"{bar}")
    mixed = pd.concat([board([0.30], version="eod_full_session"), board([0.52])], ignore_index=True)
    bar, _ = _seed_bar(mixed, champ=0.51)
    ok &= check("mixed board uses only current-scale rows", np.isclose(bar, 0.52), f"{bar}")

    print("4. degenerate boards do not produce a NaN bar")
    # Every comparison against NaN is False, which disables promotion forever.
    for label, b in (("empty", pd.DataFrame()),
                     ("all failed", board([INF, INF], status="failed: TypeError")),
                     ("non-numeric scores", board(["", "n/a"])),
                     ("all NaN", board([np.nan, np.nan])),
                     ("infinite scores", board([INF]))):
        bar, _ = _seed_bar(b, champ=0.51)
        ok &= check(f"{label} board -> +inf, not NaN", bar == INF and not np.isnan(bar), f"{bar}")

    print("5. a bar is never NaN for any input combination")
    combos = [(board([0.5]), None), (board([0.5]), 0.4), (pd.DataFrame(), None), (pd.DataFrame(), 0.4)]
    ok &= check("no combination yields NaN",
                all(not np.isnan(_seed_bar(b, c)[0]) for b, c in combos))

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
