"""
test_brief.py — the phone-sized call reads in the desk's order.

Prediction numbers, then flow, then floors and ceilings and where the path is open. Built
from the fusion result object, because the printed call carries neither flow nor walls.

Run:  python tests/test_brief.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus  # noqa: E402
from sinus import EnsembleFusionEngine  # noqa: E402

HZ = ("5m", "10m", "20m", "40m", "60m", "eod")


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def synthetic(champion=True, walls=True):
    r = {}
    for i, h in enumerate(HZ):
        t = 765.0 + 0.25 * (i + 1)
        r[h] = {"target": t, "delta": t - 765.1, "direction": 1, "confidence": 0.6 - 0.02 * i,
                "band_lo": t - 1, "band_hi": t + 1, "weights": {"lgb": .3, "cat": .3, "tft": .2, "physics": .2},
                "experts": {"lgb": t, "cat": t, "tft": t if champion else np.nan, "physics": t},
                "gravity": 0.0, "high_vol_fallback": False, "alpha_learned": 0.5}
    r["_physics"] = {"regime": "negative_gamma", "zero_gamma_level": 764.5, "max_pain": 765.0,
                     "pin_strike": 766.0, "pin_strength": 0.42, "snap_applied": True, "snap_radius": 2.0,
                     "wall_below": 764.0 if walls else np.nan, "wall_below_gex": 1.2e9,
                     "wall_above": 767.0 if walls else np.nan, "wall_above_gex": 9.0e8}
    r["_meta"] = {"spot": 765.1, "minutes_to_close": 117.0, "implied_move": 2.1, "net_prem_z": 1.3,
                  "timestamp": "2026-09-08 14:03:00-04:00"}
    if champion:
        r["_meta"]["champion"] = {"trial": 0, "test_score": 0.6368, "node": "laptop"}
    return r


def main():
    ok = True
    sinus.HORIZONS = HZ

    print("1. full live call")
    out = EnsembleFusionEngine.format_brief(synthetic())
    lines = out.splitlines()
    ok &= check("header names spot, time, trial and experts",
                lines[0].startswith("SPY 765.10") and "14:03 ET" in lines[0] and "trial 0" in lines[0] and "tft" in lines[0], lines[0])
    ok &= check("six horizon lines then EOD, in order",
                [l.split()[0] for l in lines[1:8]] == ["5m", "10m", "20m", "40m", "60m", "eod", "EOD"], str([l.split()[0] for l in lines[1:8]]))
    ok &= check("EOD line carries the pin", "pin 766.00" in lines[7], lines[7])
    i_flow = next(i for i, l in enumerate(lines) if l.startswith("flow:"))
    i_floor = next(i for i, l in enumerate(lines) if l.startswith("floor"))
    i_path = next(i for i, l in enumerate(lines) if l.startswith("path open"))
    ok &= check("order: predictions < flow < floors/ceilings < path", 7 < i_flow < i_floor < i_path, f"{i_flow} {i_floor} {i_path}")
    ok &= check("flow has net prem z, implied move, regime", "z +1.30" in lines[i_flow] and "±2.10" in lines[i_flow] and "negative gamma" in lines[i_flow], lines[i_flow])
    ok &= check("floor/ceiling with gex mass", "floor 764.00 (gex 1.2B)" in lines[i_floor] and "ceiling 767.00 (gex 900M)" in lines[i_floor], lines[i_floor])
    ok &= check("path open distances are right", "1.10 down" in lines[i_path] and "1.90 up" in lines[i_path], lines[i_path])
    ok &= check("fits a phone", len(lines) <= 11 and max(len(l) for l in lines) <= 110, f"{len(lines)} lines, widest {max(len(l) for l in lines)}")

    print("2. physics-only call (champion refused or absent)")
    out2 = EnsembleFusionEngine.format_brief(synthetic(champion=False))
    ok &= check("header says physics-only", "physics-only" in out2.splitlines()[0], out2.splitlines()[0])

    print("3. no walls resolved")
    out3 = EnsembleFusionEngine.format_brief(synthetic(walls=False))
    ok &= check("says so instead of printing nan", "no walls resolved" in out3 and "path open" not in out3)

    print("4. not served (weekend / holiday)")
    out4 = EnsembleFusionEngine.format_brief({"_meta": {"served": False, "reason": "no champion yet", "ladder_error": "no contracts"}})
    ok &= check("one line, reason and error", out4.startswith("no call — no champion yet") and "no contracts" in out4 and "\n" not in out4, out4)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
