"""
run_all.py — every check in one command.

    python tests/run_all.py            unit tests + smoke test on real bars
    python tests/run_all.py --fast     unit tests only (no data load)

Exit code is non-zero if anything failed, so it can gate a run.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAST = "--fast" in sys.argv

SUITES = [("test_scoring.py", []), ("test_promotion_gate.py", []), ("test_gitstore_guard.py", []),
          ("test_serve_gate.py", []), ("test_inbox_path.py", [])]
if not FAST:
    SUITES.append(("test_pipeline_smoke.py", ["40"]))
    SUITES.append(("test_causality.py", []))


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.setdefault("SINUS_VOLUME", r"C:\sinus\data")
    failed = []
    for name, args in SUITES:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        r = subprocess.run([sys.executable, os.path.join(HERE, name), *args], env=env)
        if r.returncode:
            failed.append(name)
    print(f"\n{'=' * 70}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"ALL SUITES PASS ({len(SUITES)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
