"""
test_gitstore_guard.py — push_champion must never publish a git clone.

GitStore's remote URL embeds the PAT, so `git clone` writes it into .git/config.
push_champion copytree's a directory into a PUBLIC repo. Hand it the parent of the
clone instead of the champion directory and the token goes public. That is exactly
what the retired push_champion_now.py did.

Run:  python tests/test_gitstore_guard.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus_gitstore  # noqa: E402
from sinus_gitstore import GitStore  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main():
    ok = True
    root = tempfile.mkdtemp(prefix="sinus_guard_")
    try:
        # A GitStore that never touches the network: stub the clone/commit machinery.
        gs = GitStore.__new__(GitStore)
        gs.enabled, gs.node = True, "test"
        gs.clone_dir = os.path.join(root, "_clone")
        os.makedirs(gs.clone_dir, exist_ok=True)
        gs.remote_champion_score = lambda: None
        gs._commit_push = lambda msg: True

        print("1. the trap: champion dir that contains a clone with the token")
        work = os.path.join(root, "champion_v2")
        os.makedirs(os.path.join(work, "_gitstore", ".git"), exist_ok=True)
        with open(os.path.join(work, "_gitstore", ".git", "config"), "w") as f:
            f.write("url = https://x-access-token:github_pat_SECRET@github.com/o/r.git\n")
        with open(os.path.join(work, "weights.txt"), "w") as f:
            f.write("model")

        pushed = gs.push_champion(work, {"trial": 1, "test_score": 0.5})
        ok &= check("push is refused", pushed is False, str(pushed))

        staged = os.path.join(gs.clone_dir, "champion")
        ok &= check("nothing left staged", not os.path.exists(staged))
        found = []
        for dirpath, _, files in os.walk(gs.clone_dir):
            for fn in files:
                p = os.path.join(dirpath, fn)
                try:
                    if "github_pat_" in open(p, errors="replace").read():
                        found.append(p)
                except OSError:
                    pass
        ok &= check("token nowhere in the clone dir", not found, str(found[:1]))

        print("2. a clean champion directory still pushes")
        clean = os.path.join(root, "champion")
        os.makedirs(os.path.join(clean, "trees"), exist_ok=True)
        for rel in ("champion.json", os.path.join("trees", "lgb_5m.txt")):
            with open(os.path.join(clean, rel), "w") as f:
                f.write("{}")
        pushed = gs.push_champion(clean, {"trial": 2, "test_score": 0.4})
        ok &= check("push succeeds", pushed is True, str(pushed))
        ok &= check("weights copied", os.path.exists(os.path.join(staged, "trees", "lgb_5m.txt")))
        ok &= check("champion.json written", os.path.exists(os.path.join(staged, "champion.json")))
        ok &= check("lineage written", os.path.isdir(os.path.join(gs.clone_dir, "lineage")))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
