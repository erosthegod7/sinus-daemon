"""
test_inbox_path.py — the inbox must hand serve() the directory the champion was pulled into.

GitStore.pull_champion() writes <work_dir>/champion/; sinus_daemon.serve(work_dir) reads
<work_dir>/champion/champion.json. process_once() used to pass VOL/champion instead of the
store's work_dir, so the champion was pulled on every run and never found — every Railway
call was physics-only. No network: the store and the model run are stubbed.

Run:  python tests/test_inbox_path.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinus_inbox  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


class _Store:
    """Just enough GitStore: a clone dir with an inbox, and a pull that writes a champion
    where the real one does. Everything else (sync, commit, push) is a no-op."""

    def __init__(self, root):
        self.work_dir = os.path.join(root, "inbox_work")
        self.clone_dir = os.path.join(self.work_dir, "_gitstore")
        os.makedirs(os.path.join(self.clone_dir, sinus_inbox.INBOX), exist_ok=True)
        self.enabled = True

    def pull_champion(self):
        d = os.path.join(self.work_dir, "champion")
        os.makedirs(d, exist_ok=True)
        json.dump({"trial": 7, "feature_set": "stub"}, open(os.path.join(d, "champion.json"), "w"))
        return {"trial": 7}

    def __getattr__(self, name):
        return lambda *a, **k: True


def main():
    ok = True
    root = tempfile.mkdtemp(prefix="sinus_inbox_")
    handed = []
    orig_run = sinus_inbox._run_model
    try:
        gs = _Store(root)
        with open(os.path.join(gs.clone_dir, sinus_inbox.INBOX, "SINUS_2026-09-05_1000.md"), "w") as fh:
            fh.write("extraction")
        sinus_inbox._run_model = lambda path, work: handed.append(work) or "stub output"

        res = sinus_inbox.process_once(gs=gs)

        ok &= check("one file was processed", res is not None and res[0] == "SINUS_2026-09-05_1000.md", str(res))
        ok &= check("serve() was handed a work_dir", len(handed) == 1, str(handed))
        champ = os.path.join(handed[0], "champion", "champion.json") if handed else ""
        ok &= check("that work_dir contains the pulled champion", bool(handed) and os.path.exists(champ),
                    handed[0] if handed else "")
        ok &= check("inbox is empty afterwards",
                    not [f for f in os.listdir(os.path.join(gs.clone_dir, sinus_inbox.INBOX)) if f != ".gitkeep"])

        print("2. /predict: run the champion with nothing in the inbox")
        text = sinus_inbox.predict_once(gs=gs)
        ok &= check("returns the run output", text == "stub output", repr(text)[:40])
        ok &= check("handed serve() the store's work_dir (champion present)",
                    len(handed) == 2 and os.path.exists(os.path.join(handed[1], "champion", "champion.json")))
        outs = [f for f in os.listdir(os.path.join(gs.clone_dir, sinus_inbox.OUT)) if f.startswith("predict_")]
        ok &= check("archived under out/predict_<stamp>.txt", len(outs) == 1, str(outs))
        ok &= check("/latest would show it", sinus_inbox.STATE.get("last_output") == "stub output")
    finally:
        sinus_inbox._run_model = orig_run
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
