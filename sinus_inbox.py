"""
sinus_inbox.py
==============
On-demand runner: GitHub `inbox/` -> SINUS -> raw output -> `old/<date>/`.

This is a SEPARATE Railway service from the search daemon. It shares the repo and the
GitStore plumbing but never imports `search_forever` and never trains anything. That
separation is the whole point: the search wants 24 CPUs for weeks, this wants almost
nothing and must be safe to leave running.

Flow, per cycle
    1. git fetch the data repo (SINUS_GIT_REPO).
    2. Look in `inbox/`. Empty -> sleep and do nothing. This is the common case and it
       costs one fetch, so idle CPU is effectively zero.
    3. Newest file wins. If more than one is sitting there, the older ones are still
       archived, just not run — inbox is meant to hold exactly one live file.
    4. Run the champion against it. stdout is captured VERBATIM and written to
       `out/<stem>.txt`. Nothing is added, reworded, or summarised — if the model
       printed it, that is what lands in the file.
    5. Move the input to `old/<YYYY-MM-DD>/`, commit, push. Inbox is empty again.

Why capture stdout instead of formatting the return value: `serve()` already prints the
call through `format_call`. Re-rendering the returned dict here would mean two places
that decide what a prediction looks like, and they would drift.

Environment
    SINUS_GIT_REPO      erosthegod7/sinus-champion    (the DATA repo, not the code repo)
    SINUS_GIT_TOKEN     fine-grained token, Contents: read/write on that repo
    SINUS_NODE          short machine name, default 'inbox'
    SINUS_VOLUME        default /data
    SINUS_SYMBOL        default SPY
    SINUS_INBOX_POLL    seconds between fetches, default 60. 0 = never poll, HTTP only.
    PORT                default 8080

Routes
    /           status: inbox contents, last run, champion in use
    /run        process the inbox right now, return the raw output as text/plain
    /latest     the most recent raw output
    /inbox      what is currently waiting
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import time
import traceback
from typing import List, Optional, Tuple

import pandas as pd

VOL = os.environ.get("SINUS_VOLUME", "/data")
SYMBOL = os.environ.get("SINUS_SYMBOL", "SPY")
POLL = int(os.environ.get("SINUS_INBOX_POLL", "60"))
INBOX = "inbox"
OLD = "old"
OUT = "out"

# Files git needs but that are not data. Anything matching these is never treated as input.
IGNORE = {".gitkeep", ".gitignore", "README.md", "readme.md"}


def _log(msg: str) -> None:
    print(f"[inbox] {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# State shared with the HTTP thread. A dict rather than globals so the handler
# reads a consistent snapshot rather than three separately-updating values.
# --------------------------------------------------------------------------- #
STATE = {"last_run": None, "last_file": None, "last_output": None, "last_error": None}


def _store():
    """GitStore against the DATA repo. Imported lazily so this module can be inspected
    (and unit-tested) on a machine with no token set."""
    from sinus_gitstore import GitStore

    gs = GitStore(work_dir=os.path.join(VOL, "inbox_work"))
    if not gs.enabled:
        raise RuntimeError(
            "gitstore disabled — set SINUS_GIT_REPO and SINUS_GIT_TOKEN on this service. "
            "Without them there is no inbox to read."
        )
    return gs


def _list_inbox(gs) -> List[str]:
    d = os.path.join(gs.clone_dir, INBOX)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f not in IGNORE and os.path.isfile(os.path.join(d, f)))


def _run_model(data_path: str, work_dir: str) -> str:
    """Run the champion and return exactly what it printed.

    Falls back to the physics path on its own if no champion has been promoted — that
    behaviour lives in serve() and is stated in its output, so it is not second-guessed
    here. A traceback is returned as text rather than raised: a bad data file should
    archive with its error attached, not wedge the queue behind it forever.
    """
    buf = io.StringIO()
    header = [
        f"# SINUS run  {pd.Timestamp.now(tz='America/New_York')}",
        f"# input      {os.path.basename(data_path)}",
        "",
    ]
    try:
        import sinus

        with contextlib.redirect_stdout(buf):
            sinus.serve(SYMBOL, work_dir=work_dir)
    except Exception:
        buf.write("\n--- run failed ---\n")
        buf.write(traceback.format_exc())
    return "\n".join(header) + buf.getvalue()


def process_once(gs=None) -> Optional[Tuple[str, str]]:
    """One full cycle. Returns (filename, raw_output) if something ran, else None."""
    gs = gs or _store()
    gs._sync()

    files = _list_inbox(gs)
    if not files:
        return None

    target = files[-1]                      # timestamped names sort chronologically
    if len(files) > 1:
        _log(f"{len(files)} files in inbox; running {target}, archiving the rest unrun")

    work = os.path.join(VOL, "champion")
    os.makedirs(work, exist_ok=True)
    gs.pull_champion()                      # newest weights the other node promoted

    src = os.path.join(gs.clone_dir, INBOX, target)
    _log(f"running {target}")
    output = _run_model(src, work)

    # raw output
    out_dir = os.path.join(gs.clone_dir, OUT)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(target)[0]
    with open(os.path.join(out_dir, f"{stem}.txt"), "w") as fh:
        fh.write(output)

    # archive every file that was sitting there, so the inbox ends up empty
    day = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    dest = os.path.join(gs.clone_dir, OLD, day)
    os.makedirs(dest, exist_ok=True)
    for f in files:
        shutil.move(os.path.join(gs.clone_dir, INBOX, f), os.path.join(dest, f))

    # .gitkeep so the empty inbox survives as a directory in git
    keep = os.path.join(gs.clone_dir, INBOX, ".gitkeep")
    if not os.path.exists(keep):
        open(keep, "w").close()

    gs._commit_push(f"inbox: ran {target}, archived {len(files)} to {OLD}/{day}")

    STATE.update(last_run=str(pd.Timestamp.now(tz="America/New_York")),
                 last_file=target, last_output=output, last_error=None)
    _log(f"done — {target} archived to {OLD}/{day}")
    return target, output


def start_http_server(port: int) -> None:
    """Status and manual trigger. Daemon thread — if it cannot bind, polling continues."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str, ctype: str = "text/plain") -> None:
            b = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self) -> None:                       # noqa: N802
            path = self.path.split("?")[0].rstrip("/") or "/"
            try:
                if path == "/run":
                    r = process_once()
                    if r is None:
                        return self._send(200, "inbox is empty — nothing to run\n")
                    return self._send(200, r[1])

                if path == "/latest":
                    if not STATE["last_output"]:
                        return self._send(404, "no run yet\n")
                    return self._send(200, STATE["last_output"])

                if path == "/inbox":
                    files = _list_inbox(_store())
                    return self._send(200, ("\n".join(files) or "(empty)") + "\n")

                lines = [f"SINUS inbox worker · {pd.Timestamp.now(tz='UTC')}",
                         f"repo      : {os.environ.get('SINUS_GIT_REPO', '(unset)')}",
                         f"poll      : {POLL}s" if POLL else "poll      : off (HTTP trigger only)",
                         f"last run  : {STATE['last_run'] or 'never'}",
                         f"last file : {STATE['last_file'] or '-'}"]
                if STATE["last_error"]:
                    lines.append(f"last error: {STATE['last_error']}")
                lines += ["", "routes: /run  /latest  /inbox"]
                return self._send(200, "\n".join(lines) + "\n")
            except Exception as e:
                self._send(500, f"error: {e}\n")

        def log_message(self, *a) -> None:
            return

    def _run() -> None:
        try:
            HTTPServer(("0.0.0.0", port), H).serve_forever()
        except Exception as e:
            _log(f"http server failed on {port}: {e}")

    threading.Thread(target=_run, daemon=True).start()
    _log(f"listening on :{port}")


def main() -> int:
    _log("inbox worker starting")
    os.makedirs(VOL, exist_ok=True)
    start_http_server(int(os.environ.get("PORT", "8080")))

    if POLL <= 0:
        _log("polling disabled — waiting for /run")
        while True:
            time.sleep(3600)

    gs = _store()
    while True:
        try:
            process_once(gs)
        except Exception as e:
            STATE["last_error"] = str(e)
            _log(f"cycle failed: {e}")
            traceback.print_exc()
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
