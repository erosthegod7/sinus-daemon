"""
sinus_matrix.py
===============
Matrix-rain wrapper for the SINUS daemon. Purely cosmetic — it launches the real daemon as a
subprocess and renders its stdout in the middle of a falling-glyph field.

Design notes worth knowing before you rely on it:
  * The daemon runs as a normal child process. If this wrapper dies, crashes, or you close the
    window, the daemon dies with it — same as running it directly. Nothing about the search
    changes.
  * Every line the daemon prints is ALSO appended to the log file, unstyled. The pretty output
    is for watching; the log is for reading later and for me to diagnose from.
  * Rendering costs a little CPU that would otherwise go to the search. On a 24-core machine
    that is noise, but if you are squeezing a slow box, run the daemon plain.

Usage
    python sinus_matrix.py                       # wraps railway_daemon.py
    python sinus_matrix.py --log C:\\sinus\\data\\run.log
    python sinus_matrix.py --plain               # no rain, just coloured log lines
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional

GLYPHS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝ0123456789"
RESET = "\033[0m"
BRIGHT = "\033[97m"          # white head of the trail
GREEN = "\033[38;5;46m"
DIM = ("\033[38;5;34m", "\033[38;5;28m", "\033[38;5;22m")
CYAN = "\033[38;5;51m"
YELLOW = "\033[38;5;226m"
RED = "\033[38;5;196m"
CLEAR = "\033[2J\033[H"
HIDE = "\033[?25l"
SHOW = "\033[?25h"


def enable_ansi() -> None:
    """Windows 10+ consoles need VT processing turned on explicitly."""
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Rain:
    """One column of falling glyphs. Independent speed and length so the field looks organic."""

    def __init__(self, row_max: int):
        self.row_max = row_max
        self.reset(initial=True)

    def reset(self, initial: bool = False) -> None:
        self.y = random.randint(-self.row_max, 0) if initial else -random.randint(0, 12)
        self.speed = random.uniform(0.25, 1.0)
        self.length = random.randint(4, max(5, self.row_max // 2))
        self.acc = 0.0

    def step(self) -> None:
        self.acc += self.speed
        while self.acc >= 1.0:
            self.acc -= 1.0
            self.y += 1
        if self.y - self.length > self.row_max:
            self.reset()

    def cells(self) -> List[tuple]:
        out = []
        for i in range(self.length):
            y = self.y - i
            if 0 <= y < self.row_max:
                if i == 0:
                    colour = BRIGHT
                elif i < 3:
                    colour = GREEN
                else:
                    colour = DIM[min(2, (i * 3) // max(self.length, 1))]
                out.append((y, colour, random.choice(GLYPHS)))
        return out


def colourise(line: str) -> str:
    """Highlight the lines that matter so they stand out of the stream."""
    low = line.lower()
    if "champion" in low or "★" in line:
        return YELLOW + line + RESET
    if "pruned" in low:
        return DIM[1] + line + RESET
    if "fatal" in low or "error" in low or "failed" in low or "traceback" in low:
        return RED + line + RESET
    if "trial" in low or "gitstore" in low:
        return CYAN + line + RESET
    return GREEN + line + RESET


def reader(proc: subprocess.Popen, q: "queue.Queue[str]", log_path: Optional[str]) -> None:
    """Drain the child's stdout into the queue and, unstyled, into the log file."""
    fh = open(log_path, "a", encoding="utf-8", errors="replace") if log_path else None
    try:
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            if fh:
                fh.write(line + "\n"); fh.flush()
            q.put(line)
    finally:
        if fh:
            fh.close()
        q.put("__EOF__")


def run(cmd: List[str], log_path: Optional[str], plain: bool, fps: float = 14.0) -> int:
    enable_ansi()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=reader, args=(proc, q, log_path), daemon=True).start()

    if plain:
        while True:
            line = q.get()
            if line == "__EOF__":
                break
            print(colourise(line), flush=True)
        return proc.wait()

    cols, rows = shutil.get_terminal_size((100, 30))
    pane_w = max(50, int(cols * 0.62))
    side = max(6, (cols - pane_w) // 2)
    rain = [Rain(rows) for _ in range(cols)]
    log: List[str] = []
    interval = 1.0 / fps
    done = False

    sys.stdout.write(HIDE + CLEAR)
    try:
        while not done:
            t0 = time.perf_counter()
            while True:                                   # drain everything queued this frame
                try:
                    line = q.get_nowait()
                except queue.Empty:
                    break
                if line == "__EOF__":
                    done = True
                    break
                for chunk in [line[i:i + pane_w] for i in range(0, max(len(line), 1), pane_w)] or [""]:
                    log.append(chunk)
            log = log[-(rows - 4):]

            grid = [[" "] * cols for _ in range(rows)]
            colour = [[""] * cols for _ in range(rows)]
            for x, col in enumerate(rain):
                if side <= x < side + pane_w:              # keep the middle pane clear of rain
                    continue
                col.step()
                for y, c, ch in col.cells():
                    grid[y][x] = ch
                    colour[y][x] = c

            buf = ["\033[H"]
            for y in range(rows):
                row = []
                for x in range(cols):
                    row.append(colour[y][x] + grid[y][x] + RESET if grid[y][x] != " " else " ")
                buf.append("".join(row))
                if y < rows - 1:
                    buf.append("\n")
            sys.stdout.write("".join(buf))

            title = " S I N U S "
            sys.stdout.write(f"\033[{1};{side + (pane_w - len(title)) // 2 + 1}H{BRIGHT}{title}{RESET}")
            for i, line in enumerate(log):
                sys.stdout.write(f"\033[{i + 3};{side + 1}H" + colourise(line[:pane_w]))
            sys.stdout.flush()

            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        sys.stdout.write(SHOW + RESET + "\n")
        sys.stdout.flush()
    return proc.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="railway_daemon.py", help="daemon entrypoint to wrap")
    ap.add_argument("--log", default=None, help="also append raw output here")
    ap.add_argument("--plain", action="store_true", help="colour only, no rain")
    ap.add_argument("--fps", type=float, default=14.0)
    a = ap.parse_args()
    sys.exit(run([sys.executable, a.script], a.log, a.plain, a.fps))
