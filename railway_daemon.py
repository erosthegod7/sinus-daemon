"""
railway_daemon.py
=================
Entrypoint for the always-on SINUS search daemon on Railway.

Responsibilities, in order:
  1. Resolve price history — from the persistent volume if a CSV was uploaded, otherwise
     pulled from Polygon's stock aggregates and cached to the volume so later restarts
     are instant.
  2. Run `search_forever` against a work_dir on the volume, so the leaderboard and champion
     survive redeploys, crashes and container recycling.
  3. Log heartbeats so `railway logs` shows progress without attaching to the process.

Environment variables (set these in the Railway dashboard):
    POLYGON_KEY        required
    SINUS_VOLUME       volume mount path            default /data
    SINUS_CSV          filename of a price CSV already on the volume
    SINUS_CSV_URL      URL to download a price CSV from on first boot (Google Drive share
                       links are auto-converted to direct downloads). Downloaded once, then
                       cached on the volume — Railway has no upload UI, so this is the
                       practical way to get a large local file onto the volume from a phone.
    SINUS_SYMBOL       default SPY
    SINUS_YEARS        years of history to pull from Polygon   default 2
    SINUS_TFT_EPOCHS   epochs per search trial       default 12
    SINUS_MIN_SESSIONS refuse to start below this    default 100

Why it refuses to start on thin data: a search daemon left running on 60 sessions will
happily produce a champion, and that champion will be luck. Failing loudly at boot is
better than burning a month of compute on a number that means nothing.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd

# Windows consoles default to cp1252, which cannot encode the arrows and box characters the
# progress lines use — an UnicodeEncodeError there would kill the daemon mid-run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

VOL = os.environ.get("SINUS_VOLUME", "/data")
SYMBOL = os.environ.get("SINUS_SYMBOL", "SPY")
YEARS = float(os.environ.get("SINUS_YEARS", "2"))
EPOCHS = int(os.environ.get("SINUS_TFT_EPOCHS", "12"))
MIN_SESSIONS = int(os.environ.get("SINUS_MIN_SESSIONS", "100"))
MAX_SESSIONS = int(os.environ.get("SINUS_MAX_SESSIONS", "300"))
PRUNE = os.environ.get("SINUS_PRUNE", "1") == "1"
PRUNE_PCT = float(os.environ.get("SINUS_PRUNE_PERCENTILE", "60"))
SCREEN_ROUNDS = int(os.environ.get("SINUS_SCREEN_ROUNDS", "150"))
CACHE_PQ = os.path.join(VOL, f"{SYMBOL.lower()}_1min_cache.parquet")
CACHE_CSV = os.path.join(VOL, f"{SYMBOL.lower()}_1min_cache.csv.gz")


def _cache_write(df: pd.DataFrame) -> str:
    """Parquet if pyarrow is available, gzipped CSV otherwise. The cache is an optimisation,
    never a hard dependency — a missing parquet engine must not stop the daemon booting."""
    try:
        df.to_parquet(CACHE_PQ, index=False)
        return CACHE_PQ
    except Exception as e:
        _log(f"parquet unavailable ({type(e).__name__}); caching as gzipped CSV")
        df.to_csv(CACHE_CSV, index=False, compression="gzip")
        return CACHE_CSV


def _cache_read() -> Optional[pd.DataFrame]:
    for p, fn in ((CACHE_PQ, pd.read_parquet), (CACHE_CSV, pd.read_csv)):
        if os.path.exists(p):
            try:
                df = fn(p)
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
                return df
            except Exception as e:
                _log(f"cache {os.path.basename(p)} unreadable ({e}) — ignoring it")
    return None


def _log(msg: str) -> None:
    print(f"[{pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_polygon_minutes(symbol: str, years: float, api_key: str) -> pd.DataFrame:
    """Pull 1-minute bars from Polygon in monthly chunks.

    Requires the STOCKS asset class on the key. Options and stocks are billed separately at
    Polygon, so an options-only key returns 403 here — the error says so plainly rather than
    letting the daemon start on an empty frame.
    """
    from sinus import _get

    end = pd.Timestamp.now(tz="America/New_York").normalize()
    start = end - pd.Timedelta(days=int(365 * years))
    frames, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=30), end)
        try:
            j = _get(f"/v2/aggs/ticker/{symbol}/range/1/minute/{cur.date()}/{nxt.date()}",
                     {"adjusted": "true", "sort": "asc", "limit": 50000}, api_key)
        except Exception as e:
            if "403" in str(e) or "NOT_AUTHORIZED" in str(e).upper():
                raise RuntimeError(
                    "Polygon returned 403 for stock aggregates. Your key covers OPTIONS but not "
                    "STOCKS (they are separate subscriptions). Either add the stocks plan, or "
                    "upload your 1-min CSV to the Railway volume and set SINUS_CSV to its filename."
                ) from e
            _log(f"chunk {cur.date()} failed: {e}")
            cur = nxt
            continue
        res = j.get("results") or []
        if res:
            frames.append(pd.DataFrame(res)[["t", "c"]])
        _log(f"  {cur.date()} → {nxt.date()}: {len(res):,} bars")
        cur = nxt
        # Polygon's free STOCKS tier allows 5 calls/min. The options subscription does not lift
        # it. 13s keeps us under the ceiling; on 2026-09-01 a 0.2s sleep lost ~half the chunks
        # to 429s. This runs once — the result caches to the volume.
        time.sleep(float(os.environ.get("SINUS_FETCH_SLEEP", "13")))
    if not frames:
        raise RuntimeError("Polygon returned no bars at all — check the key and the symbol")
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return pd.DataFrame({"ts": ts, "spot": pd.to_numeric(df["c"], errors="coerce")})


def _direct_url(url: str) -> str:
    """Rewrite a Google Drive share link into a direct download. Drive's normal /file/d/<id>/view
    URL returns an HTML page, not the file — a silent failure that would otherwise surface much
    later as an unparseable CSV."""
    if "drive.google.com" in url:
        fid = None
        if "/file/d/" in url:
            fid = url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            fid = url.split("id=")[1].split("&")[0]
        if fid:
            return f"https://drive.google.com/uc?export=download&id={fid}"
    return url


def _download(url: str, dest: str) -> str:
    """Stream a file to disk. Handles Drive's large-file confirmation interstitial, which
    otherwise saves an HTML warning page under a .csv name."""
    import requests
    url = _direct_url(url)
    _log(f"downloading price CSV from {url[:80]}...")
    with requests.Session() as ses:
        r = ses.get(url, stream=True, timeout=120)
        # Drive interposes a virus-scan warning on big files; the real bytes need a confirm token
        if "text/html" in r.headers.get("content-type", ""):
            token = next((v for k, v in r.cookies.items() if k.startswith("download_warning")), None)
            if token:
                r = ses.get(url + f"&confirm={token}", stream=True, timeout=120)
            elif "confirm=" in r.text:
                import re as _re
                m = _re.search(r"confirm=([0-9A-Za-z_-]+)", r.text)
                if m:
                    r = ses.get(url + f"&confirm={m.group(1)}", stream=True, timeout=120)
        r.raise_for_status()
        n = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk); n += len(chunk)
                if n % (32 << 20) < (1 << 20):
                    _log(f"  {n / 1e6:.0f} MB")
    _log(f"downloaded {n / 1e6:.1f} MB to {dest}")
    if n < 10000:
        raise RuntimeError(f"downloaded only {n} bytes — the URL is probably not a direct file link")
    return dest


def resolve_price_history() -> pd.DataFrame:
    """Volume cache → uploaded CSV → Polygon. Caches whatever it resolves for next boot."""
    import sinus

    csv = os.environ.get("SINUS_CSV")
    path = (csv if os.path.isabs(csv) else os.path.join(VOL, csv)) if csv else None

    # A freshly rebuilt SINUS_CSV must beat the cache. After a backfill the cache still holds
    # the old, much shorter series; reading it there means booting on a fraction of the data
    # that was just downloaded — which looks exactly like the backfill having failed.
    stale = bool(path and os.path.exists(path) and any(
        os.path.exists(c) and os.path.getmtime(path) > os.path.getmtime(c)
        for c in (CACHE_PQ, CACHE_CSV)))
    if stale:
        _log("SINUS_CSV is newer than the price cache — using the CSV and refreshing the cache")

    cached = None if stale else _cache_read()
    if cached is not None:
        _log(f"price history from cache: {len(cached):,} bars")
        return sinus._finalise(cached, verbose=True)

    if csv:
        if os.path.exists(path):
            _log(f"loading uploaded CSV {path}")
            df = sinus.load_csv(path)
            _log(f"cached to {_cache_write(df)}")
            return df
        _log(f"SINUS_CSV={csv} set but {path} not found — falling through to Polygon")

    url = os.environ.get("SINUS_CSV_URL")
    if url:
        local = os.path.join(VOL, "downloaded_prices.csv")
        if not os.path.exists(local):
            _download(url, local)
        df = sinus.load_csv(local)
        _log(f"cached to {_cache_write(df)}")
        return df

    key = os.environ.get("POLYGON_KEY")
    if not key:
        raise RuntimeError("no price source: set POLYGON_KEY, or SINUS_CSV_URL, or SINUS_CSV")
    _log(f"pulling {YEARS}y of {SYMBOL} 1-min from Polygon (first boot only)")
    raw = fetch_polygon_minutes(SYMBOL, YEARS, key)
    df = sinus._finalise(raw, verbose=True)
    _log(f"cached {len(df):,} bars to {_cache_write(df)}")
    return df



# ----------------------------------------------------------------------------- #
# Results endpoint — the volume is not Google Drive, so expose it over HTTP
# ----------------------------------------------------------------------------- #
def start_http_server(work_dir: str, port: int) -> None:
    """Serve the leaderboard and champion so Colab can pull them into Drive.

    Runs on a daemon thread: if it fails to bind, the SEARCH still runs. Monitoring must
    never be able to take down the thing being monitored.

    Routes
        /            plain-text status summary
        /leaderboard leaderboard.csv
        /champion    champion.json (which trial won and its test score)
        /model.zip   the champion weights, zipped
    """
    import io
    import json as _json
    import threading
    import zipfile
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:                       # noqa: N802
            try:
                path = self.path.split("?")[0].rstrip("/") or "/"
                board_p = os.path.join(work_dir, "leaderboard.csv")
                champ_p = os.path.join(work_dir, "champion", "champion.json")

                if path == "/leaderboard":
                    if not os.path.exists(board_p):
                        return self._send(404, b"no leaderboard yet")
                    return self._send(200, open(board_p, "rb").read(), "text/csv")

                if path == "/champion":
                    if not os.path.exists(champ_p):
                        return self._send(404, b"no champion promoted yet")
                    return self._send(200, open(champ_p, "rb").read(), "application/json")

                if path == "/model.zip":
                    d = os.path.join(work_dir, "champion")
                    if not os.path.isdir(d):
                        return self._send(404, b"no champion yet")
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                        for root, _, files in os.walk(d):
                            for f in files:
                                fp = os.path.join(root, f)
                                z.write(fp, os.path.relpath(fp, d))
                    return self._send(200, buf.getvalue(), "application/zip")

                # status
                lines = [f"SINUS daemon · {pd.Timestamp.now(tz='UTC')}"]
                if os.path.exists(board_p):
                    try:
                        b = pd.read_csv(board_p)
                        lines.append(f"trials      : {len(b)}")
                        if "status" in b and (b.status == "ok").any():
                            ok = b[b.status == "ok"]
                            lines.append(f"successful  : {len(ok)}")
                            lines.append(f"best val    : {ok['score'].min():.4f}")
                            lines.append(f"median secs : {ok['seconds'].median():.0f}")
                    except Exception as e:
                        lines.append(f"leaderboard unreadable: {e}")
                else:
                    lines.append("trials      : 0 (no leaderboard yet)")
                if os.path.exists(champ_p):
                    c = _json.load(open(champ_p))
                    lines.append(f"champion    : trial {c['trial']} · test {c['test_score']:.4f} "
                                 f"· mae {c['test_mae']:.4f} · acc {c['test_acc']:.3f}")
                    lines.append(f"promoted    : {c['promoted_at']}")
                else:
                    lines.append("champion    : none promoted yet")
                lines += ["", "routes: /leaderboard  /champion  /model.zip"]
                return self._send(200, ("\n".join(lines) + "\n").encode())
            except Exception as e:                      # never let a bad request kill the thread
                self._send(500, f"error: {e}\n".encode())

        def log_message(self, *a) -> None:               # keep request noise out of the deploy logs
            return

    def _run() -> None:
        try:
            HTTPServer(("0.0.0.0", port), H).serve_forever()
        except Exception as e:
            _log(f"http server failed to start on {port}: {e} (search continues regardless)")

    threading.Thread(target=_run, daemon=True).start()
    _log(f"results endpoint listening on :{port}")


def main() -> int:
    _log("SINUS daemon starting")
    _log(f"volume={VOL} symbol={SYMBOL} epochs/trial={EPOCHS} "
         f"prune={'on p%.0f' % PRUNE_PCT if PRUNE else 'off'}")
    os.makedirs(VOL, exist_ok=True)

    try:
        spot_df = resolve_price_history()
    except Exception as e:
        _log(f"FATAL could not resolve price history: {e}")
        traceback.print_exc()
        return 1

    n_sess = spot_df["ts"].dt.normalize().nunique()
    _log(f"{len(spot_df):,} bars across {n_sess} sessions")

    # Trim to the most recent MAX_SESSIONS. The TFT sequence tensor is
    # rows x lookback x features in float32 — at 500 sessions and a 60-bar lookback that is
    # ~7 GB in ONE array, which no reasonable container survives. 300 sessions is ample for a
    # hyper-parameter search and keeps the tensor near 4 GB. Recent sessions are also the more
    # relevant ones. Raise SINUS_MAX_SESSIONS only alongside more container memory.
    if n_sess > MAX_SESSIONS:
        days = sorted(spot_df["ts"].dt.normalize().unique())[-MAX_SESSIONS:]
        spot_df = spot_df[spot_df["ts"].dt.normalize().isin(days)].reset_index(drop=True)
        n_sess = MAX_SESSIONS
        _log(f"trimmed to the most recent {MAX_SESSIONS} sessions ({len(spot_df):,} bars) "
             f"to keep the sequence tensor inside container memory")
    if n_sess < MIN_SESSIONS:
        _log(f"FATAL only {n_sess} sessions, minimum is {MIN_SESSIONS}. A search on this little "
             f"data finds luck, not edge. Lower SINUS_MIN_SESSIONS only if you know why.")
        return 1

    import sinus_daemon as sd
    work = os.path.join(VOL, "champion")
    os.makedirs(work, exist_ok=True)
    start_http_server(work, int(os.environ.get("PORT", "8080")))
    _log(f"work_dir {work} — leaderboard and champion persist here across restarts")

    _last_err, _repeat = None, 0
    while True:                                        # outer loop: survive an unexpected crash
        try:
            # --- options-book feed (patch_daemon_chain.py) ---------------------------------
            chain_df, flow_df = None, None
            if os.environ.get('SINUS_USE_CHAIN', '1') == '1':
                try:
                    from sinus_chain_loader import load_history as _load_chain
                    _cdir = os.environ.get('SINUS_DATA', os.path.dirname(os.path.abspath(__file__)))
                    _spot2, chain_df, flow_df = _load_chain(_cdir, max_sessions=int(os.environ.get('SINUS_MAX_SESSIONS', '500')))
                    if os.environ.get('SINUS_SPOT_FROM_CHAIN', '1') == '1' and len(_spot2) > 0:
                        spot_df = _spot2      # parity spot from the chain: no stocks plan, no 5-calls/min crawl
                        # The MIN_SESSIONS check above ran on the Polygon/CSV frame. Substituting the
                        # chain's parity spot can drop the count far below the floor without tripping
                        # it — on 2026-09-02 that started a search on 5 sessions against a floor of 7.
                        _n = spot_df['ts'].dt.normalize().nunique()
                        if _n < MIN_SESSIONS:
                            _log(f"FATAL chain parity spot covers {_n} sessions, floor is {MIN_SESSIONS}. "
                                 f"Back-fill more days (polygon_chain_history.py) or set "
                                 f"SINUS_SPOT_FROM_CHAIN=0 to keep the longer price-only series.")
                            return 1
                    print(f'[chain] feed loaded: {len(chain_df):,} chain rows · {len(flow_df):,} prints · '
                          f'{spot_df["ts"].dt.normalize().nunique()} sessions', flush=True)
                except Exception as _e:
                    print(f'[chain] feed unavailable ({type(_e).__name__}: {_e}) - training price-only', flush=True)
                    chain_df, flow_df = None, None
            # ----------------------------------------------------------------------------------
            import sinus_evolve as _se
            _SEARCH = _se.evolve_forever if os.environ.get('SINUS_MODE', 'evolve') == 'evolve' else sd.search_forever
            print(f"[mode] {'EVOLVE (breed/cull)' if _SEARCH is _se.evolve_forever else 'random search'}", flush=True)
            _SEARCH(spot_df, chain_df=chain_df, flow_df=flow_df, work_dir=work, tft_epochs=EPOCHS, prune=PRUNE,
                              prune_percentile=PRUNE_PCT, screen_rounds=SCREEN_ROUNDS)
            _log("search_forever returned (stop requested) — exiting")
            return 0
        except KeyboardInterrupt:
            _log("interrupted — exiting cleanly")
            return 0
        except Exception as e:
            _log(f"daemon crashed: {e}")
            traceback.print_exc()
            # A bug that fails on every boot used to retry every 60s all night and still
            # look like a running daemon. Back off, and give up on a fault that is clearly
            # not transient so it is visible in the morning instead of buried in a log.
            sig = f"{type(e).__name__}: {e}"
            _repeat = _repeat + 1 if sig == _last_err else 1
            _last_err = sig
            if _repeat >= 5:
                _log(f"FATAL same failure {_repeat}x in a row — this is not transient. Exiting so "
                     f"you see it. The leaderboard and champion on disk are intact.")
                return 1
            wait = min(60 * 2 ** (_repeat - 1), 900)
            _log(f"restarting in {wait}s (failure {_repeat}/5); the leaderboard is intact "
                 f"so at most one trial is lost")
            time.sleep(wait)


if __name__ == "__main__":
    sys.exit(main())
