r"""
patch_daemon_chain.py
=====================
Rewires the daemon and predict_live to train and predict WITH the options book.

Run from C:\sinus-daemon after copying the four new files in:
    python patch_daemon_chain.py

It makes four edits (the fourth adds the magnitude meter print to predict_live) and prints exactly what it changed. Originals are backed up as *.bak.

1. sinus_daemon.py — search_forever gains flow_df and fits Phase 1 on
   (spot_df, chain_df, flow_df) instead of (spot_df, chain_df, None).
2. railway_daemon.py — loads the feed from SINUS_DATA/chain and passes chain_df/flow_df
   into search_forever. Spot comes from the parity series (SINUS_SPOT_FROM_CHAIN=1) so the
   free-stocks-tier crawl is no longer needed. SINUS_USE_CHAIN=0 = old behaviour.
   SINUS_MODE=evolve (default) routes the daemon to sinus_evolve.evolve_forever — the
   breed/cull generational search with horizon-weighted scoring; SINUS_MODE=random keeps
   the old loop.
3. sinus.py predict_live — `pipeline.transform(spot_df, None, None)` becomes a transform
   with the recent chain history + today's snapshots, so the trees see the book live too.

If a pattern is not found it says so and leaves the file alone — paste the printed lines
back to Claude and it will hand-fit the edit.
"""
import os
import re
import shutil
import sys

DAEMON = "railway_daemon.py"
SINUS = "sinus.py"


def _backup(p):
    if not os.path.exists(p + ".bak"):
        shutil.copy2(p, p + ".bak")


SD = "sinus_daemon.py"


def patch_sinus_daemon():
    """search_forever(spot_df, work_dir, chain_df=None, ...) already accepts a chain; add flow_df and
    pass it through to pipeline.fit_transform (currently (spot_df, chain_df, None))."""
    if not os.path.exists(SD):
        print(f"!! {SD} not found in {os.getcwd()}")
        return False
    s = open(SD, encoding="utf-8").read()
    if "flow_df=None" in s and "fit_transform(spot_df, chain_df, flow_df)" in s:
        print(f"{SD}: already patched")
        return True
    sig = re.search(r"def search_forever\(spot_df[^)]*?chain_df=None", s, flags=re.S)
    ft = re.search(r"pipeline\.fit_transform\(\s*spot_df\s*,\s*chain_df\s*,\s*None\s*\)", s)
    if not sig or not ft:
        print(f"!! {SD}: expected patterns not found (sig={bool(sig)}, fit_transform={bool(ft)}). Lines:")
        for i, line in enumerate(s.splitlines(), 1):
            if "def search_forever" in line or "fit_transform(" in line:
                print(f"   {i}: {line.strip()}")
        return False
    _backup(SD)
    s = s[:sig.end()] + ", flow_df=None" + s[sig.end():]
    s = re.sub(r"pipeline\.fit_transform\(\s*spot_df\s*,\s*chain_df\s*,\s*None\s*\)",
               "pipeline.fit_transform(spot_df, chain_df, flow_df)", s)
    open(SD, "w", encoding="utf-8").write(s)
    print(f"{SD}: search_forever now takes flow_df and fits on (spot_df, chain_df, flow_df)")
    return True


LOADER_BLOCK = """{i}# --- options-book feed (patch_daemon_chain.py) ---------------------------------
{i}chain_df, flow_df = None, None
{i}if os.environ.get('SINUS_USE_CHAIN', '1') == '1':
{i}    try:
{i}        from sinus_chain_loader import load_history as _load_chain
{i}        _cdir = os.environ.get('SINUS_DATA', os.path.dirname(os.path.abspath(__file__)))
{i}        _spot2, chain_df, flow_df = _load_chain(_cdir, max_sessions=int(os.environ.get('SINUS_MAX_SESSIONS', '500')))
{i}        if os.environ.get('SINUS_SPOT_FROM_CHAIN', '1') == '1' and len(_spot2) > 0:
{i}            spot_df = _spot2      # parity spot from the chain: no stocks plan, no 5-calls/min crawl
{i}        print(f'[chain] feed loaded: {{len(chain_df):,}} chain rows · {{len(flow_df):,}} prints · '
{i}              f'{{spot_df["ts"].dt.normalize().nunique()}} sessions', flush=True)
{i}    except Exception as _e:
{i}        print(f'[chain] feed unavailable ({{type(_e).__name__}}: {{_e}}) - training price-only', flush=True)
{i}        chain_df, flow_df = None, None
{i}# ----------------------------------------------------------------------------------
"""


def patch_daemon():
    """railway_daemon.py: load the feed, then hand chain_df/flow_df to search_forever."""
    if not os.path.exists(DAEMON):
        print(f"!! {DAEMON} not found in {os.getcwd()}")
        return False
    s = open(DAEMON, encoding="utf-8").read()
    if "sinus_chain_loader" in s:
        print(f"{DAEMON}: already patched")
        return True
    call = re.search(r"^([ \t]*)(.*\bsearch_forever\(\s*spot_df\s*,)", s, flags=re.M)
    if not call:
        print(f"!! {DAEMON}: no 'search_forever(spot_df,' call found. Lines mentioning search_forever:")
        for i, line in enumerate(s.splitlines(), 1):
            if "search_forever" in line:
                print(f"   {i}: {line.strip()}")
        return False
    _backup(DAEMON)
    indent = call.group(1)
    routed = re.sub(r"[\w\.]*search_forever\(", "_SEARCH(", call.group(2))
    s = (s[:call.start()] + LOADER_BLOCK.format(i=indent)
         + f"{indent}import sinus_evolve as _se\n"
         + f"{indent}_SEARCH = _se.evolve_forever if os.environ.get('SINUS_MODE', 'evolve') == 'evolve' else sd.search_forever\n"
         + f"{indent}print(f\"[mode] {{'EVOLVE (breed/cull)' if _SEARCH is _se.evolve_forever else 'random search'}}\", flush=True)\n"
         + call.group(1) + routed
         + " chain_df=chain_df, flow_df=flow_df," + s[call.end():])
    if not re.search(r"^\s*import os\b", s, flags=re.M):
        s = "import os\n" + s
    open(DAEMON, "w", encoding="utf-8").write(s)
    print(f"{DAEMON}: feed loader inserted; search_forever now receives chain_df/flow_df")
    return True


def patch_predict_live():
    if not os.path.exists(SINUS):
        print(f"!! {SINUS} not found")
        return False
    s = open(SINUS, encoding="utf-8").read()
    if "load_recent_for_live" in s:
        print(f"{SINUS}: already patched")
        return True
    m = re.search(r"def predict_live\(", s)
    if not m:
        print(f"!! {SINUS}: predict_live not found")
        return False
    body_start = m.start()
    seg = s[body_start:body_start + 20000]
    hit = re.search(r"^([ \t]*)out = pipeline\.transform\(spot_df, None, None\)", seg, flags=re.M)
    if not hit:
        print(f"!! {SINUS}: 'out = pipeline.transform(spot_df, None, None)' not found inside predict_live. Candidates:")
        for i, line in enumerate(seg.splitlines(), 1):
            if "pipeline.transform" in line:
                print(f"   +{i}: {line.strip()}")
        return False
    _backup(SINUS)
    ind = hit.group(1)
    new = (
        f"{ind}chain_df, flow_df = None, None\n"
        f"{ind}if os.environ.get('SINUS_USE_CHAIN', '1') == '1':\n"
        f"{ind}    try:\n"
        f"{ind}        from sinus_chain_loader import load_recent_for_live as _lrl\n"
        f"{ind}        _s2, chain_df, flow_df = _lrl(os.environ.get('SINUS_DATA', os.path.dirname(os.path.abspath(__file__))))\n"
        f"{ind}        chain_df = chain_df[chain_df['ts'] <= spot_df['ts'].max()]\n"
        f"{ind}        flow_df = flow_df[flow_df['ts'] <= spot_df['ts'].max()]\n"
        f"{ind}    except Exception as _e:\n"
        f"{ind}        log.warning('predict_live.chain_unavailable', extra={{'err': f'{{type(_e).__name__}}: {{_e}}'}})\n"
        f"{ind}        chain_df, flow_df = None, None\n"
        f"{ind}out = pipeline.transform(spot_df, chain_df, flow_df)"
    )
    seg2 = seg[:hit.start()] + new + seg[hit.end():]
    mline = re.search(r"^([ \t]*)res\[\"_meta\"\]\[\"champion\"\] = .*$", seg2, flags=re.M)
    if mline:
        mi = mline.group(1)
        mag = (
            f"\n{mi}# --- magnitude meter (patch_daemon_chain.py) ---\n"
            f"{mi}try:\n"
            f"{mi}    from sinus_magnitude import predict_magnitude as _pm, format_meter as _fm\n"
            f"{mi}    if os.path.exists(os.path.join(champion_dir, 'mag_up.pkl')):\n"
            f"{mi}        _F = out.features if hasattr(out, 'features') else out.features_raw\n"
            f"{mi}        _row = _F.iloc[idx].to_numpy() if hasattr(_F, 'iloc') else _F[idx]\n"
            f"{mi}        res['_magnitude'] = _pm(champion_dir, _row, meta_row['ts'], spot_df)\n"
            f"{mi}        print(_fm(res['_magnitude']))\n"
            f"{mi}except Exception as _e:\n"
            f"{mi}    log.warning('predict_live.magnitude_failed', extra={{'err': f'{{type(_e).__name__}}: {{_e}}'}})\n"
        )
        seg2 = seg2[:mline.end()] + mag + seg2[mline.end():]
    else:
        print(f"   note: 'res[\"_meta\"][\"champion\"] =' line not found in predict_live — magnitude print not inserted")
    s = s[:body_start] + seg2 + s[body_start + 20000:]
    open(SINUS, "w", encoding="utf-8").write(s)
    print(f"{SINUS}: predict_live now transforms with chain_df/flow_df")
    return True


if __name__ == "__main__":
    ok0 = patch_sinus_daemon()
    ok1 = patch_daemon()
    ok2 = patch_predict_live()
    for f in (SD, DAEMON, SINUS):
        try:
            import py_compile
            py_compile.compile(f, doraise=True)
            print(f"{f}: compiles")
        except Exception as e:
            print(f"!! {f} does not compile after patch: {e} — restore from {f}.bak")
    sys.exit(0 if (ok0 and ok1 and ok2) else 1)
