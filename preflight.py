"""
preflight.py — verify everything before you walk away from an overnight run.

    python preflight.py            # full check, exercises the real data path
    python preflight.py --quick    # skip the feature-pipeline build (much faster)

Exit code 0 means go. Non-zero means at least one FAIL. Every check prints what
it looked at, so a failure tells you what to change rather than just that
something is wrong.

The point of this file: the 2026-09-02 run crash-looped every 60 seconds until
morning and completed zero trials. Everything that made that possible is checked
here, before you commit the night to it.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time

QUICK = "--quick" in sys.argv

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = FAIL, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        results.append((status, name, detail))
        colour = {"PASS": "", "WARN": "", "FAIL": ""}[status]
        print(f"  [{status}] {name:<34} {detail}" + (f"  ({dt:.1f}s)" if dt > 1 else ""))
        return fn
    return deco


HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

print(f"\nSINUS preflight — {HERE}")
print(f"{'quick mode (pipeline build skipped)' if QUICK else 'full check'}\n")

# --------------------------------------------------------------------------- #
print("environment")


@check("python")
def _py():
    v = sys.version_info
    if v < (3, 10):
        return FAIL, f"{sys.version.split()[0]} — too old"
    return PASS, f"{sys.version.split()[0]}"


@check("virtualenv active")
def _venv():
    if sys.prefix == sys.base_prefix:
        return WARN, "not in a venv — packages may not be the ones you installed"
    return PASS, os.path.basename(sys.prefix)


@check("required env vars")
def _env():
    need = ["POLYGON_KEY", "SINUS_VOLUME", "SINUS_DATA"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        return FAIL, f"missing {missing} — dot-source _sinus_env.ps1 first"
    return PASS, f"volume={os.environ['SINUS_VOLUME']}"


@check("git credentials")
def _git():
    repo, tok = os.environ.get("SINUS_GIT_REPO"), os.environ.get("SINUS_GIT_TOKEN")
    if not shutil.which("git"):
        return FAIL, "git binary not on PATH — the champion cannot be pushed"
    if not (repo and tok):
        return FAIL, ("SINUS_GIT_REPO/SINUS_GIT_TOKEN unset — the champion stays on this "
                      "laptop only. This is what happened on 2026-09-02.")
    return PASS, f"{repo} as {os.environ.get('SINUS_NODE', 'node')}"


@check("margins")
def _margins():
    s = os.environ.get("SINUS_SCREEN_MARGIN", "0.002")
    p = os.environ.get("SINUS_PROMOTE_MARGIN", "0.0005")
    if float(p) > float(s):
        return WARN, f"promote {p} > screen {s} — promotions will be rarer than screens"
    return PASS, f"screen {s} · promote {p}"


# --------------------------------------------------------------------------- #
print("\npackages")


@check("numpy / pandas / sklearn")
def _core():
    vs = [f"{m} {importlib.import_module(m).__version__}"
          for m in ("numpy", "pandas", "sklearn")]
    return PASS, " · ".join(vs)


@check("lightgbm")
def _lgb():
    import lightgbm
    return PASS, lightgbm.__version__


@check("catboost")
def _cat():
    try:
        import catboost
        return PASS, catboost.__version__
    except ImportError:
        return WARN, "absent — trees run LightGBM-only, scores will differ from the champion"


@check("torch + CUDA")
def _torch():
    import torch
    if not torch.cuda.is_available():
        return WARN, (f"torch {torch.__version__} CPU-only — trials will be several times "
                      f"slower. Reinstall with cu128 for the 5070.")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    tag = f"sm_{cap[0]}{cap[1]}"
    if tag not in [a.replace("compute_", "sm_") for a in arch] and tag not in arch:
        return FAIL, (f"{name} is {tag} but this torch build supports {arch}. "
                      f"Install torch for cu128 or every TFT fit fails.")
    return PASS, f"{torch.__version__} · {name} · {tag}"


# --------------------------------------------------------------------------- #
print("\nproject files")


@check("fixes applied")
def _fixed():
    missing = []
    for f, marker in (("sinus.py", "SinusStopped"),
                      ("sinus_evolve.py", "promote_margin"),
                      ("railway_daemon.py", "_repeat >= 5")):
        if marker not in open(f, encoding="utf-8", errors="replace").read():
            missing.append(f)
    if missing:
        return FAIL, (f"unpatched: {missing} — these files lost their 2026-09-03 fixes. "
                      f"Restore from old/backups/*.prefix.bak, or re-run old/patches/sinus_fixes.py")
    return PASS, "gate, Ctrl-C, session floor, crash backoff"


@check("modules import")
def _imports():
    for m in ("sinus", "sinus_search", "sinus_daemon", "sinus_evolve",
              "sinus_gitstore", "sinus_chain_loader", "sinus_magnitude"):
        importlib.import_module(m)
    return PASS, "7 modules"


@check("disk space on volume")
def _disk():
    vol = os.environ.get("SINUS_VOLUME", ".")
    os.makedirs(vol, exist_ok=True)
    free = shutil.disk_usage(vol).free / 1e9
    if free < 5:
        return FAIL, f"{free:.1f} GB free — a night of candidates will fill this"
    if free < 20:
        return WARN, f"{free:.1f} GB free"
    return PASS, f"{free:.0f} GB free"


# --------------------------------------------------------------------------- #
print("\ndata")

_loaded = {}


@check("chain backfill on disk")
def _chain_files():
    import glob
    d = os.environ.get("SINUS_DATA", ".")
    days = sorted({os.path.basename(p).split("_")[-1].split(".")[0]
                   for p in glob.glob(os.path.join(d, "chain", "chain_*.*"))
                   if not p.endswith(".tmp")})
    if not days:
        return FAIL, f"no chain_*.parquet under {os.path.join(d, 'chain')} — run polygon_chain_history.py"
    _loaded["days"] = days
    return PASS, f"{len(days)} days · {days[0]} to {days[-1]}"


@check("session floor")
def _floor():
    from sinus_chain_loader import load_history
    d = os.environ.get("SINUS_DATA", ".")
    floor = int(os.environ.get("SINUS_MIN_SESSIONS", "7"))
    spot, chain, flow = load_history(d, max_sessions=int(os.environ.get("SINUS_MAX_SESSIONS", "500")),
                                     verbose=False)
    n = spot["ts"].dt.normalize().nunique()
    _loaded.update(spot=spot, chain=chain, flow=flow, n=n)
    if n < floor:
        return FAIL, (f"chain parity spot covers {n} sessions, floor is {floor}. This is the "
                      f"exact condition that started the 5-session run. Back-fill more days.")
    if n < 60:
        return WARN, f"{n} sessions — enough to run, thin enough that a champion may be luck"
    return PASS, f"{n} sessions · {len(spot):,} bars · {len(chain):,} chain rows · {len(flow):,} prints"


@check("feature pipeline builds")
def _pipeline():
    if QUICK:
        return WARN, "skipped (--quick)"
    if "spot" not in _loaded:
        return FAIL, "no data loaded — earlier check failed"
    from sinus import FeaturePipeline, PipelineConfig
    out = FeaturePipeline(PipelineConfig()).fit_transform(
        _loaded["spot"], _loaded["chain"], _loaded["flow"])
    _loaded["out"] = out
    nan_frac = float(out.features.isna().to_numpy().mean()) if hasattr(out.features, "isna") else 0.0
    if out.features.shape[1] < 80:
        return WARN, f"{out.features.shape[1]} features — chain features may not be landing"
    return PASS, (f"{len(out.meta):,} bars · {out.features.shape[1]} features · "
                  f"{nan_frac:.0%} NaN before impute")


@check("train/val/test splits non-empty")
def _splits():
    if QUICK or "out" not in _loaded:
        return WARN, "skipped"
    from sinus import TrainingBundle
    sizes = {}
    for split in ("train", "val", "test"):
        b = TrainingBundle.from_phase1(_loaded["out"], split, lookback=60)
        sizes[split] = len(b.Y)
    if min(sizes.values()) < 30:
        return FAIL, (f"{sizes} — score_predictions needs 30+ rows per horizon or every "
                      f"trial scores inf forever")
    return PASS, " · ".join(f"{k} {v:,}" for k, v in sizes.items())


# --------------------------------------------------------------------------- #
print("\nchampion")


@check("champion.json readable")
def _champ():
    p = os.path.join(os.environ.get("SINUS_VOLUME", "."), "champion", "champion.json")
    if not os.path.exists(p):
        return WARN, "none yet — the first promotion creates it"
    m = json.load(open(p))
    _loaded["champ"] = m
    return PASS, (f"trial {m['trial']} · val {m.get('val_score', float('nan')):.4f} · "
                  f"test {m['test_score']:.4f} · acc {m.get('test_acc', float('nan')):.3f}")


@check("champion weights load")
def _champ_load():
    if "champ" not in _loaded:
        return WARN, "no champion to load"
    from sinus_daemon import load_champion
    eng, meta = load_champion(os.environ.get("SINUS_VOLUME", "."))   # it appends "champion" itself
    return PASS, f"engine + pipeline unpickled for trial {meta['trial']}"


@check("git round-trip")
def _git_rt():
    if not (os.environ.get("SINUS_GIT_REPO") and os.environ.get("SINUS_GIT_TOKEN")):
        return WARN, "skipped — no credentials"
    from sinus_gitstore import GitStore
    gs = GitStore(os.path.join(os.environ.get("SINUS_VOLUME", "."), "_preflight"))
    if not gs.enabled:
        return FAIL, "gitstore refused to initialise — check the token's Contents scope"
    remote = gs.remote_champion_score()
    return PASS, (f"reachable · remote champion test "
                  f"{'none' if remote is None else format(remote, '.4f')}")


# --------------------------------------------------------------------------- #
print("\nhost")


@check("sleep / hibernate disabled")
def _sleep():
    if os.name != "nt":
        return WARN, "not Windows — skipped"
    try:
        out = subprocess.run(["powercfg", "/q", "SCHEME_CURRENT", "SUB_SLEEP"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return WARN, f"could not read power settings ({type(e).__name__})"
    vals = [l for l in out.splitlines() if "Current AC Power Setting Index" in l]
    if any(v.strip().split()[-1] != "0x00000000" for v in vals):
        return WARN, ("a sleep timeout is non-zero on AC. start.ps1 holds the machine awake "
                      "while it runs, but closing the lid can still suspend it.")
    return PASS, "no AC sleep timeout"


# --------------------------------------------------------------------------- #
n_fail = sum(1 for s, _, _ in results if s == FAIL)
n_warn = sum(1 for s, _, _ in results if s == WARN)
print(f"\n{'-' * 72}")
print(f"{len(results)} checks · {n_fail} FAIL · {n_warn} WARN")
if n_fail:
    print("\nfix these before starting:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"  · {name}: {detail}")
    sys.exit(1)
print("\nclear to run:  python railway_daemon.py")
sys.exit(0)
