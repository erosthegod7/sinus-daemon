p = "sinus_evolve.py"
s = open(p, encoding="utf-8").read()
old = "fit_magnitude(out, spot_df, os.path.join(work_dir, CHAMPION))"
new = "(fit_magnitude(out, spot_df, os.path.join(work_dir, CHAMPION)) if os.environ.get('SINUS_FIT_MAG','1')=='1' else None)"
assert s.count(old) == 1, f"found {s.count(old)}"
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
