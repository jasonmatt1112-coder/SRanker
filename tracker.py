"""
Hyperdrift tracker (headless, no Streamlit needed).

Logs, once per trading day:
  * our Top 15 picks (with 1-day projection) and how they did open -> close
  * the Top 10 Wall Street firms' Buy picks, scored the same way
  * the Live Trading paper-account results
into CSV files (default: data/stock_ranker_history*.csv). Missed days are back-filled automatically.

    python tracker.py          one pass, then exit   (GitHub Actions / cron)
    python tracker.py --loop   repeat every 5 minutes (any always-on machine)
    python tracker.py --parlays  only re-price / grade the $10 parlays (run every 15 minutes)

Needs app.py (the file the notebook writes) in the same folder. Set HISTORY_PATH to change where files go.
"""
import os
import sys
import time
import types

# --- a tiny stand-in for Streamlit so app.py's data + scoring code can be reused as-is
def _deco(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    return lambda f: f


st = types.ModuleType("streamlit")
st.cache_data = st.cache_resource = st.fragment = _deco
st.set_page_config = lambda *a, **k: None
st.dataframe = lambda *a, **k: None
st.secrets, st.session_state, st.__version__ = {}, {}, "0"
sys.modules["streamlit"] = st
try:
    import altair  # noqa: F401
except ImportError:
    sys.modules["altair"] = types.ModuleType("altair")

import pandas as pd  # noqa: E402

os.environ.setdefault("HISTORY_PATH", os.path.join("data", "stock_ranker_history.csv"))
os.environ.pop("DATA_URL_BASE", None)           # the tracker always writes local files

_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "app.py"), encoding="utf-8").read()
_marker = "# ----------------------------------------------------------------- sidebar"
ns = {"__name__": "hyperdrift_app"}
exec(compile(_src[:_src.index(_marker)], "app.py", "exec"), ns)   # functions only, no UI


def main():
    loop = "--loop" in sys.argv
    if "--parlays" in sys.argv:                     # the 15-minute job: parlays only (fast, no stock downloads)
        ns["parlay_step"]()
        return 0
    while True:
        errs = ns["all_steps"]()
        try:
            ns["parlay_step"]()                      # daily pass also grades any parlays still open
        except Exception as e:
            print(f"parlay_step error: {e!r}", flush=True)
        for p in (ns["HIST_PATH"], ns["EXP_PATH"], ns["LIVE_PATH"], ns["LIVE_TRADES_PATH"], ns["PARLAY_PATH"]):
            n = len(pd.read_csv(p)) if os.path.exists(p) else 0
            print(f"{p}: {n} rows", flush=True)
        if not loop:
            return 1 if errs else 0
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
