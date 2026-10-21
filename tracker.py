"""
Headless tracker: logs our Top 15 + the expert firms' picks every trading day and fills in opens/closes.
No browser or Streamlit server needed, so it can run 24/7 on any always-on machine.

    python tracker.py            # run forever (checks every 5 minutes)
    python tracker.py --once     # run a single check and exit

It reuses the exact functions in app.py, so results match the dashboard. Set HISTORY_PATH to choose
where the CSV files are saved (the experts file is saved next to it).
"""
import os
import sys
import time
import types

MARKER = "# ----------------------------------------------------------------- sidebar"


def _passthru(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    return lambda f: f


# Stand-in for streamlit so app.py's data/scoring functions load without a web server.
_st = types.ModuleType("streamlit")
_st.cache_data = _st.cache_resource = _passthru
_st.set_page_config = lambda **k: None
sys.modules["streamlit"] = _st
sys.modules.setdefault("altair", types.ModuleType("altair"))   # charts are not needed here

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")).read()
if MARKER not in _src:
    raise RuntimeError("app.py layout changed: sidebar marker not found")
core = {"__name__": "tracker_core"}
exec(compile(_src.split(MARKER)[0], "app.py", "exec"), core)


def main():
    print("Tracker started. History file:", core["HIST_PATH"], flush=True)
    while True:
        try:
            errs = core["all_steps"]()
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "check ok" if not errs else f"finished with {errs} error(s)", flush=True)
        except Exception as e:                       # keep running through network hiccups
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "tracker error:", repr(e), flush=True)
        if "--once" in sys.argv:
            break
        time.sleep(5 * 60)


if __name__ == "__main__":
    main()
