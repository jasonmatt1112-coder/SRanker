"""
End-to-end test of the parlay pipeline with a FAKE Odds API and FAKE ESPN (no network, no keys needed).

  python tests/test_integration.py

Phase 1 (pre-game):  parlay_step() pulls props + lines, builds the 10-parlay slate (5 Hit, 5 Longshot).
Phase 2 (after tip): parlay_step() locks the slate and grades every leg (game lines from scores, props from box scores).
"""
import json
import os
import shutil
import sys
import tempfile
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ET = ZoneInfo("America/New_York")

# ---- load app.py's functions the same way tracker.py does (Streamlit stand-in)
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
sys.modules.setdefault("altair", types.ModuleType("altair"))
try:
    import yfinance  # noqa: F401
except ImportError:                                    # parlay code never touches it
    sys.modules["yfinance"] = types.ModuleType("yfinance")

TMP = tempfile.mkdtemp()
os.environ["HISTORY_PATH"] = os.path.join(TMP, "stock_ranker_history.csv")
os.environ["ODDS_API_KEY"] = "fake"
os.environ.pop("DATA_URL_BASE", None)
src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
marker = "# ----------------------------------------------------------------- sidebar"
ns = {"__name__": "hyperdrift_app"}
exec(compile(src[:src.index(marker)], "app.py", "exec"), ns)
import props as P  # noqa: E402

# ---- fake world: 8 NBA games tonight, each with 2 stars, 5 books
rng = np.random.default_rng(11)
TEAMS = [("Boston Celtics", "Miami Heat"), ("New York Knicks", "Chicago Bulls"), ("Denver Nuggets", "Utah Jazz"),
         ("Dallas Mavericks", "Phoenix Suns"), ("Orlando Magic", "Atlanta Hawks"), ("Toronto Raptors", "Detroit Pistons"),
         ("Cleveland Cavaliers", "Indiana Pacers"), ("Los Angeles Lakers", "Sacramento Kings")]
NOW = datetime(2026, 10, 26, 15, 0, tzinfo=ET)
START = datetime(2026, 10, 26, 19, 30, tzinfo=ET)
BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "betrivers"]
EVENTS, STARS, TRUTH = [], {}, {}


def am(q):
    q = min(max(q, .02), .98)
    return round(-100 * q / (1 - q)) if q > .5 else round(100 * (1 - q) / q)


for i, (h, a) in enumerate(TEAMS):
    EVENTS.append(dict(id=f"ev{i}", home_team=h, away_team=a, commence_time=START.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")))
    for k in range(2):
        name = f"{h.split()[-1]} Star{k}"
        mu = float(rng.uniform(16, 30))
        STARS[(f"ev{i}", name)] = mu


def odds_for(ev):
    bms = {b: [] for b in BOOKS}
    for (eid, name), mu in STARS.items():
        if eid != ev["id"]:
            continue
        line = float(np.floor(mu) + .5)
        p_over = P.p_side("player_points", mu, line, "over")
        for b in BOOKS:
            jitter = rng.normal(0, .01)
            outs = []
            for side, pr in (("Over", p_over + jitter), ("Under", 1 - p_over - jitter)):
                outs.append(dict(name=side, description=name, price=am(pr + .025), point=line))
            bms[b].append(dict(key="player_points", outcomes=outs))
        alts = []
        for ln in (line - 10, line - 5, line + 5):
            alts.append(dict(name="Over", description=name, price=am(P.p_side("player_points", mu, ln, "over") + .04), point=ln))
        bms["draftkings"].append(dict(key="player_points_alternate", outcomes=alts))
    return dict(id=ev["id"], bookmakers=[dict(key=b, markets=m) for b, m in bms.items()])


STATE = {"phase": "pre"}


class Resp:
    def __init__(self, j): self.j = j
    def json(self): return self.j
    def raise_for_status(self): pass


def fake_get(url, params=None, timeout=None, **kw):
    params = params or {}
    if "the-odds-api.com" in url:
        if url.endswith("/events"):
            return Resp(EVENTS if "nba" in url else [])
        if "/events/" in url and url.endswith("/odds"):
            eid = url.split("/events/")[1].split("/")[0]
            return Resp(odds_for(next(e for e in EVENTS if e["id"] == eid)))
        if url.endswith("/odds"):                       # main game lines
            if "nba" not in url:
                return Resp([])
            g = []
            for e in EVENTS:
                bks = []
                for b in BOOKS:
                    sp = rng.choice([-3.5, -5.5, -7.5])
                    bks.append(dict(key=b, markets=[
                        dict(key="h2h", outcomes=[dict(name=e["home_team"], price=-170), dict(name=e["away_team"], price=145)]),
                        dict(key="spreads", outcomes=[dict(name=e["home_team"], price=-110, point=-4.5), dict(name=e["away_team"], price=-110, point=4.5)]),
                        dict(key="totals", outcomes=[dict(name="Over", price=-110, point=224.5), dict(name="Under", price=-110, point=224.5)])]))
                g.append(dict(home_team=e["home_team"], away_team=e["away_team"], commence_time=e["commence_time"], bookmakers=bks))
            return Resp(g)
    if "espn.com" in url and url.endswith("/scoreboard"):
        if "basketball/nba" not in url:
            return Resp({"events": []})
        evs = []
        for i, (h, a) in enumerate(TEAMS):
            state = STATE["phase"]
            sc = {"pre": (0, 0), "post": (112, 104)}[state]
            evs.append(dict(id=f"espn{i}", date=START.astimezone(ZoneInfo("UTC")).isoformat(), status={"type": {"state": state, "shortDetail": ""}},
                            competitions=[dict(competitors=[
                                dict(homeAway="home", score=str(sc[0]), team=dict(displayName=h, shortDisplayName=h.split()[-1])),
                                dict(homeAway="away", score=str(sc[1]), team=dict(displayName=a, shortDisplayName=a.split()[-1]))])]))
        return Resp({"events": evs})
    if "espn.com" in url and url.endswith("/summary"):
        i = int(params["event"].replace("espn", ""))
        h = TEAMS[i][0]
        ath = []
        for k in range(2):
            name = f"{h.split()[-1]} Star{k}"
            mu = STARS[(f"ev{i}", name)]
            pts = float(np.random.default_rng(i * 10 + k).normal(mu, .4 * mu))
            ath.append(dict(athlete=dict(displayName=name), stats=["34", str(max(int(pts), 0)), "8-16", "2-5", "5", "5"]))
        return Resp({"boxscore": {"players": [dict(team={}, statistics=[dict(labels=["MIN", "PTS", "FG", "3PT", "REB", "AST"], athletes=ath)])]}})
    raise AssertionError(f"unexpected url {url}")


ns["requests"].get = fake_get
P.requests.get = fake_get


class FakeDT(datetime):
    cur = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.cur.astimezone(tz) if tz else cls.cur


ns["datetime"] = FakeDT
P_run = lambda: ns["parlay_step"]()


def load():
    d = pd.read_csv(ns["PARLAY_PATH"], dtype={"parlay_id": str, "batch_date": str})
    return d[d.batch_date == "2026-10-26"]


def main():
    # ---------------- phase 1
    ns["_with_seed"] = lambda P: P if P is not None and len(P) else pd.DataFrame(columns=ns["PARLAY_COLS"])
    P_run()
    d = load()
    print(d.tier.value_counts().to_dict())
    assert (d.tier == "Hit").sum() == 5 and (d.tier == "Longshot").sum() == 5, d.tier.value_counts()
    for _, r in d.iterrows():
        legs = json.loads(r.legs)
        games = [l["game_key"] for l in legs]
        assert len(set(games)) == len(games)
        if r.tier == "Hit":
            assert 2.0 <= r.dec_odds <= 4.0, r.dec_odds
        else:
            assert 5.0 <= r.dec_odds <= 10.0, r.dec_odds
        assert abs(np.prod([l["dec"] for l in legs]) - r.dec_odds) < 1e-3
        assert abs(np.prod([l["p"] for l in legs]) - r.hit_prob) < 1e-3
    props_n = sum(1 for _, r in d.iterrows() for l in json.loads(r.legs) if l["market"] == "prop")
    print("prop legs inside the slate:", props_n)
    assert props_n > 0
    print(d[["tier", "n_legs", "dec_odds", "hit_prob", "ev"]].round(3).to_string())
    meta = pd.read_csv(ns["PARLAY_META"])
    print(meta.iloc[0].to_dict())
    assert os.path.exists(ns["PROPS_PATH"])
    ids1 = set(d.parlay_id)

    # ---------------- phase 1b: re-run inside the cache window = no new Odds-API prop pulls, same slate
    calls = {"n": 0}
    orig = fake_get

    def counting(url, params=None, **kw):
        if "/events/" in url and url.endswith("/odds"):
            calls["n"] += 1
        return orig(url, params, **kw)
    ns["requests"].get = counting
    P.requests.get = counting
    FakeDT.cur = NOW + timedelta(minutes=15)
    P_run()
    assert calls["n"] == 0, f"props should come from the cache, got {calls['n']} pulls"
    ns["requests"].get = fake_get
    P.requests.get = fake_get

    # ---------------- phase 2: games finished
    STATE["phase"] = "post"
    FakeDT.cur = START + timedelta(hours=3)
    P_run()
    d = load()
    assert (d.locked.astype(str) == "True").all()
    assert d.status.isin(["won", "lost", "push"]).all(), d.status.value_counts()
    print(d[["tier", "dec_odds", "status", "profit"]].round(2).to_string())
    for _, r in d.iterrows():
        legs = json.loads(r.legs)
        sts = [l["status"] for l in legs]
        if r.status == "won":
            assert all(s in ("won", "push") for s in sts)
            assert r.profit > 0
        if r.status == "lost":
            assert "lost" in sts and r.profit == -10.0
    # slate stays at 10 after lock (no extra parlays minted after kickoff)
    assert len(d) == 10, len(d)
    legs = [l for _, r in d.iterrows() for l in json.loads(r.legs) if l["market"] == "prop"]
    print("graded prop legs:", len(legs), {s: sum(1 for l in legs if l["status"] == s) for s in ("won", "lost", "push")})
    print(legs[0]["label"], "->", legs[0]["status"], "|", legs[0]["result"])
    print(P.calibration(d))
    print("\nINTEGRATION OK")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
