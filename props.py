"""
props.py: player-prop model + payout-band slate builder for Hyperdrift (paper tracking only).

Pipeline
  1. fetch_props()      The Odds API event-odds (player props + alternate lines), cached to disk so credits are rationed.
  2. build_prop_legs()  For every player stat: de-vig each book's Over/Under pair, take the multi-book consensus, fit a
                        count/yardage distribution to it, then price EVERY DraftKings offer (main + alternate lines)
                        with a conservative probability haircut. Suspicious "edges" (stale lines) are dropped.
  3. build_slate()      Search leg combinations (one leg per game, so legs are independent) for
                        5 "Hit" parlays paying 2-4x and 5 "Longshot" parlays paying 5-10x, each set ranked by the
                        highest joint hit probability available inside its payout band.
  4. grade_prop()       Grade legs from ESPN box scores (DNP = push, like a sportsbook void).

Nothing here beats the sportsbook's margin by itself: a parlay paying Nx can hit at most ~1/N of the time on fair
odds. The slate reports that honestly (hit chance AND expected value).
"""
import math
import os
import re
import unicodedata
from datetime import timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests

_ND = NormalDist()
BOOKS = "draftkings,fanduel,betmgm,caesars,espnbet,betrivers,fanatics"
DK = "draftkings"

# stat key -> (label, dist, shape).  dist "norm": sd = shape * mean.  dist "nb": variance = mean + mean^2 / shape.
STATS = {
    "player_points":          ("Pts",       "norm", 0.40),
    "player_rebounds":        ("Reb",       "nb",   10.0),
    "player_assists":         ("Ast",       "nb",   8.0),
    "player_threes":          ("3PM",       "nb",   6.0),
    "player_pass_yds":        ("Pass Yds",  "norm", 0.27),
    "player_rush_yds":        ("Rush Yds",  "norm", 0.65),
    "player_reception_yds":   ("Rec Yds",   "norm", 0.65),
    "player_receptions":      ("Rec",       "nb",   8.0),
}
SPORT_MARKETS = {
    "basketball_nba": ["player_points", "player_rebounds", "player_assists", "player_threes"],
    "americanfootball_nfl": ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"],
}
BOX_KEY = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_threes": "3PM",
           "player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds",
           "player_receptions": "rec"}
ESPN = {"basketball_nba": "basketball/nba", "americanfootball_nfl": "football/nfl"}

# model knobs
MIN_BOOKS = 3            # books that must post the same main line before we trust a consensus
HAIRCUT = 0.015          # flat probability haircut on every prop leg (model risk)
DISP_W = 0.5             # plus half the cross-book disagreement (st.dev. of de-vigged probabilities)
P_CAP = 0.90             # never believe a single leg is more than 90% likely (sim: >90% legs were overconfident)
SUSPECT_EV = 0.15        # |model edge| above this vs the DraftKings price is more likely a stale line than a gift
MIN_EV = -0.10           # legs priced worse than -10% EV (heavy vig) are not considered
HIT_BAND, LONG_BAND = (2.0, 4.0), (5.0, 10.0)


# ----------------------------------------------------------------- odds math
def am2dec(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def devig2(a, b):
    """Multiplicative de-vig of a two-way American market -> (p_a, p_b)."""
    pa, pb = 1 / am2dec(a), 1 / am2dec(b)
    return pa / (pa + pb), pb / (pa + pb)


# ----------------------------------------------------------------- distributions
def _nb_cdf(k, mu, r):
    """P(X <= k) for a negative binomial with mean mu and size r (k integer)."""
    if k < 0:
        return 0.0
    p = r / (r + mu)
    lp, lq = math.log(p), math.log(1 - p)
    tot, base = 0.0, math.lgamma(r)
    for i in range(int(k) + 1):
        tot += math.exp(math.lgamma(i + r) - base - math.lgamma(i + 1) + r * lp + i * lq)
    return min(tot, 1.0)


def p_side(stat, mu, line, side):
    """Probability the stat finishes over/under `line` (ties are ignored: sportsbooks push them)."""
    _, dist, shape = STATS[stat]
    lo, hi = math.ceil(line) - 1, math.floor(line) + 1          # under: X <= lo   over: X >= hi
    if dist == "nb":
        under = _nb_cdf(lo, mu, shape)
        over = 1.0 - _nb_cdf(hi - 1, mu, shape)
    else:
        sd = max(shape * mu, 1.0)
        under = _ND.cdf((lo + 0.5 - mu) / sd)
        over = 1.0 - _ND.cdf((hi - 0.5 - mu) / sd)
    return over if side == "over" else under


def fit_mu(stat, line, p_over):
    """Mean that makes P(over line) equal the consensus probability (bisection; P is increasing in the mean)."""
    p_over = min(max(p_over, 0.03), 0.97)
    a, b = 0.05, 4.0 * line + 10.0
    for _ in range(34):
        m = (a + b) / 2
        if p_side(stat, m, line, "over") < p_over:
            a = m
        else:
            b = m
    return (a + b) / 2


# ----------------------------------------------------------------- fetching (rationed)
def _get(url, params, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_event(sport, ev, odds):
    """Event-odds JSON -> long rows."""
    rows = []
    for bk in odds.get("bookmakers", []):
        for m in bk.get("markets", []):
            key = m.get("key", "")
            base = key.replace("_alternate", "")
            if base not in STATS:
                continue
            for o in m.get("outcomes", []):
                side = str(o.get("name", "")).lower()
                if side not in ("over", "under") or o.get("point") is None or not o.get("description"):
                    continue
                rows.append(dict(sport=sport, event_id=ev["id"], home=ev["home_team"], away=ev["away_team"],
                                 commence=ev["commence_time"], book=bk["key"], stat=base, alt=key.endswith("_alternate"),
                                 player=o["description"], side=side, line=float(o["point"]), price=float(o["price"])))
    return rows


COLS = ["sport", "event_id", "home", "away", "commence", "book", "stat", "alt", "player", "side", "line", "price",
        "fetched_at"]


def fetch_props(now, key, cache_path, refresh_min=90, hours_ahead=30, log=print):
    """-> (DataFrame of prop rows, note). Credits cost markets x regions per event, so each event is re-pulled only
    every `refresh_min` minutes (every 20 minutes once it starts within the hour, to catch late scratches)."""
    if not key:
        return pd.DataFrame(columns=COLS), "no ODDS_API_KEY: player props need The Odds API (ESPN has none)"
    try:
        cache = pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame(columns=COLS)
    except Exception:
        cache = pd.DataFrame(columns=COLS)
    cache["event_id"] = cache["event_id"].astype(str)
    nu, calls, errs, keep = pd.Timestamp(now).tz_convert("UTC"), 0, 0, []
    fresh = {}
    for sport, mk in SPORT_MARKETS.items():
        try:
            events = _get(f"https://api.the-odds-api.com/v4/sports/{sport}/events", dict(apiKey=key))
        except Exception as e:
            log(f"props events {sport}: {e!r}")
            errs += 1
            continue
        for ev in events:
            st = pd.Timestamp(ev["commence_time"])
            if st <= nu + timedelta(minutes=2) or st > nu + timedelta(hours=hours_ahead):
                continue
            old = cache[cache.event_id == str(ev["id"])]
            age = ((nu - pd.Timestamp(old.fetched_at.iloc[0])).total_seconds() / 60) if len(old) else 1e9
            ttl = 20 if st <= nu + timedelta(minutes=60) else refresh_min
            if age < ttl:
                continue
            markets = ",".join(mk + [m + "_alternate" for m in mk])
            try:
                odds = _get(f"https://api.the-odds-api.com/v4/sports/{sport}/events/{ev['id']}/odds",
                            dict(apiKey=key, bookmakers=BOOKS, markets=markets, oddsFormat="american",
                                 dateFormat="iso"))
                calls += 1
                rows = parse_event(sport, ev, odds)
                if rows:
                    d = pd.DataFrame(rows)
                    d["fetched_at"] = nu.isoformat()
                    fresh[str(ev["id"])] = d
            except Exception as e:
                errs += 1
                log(f"props odds {sport} {ev.get('id')}: {e!r}")
    parts = [cache[~cache.event_id.isin(fresh)]] + list(fresh.values())
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else \
        pd.DataFrame(columns=COLS)
    if len(out):
        out = out[pd.to_datetime(out.commence, utc=True) > nu]          # drop games that already started
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            out.to_csv(cache_path, index=False)
        except Exception as e:
            log(f"props cache write: {e!r}")
    note = f"The Odds API props: {out.event_id.nunique() if len(out) else 0} game(s) cached, {calls} pull(s) this run"
    return out, note + (f", {errs} error(s)" if errs else "")


# ----------------------------------------------------------------- the model
def _label(player, stat, side, line):
    return f"{player} {'Over' if side == 'over' else 'Under'} {line:g} {STATS[stat][0]}"


def consensus(g):
    """g = main-market rows of one player/stat. -> (line, p_over, dispersion, n_books) or None."""
    pairs = []
    for bk, b in g.groupby("book"):
        for ln, x in b.groupby("line"):
            o, u = x[x.side == "over"], x[x.side == "under"]
            if len(o) and len(u):
                pairs.append((bk, ln, devig2(o.price.iloc[0], u.price.iloc[0])[0]))
    if not pairs:
        return None
    d = pd.DataFrame(pairs, columns=["book", "line", "p"])
    cnt = d.groupby("line").agg(n=("p", "size"), m=("p", "mean"))
    cnt["dist"] = (cnt.m - 0.5).abs()
    ln = cnt.sort_values(["n", "dist"], ascending=[False, True]).index[0]
    s = d[d.line == ln].p
    if len(s) < MIN_BOOKS:
        return None
    return float(ln), float(s.mean()), float(s.std(ddof=0)), int(len(s))


def build_prop_legs(df, now, book=DK):
    """Prop rows -> unified leg dicts (same shape the game-line legs use) priced at `book`."""
    if df is None or len(df) == 0:
        return []
    ET = "America/New_York"
    nu, out = pd.Timestamp(now).tz_convert("UTC"), []
    df = df.copy()
    df["start"] = pd.to_datetime(df.commence, utc=True)
    df["alt"] = df["alt"].astype(str).eq("True")
    for (eid, player, stat), g in df.groupby(["event_id", "player", "stat"]):
        r0 = g.iloc[0]
        if r0.start <= nu + timedelta(minutes=2):
            continue
        c = consensus(g[~g.alt])
        if not c:
            continue
        line0, p0, disp, nb = c
        mu = fit_mu(stat, line0, p0)
        gk = f"{r0.sport}|{r0.away}@{r0.home}|{r0.start.tz_convert(ET).strftime('%Y-%m-%d')}"
        seen = set()
        for _, d in g[g.book == book].iterrows():
            if (d.side, d.line) in seen:
                continue
            seen.add((d.side, d.line))
            dec = am2dec(d.price)
            p = p_side(stat, mu, d.line, d.side)
            p = min(max(p - HAIRCUT - DISP_W * disp, 0.0), P_CAP)
            ev1 = p * dec - 1
            if p < 0.25 or ev1 < MIN_EV or ev1 > SUSPECT_EV or not (-1500 <= d.price <= 1000):
                continue
            out.append(dict(game_key=gk, sport=r0.sport, home=r0.home, away=r0.away, start=r0.start.isoformat(),
                            market="prop", side=d.side, line=float(d.line), price=float(d.price), dec=dec, p=float(p),
                            ev1=float(ev1), label=_label(player, stat, d.side, d.line), player=player, stat=stat,
                            match=f"{r0.away} @ {r0.home}".replace("  ", " "), n_books=nb, mu=round(mu, 2)))
    return out


# ----------------------------------------------------------------- slate builder
def _best_per_game(pool, k):
    by = {}
    for l in sorted(pool, key=lambda x: -x["p"]):
        by.setdefault(l["game_key"], [])
        if len(by[l["game_key"]]) < k:
            by[l["game_key"]].append(l)
    return [l for v in by.values() for l in v]


def _lid(l):
    return (l["game_key"], l.get("player") or "", l["market"], l["side"], l.get("line"))


def _search(pool, band, min_legs, max_legs, keep=3000):
    """DFS over combinations with one leg per game; prune once the payout passes the band."""
    lo, hi = band
    pool = sorted(pool, key=lambda l: -l["p"])
    best = []

    def rec(start, chosen, games, dec, p):
        if len(chosen) >= min_legs and lo <= dec <= hi:
            best.append((p, dec, list(chosen)))
            if len(best) > keep * 4:
                best.sort(key=lambda t: -t[0])
                del best[keep:]
        if len(chosen) >= max_legs:
            return
        for i in range(start, len(pool)):
            l = pool[i]
            if l["game_key"] in games or dec * l["dec"] > hi:
                continue
            chosen.append(l)
            games.add(l["game_key"])
            rec(i + 1, chosen, games, dec * l["dec"], p * l["p"])
            games.discard(l["game_key"])
            chosen.pop()

    rec(0, [], set(), 1.0, 1.0)
    best.sort(key=lambda t: -t[0])
    return best[:keep]


def build_slate(legs, n_hit=5, n_long=5, use_cap=2, pool_n=30):
    """-> list of {tier, legs, dec, hit, ev}. Hits: 2-4x. Longshots: 5-10x. Highest joint probability first."""
    legs = [l for l in legs if l["p"] > 0 and l["ev1"] >= MIN_EV]
    hit_pool = sorted(_best_per_game([l for l in legs if l["p"] >= 0.55 and 1.05 <= l["dec"] <= 2.3], 3),
                      key=lambda l: -l["p"])[:pool_n]
    long_pool = sorted(_best_per_game([l for l in legs if l["p"] >= 0.35 and 1.2 <= l["dec"] <= 4.5], 3),
                       key=lambda l: -l["p"])[:pool_n + 12]
    used_all, out = {}, []
    for tier, pool, band, n, mx in (("Hit", hit_pool, HIT_BAND, n_hit, 6), ("Longshot", long_pool, LONG_BAND, n_long, 7)):
        got, used = 0, {}                                   # per-tier reuse cap, plus a looser cap across tiers
        for p, dec, combo in _search(pool, band, 2, mx) if n > 0 else []:
            ids = [_lid(l) for l in combo]
            if got >= n:
                break
            if any(used.get(i, 0) >= use_cap or used_all.get(i, 0) >= use_cap + 2 for i in ids):
                continue
            out.append(dict(tier=tier, legs=combo, dec=dec, hit=p, ev=p * dec - 1))
            got += 1
            for i in ids:
                used[i] = used.get(i, 0) + 1
                used_all[i] = used_all.get(i, 0) + 1
    return out


# ----------------------------------------------------------------- grading (ESPN box scores)
def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def _num(x):
    try:
        return float(str(x).split("-")[0]) if "-" in str(x) and str(x)[0] != "-" else float(x)
    except (TypeError, ValueError):
        return None


def parse_box(sport, data):
    """ESPN summary JSON -> {normalized name: {stat key: value, 'played': bool}}."""
    out = {}
    for team in (data.get("boxscore", {}) or {}).get("players", []):
        for cat in team.get("statistics", []):
            labels = [str(x).upper() for x in cat.get("labels", [])]
            cname = str(cat.get("name", "")).lower()
            for a in cat.get("athletes", []):
                nm = norm_name((a.get("athlete") or {}).get("displayName", ""))
                if not nm:
                    continue
                st = a.get("stats") or []
                row = out.setdefault(nm, {"played": False})
                vals = dict(zip(labels, st))
                if sport == "basketball_nba":
                    dnp = bool(a.get("didNotPlay")) or not st
                    mins = _num(vals.get("MIN"))
                    row["played"] = (not dnp) and (mins is None or mins > 0)
                    for lab, key in (("PTS", "PTS"), ("REB", "REB"), ("AST", "AST")):
                        if lab in vals:
                            row[key] = _num(vals[lab]) or 0.0
                    if "3PT" in vals:
                        row["3PM"] = _num(vals["3PT"]) or 0.0      # "made-attempted"
                else:
                    row["played"] = True
                    if cname == "passing" and "YDS" in vals:
                        row["pass_yds"] = _num(vals["YDS"]) or 0.0
                    elif cname == "rushing" and "YDS" in vals:
                        row["rush_yds"] = _num(vals["YDS"]) or 0.0
                    elif cname == "receiving":
                        if "YDS" in vals:
                            row["rec_yds"] = _num(vals["YDS"]) or 0.0
                        if "REC" in vals:
                            row["rec"] = _num(vals["REC"]) or 0.0
    return out


def fetch_box(sport, espn_id):
    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[sport]}/summary",
                         params={"event": espn_id}, timeout=20)
        return parse_box(sport, r.json())
    except Exception:
        return {}


def grade_prop(leg, sc, box):
    """-> (status, result text). status in won/lost/push/live/pending. DNP => push (void)."""
    if not sc or sc.get("state") == "pre":
        return "pending", ""
    final = sc.get("state") == "post"
    row = box.get(norm_name(leg["player"]))
    key = BOX_KEY.get(leg["stat"])
    unit = STATS[leg["stat"]][0]
    if row is None or key not in row:
        if final:
            return "push", f"Final: {leg['player']} not in the box score (bet voided)"
        return "live", ""
    if final and not row.get("played", True):
        return "push", f"Final: {leg['player']} did not play (bet voided)"
    v, line = row[key], leg["line"]
    txt = f"{'Final' if final else 'Now'}: {v:g} {unit} (line {line:g})"
    if leg["side"] == "over":
        if v > line:
            return "won", txt                       # counting stats only go up
        return ("lost" if final else "live"), txt
    if v > line:
        return "lost", txt
    return ("won" if final else "live"), txt


# ----------------------------------------------------------------- calibration report
def calibration(parlays, bins=(0, .3, .45, .6, .75, .9, 1.01)):
    """Graded prop legs: model probability vs how often they really won. -> DataFrame."""
    import json
    rows = []
    for _, r in parlays.iterrows():
        try:
            for l in json.loads(r.legs):
                if l.get("market") == "prop" and l.get("status") in ("won", "lost"):
                    rows.append((l["p"], 1.0 if l["status"] == "won" else 0.0, l["dec"]))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows, columns=["p", "won", "dec"])
    d["bin"] = pd.cut(d.p, list(bins), right=False)
    g = d.groupby("bin", observed=True).agg(legs=("won", "size"), model_p=("p", "mean"), actual=("won", "mean"))
    return g.round(3)
