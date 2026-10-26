"""
Monte-Carlo stress test of the prop model + slate builder in a synthetic market where we KNOW the truth.

  python tests/sim_calibration.py [days]

Each simulated day: 10 games x 2 players x 4 stats. A player's true mean is hidden. Books quote the player with a noisy
view of that mean (so the consensus is imperfect) and a 4.5% vig on main lines, ~8% on DraftKings alternate lines. We then
run the REAL code (build_prop_legs -> build_slate), settle every leg against a draw from the true distribution, and ask:

  1. Is the model's leg probability calibrated (does 'model_p' match how often legs really win)?
  2. Do the slate's predicted hit rates match realised hit rates?
  3. What do $10 paper stakes earn?  (On vigged prices with no real edge the honest answer is: they lose the vig.)

Run at several levels of book noise to see how fast calibration degrades when the market is less efficient than assumed.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import props as P  # noqa: E402

NOW = pd.Timestamp("2026-10-26 12:00", tz="America/New_York")
START = "2026-10-27T00:30:00Z"
STATS = ["player_points", "player_rebounds", "player_assists", "player_threes"]
MU_RANGE = {"player_points": (12, 32), "player_rebounds": (3, 12), "player_assists": (2, 9), "player_threes": (0.8, 4)}
BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "betrivers"]


def am(q):
    q = min(max(q, .02), .98)
    return float(round(-100 * q / (1 - q)) if q > .5 else round(100 * (1 - q) / q))


def draw(stat, mu, rng):
    _, dist, shape = P.STATS[stat]
    if dist == "nb":
        return rng.negative_binomial(shape, shape / (shape + mu))
    return max(0.0, round(rng.normal(mu, shape * mu)))


def one_day(rng, book_noise, vig_main=.045, vig_alt=.08):
    rows, truth = [], {}
    for g in range(10):
        eid = f"e{g}"
        for k in range(2):
            for stat in STATS:
                pl = f"G{g}P{k}"
                mu = rng.uniform(*MU_RANGE[stat])
                truth[(eid, pl, stat)] = mu
                lo = rng.uniform(.9, 1.0) if stat != "player_threes" else 1.0
                for b in BOOKS:
                    mb = mu * np.exp(rng.normal(0, book_noise))                # this book's (noisy) view
                    line = float(np.floor(mb) + .5)
                    po = P.p_side(stat, mb, line, "over")
                    for side, pr in (("over", po), ("under", 1 - po)):
                        rows.append(dict(sport="basketball_nba", event_id=eid, home=f"H{g}", away=f"A{g}", commence=START,
                                         book=b, stat=stat, alt=False, player=pl, side=side, line=line,
                                         price=am(pr + vig_main / 2)))
                mb = mu * np.exp(rng.normal(0, book_noise))
                for ln in np.floor(mb) + .5 + np.array([-6, -4, -2, 2, 4]) * (0.35 if stat != "player_points" else 1):
                    ln = float(np.floor(ln) + .5)
                    if ln > 0:
                        rows.append(dict(sport="basketball_nba", event_id=eid, home=f"H{g}", away=f"A{g}", commence=START,
                                         book="draftkings", stat=stat, alt=True, player=pl, side="over", line=ln,
                                         price=am(P.p_side(stat, mb, ln, "over") + vig_alt / 2)))
    return pd.DataFrame(rows), truth


def run(days, book_noise, seed=7):
    rng = np.random.default_rng(seed)
    leg_rows, slate_rows = [], []
    for _ in range(days):
        df, truth = one_day(rng, book_noise)
        legs = P.build_prop_legs(df, NOW)
        outcome = {}
        for l in legs:
            eid = l["game_key"].split("@H")[1].split("|")[0]
            eid = f"e{eid}"
            key = (eid, l["player"], l["stat"])
            if key not in outcome:
                outcome[key] = draw(l["stat"], truth[key], rng)
            v = outcome[key]
            won = (v > l["line"]) if l["side"] == "over" else (v < l["line"])
            l["_won"] = bool(won)
            leg_rows.append((l["p"], l["_won"], l["dec"]))
        for s in P.build_slate(legs):
            won = all(l["_won"] for l in s["legs"])
            slate_rows.append((s["tier"], s["hit"], s["dec"], won, len(s["legs"])))
    L = pd.DataFrame(leg_rows, columns=["p", "won", "dec"])
    S = pd.DataFrame(slate_rows, columns=["tier", "p", "dec", "won", "n"])
    return L, S


def report(L, S, noise):
    print(f"\n=== book noise sigma={noise:.2f} | legs={len(L):,} | parlays={len(S):,}")
    L["bin"] = pd.cut(L.p, [0, .3, .45, .6, .75, .9, 1.0], right=False)
    g = L.groupby("bin", observed=True).agg(n=("won", "size"), model_p=("p", "mean"), actual=("won", "mean"))
    g["gap"] = (g.actual - g.model_p).round(3)
    print(g.round(3).to_string())
    for tier in ("Hit", "Longshot"):
        t = S[S.tier == tier]
        if t.empty:
            continue
        stake = 10.0
        profit = np.where(t.won, stake * (t.dec - 1), -stake).sum()
        print(f"{tier:9s} n={len(t):4d}  avg payout {t.dec.mean():.2f}x  avg legs {t.n.mean():.1f}  "
              f"predicted hit {t.p.mean():.3f}  realised hit {t.won.mean():.3f}  ROI {profit / (stake * len(t)) * 100:+.1f}%")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    t0 = time.time()
    for noise in (0.03, 0.06, 0.10):
        L, S = run(days, noise)
        report(L, S, noise)
    print(f"\n({time.time() - t0:.0f}s)")
