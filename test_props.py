"""Unit tests for props.py.   python -m unittest tests.test_props -v   (from the repo root)"""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import props as P  # noqa: E402

NOW = pd.Timestamp("2026-10-26 12:00", tz="America/New_York")


def rows_for(player, stat, line, books, p_over=0.5, vig=0.045, start="2026-10-27T00:30:00Z", alt_dk=()):
    """Synthetic prop rows: each book posts the same line with a vig; DraftKings also posts alternate overs."""
    rows = []
    for b in books:
        po = p_over
        for side, pr in (("over", po), ("under", 1 - po)):
            q = pr + vig / 2
            am = -100 * q / (1 - q) if q > .5 else 100 * (1 - q) / q
            rows.append(dict(sport="basketball_nba", event_id="e1", home="H", away="A", commence=start, book=b,
                             stat=stat, alt=False, player=player, side=side, line=line, price=round(am)))
    for ln, price in alt_dk:
        rows.append(dict(sport="basketball_nba", event_id="e1", home="H", away="A", commence=start, book="draftkings",
                         stat=stat, alt=True, player=player, side="over", line=ln, price=price))
    return rows


class TestMath(unittest.TestCase):
    def test_devig_sums_to_one(self):
        a, b = P.devig2(-110, -110)
        self.assertAlmostEqual(a, .5)
        a, b = P.devig2(-200, 170)
        self.assertAlmostEqual(a + b, 1.0)
        self.assertGreater(a, .6)

    def test_fit_roundtrip(self):
        for stat in P.STATS:
            for line, p in ((24.5, .55), (6.5, .4), (1.5, .62), (250.5, .5)):
                mu = P.fit_mu(stat, line, p)
                self.assertAlmostEqual(P.p_side(stat, mu, line, "over"), p, places=3, msg=f"{stat} {line}")

    def test_over_under_complement_on_half_lines(self):
        for stat in P.STATS:
            mu = P.fit_mu(stat, 20.5, .5)
            self.assertAlmostEqual(P.p_side(stat, mu, 20.5, "over") + P.p_side(stat, mu, 20.5, "under"), 1.0, places=6)

    def test_alt_lines_monotone(self):
        mu = P.fit_mu("player_points", 24.5, .5)
        ps = [P.p_side("player_points", mu, l, "over") for l in (14.5, 19.5, 24.5, 29.5)]
        self.assertEqual(ps, sorted(ps, reverse=True))
        self.assertGreater(ps[0], .8)


class TestLegs(unittest.TestCase):
    def test_alt_leg_priced_and_haircut(self):
        r = rows_for("Star Guy", "player_points", 24.5, ["draftkings", "fanduel", "betmgm", "caesars"],
                     alt_dk=[(14.5, -900), (19.5, -300), (29.5, 350)])
        legs = P.build_prop_legs(pd.DataFrame(r), NOW)
        by = {l["line"]: l for l in legs if l["side"] == "over"}
        self.assertIn(19.5, by)
        mu = P.fit_mu("player_points", 24.5, .5)
        raw = P.p_side("player_points", mu, 19.5, "over")
        self.assertLess(by[19.5]["p"], raw)                    # haircut applied
        self.assertLessEqual(by[14.5]["p"], P.P_CAP)
        self.assertEqual(by[19.5]["market"], "prop")
        self.assertIn("Star Guy Over 19.5 Pts", by[19.5]["label"])

    def test_needs_min_books(self):
        r = rows_for("Thin", "player_points", 20.5, ["draftkings", "fanduel"])
        self.assertEqual(P.build_prop_legs(pd.DataFrame(r), NOW), [])

    def test_suspect_edge_dropped(self):
        # DK posts +400 on an alt line the consensus says is ~50%: a stale line, not a gift
        r = rows_for("Stale", "player_points", 24.5, ["draftkings", "fanduel", "betmgm"], alt_dk=[(24.5, 400)])
        legs = P.build_prop_legs(pd.DataFrame(r), NOW)
        self.assertFalse([l for l in legs if l["price"] == 400])

    def test_started_games_skipped(self):
        r = rows_for("Late", "player_points", 20.5, ["draftkings", "fanduel", "betmgm"], start="2026-10-26T15:00:00Z")
        self.assertEqual(P.build_prop_legs(pd.DataFrame(r), NOW + pd.Timedelta(hours=5)), [])


def mk_leg(g, p, dec, name="x", market="prop"):
    return dict(game_key=g, p=p, dec=dec, ev1=p * dec - 1, market=market, side="over", line=1.5, player=name,
                label=name, start="2026-10-27T00:30:00+00:00", sport="basketball_nba", home="H", away="A", price=-150)


class TestSlate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.legs = []
        for g in range(12):
            for k in range(4):
                p = float(rng.uniform(.45, .9))
                dec = 1 / (p + 0.04)                       # ~4% vig
                self.legs.append(mk_leg(f"g{g}", p, dec, f"p{g}_{k}"))
        self.slate = P.build_slate(self.legs)

    def test_counts_and_bands(self):
        hits = [s for s in self.slate if s["tier"] == "Hit"]
        longs = [s for s in self.slate if s["tier"] == "Longshot"]
        self.assertEqual((len(hits), len(longs)), (5, 5))
        for s in hits:
            self.assertTrue(2.0 <= s["dec"] <= 4.0, s["dec"])
        for s in longs:
            self.assertTrue(5.0 <= s["dec"] <= 10.0, s["dec"])

    def test_one_leg_per_game_and_math(self):
        for s in self.slate:
            games = [l["game_key"] for l in s["legs"]]
            self.assertEqual(len(games), len(set(games)))
            self.assertGreaterEqual(len(games), 2)
            self.assertAlmostEqual(s["hit"], float(np.prod([l["p"] for l in s["legs"]])))
            self.assertAlmostEqual(s["dec"], float(np.prod([l["dec"] for l in s["legs"]])))

    def test_leg_reuse_capped(self):
        use = {}
        for s in self.slate:
            for l in s["legs"]:
                use[P._lid(l)] = use.get(P._lid(l), 0) + 1
        self.assertLessEqual(max(use.values()), 3)          # 2 per tier, 3 across both tiers

    def test_hits_more_likely_than_longshots(self):
        h = np.mean([s["hit"] for s in self.slate if s["tier"] == "Hit"])
        l = np.mean([s["hit"] for s in self.slate if s["tier"] == "Longshot"])
        self.assertGreater(h, l)

    def test_probability_ceiling_is_one_over_payout(self):
        # on ~4% vig odds no 2x parlay can be "almost 100%" likely: p <= ~1/dec
        for s in self.slate:
            self.assertLess(s["hit"], 1.0 / s["dec"] + 1e-9)

    def test_best_in_band_first(self):
        hits = [s["hit"] for s in self.slate if s["tier"] == "Hit"]
        self.assertEqual(hits, sorted(hits, reverse=True))

    def test_too_few_legs_returns_less_not_error(self):
        s = P.build_slate(self.legs[:3])
        self.assertLessEqual(len(s), 10)


class TestGrading(unittest.TestCase):
    BOX_NBA = {"boxscore": {"players": [{"team": {}, "statistics": [{
        "labels": ["MIN", "PTS", "FG", "3PT", "REB", "AST"],
        "athletes": [
            {"athlete": {"displayName": "Luka Dončić"}, "stats": ["36", "31", "11-22", "4-9", "9", "10"]},
            {"athlete": {"displayName": "Bench Guy"}, "didNotPlay": True, "stats": []},
            {"athlete": {"displayName": "Zero Min"}, "stats": ["0", "0", "0-0", "0-0", "0", "0"]}]}]}]}}
    BOX_NFL = {"boxscore": {"players": [{"team": {}, "statistics": [
        {"name": "passing", "labels": ["C/ATT", "YDS", "TD"], "athletes": [
            {"athlete": {"displayName": "Pat Passer"}, "stats": ["22/31", "287", "2"]}]},
        {"name": "rushing", "labels": ["CAR", "YDS", "TD"], "athletes": [
            {"athlete": {"displayName": "Run Back Jr."}, "stats": ["18", "-3", "0"]}]},
        {"name": "receiving", "labels": ["REC", "YDS", "TD"], "athletes": [
            {"athlete": {"displayName": "Wide Out"}, "stats": ["7", "96", "1"]}]}]}]}}

    def leg(self, player, stat, side, line):
        return dict(player=player, stat=stat, side=side, line=line)

    def test_nba_parse_and_grade(self):
        box = P.parse_box("basketball_nba", self.BOX_NBA)
        fin, live = {"state": "post"}, {"state": "in"}
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_points", "over", 29.5), fin, box)[0], "won")
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_threes", "over", 4.5), fin, box)[0], "lost")
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_threes", "over", 3.5), fin, box)[0], "won")
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_rebounds", "under", 9.5), fin, box)[0], "won")
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_points", "over", 35.5), live, box)[0], "live")
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_points", "under", 29.5), live, box)[0], "lost")
        self.assertEqual(P.grade_prop(self.leg("Bench Guy", "player_points", "over", 5.5), fin, box)[0], "push")
        self.assertEqual(P.grade_prop(self.leg("Zero Min", "player_points", "over", 5.5), fin, box)[0], "push")
        self.assertEqual(P.grade_prop(self.leg("Luka Doncic", "player_points", "over", 20.5), {"state": "pre"}, box)[0],
                         "pending")

    def test_nfl_parse_and_grade(self):
        box = P.parse_box("americanfootball_nfl", self.BOX_NFL)
        fin = {"state": "post"}
        self.assertEqual(box["run back"]["rush_yds"], -3.0)                    # suffix stripped, negative kept
        self.assertEqual(P.grade_prop(self.leg("Pat Passer", "player_pass_yds", "over", 274.5), fin, box)[0], "won")
        self.assertEqual(P.grade_prop(self.leg("Run Back", "player_rush_yds", "under", 40.5), fin, box)[0], "won")
        self.assertEqual(P.grade_prop(self.leg("Wide Out", "player_receptions", "over", 6.5), fin, box)[0], "won")
        self.assertEqual(P.grade_prop(self.leg("Wide Out", "player_reception_yds", "over", 96.5), fin, box)[0], "lost")
        self.assertEqual(P.grade_prop(self.leg("Missing Man", "player_receptions", "over", 2.5), fin, box)[0], "push")


if __name__ == "__main__":
    unittest.main()
