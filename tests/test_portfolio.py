from collections import Counter

import pandas as pd

from models.portfolio import build_spread_portfolio, qb_plan

TEAMS = [("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "DDD"), ("DDD", "CCC"), ("EEE", "FFF"), ("FFF", "EEE")]


def _pool():
    pool, n = [], 0
    for i, (team, opp) in enumerate(TEAMS):
        def add(pos, salary, pts):
            nonlocal n
            n += 1
            pool.append({"player_id": f"p{n}", "name": f"{team}-{pos}{n}", "position": pos, "salary": salary,
                         "team": team, "opponent": opp, "points": pts})
        add("QB", 6000 + 100 * i, 20 + i)
        for k in range(3):
            add("RB", 5000 + 300 * k, 12 + k + i * 0.3)
        for k in range(4):
            add("WR", 4500 + 400 * k, 11 + k + i * 0.2)
        for k in range(2):
            add("TE", 3500 + 500 * k, 8 + k)
        add("DST", 2500 + 100 * i, 7 + i * 0.1)
    return pool


def test_qb_plan_takes_both_qbs_of_the_best_games_first():
    pool = _pool()
    ranked = pd.DataFrame([{"home": "EEE", "away": "FFF"}, {"home": "AAA", "away": "BBB"}, {"home": "CCC", "away": "DDD"}])
    plan = qb_plan(ranked, pool, 4, top_games=2)
    teams = [next(p["team"] for p in pool if p["player_id"] == q) for q in plan]
    assert teams == ["FFF", "EEE", "BBB", "AAA"]  # best game first, better-projected QB first within it


def test_spread_portfolio_caps_every_player_and_uses_a_different_qb_each_lineup():
    pool = _pool()
    ranked = pd.DataFrame([{"home": a, "away": b} for a, b in (("EEE", "FFF"), ("AAA", "BBB"), ("CCC", "DDD"))])
    plan = qb_plan(ranked, pool, 6, top_games=3)
    lineups, skipped = build_spread_portfolio(pool, plan, max_player_share=0.5, max_dst_share=0.34)
    assert not skipped and len(lineups) == 6
    counts = Counter(p["player_id"] for lu in lineups for _, p in lu["roster"])
    positions = {p["player_id"]: p["position"] for p in pool}
    assert all(c <= (2 if positions[pid] == "DST" else 3) for pid, c in counts.items() if positions[pid] != "QB")
    assert len({p["player_id"] for lu in lineups for s, p in lu["roster"] if s == "QB"}) == 6
