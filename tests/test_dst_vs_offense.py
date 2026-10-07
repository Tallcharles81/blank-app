from models.optimizer import build_lineups_from_pool


def _pool():
    # Team A plays team B. B's DST is by far the best DST and A's WR1 the best
    # receiver, so an unconstrained build pairs them.
    rows = [("qbA", "QB", "A", "B", 6000, 20), ("rbA", "RB", "A", "B", 5000, 15), ("rbC", "RB", "C", "D", 5000, 14),
            ("rbD", "RB", "D", "C", 5000, 13), ("wr1A", "WR", "A", "B", 6000, 25), ("wrC", "WR", "C", "D", 5000, 12),
            ("wrD", "WR", "D", "C", 5000, 11), ("wrB", "WR", "B", "A", 4000, 10), ("teC", "TE", "C", "D", 4000, 8),
            ("teD", "TE", "D", "C", 3000, 7), ("dstB", "DST", "B", "A", 3000, 20), ("dstC", "DST", "C", "D", 3000, 5),
            ("dstE", "DST", "E", "F", 3000, 4)]
    return [{"player_id": pid, "name": pid, "position": pos, "team": t, "opponent": o, "salary": sal, "points": pts}
            for pid, pos, t, o, sal, pts in rows]


def _ids(lu):
    return {p["player_id"] for _, p in lu["roster"]}


def test_default_build_can_pair_a_dst_with_the_offense_it_faces():
    lu = build_lineups_from_pool(_pool(), num_lineups=1)[0][0]
    assert {"dstB", "wr1A"} <= _ids(lu)


def test_opt_in_constraint_keeps_dst_away_from_its_opponents():
    lu = build_lineups_from_pool(_pool(), num_lineups=1, avoid_dst_vs_offense=True)[0][0]
    ids = _ids(lu)
    dst = next(p for _, p in lu["roster"] if p["position"] == "DST")
    assert not any(p["team"] == dst["opponent"] for _, p in lu["roster"] if p["position"] != "DST")
    assert "wr1A" in ids  # keeps the better player, drops the conflicting DST
