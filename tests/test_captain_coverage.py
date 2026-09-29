from collections import Counter

import pytest
from sqlalchemy import text

from models.optimizer import (
    build_lineups_from_pool,
    select_with_captain_coverage,
    showdown_qb_captain_ids,
)


def _pool():
    # One Showdown game: QB "Star" projects far above QB "Backup", so a
    # pure-projection build never captains Backup.
    base = [
        ("Star", "QB", "A", 10000, 20.0),
        ("Backup", "QB", "B", 8400, 12.0),
        ("RB1", "RB", "A", 9000, 15.0),
        ("RB2", "RB", "B", 7000, 11.0),
        ("WR1", "WR", "A", 7500, 12.0),
        ("WR2", "WR", "B", 6000, 10.0),
        ("WR3", "WR", "A", 4000, 7.0),
        ("TE1", "TE", "B", 3000, 5.0),
        ("K1", "K", "A", 4500, 7.5),
        ("DST", "DST", "B", 4000, 6.0),
    ]
    pool = []
    for name, _, team, salary, pts in base:
        opp = "B" if team == "A" else "A"
        pool.append({"player_id": f"F_{name}", "name": name, "position": "FLEX", "team": team, "opponent": opp,
                     "salary": salary, "points": pts})
        pool.append({"player_id": f"C_{name}", "name": name, "position": "CPT", "team": team, "opponent": opp,
                     "salary": int(salary * 1.5), "points": pts * 1.5})
    return pool


def _captains(lineups):
    return Counter(lu["roster"][0][1]["player_id"] for lu in lineups)


def test_projection_only_build_never_captains_the_backup_qb():
    lineups, _ = build_lineups_from_pool(_pool(), num_lineups=6, min_uniques=1)
    assert _captains(lineups)["C_Backup"] == 0


def test_each_required_captain_gets_its_lineups():
    lineups, _ = build_lineups_from_pool(
        _pool(), num_lineups=6, min_uniques=1, min_captain_lineups={"C_Star": 2, "C_Backup": 2}
    )
    caps = _captains(lineups)
    assert len(lineups) == 6
    assert caps["C_Star"] >= 2 and caps["C_Backup"] >= 2


def test_two_captains_owed_the_final_slots_are_scheduled_not_both_locked():
    # 4 lineups, 2 + 2 owed: min_exposure-style locking would force both
    # captains into one lineup and fail; scheduling fills exactly.
    lineups, _ = build_lineups_from_pool(
        _pool(), num_lineups=4, min_uniques=1, min_captain_lineups={"C_Star": 2, "C_Backup": 2}
    )
    assert len(lineups) == 4
    assert _captains(lineups) == Counter({"C_Star": 2, "C_Backup": 2})


def test_asking_for_more_captain_slots_than_lineups_is_an_error():
    with pytest.raises(ValueError):
        build_lineups_from_pool(_pool(), num_lineups=3, min_captain_lineups={"C_Star": 2, "C_Backup": 2})


def test_select_with_captain_coverage_keeps_rank_order_and_reserves_captains():
    def lu(cpt, tag):
        return {"roster": [("CPT", {"player_id": cpt}), ("FLEX", {"player_id": tag})]}
    ranked = [lu("C_Star", "a"), lu("C_Star", "b"), lu("C_Star", "c"), lu("C_Backup", "d"), lu("C_Backup", "e")]
    picked = select_with_captain_coverage(ranked, 3, {"C_Backup": 1})
    assert [p["roster"][1][1]["player_id"] for p in picked] == ["a", "b", "d"]


def test_showdown_qb_captain_ids_finds_real_qbs_by_game_history(engine):
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT player_id, name, position, team, opponent, salary FROM slate_player_pool "
            "WHERE slate_id = 'dk_showdown_phi_chi_2026_09_28' AND position = 'CPT'"
        )).mappings().fetchall()
    if not rows:
        pytest.skip("PHI@CHI Showdown slate not loaded")
    by_name = {r["name"]: r["player_id"] for r in rows}
    qb_ids = set(showdown_qb_captain_ids([dict(r) for r in rows], engine))
    assert by_name["Jalen Hurts"] in qb_ids and by_name["Tyson Bagent"] in qb_ids
    assert by_name["D'Andre Swift"] not in qb_ids and by_name["DeVonta Smith"] not in qb_ids
