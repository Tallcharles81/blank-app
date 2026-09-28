import csv

import pytest
from sqlalchemy import text

from data.lineup_tracking import (
    MIN_SLATES_FOR_VERDICT,
    compare_build_groups,
    register_built_lineups,
    score_builds_against_contest,
)

SLATE = "TEST_TRACK_CLASSIC"
SD_SLATE = "TEST_TRACK_SHOWDOWN"
SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]


def _lineup(names, slots=SLOTS):
    return {"roster": [(s, {"player_id": f"id_{n}", "name": n}) for s, n in zip(slots, names)]}


def _write_standings(path, entries, players):
    """entries: [(rank, entry_name, points, lineup_text)]; players:
    [(name, roster_position, fpts)] - DK's bolted-on side table."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "EntryId", "EntryName", "TimeRemaining", "Points", "Lineup", "", "Player", "Roster Position", "%Drafted", "FPTS"])
        for i in range(max(len(entries), len(players))):
            e = entries[i] if i < len(entries) else ("", "", "", "", "")
            p = players[i] if i < len(players) else ("", "", "")
            w.writerow([e[0], i + 1 if e[0] != "" else "", e[1], 0, e[2], e[3], "", p[0], p[1], "1%" if p[0] else "", p[2]])


@pytest.fixture
def cleanup(engine):
    yield
    with engine.begin() as conn:
        for t in ("built_lineups", "lineup_contest_results"):
            conn.execute(text(f"DELETE FROM {t} WHERE slate_id LIKE 'TEST_TRACK_%'"))


OPT = ["Q1", "R1", "R2", "W1", "W2", "W3", "T1", "W4", "D1"]
SIM = ["Q2", "R1", "R3", "W1", "W5", "W6", "T2", "R4", "D2"]
PLAYERS = [(n, pos, v) for n, pos, v in [
    ("Q1", "QB", 20), ("Q2", "QB", 30), ("R1", "RB", 15), ("R2", "RB", 5), ("R3", "RB", 12), ("R4", "FLEX", 8),
    ("W1", "WR", 18), ("W2", "WR", 4), ("W3", "WR", 6), ("W4", "FLEX", 10), ("W5", "WR", 14), ("W6", "WR", 9),
    ("T1", "TE", 3), ("T2", "TE", 11), ("D1", "DST", 5), ("D2", "DST", 7)]]
OPT_POINTS = 20 + 15 + 5 + 18 + 4 + 6 + 3 + 10 + 5  # 86
SIM_POINTS = 30 + 15 + 12 + 18 + 14 + 9 + 11 + 8 + 7  # 124


def _lineup_text(names):
    return " ".join(f"{s} {n}" for s, n in zip(SLOTS, names))


def test_scores_each_group_against_the_real_field_and_matches_entries(engine, tmp_path, cleanup):
    register_built_lineups(SLATE, "b1", {"optimizer": [_lineup(OPT)], "simulator": [_lineup(SIM)]}, engine=engine)
    swapped = [*SIM[:5], "W2", *SIM[6:]]  # late swap of one WR
    entries = [
        (1, "someone", 150.0, _lineup_text(OPT)),
        (2, "tallcharles81 (1/2)", SIM_POINTS, _lineup_text(SIM)),
        (3, "tallcharles81 (2/2)", 119.0, _lineup_text(swapped)),
        (4, "someone2", 100.0, _lineup_text(OPT)),
        (5, "someone3", 50.0, _lineup_text(OPT)),
    ]
    path = tmp_path / "contest-standings-1.csv"
    _write_standings(path, entries, PLAYERS)

    r = score_builds_against_contest(str(path), SLATE, "TESTC1", engine=engine)
    assert r["field_size"] == 5
    assert r["by_group"]["optimizer"]["avg_points"] == pytest.approx(OPT_POINTS)
    assert r["by_group"]["simulator"]["avg_points"] == pytest.approx(SIM_POINTS)
    # 124 is beaten only by 150 -> rank 2; 86 is beaten by 150, 124, 119, 100 -> rank 5
    assert r["by_group"]["simulator"]["best_rank"] == 2
    assert r["by_group"]["optimizer"]["best_rank"] == 5
    sources = [e["source"] for e in r["entries"]]
    assert sources[0].startswith("simulator")
    assert sources[1].startswith("edited from simulator")

    with engine.connect() as conn:
        entered = dict(conn.execute(text(
            "SELECT build_group, times_entered FROM lineup_contest_results WHERE contest_id = 'TESTC1'")).fetchall())
    assert entered == {"simulator": 1, "optimizer": 0}


def test_showdown_captain_uses_its_own_multiplied_row(engine, tmp_path, cleanup):
    slots = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
    names = ["A", "B", "C", "D", "E", "F"]
    register_built_lineups(SD_SLATE, "sd", {"optimizer": [_lineup(names, slots)]}, engine=engine)
    players = [("A", "CPT", 30.0), ("A", "FLEX", 20.0)] + [(n, "FLEX", 5.0) for n in names[1:]]
    path = tmp_path / "contest-standings-2.csv"
    _write_standings(path, [(1, "x", 10.0, "CPT A FLEX B FLEX C FLEX D FLEX E FLEX F")], players)
    r = score_builds_against_contest(str(path), SD_SLATE, "TESTC2", engine=engine)
    assert r["by_group"]["optimizer"]["avg_points"] == pytest.approx(30 + 5 * 5)


def test_compare_reports_insufficient_sample_until_enough_slates(engine, cleanup):
    with engine.begin() as conn:
        for i in range(MIN_SLATES_FOR_VERDICT):
            for group, pct in (("optimizer", 0.50), ("simulator", 0.30)):
                conn.execute(text(
                    """INSERT INTO lineup_contest_results (contest_id, slate_id, build_id, lineup_id, build_group, points,
                       finish_rank, field_size, finish_pct) VALUES (:c, :s, 'b', :l, :g, 100, 1, 100, :p)"""),
                    {"c": f"TC{i}", "s": f"TEST_TRACK_S{i}", "l": f"{group}-01", "g": group, "p": pct + 0.01 * i})
    few = compare_build_groups(slate_ids=[f"TEST_TRACK_S{i}" for i in range(2)], engine=engine)
    assert few["verdict"].startswith("INSUFFICIENT SAMPLE")
    full = compare_build_groups(slate_ids=[f"TEST_TRACK_S{i}" for i in range(MIN_SLATES_FOR_VERDICT)], engine=engine)
    assert full["n_slates"] == MIN_SLATES_FOR_VERDICT
    assert full["mean_finish_pct_diff"] == pytest.approx(-0.20)
    assert full["slates_group_b_better"] == MIN_SLATES_FOR_VERDICT
