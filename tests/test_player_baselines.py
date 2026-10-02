import json

import pytest
from sqlalchemy import text

from models.player_baselines import (
    BASELINE_UPDATE_WEIGHT,
    blend_ladders,
    smooth_ladder,
    update_baselines,
)

LADDER = {"10": 6.0, "25": 10.0, "50": 15.0, "75": 21.0, "90": 28.0}
GSIS = "TEST_BASELINE_PLAYER"


def test_smoothing_moves_the_median_part_way_and_keeps_the_shape():
    new = smooth_ladder(LADDER, 25.0)
    expected_median = (1 - BASELINE_UPDATE_WEIGHT) * 15.0 + BASELINE_UPDATE_WEIGHT * 25.0
    assert new["50"] == pytest.approx(expected_median)
    assert new["90"] / new["50"] == pytest.approx(28.0 / 15.0, abs=0.01)


def test_smoothing_a_zero_median_shifts_instead_of_dividing_by_zero():
    new = smooth_ladder({"10": 0, "25": 0, "50": 0, "75": 1, "90": 3}, 10.0)
    assert new["50"] == pytest.approx(BASELINE_UPDATE_WEIGHT * 10.0)


def test_blend_is_an_even_mix_by_default():
    other = {k: v + 4 for k, v in LADDER.items()}
    assert blend_ladders(LADDER, other) == {k: v + 2 for k, v in LADDER.items()}


@pytest.fixture
def baseline_player(engine):
    with engine.begin() as conn:
        for week, pts in ((1, 20.0), (2, None), (3, 5.0)):
            if pts is None:
                continue  # week 2: didn't play
            conn.execute(text("INSERT INTO player_weekly_stats (player_id, player_name, position, team, season, week, fantasy_points_ppr) "
                              "VALUES (:g, 'Test Baseline', 'RB', 'ZZ', 2099, :w, :p)"), {"g": GSIS, "w": week, "p": pts})
        conn.execute(text("INSERT INTO player_baselines (gsis_id, name, team, position, ladder, source, seeded_season, seeded_week, "
                          "updated_season, updated_week) VALUES (:g, 'Test Baseline', 'ZZ', 'RB', :l, 'test', 2099, 1, 2099, 0)"),
                     {"g": GSIS, "l": json.dumps(LADDER)})
    yield
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM player_weekly_stats WHERE player_id = :g"), {"g": GSIS})
        conn.execute(text("DELETE FROM player_baselines WHERE gsis_id = :g"), {"g": GSIS})


def _baseline(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT ladder, games_applied, updated_week FROM player_baselines WHERE gsis_id = :g"), {"g": GSIS}).one()


def test_update_applies_each_played_game_once_in_order(engine, baseline_player):
    update_baselines((2099, 3), engine=engine, gsis_ids=[GSIS])
    ladder, games, week = _baseline(engine)
    expected = smooth_ladder(smooth_ladder(LADDER, 20.0), 5.0)
    assert ladder["50"] == pytest.approx(expected["50"]) and games == 2 and week == 3
    update_baselines((2099, 3), engine=engine, gsis_ids=[GSIS])  # re-running the same week changes nothing
    assert _baseline(engine)[1] == 2


def test_newer_seed_replaces_an_older_baseline_only_when_asked(engine, baseline_player, tmp_path):
    from models.player_baselines import seed_from_sabersim
    with engine.begin() as conn:  # make the fixture player resolvable by name like a real one
        conn.execute(text("UPDATE player_baselines SET seeded_week = 1 WHERE gsis_id = :g"), {"g": GSIS})
    csv_path = tmp_path / "ss.csv"
    csv_path.write_text("DFS ID,Name,Pos,Team,Opp,Status,Salary,SS Proj,Adj Own,dk_25_percentile,dk_50_percentile,"
                        "dk_75_percentile,dk_85_percentile,dk_95_percentile,dk_99_percentile\n"
                        "1,Test Baseline,RB,ZZ,YY,,5000,30,10,25,30,35,38,42,50\n")
    seed_from_sabersim(str(csv_path), 2099, {"ZZ": 4}, engine=engine)
    assert _baseline(engine)[0]["50"] == 15.0  # default: existing baseline kept
    seed_from_sabersim(str(csv_path), 2099, {"ZZ": 4}, engine=engine, replace_older=True)
    ladder, games, week = _baseline(engine)
    assert ladder["50"] == 30.0 and games == 0 and week == 3


def test_a_game_imported_after_an_earlier_update_of_its_week_still_applies(engine, baseline_player):
    # Thursday's game is in, this player's Sunday game isn't yet: updating
    # through week 4 must not mark week 4 done for him.
    update_baselines((2099, 4), engine=engine, gsis_ids=[GSIS])
    assert _baseline(engine)[1:] == (2, 3)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO player_weekly_stats (player_id, player_name, position, team, season, week, fantasy_points_ppr) "
                          "VALUES (:g, 'Test Baseline', 'RB', 'ZZ', 2099, 4, 30.0)"), {"g": GSIS})
    update_baselines((2099, 4), engine=engine, gsis_ids=[GSIS])
    ladder, games, week = _baseline(engine)
    expected = smooth_ladder(smooth_ladder(smooth_ladder(LADDER, 20.0), 5.0), 30.0)
    assert ladder["50"] == pytest.approx(expected["50"]) and games == 3 and week == 4
