import pytest
from sqlalchemy import text

from models.projections import SHOWDOWN_CAPTAIN_MULTIPLIER, generate_projections

# Regression coverage for a real bug: models/projections.py's own generated
# proj_floor/median/ceiling are computed from a player's real historical
# fantasy-point history, entirely independent of DraftKings' own salary/
# AvgPointsPerGame columns (which DO already bake in the real 1.5x Captain
# multiplier - see models/optimizer.py's _solve_showdown comment). Before
# this fix, a Showdown player's CPT and FLEX row got the IDENTICAL
# generated projection despite the CPT row costing 1.5x the salary for the
# SAME real outcome - caught for real building the first actual GPP
# lineups from a real Showdown CSV, where the solver dumped a $1,500 punt
# play at Captain (paying nothing extra for a real stud there looked like a
# bad trade, since both rows "projected" the same).

TEST_SHOWDOWN_PROJECTIONS_SLATE_ID = "TEST_SHOWDOWN_PROJECTIONS"

# Jameis Winston - real, extensive player_weekly_stats history already
# relied on elsewhere in this test suite (tests/test_showdown_gates.py).
WINSTON_CPT_ID = "SDPROJ_CPT_WINSTON"
WINSTON_FLEX_ID = "SDPROJ_FLEX_WINSTON"


@pytest.fixture
def showdown_projection_slate(engine):
    rows = [
        {"player_id": WINSTON_CPT_ID, "name": "Jameis Winston", "position": "CPT", "team": "NYG", "salary": 13500},
        {"player_id": WINSTON_FLEX_ID, "name": "Jameis Winston", "position": "FLEX", "team": "NYG", "salary": 9000},
    ]
    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team)
                    VALUES (:slate_id, :player_id, :name, :position, :salary, :team)
                    ON CONFLICT (slate_id, player_id) DO NOTHING
                    """
                ),
                {"slate_id": TEST_SHOWDOWN_PROJECTIONS_SLATE_ID, **row},
            )
    try:
        yield TEST_SHOWDOWN_PROJECTIONS_SLATE_ID
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM projections WHERE slate_id = :slate_id"),
                {"slate_id": TEST_SHOWDOWN_PROJECTIONS_SLATE_ID},
            )
            conn.execute(
                text("DELETE FROM slate_player_pool WHERE slate_id = :slate_id"),
                {"slate_id": TEST_SHOWDOWN_PROJECTIONS_SLATE_ID},
            )


def test_generate_projections_scales_captain_row_by_1_5x(engine, showdown_projection_slate):
    generate_projections(showdown_projection_slate, engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT player_id, proj_floor, proj_median, proj_ceiling FROM projections "
                "WHERE slate_id = :slate_id"
            ),
            {"slate_id": showdown_projection_slate},
        ).mappings().fetchall()
    by_id = {r["player_id"]: r for r in rows}

    cpt = by_id[WINSTON_CPT_ID]
    flex = by_id[WINSTON_FLEX_ID]
    for field in ("proj_floor", "proj_median", "proj_ceiling"):
        assert float(cpt[field]) == pytest.approx(float(flex[field]) * SHOWDOWN_CAPTAIN_MULTIPLIER, abs=0.02)
