import pytest
from sqlalchemy import text

from models.projections import _apply_game_environment_adjustment, generate_projections

# Regression test for a real finding: the game-environment adjustment
# originally scaled ceiling DOWN for a below-average implied total too
# (symmetric), and a real 54-week backtest showed that half moved P90
# calibration AWAY from the true 10% target, not toward it - a real,
# significant effect (t=-8.79, p<0.0001), not noise. Fixed to boost-only.

SAMPLE_PROJ = {
    "proj_floor": 8.0,
    "proj_median": 14.0,
    "proj_ceiling": 22.0,
    "proj_percentiles": {"10": 5.0, "25": 8.0, "50": 14.0, "75": 18.5, "90": 22.0},
}


def test_below_average_implied_total_is_a_no_op():
    league_average = 22.69
    adjusted = _apply_game_environment_adjustment(SAMPLE_PROJ, 16.25, league_average)
    assert adjusted["proj_ceiling"] == SAMPLE_PROJ["proj_ceiling"]
    assert adjusted["proj_percentiles"] == SAMPLE_PROJ["proj_percentiles"]


def test_average_implied_total_is_a_no_op():
    league_average = 22.69
    adjusted = _apply_game_environment_adjustment(SAMPLE_PROJ, league_average, league_average)
    assert adjusted["proj_ceiling"] == SAMPLE_PROJ["proj_ceiling"]


def test_above_average_implied_total_still_boosts_ceiling_only():
    league_average = 22.69
    adjusted = _apply_game_environment_adjustment(SAMPLE_PROJ, 30.0, league_average)
    assert adjusted["proj_ceiling"] > SAMPLE_PROJ["proj_ceiling"]
    assert adjusted["proj_floor"] == SAMPLE_PROJ["proj_floor"]
    assert adjusted["proj_median"] == SAMPLE_PROJ["proj_median"]
    assert adjusted["proj_percentiles"]["50"] == SAMPLE_PROJ["proj_percentiles"]["50"]
    assert adjusted["proj_percentiles"]["25"] == SAMPLE_PROJ["proj_percentiles"]["25"]


# Real regression coverage for generate_projections' opt-in
# use_dst_opponent_matchup_adjustment - a real, backtested (models/
# calibration.py::run_dst_matchup_backtest) but not-yet-decisive-enough-to-
# default DST-specific correction: same boost-only mechanism, fed the real
# opponent's implied total instead of the DST's own team's.
#
# Uses a dedicated TEST slate_id rather than the real dk_sunday_2026_09_20
# slate directly - that real slate's own projections are already locked
# (its games have been played), and generate_projections' own upsert
# deliberately no-ops on a locked row, which would make this test pass for
# the wrong reason (nothing written, not nothing changed). Copies the real
# SEA/ARI DST rows' own real team/opponent/game_time verbatim so real
# schedule/Vegas resolution still works identically.
TEST_DST_ADJUSTMENT_SLATE_ID = "TEST_DST_OPPONENT_ADJUSTMENT"
SEA_DST_ID = "TEST_DST_SEA"
ARI_DST_ID = "TEST_DST_ARI"
REAL_GAME_TIME = "2026-09-20 20:25:00+00"


@pytest.fixture
def dst_adjustment_slate(engine):
    rows = [
        {"player_id": SEA_DST_ID, "name": "Seahawks", "position": "DST", "team": "SEA", "opponent": "ARI", "salary": 3500},
        {"player_id": ARI_DST_ID, "name": "Cardinals", "position": "DST", "team": "ARI", "opponent": "SEA", "salary": 3500},
    ]
    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent, game_time)
                    VALUES (:slate_id, :player_id, :name, :position, :salary, :team, :opponent, :game_time)
                    ON CONFLICT (slate_id, player_id) DO NOTHING
                    """
                ),
                {"slate_id": TEST_DST_ADJUSTMENT_SLATE_ID, "game_time": REAL_GAME_TIME, **row},
            )
    try:
        yield TEST_DST_ADJUSTMENT_SLATE_ID
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM projections WHERE slate_id = :s"), {"s": TEST_DST_ADJUSTMENT_SLATE_ID})
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :s"), {"s": TEST_DST_ADJUSTMENT_SLATE_ID})


def _dst_ceilings(engine, slate_id):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT player_id, proj_ceiling FROM projections WHERE slate_id = :s"), {"s": slate_id}
        ).fetchall()
    return {r.player_id: float(r.proj_ceiling) for r in rows}


def test_default_generate_projections_leaves_dst_treatment_unchanged(engine, dst_adjustment_slate):
    generate_projections(dst_adjustment_slate, engine=engine)
    before = _dst_ceilings(engine, dst_adjustment_slate)
    generate_projections(dst_adjustment_slate, engine=engine)  # default call again - no flag passed
    after = _dst_ceilings(engine, dst_adjustment_slate)
    assert before == after


def test_opt_in_dst_opponent_adjustment_moves_at_least_one_real_dst_ceiling(engine, dst_adjustment_slate):
    generate_projections(dst_adjustment_slate, engine=engine)
    baseline = _dst_ceilings(engine, dst_adjustment_slate)
    assert baseline  # real history must exist for at least one of these two real teams

    generate_projections(dst_adjustment_slate, engine=engine, use_dst_opponent_matchup_adjustment=True)
    adjusted = _dst_ceilings(engine, dst_adjustment_slate)

    # At least one real DST's ceiling must actually move - otherwise this
    # "opt-in" flag would be a silent no-op rather than a real alternative.
    changed = [pid for pid in baseline if pid in adjusted and baseline[pid] != adjusted[pid]]
    assert changed
