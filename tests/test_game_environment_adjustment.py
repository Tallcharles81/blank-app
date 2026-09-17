from models.projections import _apply_game_environment_adjustment

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
