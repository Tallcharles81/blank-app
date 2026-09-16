from models.calibration import (
    MIN_PAIRS_FOR_FITTED_CORRELATION,
    fit_real_correlation_matrix,
    run_game_environment_backtest_comparison,
    run_hard_exclude_backtest,
)

# These validate that the backtest MACHINERY works correctly (real as-of
# data, no crashes, sane structure, real relationships hold) - not the
# specific numeric results from the full multi-season run disclosed in
# models/projections.py (GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT) and
# data/pre_lock_check.py (HARD_EXCLUDE_MAX_SNAP_PCT/MIN_SNAP_PCT_FOR_BENCHED),
# which will drift slightly as more real weeks accumulate. Scoped to a
# single season to stay fast in a regular test run.


def test_hard_exclude_backtest_runs_and_shows_excluded_players_score_lower(engine):
    result = run_hard_exclude_backtest("dk_thu_mon_2026_09_17", seasons={2026}, engine=engine)

    assert result["weeks_evaluated"] >= 1
    assert result["would_be_excluded_player_weeks"] > 0
    assert result["not_excluded_player_weeks"] > 0
    # The whole premise of a hard, silent exclusion: players it catches must
    # score meaningfully lower on average than players it doesn't.
    assert result["avg_actual_score_if_excluded"] < result["avg_actual_score_if_not_excluded"]
    assert 0.0 <= result["miss_rate"] <= 1.0


def test_game_environment_backtest_runs_and_reports_tercile_structure(engine):
    result = run_game_environment_backtest_comparison("dk_thu_mon_2026_09_17", seasons={2026}, engine=engine)

    assert result["weeks_evaluated"] >= 1
    assert result["players_evaluated"] > 0
    assert "pooled" in result
    assert 0.0 <= result["pooled"]["baseline_p90_hit_rate"]["rate"] <= 1.0
    assert 0.0 <= result["pooled"]["adjusted_p90_hit_rate"]["rate"] <= 1.0
    # Terciles are only reported once there's enough real data to make them
    # meaningful (see the function's own tercile_size >= 20 guard) - present
    # here since a real week has hundreds of eligible players.
    if "high_implied_total_tercile" in result:
        assert result["high_implied_total_tercile"]["avg_gap"] > 0
        assert result["low_implied_total_tercile"]["avg_gap"] < 0


def test_fit_real_correlation_matrix_runs_and_shows_real_relationships(engine):
    # Full multi-season run (not scoped to one season - this needs the whole
    # real sample to clear MIN_PAIRS_FOR_FITTED_CORRELATION comfortably); the
    # specific numeric results are disclosed in models/simulation.py's
    # QB_PASS_CATCHER_CORR/etc. comment, not re-asserted exactly here since
    # they'll drift as more real seasons accumulate - same reasoning as the
    # two tests above.
    result = fit_real_correlation_matrix(engine=engine)

    expected_relationships = {"qb_pass_catcher", "qb_rb", "same_team", "bring_back", "dst_vs_opponent_offense"}
    assert set(result.keys()) == expected_relationships

    for relationship, stats in result.items():
        assert stats["sample_size"] >= MIN_PAIRS_FOR_FITTED_CORRELATION, (
            f"{relationship} has too little real data to trust its fitted correlation"
        )
        assert -1.0 <= stats["correlation"] <= 1.0

    # Real, structural relationships that should hold regardless of exactly
    # how much data has accumulated: a QB correlates positively and clearly
    # with his own pass-catchers (they score off the same real completions),
    # and a defense anti-correlates with its opponent's real offensive output.
    assert result["qb_pass_catcher"]["correlation"] > 0.1
    assert result["dst_vs_opponent_offense"]["correlation"] < -0.2


def test_fit_real_correlation_matrix_can_be_scoped_to_a_season(engine):
    # seasons filtering must actually narrow the real sample, not silently
    # no-op and return the full-history numbers regardless of what's passed.
    full = fit_real_correlation_matrix(engine=engine)
    scoped = fit_real_correlation_matrix(seasons={2026}, engine=engine)

    assert scoped["same_team"]["sample_size"] < full["same_team"]["sample_size"]
