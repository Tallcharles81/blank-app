from data.nflverse_fetch import fetch_schedules
from models.calibration import (
    MIN_PAIRS_FOR_FITTED_CORRELATION,
    _classify_situational_games,
    fit_real_correlation_matrix,
    run_game_environment_backtest_comparison,
    run_hard_exclude_backtest,
    run_situational_backtest,
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


# --- run_situational_backtest / _classify_situational_games ---------------


def test_classify_situational_games_finds_real_examples_of_all_three():
    situational = _classify_situational_games(fetch_schedules())

    short_rest = [k for k, v in situational.items() if v["short_rest_both"]]
    rematches = [k for k, v in situational.items() if v["divisional_rematch"]]
    cross_country = [k for k, v in situational.items() if v["cross_country_early"]]

    assert short_rest, "real short-rest-into-short-rest games must exist (Thursday games are common)"
    assert rematches, "real divisional rematches must exist (every division pair plays twice a season)"
    assert cross_country, "real Pacific-away/Eastern-home 1pm ET games must exist"

    # cross_country_early is scoped to the TRAVELING team only - the real
    # Eastern home team in that same game must never be tagged (it isn't
    # the one crossing time zones).
    cross_country_teams = {team for team, _, _ in cross_country}
    assert cross_country_teams <= {"SEA", "SF", "LAR", "LAC", "LV"}


def test_run_situational_backtest_runs_and_reports_all_three_categories(engine):
    # Scoped to 2023 (a full 18-week real season), not 2026 like the other
    # backtests in this file - 2026 only has one real week of data so far,
    # which can't contain a real divisional rematch (needs two meetings) or
    # a meaningful short-rest sample, and would make every assertion below
    # vacuous (None vs None).
    result = run_situational_backtest("dk_thu_mon_2026_09_17", seasons={2023}, engine=engine)

    assert set(result.keys()) == {"short_rest_both", "divisional_rematch", "cross_country_early"}
    for category, stats in result.items():
        assert stats["n_in_situation"] > 0, f"{category} found zero real examples in 2023 - suspicious"
        assert stats["n_complement"] > 0
        assert 0.0 <= stats["p90_hit_rate_in_situation"] <= 1.0
        assert 0.0 <= stats["p90_hit_rate_complement"] <= 1.0
        # A real p-value must be a real probability, not a leftover of a
        # broken significance calculation.
        if stats["hit_rate_p_value"] is not None:
            assert 0.0 <= stats["hit_rate_p_value"] <= 1.0
        if stats["bias_p_value"] is not None:
            assert 0.0 <= stats["bias_p_value"] <= 1.0
