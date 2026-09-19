from collections import defaultdict

import pytest

from data.nflverse_fetch import fetch_schedules
from models.calibration import (
    MIN_PAIRS_FOR_FITTED_CORRELATION,
    _classify_situational_games,
    _real_divisional_rematch_torch_margins,
    _team_qb_availability_by_season,
    _teammate_qb_starter_unavailable,
    fit_real_correlation_matrix,
    run_divisional_rematch_torch_backtest,
    run_game_environment_backtest_comparison,
    run_game_environment_p80_hit_rate_backtest,
    run_hard_exclude_backtest,
    run_opportunity_score_backtest,
    run_playing_time_floor_backtest,
    run_salary_left_backtest,
    run_shrinkage_backtest,
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

    # Real, named false-exclusion examples - a miss_rate percentage alone
    # doesn't tell you whether a "miss" is a real difference-maker or a
    # replacement-level scrub who scraped 8.1 once. Every entry must be a
    # real player above the real threshold, sorted worst-case-first.
    assert len(result["top_misses"]) <= 15
    for miss in result["top_misses"]:
        assert miss["actual_score"] > result["meaningful_score_threshold"]
    scores = [m["actual_score"] for m in result["top_misses"]]
    assert scores == sorted(scores, reverse=True)


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


def test_game_environment_p80_backtest_runs_and_reports_tercile_structure(engine):
    result = run_game_environment_p80_hit_rate_backtest("dk_thu_mon_2026_09_17", seasons={2026}, engine=engine)

    assert result["weeks_evaluated"] >= 1
    assert result["players_evaluated"] > 0
    assert result["true_target_hit_rate"] == 0.20
    assert "pooled" in result
    assert 0.0 <= result["pooled"]["baseline_p80_hit_rate"]["rate"] <= 1.0
    assert 0.0 <= result["pooled"]["adjusted_p80_hit_rate"]["rate"] <= 1.0
    if "high_implied_total_tercile" in result:
        assert result["high_implied_total_tercile"]["avg_gap"] > 0
        assert result["low_implied_total_tercile"]["avg_gap"] < 0


def test_game_environment_p80_backtest_low_tercile_shows_zero_change(engine):
    # The adjustment is boost-only (gap <= 0 is a no-op) - the low tercile's
    # P80 hit rate must be IDENTICAL under baseline and adjusted, the same
    # real property already confirmed for P90 in models/projections.py's
    # GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT disclosure comment. Needs the
    # full real dataset (not seasons={2026}) to clear the >=20-per-tercile
    # reporting threshold reliably.
    result = run_game_environment_p80_hit_rate_backtest("dk_thu_mon_2026_09_17", engine=engine)
    assert "low_implied_total_tercile" in result
    low = result["low_implied_total_tercile"]
    assert low["baseline_p80_hit_rate"]["hits"] == low["adjusted_p80_hit_rate"]["hits"]
    assert low["baseline_minus_adjusted_hit_rate_diff"] == 0.0


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


# --- run_divisional_rematch_torch_backtest ---------------------------------
# A sharper refinement of divisional_rematch above: within real rematches,
# does it matter whether the offense scored well above its own season norm
# against this same rival in meeting 1?


def test_real_divisional_rematch_torch_margins_finds_real_observations():
    observations = _real_divisional_rematch_torch_margins(fetch_schedules())
    assert observations, "real divisional rematches with a computable margin must exist"

    for obs in observations[:5]:
        assert obs["week2"] > obs["week1"]  # the rematch is chronologically after meeting 1
        assert obs["team"] != obs["rival"]

    # Both teams in a real rematch pair get their own independent margin -
    # team A's offense-vs-team-B's-defense is a different real question
    # from team B's offense-vs-team-A's-defense, even in the same two games.
    pairs = {(o["season"], frozenset({o["team"], o["rival"]})) for o in observations}
    teams_per_pair = defaultdict(set)
    for o in observations:
        teams_per_pair[(o["season"], frozenset({o["team"], o["rival"]}))].add(o["team"])
    both_sides_present = sum(1 for pair in pairs if len(teams_per_pair[pair]) == 2)
    assert both_sides_present > 0


def test_run_divisional_rematch_torch_backtest_runs_and_reports_real_numbers(engine):
    # Scoped to 2023 for the same reason as the situational backtest above -
    # a full real season is needed for a meaningful rematch sample.
    result = run_divisional_rematch_torch_backtest("dk_thu_mon_2026_09_17", seasons={2023}, engine=engine)

    assert "note" not in result, f"expected a real comparison, got: {result}"
    assert result["n_torched_player_weeks"] >= 20
    assert result["n_did_not_torch_player_weeks"] >= 20
    assert 0.0 <= result["p90_hit_rate_torched"] <= 1.0
    assert 0.0 <= result["p90_hit_rate_did_not_torch"] <= 1.0
    if result["hit_rate_p_value"] is not None:
        assert 0.0 <= result["hit_rate_p_value"] <= 1.0
    if result["bias_p_value"] is not None:
        assert 0.0 <= result["bias_p_value"] <= 1.0


def test_run_divisional_rematch_torch_backtest_reports_too_small_a_sample_honestly(engine):
    # 2026 has only one real week of data so far - no real rematch (which
    # needs two meetings) can exist yet. Must report that honestly via the
    # "note" field rather than silently returning a comparison built on an
    # empty or near-empty sample.
    with pytest.raises(ValueError, match="No real divisional rematch observations"):
        run_divisional_rematch_torch_backtest("dk_thu_mon_2026_09_17", seasons={2026}, engine=engine)


# --- _team_qb_availability_by_season / _teammate_qb_starter_unavailable ----
# The teammate-starter-unavailable check that feeds run_hard_exclude_backtest
# above (via data/pre_lock_check.py's _would_be_hard_excluded) - sourced from
# nflverse's real weekly roster feed, not player_weekly_stats (which is
# blind to a QB who recorded zero real box-score participation that week -
# see _team_qb_availability_by_season's own docstring). These pin down real,
# specific, already-investigated cases so a future change to the recency/
# established-starter thresholds can't silently regress any of them.

# Real gsis_ids (from player_weekly_stats), not re-looked-up per test run -
# stable real nflverse identifiers, not names that could collide/typo.
_DREW_LOCK = "00-0035704"
_DESHAUN_WATSON = "00-0033537"
_JAMEIS_WINSTON = "00-0031503"
_RUSSELL_WILSON = "00-0029263"
_JAXSON_DART = "00-0040691"
_MARCUS_MARIOTA = "00-0032268"


def test_teammate_qb_starter_unavailable_true_when_real_starter_released():
    # Real 2024 case: Daniel Jones was NYG's real QB1 for most of the season,
    # then released and signed to Minnesota's practice squad before week 17 -
    # by week 17 he no longer has ANY roster row under team "NYG" at all,
    # which is real, unambiguous evidence Drew Lock's week-17 start wasn't a
    # random appearance. This is the exact real case that exposed
    # player_weekly_stats' blind spot (Jones has zero row there for week 17
    # either way, injured or not, since he recorded no real stats for NYG).
    _, season_team_qbs, status_by_team_gsis = _team_qb_availability_by_season(2024)
    assert _teammate_qb_starter_unavailable(
        "NYG", _DREW_LOCK, 17, season_team_qbs, status_by_team_gsis
    ) is True


def test_teammate_qb_starter_unavailable_true_when_real_starter_on_ir():
    # Real 2024 case: Deshaun Watson tore his Achilles and was placed on
    # real injured reserve ("RES") well before Jameis Winston's week-11 spot
    # start - a persistent roster-level exit, not a single stale INA week.
    _, season_team_qbs, status_by_team_gsis = _team_qb_availability_by_season(2024)
    assert _teammate_qb_starter_unavailable(
        "CLE", _JAMEIS_WINSTON, 11, season_team_qbs, status_by_team_gsis
    ) is True


def test_teammate_qb_starter_unavailable_false_for_real_healthy_benching():
    # Real 2025 case: Russell Wilson was real-ACT (healthy, active gameday
    # roster) the same week Jaxson Dart took over as NYG's starter - a real
    # performance-based benching, confirmed directly against nflverse's raw
    # injuries data (Wilson never appears with any real injury designation
    # around this change). No real injury/roster signal can or should catch
    # this - the check must correctly report False here, not just find SOME
    # excuse to flip the exclusion off.
    _, season_team_qbs, status_by_team_gsis = _team_qb_availability_by_season(2025)
    assert _teammate_qb_starter_unavailable(
        "NYG", _JAXSON_DART, 7, season_team_qbs, status_by_team_gsis
    ) is False


def test_teammate_qb_starter_unavailable_ignores_routinely_inactive_third_string_qb():
    # Real 2025 case, same week as above: NYG's real roster also carried
    # Jameis Winston as a third QB who was real-INA in every week here
    # except one (the normal, meaningless-by-itself inactivity of a real
    # emergency third arm). Without the established-starter threshold, his
    # routine inactivity alone would wrongly "explain" Dart's start even
    # though the actual competitor (Wilson) was healthy - this is the exact
    # false positive this test guards against regressing.
    _, season_team_qbs, status_by_team_gsis = _team_qb_availability_by_season(2025)
    assert _JAMEIS_WINSTON in season_team_qbs["NYG"]  # sanity: he's a real roster QB this season
    assert _teammate_qb_starter_unavailable(
        "NYG", _JAXSON_DART, 7, season_team_qbs, status_by_team_gsis
    ) is False


def test_teammate_qb_starter_unavailable_ignores_stale_bench_demotion():
    # Real 2024 case: Washington's Jeff Driskel was real-ACT (an established
    # real backup) for weeks 1-4, then real-INA for the rest of the season
    # including week 18 - but that's a permanent, months-old bench
    # demotion, not fresh week-18 news. The real Jayden Daniels was himself
    # real-ACT that same week 18 (just given a lighter snap share in an
    # already-clinched game) - Driskel's stale inactivity must not be
    # mistaken for Daniels being unavailable.
    _, season_team_qbs, status_by_team_gsis = _team_qb_availability_by_season(2024)
    assert _teammate_qb_starter_unavailable(
        "WAS", _MARCUS_MARIOTA, 18, season_team_qbs, status_by_team_gsis
    ) is False


# --- run_playing_time_floor_backtest ----------------------------------------


def test_run_salary_left_backtest_validates_the_cash_mode_salary_floor(engine):
    # Real regression coverage for the exact real numbers this backtest
    # produced against dk_sunday_2026_09_20 (see models/optimizer.py's
    # generate_cash_lineups docstring, updated from this same real run):
    # cash mode's risk-averse objective leaves real cap unspent and a real
    # negative salary-left/actual-score correlation, while the ceiling
    # objective shows neither - and applying the shipped min_salary_
    # fraction=0.95 default produces a real, significant improvement.
    result = run_salary_left_backtest("dk_sunday_2026_09_20", seasons={2023, 2024, 2025}, engine=engine)

    assert result["weeks_considered"] >= 1
    assert result["ceiling_objective"]["sample_size"] > 0
    assert result["cash_objective"]["sample_size"] > 0

    # The real premise: an unconstrained risk-averse objective leaves
    # meaningfully more cap on the table than a pure-ceiling objective, and
    # doing so is real bad news for it (negative correlation with real
    # actual score) - the ceiling objective shows no such relationship.
    assert result["cash_objective"]["avg_salary_left"] > result["ceiling_objective"]["avg_salary_left"]
    assert result["cash_objective"]["correlation"] < -0.3
    assert abs(result["ceiling_objective"]["correlation"]) < 0.3

    # The real A/B: generate_cash_lineups' own shipped min_salary_fraction=
    # 0.95 default must show a real, positive, significant improvement over
    # the unconstrained version of the exact same real objective/weeks -
    # otherwise that default isn't earning its complexity.
    ab_result = result["cash_min_salary_fraction_0_95_vs_unconstrained"]
    assert ab_result["sample_size"] > 0
    assert ab_result["avg_actual_score_improvement"] > 0
    assert ab_result["p_value"] < 0.01


def test_run_opportunity_score_backtest_shows_real_signal_among_eligible_players(engine):
    # Real regression coverage: compute_opportunity_score carries real
    # predictive signal even among players who already clear the hard
    # playing-time floor - the whole premise behind treating it as a
    # distinct, separate signal from the binary floor itself.
    result = run_opportunity_score_backtest("dk_sunday_2026_09_20", seasons={2023, 2024, 2025}, engine=engine)

    assert result["weeks_evaluated"] >= 1
    assert result["eligible_player_weeks"] > 0
    assert result["correlation"] > 0.2
    for pos in ("QB", "RB", "WR", "TE"):
        assert result["correlation_by_position"][pos]["correlation"] > 0
    assert result["avg_actual_score_high_tercile"] > result["avg_actual_score_low_tercile"]
    assert result["p_value"] < 0.01


def test_run_shrinkage_backtest_reports_the_real_negative_result_honestly(engine):
    # Real regression coverage for a genuinely surprising, disclosed finding
    # (see models/playing_time_engine.py's own SHRINKAGE_K comment): the
    # empirical-Bayes blend does NOT beat the naive "trust current season
    # the instant it exists" baseline at predicting real next-week
    # snap_pct - it's measurably worse. This test locks in that this
    # backtest keeps reporting that real result rather than silently
    # flipping sign if the underlying data or logic ever changes without
    # anyone noticing.
    result = run_shrinkage_backtest("dk_sunday_2026_09_20", seasons={2023, 2024, 2025}, engine=engine)

    assert result["weeks_evaluated"] >= 1
    assert result["player_weeks_evaluated"] > 0
    assert result["avg_absolute_error_shrinkage"] > result["avg_absolute_error_naive_hard_cutoff"]
    assert result["avg_error_reduction"] < 0
    assert result["p_value"] < 0.01


def test_run_playing_time_floor_backtest_runs_and_shows_a_real_effect(engine):
    result = run_playing_time_floor_backtest("dk_thu_mon_2026_09_17", seasons={2023, 2024, 2025}, engine=engine)

    assert result["weeks_evaluated"] >= 1
    assert result["would_be_excluded_player_weeks"] > 0
    assert result["not_excluded_player_weeks"] > 0
    # The whole premise of this real hard floor: players it catches must
    # score meaningfully lower on average than players it doesn't - the
    # same real standard run_hard_exclude_backtest already holds itself to.
    assert result["avg_actual_score_if_excluded"] < result["avg_actual_score_if_not_excluded"]
    assert 0.0 <= result["miss_rate"] <= 1.0

    assert len(result["top_misses"]) <= 15
    for miss in result["top_misses"]:
        assert miss["actual_score"] > result["meaningful_score_threshold"]
        assert miss["position"] in ("QB", "RB", "WR", "TE")
    scores = [m["actual_score"] for m in result["top_misses"]]
    assert scores == sorted(scores, reverse=True)
