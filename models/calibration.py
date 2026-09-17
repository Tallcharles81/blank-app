import json
import math
import random
from collections import defaultdict

from sqlalchemy import text

from data.nflverse_fetch import _team_implied_totals, fetch_schedules
from data.player_crosswalk import resolve_dk_players_to_gsis
from data.pre_lock_check import MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE, _load_recent_usage_batch, _would_be_hard_excluded
from db.migrate import get_engine
from models.backtest import DEFAULT_RANDOM_FIELD_SIZE, _asof_projected_points, load_actual_scores, load_slate_pool
from models.optimizer import SALARY_CAP, build_lineups_from_pool
from models.projections import (
    GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT,
    _apply_game_environment_adjustment,
    _load_recent_stats,
    _project_from_history,
)
from models.simulation import _standard_normal_cdf

# The stored percentile ladder (models/projections.py's PERCENTILE_Z) has
# 10/25/50/75/90 but not 20 or 80 directly - both interpolated between their
# neighboring stored points the same way models/simulation.py already
# interpolates arbitrary percentiles from that ladder, rather than
# approximated by the nearest stored bucket.
_P20_LOWER = (0.10, "10")
_P20_UPPER = (0.25, "25")
_P80_LOWER = (0.75, "75")
_P80_UPPER = (0.90, "90")


def _interpolate(percentiles, target, lower, upper):
    x0, key0 = lower
    x1, key1 = upper
    y0, y1 = percentiles[key0], percentiles[key1]
    return y0 + (y1 - y0) * (target - x0) / (x1 - x0)


def _available_weeks(engine):
    query = text("SELECT DISTINCT season, week FROM player_weekly_stats ORDER BY season, week")
    with engine.connect() as conn:
        return [(row.season, row.week) for row in conn.execute(query)]


def _asof_projections(slate_players, gsis_by_dk_id, season, week, engine):
    history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))

    result = {}
    for player in slate_players:
        gsis_id = gsis_by_dk_id.get(player["player_id"])
        games = history.get(gsis_id) if gsis_id else None
        if not games:
            continue
        proj = _project_from_history(games, player["position"])
        percentiles = proj["proj_percentiles"]
        result[player["player_id"]] = {
            "median": proj["proj_median"],
            "p20": _interpolate(percentiles, 0.20, _P20_LOWER, _P20_UPPER),
            "p80": _interpolate(percentiles, 0.80, _P80_LOWER, _P80_UPPER),
        }
    return result


def _played_gsis_ids(gsis_ids, season, week, engine):
    # snap_pct > 0 is a much more precise "did they actually play" signal than
    # "scored exactly 0 points" - checked against real data first: of all
    # exact-zero-point rows, 98% also had zero recorded snaps, confirming this
    # correlates the way it should before relying on it.
    query = text(
        "SELECT player_id FROM player_weekly_stats "
        "WHERE player_id = ANY(:ids) AND season = :season AND week = :week AND snap_pct > 0"
    )
    with engine.connect() as conn:
        return {row.player_id for row in conn.execute(query, {"ids": list(gsis_ids), "season": season, "week": week})}


def run_weekly_calibration(source_slate_id, seasons=None, num_lineups=1, random_field_size=100, engine=None):
    """Backtest source_slate_id's real salaries against every available real
    historical week (or just `seasons` if given), one independent backtest per
    week. A week is skipped (not failed) if there's no overlap between
    as-of-projectable and actually-scored players for it - e.g. the very
    first available week has no prior history to project from at all.

    MAE and the p20/p50/p80 hit rates only count players who actually played
    that week (snap_pct > 0, or any DST - see _played_gsis_ids). A bye/
    inactive/injured week is a real, correct 0 for lineup-scoring purposes
    (see avg_field_percentile below, which does NOT apply this filter - a real
    lineup that rosters a player who goes inactive should be penalized for
    that), but it isn't a fair test of projection accuracy: the projection
    model has no injury/inactivity awareness, so comparing its "if they play"
    estimate against a "didn't play" actual isn't measuring the same thing.
    This was caught for real: MAE and hit rates without this filter looked
    biased, but excluding confirmed-DNP weeks alone moved the median hit rate
    from ~37% to ~53.5%, right around the 50% target.

    Returns (weekly_results, skipped_weeks). Each weekly_results entry has the
    per-week counts needed to pool correctly across weeks afterward (see
    summarize_calibration) - not pre-averaged.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(slate_players, engine)

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    rng = random.Random()
    weekly_results = []
    skipped = []

    for season, week in weeks:
        asof = _asof_projections(slate_players, gsis_by_dk_id, season, week, engine)
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)

        eligible_ids = set(asof) & set(actual_points)
        if not eligible_ids:
            skipped.append((season, week))
            continue

        eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]

        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        accuracy_players = [
            p
            for p in eligible_players
            if p["position"] == "DST" or gsis_by_dk_id.get(p["player_id"]) in played_gsis_ids
        ]

        abs_error_by_position = defaultdict(lambda: [0.0, 0])
        p20_hits = p50_hits = p80_hits = 0
        for p in accuracy_players:
            pid = p["player_id"]
            actual = actual_points[pid]
            error = abs(asof[pid]["median"] - actual)
            abs_error_by_position[p["position"]][0] += error
            abs_error_by_position[p["position"]][1] += 1

            if actual <= asof[pid]["p20"]:
                p20_hits += 1
            if actual >= asof[pid]["median"]:
                p50_hits += 1
            if actual >= asof[pid]["p80"]:
                p80_hits += 1
        opportunities = len(accuracy_players)

        asof_pool = [{**p, "points": asof[p["player_id"]]["median"]} for p in eligible_players]
        try:
            lineups, _ = build_lineups_from_pool(asof_pool, num_lineups=num_lineups, salary_cap=SALARY_CAP)
        except ValueError:
            skipped.append((season, week))
            continue

        def score_actual(roster):
            return sum(actual_points[p["player_id"]] for _, p in roster)

        field_scores = []
        for _ in range(random_field_size):
            randomized_pool = [{**p, "points": rng.random()} for p in eligible_players]
            try:
                field_lineup, _ = build_lineups_from_pool(randomized_pool, num_lineups=1, salary_cap=SALARY_CAP)
            except ValueError:
                continue
            field_scores.append(score_actual(field_lineup[0]["roster"]))

        lineup_percentiles = []
        for lu in lineups:
            actual = score_actual(lu["roster"])
            if field_scores:
                better_than = sum(1 for s in field_scores if actual > s)
                lineup_percentiles.append(better_than / len(field_scores))

        weekly_results.append(
            {
                "season": season,
                "week": week,
                "num_players_evaluated": len(eligible_ids),
                "mae_by_position": {
                    pos: {"mae": round(total / n, 4), "n": n} for pos, (total, n) in abs_error_by_position.items()
                },
                "p20_hits": p20_hits,
                "p20_opportunities": opportunities,
                "p50_hits": p50_hits,
                "p50_opportunities": opportunities,
                "p80_hits": p80_hits,
                "p80_opportunities": opportunities,
                "avg_field_percentile": (
                    round(sum(lineup_percentiles) / len(lineup_percentiles), 4) if lineup_percentiles else None
                ),
                "field_size": len(field_scores),
            }
        )

    return weekly_results, skipped


def store_calibration_results(weekly_results, engine=None):
    engine = engine or get_engine()
    upsert_sql = text(
        """
        INSERT INTO calibration_weekly
            (season, week, num_players_evaluated, mae_by_position,
             p20_hits, p20_opportunities, p50_hits, p50_opportunities, p80_hits, p80_opportunities,
             avg_field_percentile, field_size)
        VALUES
            (:season, :week, :num_players_evaluated, :mae_by_position,
             :p20_hits, :p20_opportunities, :p50_hits, :p50_opportunities, :p80_hits, :p80_opportunities,
             :avg_field_percentile, :field_size)
        ON CONFLICT (season, week) DO UPDATE SET
            num_players_evaluated = EXCLUDED.num_players_evaluated,
            mae_by_position = EXCLUDED.mae_by_position,
            p20_hits = EXCLUDED.p20_hits,
            p20_opportunities = EXCLUDED.p20_opportunities,
            p50_hits = EXCLUDED.p50_hits,
            p50_opportunities = EXCLUDED.p50_opportunities,
            p80_hits = EXCLUDED.p80_hits,
            p80_opportunities = EXCLUDED.p80_opportunities,
            avg_field_percentile = EXCLUDED.avg_field_percentile,
            field_size = EXCLUDED.field_size,
            computed_at = now()
        """
    )
    with engine.begin() as conn:
        for row in weekly_results:
            conn.execute(upsert_sql, {**row, "mae_by_position": json.dumps(row["mae_by_position"])})


def summarize_calibration(seasons=None, engine=None):
    """Pools stored calibration_weekly rows into the cross-week numbers: MAE by
    position (weighted by each week's player count, not a plain average of
    per-week MAEs), the overall p80 hit rate (total hits / total
    opportunities), and the average/worst/best week by field percentile.
    """
    engine = engine or get_engine()
    query = "SELECT * FROM calibration_weekly"
    params = {}
    if seasons is not None:
        query += " WHERE season = ANY(:seasons)"
        params["seasons"] = list(seasons)
    query += " ORDER BY season, week"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().fetchall()
    if not rows:
        raise ValueError("No calibration_weekly rows to summarize")

    pooled_error = defaultdict(lambda: [0.0, 0])
    totals = {"p20": [0, 0], "p50": [0, 0], "p80": [0, 0]}
    field_percentiles_by_week = []

    for row in rows:
        mae_by_position = row["mae_by_position"]
        if isinstance(mae_by_position, str):
            mae_by_position = json.loads(mae_by_position)
        for pos, info in (mae_by_position or {}).items():
            pooled_error[pos][0] += info["mae"] * info["n"]
            pooled_error[pos][1] += info["n"]

        for key in totals:
            totals[key][0] += row[f"{key}_hits"] or 0
            totals[key][1] += row[f"{key}_opportunities"] or 0

        if row["avg_field_percentile"] is not None:
            field_percentiles_by_week.append((row["season"], row["week"], float(row["avg_field_percentile"])))

    mae_by_position_overall = {pos: round(total / n, 4) for pos, (total, n) in pooled_error.items() if n}
    hit_rates = {key: round(hits / opps, 4) if opps else None for key, (hits, opps) in totals.items()}

    field_only = [fp for _, _, fp in field_percentiles_by_week]
    worst = min(field_percentiles_by_week, key=lambda t: t[2]) if field_percentiles_by_week else None
    best = max(field_percentiles_by_week, key=lambda t: t[2]) if field_percentiles_by_week else None

    return {
        "weeks_evaluated": len(rows),
        "mae_by_position": mae_by_position_overall,
        # p20: fraction of players scoring AT OR BELOW their as-of 20th-
        # percentile (floor) projection - should land near 20% if calibrated.
        "p20_hits": totals["p20"][0],
        "p20_opportunities": totals["p20"][1],
        "p20_hit_rate": hit_rates["p20"],
        # p50: fraction scoring AT OR ABOVE their median projection - should
        # land near 50%. Well below 50% means the median itself runs high
        # (a bias problem); near 50% while p20/p80 still miss their targets
        # means the distribution is too narrow (a spread/variance problem).
        "p50_hits": totals["p50"][0],
        "p50_opportunities": totals["p50"][1],
        "p50_hit_rate": hit_rates["p50"],
        "p80_hits": totals["p80"][0],
        "p80_opportunities": totals["p80"][1],
        "p80_hit_rate": hit_rates["p80"],
        "avg_field_percentile": round(sum(field_only) / len(field_only), 4) if field_only else None,
        "worst_week": {"season": worst[0], "week": worst[1], "avg_field_percentile": worst[2]} if worst else None,
        "best_week": {"season": best[0], "week": best[1], "avg_field_percentile": best[2]} if best else None,
    }


def run_gpp_ceiling_backtest(source_slate_id, seasons=None, random_field_size=DEFAULT_RANDOM_FIELD_SIZE, engine=None):
    """The GPP equivalent of the projection engine's own P80 calibration
    check, applied to whole lineups instead of individual players: build the
    as-of proj_ceiling-optimized lineup for every real historical week (the
    same field real GPP-mode lineups use - see
    models/optimizer.generate_lineups), score it against that week's real
    actual results, and rank it against a real random-legal-lineup field
    (same technique as models/backtest.py's backtest_slate) scored on the
    same real outcomes.

    A random lineup finishes in the top 20% of a random field exactly 20% of
    the time by construction - that's the baseline. If the ceiling-optimized
    lineup is doing what it claims, its top-20%-finish rate should be
    meaningfully ABOVE 20%. Also builds the as-of proj_median lineup for the
    same weeks as a same-methodology comparison point (not pulled from the
    separately-computed calibration_weekly table, to keep this apples-to-
    apples on identical weeks/field draws).

    Returns (weekly_results, summary). Never writes to any table - a
    diagnostic, not a stored calibration.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(slate_players, engine)

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    rng = random.Random()
    weekly_results = []
    skipped = []

    for season, week in weeks:
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)
        ceiling_points = _asof_projected_points(slate_players, gsis_by_dk_id, season, week, engine, field="proj_ceiling")
        median_points = _asof_projected_points(slate_players, gsis_by_dk_id, season, week, engine, field="proj_median")

        eligible_ids = set(actual_points) & set(ceiling_points) & set(median_points)
        if not eligible_ids:
            skipped.append((season, week))
            continue
        eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]

        def score_actual(roster):
            return sum(actual_points[p["player_id"]] for _, p in roster)

        field_scores = []
        for _ in range(random_field_size):
            randomized_pool = [{**p, "points": rng.random()} for p in eligible_players]
            try:
                field_lineup, _ = build_lineups_from_pool(randomized_pool, num_lineups=1, salary_cap=SALARY_CAP)
            except ValueError:
                continue
            field_scores.append(score_actual(field_lineup[0]["roster"]))
        if not field_scores:
            skipped.append((season, week))
            continue

        def build_and_rank(points_by_id):
            pool = [{**p, "points": points_by_id[p["player_id"]]} for p in eligible_players]
            try:
                lineups, _ = build_lineups_from_pool(pool, num_lineups=1, salary_cap=SALARY_CAP)
            except ValueError:
                return None
            actual = score_actual(lineups[0]["roster"])
            better_than = sum(1 for s in field_scores if actual > s)
            return round(better_than / len(field_scores), 4)

        ceiling_percentile = build_and_rank(ceiling_points)
        median_percentile = build_and_rank(median_points)
        if ceiling_percentile is None or median_percentile is None:
            skipped.append((season, week))
            continue

        weekly_results.append(
            {
                "season": season,
                "week": week,
                "ceiling_field_percentile": ceiling_percentile,
                "median_field_percentile": median_percentile,
                "ceiling_top20": ceiling_percentile >= 0.80,
                "field_size": len(field_scores),
            }
        )

    n = len(weekly_results)
    ceiling_top20_rate = round(sum(1 for r in weekly_results if r["ceiling_top20"]) / n, 4) if n else None
    avg_ceiling_percentile = round(sum(r["ceiling_field_percentile"] for r in weekly_results) / n, 4) if n else None
    avg_median_percentile = round(sum(r["median_field_percentile"] for r in weekly_results) / n, 4) if n else None

    summary = {
        "weeks_evaluated": n,
        "skipped_weeks": skipped,
        "ceiling_top20_finish_rate": ceiling_top20_rate,
        "baseline_random_top20_rate": 0.20,
        "avg_ceiling_lineup_field_percentile": avg_ceiling_percentile,
        "avg_median_lineup_field_percentile": avg_median_percentile,
    }
    return weekly_results, summary


# ---------------------------------------------------------------------------
# Backtests for two real additions that shipped tonight without ever being
# run through this module - the Vegas ceiling adjustment
# (models/projections.py) and the hard-exclude thresholds
# (data/pre_lock_check.py). Both were reasoned through and spot-checked
# against the live pool, which is a different, weaker claim than "validated
# against real historical outcomes" - the standard models/matchups.py's TE
# adjustment was held to before it shipped. These two functions apply that
# same standard after the fact.
#
# Neither this project's requirements.txt nor any other module here depends
# on scipy (see models/simulation.py's own hand-rolled erf-based normal
# CDF, kept specifically to avoid that dependency) - _paired_significance/
# _two_sample_significance below reuse that exact function rather than
# duplicate it, and use a normal approximation to the t-distribution, which
# is a reasonable approximation at the sample sizes (hundreds of real
# player-weeks) these backtests actually produce.
# ---------------------------------------------------------------------------


def _paired_significance(differences):
    """Two-tailed significance test for whether paired `differences` (e.g.
    baseline_error - adjusted_error for the SAME player-week under both
    conditions) has a nonzero mean. Returns (mean_diff, t_stat, p_value);
    t_stat/p_value are None if there's too little data or zero variance to
    say anything.
    """
    n = len(differences)
    if n < 2:
        return (differences[0] if differences else 0.0), None, None
    mean_diff = sum(differences) / n
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev == 0:
        return mean_diff, None, None
    t_stat = mean_diff / (stdev / math.sqrt(n))
    p_value = float(2 * (1 - _standard_normal_cdf(abs(t_stat))))
    return mean_diff, t_stat, p_value


def _two_sample_significance(group_a, group_b):
    """Welch's t-test (unequal variance, normal approximation) for whether
    two INDEPENDENT samples (e.g. would-be-excluded vs not-excluded
    player-weeks - different players/weeks, not a paired comparison) have
    different means. Returns (mean_a, mean_b, t_stat, p_value); the latter
    two are None if either group has too little data or the pooled
    variance is zero.
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a = sum(group_a) / n_a if n_a else None
    mean_b = sum(group_b) / n_b if n_b else None
    if n_a < 2 or n_b < 2:
        return mean_a, mean_b, None, None
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return mean_a, mean_b, None, None
    t_stat = (mean_a - mean_b) / se
    p_value = float(2 * (1 - _standard_normal_cdf(abs(t_stat))))
    return mean_a, mean_b, t_stat, p_value


def _teams_for_week(gsis_ids, season, week, engine):
    # A player's CURRENT slate team can differ from their real team in a
    # past backtested week after a trade - the same real issue
    # models/matchups.py's _opponents_for_week already solves by pulling
    # the real historical value per week rather than trusting the current
    # slate's stored field. Vegas implied totals are team-specific, so
    # using the wrong team for a traded player would silently test the
    # adjustment against the wrong number.
    query = text(
        "SELECT player_id, team FROM player_weekly_stats WHERE player_id = ANY(:ids) AND season = :season AND week = :week"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"ids": list(gsis_ids), "season": season, "week": week}).fetchall()
    return {row.player_id: row.team for row in rows}


def run_game_environment_backtest_comparison(source_slate_id, seasons=None, engine=None):
    """Does the Vegas ceiling adjustment (models/projections.py's
    GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT) improve real calibration
    versus the unadjusted baseline, across every real historical week this
    can be tested against? Mirrors models/matchups.py's
    run_matchup_backtest_comparison methodology (same as-of/no-lookahead
    pattern via _load_recent_stats(before=...), same played-only filter,
    pooled + adjusted-only subset) so this is a fair before/after
    comparison, not a new, incomparable test design.

    The adjustment only touches the ceiling-side percentiles (see
    _apply_game_environment_adjustment) - proj_ceiling is meant to be a
    90th-percentile target, not a central-tendency point estimate, so the
    metric here is the P90 HIT RATE (fraction of real actuals >=
    proj_ceiling), which should sit near 10% for a well-calibrated target,
    not MAE.

    Player-weeks with a real implied total are split into terciles by
    (implied_total - that week's league-average implied total) - the exact
    quantity the adjustment scales on - to test the real, DIRECTIONAL
    claim: in the highest tercile (real projected shootouts), does the
    adjusted ceiling's P90 hit rate land closer to the true 10% target than
    the unadjusted baseline? Symmetrically for the lowest tercile. A pooled
    all-weeks number alone can hide this - a real effect in both tails can
    still average out to "no difference" overall.

    Returns a dict with pooled hit rates, tercile-level hit rates, and a
    paired significance test on the per-player-week P90-hit indicator
    (baseline - adjusted) within the top and bottom terciles. Never writes
    anywhere - a diagnostic, like run_matchup_backtest_comparison.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(slate_players, engine)
    players_by_id = {p["player_id"]: p for p in slate_players}

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    # Fetched ONCE and filtered per week in Python below, not re-fetched
    # inside the loop - real historical spread_line/total_line data for
    # every season/week lives in the same schedules file, and this loop
    # covers 50+ real weeks; re-downloading it that many times would be
    # pure network waste for data that never changes within one run.
    try:
        implied_totals_by_team_season_week = _team_implied_totals(fetch_schedules())
    except Exception:
        implied_totals_by_team_season_week = {}

    records = []  # (position, gap_or_None, baseline_hit, adjusted_hit)
    weeks_evaluated = 0

    for season, week in weeks:
        history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        teams_by_gsis = _teams_for_week(gsis_by_dk_id.values(), season, week, engine)

        implied_totals_by_team = {
            team: total
            for (team, s, w), total in implied_totals_by_team_season_week.items()
            if s == season and w == week
        }
        league_average = (
            sum(implied_totals_by_team.values()) / len(implied_totals_by_team) if implied_totals_by_team else None
        )

        week_had_data = False
        for dk_id, gsis_id in gsis_by_dk_id.items():
            player = players_by_id.get(dk_id)
            if player is None or dk_id not in actual_points:
                continue
            if player["position"] != "DST" and gsis_id not in played_gsis_ids:
                continue
            games = history.get(gsis_id)
            if not games:
                continue

            proj = _project_from_history(games, player["position"])
            actual = actual_points[dk_id]
            baseline_hit = 1 if actual >= proj["proj_ceiling"] else 0

            real_team = teams_by_gsis.get(gsis_id)
            implied_total = implied_totals_by_team.get(real_team) if real_team else None
            if implied_total is not None and league_average is not None:
                adjusted = _apply_game_environment_adjustment(proj, implied_total, league_average)
                gap = implied_total - league_average
            else:
                adjusted = proj
                gap = None
            adjusted_hit = 1 if actual >= adjusted["proj_ceiling"] else 0

            records.append((player["position"], gap, baseline_hit, adjusted_hit))
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if not records:
        raise ValueError(f"No real player-weeks could be evaluated for slate {source_slate_id}")

    def _hit_rate_summary(rows, hit_index):
        hits = sum(r[hit_index] for r in rows)
        return {"hits": hits, "opportunities": len(rows), "rate": round(hits / len(rows), 4) if rows else None}

    with_gap = [r for r in records if r[1] is not None]
    with_gap.sort(key=lambda r: r[1])
    n = len(with_gap)
    tercile_size = n // 3

    result = {
        "weeks_evaluated": weeks_evaluated,
        "players_evaluated": len(records),
        "players_with_real_implied_total": n,
        "pooled": {
            "baseline_p90_hit_rate": _hit_rate_summary(records, 2),
            "adjusted_p90_hit_rate": _hit_rate_summary(records, 3),
        },
    }

    if tercile_size >= 20:  # not worth reporting a tercile split on a tiny sample
        low_tercile = with_gap[:tercile_size]
        high_tercile = with_gap[-tercile_size:]
        for label, tercile in (("low_implied_total_tercile", low_tercile), ("high_implied_total_tercile", high_tercile)):
            baseline_diffs = [r[2] - r[3] for r in tercile]  # positive = baseline hit more than adjusted
            mean_diff, t_stat, p_value = _paired_significance(baseline_diffs)
            result[label] = {
                "n": len(tercile),
                "avg_gap": round(sum(r[1] for r in tercile) / len(tercile), 2),
                "baseline_p90_hit_rate": _hit_rate_summary(tercile, 2),
                "adjusted_p90_hit_rate": _hit_rate_summary(tercile, 3),
                "baseline_minus_adjusted_hit_rate_diff": round(mean_diff, 4),
                "t_stat": round(t_stat, 4) if t_stat is not None else None,
                "p_value": round(p_value, 4) if p_value is not None else None,
            }

    return result


# Roughly a usable FLEX/bench score at PPR scoring - the real cost of a
# hard exclusion being wrong is a player who'd have been benched anyway
# scoring nothing (no real cost) vs. one who goes on to have a genuinely
# relevant real week despite the exclusion (a real cost). Not a precise
# science - a round, defensible "this would have mattered" line.
MEANINGFUL_SCORE_THRESHOLD = 8.0


def run_hard_exclude_backtest(source_slate_id, seasons=None, engine=None):
    """Does data/pre_lock_check.py's hard_role_exclusions criteria
    (HARD_EXCLUDE_MAX_SNAP_PCT for RB/WR/TE, the MIN_SNAP_PCT_FOR_BENCHED/
    MAX_SNAP_PCT_FOR_FULL_GAME pattern for QB) actually correspond to real
    subsequent futility? For every real historical played player-week,
    would_be_hard_excluded() is evaluated using ONLY data available
    strictly before that week (see _load_recent_usage_batch's `before`),
    then compared against what the player actually scored - the same
    no-lookahead standard as every other backtest in this module.

    A silent, permanent, hard exclusion is only defensible if the players
    it would catch genuinely tend not to produce. Reports the real average
    actual score for would-be-excluded vs not-excluded player-weeks (the
    effect size), a two-sample significance test on that gap, and the
    MISS RATE - the fraction of would-be-excluded player-weeks that
    nonetheless scored above MEANINGFUL_SCORE_THRESHOLD real PPR points -
    the real cost of this gate being wrong, which the average alone can
    hide.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    non_dst_players = [p for p in slate_players if p["position"] != "DST"]
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(non_dst_players, engine)
    players_by_id = {p["player_id"]: p for p in non_dst_players}

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    excluded_scores = []
    not_excluded_scores = []
    excluded_scores_by_position = defaultdict(list)
    weeks_evaluated = 0

    for season, week in weeks:
        games_by_gsis = _load_recent_usage_batch(gsis_by_dk_id.values(), engine, before=(season, week))
        actual_points, _ = load_actual_scores(non_dst_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)

        week_had_data = False
        for dk_id, gsis_id in gsis_by_dk_id.items():
            if dk_id not in actual_points or gsis_id not in played_gsis_ids:
                continue
            player = players_by_id[dk_id]
            games = games_by_gsis.get(gsis_id, [])
            if len(games) < MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE:
                continue  # not enough as-of history to make this call either way - same as the live gate

            is_excluded, _reason = _would_be_hard_excluded(player["position"], games)
            actual = actual_points[dk_id]
            if is_excluded:
                excluded_scores.append(actual)
                excluded_scores_by_position[player["position"]].append(actual)
            else:
                not_excluded_scores.append(actual)
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if not excluded_scores:
        raise ValueError(
            f"No real player-weeks would have been hard-excluded for slate {source_slate_id} - "
            "nothing to evaluate (this itself is worth knowing, not just an error)"
        )

    mean_excluded, mean_not_excluded, t_stat, p_value = _two_sample_significance(excluded_scores, not_excluded_scores)
    misses = sum(1 for s in excluded_scores if s > MEANINGFUL_SCORE_THRESHOLD)

    return {
        "weeks_evaluated": weeks_evaluated,
        "would_be_excluded_player_weeks": len(excluded_scores),
        "not_excluded_player_weeks": len(not_excluded_scores),
        "avg_actual_score_if_excluded": round(mean_excluded, 2) if mean_excluded is not None else None,
        "avg_actual_score_if_not_excluded": round(mean_not_excluded, 2) if mean_not_excluded is not None else None,
        "t_stat": round(t_stat, 4) if t_stat is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
        "meaningful_score_threshold": MEANINGFUL_SCORE_THRESHOLD,
        "miss_rate": round(misses / len(excluded_scores), 4),
        "misses": misses,
        "avg_actual_score_if_excluded_by_position": {
            pos: round(sum(scores) / len(scores), 2) for pos, scores in excluded_scores_by_position.items()
        },
        "would_be_excluded_count_by_position": {
            pos: len(scores) for pos, scores in excluded_scores_by_position.items()
        },
        # Pooled miss_rate above blends positions with very different real
        # exclusion criteria (RB/WR/TE: a volume floor claiming "no real
        # role"; QB: an intermittent-starter pattern claiming "can't
        # confirm who starts," a genuinely different, weaker claim) - a
        # per-position breakdown is what actually shows whether either one
        # is carrying the pooled number.
        "miss_rate_by_position": {
            pos: round(sum(1 for s in scores if s > MEANINGFUL_SCORE_THRESHOLD) / len(scores), 4)
            for pos, scores in excluded_scores_by_position.items()
        },
    }


# Real sample size floor for a fitted correlation to be trusted at all - below
# this, a single unusual week could swing the number a lot. Mirrors
# MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE's role elsewhere in this codebase: a
# minimum, not a claim that more data wouldn't still help.
MIN_PAIRS_FOR_FITTED_CORRELATION = 200

# "Relevant" here filters out players with token/garbage-time usage before
# computing a correlation - matches LOW_TARGET_SHARE_THRESHOLD/LOW_CARRIES_
# PER_GAME_THRESHOLD's reasoning in data/pre_lock_check.py: a real generated
# lineup only ever contains players who'd pass hard_role_exclusions in the
# first place, so pooling in every 4th-string, near-zero-usage player here
# would measure a different, less relevant population than what this
# correlation matrix actually gets applied to at simulation time.
FITTED_CORR_MIN_TARGET_SHARE = 0.10
FITTED_CORR_MIN_CARRIES = 5


def fit_real_correlation_matrix(seasons=None, engine=None):
    """Real, empirically-fitted replacement for models/simulation.py's
    hand-picked QB_PASS_CATCHER_CORR/QB_RB_CORR/SAME_TEAM_CORR/BRING_BACK_
    CORR/DST_VS_OPPONENT_OFFENSE_CORR - see that module's docstring for why
    those started as rule-of-thumb GPP-stacking assumptions rather than
    fitted values ("we don't have enough real joint outcome data yet").
    This is that fit, now that real multi-season player_weekly_stats data
    (with a real `opponent` column per player-week) exists to compute it
    from directly.

    This is NOT a no-lookahead backtest the way run_hard_exclude_backtest
    and run_game_environment_backtest_comparison above are - those evaluate
    whether a DECISION RULE would have correctly predicted an outcome using
    only data available beforehand, which lookahead bias would make
    meaningless. A correlation matrix is different in kind: it's a fixed
    structural property of how teammates'/opponents' real outcomes move
    together, applied identically to every future week regardless of when
    it was fit, not a per-week prediction that could leak future
    information into an earlier decision. Same reasoning nflverse-derived
    Vegas totals or any other static model parameter would get.

    For every real (team, season, week) with exactly one QB who recorded
    fantasy_points_ppr > 3 (filters out garbage-time/kneel-down backup
    relief appearances that aren't a real joint-outcome sample), computes
    real Pearson correlations across the whole real sample for:
      - qb_pass_catcher: that QB vs each of his own team's relevant WR/TE
      - qb_rb: that QB vs each of his own team's relevant RB
      - same_team: relevant RB/WR/TE pairs on the same team, excluding the
        QB (what SAME_TEAM_CORR covers in _player_correlation)
      - bring_back: that QB vs the OPPOSING team's relevant WR/TE that same
        week (uses player_weekly_stats.opponent)
      - dst_vs_opponent_offense: each team's DST fantasy_points_ppr vs the
        OPPOSING team's total real QB+RB+WR+TE fantasy_points_ppr that week

    Returns {relationship: {"correlation": float, "sample_size": int}} -
    caller decides what to do with a relationship whose sample_size is below
    MIN_PAIRS_FOR_FITTED_CORRELATION (too little real data to trust yet).
    """
    engine = engine or get_engine()

    query = text(
        """
        SELECT player_id, player_name, position, team, opponent, season, week,
               fantasy_points_ppr, target_share, carries
        FROM player_weekly_stats
        WHERE fantasy_points_ppr IS NOT NULL AND position IN ('QB', 'RB', 'WR', 'TE', 'DST')
        """
        + (" AND season = ANY(:seasons)" if seasons is not None else "")
    )
    params = {"seasons": list(seasons)} if seasons is not None else {}
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(query, params).mappings().fetchall()]
    for r in rows:
        r["fantasy_points_ppr"] = float(r["fantasy_points_ppr"])

    by_team_week = defaultdict(list)
    for r in rows:
        by_team_week[(r["team"], r["season"], r["week"])].append(r)

    team_offense_total = {
        key: sum(p["fantasy_points_ppr"] for p in plist if p["position"] in ("QB", "RB", "WR", "TE"))
        for key, plist in by_team_week.items()
    }

    def is_relevant_pass_catcher(p):
        return p["target_share"] is not None and float(p["target_share"]) >= FITTED_CORR_MIN_TARGET_SHARE

    def is_relevant_rb(p):
        return p["carries"] is not None and p["carries"] >= FITTED_CORR_MIN_CARRIES

    pairs = {
        "qb_pass_catcher": ([], []),
        "qb_rb": ([], []),
        "same_team": ([], []),
        "bring_back": ([], []),
        "dst_vs_opponent_offense": ([], []),
    }

    for (team, season, week), plist in by_team_week.items():
        qbs = [p for p in plist if p["position"] == "QB" and p["fantasy_points_ppr"] and p["fantasy_points_ppr"] > 3]
        if len(qbs) != 1:
            continue
        qb = qbs[0]
        rbs = [p for p in plist if p["position"] == "RB" and is_relevant_rb(p)]
        pass_catchers = [p for p in plist if p["position"] in ("WR", "TE") and is_relevant_pass_catcher(p)]
        others = rbs + pass_catchers

        for p in pass_catchers:
            pairs["qb_pass_catcher"][0].append(qb["fantasy_points_ppr"])
            pairs["qb_pass_catcher"][1].append(p["fantasy_points_ppr"])
        for p in rbs:
            pairs["qb_rb"][0].append(qb["fantasy_points_ppr"])
            pairs["qb_rb"][1].append(p["fantasy_points_ppr"])
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                pairs["same_team"][0].append(others[i]["fantasy_points_ppr"])
                pairs["same_team"][1].append(others[j]["fantasy_points_ppr"])

        opponent = qb.get("opponent")
        if opponent:
            opp_pass_catchers = [
                p
                for p in by_team_week.get((opponent, season, week), [])
                if p["position"] in ("WR", "TE") and is_relevant_pass_catcher(p)
            ]
            for p in opp_pass_catchers:
                pairs["bring_back"][0].append(qb["fantasy_points_ppr"])
                pairs["bring_back"][1].append(p["fantasy_points_ppr"])

    for (team, season, week), plist in by_team_week.items():
        dsts = [p for p in plist if p["position"] == "DST"]
        if not dsts:
            continue
        opponent = dsts[0].get("opponent")
        opponent_total = team_offense_total.get((opponent, season, week)) if opponent else None
        if opponent_total is None:
            continue
        pairs["dst_vs_opponent_offense"][0].append(dsts[0]["fantasy_points_ppr"])
        pairs["dst_vs_opponent_offense"][1].append(opponent_total)

    results = {}
    for relationship, (xs, ys) in pairs.items():
        n = len(xs)
        if n < 2:
            results[relationship] = {"correlation": None, "sample_size": n}
            continue
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n - 1)
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / (n - 1))
        std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / (n - 1))
        correlation = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else None
        results[relationship] = {
            "correlation": round(correlation, 4) if correlation is not None else None,
            "sample_size": n,
        }

    return results


# Three situational categories cleanly derivable from data already fetched
# for other purposes (schedules.csv.gz's home_rest/away_rest/div_game/
# gametime columns - no new data source needed) - checked whether real
# baseline projections are miscalibrated in each, the same way the Vegas
# ceiling adjustment was checked before being trusted.
#
# SHORT_REST_MAX_DAYS=6: real distribution of home_rest/away_rest across
# 2023-2026 (schedules.csv.gz) clusters sharply at 7 (703/697 real games -
# the normal Sunday-to-Sunday week) with the next-biggest cluster at 4 (70/72
# games - a Thursday game following a normal week). 6 cleanly separates
# "short week" from "normal or bye-adjusted week" without cutting through
# either real cluster.
#
# PACIFIC_TEAMS/EASTERN_TEAMS: real, static NFL team-to-timezone groupings
# (Las Vegas observes Pacific time; Indianapolis observes Eastern) - not
# derived from any fetched data, since nflverse's schedule file has no
# stadium timezone column, but a fixed real-world fact, same status as
# other static team mappings already in this codebase (e.g.
# data/player_crosswalk.py's team/city tables).
#
# EARLY_KICKOFF_TIME="13:00": confirmed empirically before relying on it -
# gametime in schedules.csv.gz is Eastern-normalized regardless of stadium
# (real Pacific-market HOME games show "16:05"/"16:25", the standard late-
# afternoon window scheduled so West Coast fans get an early local start -
# never "10:xx" local time), and "13:00" is the single most common real
# Sunday kickoff slot (542 of ~1,700 real 2023-2026 REG Sunday games),
# cleanly distinct from the late-afternoon (16:05/16:25) and primetime
# (20:xx) windows. A Pacific AWAY team at a "13:00" kickoff in an Eastern
# stadium is a real 10am-body-clock start - confirmed 44 such real games
# exist across 2023-2026, not a hypothetical scenario with zero real cases.
SHORT_REST_MAX_DAYS = 6
PACIFIC_TEAMS = {"SEA", "SF", "LAR", "LAC", "LV"}
EASTERN_TEAMS = {
    "BUF", "MIA", "NE", "NYJ", "NYG", "PHI", "PIT", "BAL",
    "CIN", "CLE", "ATL", "CAR", "JAX", "TB", "WAS", "IND",
}
EARLY_KICKOFF_TIME = "13:00"


def _classify_situational_games(schedules_df):
    """{(team, season, week): {"short_rest_both": bool, "cross_country_
    early": bool, "divisional_rematch": bool}} for every real REG-season
    game. short_rest_both and divisional_rematch apply to BOTH teams in the
    game (both squads are equally short-rested; both are equally facing a
    familiar division opponent); cross_country_early applies ONLY to the
    traveling Pacific team's own (team, season, week) entry, not the
    Eastern home team's - the home team isn't the one crossing time zones.

    divisional_rematch: the SECOND (by week) meeting between the same two
    teams in the same real season among real div_game=1 rows - grouped by
    (season, frozenset({home_team, away_team})) rather than assuming a
    fixed team order, since a division pair's two real meetings swap home/
    away.
    """
    df = schedules_df[schedules_df["game_type"] == "REG"]

    weeks_by_matchup = defaultdict(list)
    for row in df[df["div_game"] == 1].itertuples():
        weeks_by_matchup[(row.season, frozenset({row.home_team, row.away_team}))].append(row.week)
    rematch_weeks = set()
    for (season, teams), weeks_list in weeks_by_matchup.items():
        for week in sorted(weeks_list)[1:]:
            rematch_weeks.add((season, teams, week))

    result = {}
    for row in df.itertuples():
        short_rest_both = row.home_rest <= SHORT_REST_MAX_DAYS and row.away_rest <= SHORT_REST_MAX_DAYS
        is_rematch = bool(row.div_game) and (row.season, frozenset({row.home_team, row.away_team}), row.week) in rematch_weeks
        cross_country_early = (
            row.gametime == EARLY_KICKOFF_TIME
            and row.away_team in PACIFIC_TEAMS
            and row.home_team in EASTERN_TEAMS
        )
        for team in (row.home_team, row.away_team):
            result[(team, row.season, row.week)] = {
                "short_rest_both": short_rest_both,
                "divisional_rematch": is_rematch,
                "cross_country_early": cross_country_early and team == row.away_team,
            }
    return result


def run_situational_backtest(source_slate_id, seasons=None, engine=None):
    """For each of the three situational categories in _classify_
    situational_games, splits every real, no-lookahead as-of player-week
    into "in this situation" vs every other player-week (the complement -
    the natural baseline for "is this specific situation different from
    everything else," not a hand-picked control group), and reports real
    P90 ceiling calibration (hit rate - see run_game_environment_backtest_
    comparison for why hit rate, not MAE, is the right ceiling metric) and
    real central-tendency bias (actual - proj_median) for both sides, with
    a two-sample significance test on each (models/calibration.py's own
    _two_sample_significance, same normal-approximation method used
    throughout this module).

    Same as-of/no-lookahead methodology as every other backtest here
    (_load_recent_stats(before=(season, week)), played-only filter via
    _played_gsis_ids, DST exempted from that filter). Never writes
    anywhere - a diagnostic, like the other run_*_backtest* functions.

    Returns {category: {"n_in_situation", "n_complement",
    "p90_hit_rate_in_situation", "p90_hit_rate_complement",
    "hit_rate_t_stat", "hit_rate_p_value", "mean_bias_in_situation",
    "mean_bias_complement", "bias_t_stat", "bias_p_value"}}.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(slate_players, engine)
    players_by_id = {p["player_id"]: p for p in slate_players}

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    # Fetched once, same reasoning as run_game_environment_backtest_
    # comparison's own implied-totals fetch - real schedule data doesn't
    # change within one backtest run, so there's no reason to re-fetch it
    # once per evaluated week.
    situational_by_team_week = _classify_situational_games(fetch_schedules())

    categories = ("short_rest_both", "divisional_rematch", "cross_country_early")
    records = defaultdict(list)  # category -> [(in_situation: bool, p90_hit: 0/1, bias: float)]

    for season, week in weeks:
        history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        teams_by_gsis = _teams_for_week(gsis_by_dk_id.values(), season, week, engine)

        for dk_id, gsis_id in gsis_by_dk_id.items():
            player = players_by_id.get(dk_id)
            if player is None or dk_id not in actual_points:
                continue
            if player["position"] != "DST" and gsis_id not in played_gsis_ids:
                continue
            games = history.get(gsis_id)
            if not games:
                continue

            real_team = teams_by_gsis.get(gsis_id)
            situational = situational_by_team_week.get((real_team, season, week)) if real_team else None
            if situational is None:
                continue  # real team not found in this week's real schedule (shouldn't happen, but no data to classify with)

            proj = _project_from_history(games, player["position"])
            actual = actual_points[dk_id]
            p90_hit = 1 if actual >= proj["proj_ceiling"] else 0
            bias = actual - proj["proj_median"]

            for category in categories:
                records[category].append((situational[category], p90_hit, bias))

    if not any(records.values()):
        raise ValueError(f"No real player-weeks could be evaluated for slate {source_slate_id}")

    result = {}
    for category in categories:
        rows = records[category]
        in_situation = [r for r in rows if r[0]]
        complement = [r for r in rows if not r[0]]

        mean_bias_in, mean_bias_comp, bias_t, bias_p = _two_sample_significance(
            [r[2] for r in in_situation], [r[2] for r in complement]
        )
        _, _, hit_t, hit_p = _two_sample_significance([r[1] for r in in_situation], [r[1] for r in complement])

        result[category] = {
            "n_in_situation": len(in_situation),
            "n_complement": len(complement),
            "p90_hit_rate_in_situation": round(sum(r[1] for r in in_situation) / len(in_situation), 4) if in_situation else None,
            "p90_hit_rate_complement": round(sum(r[1] for r in complement) / len(complement), 4) if complement else None,
            "hit_rate_t_stat": round(hit_t, 4) if hit_t is not None else None,
            "hit_rate_p_value": round(hit_p, 4) if hit_p is not None else None,
            "mean_bias_in_situation": round(mean_bias_in, 3) if mean_bias_in is not None else None,
            "mean_bias_complement": round(mean_bias_comp, 3) if mean_bias_comp is not None else None,
            "bias_t_stat": round(bias_t, 4) if bias_t is not None else None,
            "bias_p_value": round(bias_p, 4) if bias_p is not None else None,
        }

    return result
