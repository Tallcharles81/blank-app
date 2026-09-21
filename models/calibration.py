import json
import math
import random
from collections import defaultdict

import numpy as np
from sqlalchemy import text

from data.nflverse_fetch import _team_implied_totals, _team_points_scored, fetch_schedules
from data.player_availability import HARD_EXCLUDE_ROSTER_STATUSES, fetch_roster_status_for_season
from data.player_crosswalk import resolve_dk_players_to_gsis
from data.pre_lock_check import MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE, _load_recent_usage_batch, _would_be_hard_excluded
from db.migrate import get_engine
from models import playing_time_engine
from models.backtest import DEFAULT_RANDOM_FIELD_SIZE, _asof_projected_points, load_actual_scores, load_slate_pool
from models.optimizer import SALARY_CAP, build_lineups_from_pool
from models.projections import (
    GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT,
    _apply_game_environment_adjustment,
    _load_recent_stats,
    _project_from_history,
)
from models.simulation import _correlated_percentile_ranks, _quantile_to_scores, _standard_normal_cdf

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


def _team_qb_availability_by_season(season):
    """Real per-week QB roster/gameday-active data for one season, sourced
    directly from nflverse's actual weekly roster feed (the same real feed
    data/player_availability.py's live gate uses) - NOT player_weekly_stats.
    player_weekly_stats only gets a row for a player who ALSO recorded real
    box-score participation that week (see fetch_player_stats), so a QB who
    didn't play AT ALL that week - traded away, released, on IR, or a
    healthy scratch who never dressed - is invisible to it regardless of
    what nflverse's real roster/injury data actually says. Confirmed for
    real: Daniel Jones (NYG's real QB1 for most of 2024) has zero
    player_weekly_stats row for week 17 2024 because he'd been released and
    signed to Minnesota's practice squad by then - but the real roster feed
    shows exactly that (status "DEV" under team "MIN", not "NYG") - which is
    what let Drew Lock's real spot start that week happen.

    Returns (team_by_gsis_week, season_team_qbs, status_by_team_gsis):
    - team_by_gsis_week: {(gsis_id, week): team} for every real player that
      week, any position - used to find a QB's own REAL team for the
      specific historical week being evaluated, not whatever team he's on
      TODAY (a real gap in this backtest before this fix: a well-traveled
      backup QB - exactly the population this whole check exists for - can
      easily be on a different team now than the one his flagged historical
      week actually happened on).
    - season_team_qbs: {team: set of real gsis_ids who appeared at
      position "QB" for that team in ANY week this season}.
    - status_by_team_gsis: {(team, gsis_id): {week: status}} - the real
      per-week roster status ("ACT"/"INA"/"RES"/"CUT"/"DEV"/etc) for every
      real QB, keyed to the specific team he held that status under (a
      traded/released player's row moves to his new team, so a week key
      missing here for his OLD team is itself real signal - see
      _teammate_qb_starter_unavailable).
    """
    try:
        roster_df = fetch_roster_status_for_season(season)
    except RuntimeError:
        return {}, defaultdict(set), {}

    team_by_gsis_week = {(row.gsis_id, row.week): row.team for row in roster_df.itertuples()}

    qb_df = roster_df[roster_df["position"] == "QB"]
    season_team_qbs = defaultdict(set)
    status_by_team_gsis = defaultdict(dict)
    for row in qb_df.itertuples():
        season_team_qbs[row.team].add(row.gsis_id)
        status_by_team_gsis[(row.team, row.gsis_id)][row.week] = row.status
    return team_by_gsis_week, season_team_qbs, status_by_team_gsis


# How many of a team's real PRIOR weeks a teammate QB must have actually been
# ACT before his current absence counts as a real "the starter is out"
# signal at all. Needed for real: an every-week-inactive QB3 (completely
# normal NFL roster construction, not evidence of anything) would otherwise
# trigger a false positive on almost every team every week - confirmed for
# real on the 2025 Giants, who carried 3 real roster QBs (Wilson, Dart,
# Winston) - Winston sat real-INA in 9 of his first 10 weeks as the
# emergency third arm, which would wrongly "explain" Dart's real Week 7
# start (Wilson, the guy Dart actually took the job from, was still
# real-ACT and healthy that week - a real benching, not an injury - see
# this module's own backtest disclosure) if Winston's routine inactivity
# alone were treated as a signal.
MIN_PRIOR_ACTIVE_WEEKS_FOR_ESTABLISHED_STARTER = 2

# A single "INA" week only counts as fresh news if the OTHER established
# teammate QB was real-ACT within this many weeks beforehand - otherwise a
# QB who's simply been the permanent, months-stale inactive QB2 (e.g. real
# 2024 Washington: Jeff Driskel was ACT weeks 1-4 backing up an uninjured
# Jayden Daniels, then real-INA every week after including week 18 - still
# "established" by the >=2 rule above, but his week-18 inactivity is old
# news, not a signal that Daniels was unavailable that week - confirmed
# real: Daniels himself was real-ACT that same week 18, just given a lighter
# real snap share in a game already clinched) doesn't wrongly count. A
# persistent roster-level exit (RES/CUT/etc, or no longer even listed under
# this team - see PERSISTENT_QB_UNAVAILABLE_STATUSES below) has no such
# recency requirement: once a real starter is on IR or released, that's not
# a "stale, no longer relevant" fact the way a bench demotion can be.
INA_RECENCY_WINDOW_WEEKS = 5

# Real roster statuses that mean "not competing for this team's starting job
# this week" regardless of how long ago that became true - unlike a single
# "INA" week (see INA_RECENCY_WINDOW_WEEKS above), these don't get stale.
# Reuses data/player_availability.py's own real HARD_EXCLUDE_ROSTER_STATUSES
# (RES/PUP/SUS/RET/CUT/etc - the exact same live-gate definition) plus "DEV"
# (practice squad) - a real demotion off the 53-man roster, the same real
# signal as being released, even though the live gate itself doesn't
# hard-exclude a DK-listed player over it (a live DK slate would never list
# a practice-squad player as rosterable in the first place, so the live gate
# has never needed to consider it - this backtest is asking a different
# question: was the historical TEAMMATE, not the DK pool candidate, off the
# team's active roster).
PERSISTENT_QB_UNAVAILABLE_STATUSES = HARD_EXCLUDE_ROSTER_STATUSES | {"DEV"}


def _teammate_qb_starter_unavailable(team, own_gsis_id, week, season_team_qbs, status_by_team_gsis):
    """True if some OTHER real QB who was an ESTABLISHED recent starter for
    `team` (real-ACT in at least MIN_PRIOR_ACTIVE_WEEKS_FOR_ESTABLISHED_
    STARTER weeks strictly before this one) is genuinely unavailable this
    specific week - either a persistent roster-level exit (see
    PERSISTENT_QB_UNAVAILABLE_STATUSES, including no longer being listed
    under this team's roster at all that week - e.g. traded or released) or
    a single real "INA" week that's still recent enough to be this week's
    actual news (see INA_RECENCY_WINDOW_WEEKS) rather than a long-stale
    bench demotion. This is the real, week-of signal that `own_gsis_id`'s
    intermittent season-long snap pattern might genuinely be this week's
    confirmed starter rather than a random spot appearance."""
    for other_gsis_id in season_team_qbs.get(team, set()) - {own_gsis_id}:
        timeline = status_by_team_gsis.get((team, other_gsis_id), {})
        prior_active_weeks = [w for w, status in timeline.items() if w < week and status == "ACT"]
        if len(prior_active_weeks) < MIN_PRIOR_ACTIVE_WEEKS_FOR_ESTABLISHED_STARTER:
            continue  # never really an established starter for this team either way

        status_this_week = timeline.get(week)
        if status_this_week is None or status_this_week in PERSISTENT_QB_UNAVAILABLE_STATUSES:
            return True
        if status_this_week == "INA" and (week - max(prior_active_weeks)) <= INA_RECENCY_WINDOW_WEEKS:
            return True
    return False


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


def _asof_full_projection(slate_players, gsis_by_dk_id, season, week, engine):
    """Same real, as-of (strictly-before-this-week) history restriction as
    models/backtest.py's _asof_projected_points, but keeps _project_from_
    history's FULL return (proj_floor/median/ceiling AND the complete
    proj_percentiles ladder) instead of collapsing to one field. The ladder
    is exactly what models/simulation.py's correlated Monte Carlo engine
    needs (see _quantile_to_scores) - there's no way to read it from the
    live projections table the way models/simulation.py::_load_players
    does, since a backtest's as-of projection for a past week was never
    written there (and must not be - it needs to be blind to data after
    that week, which the live table isn't).
    """
    history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))
    result = {}
    for player in slate_players:
        gsis_id = gsis_by_dk_id.get(player["player_id"])
        games = history.get(gsis_id) if gsis_id else None
        if not games:
            continue
        result[player["player_id"]] = _project_from_history(games, player["position"])
    return result


def _simulate_candidates_asof(candidates, players_meta_by_id, full_projections, num_simulations, seed):
    """models/simulation.py's simulate_lineups + win_rates, replayed against
    AS-OF (blind, backtest-only) percentile projections instead of a live
    slate's stored projections table - simulate_lineups's own _load_players
    hard-requires a real DB projections row for slate_id, which a backtest
    week's synthetic as-of projection never has. Same correlation model
    (_correlated_percentile_ranks/_quantile_to_scores) applied to an
    in-memory dict instead of a query; every candidate here comes from the
    same real Classic slate_player_pool, so real_position is just each
    player's own stored position (no Showdown CPT/FLEX resolution needed).
    """
    player_ids = sorted({p["player_id"] for lu in candidates for _, p in lu["roster"]})
    ordered = [
        {
            "player_id": pid,
            "real_position": players_meta_by_id[pid]["position"],
            "team": players_meta_by_id[pid]["team"],
            "opponent": players_meta_by_id[pid].get("opponent"),
            "percentiles": full_projections[pid]["proj_percentiles"],
        }
        for pid in player_ids
    ]
    percentile_ranks = _correlated_percentile_ranks(ordered, num_simulations, seed)
    scores_by_id = {
        p["player_id"]: _quantile_to_scores(percentile_ranks[:, j], p["percentiles"]) for j, p in enumerate(ordered)
    }
    stacked = np.vstack([sum(scores_by_id[p["player_id"]] for _, p in lu["roster"]) for lu in candidates])
    winner_idx = np.argmax(stacked, axis=0)
    counts = np.bincount(winner_idx, minlength=len(candidates))
    return [count / stacked.shape[1] for count in counts]


def run_simulation_selection_backtest(
    source_slate_id,
    seasons=None,
    num_candidates=10,
    num_simulations=2000,
    seed=None,
    random_field_size=DEFAULT_RANDOM_FIELD_SIZE,
    engine=None,
):
    """Does models/optimizer.py::generate_simulation_selected_lineup's real
    selection mechanism (build num_candidates diverse legal lineups, then
    pick by simulated win rate) produce better REAL historical outcomes
    than the CURRENT default - just taking the single highest-as-of-
    ceiling-sum lineup - across many real historical weeks? Same real
    backtest methodology as run_gpp_ceiling_backtest (one slate's real
    salary structure, replayed against every real historical week's actual
    results, ranked against the same real random-legal field each week),
    extended to compare TWO selection rules on IDENTICAL candidates/field/
    actual-outcomes per week, so any difference found is attributable to
    the selection rule itself, not a confound from different lineups or a
    different random field.

    "current" = candidates[0]. build_lineups_from_pool's diversity
    constraints are only applied to lineups AFTER the first, so candidates[0]
    is unconstrained - the single highest-raw-ceiling-sum roster, identical
    to what generate_lineups(num_lineups=1) would return. No separate build
    needed to get the "current" comparison point.

    num_simulations defaults far lower here (2000, not simulate_lineups's
    own live-path default of 10000) - this runs once per real historical
    week (dozens of weeks in one call), and the selection question only
    needs the winning CANDIDATE identified correctly, not a precise win-rate
    percentage to two decimal places.

    Returns (weekly_results, summary). Never writes to any table - a
    diagnostic, not a stored calibration, same as run_gpp_ceiling_backtest.

    SCOPE OF THE REAL NULL RESULT THIS FUNCTION PRODUCES (55 real weeks on
    dk_sunday_2026_09_20, t=0.20, mean field-percentile diff +0.0016 -
    essentially zero despite the two rules picking a DIFFERENT lineup 85%
    of the time): this says the CORRELATION-COPULA simulation engine in
    models/simulation.py (see that module's own disclosed limitation, right
    above QB_PASS_CATCHER_CORR) doesn't beat the deterministic ceiling pick
    on real historical outcomes. It is not evidence about simulation-based
    selection in general - a genuinely deeper engine (real play-by-play, or
    an explicit shared-game-environment factor) is architecturally capable
    of capturing correlation structure this one cannot, and could show a
    real effect this backtest was never able to detect. Don't cite this
    result as "simulation doesn't help GPPs" - only as "this specific,
    simpler simulation implementation doesn't, as tested."
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
        full_projections = _asof_full_projection(slate_players, gsis_by_dk_id, season, week, engine)

        eligible_ids = set(actual_points) & set(full_projections)
        if not eligible_ids:
            skipped.append((season, week))
            continue
        eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]
        players_meta_by_id = {p["player_id"]: p for p in eligible_players}

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

        ceiling_pool = [{**p, "points": full_projections[p["player_id"]]["proj_ceiling"]} for p in eligible_players]
        try:
            candidates, _ = build_lineups_from_pool(
                ceiling_pool, num_lineups=num_candidates, min_uniques=3, salary_cap=SALARY_CAP
            )
        except ValueError:
            skipped.append((season, week))
            continue
        if len(candidates) < 2:
            skipped.append((season, week))
            continue

        current_lineup = candidates[0]
        candidate_win_rates = _simulate_candidates_asof(
            candidates, players_meta_by_id, full_projections, num_simulations, seed
        )
        sim_best_idx = max(range(len(candidates)), key=lambda i: candidate_win_rates[i])
        sim_selected_lineup = candidates[sim_best_idx]

        def field_percentile(roster):
            actual = score_actual(roster)
            better_than = sum(1 for s in field_scores if actual > s)
            return round(better_than / len(field_scores), 4), actual

        current_percentile, current_actual = field_percentile(current_lineup["roster"])
        sim_percentile, sim_actual = field_percentile(sim_selected_lineup["roster"])

        weekly_results.append(
            {
                "season": season,
                "week": week,
                "current_field_percentile": current_percentile,
                "sim_selected_field_percentile": sim_percentile,
                "current_actual_points": round(current_actual, 2),
                "sim_selected_actual_points": round(sim_actual, 2),
                "same_lineup_picked": sim_best_idx == 0,
                "sim_selected_win_rate": round(candidate_win_rates[sim_best_idx], 4),
                "current_win_rate_among_candidates": round(candidate_win_rates[0], 4),
                "field_size": len(field_scores),
                "num_candidates_built": len(candidates),
            }
        )

    n = len(weekly_results)

    def avg(key):
        return round(sum(r[key] for r in weekly_results) / n, 4) if n else None

    sim_beats_current = sum(1 for r in weekly_results if r["sim_selected_field_percentile"] > r["current_field_percentile"])
    current_beats_sim = sum(1 for r in weekly_results if r["current_field_percentile"] > r["sim_selected_field_percentile"])
    tied_weeks = n - sim_beats_current - current_beats_sim

    diffs = [r["sim_selected_field_percentile"] - r["current_field_percentile"] for r in weekly_results]
    mean_diff = sum(diffs) / n if n else None
    t_stat = None
    if n and n >= 2:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std = math.sqrt(variance)
        if std > 0:
            t_stat = mean_diff / (std / math.sqrt(n))

    summary = {
        "weeks_evaluated": n,
        "skipped_weeks": skipped,
        "same_lineup_picked_rate": round(sum(1 for r in weekly_results if r["same_lineup_picked"]) / n, 4) if n else None,
        "avg_current_field_percentile": avg("current_field_percentile"),
        "avg_sim_selected_field_percentile": avg("sim_selected_field_percentile"),
        "sim_beats_current_weeks": sim_beats_current,
        "current_beats_sim_weeks": current_beats_sim,
        "tied_weeks": tied_weeks,
        "mean_percentile_diff_sim_minus_current": round(mean_diff, 4) if mean_diff is not None else None,
        "t_stat": round(t_stat, 4) if t_stat is not None else None,
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


def _pearson_r(xs, ys):
    """Shared real Pearson correlation - the same formula fit_real_
    correlation_matrix already computes inline for its five stacking
    relationships; factored out here since run_salary_left_backtest below
    needs the identical computation a third time.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n - 1)
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / (n - 1))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / (n - 1))
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


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

    Includes the real teammate-starter-unavailable check (see
    data/pre_lock_check.py's hard_role_exclusions/_would_be_hard_excluded):
    for each real historical week, a QB matching the intermittent-backup
    pattern is checked against whether his own real team's OTHER
    ESTABLISHED real QB(s) that season (see
    MIN_PRIOR_ACTIVE_WEEKS_FOR_ESTABLISHED_STARTER) were genuinely
    unavailable THAT SAME week - a persistent roster-level exit (IR/cut/
    practice-squad/no longer even listed under this team - see
    PERSISTENT_QB_UNAVAILABLE_STATUSES) or a recent real "inactive" week
    (see INA_RECENCY_WINDOW_WEEKS) - not "before", since real roster/
    inactive status is legitimately known before kickoff, the same real
    nflverse source data/player_availability.py's live gate uses (see
    _team_qb_availability_by_season/_teammate_qb_starter_unavailable for
    the full real logic and why each piece exists). Sourced directly from
    nflverse's real weekly roster feed, not player_weekly_stats - that
    table only has a row for a player who ALSO recorded real box-score
    participation that week, so it's structurally blind to exactly the
    cases this check exists for (a starter who didn't play AT ALL: traded,
    released, on IR, or a healthy scratch) - confirmed for real and fixed;
    see _team_qb_availability_by_season's docstring for the Daniel
    Jones/Drew Lock example that exposed this. Real, remaining limitation:
    this can only ever catch an injury/roster-driven change - a real,
    healthy benching for poor play (confirmed real cases in this exact
    dataset: Michael Penix Jr. over an ACT, healthy Kirk Cousins;
    Jaxson Dart over an ACT, healthy Russell Wilson; Andy Dalton over an
    ACT, healthy Bryce Young) is structurally invisible to any real
    injury/roster-status data source and will remain a real miss no matter
    how this check is refined.

    Uses player["position"] directly (not a Showdown-safe resolved real
    position) for the QB-pattern check - a real, separate, pre-existing gap
    in this specific backtest function (it's only ever been run against
    Classic slates in practice, where that distinction doesn't arise)
    noticed while making this change, not introduced by it - flagged here
    rather than silently left undocumented, fixing it is a separate task.
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

    roster_by_season = {season: _team_qb_availability_by_season(season) for season in {s for s, _ in weeks}}

    excluded_records = []  # (name, position, season, week, actual_score, reason)
    not_excluded_scores = []
    weeks_evaluated = 0

    for season, week in weeks:
        games_by_gsis = _load_recent_usage_batch(gsis_by_dk_id.values(), engine, before=(season, week))
        actual_points, _ = load_actual_scores(non_dst_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        team_by_gsis_week, season_team_qbs, status_by_team_gsis = roster_by_season[season]

        week_had_data = False
        for dk_id, gsis_id in gsis_by_dk_id.items():
            if dk_id not in actual_points or gsis_id not in played_gsis_ids:
                continue
            player = players_by_id[dk_id]
            games = games_by_gsis.get(gsis_id, [])
            if len(games) < MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE:
                continue  # not enough as-of history to make this call either way - same as the live gate

            teammate_starter_unavailable = False
            if player["position"] == "QB":
                # The player's REAL team for THIS historical week, not
                # whatever team he's on in today's slate - a well-traveled
                # backup (exactly this check's target population) can easily
                # have changed teams since. Falls back to the slate's team
                # only if he's altogether missing from that week's real
                # roster feed, which shouldn't happen for a game he actually
                # played.
                real_team = team_by_gsis_week.get((gsis_id, week), player["team"])
                teammate_starter_unavailable = _teammate_qb_starter_unavailable(
                    real_team, gsis_id, week, season_team_qbs, status_by_team_gsis
                )

            is_excluded, reason = _would_be_hard_excluded(player["position"], games, teammate_starter_unavailable)
            actual = actual_points[dk_id]
            if is_excluded:
                excluded_records.append((player["name"], player["position"], season, week, actual, reason))
            else:
                not_excluded_scores.append(actual)
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if not excluded_records:
        raise ValueError(
            f"No real player-weeks would have been hard-excluded for slate {source_slate_id} - "
            "nothing to evaluate (this itself is worth knowing, not just an error)"
        )

    excluded_scores = [r[4] for r in excluded_records]
    excluded_scores_by_position = defaultdict(list)
    for r in excluded_records:
        excluded_scores_by_position[r[1]].append(r[4])

    mean_excluded, mean_not_excluded, t_stat, p_value = _two_sample_significance(excluded_scores, not_excluded_scores)
    misses = [r for r in excluded_records if r[4] > MEANINGFUL_SCORE_THRESHOLD]

    return {
        "weeks_evaluated": weeks_evaluated,
        "would_be_excluded_player_weeks": len(excluded_scores),
        "not_excluded_player_weeks": len(not_excluded_scores),
        "avg_actual_score_if_excluded": round(mean_excluded, 2) if mean_excluded is not None else None,
        "avg_actual_score_if_not_excluded": round(mean_not_excluded, 2) if mean_not_excluded is not None else None,
        "t_stat": round(t_stat, 4) if t_stat is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
        "meaningful_score_threshold": MEANINGFUL_SCORE_THRESHOLD,
        "miss_rate": round(len(misses) / len(excluded_scores), 4),
        "misses": len(misses),
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
        # The real, named worst cases - a miss_rate percentage alone doesn't
        # tell you whether "false exclusion" means "a real bench player
        # scraped 8.1 points once" or "a real difference-maker had a huge
        # week and got cut anyway." Sorted by real actual score descending,
        # capped at 15 so this stays a spot-check list, not a dump of every
        # miss (miss_rate/misses above already give the real, complete count).
        "top_misses": [
            {"name": name, "position": position, "season": season, "week": week, "actual_score": actual, "reason": reason}
            for name, position, season, week, actual, reason in sorted(misses, key=lambda r: -r[4])[:15]
        ],
    }


def run_playing_time_floor_backtest(source_slate_id, seasons=None, engine=None):
    """Does models/playing_time_engine.py's real hard floor
    (meets_playing_time_floor - WR/TE real blended snap_pct <50%/<40%, RB
    real blended snap_pct <30% AND real blended touches <8, QB not the real
    depth-chart-expected starter) correspond to real subsequent futility,
    the same no-lookahead standard as run_hard_exclude_backtest above
    (estimate_role is evaluated via _load_role_history's before_current_week
    cutoff - only real data strictly before the week under test).

    Real, necessary scope limitation: the QB "not expected starter" branch
    needs nflverse's real depth-chart feed, which only exists for the
    CURRENT season (see data/depth_charts.py) - it is never available for a
    real historical backtest week, so every historical QB player-week here
    has depth_chart_rank=None and is never excluded by that branch. This
    backtest is therefore a real test of the RB/WR/TE snap/touch floor
    only - the QB depth-chart branch is real but, by nflverse's own real
    data limits, untestable against history.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    non_dst_players = [p for p in slate_players if p["position"] != "DST"]
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(non_dst_players, engine)
    players_by_id = {p["player_id"]: p for p in non_dst_players}

    games_by_gsis = _load_recent_usage_batch(gsis_by_dk_id.values(), engine)
    real_position_by_gsis = {
        gsis_id: games[0]["position"] for gsis_id, games in games_by_gsis.items() if games
    }

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    excluded_records = []  # (name, position, season, week, actual_score, reason)
    not_excluded_scores = []
    weeks_evaluated = 0

    for season, week in weeks:
        actual_points, _ = load_actual_scores(non_dst_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        history_by_gsis = playing_time_engine._load_role_history(
            gsis_by_dk_id.values(), season, week, engine, before_current_week=True
        )

        week_had_data = False
        for dk_id, gsis_id in gsis_by_dk_id.items():
            if dk_id not in actual_points or gsis_id not in played_gsis_ids:
                continue
            real_position = real_position_by_gsis.get(gsis_id)
            if real_position not in ("QB", "RB", "WR", "TE"):
                continue  # this floor doesn't apply to DST/K, and an unresolved position isn't this test's call

            history = history_by_gsis.get(gsis_id, {"current": [], "prior": []})
            if not history["current"] and not history["prior"]:
                continue  # no real history on either side - same as the live gate's "not this gate's call" case

            player = players_by_id[dk_id]
            role_estimate = playing_time_engine.estimate_role(real_position, history, None)  # no real depth chart for a historical week - see docstring
            meets_floor, reason = playing_time_engine.meets_playing_time_floor(real_position, role_estimate)
            actual = actual_points[dk_id]
            if not meets_floor:
                excluded_records.append((player["name"], real_position, season, week, actual, reason))
            else:
                not_excluded_scores.append(actual)
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if not excluded_records:
        raise ValueError(
            f"No real player-weeks would have been excluded by the playing-time floor for slate {source_slate_id} - "
            "nothing to evaluate (this itself is worth knowing, not just an error)"
        )

    excluded_scores = [r[4] for r in excluded_records]
    excluded_scores_by_position = defaultdict(list)
    for r in excluded_records:
        excluded_scores_by_position[r[1]].append(r[4])

    mean_excluded, mean_not_excluded, t_stat, p_value = _two_sample_significance(excluded_scores, not_excluded_scores)
    misses = [r for r in excluded_records if r[4] > MEANINGFUL_SCORE_THRESHOLD]

    return {
        "weeks_evaluated": weeks_evaluated,
        "would_be_excluded_player_weeks": len(excluded_scores),
        "not_excluded_player_weeks": len(not_excluded_scores),
        "avg_actual_score_if_excluded": round(mean_excluded, 2) if mean_excluded is not None else None,
        "avg_actual_score_if_not_excluded": round(mean_not_excluded, 2) if mean_not_excluded is not None else None,
        "t_stat": round(t_stat, 4) if t_stat is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
        "meaningful_score_threshold": MEANINGFUL_SCORE_THRESHOLD,
        "miss_rate": round(len(misses) / len(excluded_scores), 4),
        "misses": len(misses),
        "would_be_excluded_count_by_position": {
            pos: len(scores) for pos, scores in excluded_scores_by_position.items()
        },
        "avg_actual_score_if_excluded_by_position": {
            pos: round(sum(scores) / len(scores), 2) for pos, scores in excluded_scores_by_position.items()
        },
        "miss_rate_by_position": {
            pos: round(sum(1 for s in scores if s > MEANINGFUL_SCORE_THRESHOLD) / len(scores), 4)
            for pos, scores in excluded_scores_by_position.items()
        },
        "top_misses": [
            {"name": name, "position": position, "season": season, "week": week, "actual_score": actual, "reason": reason}
            for name, position, season, week, actual, reason in sorted(misses, key=lambda r: -r[4])[:15]
        ],
    }


def run_salary_left_backtest(source_slate_id, risk_aversion=1.0, seasons=None, engine=None):
    """Does leaving real DK salary cap unspent predict a worse real
    subsequent outcome? Motivated by a real, single-anecdote finding
    already baked into generate_cash_lineups (min_salary_fraction=0.95 -
    see that function's own docstring: "caught for real on the first run:
    risk_aversion=1.0 with no salary floor left $17,200 of a $50,000 cap
    unused") that was never backtested against real historical data before
    now - exactly the kind of one-slate threshold this codebase's own
    standing rule says never to trust without a real backtest.

    For every real historical week (no-lookahead - _asof_projected_points
    only ever uses data available strictly before that week, same as every
    other backtest in this module), builds this source slate's real salary
    structure into two lineups with NO min_salary constraint applied (so
    real natural leftover-salary behavior can actually show up to be
    measured, rather than being forced away before it's observed):
      - ceiling_objective: the same real objective generate_lineups (GPP
        mode) uses - maximize sum(proj_ceiling).
      - cash_objective: the same real objective generate_cash_lineups uses
        - maximize sum(proj_floor - risk_aversion*(proj_ceiling-proj_floor))
        - MINUS its own min_salary_fraction floor, deliberately, so this
        backtest measures the same real unconstrained behavior the
        anecdote described.

    For each, Pearson-correlates the real lineup's own leftover salary
    (SALARY_CAP - real total salary spent) against its own real subsequent
    actual score across every real historical week. A negative real
    correlation (more real salary left behind -> lower real actual score)
    would validate the existing anecdote-based fix; no real correlation
    would mean that fix isn't earning its complexity for that objective -
    reported honestly either way, not assumed from the one anecdote it
    started from.

    Also runs a direct real paired A/B: the same cash_objective pool, same
    real week, WITH generate_cash_lineups' own current min_salary_fraction=
    0.95 constraint applied vs. without it - a real significance test of
    whether that specific, currently-shipped default actually improves real
    subsequent actual score, not just whether salary-left correlates with
    anything in the abstract.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(slate_players, engine)

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    ceiling_records = []  # (salary_left, actual_score)
    cash_records = []
    constrained_score_diffs = []  # constrained_actual_score - unconstrained_actual_score, same real week both sides
    skipped = []

    for season, week in weeks:
        actual_points, _ = load_actual_scores(slate_players, season, week, engine)
        ceiling_points = _asof_projected_points(slate_players, gsis_by_dk_id, season, week, engine, field="proj_ceiling")
        floor_points = _asof_projected_points(slate_players, gsis_by_dk_id, season, week, engine, field="proj_floor")

        eligible_ids = set(actual_points) & set(ceiling_points) & set(floor_points)
        if not eligible_ids:
            skipped.append((season, week))
            continue
        eligible_players = [p for p in slate_players if p["player_id"] in eligible_ids]

        def score_actual(roster):
            return sum(actual_points[p["player_id"]] for _, p in roster)

        ceiling_pool = [{**p, "points": ceiling_points[p["player_id"]]} for p in eligible_players]
        try:
            lu = build_lineups_from_pool(ceiling_pool, num_lineups=1, salary_cap=SALARY_CAP)[0][0]
            ceiling_records.append((SALARY_CAP - lu["total_salary"], score_actual(lu["roster"])))
        except ValueError:
            pass

        cash_pool = [
            {
                **p,
                "points": floor_points[p["player_id"]]
                - risk_aversion * (ceiling_points[p["player_id"]] - floor_points[p["player_id"]]),
            }
            for p in eligible_players
        ]
        unconstrained_score = None
        try:
            lu = build_lineups_from_pool(
                cash_pool, num_lineups=1, salary_cap=SALARY_CAP, max_players_per_team=3
            )[0][0]
            unconstrained_score = score_actual(lu["roster"])
            cash_records.append((SALARY_CAP - lu["total_salary"], unconstrained_score))
        except ValueError:
            pass

        # Direct real A/B test of generate_cash_lineups' OWN current default
        # (min_salary_fraction=0.95) against the unconstrained build above,
        # paired on the identical real week/pool/objective - only the
        # constraint differs. Only recorded when both sides actually built,
        # so the pairing stays real (same week on both sides of the diff).
        try:
            constrained_lu = build_lineups_from_pool(
                cash_pool,
                num_lineups=1,
                salary_cap=SALARY_CAP,
                max_players_per_team=3,
                min_salary=0.95 * SALARY_CAP,
            )[0][0]
            if unconstrained_score is not None:
                constrained_score_diffs.append(score_actual(constrained_lu["roster"]) - unconstrained_score)
        except ValueError:
            pass

    if not ceiling_records and not cash_records:
        raise ValueError(f"No real historical weeks produced a buildable lineup for slate {source_slate_id}")

    def _summarize(records):
        n = len(records)
        if n < 2:
            return {"sample_size": n, "correlation": None, "avg_salary_left": None, "avg_actual_score": None}
        salary_lefts = [r[0] for r in records]
        scores = [r[1] for r in records]
        correlation = _pearson_r(salary_lefts, scores)
        return {
            "sample_size": n,
            "correlation": round(correlation, 4) if correlation is not None else None,
            "avg_salary_left": round(sum(salary_lefts) / n, 2),
            "avg_actual_score": round(sum(scores) / n, 2),
        }

    mean_diff, t_stat, p_value = _paired_significance(constrained_score_diffs)

    return {
        "weeks_considered": len(weeks),
        "skipped_weeks": skipped,
        "ceiling_objective": _summarize(ceiling_records),
        "cash_objective": _summarize(cash_records),
        "cash_min_salary_fraction_0_95_vs_unconstrained": {
            "sample_size": len(constrained_score_diffs),
            "avg_actual_score_improvement": round(mean_diff, 2) if constrained_score_diffs else None,
            "t_stat": round(t_stat, 4) if t_stat is not None else None,
            "p_value": round(p_value, 4) if p_value is not None else None,
        },
    }


def run_opportunity_score_backtest(source_slate_id, seasons=None, engine=None):
    """Does models/playing_time_engine.py's compute_opportunity_score carry
    real predictive signal of its own, among players who ALREADY clear the
    hard playing-time floor? So far opportunity_score only ever reaches a
    debug-log field and PUNT MODE's own binary gate (apply_playing_time_gate
    only checks meets_playing_time_floor, not the score, to decide who gets
    punt-flagged) - never anything that changes NORMAL MODE's real
    eligible-but-thin players. Before wiring it into anything that would
    change what the optimizer picks, this checks whether the score itself
    means anything real: item 7's own language calls a floor-passing-but-
    low-opportunity player HIGH-VARIANCE-PUNT territory - this either shows
    that's true of real historical outcomes or it doesn't. Same no-lookahead
    standard as every other backtest in this module (estimate_role only
    ever sees _load_role_history's before_current_week=True data).

    Scoped to QB/RB/WR/TE only (compute_opportunity_score's own real scope -
    see meets_playing_time_floor for why DST/K never reach this floor at
    all), and to player-weeks that already pass meets_playing_time_floor -
    the floor's OWN real effect is run_playing_time_floor_backtest's job,
    not this one's; this backtest is specifically about the players that
    gate already lets through.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    non_dst_players = [p for p in slate_players if p["position"] != "DST"]
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(non_dst_players, engine)

    games_by_gsis = _load_recent_usage_batch(gsis_by_dk_id.values(), engine)
    real_position_by_gsis = {
        gsis_id: games[0]["position"] for gsis_id, games in games_by_gsis.items() if games
    }

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    records = []  # (opportunity_score, actual_score, position)
    weeks_evaluated = 0

    for season, week in weeks:
        actual_points, _ = load_actual_scores(non_dst_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        history_by_gsis = playing_time_engine._load_role_history(
            gsis_by_dk_id.values(), season, week, engine, before_current_week=True
        )

        week_had_data = False
        for dk_id, gsis_id in gsis_by_dk_id.items():
            if dk_id not in actual_points or gsis_id not in played_gsis_ids:
                continue
            real_position = real_position_by_gsis.get(gsis_id)
            if real_position not in ("QB", "RB", "WR", "TE"):
                continue

            history = history_by_gsis.get(gsis_id, {"current": [], "prior": []})
            if not history["current"] and not history["prior"]:
                continue  # no real history either side - not this backtest's call, same as the floor's own exemption

            role_estimate = playing_time_engine.estimate_role(real_position, history, None)
            meets_floor, _ = playing_time_engine.meets_playing_time_floor(real_position, role_estimate)
            if not meets_floor:
                continue

            opportunity_score = playing_time_engine.compute_opportunity_score(real_position, role_estimate)
            records.append((opportunity_score, actual_points[dk_id], real_position))
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if len(records) < 2:
        raise ValueError(
            f"Not enough real eligible player-weeks to evaluate opportunity_score for slate {source_slate_id}"
        )

    scores = [r[0] for r in records]
    actuals = [r[1] for r in records]
    correlation = _pearson_r(scores, actuals)

    by_position = defaultdict(list)
    for opp, actual, pos in records:
        by_position[pos].append((opp, actual))
    correlation_by_position = {}
    for pos, pairs in by_position.items():
        r = _pearson_r([p[0] for p in pairs], [p[1] for p in pairs])
        correlation_by_position[pos] = {
            "correlation": round(r, 4) if r is not None else None,
            "sample_size": len(pairs),
        }

    # Real, evidence-derived tercile split of the real observed opportunity_
    # score distribution among eligible players - not an invented fixed
    # cutoff, so the low/high comparison below reflects this slate's own
    # real data rather than a guessed threshold.
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    low_cut = sorted_scores[n // 3]
    high_cut = sorted_scores[2 * n // 3]
    low_actuals = [a for o, a, _ in records if o <= low_cut]
    high_actuals = [a for o, a, _ in records if o >= high_cut]
    mean_low, mean_high, t_stat, p_value = _two_sample_significance(low_actuals, high_actuals)

    return {
        "weeks_evaluated": weeks_evaluated,
        "eligible_player_weeks": len(records),
        "correlation": round(correlation, 4) if correlation is not None else None,
        "correlation_by_position": correlation_by_position,
        "low_opportunity_tercile_cutoff": round(low_cut, 2),
        "high_opportunity_tercile_cutoff": round(high_cut, 2),
        "avg_actual_score_low_tercile": round(mean_low, 2) if mean_low is not None else None,
        "avg_actual_score_high_tercile": round(mean_high, 2) if mean_high is not None else None,
        "t_stat": round(t_stat, 4) if t_stat is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
    }


def _actual_snap_pct_for_week(gsis_ids, season, week, engine):
    query = text(
        "SELECT player_id, snap_pct FROM player_weekly_stats "
        "WHERE player_id = ANY(:ids) AND season = :season AND week = :week AND snap_pct IS NOT NULL"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"ids": list(gsis_ids), "season": season, "week": week}).fetchall()
    return {row.player_id: float(row.snap_pct) for row in rows}


def run_shrinkage_backtest(source_slate_id, seasons=None, engine=None):
    """The real backtest models/playing_time_engine.py's own SHRINKAGE_K
    comment has promised since it shipped ("see run_shrinkage_backtest in
    models/calibration.py for the real test of whether this actually beats
    the old hard cutoff") but that function never actually existed - a real
    gap between what the code claimed and what was there, found by mapping
    this codebase against a later request rather than by design. This is
    that backtest, finally written.

    Compares _shrinkage_blend's real empirical-Bayes blend of current- and
    prior-season snap_pct against the naive approach it was built to
    replace: trust current-season data the moment ANY of it exists, ignore
    prior season entirely - a real, simple hard-cutoff baseline, not an
    invented strawman (this module's own SHRINKAGE_K comment calls it "an
    arbitrary game-count cutoff"). Both predict a player's real snap_pct in
    `week` using only real games strictly before it (before_current_week=
    True, the same no-lookahead standard as every other backtest here),
    scored by absolute error against that player's own real actual snap_pct
    in `week`.

    Restricted to n_current in [1, SHRINKAGE_K) - the real early-season
    regime where the two methods actually disagree. At n_current=0 both
    fall back identically to prior-season data (no real disagreement to
    measure); at n_current >= SHRINKAGE_K, shrinkage's own weight on
    current-season data approaches 1.0 and converges with the naive
    baseline by construction - this only tests the window where blending
    could plausibly help or hurt.
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    non_dst_players = [p for p in slate_players if p["position"] != "DST"]
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(non_dst_players, engine)

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    shrinkage_errors = []
    naive_errors = []
    weeks_evaluated = 0

    for season, week in weeks:
        history_by_gsis = playing_time_engine._load_role_history(
            gsis_by_dk_id.values(), season, week, engine, before_current_week=True
        )
        actual_snaps = _actual_snap_pct_for_week(gsis_by_dk_id.values(), season, week, engine)

        week_had_data = False
        for gsis_id in set(gsis_by_dk_id.values()):
            if gsis_id is None:
                continue
            actual = actual_snaps.get(gsis_id)
            if actual is None:
                continue
            history = history_by_gsis.get(gsis_id, {"current": [], "prior": []})
            # player_weekly_stats.snap_pct is NUMERIC - psycopg2 returns
            # Decimal, which _shrinkage_blend's float arithmetic can't mix
            # with (same real cast estimate_role already applies to this
            # exact data before calling the same function).
            current_snaps = [float(g["snap_pct"]) for g in history["current"] if g["snap_pct"] is not None]
            prior_snaps = [float(g["snap_pct"]) for g in history["prior"] if g["snap_pct"] is not None]
            n_current = len(current_snaps)
            if not (1 <= n_current < playing_time_engine.SHRINKAGE_K):
                continue
            if not prior_snaps:
                continue  # both methods degenerate to the same current-only average here - no real disagreement to measure

            shrinkage_pred, _, _, _ = playing_time_engine._shrinkage_blend(current_snaps, prior_snaps)
            naive_pred = sum(current_snaps) / len(current_snaps)

            shrinkage_errors.append(abs(shrinkage_pred - actual))
            naive_errors.append(abs(naive_pred - actual))
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if len(shrinkage_errors) < 2:
        raise ValueError(
            f"Not enough real early-season player-weeks to evaluate shrinkage for slate {source_slate_id}"
        )

    differences = [naive_e - shrink_e for naive_e, shrink_e in zip(naive_errors, shrinkage_errors)]
    mean_diff, t_stat, p_value = _paired_significance(differences)

    return {
        "weeks_evaluated": weeks_evaluated,
        "player_weeks_evaluated": len(shrinkage_errors),
        "avg_absolute_error_shrinkage": round(sum(shrinkage_errors) / len(shrinkage_errors), 4),
        "avg_absolute_error_naive_hard_cutoff": round(sum(naive_errors) / len(naive_errors), 4),
        "avg_error_reduction": round(mean_diff, 4),
        "t_stat": round(t_stat, 4) if t_stat is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
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


# A sharper, more mechanistic version of run_situational_backtest's blunt
# rematch flag: within real divisional rematches specifically, does it
# matter HOW the first meeting went - specifically, did the offense score
# well above its own season norm against this same rival? "Season norm"
# deliberately excludes both meetings against that rival (not just meeting
# 1) - the point is comparing the rival matchup to how this offense
# performs against everyone else, not letting the rematch itself leak into
# its own baseline.
#
# MIN_OTHER_GAMES_FOR_SEASON_BASELINE=3: a team's non-rival scoring average
# needs a real minimum sample to mean anything (a rookie-season team with
# only 1-2 non-rival games so far, e.g. very early in a season, would make
# "season average" mostly noise) - mirrors MIN_RECENT_GAMES_FOR_ROLE_
# CONFIDENCE's role elsewhere in this codebase.
MIN_OTHER_GAMES_FOR_SEASON_BASELINE = 3


def _real_divisional_rematch_torch_margins(schedules_df):
    """For every real divisional rematch pair (season, {team_a, team_b}),
    both teams' own real "did this offense torch this rival in meeting 1"
    margin - real meeting-1 points scored against the rival, minus that
    team's own real season average points scored against everyone ELSE
    that season (see MIN_OTHER_GAMES_FOR_SEASON_BASELINE). Both teams in
    the pair are evaluated independently (team A's offense vs team B's
    defense is a different real question from team B's offense vs team
    A's defense, even though it's the same two games) - a real division
    rivalry produces two independent torch/no-torch observations, not one.

    Returns [{"team", "season", "rival", "week1", "week2", "margin"}, ...] -
    week2 is the real rematch week whose player-level outcomes get
    evaluated; week1 is the meeting being scored for "did they torch them."
    """
    df = schedules_df[(schedules_df["game_type"] == "REG")].dropna(subset=["home_score", "away_score"])
    points_scored_by_team_week = _team_points_scored(df)

    team_season_games = defaultdict(list)  # (team, season) -> [(week, points_scored, opponent)]
    for (team, season, week), (opponent, points) in points_scored_by_team_week.items():
        team_season_games[(team, season)].append((week, points, opponent))

    weeks_by_matchup = defaultdict(list)
    for row in df[df["div_game"] == 1].itertuples():
        weeks_by_matchup[(row.season, frozenset({row.home_team, row.away_team}))].append(row.week)

    observations = []
    for (season, teams), weeks in weeks_by_matchup.items():
        weeks = sorted(weeks)
        if len(weeks) < 2:
            continue  # only one real meeting this season - no rematch to evaluate
        week1, week2 = weeks[0], weeks[1]
        for team, rival in ((t, next(iter(teams - {t}))) for t in teams):
            games = team_season_games.get((team, season), [])
            meeting1 = next((pts for w, pts, opp in games if w == week1 and opp == rival), None)
            if meeting1 is None:
                continue
            other_games = [pts for w, pts, opp in games if opp != rival]
            if len(other_games) < MIN_OTHER_GAMES_FOR_SEASON_BASELINE:
                continue
            season_baseline = sum(other_games) / len(other_games)
            observations.append(
                {
                    "team": team,
                    "season": season,
                    "rival": rival,
                    "week1": week1,
                    "week2": week2,
                    "margin": meeting1 - season_baseline,
                }
            )
    return observations


def run_divisional_rematch_torch_backtest(source_slate_id, seasons=None, engine=None):
    """Splits real divisional rematches by HOW the first meeting went,
    rather than treating every rematch identically (see
    run_situational_backtest's blunt divisional_rematch flag, which found
    no significant effect pooled). Real observations are split by a median
    cut of _real_divisional_rematch_torch_margins' real margin distribution
    into "torched" (this offense scored well above its own season norm
    against this rival in meeting 1) vs "did_not_torch" (at or below
    median) - a median split rather than a hand-picked absolute point
    threshold, so it reflects the real distribution's own shape rather
    than an invented cutoff.

    For each group, evaluates real, no-lookahead as-of P90 ceiling hit
    rate and central-tendency bias (actual - proj_median) for that
    offense's own real skill-position players (QB/RB/WR/TE - not DST,
    which doesn't have an "offense torched them" mechanism) in the REMATCH
    week specifically, then a two-sample significance test comparing the
    two groups directly - the sharper, mechanistic version of "does
    familiarity cap upside" the user asked for, versus the blunt yes/no
    rematch flag already tested.

    Returns {"n_torched_player_weeks", "n_did_not_torch_player_weeks",
    "median_margin_split", "p90_hit_rate_torched",
    "p90_hit_rate_did_not_torch", "hit_rate_t_stat", "hit_rate_p_value",
    "mean_bias_torched", "mean_bias_did_not_torch", "bias_t_stat",
    "bias_p_value"} - or a dict with None numbers and a "note" if the real
    sample is too small to trust (see MIN_MATCHED_PLAYERS_FOR_CORRELATION's
    role elsewhere in this codebase for the same reasoning applied here).
    """
    engine = engine or get_engine()

    slate_players = load_slate_pool(source_slate_id, engine)
    if not slate_players:
        raise ValueError(f"No players found in slate_player_pool for slate {source_slate_id}")
    skill_players = [p for p in slate_players if p["position"] in ("QB", "RB", "WR", "TE")]
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(skill_players, engine)
    players_by_id = {p["player_id"]: p for p in skill_players}

    available_weeks = set(_available_weeks(engine))
    if seasons is not None:
        available_weeks = {(s, w) for s, w in available_weeks if s in seasons}

    observations = _real_divisional_rematch_torch_margins(fetch_schedules())
    observations = [o for o in observations if (o["season"], o["week2"]) in available_weeks]
    if not observations:
        raise ValueError(f"No real divisional rematch observations fall within the evaluated weeks for {source_slate_id}")

    margins = sorted(o["margin"] for o in observations)
    median_margin = margins[len(margins) // 2] if len(margins) % 2 else (margins[len(margins) // 2 - 1] + margins[len(margins) // 2]) / 2
    torch_by_team_season_week2 = {
        (o["team"], o["season"], o["week2"]): o["margin"] > median_margin for o in observations
    }

    records = {"torched": [], "did_not_torch": []}  # each: (p90_hit, bias)
    for season, week in sorted(available_weeks):
        relevant = {(team, s, w): torched for (team, s, w), torched in torch_by_team_season_week2.items() if s == season and w == week}
        if not relevant:
            continue

        history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))
        actual_points, _ = load_actual_scores(skill_players, season, week, engine)
        played_gsis_ids = _played_gsis_ids(gsis_by_dk_id.values(), season, week, engine)
        teams_by_gsis = _teams_for_week(gsis_by_dk_id.values(), season, week, engine)

        for dk_id, gsis_id in gsis_by_dk_id.items():
            if dk_id not in actual_points or gsis_id not in played_gsis_ids:
                continue
            real_team = teams_by_gsis.get(gsis_id)
            torched = relevant.get((real_team, season, week))
            if torched is None:
                continue  # this player's real team wasn't in a rematch this week

            games = history.get(gsis_id)
            if not games:
                continue
            player = players_by_id[dk_id]
            proj = _project_from_history(games, player["position"])
            actual = actual_points[dk_id]
            p90_hit = 1 if actual >= proj["proj_ceiling"] else 0
            bias = actual - proj["proj_median"]
            records["torched" if torched else "did_not_torch"].append((p90_hit, bias))

    n_torched, n_did_not = len(records["torched"]), len(records["did_not_torch"])
    if n_torched < 20 or n_did_not < 20:
        return {
            "n_torched_player_weeks": n_torched,
            "n_did_not_torch_player_weeks": n_did_not,
            "note": "too few real player-weeks in one or both groups to trust a comparison (need >= 20 each)",
        }

    torched_hits = [r[0] for r in records["torched"]]
    did_not_hits = [r[0] for r in records["did_not_torch"]]
    torched_bias = [r[1] for r in records["torched"]]
    did_not_bias = [r[1] for r in records["did_not_torch"]]

    _, _, hit_t, hit_p = _two_sample_significance(torched_hits, did_not_hits)
    mean_bias_torched, mean_bias_did_not, bias_t, bias_p = _two_sample_significance(torched_bias, did_not_bias)

    return {
        "n_torched_player_weeks": n_torched,
        "n_did_not_torch_player_weeks": n_did_not,
        "median_margin_split": round(median_margin, 2),
        "p90_hit_rate_torched": round(sum(torched_hits) / n_torched, 4),
        "p90_hit_rate_did_not_torch": round(sum(did_not_hits) / n_did_not, 4),
        "hit_rate_t_stat": round(hit_t, 4) if hit_t is not None else None,
        "hit_rate_p_value": round(hit_p, 4) if hit_p is not None else None,
        "mean_bias_torched": round(mean_bias_torched, 3) if mean_bias_torched is not None else None,
        "mean_bias_did_not_torch": round(mean_bias_did_not, 3) if mean_bias_did_not is not None else None,
        "bias_t_stat": round(bias_t, 4) if bias_t is not None else None,
        "bias_p_value": round(bias_p, 4) if bias_p is not None else None,
    }


def run_game_environment_p80_hit_rate_backtest(source_slate_id, seasons=None, engine=None):
    """Does the Vegas ceiling adjustment actually improve REAL calibration
    on a DIFFERENT percentile than the one it was tuned against, or does
    boosting the "75"/"90" percentiles by the same multiplier just shift
    every number up without the shape of the distribution actually getting
    more accurate? run_game_environment_backtest_comparison only ever
    checked P90 (proj_ceiling itself); _apply_game_environment_adjustment
    boosts BOTH "75" and "90" (every PERCENTILE_Z label with z > 0) by the
    same multiplier, so P80 - interpolated between them, the same way
    run_weekly_calibration's own p80_hits already does - is a real, held-
    out check of whether the fix generalizes across the percentile ladder
    or was only ever validated at the one point it happened to improve.

    Same as-of/no-lookahead methodology and same real implied-total-gap
    tercile split as run_game_environment_backtest_comparison (fetches
    schedules once, not per week, for the same reason). True target for a
    well-calibrated P80 is 20%, not P90's 10%.

    Returns the same shape as run_game_environment_backtest_comparison
    (pooled + tercile p80 hit rates, paired significance test on the
    per-player-week hit-indicator difference) so the two are directly
    comparable side by side.
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

    try:
        implied_totals_by_team_season_week = _team_implied_totals(fetch_schedules())
    except Exception:
        implied_totals_by_team_season_week = {}

    def _p80(percentiles):
        return _interpolate(percentiles, 0.80, _P80_LOWER, _P80_UPPER)

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
            baseline_hit = 1 if actual >= _p80(proj["proj_percentiles"]) else 0

            real_team = teams_by_gsis.get(gsis_id)
            implied_total = implied_totals_by_team.get(real_team) if real_team else None
            if implied_total is not None and league_average is not None:
                adjusted = _apply_game_environment_adjustment(proj, implied_total, league_average)
                gap = implied_total - league_average
            else:
                adjusted = proj
                gap = None
            adjusted_hit = 1 if actual >= _p80(adjusted["proj_percentiles"]) else 0

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
        "true_target_hit_rate": 0.20,
        "pooled": {
            "baseline_p80_hit_rate": _hit_rate_summary(records, 2),
            "adjusted_p80_hit_rate": _hit_rate_summary(records, 3),
        },
    }

    if tercile_size >= 20:
        low_tercile = with_gap[:tercile_size]
        high_tercile = with_gap[-tercile_size:]
        for label, tercile in (("low_implied_total_tercile", low_tercile), ("high_implied_total_tercile", high_tercile)):
            baseline_diffs = [r[2] - r[3] for r in tercile]
            mean_diff, t_stat, p_value = _paired_significance(baseline_diffs)
            result[label] = {
                "n": len(tercile),
                "avg_gap": round(sum(r[1] for r in tercile) / len(tercile), 2),
                "baseline_p80_hit_rate": _hit_rate_summary(tercile, 2),
                "adjusted_p80_hit_rate": _hit_rate_summary(tercile, 3),
                "baseline_minus_adjusted_hit_rate_diff": round(mean_diff, 4),
                "t_stat": round(t_stat, 4) if t_stat is not None else None,
                "p_value": round(p_value, 4) if p_value is not None else None,
            }

    return result


def run_dst_matchup_backtest(source_slate_id, seasons=None, engine=None):
    """Does a DST-specific ceiling adjustment - reusing the exact same
    validated boost-only mechanism as _apply_game_environment_adjustment,
    just fed the real OPPONENT's implied total instead of the DST's own
    team's - actually improve real P90 calibration for DST specifically?

    Motivated by two real findings this session: (1) two real weeks of
    contest-standings data show real GPP winners concentrate heavily on
    just 1-2 real "best matchup" DSTs a week, not spread evenly across all
    32 - matching data/ownership_calibration.py's own already-documented
    finding that real DST ownership tracks matchup quality, not raw
    points-per-salary; (2) the CURRENTLY SHIPPED Vegas ceiling adjustment
    (models/projections.py's generate_projections) applies to DST using
    the DST's own team's implied total - the wrong side of the ball for a
    defense, whose real upside comes from the OPPONENT's offense being
    bad, not its own offense scoring a lot. That current behavior WAS part
    of run_game_environment_backtest_comparison's real pooled sample (DST
    is included there), so it is not simply "untested" - but no position-
    specific breakout of that pooled result has ever been run, so whether
    the current own-team-based DST treatment actually helps, hurts, or
    just rides along with the skill-position effect was a real unknown
    before this function existed.

    Compares three real, no-lookahead variants for every real historical
    DST player-week:
      - baseline: raw historical-percentile proj_ceiling, no Vegas
        adjustment at all.
      - current (shipped today): _apply_game_environment_adjustment fed
        the DST's own team's real implied total - reproduces exactly what
        generate_projections already does for a real DST row.
      - proposed: the SAME real, already-validated boost-only mechanism
        and coefficient (GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT is not
        re-tuned or duplicated here - no new signal, per the explicit
        instruction this was built under), fed a synthetic implied_total
        constructed so the function's own internal gap
        (implied_total - league_average) equals (league_average - real
        opponent implied total) - i.e. boosts a DST's ceiling exactly when
        its real opponent's own implied total is BELOW the week's average,
        using only the opponent's own already-fetched real Vegas number.

    True target for a well-calibrated proj_ceiling (90th percentile) is a
    real 10% P90 hit rate, matching run_game_environment_backtest_
    comparison's own established standard for this exact mechanism.
    """
    engine = engine or get_engine()
    from models.matchups import _opponents_for_week  # local import - avoids a module-load cycle (models/matchups.py imports FROM this module)

    slate_players = load_slate_pool(source_slate_id, engine)
    dst_players = [p for p in slate_players if p["position"] == "DST"]
    if not dst_players:
        raise ValueError(f"No real DST rows found in slate_player_pool for slate {source_slate_id}")
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(dst_players, engine)
    players_by_id = {p["player_id"]: p for p in dst_players}

    weeks = _available_weeks(engine)
    if seasons is not None:
        weeks = [(s, w) for s, w in weeks if s in seasons]

    try:
        implied_totals_by_team_season_week = _team_implied_totals(fetch_schedules())
    except Exception:
        implied_totals_by_team_season_week = {}

    records = []  # (gap_or_None, baseline_hit, current_hit, proposed_hit)
    weeks_evaluated = 0

    for season, week in weeks:
        history = _load_recent_stats(gsis_by_dk_id.values(), engine, before=(season, week))
        actual_points, _ = load_actual_scores(dst_players, season, week, engine)
        teams_by_gsis = _teams_for_week(gsis_by_dk_id.values(), season, week, engine)
        opponents_by_gsis = _opponents_for_week(gsis_by_dk_id.values(), season, week, engine)

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
            games = history.get(gsis_id)
            if not games:
                continue

            proj = _project_from_history(games, "DST")
            actual = actual_points[dk_id]
            baseline_hit = 1 if actual >= proj["proj_ceiling"] else 0

            real_team = teams_by_gsis.get(gsis_id)
            own_implied_total = implied_totals_by_team.get(real_team) if real_team else None
            if own_implied_total is not None and league_average is not None:
                current_adjusted = _apply_game_environment_adjustment(proj, own_implied_total, league_average)
            else:
                current_adjusted = proj
            current_hit = 1 if actual >= current_adjusted["proj_ceiling"] else 0

            opponent = opponents_by_gsis.get(gsis_id)
            opponent_implied_total = implied_totals_by_team.get(opponent) if opponent else None
            gap = None
            if opponent_implied_total is not None and league_average is not None:
                gap = league_average - opponent_implied_total
                synthetic_implied_total = league_average + gap
                proposed_adjusted = _apply_game_environment_adjustment(proj, synthetic_implied_total, league_average)
            else:
                proposed_adjusted = proj
            proposed_hit = 1 if actual >= proposed_adjusted["proj_ceiling"] else 0

            records.append((gap, baseline_hit, current_hit, proposed_hit))
            week_had_data = True

        if week_had_data:
            weeks_evaluated += 1

    if not records:
        raise ValueError(f"No real DST player-weeks could be evaluated for slate {source_slate_id}")

    def _hit_rate_summary(rows, hit_index):
        hits = sum(r[hit_index] for r in rows)
        return {"hits": hits, "opportunities": len(rows), "rate": round(hits / len(rows), 4) if rows else None}

    with_gap = [r for r in records if r[0] is not None]
    if len(with_gap) < 20:
        raise ValueError(
            f"Only {len(with_gap)} real DST player-weeks had both a real opponent and league-average "
            "implied total - too few to trust a real hit-rate comparison"
        )

    baseline_vs_proposed_diffs = [r[1] - r[3] for r in with_gap]
    current_vs_proposed_diffs = [r[2] - r[3] for r in with_gap]
    mean_diff_bp, t_bp, p_bp = _paired_significance(baseline_vs_proposed_diffs)
    mean_diff_cp, t_cp, p_cp = _paired_significance(current_vs_proposed_diffs)

    return {
        "weeks_evaluated": weeks_evaluated,
        "dst_player_weeks_evaluated": len(records),
        "dst_player_weeks_with_real_opponent_implied_total": len(with_gap),
        "true_target_p90_hit_rate": 0.10,
        "baseline_p90_hit_rate": _hit_rate_summary(with_gap, 1),
        "current_shipped_p90_hit_rate": _hit_rate_summary(with_gap, 2),
        "proposed_opponent_based_p90_hit_rate": _hit_rate_summary(with_gap, 3),
        "baseline_minus_proposed_diff": round(mean_diff_bp, 4),
        "baseline_vs_proposed_t_stat": round(t_bp, 4) if t_bp is not None else None,
        "baseline_vs_proposed_p_value": round(p_bp, 4) if p_bp is not None else None,
        "current_minus_proposed_diff": round(mean_diff_cp, 4),
        "current_vs_proposed_t_stat": round(t_cp, 4) if t_cp is not None else None,
        "current_vs_proposed_p_value": round(p_cp, 4) if p_cp is not None else None,
    }
