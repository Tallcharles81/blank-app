import json
import math
from collections import defaultdict

from sqlalchemy import text

from data.nflverse_fetch import fetch_team_implied_totals
from data.player_availability import resolve_slate_season_week
from data.player_crosswalk import resolve_dk_players_to_gsis
from data.pre_lock_check import _load_recent_usage_batch
from db.migrate import get_engine

# player_weekly_stats.player_id is nflverse's GSIS ID (e.g. "00-0034857"), which
# does not match slate_player_pool.player_id's DraftKings numeric ID (e.g.
# "44132656") - there is no published GSIS<->DK crosswalk, so
# generate_projections() below resolves DK player_ids to GSIS ids by name+
# position via data/player_crosswalk.py before querying player_weekly_stats.

MAX_HISTORY_WEEKS = 10
# A player whose most recent recorded game is older than this many real
# played weeks (counted across season boundaries, not raw week-number
# arithmetic - see _real_week_sequence) is treated as having no usable
# history, not silently projected from a stale snapshot. 4 weeks is enough
# to cover a normal bye week plus a short DNP stretch without flagging every
# ordinary absence, while still catching a real multi-week-plus injury
# (exactly the Christian McCaffrey 2024/Brandon Aiyuk case this was added
# for - see _load_recent_stats).
MAX_STALENESS_WEEKS = 4
MIN_GAMES_FOR_OWN_VARIANCE = 3
# Recent weeks matter more than older ones - an exponential decay with a 4-week
# half-life weights last week roughly 1.19x more than 4 weeks ago.
RECENCY_HALF_LIFE_WEEKS = 4

# Fallback coefficient of variation (stdev / mean) by position, used only when a
# player doesn't have enough games of their own history to compute a reliable
# variance (rookies, new starters). These are rough, commonly-cited approximations
# for week-to-week fantasy scoring volatility by position - a real historical
# dataset should replace these once enough games exist for every player.
POSITION_COV_FALLBACK = {"QB": 0.35, "RB": 0.45, "WR": 0.50, "TE": 0.55, "DST": 0.60}
DEFAULT_COV_FALLBACK = 0.5

# z-scores for a standard normal distribution at each percentile.
PERCENTILE_Z = {"10": -1.2816, "25": -0.6745, "50": 0.0, "75": 0.6745, "90": 1.2816}
# Fantasy scoring is right-skewed (huge games happen more often than a symmetric
# distribution predicts) - nudge percentiles above the median up to reflect that,
# and leave the median/below alone since busts are bounded near zero anyway.
CEILING_Z_BOOST = 0.3

# Real, current-week Vegas lines (spread_line/total_line, via
# data/nflverse_fetch.py's fetch_team_implied_totals) were being fetched and
# stored in player_weekly_stats.vegas_implied_total this whole time but never
# actually read by anything in this module - a real signal sitting unused. A
# team's implied point total is a standard, well-established correlate of
# offensive ceiling outcomes (more expected plays and scoring in a real
# projected shootout, fewer in a real projected defensive grind) - applied
# here ONLY to the ceiling-side percentiles (75th/90th, i.e. proj_ceiling),
# never floor/median, matching "a shootout raises the boom-game upside" more
# precisely than "a shootout raises the median outcome," which isn't the same
# claim. Scaled off the GAP between this player's own team's implied total
# and the field's average for the week, not the raw number, so an average-
# implied-total player's ceiling is unaffected either way.
#
# BACKTESTED (models/calibration.py::run_game_environment_backtest_comparison,
# 54 real historical weeks, 14,547 real played player-weeks, split into
# terciles by the real gap this coefficient scales on). First version was
# symmetric (also scaled ceiling DOWN for a below-average implied total)
# and came back MIXED, not a clean pass:
#   - High-implied-total tercile (avg gap +4.16, real projected shootouts,
#     n=4,818): baseline P90 hit rate 14.78% -> adjusted 12.72% - moved
#     TOWARD the true 10% target, the intended direction, real effect
#     (t=10.05, p<0.0001).
#   - Low-implied-total tercile (avg gap -3.84, real projected grinds,
#     n=4,818): baseline P90 hit rate 15.34% -> adjusted 16.92% - moved
#     AWAY from the 10% target, the OPPOSITE of the intended direction,
#     also real and highly significant (t=-8.79, p<0.0001), not noise.
# Fixed to boost-only (see _apply_game_environment_adjustment: gap <= 0 is
# now a no-op) based on that result, then RE-backtested to confirm rather
# than assumed: low tercile now shows EXACTLY zero change (736/4818 hits
# under both baseline and adjusted - no variance in the difference at all,
# t_stat/p_value both None), high tercile keeps its same validated
# improvement, and the pooled P90 hit rate improved too (15.32% -> 14.57%,
# closer to the true 10% target than before the fix). Unlike models/
# matchups.py's TE adjustment, which cleared its backtest before shipping,
# this one shipped first, was backtested after, found mixed, and was fixed
# and re-verified in the same pass - all three steps disclosed here rather
# than only the final "it's fine now."
#
# Re-run again after data/nflverse_fetch.py's real-DK-scoring fix (the
# historical actual_score ground truth this whole backtest depends on was
# previously generic-PPR-derived, not real DK Classic scoring - missing
# DK's real yardage bonuses and using -2 instead of DK's real -1 for INTs/
# fumbles lost). High tercile: baseline 14.88% -> adjusted 12.93% (t=9.79,
# p<0.0001, was 14.78%->12.72%). Low tercile and pooled numbers above are
# already the post-fix values. Same conclusion holds - the fix didn't change
# which direction this adjustment should go, only these numbers by tenths of
# a point.
#
# SPLIT PER PERCENTILE (previously one shared coefficient for both "75" and
# "90"): a real held-out check (models/calibration.py::
# run_game_environment_p80_hit_rate_backtest) found the single shared 0.015
# correctly helps P90 (its own tuned target, above) but actively HURTS P80 -
# interpolated between "75" and "90" the same way run_weekly_calibration's
# own p80_hits already does, true target 20%, not 10%. High-implied-total
# tercile: baseline P80 hit rate 20.32% -> adjusted (shared 0.015) 18.61%,
# moving AWAY from target, real and significant (t=8.32, p<0.0001).
#
# Real sweep across candidate "75"-only coefficients (n=12,018 pooled /
# 3,971 high-tercile real player-weeks, dk_sunday_2026_09_20) confirmed a
# clean, monotonic real relationship - every step down from 0.015 toward
# 0.0 moved the real P80 hit rate closer to its true 20% target, with 0.0
# itself the best of everything tested (pooled 19.67%->20.04%, high tercile
# 18.61%->19.52%). "75" gets NO boost at all now; "90" keeps its own
# already-validated 0.015 unchanged - this doesn't reopen or re-tune the
# P90 result above, it only stops that same coefficient from being wrongly
# reused on a percentile it was never tuned for.
GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT = {"75": 0.0, "90": 0.015}

# DraftKings' real Showdown Captain slot scores the SAME real outcome at
# 1.5x - a real fact already relied on elsewhere (models/optimizer.py's
# _solve_showdown deliberately does NOT re-apply this to DK's own
# AvgPointsPerGame/salary columns, since DK already bakes it in there) but
# never applied to THIS module's own generated proj_floor/median/ceiling,
# which are computed from a player's real historical fantasy-point history
# independent of DK's columns entirely, and are IDENTICAL for a player's
# CPT and FLEX row (same real gsis_id, same history) before this scaling.
# Caught for real: a first real Showdown build using these projections
# unscaled put a $1,500 punt play at Captain in 4 of 5 lineups, real
# stud plays. Paying 1.5x salary for 1.5x projected points is normally a
# strong real captain choice - the solver could never see that trade at
# all while both rows carried an identical points estimate.
SHOWDOWN_CAPTAIN_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# NOT IMPLEMENTED - a real future upgrade path, not urgent.
#
# _project_from_history()/_apply_game_environment_adjustment() below combine
# real signals (own recency-weighted history, real Vegas-implied-total gap)
# through hand-picked formulas (RECENCY_HALF_LIFE_WEEKS's decay constant,
# GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT's coefficient, CEILING_Z_BOOST's
# skew). A trained model (real regression/gradient-boosted trees on real
# historical (features -> actual outcome) pairs, not an LLM) could in
# principle blend these same signals - plus others already computed
# elsewhere in this codebase but not fed into projections today, e.g.
# models/matchups.py's opponent-strength read - into one calibrated
# projection the way real commercial projection sites do, instead of the
# formulas here being fixed constants tuned once and left alone.
#
# This is a real, legitimate upgrade path ONLY once there is enough real
# historical data to train and validate it properly, and it must clear the
# exact same bar as every other change to this codebase: backtested against
# the CURRENT formula-based approach using this module's own real methods
# (models/calibration.py's run_game_environment_p80_hit_rate_backtest and
# friends), on the same real historical player-weeks, and kept only if it
# measurably improves real calibration (hit rates closer to their true
# target percentages, not just a lower training-set error). Not attempted
# here - flagged as a real direction, not built speculatively.
# ---------------------------------------------------------------------------


def _scale_projection(proj, factor):
    return {
        "proj_floor": round(proj["proj_floor"] * factor, 2),
        "proj_median": round(proj["proj_median"] * factor, 2),
        "proj_ceiling": round(proj["proj_ceiling"] * factor, 2),
        "proj_percentiles": {label: round(value * factor, 2) for label, value in proj["proj_percentiles"].items()},
    }


def _real_week_sequence(engine):
    # The full, real ordered sequence of (season, week) pairs that actually
    # have data - used to count staleness in real, played NFL weeks rather
    # than raw week-number arithmetic, which breaks across a season boundary
    # (2025 week 18 -> 2026 week 1 is one real week later, not a ~35-week gap).
    query = text("SELECT DISTINCT season, week FROM player_weekly_stats ORDER BY season, week")
    with engine.connect() as conn:
        return [(row.season, row.week) for row in conn.execute(query)]


def _staleness_reference_week(week_sequence, before):
    # The reference point staleness is measured against: the latest real
    # completed week strictly before `before` (backtesting), or the latest
    # real completed week in the whole table at all (live mode, before=None -
    # i.e. "as of right now"). Returns None if no such week exists.
    candidates = [w for w in week_sequence if before is None or w < before] if before is not None else week_sequence
    return candidates[-1] if candidates else None


def _load_recent_stats(player_ids, engine, max_weeks=MAX_HISTORY_WEEKS, before=None, max_staleness_weeks=MAX_STALENESS_WEEKS):
    # A single windowed query instead of one query per player - RANK gives each
    # player's own games a recency rank so we can cap history length per player
    # without N+1 round trips for a slate of hundreds of players.
    #
    # `before`, an optional (season, week) cutoff, exists for backtesting
    # (models/backtest.py): without it, "recent" always means the most recent
    # data in the table, which for a past week under test would include that
    # week itself and everything after it - lookahead bias that would make a
    # backtest meaningless (projecting a week partly from its own result).
    cutoff_sql = ""
    params = {"player_ids": list(player_ids), "max_weeks": max_weeks}
    if before is not None:
        cutoff_sql = "AND (season < :before_season OR (season = :before_season AND week < :before_week))"
        params["before_season"], params["before_week"] = before

    query = text(
        f"""
        SELECT player_id, season, week, fantasy_points_ppr, recency_rank FROM (
            SELECT player_id, season, week, fantasy_points_ppr,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id ORDER BY season DESC, week DESC
                   ) AS recency_rank
            FROM player_weekly_stats
            WHERE player_id = ANY(:player_ids) AND fantasy_points_ppr IS NOT NULL
            {cutoff_sql}
        ) ranked
        WHERE recency_rank <= :max_weeks
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    by_player = defaultdict(list)
    most_recent_week_by_player = {}
    for row in rows:
        by_player[row.player_id].append((row.recency_rank, float(row.fantasy_points_ppr)))
        if row.recency_rank == 1:
            most_recent_week_by_player[row.player_id] = (row.season, row.week)

    # Staleness check: a player whose most recent available game is older
    # than max_staleness_weeks real played weeks is treated as having no
    # usable history at all, the same as a player with zero games - not
    # silently projected from a stale snapshot. Caught for real: Christian
    # McCaffrey's 2024 week-8 as-of projection was being built entirely from
    # late-2023 games (his real 2024 games didn't start until week 10, after
    # a season-long Achilles/PCL injury) with no signal anywhere that the
    # data was ~30 real weeks old - and on the live slate right now, Brandon
    # Aiyuk (last real game 2024 week 7, torn ACL/MCL/meniscus, hasn't played
    # since, his own GM has said he'll never play for the team again) was
    # sitting in projections with a real proj_ceiling of 20.72 at a $3,000
    # salary - exactly the "cheap ceiling dart" a GPP optimizer would love,
    # built from data 30 real weeks stale.
    week_sequence = _real_week_sequence(engine)
    reference_week = _staleness_reference_week(week_sequence, before)
    week_index = {w: i for i, w in enumerate(week_sequence)}

    fresh_players = {}
    for pid, games in by_player.items():
        most_recent = most_recent_week_by_player.get(pid)
        if reference_week is not None and most_recent in week_index and reference_week in week_index:
            weeks_stale = week_index[reference_week] - week_index[most_recent]
            if weeks_stale > max_staleness_weeks:
                continue  # too stale - treated as no usable history, not silently used
        fresh_players[pid] = games

    # recency_rank 1 = most recent game; sort so index 0 is always the most recent.
    return {pid: [pts for _, pts in sorted(games)] for pid, games in fresh_players.items()}


def _weighted_median(games_most_recent_first):
    # This used to compute a weighted MEAN despite the name. Fantasy scores
    # are right-skewed (a few boom games among many modest ones - see
    # CEILING_Z_BOOST below, which already exists to account for that skew
    # elsewhere), and the mean of a right-skewed sample sits above its true
    # median because outlier boom games pull an average up more than they
    # move the middle value. That mismatch was the real cause of a calibration
    # bias caught via models/calibration.py: backtested real outcomes cleared
    # the claimed "median" only ~32% of the time, not ~50%, and every
    # percentile built on top of that inflated anchor missed high in the same
    # direction. A true weighted median - the value where cumulative recency
    # weight first crosses the halfway point - isn't pulled by outliers the
    # way an average is.
    decay = math.log(2) / RECENCY_HALF_LIFE_WEEKS
    weighted = sorted(
        ((points, math.exp(-decay * i)) for i, points in enumerate(games_most_recent_first)),
        key=lambda pw: pw[0],
    )
    half_weight = sum(w for _, w in weighted) / 2
    cumulative = 0.0
    for points, weight in weighted:
        cumulative += weight
        if cumulative >= half_weight:
            return points
    return weighted[-1][0]


def _sample_stdev(games, mean):
    if len(games) < 2:
        return None
    variance = sum((p - mean) ** 2 for p in games) / (len(games) - 1)
    return math.sqrt(variance)


def _project_from_history(games, position):
    median = _weighted_median(games)

    stdev = _sample_stdev(games, median) if len(games) >= MIN_GAMES_FOR_OWN_VARIANCE else None
    if not stdev:
        stdev = median * POSITION_COV_FALLBACK.get(position, DEFAULT_COV_FALLBACK)

    percentiles = {}
    for label, z in PERCENTILE_Z.items():
        z_adjusted = z + CEILING_Z_BOOST if z > 0 else z
        percentiles[label] = round(max(median + z_adjusted * stdev, 0.0), 2)

    return {
        "proj_floor": percentiles["25"],
        "proj_median": percentiles["50"],
        "proj_ceiling": percentiles["90"],
        "proj_percentiles": percentiles,
    }


def _apply_game_environment_adjustment(proj, implied_total, league_average_implied_total):
    """Nudge only the ceiling-side percentiles (PERCENTILE_Z labels with
    z > 0 - "75"/"90", i.e. proj_ceiling) UP for a real projected shootout
    (this player's team's current-week Vegas-implied total above the
    week's average). floor/median are untouched.

    Boost-only, not the originally-shipped symmetric version that also
    scaled ceiling DOWN for a below-average implied total - see
    GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT for the real backtest that
    found the downward half backwards (it moved P90 calibration away from
    the true target, not toward it, a real and significant effect, not
    noise) while the upward half was validated. A team at or below the
    week's average is left completely unchanged now, rather than
    penalized on a direction the data didn't support.

    Per-percentile coefficient (GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT is
    a dict keyed by label, not one shared scalar) - a real backtest found
    the single-coefficient version correctly helped P90 (what it was tuned
    against) while actively hurting P80's own, different true target. A
    label missing from the dict gets no boost at all (0.0), the same real
    fail-safe default as models/field_simulation.py's calibrated_ownership_
    proxy for an unrecognized key.
    """
    gap = implied_total - league_average_implied_total
    if gap <= 0:
        return dict(proj)
    adjusted_percentiles = dict(proj["proj_percentiles"])
    for label, z in PERCENTILE_Z.items():
        if z > 0:
            coeff = GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT.get(label, 0.0)
            multiplier = 1.0 + coeff * gap
            adjusted_percentiles[label] = round(adjusted_percentiles[label] * multiplier, 2)
    return {**proj, "proj_ceiling": adjusted_percentiles["90"], "proj_percentiles": adjusted_percentiles}


def generate_projections(slate_id, engine=None, use_dst_opponent_matchup_adjustment=False):
    """use_dst_opponent_matchup_adjustment: opt-in, default False. Applies
    the same real, already-validated boost-only Vegas mechanism
    (_apply_game_environment_adjustment) to a DST row using its real
    OPPONENT's implied total instead of the DST's own team's - real
    backtest (models/calibration.py::run_dst_matchup_backtest, n=1326 real
    DST player-weeks) shows this beats no adjustment decisively (P90 hit
    rate 11.84% -> 10.78%, true target 10%, t=3.76, p=0.0002) and beats the
    default own-team-based version too, but only at p=0.0575 - real and
    directionally correct, not yet decisive enough to become the default,
    same standard models/matchups.py's TE adjustment was held to. Leave
    False to keep today's shipped behavior unchanged.
    """
    engine = engine or get_engine()

    with engine.connect() as conn:
        players = conn.execute(
            text("SELECT player_id, name, position, team, opponent FROM slate_player_pool WHERE slate_id = :slate_id"),
            {"slate_id": slate_id},
        ).mappings().fetchall()

    gsis_by_dk_id, unmatched, ambiguous = resolve_dk_players_to_gsis(players, engine)
    history_by_gsis = _load_recent_stats(gsis_by_dk_id.values(), engine)

    # Real position resolution for the COV fallback below (POSITION_COV_
    # FALLBACK is keyed by real position - "CPT"/"FLEX" would silently miss
    # every lookup and always fall back to DEFAULT_COV_FALLBACK for every
    # Showdown player) and for the Captain-multiplier check right after it -
    # a Showdown row's own position is "CPT"/"FLEX" regardless of what the
    # real player plays (see data/player_crosswalk.py's
    # SHOWDOWN_PSEUDO_POSITIONS), the same real gap already fixed elsewhere
    # in this codebase (data/pre_lock_check.py's hard_role_exclusions).
    usage_by_gsis = _load_recent_usage_batch(
        [gid for gid in gsis_by_dk_id.values() if gid is not None and not gid.startswith("DST_")], engine
    )

    # Real, current-week game-environment signal (see
    # GAME_ENVIRONMENT_CEILING_BOOST_PER_POINT) - best-effort per this
    # project's convention of wrapping external-data fetches in try/except:
    # projections generating successfully matters far more than this one
    # adjustment, so a fetch failure or a week with no posted lines yet
    # (empty dict back, not an error - see fetch_team_implied_totals) just
    # skips the adjustment rather than blocking projections entirely.
    implied_totals_by_team = {}
    try:
        season_week = resolve_slate_season_week(slate_id, engine)
        if season_week is not None:
            implied_totals_by_team = fetch_team_implied_totals(*season_week)
    except Exception:
        implied_totals_by_team = {}
    league_average_implied_total = (
        sum(implied_totals_by_team.values()) / len(implied_totals_by_team) if implied_totals_by_team else None
    )

    upsert_sql = text(
        """
        INSERT INTO projections (slate_id, player_id, proj_floor, proj_median, proj_ceiling, proj_percentiles)
        VALUES (:slate_id, :player_id, :proj_floor, :proj_median, :proj_ceiling, :proj_percentiles)
        ON CONFLICT (slate_id, player_id) DO UPDATE SET
            proj_floor = EXCLUDED.proj_floor,
            proj_median = EXCLUDED.proj_median,
            proj_ceiling = EXCLUDED.proj_ceiling,
            proj_percentiles = EXCLUDED.proj_percentiles
        -- Once a slate's projections are locked (locked_at set), leave them alone -
        -- re-running this shouldn't silently rewrite the numbers a lineup was built
        -- against, mirroring the DB trigger that protects locked_at itself.
        WHERE projections.locked_at IS NULL
        """
    )

    projected = 0
    skipped_no_history = []
    with engine.begin() as conn:
        for player in players:
            gsis_id = gsis_by_dk_id.get(player["player_id"])
            games = history_by_gsis.get(gsis_id) if gsis_id else None
            if not games:
                skipped_no_history.append(player["player_id"])
                continue
            if player["position"] in ("CPT", "FLEX"):
                usage_games = usage_by_gsis.get(gsis_id, [])
                real_position = usage_games[0]["position"] if usage_games else None
            else:
                real_position = player["position"]

            proj = _project_from_history(games, real_position)
            if use_dst_opponent_matchup_adjustment and real_position == "DST":
                # Opt-in only - see this function's own docstring for the
                # real backtest behind this. Synthetic implied_total so
                # _apply_game_environment_adjustment's own internal gap
                # (implied_total - league_average) comes out to
                # (league_average - opponent_implied_total) - boosts this
                # DST's ceiling exactly when its real opponent is BELOW the
                # week's average implied total, reusing that function's
                # already-validated boost-only mechanism unchanged.
                opponent_implied_total = implied_totals_by_team.get(player["opponent"])
                if opponent_implied_total is not None and league_average_implied_total is not None:
                    gap = league_average_implied_total - opponent_implied_total
                    synthetic_implied_total = league_average_implied_total + gap
                    proj = _apply_game_environment_adjustment(proj, synthetic_implied_total, league_average_implied_total)
            else:
                implied_total = implied_totals_by_team.get(player["team"])
                if implied_total is not None and league_average_implied_total is not None:
                    proj = _apply_game_environment_adjustment(proj, implied_total, league_average_implied_total)
            if player["position"] == "CPT":
                proj = _scale_projection(proj, SHOWDOWN_CAPTAIN_MULTIPLIER)
            conn.execute(
                upsert_sql,
                {
                    "slate_id": slate_id,
                    "player_id": player["player_id"],
                    "proj_floor": proj["proj_floor"],
                    "proj_median": proj["proj_median"],
                    "proj_ceiling": proj["proj_ceiling"],
                    "proj_percentiles": json.dumps(proj["proj_percentiles"]),
                },
            )
            projected += 1

    return projected, skipped_no_history, ambiguous


def lock_projections(slate_id, engine=None):
    engine = engine or get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE projections SET locked_at = now() WHERE slate_id = :slate_id AND locked_at IS NULL"),
            {"slate_id": slate_id},
        )
    return result.rowcount
