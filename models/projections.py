import json
import math
from collections import defaultdict

from sqlalchemy import text

from data.player_crosswalk import resolve_dk_players_to_gsis
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


def generate_projections(slate_id, engine=None):
    engine = engine or get_engine()

    with engine.connect() as conn:
        players = conn.execute(
            text("SELECT player_id, name, position, team FROM slate_player_pool WHERE slate_id = :slate_id"),
            {"slate_id": slate_id},
        ).mappings().fetchall()

    gsis_by_dk_id, unmatched, ambiguous = resolve_dk_players_to_gsis(players, engine)
    history_by_gsis = _load_recent_stats(gsis_by_dk_id.values(), engine)

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
            proj = _project_from_history(games, player["position"])
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
