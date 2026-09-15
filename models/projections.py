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


def _load_recent_stats(player_ids, engine, max_weeks=MAX_HISTORY_WEEKS):
    # A single windowed query instead of one query per player - RANK gives each
    # player's own games a recency rank so we can cap history length per player
    # without N+1 round trips for a slate of hundreds of players.
    query = text(
        """
        SELECT player_id, fantasy_points_ppr, recency_rank FROM (
            SELECT player_id, fantasy_points_ppr,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id ORDER BY season DESC, week DESC
                   ) AS recency_rank
            FROM player_weekly_stats
            WHERE player_id = ANY(:player_ids) AND fantasy_points_ppr IS NOT NULL
        ) ranked
        WHERE recency_rank <= :max_weeks
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"player_ids": list(player_ids), "max_weeks": max_weeks}).fetchall()

    by_player = defaultdict(list)
    for row in rows:
        by_player[row.player_id].append((row.recency_rank, float(row.fantasy_points_ppr)))
    # recency_rank 1 = most recent game; sort so index 0 is always the most recent.
    return {pid: [pts for _, pts in sorted(games)] for pid, games in by_player.items()}


def _weighted_median(games_most_recent_first):
    decay = math.log(2) / RECENCY_HALF_LIFE_WEEKS
    weights = [math.exp(-decay * i) for i in range(len(games_most_recent_first))]
    return sum(p * w for p, w in zip(games_most_recent_first, weights)) / sum(weights)


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
