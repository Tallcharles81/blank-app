import gzip
import io
import logging

import pandas as pd
import requests
from sqlalchemy import text

from db.migrate import get_engine

logger = logging.getLogger(__name__)

NFLVERSE_RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
REQUEST_TIMEOUT_SECONDS = 60

# Confirmed by hand against https://github.com/nflverse/nflverse-data/releases
# before writing this module (nflverse has renamed/retired release tags before,
# e.g. the legacy "player_stats" naming stops at 2022 in favor of per-season
# files under the same tag - these are the tags/filenames that actually exist
# today, not assumed ones):
#   - player_stats/player_stats_{season}.csv.gz  (weekly, REG+POST, gsis_id-keyed)
#   - snap_counts/snap_counts_{season}.csv.gz    (weekly, pfr_player_id-keyed)
#   - injuries/injuries_{season}.csv.gz          (weekly, gsis_id-keyed)
#   - players/players.csv.gz                     (single file, gsis_id<->pfr_id crosswalk)
#   - schedules/games.csv.gz                     (single all-seasons file - note the
#     asset is named "games.csv.gz", not "schedules.csv.gz", despite the tag name)
# A given season's file may simply not exist yet (e.g. player_stats has no 2025/2026
# release as of writing, since nflfastR can't compute weekly stats for games that
# haven't been played) - callers must treat a 404 as "skip this season", not an error.


def _download_csv(tag, filename):
    url = f"{NFLVERSE_RELEASE_BASE}/{tag}/{filename}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    content = response.content
    if filename.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            raise RuntimeError(f"Failed to gunzip {url}: {exc}") from exc

    try:
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Failed to parse {url} as CSV: {exc}") from exc


def _fetch_per_season(tag, filename_template, seasons):
    frames = []
    for season in seasons:
        try:
            df = _download_csv(tag, filename_template.format(season=season))
        except RuntimeError as exc:
            # Most commonly a season that hasn't been played yet (no release
            # asset exists) - skip it rather than fail the whole fetch over one
            # missing season. Seasons that DID download successfully still get
            # used; this season's existing DB rows (if any) are left untouched.
            logger.warning("Skipping %s season %s: %s", tag, season, exc)
            continue
        frames.append(df)
    return frames


def fetch_player_stats(seasons):
    frames = _fetch_per_season("player_stats", "player_stats_{season}.csv.gz", seasons)
    if not frames:
        raise RuntimeError(f"No player_stats data could be fetched for any of seasons {seasons}")
    df = pd.concat(frames, ignore_index=True)
    # "week" numbering restarts each season_type (REG week 1 and POST week 1 both
    # exist), which would collide with player_weekly_stats' (player_id, season,
    # week) primary key for anyone who made the playoffs - keep regular season
    # only, which also matches how DK slates number their weeks.
    return df[df["season_type"] == "REG"].copy()


def fetch_injuries(seasons):
    frames = _fetch_per_season("injuries", "injuries_{season}.csv.gz", seasons)
    if not frames:
        raise RuntimeError(f"No injuries data could be fetched for any of seasons {seasons}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["game_type"] == "REG"].copy()
    # Multiple injury reports get filed across a week (Wed/Thu/Fri practice
    # reports) - keep only the most recent one per player-week, since that's the
    # one that actually reflects their final status for that week's games.
    df = df.sort_values("date_modified").drop_duplicates(subset=["gsis_id", "season", "week"], keep="last")
    return df


def fetch_snap_counts(seasons):
    frames = _fetch_per_season("snap_counts", "snap_counts_{season}.csv.gz", seasons)
    if not frames:
        raise RuntimeError(f"No snap_counts data could be fetched for any of seasons {seasons}")
    return pd.concat(frames, ignore_index=True)


def fetch_players_crosswalk():
    return _download_csv("players", "players.csv.gz")


def fetch_schedules():
    return _download_csv("schedules", "games.csv.gz")


def fetch_stats_team_week(seasons):
    # Confirmed to exist by direct download (stats_team_week_{season}.csv.gz),
    # despite not appearing in the truncated release-asset listing checked
    # earlier - team-level, per-week defense/special-teams box score stats,
    # which is what DK's DST scoring needs and player_stats (per-player,
    # offense-only) doesn't have at all.
    frames = _fetch_per_season("stats_team", "stats_team_week_{season}.csv.gz", seasons)
    if not frames:
        raise RuntimeError(f"No stats_team_week data could be fetched for any of seasons {seasons}")
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"].copy()


def _team_points_allowed(schedules_df):
    df = schedules_df.dropna(subset=["home_score", "away_score"])
    home = df[["season", "week", "home_team", "away_score"]].rename(
        columns={"home_team": "team", "away_score": "points_allowed"}
    )
    away = df[["season", "week", "away_team", "home_score"]].rename(
        columns={"away_team": "team", "home_score": "points_allowed"}
    )
    combined = pd.concat([home, away], ignore_index=True)
    return {(row.team, row.season, row.week): row.points_allowed for row in combined.itertuples()}


# DraftKings' published Classic-contest DST scoring: tiers are keyed by the
# upper bound of a "points allowed" bracket, first match wins.
_POINTS_ALLOWED_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1)]
_POINTS_ALLOWED_BLOWOUT_PENALTY = -4


def _points_allowed_bonus(points_allowed):
    for upper_bound, bonus in _POINTS_ALLOWED_TIERS:
        if points_allowed <= upper_bound:
            return bonus
    return _POINTS_ALLOWED_BLOWOUT_PENALTY


def _dst_fantasy_points(team_row, points_allowed):
    # Verified against real stats_team_week data before relying on it: def_tds,
    # fumble_recovery_tds, and special_teams_tds are independent, non-
    # overlapping touchdown categories (e.g. def_tds never exceeds
    # def_interceptions across a full season, and a real game exists with both
    # a def_td and a fumble_recovery_td counted separately in the same row) -
    # summing all three double-counts nothing.
    score = (
        1 * team_row.def_sacks
        + 2 * team_row.def_interceptions
        + 2 * team_row.fumble_recovery_opp
        + 6 * (team_row.def_tds + team_row.fumble_recovery_tds + team_row.special_teams_tds)
        + 2 * team_row.def_safeties
        + 2 * (team_row.def_punt_blocks + team_row.def_pat_blocks + team_row.def_fg_blocks)
    )
    if points_allowed is not None:
        score += _points_allowed_bonus(points_allowed)
    return round(float(score), 2)


def _team_implied_totals(schedules_df):
    # spread_line is the HOME team's favored margin (confirmed empirically: it
    # correlates positively with home_score - away_score across real games, not
    # assumed from nflverse's docs) - so each team's Vegas-implied point total is
    # half the game total, adjusted by half the spread in their favor/against.
    df = schedules_df.dropna(subset=["spread_line", "total_line"])

    home = df[["season", "week", "home_team", "spread_line", "total_line"]].copy()
    home["implied_total"] = home["total_line"] / 2 + home["spread_line"] / 2
    home = home.rename(columns={"home_team": "team"})

    away = df[["season", "week", "away_team", "spread_line", "total_line"]].copy()
    away["implied_total"] = away["total_line"] / 2 - away["spread_line"] / 2
    away = away.rename(columns={"away_team": "team"})

    combined = pd.concat([home, away], ignore_index=True)
    return {(row.team, row.season, row.week): row.implied_total for row in combined.itertuples()}


def _clear_and_insert_season(conn, season, rows):
    # Delete-then-insert in the same transaction as the caller's `conn`, so a
    # failure partway through this season's insert rolls back the delete too -
    # we never end up with a season's real data half-written, or worse, deleted
    # with nothing to replace it. Scoped to position != 'DST' so this never
    # wipes out the DST rows refresh_dst_weekly_stats() writes to this same
    # table - the two refreshes must not delete each other's data.
    conn.execute(
        text("DELETE FROM player_weekly_stats WHERE season = :season AND position != 'DST'"),
        {"season": season},
    )
    if not rows:
        return
    conn.execute(
        text(
            """
            INSERT INTO player_weekly_stats (
                player_id, player_name, position, team, season, week,
                targets, target_share, air_yards_share, carries,
                rushing_yards, receiving_yards, fantasy_points_ppr,
                opponent, snap_pct, injury_status, vegas_implied_total
            ) VALUES (
                :player_id, :player_name, :position, :team, :season, :week,
                :targets, :target_share, :air_yards_share, :carries,
                :rushing_yards, :receiving_yards, :fantasy_points_ppr,
                :opponent, :snap_pct, :injury_status, :vegas_implied_total
            )
            """
        ),
        rows,
    )


def _clear_and_insert_dst_season(conn, season, rows):
    # Scoped to position = 'DST' only - the counterpart to
    # _clear_and_insert_season's "!= 'DST'" scoping, so refreshing DST never
    # touches the offensive player rows in the same table.
    conn.execute(
        text("DELETE FROM player_weekly_stats WHERE season = :season AND position = 'DST'"),
        {"season": season},
    )
    if not rows:
        return
    conn.execute(
        text(
            """
            INSERT INTO player_weekly_stats (
                player_id, player_name, position, team, season, week,
                fantasy_points_ppr, opponent, vegas_implied_total
            ) VALUES (
                :player_id, :player_name, :position, :team, :season, :week,
                :fantasy_points_ppr, :opponent, :vegas_implied_total
            )
            """
        ),
        rows,
    )


def refresh_dst_weekly_stats(seasons, engine=None):
    """Replace player_weekly_stats' DST rows for `seasons` with real nflverse data.

    DST fantasy points are computed here (see _dst_fantasy_points), not sourced
    directly, since no nflverse release publishes DK's DST fantasy score - only
    the raw box-score categories DK's formula is built from. player_id is a
    synthetic f"DST_{team}" (team defenses aren't people, so there's no GSIS id
    to reuse); data/player_crosswalk.py resolves DK's DST rows to this same id
    by team code, not name matching.
    """
    engine = engine or get_engine()

    team_stats_df = fetch_stats_team_week(seasons)
    schedules_df = fetch_schedules()
    points_allowed_by_key = _team_points_allowed(schedules_df)

    vegas_by_team_key = {}
    try:
        vegas_by_team_key = _team_implied_totals(schedules_df)
    except KeyError as exc:
        logger.warning("Continuing without vegas_implied_total: %s", exc)

    rows_by_season = {}
    for row in team_stats_df.itertuples():
        points_allowed = points_allowed_by_key.get((row.team, row.season, row.week))
        rows_by_season.setdefault(row.season, []).append(
            {
                "player_id": f"DST_{row.team}",
                "player_name": f"{row.team} DST",
                "position": "DST",
                "team": row.team,
                "season": row.season,
                "week": row.week,
                "opponent": row.opponent_team,
                "fantasy_points_ppr": _dst_fantasy_points(row, points_allowed),
                "vegas_implied_total": _nan_to_none(vegas_by_team_key.get((row.team, row.season, row.week))),
            }
        )

    row_counts = {}
    for season, rows in rows_by_season.items():
        with engine.begin() as conn:
            _clear_and_insert_dst_season(conn, season, rows)
        row_counts[season] = len(rows)

    return row_counts


def refresh_player_weekly_stats(seasons, engine=None):
    """Replace player_weekly_stats' rows for `seasons` with real nflverse data.

    Returns {season: row_count} for seasons actually refreshed. A season is left
    completely untouched (old data and all) if its player_stats fetch fails -
    see _fetch_per_season - so a bad download never leaves that season's table
    rows deleted with nothing to replace them, nor a silent mix of stale rows
    sitting next to fresh ones.
    """
    engine = engine or get_engine()

    stats_df = fetch_player_stats(seasons)
    fetched_seasons = sorted(stats_df["season"].unique().tolist())

    try:
        injuries_df = fetch_injuries(fetched_seasons)
    except RuntimeError as exc:
        logger.warning("Continuing without injury status: %s", exc)
        injuries_df = None

    snap_pct_by_key = {}
    try:
        snaps_df = fetch_snap_counts(fetched_seasons)
        crosswalk_df = fetch_players_crosswalk()
        pfr_to_gsis = dict(zip(crosswalk_df["pfr_id"], crosswalk_df["gsis_id"]))
        snaps_df = snaps_df[snaps_df["game_type"] == "REG"].copy()
        snaps_df["gsis_id"] = snaps_df["pfr_player_id"].map(pfr_to_gsis)
        snaps_df = snaps_df.dropna(subset=["gsis_id"])
        for row in snaps_df.itertuples():
            snap_pct_by_key[(row.gsis_id, row.season, row.week)] = row.offense_pct
    except (RuntimeError, KeyError) as exc:
        logger.warning("Continuing without snap_pct: %s", exc)

    injury_by_key = {}
    if injuries_df is not None:
        for row in injuries_df.itertuples():
            injury_by_key[(row.gsis_id, row.season, row.week)] = row.report_status

    vegas_by_team_key = {}
    try:
        vegas_by_team_key = _team_implied_totals(fetch_schedules())
    except (RuntimeError, KeyError) as exc:
        logger.warning("Continuing without vegas_implied_total: %s", exc)

    rows_by_season = {}
    for row in stats_df.itertuples():
        key = (row.player_id, row.season, row.week)
        team_key = (row.recent_team, row.season, row.week)
        rows_by_season.setdefault(row.season, []).append(
            {
                "player_id": row.player_id,
                "player_name": row.player_display_name,
                "position": row.position,
                "team": row.recent_team,
                "season": row.season,
                "week": row.week,
                "targets": _nan_to_none(row.targets),
                "target_share": _nan_to_none(row.target_share),
                "air_yards_share": _nan_to_none(row.air_yards_share),
                "carries": _nan_to_none(row.carries),
                "rushing_yards": _nan_to_none(row.rushing_yards),
                "receiving_yards": _nan_to_none(row.receiving_yards),
                "fantasy_points_ppr": _nan_to_none(row.fantasy_points_ppr),
                "opponent": row.opponent_team,
                "snap_pct": _nan_to_none(snap_pct_by_key.get(key)),
                "injury_status": _nan_to_none(injury_by_key.get(key)),
                "vegas_implied_total": _nan_to_none(vegas_by_team_key.get(team_key)),
            }
        )

    # One transaction per season, not one big transaction for the whole run - a
    # DB-level failure partway through one season's insert rolls back only that
    # season, instead of also undoing another season that already wrote cleanly.
    row_counts = {}
    for season, rows in rows_by_season.items():
        with engine.begin() as conn:
            _clear_and_insert_season(conn, season, rows)
        row_counts[season] = len(rows)

    return row_counts


def _nan_to_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
