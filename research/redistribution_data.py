"""Data layer for the defensive-redistribution research (research/
redistribution_study.py). Downloads nflverse release files once into a
local, gitignored cache and keeps only the columns the study uses.

Sources (all public nflverse releases):
  - play-by-play (tag pbp): targets, air yards, pass location/length,
    sacks, QB hits, scrambles, rushing, TDs, red zone.
  - participation (tag pbp_participation, 2016-2025): on-field offensive
    player ids per play (-> snap and dropback participation, the route
    proxy), was_pressure, defense man/zone and coverage type.
  - FTN charting (tag ftn_charting, 2022+): blitzers, pass rushers.
  - player stats (tag stats_player): per-player weekly box scores, scored
    with the live engine's own DK formula.
  - players (tag players): gsis id -> position.
  - schedules: closing spread/total, roof, wind, temp.
  - PFR defensive advanced stats (tag pfr_advstats): per-defender coverage
    charting (targets/yards/TDs allowed).
Nothing here is written to the project database.
"""
import os
from pathlib import Path

import pandas as pd

from data.nflverse_fetch import _skill_player_fantasy_points, download_csv

CACHE_DIR = Path(os.environ.get("DK_EDGE_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "nflverse"))

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "game_date", "posteam", "defteam", "home_team",
    "away_team", "play_type", "pass", "rush", "qb_dropback", "sack", "qb_hit", "qb_scramble", "interception",
    "complete_pass", "pass_attempt", "receiver_player_id", "rusher_player_id", "passer_player_id", "air_yards",
    "yards_gained", "receiving_yards", "rushing_yards", "passing_yards", "pass_location", "pass_length",
    "pass_touchdown", "rush_touchdown", "yardline_100", "two_point_attempt", "qb_kneel", "qb_spike", "down",
    "score_differential", "spread_line", "total_line", "roof", "wind", "temp",
]
PARTICIPATION_COLUMNS = [
    "nflverse_game_id", "play_id", "offense_players", "was_pressure", "defense_man_zone_type",
    "defense_coverage_type", "number_of_pass_rushers", "time_to_throw",
]
FTN_COLUMNS = ["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "is_qb_out_of_pocket", "read_thrown"]
STATS_COLUMNS = [
    "player_id", "season", "week", "season_type", "position", "team", "opponent_team", "passing_yards",
    "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds",
    "receptions", "targets", "carries", "attempts", "special_teams_tds", "fumble_recovery_tds",
    "fumbles_lost_total", "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
]


def _cached(name, loader):
    """Pickle-cache a DataFrame under CACHE_DIR/name.pkl. The loader's own
    download errors (RuntimeError from download_csv) propagate so a caller
    can decide whether a missing season is fatal."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    df = loader()
    df.to_pickle(path)
    return df


def load_pbp(season):
    def loader():
        df = download_csv("pbp", f"play_by_play_{season}.csv.gz", usecols=PBP_COLUMNS)
        return df[df["season_type"] == "REG"].reset_index(drop=True)

    return _cached(f"pbp_{season}", loader)


def load_participation(season):
    def loader():
        df = download_csv("pbp_participation", f"pbp_participation_{season}.csv", usecols=PARTICIPATION_COLUMNS)
        return df.rename(columns={"nflverse_game_id": "game_id"})

    return _cached(f"participation_{season}", loader)


def load_ftn(season):
    def loader():
        df = download_csv("ftn_charting", f"ftn_charting_{season}.csv", usecols=FTN_COLUMNS)
        return df.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})

    return _cached(f"ftn_{season}", loader)


def load_player_stats(season):
    def loader():
        df = download_csv("stats_player", f"stats_player_week_{season}.csv.gz")
        df = df[df["season_type"] == "REG"]
        df = df[[c for c in STATS_COLUMNS if c in df.columns]].copy()
        for col in STATS_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        numeric = [c for c in STATS_COLUMNS if c not in ("player_id", "season_type", "position", "team", "opponent_team")]
        df[numeric] = df[numeric].fillna(0)
        # Same DK formula the live projection engine is trained on.
        df["dk_points"] = [_skill_player_fantasy_points(row) for row in df.itertuples()]
        return df.reset_index(drop=True)

    return _cached(f"stats_{season}", loader)


def load_players():
    return _cached("players", lambda: download_csv("players", "players.csv.gz", usecols=["gsis_id", "position", "display_name"]))


def load_schedules():
    return _cached("schedules", lambda: download_csv("schedules", "games.csv.gz"))


def load_pfr_def(season):
    def loader():
        # Only recent seasons are published gzipped; older ones are plain .csv.
        try:
            df = download_csv("pfr_advstats", f"advstats_week_def_{season}.csv.gz")
        except RuntimeError:
            df = download_csv("pfr_advstats", f"advstats_week_def_{season}.csv")
        return df[df["game_type"] == "REG"].reset_index(drop=True)

    return _cached(f"pfr_def_{season}", loader)


def warm_cache(seasons, include_ftn_from=2022):
    """Download everything the study needs for `seasons` once. Returns
    {source: [seasons that failed]} so a gap is reported, not silently
    skipped."""
    failures = {}
    for season in seasons:
        loaders = [("pbp", load_pbp), ("participation", load_participation), ("stats", load_player_stats), ("pfr_def", load_pfr_def)]
        if season >= include_ftn_from:
            loaders.append(("ftn", load_ftn))
        for name, fn in loaders:
            try:
                fn(season)
            except RuntimeError:
                failures.setdefault(name, []).append(season)
    load_players()
    load_schedules()
    return failures
