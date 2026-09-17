"""
fetch_data.py
-------------
Pulls recent player game logs from the NBA stats API (via nba_api) and
saves them locally so the projection model can use them.

Usage:
    python fetch_data.py --season 2025-26 --last-n-games 15

Output:
    data/player_game_logs.csv
    data/team_defense_ranks.csv
"""

import argparse
import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats
from nba_api.stats.static import players, teams


def fetch_player_game_logs(season: str) -> pd.DataFrame:
    """Pull every player's game log for the given season."""
    print(f"Fetching player game logs for {season}...")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
    )
    df = log.get_data_frames()[0]
    return df


def fetch_team_defense_ranks(season: str) -> pd.DataFrame:
    """
    Pull team defensive stats (points allowed, defensive rating) so we can
    adjust projections based on matchup difficulty.
    """
    print(f"Fetching team defensive stats for {season}...")
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Defense",
        per_mode_detailed="PerGame",
    )
    df = stats.get_data_frames()[0]
    return df


def compute_recent_form(game_logs: pd.DataFrame, last_n_games: int) -> pd.DataFrame:
    """
    For each player, compute average fantasy-relevant stats over their
    last N games vs. their full-season average. DraftKings NBA scoring:
        PTS*1 + REB*1.25 + AST*1.5 + STL*2 + BLK*2 - TOV*0.5
        + bonuses for double/triple-doubles (+1.5 / +3)
    """
    game_logs = game_logs.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=[True, False])

    def dk_points(row):
        pts = row.get("PTS", 0) or 0
        reb = row.get("REB", 0) or 0
        ast = row.get("AST", 0) or 0
        stl = row.get("STL", 0) or 0
        blk = row.get("BLK", 0) or 0
        tov = row.get("TOV", 0) or 0
        score = pts + reb * 1.25 + ast * 1.5 + stl * 2 + blk * 2 - tov * 0.5
        double_digit_cats = sum(x >= 10 for x in [pts, reb, ast, stl, blk])
        if double_digit_cats >= 3:
            score += 3.0  # triple-double bonus
        elif double_digit_cats == 2:
            score += 1.5  # double-double bonus
        return score

    game_logs["DK_POINTS"] = game_logs.apply(dk_points, axis=1)

    season_avg = game_logs.groupby("PLAYER_ID")["DK_POINTS"].mean().rename("SEASON_AVG_DK")
    recent_avg = (
        game_logs.groupby("PLAYER_ID")
        .head(last_n_games)
        .groupby("PLAYER_ID")["DK_POINTS"]
        .mean()
        .rename(f"LAST_{last_n_games}_AVG_DK")
    )
    games_played = game_logs.groupby("PLAYER_ID")["DK_POINTS"].count().rename("GAMES_PLAYED")

    names = game_logs.groupby("PLAYER_ID")["PLAYER_NAME"].first()
    teams_ = game_logs.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].first()

    summary = pd.concat([names, teams_, season_avg, recent_avg, games_played], axis=1).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2025-26", help="e.g. 2025-26")
    parser.add_argument("--last-n-games", type=int, default=15)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    game_logs = fetch_player_game_logs(args.season)
    game_logs.to_csv("data/player_game_logs_raw.csv", index=False)

    summary = compute_recent_form(game_logs, args.last_n_games)
    summary.to_csv("data/player_form_summary.csv", index=False)
    print(f"Saved {len(summary)} players to data/player_form_summary.csv")

    time.sleep(1)  # be polite to the API
    defense = fetch_team_defense_ranks(args.season)
    defense.to_csv("data/team_defense_ranks.csv", index=False)
    print(f"Saved {len(defense)} teams to data/team_defense_ranks.csv")


if __name__ == "__main__":
    main()
