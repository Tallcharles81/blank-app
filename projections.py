"""
projections.py
---------------
Builds a blended fantasy point projection for each player using:
    - Recent form (last N games) - weighted heavier, captures hot/cold streaks
    - Season-long average - stabilizes against small-sample noise
    - Opponent matchup adjustment - based on how many DK points that
      position/team allows relative to league average

Usage:
    python projections.py --slate data/dk_salaries.csv --recent-weight 0.65

Output:
    data/projections.csv  (ready to feed into optimizer.py)
"""

import argparse
import pandas as pd


def load_form_summary(path="data/player_form_summary.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_defense_ranks(path="data/team_defense_ranks.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_slate(path: str) -> pd.DataFrame:
    """
    Expects a DraftKings salary export CSV with at minimum:
    Name, Salary, Position, TeamAbbrev, Opponent (or OppTeam)
    Column names vary by DK export version, so adjust the rename map below
    if your CSV headers differ.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_matchup_multiplier(defense_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns team defensive rating into a simple multiplier:
    league-average defense = 1.0x, worse defense (allows more) = >1.0x,
    elite defense = <1.0x. This is intentionally simple to start —
    swap in DVP-by-position data later for real precision.
    """
    if "DEF_RATING" not in defense_df.columns:
        raise ValueError("Expected DEF_RATING column from leaguedashteamstats output")

    league_avg_def = defense_df["DEF_RATING"].mean()
    defense_df = defense_df.copy()
    # Higher DEF_RATING = worse defense = should boost opposing player projections
    defense_df["MATCHUP_MULTIPLIER"] = defense_df["DEF_RATING"] / league_avg_def
    return defense_df[["TEAM_ABBREVIATION", "MATCHUP_MULTIPLIER"]]


def compute_projections(slate: pd.DataFrame, form: pd.DataFrame,
                         matchup: pd.DataFrame, recent_weight: float) -> pd.DataFrame:
    recent_col = [c for c in form.columns if c.startswith("LAST_")][0]

    merged = slate.merge(
        form, left_on="Name", right_on="PLAYER_NAME", how="left"
    )

    # Fill missing (rookies, low-sample players) with a conservative default
    merged[recent_col] = merged[recent_col].fillna(merged["SEASON_AVG_DK"])
    merged["SEASON_AVG_DK"] = merged["SEASON_AVG_DK"].fillna(0)
    merged[recent_col] = merged[recent_col].fillna(0)

    merged["BASE_PROJECTION"] = (
        merged[recent_col] * recent_weight
        + merged["SEASON_AVG_DK"] * (1 - recent_weight)
    )

    # Join opponent matchup multiplier if an Opponent column exists
    opp_col = None
    for candidate in ["Opponent", "OppTeam", "Opp"]:
        if candidate in merged.columns:
            opp_col = candidate
            break

    if opp_col:
        merged = merged.merge(
            matchup, left_on=opp_col, right_on="TEAM_ABBREVIATION", how="left"
        )
        merged["MATCHUP_MULTIPLIER"] = merged["MATCHUP_MULTIPLIER"].fillna(1.0)
    else:
        merged["MATCHUP_MULTIPLIER"] = 1.0

    merged["PROJECTION"] = merged["BASE_PROJECTION"] * merged["MATCHUP_MULTIPLIER"]

    # Value metric: points per $1000 salary, useful for spotting salary-efficient plays
    merged["VALUE"] = merged["PROJECTION"] / (merged["Salary"] / 1000)

    return merged.sort_values("PROJECTION", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slate", required=True, help="Path to DK salary CSV export")
    parser.add_argument("--recent-weight", type=float, default=0.65)
    parser.add_argument("--output", default="data/projections.csv")
    args = parser.parse_args()

    slate = load_slate(args.slate)
    form = load_form_summary()
    defense = load_defense_ranks()
    matchup = build_matchup_multiplier(defense)

    projections = compute_projections(slate, form, matchup, args.recent_weight)
    projections.to_csv(args.output, index=False)
    print(f"Wrote {len(projections)} projections to {args.output}")
    print(projections[["Name", "Salary", "PROJECTION", "VALUE"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
