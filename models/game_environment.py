"""Rank a slate's games by scoring environment, so multi-lineup builds can
spread QB stacks across the best games instead of piling into one or two.

Composite of the Vegas total (double weight), plays per game (pace),
neutral-situation pass rate, explosive-play rate for and allowed, offensive
EPA/play and EPA/play allowed - each a z-score across the slate's games,
from this season's nflverse play-by-play before the slate's week. Added
after 2026-10-04, when the #2-ranked game (DAL@HOU, 34-30) got 4 of 180
player slots across 20 lineups and the contest leader stacked it.
"""
import pandas as pd

from data.nflverse_fetch import download_csv, fetch_schedules

PBP_COLUMNS = ["game_id", "week", "posteam", "defteam", "play_type", "yards_gained", "epa", "wp", "down"]
# A run of 12+ or a pass of 20+ yards - the usual explosive-play cutoffs.
EXPLOSIVE_RUN_YARDS = 12
EXPLOSIVE_PASS_YARDS = 20


def team_environment(pbp):
    """Per-team pace, neutral pass rate, explosive rates and EPA from a pbp frame."""
    p = pbp[pbp.play_type.isin(["pass", "run"])].copy()
    p["explosive"] = ((p.play_type == "pass") & (p.yards_gained >= EXPLOSIVE_PASS_YARDS)) | (
        (p.play_type == "run") & (p.yards_gained >= EXPLOSIVE_RUN_YARDS)
    )
    games = p.groupby("posteam").game_id.nunique()
    # Neutral = early downs with the game still in doubt, so garbage-time passing doesn't count.
    neutral = p[p.wp.between(0.2, 0.8) & p.down.isin([1, 2])]
    return pd.DataFrame({
        "plays_pg": p.groupby("posteam").size() / games,
        "neutral_pass_rate": neutral.groupby("posteam").play_type.apply(lambda s: (s == "pass").mean()),
        "explosive_rate": p.groupby("posteam").explosive.mean(),
        "epa_play": p.groupby("posteam").epa.mean(),
        "def_explosive_rate": p.groupby("defteam").explosive.mean(),
        "def_epa_play": p.groupby("defteam").epa.mean(),
    })


def rank_games(schedule_week, team_env):
    """schedule_week: nflverse schedule rows for the slate's games (home_team,
    away_team, spread_line, total_line). Returns a DataFrame sorted best-first."""
    rows = []
    for _, g in schedule_week.iterrows():
        h, a = g.home_team, g.away_team
        if h not in team_env.index or a not in team_env.index:
            continue
        th, ta = team_env.loc[h], team_env.loc[a]
        imp_h = (g.total_line + g.spread_line) / 2  # nflverse spread_line is positive when the home team is favored
        rows.append({
            "game": f"{a}@{h}", "away": a, "home": h, "total": g.total_line,
            "implied_away": round(g.total_line - imp_h, 1), "implied_home": round(imp_h, 1),
            "plays_pg": th.plays_pg + ta.plays_pg,
            "neutral_pass": (th.neutral_pass_rate + ta.neutral_pass_rate) / 2,
            "expl_off": (th.explosive_rate + ta.explosive_rate) / 2,
            "expl_allowed": (th.def_explosive_rate + ta.def_explosive_rate) / 2,
            "off_epa": (th.epa_play + ta.epa_play) / 2,
            "def_epa_allowed": (th.def_epa_play + ta.def_epa_play) / 2,
        })
    g = pd.DataFrame(rows)
    if g.empty:
        return g
    z = lambda c: (g[c] - g[c].mean()) / (g[c].std() or 1.0)
    g["env_score"] = (2 * z("total") + z("plays_pg") + z("neutral_pass") + z("expl_off") + z("expl_allowed")
                      + z("off_epa") + z("def_epa_allowed")) / 8
    return g.sort_values("env_score", ascending=False).reset_index(drop=True)


def rank_slate_games(season, week, teams):
    """Rank the games on a slate (`teams`: nflverse codes of every team on it)
    for `season`/`week`, from play-by-play strictly before that week. Returns
    an empty DataFrame if either feed can't be fetched - callers fall back to
    the plain build rather than failing."""
    try:
        pbp = download_csv("pbp", f"play_by_play_{season}.csv.gz", usecols=PBP_COLUMNS)
        schedules = fetch_schedules()
    except Exception:
        return pd.DataFrame()
    pbp = pbp[pbp.week < week]
    if pbp.empty:
        return pd.DataFrame()
    sched = schedules[(schedules.season == season) & (schedules.week == week)
                      & schedules.home_team.isin(teams) & schedules.away_team.isin(teams)]
    return rank_games(sched, team_environment(pbp))
