import pandas as pd

from models.game_environment import rank_games, team_environment


def _plays(team, opp, n_pass, n_run, pass_yards, epa):
    rows = []
    for i in range(n_pass):
        rows.append({"game_id": f"{team}{opp}", "week": 1, "posteam": team, "defteam": opp, "play_type": "pass",
                     "yards_gained": pass_yards, "epa": epa, "wp": 0.5, "down": 1})
    for i in range(n_run):
        rows.append({"game_id": f"{team}{opp}", "week": 1, "posteam": team, "defteam": opp, "play_type": "run",
                     "yards_gained": 4, "epa": epa, "wp": 0.5, "down": 1})
    return rows


def test_the_fast_explosive_high_total_game_ranks_first():
    pbp = pd.DataFrame(
        _plays("AAA", "BBB", 45, 25, 22, 0.3) + _plays("BBB", "AAA", 45, 25, 22, 0.3)  # fast, pass-heavy, explosive
        + _plays("CCC", "DDD", 20, 40, 5, -0.2) + _plays("DDD", "CCC", 20, 40, 5, -0.2)  # slow, run-heavy
    )
    env = team_environment(pbp)
    assert env.loc["AAA", "neutral_pass_rate"] > env.loc["CCC", "neutral_pass_rate"]
    assert env.loc["AAA", "explosive_rate"] == 45 / 70
    sched = pd.DataFrame([{"home_team": "AAA", "away_team": "BBB", "spread_line": 3.0, "total_line": 51.5},
                          {"home_team": "CCC", "away_team": "DDD", "spread_line": -2.5, "total_line": 38.5}])
    ranked = rank_games(sched, env)
    assert list(ranked.game) == ["BBB@AAA", "DDD@CCC"]
    assert ranked.loc[0, "implied_home"] == 27.2 and ranked.loc[0, "implied_away"] == 24.2
