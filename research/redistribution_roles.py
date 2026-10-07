"""Time-aware offensive role classification and the per-(game, offense,
role) production table the redistribution study runs on.

Role rules (all ranking inputs come from games strictly BEFORE the game
being classified - nothing from the game itself except who was active):
  WR1/WR2/WR3+  active WRs ranked by trailing target share
  TE1/TE2+      active TEs ranked by trailing offensive snap share
  RB1/RB2/RB3+  active RBs/FBs ranked by trailing share of team
                carries+targets
  QB            the active QB with the most trailing dropbacks
  OTHER         everyone else (e.g. a QB target, an OL trick play)
"Active" = on the field for at least one offensive play (participation
data) or credited with a target/carry. That's the one piece of same-day
information used, and it's knowable at lock (inactives are announced ~90
minutes before kickoff); a player who was active but never took a snap is
missed, which only affects the pooled WR3+/TE2+/RB3+ buckets.

Stratifier flags (also trailing-only): TE1 receiving vs blocking-heavy,
RB committee, receiving-RB identity, rushing QB.
"""
import numpy as np
import pandas as pd

from research.redistribution_data import (
    load_participation,
    load_pbp,
    load_player_stats,
    load_players,
)

TRAILING_GAMES = 8
ROLES = ["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2", "RB3+", "QB", "OTHER"]
ROLE_GROUP = {"WR1": "WR", "WR2": "WR", "WR3+": "WR", "TE1": "TE", "TE2+": "TE", "RB1": "RB", "RB2": "RB",
              "RB3+": "RB", "QB": "QB", "OTHER": "OTHER"}
EXPLOSIVE_RECEPTION_YARDS = 20
EXPLOSIVE_RUN_YARDS = 10
DEEP_AIR_YARDS = 20
# Pseudo-count (team targets) pulling a thin trailing target share toward
# the position's typical share for a player with little history.
SHARE_PRIOR_TEAM_TARGETS = 30
POSITION_PRIOR_TARGET_SHARE = {"WR": 0.08, "TE": 0.06, "RB": 0.05, "QB": 0.0, "OTHER": 0.0}
POSITION_PRIOR_TOUCH_SHARE = {"RB": 0.08}


def _position_map():
    players = load_players()
    pos = dict(zip(players["gsis_id"], players["position"]))
    return {k: ("RB" if v == "FB" else v) for k, v in pos.items() if isinstance(k, str)}


def _clean_plays(pbp):
    plays = pbp[(pbp["two_point_attempt"] != 1) & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)].copy()
    plays = plays[plays["play_type"].isin(["pass", "run"]) | (plays["sack"] == 1) | (plays["qb_scramble"] == 1)]
    return plays


def player_game_usage(season):
    """One row per (game, team, player) with raw usage/production from pbp
    plus participation snap counts. Includes a `game_order` key
    (season*100+week) for trailing windows."""
    pbp = load_pbp(season)
    plays = _clean_plays(pbp)
    targets = plays[(plays["pass_attempt"] == 1) & plays["receiver_player_id"].notna() & (plays["sack"] != 1)]

    rec = targets.assign(
        player_id=targets["receiver_player_id"],
        tgt=1,
        rec=targets["complete_pass"].fillna(0),
        rec_yds=targets["receiving_yards"].fillna(0),
        air=targets["air_yards"].fillna(0),
        rec_td=targets["pass_touchdown"].fillna(0),
        rz_tgt=(targets["yardline_100"] <= 20).astype(int),
        ez_tgt=(targets["air_yards"] >= targets["yardline_100"]).astype(int),
        deep_tgt=(targets["air_yards"] >= DEEP_AIR_YARDS).astype(int),
        mid_tgt=(targets["pass_location"] == "middle").astype(int),
        expl_rec=((targets["complete_pass"] == 1) & (targets["receiving_yards"] >= EXPLOSIVE_RECEPTION_YARDS)).astype(int),
    ).groupby(["game_id", "posteam", "player_id"])[["tgt", "rec", "rec_yds", "air", "rec_td", "rz_tgt", "ez_tgt", "deep_tgt", "mid_tgt", "expl_rec"]].sum()

    runs = plays[(plays["rush"] == 1) & plays["rusher_player_id"].notna()]
    rush = runs.assign(
        player_id=runs["rusher_player_id"],
        car=1,
        designed=(runs["qb_scramble"] != 1).astype(int),
        rush_yds=runs["rushing_yards"].fillna(0),
        rush_td=runs["rush_touchdown"].fillna(0),
        expl_run=(runs["rushing_yards"] >= EXPLOSIVE_RUN_YARDS).astype(int),
        i5_car=(runs["yardline_100"] <= 5).astype(int),
        rz_car=(runs["yardline_100"] <= 20).astype(int),
    ).groupby(["game_id", "posteam", "player_id"])[["car", "designed", "rush_yds", "rush_td", "expl_run", "i5_car", "rz_car"]].sum()

    dropbacks = plays[plays["qb_dropback"] == 1]
    qb = dropbacks.assign(
        player_id=dropbacks["passer_player_id"].fillna(dropbacks["rusher_player_id"]),
        db=1,
        att=dropbacks["pass_attempt"].fillna(0) * (dropbacks["sack"] != 1),
        cmp=dropbacks["complete_pass"].fillna(0),
        pass_yds=dropbacks["passing_yards"].fillna(0),
        pass_td=dropbacks["pass_touchdown"].fillna(0),
        ints=dropbacks["interception"].fillna(0),
        sacks=dropbacks["sack"].fillna(0),
        qb_hits=dropbacks["qb_hit"].fillna(0),
        scrambles=dropbacks["qb_scramble"].fillna(0),
    ).dropna(subset=["player_id"]).groupby(["game_id", "posteam", "player_id"])[
        ["db", "att", "cmp", "pass_yds", "pass_td", "ints", "sacks", "qb_hits", "scrambles"]
    ].sum()

    usage = rec.join(rush, how="outer").join(qb, how="outer").fillna(0)

    part = load_participation(season)
    part = part.merge(plays[["game_id", "play_id", "posteam", "qb_dropback"]], on=["game_id", "play_id"], how="inner")
    part = part[part["offense_players"].notna() & (part["offense_players"] != "")]
    exploded = part.assign(player_id=part["offense_players"].str.split(";")).explode("player_id")
    snaps = exploded.groupby(["game_id", "posteam", "player_id"]).agg(snaps=("play_id", "size"), db_snaps=("qb_dropback", "sum"))
    team_snaps = part.groupby(["game_id", "posteam"]).agg(team_snaps=("play_id", "size"), team_db=("qb_dropback", "sum"))

    usage = usage.join(snaps, how="outer").fillna(0).reset_index()
    usage = usage.merge(team_snaps.reset_index(), on=["game_id", "posteam"], how="left")
    team = plays.groupby(["game_id", "posteam"]).agg(
        team_tgt=("receiver_player_id", lambda s: s.notna().sum()),
    ).reset_index()
    team_car = usage.groupby(["game_id", "posteam"])["car"].sum().rename("team_car").reset_index()
    team_tgt = usage.groupby(["game_id", "posteam"])["tgt"].sum().rename("team_tgt").reset_index()
    usage = usage.merge(team_tgt, on=["game_id", "posteam"]).merge(team_car, on=["game_id", "posteam"])
    del team

    meta = pbp.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                      home_team=("home_team", "first"), away_team=("away_team", "first"))
    usage = usage.merge(meta.reset_index(), on="game_id", how="left")
    usage["defteam"] = np.where(usage["posteam"] == usage["home_team"], usage["away_team"], usage["home_team"])
    usage["game_order"] = usage["season"] * 100 + usage["week"]

    positions = _position_map()
    usage["position"] = usage["player_id"].map(positions).fillna("OTHER")
    usage.loc[~usage["position"].isin(["WR", "TE", "RB", "QB"]), "position"] = "OTHER"

    stats = load_player_stats(season)[["player_id", "week", "dk_points"]]
    usage = usage.merge(stats, on=["player_id", "week"], how="left")
    usage["dk_points"] = usage["dk_points"].fillna(0.0)
    return usage


def add_trailing_usage(usage):
    """Trailing (strictly prior-game) usage per player over his last
    TRAILING_GAMES active games, across season boundaries. Ratio-of-sums,
    not mean-of-ratios, so a 2-target game doesn't count like a 12-target
    one."""
    usage = usage.sort_values(["player_id", "game_order"]).copy()
    cols = ["tgt", "team_tgt", "car", "team_car", "snaps", "team_snaps", "db_snaps", "team_db", "db", "designed", "i5_car",
            "rec_yds", "dk_points"]
    grouped = usage.groupby("player_id")
    for col in cols:
        usage[f"prior_{col}"] = grouped[col].transform(lambda s: s.shift(1).rolling(TRAILING_GAMES, min_periods=1).sum()).fillna(0.0)
    usage["prior_games"] = grouped.cumcount().clip(upper=TRAILING_GAMES)

    prior_share = usage["position"].map(POSITION_PRIOR_TARGET_SHARE).fillna(0.0)
    usage["tr_tgt_share"] = (usage["prior_tgt"] + SHARE_PRIOR_TEAM_TARGETS * prior_share) / (usage["prior_team_tgt"] + SHARE_PRIOR_TEAM_TARGETS)
    touch_prior = usage["position"].map(POSITION_PRIOR_TOUCH_SHARE).fillna(0.0)
    usage["tr_touch_share"] = (usage["prior_tgt"] + usage["prior_car"] + SHARE_PRIOR_TEAM_TARGETS * touch_prior) / (
        usage["prior_team_tgt"] + usage["prior_team_car"] + SHARE_PRIOR_TEAM_TARGETS
    )
    usage["tr_car_share"] = usage["prior_car"] / usage["prior_team_car"].replace(0, np.nan)
    usage["tr_snap_share"] = usage["prior_snaps"] / usage["prior_team_snaps"].replace(0, np.nan)
    usage["tr_db_share"] = usage["prior_db_snaps"] / usage["prior_team_db"].replace(0, np.nan)
    usage["tr_tgt_per_db_snap"] = usage["prior_tgt"] / usage["prior_db_snaps"].replace(0, np.nan)
    usage["tr_dropbacks_pg"] = usage["prior_db"] / usage["prior_games"].replace(0, np.nan)
    usage["tr_designed_runs_pg"] = usage["prior_designed"] / usage["prior_games"].replace(0, np.nan)
    usage["tr_i5_share"] = usage["prior_i5_car"] / usage["prior_team_car"].replace(0, np.nan)
    usage["tr_dk_pg"] = usage["prior_dk_points"] / usage["prior_games"].replace(0, np.nan)
    return usage


def assign_roles(usage):
    """Adds a `role` column. Rank ties and players with no history fall to
    the bottom of their position (sorted by trailing share, NaN last)."""
    usage = usage.copy()
    usage["role"] = "OTHER"
    keys = ["game_id", "posteam"]

    def rank_within(pos, score_col):
        sub = usage[usage["position"] == pos]
        order = sub.sort_values(score_col, ascending=False, na_position="last").groupby(keys).cumcount()
        return order.reindex(sub.index)

    wr_rank = rank_within("WR", "tr_tgt_share")
    usage.loc[wr_rank.index, "role"] = np.select([wr_rank == 0, wr_rank == 1], ["WR1", "WR2"], "WR3+")
    te_rank = rank_within("TE", "tr_snap_share")
    usage.loc[te_rank.index, "role"] = np.where(te_rank == 0, "TE1", "TE2+")
    rb_rank = rank_within("RB", "tr_touch_share")
    usage.loc[rb_rank.index, "role"] = np.select([rb_rank == 0, rb_rank == 1], ["RB1", "RB2"], "RB3+")
    qb_rank = rank_within("QB", "tr_dropbacks_pg")
    usage.loc[qb_rank.index, "role"] = np.where(qb_rank == 0, "QB", "OTHER")
    return usage


def build_player_games(seasons):
    """Player-game rows with trailing usage and roles for `seasons`
    (trailing windows use every season passed, so pass one season earlier
    than the first one you analyze)."""
    frames = [player_game_usage(s) for s in seasons]
    usage = pd.concat(frames, ignore_index=True)
    usage = usage[usage["player_id"].notna() & (usage["player_id"] != "")]
    usage = add_trailing_usage(usage)
    return assign_roles(usage)


ROLE_METRICS = ["tgt", "rec", "rec_yds", "air", "rec_td", "rz_tgt", "ez_tgt", "deep_tgt", "mid_tgt", "expl_rec",
                "car", "rush_yds", "rush_td", "expl_run", "i5_car", "dk_points"]


def team_game_roles(player_games):
    """One row per (game, offense, role): summed production plus the
    EXPECTED target and carry share of that role from its players'
    trailing shares, normalized over every active player so each team-
    game's expected shares sum to 1 (the redistribution-must-add-up
    constraint)."""
    pg = player_games.copy()
    keys = ["game_id", "posteam"]
    pg["w_tgt"] = pg["tr_tgt_share"].fillna(0.0).clip(lower=0)
    pg["w_car"] = pg["tr_car_share"].fillna(0.0).clip(lower=0)
    pg["exp_tgt_share"] = pg["w_tgt"] / pg.groupby(keys)["w_tgt"].transform("sum").replace(0, np.nan)
    pg["exp_car_share"] = pg["w_car"] / pg.groupby(keys)["w_car"].transform("sum").replace(0, np.nan)

    agg = pg.groupby([*keys, "role"]).agg(
        **{m: (m, "sum") for m in ROLE_METRICS},
        exp_tgt_share=("exp_tgt_share", "sum"),
        exp_car_share=("exp_car_share", "sum"),
        n_players=("player_id", "size"),
        tr_dk_pg=("tr_dk_pg", "sum"),
    ).reset_index()

    # Fill missing roles (e.g. a team with no active TE2) with zero rows so
    # every team-game has all ROLES and shares sum correctly.
    games = pg[[*keys, "season", "week", "game_order", "defteam", "home_team", "away_team", "team_tgt", "team_car"]].drop_duplicates(keys)
    full = games.assign(_k=1).merge(pd.DataFrame({"role": ROLES, "_k": 1}), on="_k").drop(columns="_k")
    out = full.merge(agg, on=[*keys, "role"], how="left")
    fill = [*ROLE_METRICS, "exp_tgt_share", "exp_car_share", "n_players", "tr_dk_pg"]
    out[fill] = out[fill].fillna(0.0)
    out["tgt_share"] = out["tgt"] / out["team_tgt"].replace(0, np.nan)
    out["car_share"] = out["car"] / out["team_car"].replace(0, np.nan)
    out["group"] = out["role"].map(ROLE_GROUP)
    return out


def team_game_flags(player_games):
    """Trailing-only stratifier flags per (game, offense)."""
    pg = player_games
    keys = ["game_id", "posteam"]
    te1 = pg[pg["role"] == "TE1"].set_index(keys)["tr_tgt_per_db_snap"].rename("te1_tgt_per_db_snap")
    rb = pg[pg["position"] == "RB"].copy()
    rb_car = rb.groupby(keys)["tr_car_share"].agg(["max", "sum"])
    committee = (rb_car["max"] / rb_car["sum"].replace(0, np.nan) < 0.6).rename("rb_committee")
    rb_recv = rb.sort_values("tr_tgt_share", ascending=False).groupby(keys).head(1).set_index(keys)["role"].rename("receiving_rb_role")
    qb = pg[pg["role"] == "QB"].set_index(keys)["tr_designed_runs_pg"].rename("qb_designed_runs_pg")
    return pd.concat([te1, committee, rb_recv, qb], axis=1).reset_index()
