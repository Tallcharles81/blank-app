import pandas as pd

from data.nflverse_fetch import download_csv

# Real, official-source depth chart data (nflverse's real "depth_charts"
# release, sourced from ESPN) - genuinely new to this codebase, never
# fetched before models/playing_time_engine.py. Confirmed for real: only
# available for the CURRENT season (2026) - a real request for 2023/2024/
# 2025 404s, nflverse does not publish historical depth-chart snapshots for
# past seasons the way it does for box scores. This means depth-chart-based
# role classification is real and usable for a LIVE, upcoming slate, but
# cannot be backtested against past seasons the way every other threshold
# in this codebase has been - that limitation is disclosed everywhere this
# module's data is used, not glossed over.
#
# The feed refreshes multiple times per day (real distinct `dt` timestamps),
# so "the current depth chart" means the most recent row per (team, player)
# as of whenever this is called - there is no lookahead risk for a LIVE
# call (there's nothing to look ahead of), but see fetch_depth_chart's own
# note on the one real case where this matters: a snapshot fetched well
# before kickoff can still be stale by kickoff if a real in-week move
# happens after the fetch.


def fetch_depth_chart(season):
    """Every real depth-chart row nflverse has for `season`, one row per
    real (team, player, snapshot time) - NOT deduplicated to the latest
    snapshot per player here, since a caller backtesting/inspecting history
    within the season may want every snapshot, not just the newest. See
    latest_depth_chart_by_player for the "current role" view most callers
    actually want.
    """
    return download_csv("depth_charts", f"depth_charts_{season}.csv.gz")


# Real offensive skill positions this engine cares about - a real player
# (e.g. a real RB who also returns kicks/punts) can have MULTIPLE real
# depth-chart rows at the exact same latest timestamp, one per unit he's
# on (offense, special teams, occasionally defense for a two-way player) -
# confirmed for real on the actual 2026 BUF depth chart (Ray Davis: RB in
# "3WR 1TE" AND KR/PR in "Special Teams", same real dt). Filtering to these
# real offensive pos_abb values BEFORE deduping to "latest row per player"
# is what keeps his OFFENSIVE role, not whichever row happened to sort last.
_OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}


def latest_depth_chart_by_player(season):
    """{gsis_id: {"team", "position": pos_abb, "pos_grp", "pos_rank",
    "pos_slot", "as_of"}} - the single most recent real OFFENSIVE depth-
    chart row per real player for `season`, which is what "his current
    role" means for every caller in models/playing_time_engine.py.
    `pos_rank` is nflverse's own real depth ranking within that position
    group (1 = the real starter/top of the group) - not this codebase's
    invention. A real special-teams-only player (a pure kicker/punter/long
    snapper) has no offensive row at all and is simply absent here.
    """
    df = fetch_depth_chart(season)
    df = df[df["pos_abb"].isin(_OFFENSIVE_POSITIONS)]
    df = df.sort_values("dt")
    latest = df.drop_duplicates(subset=["gsis_id"], keep="last")
    return {
        row.gsis_id: {
            "team": row.team,
            "position": row.pos_abb,
            "pos_grp": row.pos_grp,
            "pos_rank": int(row.pos_rank),
            "pos_slot": int(row.pos_slot),
            "as_of": row.dt,
        }
        for row in latest.itertuples()
        if pd.notna(row.gsis_id)  # real practice-squad/futures rows sometimes carry no gsis_id at all
    }
