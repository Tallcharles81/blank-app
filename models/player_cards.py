"""Player cards for a build report: what each rostered player has actually
done and who he plays, next to what we project.

Asked for by the user 2026-10-12 after the 10/11 Sunday: every report must
show average points (not only ceiling), usage (targets, carries, snaps),
the opponent and how that defense has played the position, and for each
DST its matchup. What the history says each column is worth (2023-2025
fit, tested on 2025-26; scratchpad matchup_skill.py / dst_predictors.py):
  - Season and recent averages and usage carry most of the signal for
    QB/RB/WR/TE; targets + carries help TEs and WRs a bit beyond average.
  - "Opponent allows X to the position" did NOT improve skill-player
    predictions (MAE unchanged at every position), so it is shown as
    context only and is not in the projection.
  - For a DST the opponent's implied total is the strongest signal we have
    (corr 0.31 vs 0.06 for the DST's own average), and it now drives the
    DST projection (models/dst_matchup.py).
"""
from collections import defaultdict

import pandas as pd
from sqlalchemy import text

from data.nflverse_fetch import fetch_team_implied_totals, to_nflverse_team
from data.player_availability import resolve_slate_season_week
from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine

RECENT_GAMES = 3


def _mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else None


def _fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _slate_rows(slate_id, player_ids, engine):
    with engine.connect() as conn:
        rows = conn.execute(text(
            """SELECT sp.player_id, sp.name, sp.position, sp.team, sp.opponent, sp.salary,
                      p.proj_median, p.proj_ceiling
               FROM slate_player_pool sp LEFT JOIN projections p USING (slate_id, player_id)
               WHERE sp.slate_id = :s AND sp.position <> 'CPT'"""), {"s": slate_id}).mappings().fetchall()
    rows = [dict(r) for r in rows]
    if player_ids is not None:
        wanted = set(player_ids)
        rows = [r for r in rows if r["player_id"] in wanted]
    return rows


def _season_games(gsis_ids, season, week, engine):
    if not gsis_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text(
            """SELECT player_id, position, week, fantasy_points_ppr::float AS pts, targets, carries, snap_pct::float AS snap
               FROM player_weekly_stats WHERE player_id = ANY(:ids) AND season = :s AND week < :w
               ORDER BY week"""), {"ids": list(gsis_ids), "s": season, "w": week}).fetchall()
    out = defaultdict(list)
    for r in rows:
        out[r.player_id].append(r)
    return out


def _defense_vs_position(season, week, engine):
    """{(defense_team, position): (DK points allowed per game, rank)} -
    rank 1 = allows the most (the easiest matchup)."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            """SELECT opponent AS defense, position, week, sum(fantasy_points_ppr)::float AS pts
               FROM player_weekly_stats
               WHERE season = :s AND week < :w AND position IN ('QB','RB','WR','TE','DST')
               GROUP BY 1, 2, 3"""), {"s": season, "w": week}).fetchall()
    per_game = defaultdict(list)
    for r in rows:
        per_game[(r.defense, r.position)].append(r.pts)
    avg = {k: sum(v) / len(v) for k, v in per_game.items()}
    out = {}
    for pos in {k[1] for k in avg}:
        ranked = sorted((k for k in avg if k[1] == pos), key=lambda k: -avg[k])
        for i, k in enumerate(ranked, start=1):
            out[k] = (round(avg[k], 1), i)
    return out


def player_cards(slate_id, player_ids=None, engine=None):
    """One row per player (CPT rows folded into FLEX): salary, projected
    median and ceiling, season and last-3 DK averages, per-game usage, the
    opponent, the opponent defense's points allowed to the position (rank 1
    = most allowed) and both teams' implied totals."""
    engine = engine or get_engine()
    season_week = resolve_slate_season_week(slate_id, engine)
    if season_week is None:
        raise ValueError(f"Can't resolve season/week for slate {slate_id}")
    season, week = season_week
    rows = _slate_rows(slate_id, player_ids, engine)
    gsis, _, _ = resolve_dk_players_to_gsis(rows, engine)
    games = _season_games({g for g in gsis.values() if g}, season, week, engine)
    dvp = _defense_vs_position(season, week, engine)
    gives = _offense_gives_dst(engine, season, week)
    try:
        implied = fetch_team_implied_totals(season, week)
    except Exception:  # lines are context here; a failed fetch shouldn't stop the report
        implied = {}
    cards = []
    for r in rows:
        gs = games.get(gsis.get(r["player_id"]), [])
        pos = r["position"] if r["position"] != "FLEX" else (gs[-1].position if gs else "FLEX")
        recent = gs[-RECENT_GAMES:]
        team, opp = to_nflverse_team(r["team"]), to_nflverse_team(r["opponent"])
        card = {
            "player": r["name"], "pos": pos, "team": r["team"], "opp": r["opponent"], "salary": r["salary"],
            "proj_median": float(r["proj_median"]) if r["proj_median"] is not None else None,
            "proj_ceiling": float(r["proj_ceiling"]) if r["proj_ceiling"] is not None else None,
            "games": len(gs),
            "season_avg": _mean([g.pts for g in gs]),
            "last3_avg": _mean([g.pts for g in recent]),
            "tgt_pg": _mean([g.targets or 0 for g in gs]) if pos in ("RB", "WR", "TE") else None,
            "car_pg": _mean([g.carries or 0 for g in gs]) if pos in ("QB", "RB") else None,
            "snap_last3": _mean([round(100 * g.snap, 0) for g in recent if g.snap is not None]) if pos != "DST" else None,
            "team_implied": implied.get(team),
            "opp_implied": implied.get(opp),
        }
        if pos == "DST":
            # A DST's matchup is the offense it faces: DK points that offense
            # has GIVEN UP to opposing defenses (rank 1 = most given up).
            card["opp_allows_to_pos"], card["opp_rank_vs_pos"] = gives.get(opp, (None, None))
        else:
            card["opp_allows_to_pos"], card["opp_rank_vs_pos"] = dvp.get((opp, pos), (None, None))
        cards.append(card)
    return pd.DataFrame(cards)


def _offense_gives_dst(engine, season, week):
    """{offense_team: (DST points scored against it per game, rank)} - rank 1
    = gives up the most to defenses."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            """SELECT opponent AS offense, avg(fantasy_points_ppr)::float AS pts FROM player_weekly_stats
               WHERE position = 'DST' AND season = :s AND week < :w GROUP BY 1"""), {"s": season, "w": week}).fetchall()
    ranked = sorted(rows, key=lambda r: -r.pts)
    return {r.offense: (round(r.pts, 1), i) for i, r in enumerate(ranked, start=1)}


def format_cards(cards):
    """Markdown table, grouped by position, for a build report."""
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 4}
    df = cards.assign(_o=cards.pos.map(order).fillna(9)).sort_values(["_o", "proj_median"], ascending=[True, False])
    header = ("| Player | Pos | Team | Opp | Salary | Proj median | Proj ceiling | Season avg (G) | Last 3 | "
              "Tgt/g | Car/g | Snap% L3 | Opp allows to pos (rank) | Team / Opp implied |")
    lines = [header, "|" + "---|" * 14]
    for _, c in df.iterrows():
        lines.append(
            f"| {c.player} | {c.pos} | {c.team} | {c.opp} | ${int(c.salary):,} | {_fmt(c.proj_median)} | {_fmt(c.proj_ceiling)} | "
            f"{_fmt(c.season_avg)} ({int(c.games)}) | {_fmt(c.last3_avg)} | {_fmt(c.tgt_pg)} | {_fmt(c.car_pg)} | {_fmt(c.snap_last3)} | "
            f"{_fmt(c.opp_allows_to_pos)} (#{_fmt(c.opp_rank_vs_pos)}) | {_fmt(c.team_implied)} / {_fmt(c.opp_implied)} |")
    return "\n".join(lines)
