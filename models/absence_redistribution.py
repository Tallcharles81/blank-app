"""Move a ruled-out lead running back's production to his backups, using
measured shares instead of a judgment call.

Measured on 2023-2025 (scratchpad next_man_up.py): in the 32 team-games
where a team's lead back (most carries + targets over the prior 4 games,
8+ DK points a game) sat, the next back in line gained a median 34% of the
lead's average points (mean 53%) and the third back 15% (mean 20%). In the
66 games a lead WR sat, the other WRs/TEs gained a median of about nothing
(WR2 -3%, WR3 +1%), so a pass catcher's absence moves nothing here - his
targets mostly go to worse outcomes, not to a teammate's ceiling.

Why this exists: two hand-made splits missed in opposite directions. DET@CAR
2026-10-04 gave 60% of Jalen Coker's median to the other pass catchers (the
data says ~0). ATL@NO 2026-10-05 gave Kamara 33% of Travis Etienne's median
(the data says 34-53%); Kamara scored 22.8 against our 11.9.
"""
import json
from collections import defaultdict

from sqlalchemy import text

from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine

# Median share of the absent lead back's median projection, by depth (2 = next man up).
RB_SHARES = {2: 0.34, 3: 0.15}
RECENT_WEEKS = 4


def _slate_rows(slate_id, engine):
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(
            """SELECT sp.player_id, sp.name, sp.position, sp.team, p.proj_median, p.proj_percentiles, p.locked_at
               FROM slate_player_pool sp JOIN projections p USING (slate_id, player_id) WHERE sp.slate_id = :s"""),
            {"s": slate_id}).mappings()]


def _recent_backfield(gsis_ids, engine):
    """{gsis_id: (position, avg carries+targets over his last RECENT_WEEKS games)}."""
    if not gsis_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text(
            """SELECT player_id, position, coalesce(carries,0) + coalesce(targets,0) AS opp,
                      rank() OVER (PARTITION BY player_id ORDER BY season DESC, week DESC) AS rk
               FROM player_weekly_stats WHERE player_id = ANY(:ids)"""), {"ids": list(gsis_ids)}).fetchall()
    games = defaultdict(list)
    pos = {}
    for r in rows:
        if r.rk <= RECENT_WEEKS:
            games[r.player_id].append(float(r.opp))
            pos[r.player_id] = r.position
    return {g: (pos[g], sum(v) / len(v)) for g, v in games.items()}


def redistribute_absence(slate_id, out_name, excluded_names=(), engine=None):
    """Raise the projections of `out_name`'s backup RBs by RB_SHARES of his
    median (whole ladder scaled, CPT and FLEX rows alike; locked rows are
    left alone). A non-RB absence changes nothing - see module docstring.
    `excluded_names` are other ruled-out players who can't absorb work.
    Returns [(name, old_median, new_median)] for the FLEX/classic row."""
    engine = engine or get_engine()
    rows = _slate_rows(slate_id, engine)
    base_rows = [r for r in rows if r["position"] != "CPT"]
    out_rows = [r for r in base_rows if r["name"] == out_name]
    if not out_rows:
        raise ValueError(f"{out_name} is not on slate {slate_id}")
    out_row = out_rows[0]
    team = out_row["team"]
    teammates = [r for r in base_rows if r["team"] == team and r["position"] != "DST"]
    gsis, _, _ = resolve_dk_players_to_gsis(
        [{"player_id": r["player_id"], "name": r["name"], "position": r["position"], "team": r["team"]} for r in teammates], engine
    )
    usage = _recent_backfield({g for g in gsis.values() if g}, engine)
    out_gsis = gsis.get(out_row["player_id"])
    if not out_gsis or usage.get(out_gsis, ("",))[0] != "RB":
        return []
    skip = {out_name, *excluded_names}
    backs = sorted(
        (r for r in teammates if r["name"] not in skip and usage.get(gsis.get(r["player_id"]), ("",))[0] == "RB"),
        key=lambda r: -usage[gsis[r["player_id"]]][1],
    )
    out_median = float(out_row["proj_median"])
    changes = []
    with engine.begin() as conn:
        for depth, back in enumerate(backs[: len(RB_SHARES)], start=2):
            old = float(back["proj_median"])
            if old <= 0:
                continue
            mult = (old + RB_SHARES[depth] * out_median) / old
            for r in rows:
                if r["name"] != back["name"] or r["team"] != team or r["locked_at"] is not None:
                    continue
                ladder = r["proj_percentiles"]
                ladder = json.loads(ladder) if isinstance(ladder, str) else ladder
                ladder = {k: round(float(v) * mult, 2) for k, v in ladder.items()}
                conn.execute(text(
                    """UPDATE projections SET proj_floor = :f, proj_median = :m, proj_ceiling = :c, proj_percentiles = :l
                       WHERE slate_id = :s AND player_id = :p"""),
                    {"f": ladder["25"], "m": ladder["50"], "c": ladder["90"], "l": json.dumps(ladder), "s": slate_id, "p": r["player_id"]})
            changes.append((back["name"], round(old, 2), round(old * mult, 2)))
    return changes
