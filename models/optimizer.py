from collections import defaultdict

import pulp
from sqlalchemy import text

from data.dk_salary_csv import CLASSIC_ROSTER, SHOWDOWN_ROSTER
from db.migrate import get_engine

SALARY_CAP = 50000
CLASSIC_ROSTER_SIZE = len(CLASSIC_ROSTER)
SHOWDOWN_ROSTER_SIZE = len(SHOWDOWN_ROSTER)
CLASSIC_POSITION_MINIMUMS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
CLASSIC_FLEX_ELIGIBLE = {"RB", "WR", "TE"}

ALLOWED_PROJECTION_FIELDS = {"proj_floor", "proj_median", "proj_ceiling"}


def _load_player_pool(slate_id, projection_field, engine):
    if projection_field not in ALLOWED_PROJECTION_FIELDS:
        raise ValueError(f"projection_field must be one of {sorted(ALLOWED_PROJECTION_FIELDS)}")
    # projection_field is checked against the fixed whitelist above before use, so
    # it's safe to interpolate here - column names can't be passed as bind params.
    query = text(
        f"""
        SELECT p.player_id, p.name, p.position, p.salary, p.team,
               proj.{projection_field} AS points
        FROM slate_player_pool p
        JOIN projections proj
            ON proj.slate_id = p.slate_id AND proj.player_id = p.player_id
        WHERE p.slate_id = :slate_id AND proj.{projection_field} IS NOT NULL
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"slate_id": slate_id}).mappings().fetchall()
    # proj.* columns are NUMERIC in Postgres, so psycopg2 returns Decimal - PuLP's
    # LpAffineExpression only accepts float/int coefficients and errors on Decimal.
    return [{**dict(row), "points": float(row["points"])} for row in rows]


def _apply_common_constraints(prob, x, players, locked_ids, excluded_ids, max_players_per_team):
    for pid in locked_ids or []:
        if pid in x:
            prob += x[pid] == 1
    for pid in excluded_ids or []:
        if pid in x:
            prob += x[pid] == 0
    if max_players_per_team:
        teams = {p["team"] for p in players}
        for team in teams:
            prob += (
                pulp.lpSum(x[p["player_id"]] for p in players if p["team"] == team)
                <= max_players_per_team
            )


def _apply_diversity_constraints(prob, x, previous_lineups, min_uniques, roster_size):
    # Force each new lineup to swap out at least `min_uniques` players versus every
    # lineup already generated, otherwise the solver just returns the same optimal
    # lineup num_lineups times over.
    for prev_ids in previous_lineups:
        overlap_vars = [x[pid] for pid in prev_ids if pid in x]
        prob += pulp.lpSum(overlap_vars) <= roster_size - min_uniques


def _solve(prob):
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError("No feasible lineup found for the given constraints")


def _solve_classic(players, salary_cap, locked_ids, excluded_ids, max_players_per_team, previous_lineups, min_uniques):
    prob = pulp.LpProblem("dfs_classic", pulp.LpMaximize)
    x = {p["player_id"]: pulp.LpVariable(f"x_{p['player_id']}", cat="Binary") for p in players}

    prob += pulp.lpSum(x[p["player_id"]] * p["points"] for p in players)
    prob += pulp.lpSum(x[p["player_id"]] * p["salary"] for p in players) <= salary_cap
    prob += pulp.lpSum(x.values()) == CLASSIC_ROSTER_SIZE

    prob += pulp.lpSum(x[p["player_id"]] for p in players if p["position"] == "QB") == 1
    prob += pulp.lpSum(x[p["player_id"]] for p in players if p["position"] == "DST") == 1
    for pos in ("RB", "WR", "TE"):
        prob += (
            pulp.lpSum(x[p["player_id"]] for p in players if p["position"] == pos)
            >= CLASSIC_POSITION_MINIMUMS[pos]
        )
    # One extra RB/WR/TE beyond the position minimums fills the FLEX slot; which
    # position it comes from is left to the solver rather than fixed in advance.
    prob += (
        pulp.lpSum(x[p["player_id"]] for p in players if p["position"] in CLASSIC_FLEX_ELIGIBLE)
        == sum(CLASSIC_POSITION_MINIMUMS[pos] for pos in CLASSIC_FLEX_ELIGIBLE) + 1
    )

    _apply_common_constraints(prob, x, players, locked_ids, excluded_ids, max_players_per_team)
    _apply_diversity_constraints(prob, x, previous_lineups, min_uniques, CLASSIC_ROSTER_SIZE)

    _solve(prob)
    selected = [p for p in players if x[p["player_id"]].value() == 1]
    return _assign_classic_slots(selected)


def _assign_classic_slots(selected):
    by_pos = defaultdict(list)
    for p in selected:
        by_pos[p["position"]].append(p)

    slots = [("QB", by_pos["QB"].pop())]
    slots += [("RB", by_pos["RB"].pop()) for _ in range(CLASSIC_POSITION_MINIMUMS["RB"])]
    slots += [("WR", by_pos["WR"].pop()) for _ in range(CLASSIC_POSITION_MINIMUMS["WR"])]
    slots.append(("TE", by_pos["TE"].pop()))
    # Exactly one RB/WR/TE is left over here - the FLEX constraint above guarantees it.
    flex_pool = by_pos["RB"] + by_pos["WR"] + by_pos["TE"]
    slots.append(("FLEX", flex_pool[0]))
    slots.append(("DST", by_pos["DST"].pop()))
    return slots


def _solve_showdown(players, salary_cap, locked_ids, excluded_ids, max_players_per_team, previous_lineups, min_uniques):
    prob = pulp.LpProblem("dfs_showdown", pulp.LpMaximize)
    x = {p["player_id"]: pulp.LpVariable(f"x_{p['player_id']}", cat="Binary") for p in players}

    # DraftKings already bakes the 1.5x Captain multiplier into both the salary and
    # AvgPointsPerGame columns for the CPT row (a distinct player_id from that same
    # player's FLEX row), so no multiplier is applied here - doing so would double it.
    prob += pulp.lpSum(x[p["player_id"]] * p["points"] for p in players)
    prob += pulp.lpSum(x[p["player_id"]] * p["salary"] for p in players) <= salary_cap
    prob += pulp.lpSum(x.values()) == SHOWDOWN_ROSTER_SIZE
    prob += pulp.lpSum(x[p["player_id"]] for p in players if p["position"] == "CPT") == 1

    # The CPT and FLEX rows for the same real player have different IDs but are the
    # same person - a lineup can only include them once, in one slot or the other.
    by_name = defaultdict(list)
    for p in players:
        by_name[p["name"]].append(p)
    for rows in by_name.values():
        if len(rows) > 1:
            prob += pulp.lpSum(x[p["player_id"]] for p in rows) <= 1

    _apply_common_constraints(prob, x, players, locked_ids, excluded_ids, max_players_per_team)
    _apply_diversity_constraints(prob, x, previous_lineups, min_uniques, SHOWDOWN_ROSTER_SIZE)

    _solve(prob)
    selected = [p for p in players if x[p["player_id"]].value() == 1]
    cpt = next(p for p in selected if p["position"] == "CPT")
    flex = [p for p in selected if p["position"] != "CPT"]
    return [("CPT", cpt)] + [("FLEX", p) for p in flex]


def generate_lineups(
    slate_id,
    num_lineups=1,
    projection_field="proj_median",
    salary_cap=SALARY_CAP,
    locked_player_ids=None,
    excluded_player_ids=None,
    max_players_per_team=None,
    min_uniques=1,
    max_exposure=None,
    engine=None,
):
    engine = engine or get_engine()
    players = _load_player_pool(slate_id, projection_field, engine)
    if not players:
        raise ValueError(f"No players with a '{projection_field}' projection found for slate {slate_id}")

    slate_type = "showdown" if any(p["position"] == "CPT" for p in players) else "classic"
    solve_fn = _solve_showdown if slate_type == "showdown" else _solve_classic

    lineups = []
    previous_lineups = []
    exposure_counts = defaultdict(int)
    excluded_ids = set(excluded_player_ids or [])

    for i in range(num_lineups):
        # max_exposure is enforced by hard-excluding a player once they've hit their
        # cap across lineups generated so far, rather than a true exposure-aware
        # MILP - much simpler, and good enough at the lineup counts DFS players
        # actually build (dozens, not thousands).
        exposure_excluded = (
            {pid for pid, count in exposure_counts.items() if count >= max_exposure * num_lineups}
            if max_exposure is not None
            else set()
        )
        try:
            slots = solve_fn(
                players,
                salary_cap,
                locked_player_ids,
                excluded_ids | exposure_excluded,
                max_players_per_team,
                previous_lineups,
                min_uniques,
            )
        except ValueError:
            if i == 0:
                raise
            break  # diversity/exposure constraints have exhausted the feasible pool

        lineup_ids = [p["player_id"] for _, p in slots]
        previous_lineups.append(set(lineup_ids))
        for pid in lineup_ids:
            exposure_counts[pid] += 1

        lineups.append(
            {
                "roster": slots,
                "total_salary": sum(p["salary"] for _, p in slots),
                "total_points": sum(p["points"] for _, p in slots),
            }
        )

    return lineups


def lineup_player_ids(lineup):
    return [p["player_id"] for _, p in lineup["roster"]]
