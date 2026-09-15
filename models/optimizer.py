import math
from collections import defaultdict

import pulp
from sqlalchemy import text

from data.dk_salary_csv import CLASSIC_ROSTER, SHOWDOWN_ROSTER
from data.player_availability import get_availability_gate, resolve_slate_season_week
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
    players = [{**dict(row), "points": float(row["points"])} for row in rows]

    # Hard blocker, not opt-in: a live slate's player pool is never handed to
    # the solver without checking current roster/injury status first. Refusing
    # to guess the slate's (season, week) rather than silently skipping the
    # gate if it can't be resolved - see data/player_availability.py.
    season_week = resolve_slate_season_week(slate_id, engine)
    if season_week is None:
        raise RuntimeError(
            f"Could not determine (season, week) for slate {slate_id} - refusing to build a lineup "
            "without checking current roster/injury status first"
        )
    season, week = season_week
    excluded, flagged, injury_report_available = get_availability_gate(players, season, week, engine)

    available_players = [p for p in players if p["player_id"] not in excluded]
    for p in available_players:
        p["availability_flag"] = flagged.get(p["player_id"])

    return available_players, excluded, injury_report_available


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


def _resolve_exposure(spec, player_id):
    # spec is None (no cap/floor at all), a single float/int (applies to every
    # player), or a dict of per-player overrides with an optional "default" for
    # players not explicitly listed.
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec[player_id] if player_id in spec else spec.get("default")
    return spec


def build_lineups_from_pool(
    players,
    num_lineups=1,
    salary_cap=SALARY_CAP,
    locked_player_ids=None,
    excluded_player_ids=None,
    max_players_per_team=None,
    min_uniques=1,
    max_exposure=None,
    min_exposure=None,
):
    """Core multi-lineup builder, operating on an in-memory player pool (dicts
    with player_id/name/position/salary/team/points) instead of loading from
    the DB - shared by generate_lineups() (points from the projections table)
    and models/backtest.py (points from as-of projections or real actual
    scores computed in Python, never written to the DB at all).

    max_exposure/min_exposure: None, a single fraction (0-1) applied to every
    player, or {player_id: fraction, "default": fraction}.

    Returns (lineups, exposure_report) where exposure_report is
    {player_id: {"count": n, "fraction": n / lineups_actually_built}} - the
    achieved exposure, which may fall short of requested max/min_exposure if
    the caps made later lineups infeasible (see the ValueError handling
    below); the report is what lets a caller notice that happened.
    """
    if not players:
        raise ValueError("No players in the given pool")

    slate_type = "showdown" if any(p["position"] == "CPT" for p in players) else "classic"
    solve_fn = _solve_showdown if slate_type == "showdown" else _solve_classic

    lineups = []
    previous_lineups = []
    exposure_counts = defaultdict(int)
    base_excluded_ids = set(excluded_player_ids or [])
    base_locked_ids = list(locked_player_ids or [])

    for i in range(num_lineups):
        lineups_remaining = num_lineups - i

        # max_exposure is checked against the final target lineup count, not
        # lineups built so far: a player is excluded once their count would
        # put them over cap * num_lineups. This does mean a popular player
        # gets included in the first cap*num_lineups solves and hard-excluded
        # after that, rather than interleaved throughout the run - but that's
        # just an ordering artifact; the final achieved exposure fraction is
        # identical either way, and comparing against lineups-built-so-far
        # instead would make cap enforcement impossible on lineup 1 (every
        # player's "fraction so far" starts at 100% the moment they're used
        # once out of one lineup built).
        exposure_excluded = set()
        forced_locks = []
        for p in players:
            pid = p["player_id"]
            cap = _resolve_exposure(max_exposure, pid)
            if cap is not None and exposure_counts[pid] >= cap * num_lineups:
                exposure_excluded.add(pid)

            floor = _resolve_exposure(min_exposure, pid)
            if floor is not None:
                # Lineups still needed to reach this player's floor by the end
                # of the run - once that equals the lineups left to build,
                # they must be locked into every remaining one to make it.
                needed = math.ceil(floor * num_lineups) - exposure_counts[pid]
                if needed >= lineups_remaining:
                    forced_locks.append(pid)

        conflicts = exposure_excluded & set(forced_locks)
        if conflicts:
            # A configuration error, not something to silently resolve one way -
            # the caller asked for this player both capped out and floored in
            # for the same lineup.
            raise ValueError(f"max_exposure and min_exposure conflict for player(s): {sorted(conflicts)}")

        locked_ids = list(dict.fromkeys(base_locked_ids + forced_locks))

        try:
            slots = solve_fn(
                players,
                salary_cap,
                locked_ids,
                base_excluded_ids | exposure_excluded,
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

    built = len(lineups)
    exposure_report = {
        pid: {"count": count, "fraction": round(count / built, 4) if built else 0.0}
        for pid, count in exposure_counts.items()
    }

    return lineups, exposure_report


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
    min_exposure=None,
    engine=None,
):
    engine = engine or get_engine()
    players, availability_excluded, injury_report_available = _load_player_pool(slate_id, projection_field, engine)
    if not players:
        raise ValueError(f"No players with a '{projection_field}' projection found for slate {slate_id}")

    lineups, exposure_report = build_lineups_from_pool(
        players,
        num_lineups=num_lineups,
        salary_cap=salary_cap,
        locked_player_ids=locked_player_ids,
        excluded_player_ids=excluded_player_ids,
        max_players_per_team=max_players_per_team,
        min_uniques=min_uniques,
        max_exposure=max_exposure,
        min_exposure=min_exposure,
    )

    availability_report = {
        # Hard-excluded before the solver ever saw them - see
        # data/player_availability.py. {dk_player_id: reason}.
        "excluded": availability_excluded,
        # Still eligible, but risky - Questionable/Doubtful on the real
        # current injury report. {dk_player_id: status}.
        "flagged": {p["player_id"]: p["availability_flag"] for p in players if p.get("availability_flag")},
        # Injury reports are filed Wed-Fri of game week - False here means
        # nflverse hasn't published this week's report yet, not that everyone
        # is confirmed healthy. Roster-status exclusions (IR/PUP/Suspended)
        # are unaffected either way.
        "injury_report_available": injury_report_available,
    }
    return lineups, exposure_report, availability_report


def lineup_player_ids(lineup):
    return [p["player_id"] for _, p in lineup["roster"]]
