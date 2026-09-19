import math
from collections import Counter, defaultdict

import pulp
from sqlalchemy import text

from data.dk_salary_csv import CLASSIC_ROSTER, SHOWDOWN_ROSTER
from data.player_availability import game_lock_status, get_availability_gate, resolve_slate_season_week
from data.player_crosswalk import resolve_dk_players_to_gsis
from data.pre_lock_check import hard_role_exclusions
from db.migrate import get_engine
from models.playing_time_engine import apply_playing_time_gate
from models.simulation import DEFAULT_NUM_SIMULATIONS, select_best_by_simulation

SALARY_CAP = 50000
CLASSIC_ROSTER_SIZE = len(CLASSIC_ROSTER)
SHOWDOWN_ROSTER_SIZE = len(SHOWDOWN_ROSTER)
CLASSIC_POSITION_MINIMUMS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
CLASSIC_FLEX_ELIGIBLE = {"RB", "WR", "TE"}
CLASSIC_SLOT_ELIGIBLE_POSITIONS = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "FLEX": CLASSIC_FLEX_ELIGIBLE,
    "DST": {"DST"},
}

ALLOWED_PROJECTION_FIELDS = {"proj_floor", "proj_median", "proj_ceiling"}


class LineupValidationError(ValueError):
    """Raised by validate_lineup() when a lineup isn't actually legal to
    submit. A subclass of ValueError so it's still catchable anywhere that
    already handles ValueError, but named distinctly so it's never confused
    with - or silently swallowed by - the solver-infeasibility ValueError
    build_lineups_from_pool's own loop already catches (see that function
    for why validate_lineup is called OUTSIDE that try/except block: a real
    validation failure must always raise, never be treated as "diversity/
    exposure constraints exhausted the feasible pool").
    """


def _load_player_pool(slate_id, projection_field, engine, punt_mode=False):
    if projection_field not in ALLOWED_PROJECTION_FIELDS:
        raise ValueError(f"projection_field must be one of {sorted(ALLOWED_PROJECTION_FIELDS)}")
    # projection_field is checked against the fixed whitelist above before use, so
    # it's safe to interpolate here - column names can't be passed as bind params.
    # proj_floor/proj_ceiling are always fetched alongside whichever field
    # drives the main objective, not just the one field - generate_cash_lineups
    # below needs both to build its variance-penalized objective.
    query = text(
        f"""
        SELECT p.player_id, p.name, p.position, p.salary, p.team, p.opponent, p.game_time,
               proj.{projection_field} AS points,
               proj.proj_floor, proj.proj_ceiling
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
    players = [
        {
            **dict(row),
            "points": float(row["points"]),
            "proj_floor": float(row["proj_floor"]),
            "proj_ceiling": float(row["proj_ceiling"]),
        }
        for row in rows
    ]

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

    # Real "is a teammate QB confirmed out this week" signal for hard_role_
    # exclusions' QB pattern check - reuses the availability gate's OWN
    # real, already-computed exclusions above (no separate fetch) rather
    # than re-deriving injury status a second way. Resolved for every
    # excluded player regardless of position (not just ones already known
    # to be QBs) - hard_role_exclusions' own real_position resolution
    # (Showdown-safe) decides which of these are actually teammate QBs; a
    # DST or WR's gsis_id here is harmless, it just won't match anything.
    excluded_players = [p for p in players if p["player_id"] in excluded]
    gsis_by_excluded_id, _, _ = resolve_dk_players_to_gsis(excluded_players, engine) if excluded_players else ({}, [], [])
    unavailable_gsis_ids = {gid for gid in gsis_by_excluded_id.values() if gid is not None}

    # Second hard gate, same standard as the roster/IR check above: a player
    # can be rostered and healthy (passes get_availability_gate) and still
    # have no real current role - a real backup with a starter-level
    # projection (Jameis Winston, see data/pre_lock_check.py's module
    # docstring). Checked against the pool that already survived the
    # availability gate, not the raw pool, so this never does redundant work
    # resolving a player who's excluded already.
    role_excluded = hard_role_exclusions(available_players, engine, unavailable_gsis_ids=unavailable_gsis_ids)
    excluded = {**excluded, **role_excluded}
    available_players = [p for p in available_players if p["player_id"] not in role_excluded]

    # Fourth hard gate, a distinct real claim from the two above: ACTIVE
    # (roster/injury-clear, already true of every player still in
    # available_players here) is not the same real fact as "has enough real
    # expected offensive opportunity to be a DFS play" - see models/
    # playing_time_engine.py's own module docstring for the real
    # distinction and the real backtest behind it
    # (models/calibration.py::run_playing_time_floor_backtest). Every
    # player reaching this point already cleared the roster/injury gate
    # above, so no roster_status/injury_status dicts are passed through -
    # classify_player_status correctly reads that as ACTIVE for all of
    # them, which is already the real, established fact at this point in
    # the pipeline; this gate's only new real work is the playing-time
    # floor itself.
    playing_time_result = apply_playing_time_gate(
        available_players, season, week, engine, punt_mode=punt_mode
    )
    playing_time_excluded_ids = {e["player_id"]: e["reason"] for e in playing_time_result["excluded"]}
    excluded = {**excluded, **playing_time_excluded_ids}
    available_players = playing_time_result["eligible"]
    playing_time_debug_log = playing_time_result["debug_log"]
    playing_time_punt_flagged = playing_time_result["punt_flagged"]

    # PUNT MODE only (punt_mode=False leaves punt_flagged empty and this is
    # a no-op): a player who failed the real playing-time floor stays
    # eligible but gets his own points/floor/ceiling scaled down by the
    # same severe, fixed real penalty_multiplier the debug entry already
    # discloses - applied to every projection field the solver could
    # optimize on, not just whichever one this call happens to use, so
    # switching projection_field later can't silently undo the penalty.
    if playing_time_punt_flagged:
        penalty_by_id = {e["player_id"]: e["penalty_multiplier"] for e in playing_time_punt_flagged}
        for p in available_players:
            multiplier = penalty_by_id.get(p["player_id"])
            if multiplier is not None:
                p["points"] *= multiplier
                p["proj_floor"] *= multiplier
                p["proj_ceiling"] *= multiplier

    # Fifth hard gate: a player whose real game has already started can
    # never legally be newly rostered - DraftKings itself enforces this
    # (late swap only lets you touch a still-open slot), and unlike the two
    # gates above this is a plain, deterministic fact (a kickoff time has
    # passed or it hasn't), not a threshold call. Matters on any multi-day
    # slate (Thu-Mon) where some games lock while others are still hours
    # away - see data/player_availability.py's lock-time section.
    locked = game_lock_status(available_players)
    game_locked_ids = {pid for pid, is_locked in locked.items() if is_locked}
    excluded = {**excluded, **{pid: "game already started" for pid in game_locked_ids}}
    available_players = [p for p in available_players if p["player_id"] not in game_locked_ids]

    for p in available_players:
        p["availability_flag"] = flagged.get(p["player_id"])

    return available_players, excluded, injury_report_available, playing_time_debug_log


def _apply_common_constraints(prob, x, players, locked_ids, excluded_ids, max_players_per_team, min_salary=None):
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
    if min_salary:
        # Without this, a risk-averse objective (see generate_cash_lineups) can
        # rationally prefer leaving salary unspent over rostering an expensive
        # player whose upside also brings volatility - caught for real: an
        # early cash-mode run left $17k of a $50k cap unused, filling slots
        # with zero-projection scrubs instead, since a guaranteed zero has
        # zero spread too. This forces the solver to still spend real budget.
        prob += pulp.lpSum(x[p["player_id"]] * p["salary"] for p in players) >= min_salary


def _apply_diversity_constraints(prob, x, previous_lineups, min_uniques, roster_size):
    # Force each new lineup to swap out at least `min_uniques` players versus every
    # lineup already generated, otherwise the solver just returns the same optimal
    # lineup num_lineups times over.
    for prev_ids in previous_lineups:
        overlap_vars = [x[pid] for pid in prev_ids if pid in x]
        prob += pulp.lpSum(overlap_vars) <= roster_size - min_uniques


def _apply_qb_stack_constraint(prob, x, players):
    # Correlated ceiling outcomes are the actual mechanism by which a GPP
    # lineup wins a tournament (see models/simulation.py's real correlation
    # model - QB_PASS_CATCHER_CORR=0.55, the single strongest same-game
    # correlation it encodes) - but until this existed, nothing in the
    # solver's own objective knew that: it summed independent point
    # estimates with no covariance term, so a ceiling-maximizing build had
    # no reason to prefer a QB+pass-catcher combo over 9 unrelated high-
    # ceiling players. Verified for real: 19 of 20 lineups from a fresh live
    # GPP run had zero same-team QB+WR/TE pairing before this existed.
    #
    # PuLP/CBC only solves LINEAR programs - there's no way to directly
    # optimize for the simulated correlation benefit itself inside the MILP
    # (that would need a quadratic/covariance term). What IS linear and
    # captures the same real, standard DFS construction principle: require
    # at least one same-team WR/TE alongside whichever QB gets selected.
    # For binary x, `sum(same-team pass catchers) >= x[qb]` is a standard
    # linear implication - it's automatically satisfied when the QB isn't
    # selected (x[qb]=0), and forces at least one pass-catcher in when he
    # is (x[qb]=1).
    #
    # Skipped for a team with zero WR/TE candidates left in the pool (every
    # one hard-excluded, or genuinely none on the slate) rather than forcing
    # that QB to zero - an interaction with an unrelated gate shouldn't
    # silently make an otherwise-legal QB unselectable.
    by_team = defaultdict(lambda: {"qb": [], "pass_catchers": []})
    for p in players:
        if p["position"] == "QB":
            by_team[p["team"]]["qb"].append(p["player_id"])
        elif p["position"] in ("WR", "TE"):
            by_team[p["team"]]["pass_catchers"].append(p["player_id"])

    for team_players in by_team.values():
        if not team_players["pass_catchers"]:
            continue
        pass_catcher_sum = pulp.lpSum(x[pid] for pid in team_players["pass_catchers"])
        for qb_id in team_players["qb"]:
            prob += pass_catcher_sum >= x[qb_id]


def _apply_bring_back_constraint(prob, x, players):
    # The other half of a real "game stack": a pass-catcher from the QB's
    # OPPONENT, alongside his own team's pass-catcher from
    # _apply_qb_stack_constraint. This is the second correlation
    # models/simulation.py's real model already encodes (BRING_BACK_CORR=
    # 0.20 - the opposing offense scoring back in a real shootout) but,
    # like the primary stack before it, was never used in construction,
    # only in post-hoc scoring. Same linear-implication technique: require
    # at least one of the QB's opponent's WR/TE alongside him. Deliberately
    # opt-in (unlike require_qb_stack, which defaults on in
    # generate_lineups) - dedicating a THIRD roster slot to one game's
    # correlation is a real construction cost, not something every lineup
    # should pay by default.
    #
    # Needs each player's real `opponent` field, which _apply_qb_stack_
    # constraint never needed - skipped (not forced to zero) for a QB
    # whose opponent has no pass-catcher candidates left in the pool, same
    # reasoning as the primary stack's skip case.
    qbs_by_team = defaultdict(list)
    pass_catchers_by_team = defaultdict(list)
    opponent_by_team = {}
    for p in players:
        if p["position"] == "QB":
            qbs_by_team[p["team"]].append(p["player_id"])
            if p.get("opponent"):
                opponent_by_team[p["team"]] = p["opponent"]
        elif p["position"] in ("WR", "TE"):
            pass_catchers_by_team[p["team"]].append(p["player_id"])

    for team, qb_ids in qbs_by_team.items():
        opponent = opponent_by_team.get(team)
        opponent_pass_catchers = pass_catchers_by_team.get(opponent, []) if opponent else []
        if not opponent_pass_catchers:
            continue
        pass_catcher_sum = pulp.lpSum(x[pid] for pid in opponent_pass_catchers)
        for qb_id in qb_ids:
            prob += pass_catcher_sum >= x[qb_id]


def _solve(prob):
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise ValueError("No feasible lineup found for the given constraints")


def _solve_classic(
    players,
    salary_cap,
    locked_ids,
    excluded_ids,
    max_players_per_team,
    previous_lineups,
    min_uniques,
    min_salary=None,
    require_qb_stack=False,
    require_bring_back=False,
):
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

    _apply_common_constraints(prob, x, players, locked_ids, excluded_ids, max_players_per_team, min_salary)
    _apply_diversity_constraints(prob, x, previous_lineups, min_uniques, CLASSIC_ROSTER_SIZE)
    if require_qb_stack:
        _apply_qb_stack_constraint(prob, x, players)
    if require_bring_back:
        _apply_bring_back_constraint(prob, x, players)

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


def _solve_showdown(
    players,
    salary_cap,
    locked_ids,
    excluded_ids,
    max_players_per_team,
    previous_lineups,
    min_uniques,
    min_salary=None,
    require_qb_stack=False,
    require_bring_back=False,
):
    # require_qb_stack/require_bring_back are accepted (not just missing) so
    # build_lineups_from_pool can call either solver with the same keyword
    # args, but both are no-ops here: a showdown roster is only 6 players,
    # all drawn from the SAME 2 teams in one game, so "stack the QB with a
    # teammate" (or an opponent) doesn't carve out a distinct correlated
    # subset the way it does in a 9-player classic roster pulled from up to
    # 8 different games - most showdown rosters already include a
    # meaningful share of both teams by construction.
    del require_qb_stack, require_bring_back
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

    _apply_common_constraints(prob, x, players, locked_ids, excluded_ids, max_players_per_team, min_salary)
    _apply_diversity_constraints(prob, x, previous_lineups, min_uniques, SHOWDOWN_ROSTER_SIZE)

    _solve(prob)
    selected = [p for p in players if x[p["player_id"]].value() == 1]
    cpt = next(p for p in selected if p["position"] == "CPT")
    flex = [p for p in selected if p["position"] != "CPT"]
    return [("CPT", cpt)] + [("FLEX", p) for p in flex]


def _validate_classic_roster(roster):
    problems = []
    players = [p for _, p in roster]

    if len(roster) != CLASSIC_ROSTER_SIZE:
        problems.append(f"roster has {len(roster)} players, expected {CLASSIC_ROSTER_SIZE}")

    # The slot-label template itself (exactly one QB/FLEX/TE/DST, two RB,
    # three WR) - checked BEFORE per-slot eligibility below, because
    # eligibility alone isn't enough: a real TE labeled "TE" twice (and
    # "FLEX" zero times) passes every individual eligibility check (a TE
    # really is eligible for a slot called "TE") while still not being the
    # roster DraftKings actually requires. This is the check that would
    # have caught the real incident directly - a FLEX slot mislabeled as a
    # second dedicated TE slot in a human-written summary.
    slot_counts = Counter(slot for slot, _ in roster)
    expected_slot_counts = Counter(CLASSIC_ROSTER)
    if slot_counts != expected_slot_counts:
        problems.append(f"slot labels {dict(slot_counts)} don't match the required {dict(expected_slot_counts)}")

    for slot, p in roster:
        eligible = CLASSIC_SLOT_ELIGIBLE_POSITIONS.get(slot)
        if eligible is None:
            problems.append(f"unrecognized slot {slot!r} for {p.get('name')}")
        elif p["position"] not in eligible:
            problems.append(f"{p.get('name')} ({p['position']}) is not eligible for slot {slot!r}")

    # Real position counts, derived from each player's own position - never
    # from slot labels, which is exactly what let a legal roster get
    # mislabeled as illegal in a human-written summary once already (see
    # validate_lineup's docstring). A slot can say whatever it wants; this
    # is the actual, independent source of truth.
    position_counts = Counter(p["position"] for p in players)
    if position_counts.get("QB", 0) != 1:
        problems.append(f"expected exactly 1 QB, got {position_counts.get('QB', 0)}")
    if position_counts.get("DST", 0) != 1:
        problems.append(f"expected exactly 1 DST, got {position_counts.get('DST', 0)}")
    for pos in ("RB", "WR", "TE"):
        if position_counts.get(pos, 0) < CLASSIC_POSITION_MINIMUMS[pos]:
            problems.append(
                f"expected at least {CLASSIC_POSITION_MINIMUMS[pos]} {pos}, got {position_counts.get(pos, 0)}"
            )
    # The one extra RB/WR/TE beyond the position minimums fills FLEX - could
    # legally be a 3rd RB, a 4th WR, or a 2nd TE. Any of those is fine;
    # anything else (too many or too few total across RB+WR+TE) isn't.
    flex_pool_count = sum(position_counts.get(pos, 0) for pos in CLASSIC_FLEX_ELIGIBLE)
    expected_flex_pool_count = sum(CLASSIC_POSITION_MINIMUMS[pos] for pos in CLASSIC_FLEX_ELIGIBLE) + 1
    if flex_pool_count != expected_flex_pool_count:
        problems.append(f"RB+WR+TE count is {flex_pool_count}, expected exactly {expected_flex_pool_count}")

    return problems


def _validate_showdown_roster(roster):
    problems = []
    players = [p for _, p in roster]

    if len(roster) != SHOWDOWN_ROSTER_SIZE:
        problems.append(f"roster has {len(roster)} players, expected {SHOWDOWN_ROSTER_SIZE}")

    cpt_count = sum(1 for p in players if p["position"] == "CPT")
    if cpt_count != 1:
        problems.append(f"expected exactly 1 CPT, got {cpt_count}")

    # The CPT and FLEX rows for the same real player have different
    # player_ids but are the same person - a real name showing up twice
    # means that person got rostered in both slots.
    names = [p["name"] for p in players]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        problems.append(f"same real player rostered twice (CPT and FLEX): {dupes}")

    for slot, p in roster:
        if slot not in ("CPT", "FLEX"):
            problems.append(f"unrecognized slot {slot!r} for {p.get('name')}")

    return problems


def validate_lineup(lineup, salary_cap=SALARY_CAP):
    """The final, independent legality check every lineup passes through
    before it can ever leave build_lineups_from_pool - real position counts
    and salary re-derived from the actual roster data (never from slot
    labels, never from solver internals, never from a caller's own summary
    of it), so this catches a real solver bug the same way it would catch a
    downstream reporting bug.

    Added after a real incident: a lineup reported to the user labeled a
    FLEX slot - correctly filled by a second TE, which is legal under
    DraftKings' own rules (FLEX accepts RB/WR/TE) - as a second dedicated
    "TE" slot in a human-written summary table, making a fully legal 2 RB/
    3 WR/1 TE/1 FLEX-as-TE roster read as an illegal 2 RB/2 TE roster
    missing an RB. The underlying optimizer output was correct the whole
    time - nothing had ever independently re-verified the REPORTED lineup
    against real DK roster rules before this, so a presentation mistake
    was indistinguishable from a real solver bug until counted by hand.

    Auto-detects classic vs showdown the same way build_lineups_from_pool
    does (a "CPT" position anywhere in the roster). Raises
    LineupValidationError listing every problem found at once (not just
    the first) if the lineup is illegal - never returns a partial or
    best-effort verdict, and never silently drops a bad lineup instead of
    raising.
    """
    roster = lineup.get("roster", [])
    players = [p for _, p in roster]
    problems = []

    player_ids = [p["player_id"] for p in players]
    if len(set(player_ids)) != len(player_ids):
        dupes = sorted({pid for pid in player_ids if player_ids.count(pid) > 1})
        problems.append(f"duplicate player(s) in roster: {dupes}")

    total_salary = sum(p["salary"] for p in players)
    if total_salary > salary_cap:
        problems.append(f"total salary ${total_salary} exceeds cap ${salary_cap}")

    is_showdown = any(p["position"] == "CPT" for p in players)
    problems.extend(_validate_showdown_roster(roster) if is_showdown else _validate_classic_roster(roster))

    if problems:
        raise LineupValidationError("; ".join(problems))
    return True


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
    min_salary=None,
    require_qb_stack=False,
    require_bring_back=False,
):
    """Core multi-lineup builder, operating on an in-memory player pool (dicts
    with player_id/name/position/salary/team/points) instead of loading from
    the DB - shared by generate_lineups() (points from the projections table)
    and models/backtest.py (points from as-of projections or real actual
    scores computed in Python, never written to the DB at all).

    max_exposure/min_exposure: None, a single fraction (0-1) applied to every
    player, or {player_id: fraction, "default": fraction}.

    require_qb_stack: classic slates only (see _apply_qb_stack_constraint) -
    require at least one same-team WR/TE alongside whichever QB is selected,
    the correlation-aware construction real GPP play depends on. Defaults to
    False here so every existing caller (models/backtest.py, calibration.py,
    field_simulation.py all call this directly, not through generate_lineups)
    keeps its already-validated behavior unchanged; generate_lineups()
    defaults it to True instead, since that's the actual live GPP path.

    require_bring_back: classic slates only (see _apply_bring_back_constraint)
    - require at least one WR/TE from the QB's OPPONENT too, the other half
    of a real "game stack." Unlike require_qb_stack, this defaults to False
    everywhere, including generate_lineups() - dedicating a third roster
    slot to one game's correlation is a real construction cost that not
    every lineup should pay by default; pass True explicitly for a
    deliberate full-game-stack build.

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
                min_salary,
                require_qb_stack,
                require_bring_back,
            )
        except ValueError:
            if i == 0:
                raise
            break  # diversity/exposure constraints have exhausted the feasible pool

        lineup = {
            "roster": slots,
            "total_salary": sum(p["salary"] for _, p in slots),
            "total_points": sum(p["points"] for _, p in slots),
        }
        # The first, hard gate on the OUTPUT side, the same way the availability/
        # role/lock-time gates are hard gates on the input side - deliberately
        # OUTSIDE the try/except above, so a real validation failure always
        # raises LineupValidationError and is never mistaken for the solver
        # simply running out of feasible lineups (see that exception's
        # docstring). No lineup this function returns has skipped this check.
        validate_lineup(lineup, salary_cap=salary_cap)

        lineup_ids = [p["player_id"] for _, p in slots]
        previous_lineups.append(set(lineup_ids))
        for pid in lineup_ids:
            exposure_counts[pid] += 1

        lineups.append(lineup)

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
    require_qb_stack=True,
    require_bring_back=False,
    punt_mode=False,
    engine=None,
):
    """The live GPP-style lineup path (models/backtest.py, calibration.py,
    and field_simulation.py all call build_lineups_from_pool directly for
    their own validation runs, never through here - see that function's
    docstring). require_qb_stack defaults to True here specifically because
    this is the function real lineup generation actually goes through:
    without it, the solver's objective has no covariance term at all and
    has no reason to prefer a correlated QB+pass-catcher combo over 9
    unrelated high-projection players, even though correlated ceiling
    outcomes are the real mechanism a GPP lineup wins by (see
    _apply_qb_stack_constraint). Pass False to opt back out.

    require_bring_back defaults to False even here (see
    build_lineups_from_pool's docstring) - pass True for a deliberate,
    opt-in full-game-stack build (QB + own pass-catcher + opponent
    pass-catcher).

    punt_mode: see models/playing_time_engine.py's own docstring. False
    (default, NORMAL MODE) hard-excludes anyone who fails the real
    playing-time floor before the solver ever sees them. True lets them
    back in with a severe, real penalty applied to their own points/floor/
    ceiling instead - an explicit opt-in, never the default.
    """
    engine = engine or get_engine()
    players, availability_excluded, injury_report_available, playing_time_debug_log = _load_player_pool(
        slate_id, projection_field, engine, punt_mode=punt_mode
    )
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
        require_qb_stack=require_qb_stack,
        require_bring_back=require_bring_back,
    )

    availability_report = _build_availability_report(players, availability_excluded, injury_report_available)
    # Item 13's own requirement: this debug information must be visible to
    # the user, not just logged - every real playing-time verdict (excluded
    # AND eligible alike), not only the ones that ended up excluded.
    availability_report["playing_time_debug_log"] = playing_time_debug_log
    return lineups, exposure_report, availability_report


def _build_availability_report(players, availability_excluded, injury_report_available):
    return {
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


def generate_simulation_selected_lineup(
    slate_id,
    num_candidates=10,
    projection_field="proj_ceiling",
    num_simulations=DEFAULT_NUM_SIMULATIONS,
    seed=None,
    engine=None,
    **generate_lineups_kwargs,
):
    """The real selection layer this codebase's MILP-only path was missing:
    build `num_candidates` diverse, legal GPP lineups the normal way (via
    generate_lineups - same hard gates, same stacking), then rank them by
    REAL SIMULATED win rate (models/simulation.py's correlation-aware
    Monte Carlo engine) instead of just handing back whichever one
    happened to have the highest raw linear-objective ceiling sum.

    This is the actual mechanism a simulation-driven optimizer uses -
    score correlated outcomes, then pick the lineup that wins the most
    simulated worlds - applied as a selection step on top of the existing
    MILP candidate pool rather than a full simulation-native rebuild of
    the solver itself (which would need a quadratic objective CBC can't
    solve). min_uniques defaults to 3 here specifically (overridable via
    generate_lineups_kwargs) so the candidates represent meaningfully
    different bets, not near-duplicates of the same core with one swap -
    ranking near-identical lineups by simulation wouldn't tell you much.

    Returns (ranked, exposure_report, availability_report) - ranked[0] is
    the recommended lineup, with its real win_rate/mean_score/p10/p50/p90
    attached (see select_best_by_simulation), not a black-box pick.
    """
    engine = engine or get_engine()
    generate_lineups_kwargs.setdefault("min_uniques", 3)
    lineups, exposure_report, availability_report = generate_lineups(
        slate_id,
        num_lineups=num_candidates,
        projection_field=projection_field,
        engine=engine,
        **generate_lineups_kwargs,
    )
    if len(lineups) < 2:
        raise ValueError(
            f"Only {len(lineups)} legal candidate lineup(s) could be built for slate {slate_id} - "
            "need at least 2 to rank by simulation (try a smaller min_uniques or max_exposure)"
        )

    ranked = select_best_by_simulation(lineups, slate_id, num_simulations=num_simulations, seed=seed, engine=engine)
    return ranked, exposure_report, availability_report


def generate_cash_lineups(
    slate_id,
    num_lineups=1,
    risk_aversion=1.0,
    salary_cap=SALARY_CAP,
    min_salary_fraction=0.95,
    locked_player_ids=None,
    excluded_player_ids=None,
    max_players_per_team=3,
    min_uniques=1,
    engine=None,
):
    """Cash-mode lineup builder with an explicit variance penalty, not just a
    different projection field.

    The solver is a linear MILP (PuLP/CBC) - it can't optimize true lineup
    variance directly, since that needs the covariance between every pair of
    rostered players (a quadratic term). Instead, each player's own
    (ceiling - floor) spread is used as a linear proxy for how much risk they
    add, and the objective becomes:

        maximize  sum(floor_i * x_i) - risk_aversion * sum((ceiling_i - floor_i) * x_i)

    which rewards floor and directly punishes width in the same pass, instead
    of just picking a safer percentile to maximize and hoping the width comes
    along for the ride. max_players_per_team defaults to 3 here (vs.
    unrestricted for GPP) since a same-team stack raises correlated bust risk,
    which cash mode should avoid by default.

    min_salary_fraction (as a fraction of salary_cap) is not optional padding -
    without it, this objective can rationally leave real cap unspent rather
    than pay for an expensive-but-volatile player, and a guaranteed-zero
    player has zero spread too, so nothing stops the solver from filling
    slots with worthless $2,500 scrubs instead. First caught anecdotally on a
    single live run (risk_aversion=1.0 with no salary floor left $17,200 of a
    $50,000 cap unused), then properly backtested against 55 real historical
    weeks (models/calibration.py::run_salary_left_backtest) rather than left
    on that one anecdote: unconstrained, this same objective averages
    $12,356 of real cap left unspent with a real -0.70 Pearson correlation
    between salary left and real actual score; applying this exact
    min_salary_fraction=0.95 default improves real actual score by a real,
    paired +33.99 points on average across those same 55 weeks (t=9.27,
    p<0.0001). The GPP ceiling objective (generate_lineups) was checked the
    same way and shows no such effect (correlation 0.03, avg $907 left
    unspent on its own) - it naturally spends the cap without needing this
    constraint, which is why generate_lineups has no min_salary of its own.

    lineup['total_points'] on the results is the real sum of proj_floor (what
    you'd actually expect), not the risk-adjusted objective value used
    internally to pick the roster.
    """
    engine = engine or get_engine()
    players, availability_excluded, injury_report_available, playing_time_debug_log = _load_player_pool(
        slate_id, "proj_floor", engine
    )
    if not players:
        raise ValueError(f"No players with a 'proj_floor' projection found for slate {slate_id}")

    for p in players:
        p["points"] = p["proj_floor"] - risk_aversion * (p["proj_ceiling"] - p["proj_floor"])

    lineups, exposure_report = build_lineups_from_pool(
        players,
        num_lineups=num_lineups,
        salary_cap=salary_cap,
        min_salary=min_salary_fraction * salary_cap if min_salary_fraction else None,
        locked_player_ids=locked_player_ids,
        excluded_player_ids=excluded_player_ids,
        max_players_per_team=max_players_per_team,
        min_uniques=min_uniques,
    )
    # Report the real floor sum, not the risk-adjusted solver objective.
    for lu in lineups:
        lu["total_points"] = sum(p["proj_floor"] for _, p in lu["roster"])

    availability_report = _build_availability_report(players, availability_excluded, injury_report_available)
    availability_report["playing_time_debug_log"] = playing_time_debug_log
    return lineups, exposure_report, availability_report


def lineup_player_ids(lineup):
    return [p["player_id"] for _, p in lineup["roster"]]
