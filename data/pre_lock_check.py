from collections import defaultdict

from sqlalchemy import text

from data.player_availability import get_availability_gate, resolve_slate_season_week
from db.migrate import get_engine

# ---------------------------------------------------------------------------
# What this module is, and what it is NOT.
#
# This is the second, independent check for the hours-before-lock window,
# separate from data/player_availability.py's roster/injury-report gate. That
# gate is real and correct, but it's only as fresh as nflverse's last fetch -
# the A.J. Brown and Brandon Aiyuk incidents were both cases where the
# player's true status was knowable, just not yet reflected in our data feed.
# In the hours right before lock, real news (a surprise inactive, a scratch,
# a game-affecting weather report) can move faster than any data feed
# updates. This module does NOT replace the gate - it cross-checks it.
#
# What this module CANNOT do on its own: search the web. There is no web-
# search capability inside this codebase's Python runtime - build_pre_lock_
# checklist() only prepares the real, structured list of what needs
# checking (every currently-eligible player, grouped by game, with the
# gate's own current verdict attached) so an agent with real search access
# can work through it efficiently (per game, not per player) and call
# record_finding() for anything it turns up. The actual research step is a
# human- or agent-driven procedure, not an automated function - see
# scripts/run_pre_lock_check.md (if this has been run before) or the
# procedure documented in the module docstring below for what that step
# should do.
#
# PROCEDURE for the agent step, each time this is run:
#   1. Call build_pre_lock_checklist(slate_id) to get the real game list.
#      Every player in it now carries needs_starter_check/starter_check_reason
#      (see _needs_starter_check below) - treat those as a required part of
#      the same pass, not an optional extra.
#   2. For each game (not each player - search once per matchup), search for:
#      - "<team> <team> inactive list <date>" (and again closer to actual
#        kickoff, since inactives are official ~90 min before)
#      - "<team> <team> injury report <date>" for any last-minute change
#      - "<city> weather forecast <date>" if the game is outdoors (this
#        codebase has no stadium/roof-type data - note that gap explicitly
#        in the finding if it matters, don't guess dome vs outdoor)
#   3. For every player flagged needs_starter_check, ALSO search specifically
#      for current-week STARTER status, not just health - "is he playing"
#      is not the same question as "is he confirmed as this week's starter/
#      lead back/lead WR." Use real current sources (official team site,
#      ESPN, NFL.com, beat reporters covering that team this week), not this
#      codebase's own roster/injury data, which only knows who's ON a roster,
#      never who's actually starting. Caught for real: Jameis Winston was
#      rostered (passed the gate - on the active roster, no injury flag) and
#      given a real QB1-level projection by this system for a real GPP entry,
#      with nothing anywhere checking whether he was actually that week's
#      confirmed starter versus a backup who happens to be on the 53-man
#      roster. Rostered and starting are different facts; the gate only
#      knows the first one.
#   4. For every eligible player where a real, credible source contradicts
#      the gate's current verdict OR contradicts the starter assumption this
#      system would otherwise make (e.g. a confirmed backup getting this
#      system's starter-level projection), call record_finding(...,
#      contradicts_gate=True) with the real source and quote/claim - this is
#      what get_contradictions() surfaces loudly. Use source="starter_status"
#      for this category specifically, distinct from "injury_news"/
#      "inactive_list"/"weather", so a contradiction's cause is legible from
#      the stored row alone.
#   5. Also record a finding (contradicts_gate=False) for games/players
#      checked with nothing new, so a later run of get_contradictions() next
#      to summarize_checks() shows real coverage, not just silence.
#   6. Report get_contradictions(slate_id) results to the user prominently -
#      a contradiction found this close to lock is exactly the scenario this
#      check exists for.
# ---------------------------------------------------------------------------

# A team fielding a real committee (not a clean lead back/receiver) means
# more than one player at that position is worth a starter-role check, not
# just whoever happens to be priced highest. 0.85 is a deliberately loose
# band - catching one extra player worth double-checking is cheap; missing a
# real committee split isn't.
COMMITTEE_SALARY_TOLERANCE = 0.85
STARTER_CHECK_POSITIONS = {"QB", "RB", "WR"}


def _needs_starter_check(player, team_position_salaries):
    # QB: always check - only one player takes meaningful starter snaps most
    # weeks, and this codebase's own data has no way to tell which one
    # that is, only who's on the roster (see the Jameis Winston case in the
    # module docstring above).
    if player["position"] == "QB":
        return True, "QB - confirm which player is this week's actual starter, not just who's rostered"

    # RB/WR: check whoever is priced within COMMITTEE_SALARY_TOLERANCE of the
    # top salary at that team+position in the current pool - this system's
    # own market-driven proxy for "who it's currently treating as the lead."
    if player["position"] in ("RB", "WR"):
        top_salary = max(team_position_salaries)
        if player["salary"] >= COMMITTEE_SALARY_TOLERANCE * top_salary:
            reason = (
                f"top-{player['position']} salary tier at {player['team']} "
                f"(${player['salary']} vs team-high ${top_salary}) - confirm current role/snap "
                "share, not just health"
            )
            return True, reason

    return False, None


def build_pre_lock_checklist(slate_id, engine=None):
    """The real, structured list of what a pre-lock check needs to verify:
    every currently-eligible (post-gate) player in `slate_id`, grouped by
    real game matchup, with the gate's own current verdict AND a starter-
    status check flag attached, so an agent doing the research step knows
    what it's trying to confirm or contradict for each one - "is this
    player healthy" (the gate) and "is this player actually this week's
    starter" (needs_starter_check) are different questions, and this
    codebase's own data can only answer the first one.

    Returns {(team, opponent, game_time): {"players": [...], "game_time":
    ...}} - grouped by game so the research step can search once per
    matchup instead of once per player.
    """
    engine = engine or get_engine()

    with engine.connect() as conn:
        all_players = conn.execute(
            text(
                "SELECT player_id, name, position, team, opponent, game_time, salary "
                "FROM slate_player_pool WHERE slate_id = :slate_id"
            ),
            {"slate_id": slate_id},
        ).mappings().fetchall()
    all_players = [dict(row) for row in all_players]
    if not all_players:
        raise ValueError(f"No players found in slate_player_pool for slate {slate_id}")

    season_week = resolve_slate_season_week(slate_id, engine)
    if season_week is None:
        raise RuntimeError(f"Could not determine (season, week) for slate {slate_id}")
    season, week = season_week
    excluded, flagged, injury_report_available = get_availability_gate(all_players, season, week, engine)

    eligible = [p for p in all_players if p["player_id"] not in excluded]

    # Real team+position salary groups, over the WHOLE eligible pool (not
    # per-game) - the natural unit for "who's competing for this role" is a
    # team's own depth chart, not the game grouping used below.
    salaries_by_team_position = defaultdict(list)
    for p in eligible:
        if p["position"] in STARTER_CHECK_POSITIONS:
            salaries_by_team_position[(p["team"], p["position"])].append(p["salary"])

    games = defaultdict(lambda: {"game_time": None, "players": []})
    for p in eligible:
        key = (p["team"], p["opponent"], p["game_time"])
        games[key]["game_time"] = p["game_time"]
        gate_status = flagged.get(p["player_id"], "CLEAR")
        team_position_salaries = salaries_by_team_position.get((p["team"], p["position"]), [])
        needs_starter_check, starter_check_reason = _needs_starter_check(p, team_position_salaries)
        games[key]["players"].append(
            {
                "player_id": p["player_id"],
                "name": p["name"],
                "position": p["position"],
                "team": p["team"],
                "salary": p["salary"],
                "gate_status": gate_status,
                "needs_starter_check": needs_starter_check,
                "starter_check_reason": starter_check_reason,
            }
        )

    return dict(games), {"season": season, "week": week, "injury_report_available": injury_report_available}


def record_finding(slate_id, player_id, source, finding, gate_status_at_check, contradicts_gate, engine=None):
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pre_lock_checks
                    (slate_id, player_id, source, finding, gate_status_at_check, contradicts_gate)
                VALUES (:slate_id, :player_id, :source, :finding, :gate_status_at_check, :contradicts_gate)
                """
            ),
            {
                "slate_id": slate_id,
                "player_id": player_id,
                "source": source,
                "finding": finding,
                "gate_status_at_check": gate_status_at_check,
                "contradicts_gate": contradicts_gate,
            },
        )


def get_contradictions(slate_id, engine=None):
    """Every recorded finding that contradicted the gate's status at check
    time - what a caller should surface loudly. Ordered most recent first
    since a later, more-current finding matters more than an earlier one for
    the same player.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT plc.player_id, sp.name, plc.source, plc.finding,
                       plc.gate_status_at_check, plc.checked_at
                FROM pre_lock_checks plc
                JOIN slate_player_pool sp ON sp.slate_id = plc.slate_id AND sp.player_id = plc.player_id
                WHERE plc.slate_id = :slate_id AND plc.contradicts_gate = TRUE
                ORDER BY plc.checked_at DESC
                """
            ),
            {"slate_id": slate_id},
        ).mappings().fetchall()
    return [dict(row) for row in rows]


def summarize_checks(slate_id, engine=None):
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT count(*) AS total_findings,
                       count(*) FILTER (WHERE contradicts_gate) AS contradictions,
                       count(DISTINCT player_id) AS players_checked,
                       max(checked_at) AS last_checked_at
                FROM pre_lock_checks WHERE slate_id = :slate_id
                """
            ),
            {"slate_id": slate_id},
        ).mappings().first()
    return dict(row) if row else None
