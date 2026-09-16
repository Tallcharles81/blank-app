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
#   2. For each game (not each player - search once per matchup), search for:
#      - "<team> <team> inactive list <date>" (and again closer to actual
#        kickoff, since inactives are official ~90 min before)
#      - "<team> <team> injury report <date>" for any last-minute change
#      - "<city> weather forecast <date>" if the game is outdoors (this
#        codebase has no stadium/roof-type data - note that gap explicitly
#        in the finding if it matters, don't guess dome vs outdoor)
#   3. For every eligible player where a real, credible source contradicts
#      the gate's current verdict (e.g. reported "Out" or a surprise
#      inactive, when the gate currently has them CLEAR or FLAGGED-only),
#      call record_finding(..., contradicts_gate=True) with the real source
#      and quote/claim - this is what get_contradictions() surfaces loudly.
#   4. Also record a finding (contradicts_gate=False) for games/players
#      checked with nothing new, so a later run of get_contradictions() next
#      to summarize_checks() shows real coverage, not just silence.
#   5. Report get_contradictions(slate_id) results to the user prominently -
#      a contradiction found this close to lock is exactly the scenario this
#      check exists for.
# ---------------------------------------------------------------------------


def build_pre_lock_checklist(slate_id, engine=None):
    """The real, structured list of what a pre-lock check needs to verify:
    every currently-eligible (post-gate) player in `slate_id`, grouped by
    real game matchup, with the gate's own current verdict attached so an
    agent doing the research step knows what it's trying to confirm or
    contradict for each one.

    Returns {(team, opponent, game_time): {"players": [...], "game_time":
    ...}} - grouped by game so the research step can search once per
    matchup instead of once per player.
    """
    engine = engine or get_engine()

    with engine.connect() as conn:
        all_players = conn.execute(
            text(
                "SELECT player_id, name, position, team, opponent, game_time "
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

    games = defaultdict(lambda: {"game_time": None, "players": []})
    for p in all_players:
        if p["player_id"] in excluded:
            continue  # already excluded - not what this check needs to re-verify
        key = (p["team"], p["opponent"], p["game_time"])
        games[key]["game_time"] = p["game_time"]
        gate_status = flagged.get(p["player_id"], "CLEAR")
        games[key]["players"].append(
            {
                "player_id": p["player_id"],
                "name": p["name"],
                "position": p["position"],
                "team": p["team"],
                "gate_status": gate_status,
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
