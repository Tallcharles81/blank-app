import math
from collections import defaultdict

from sqlalchemy import text

from data.player_availability import get_availability_gate, resolve_slate_season_week
from data.player_crosswalk import resolve_dk_players_to_gsis
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
#   1. Call build_pre_lock_checklist(slate_id) to get the real game list
#      (every currently-eligible player, grouped by game) for the injury/
#      inactive/weather sweep below.
#   2. For each game (not each player - search once per matchup), search for:
#      - "<team> <team> inactive list <date>" (and again closer to actual
#        kickoff, since inactives are official ~90 min before)
#      - "<team> <team> injury report <date>" for any last-minute change
#      - "<city> weather forecast <date>" if the game is outdoors (this
#        codebase has no stadium/roof-type data - note that gap explicitly
#        in the finding if it matters, don't guess dome vs outdoor)
#   3. Once a specific lineup has been generated (not the whole eligible
#      pool - just the players actually being entered), call
#      build_lineup_role_checklist(slate_id, player_ids) and, for EVERY
#      player it returns with needs_check=True, search specifically for
#      current-week STARTER/ROLE status, not just health - "is he playing"
#      is not the same question as "is he confirmed as this week's starter/
#      lead back/lead WR/every-down role." Use real current sources
#      (official team site, ESPN, NFL.com, beat reporters covering that team
#      this week), not this codebase's own roster/injury data, which only
#      knows who's ON a roster or what this week's aggregate usage numbers
#      were, never who's actually starting or how a coach is using someone
#      THIS week. Price is never the trigger for this step - a player's
#      presence in the actual lineup is (see build_lineup_role_checklist's
#      docstring for why the old salary-proximity trigger was replaced).
#      Caught for real: Jameis Winston was rostered (passed the gate - on
#      the active roster, no injury flag) and given a real QB1-level
#      projection by this system for a real GPP entry, with nothing
#      checking whether he was actually that week's confirmed starter
#      versus a backup who happens to be on the 53-man roster. Rostered and
#      starting are different facts; the gate only knows the first one.
#   4. For every eligible player where a real, credible source contradicts
#      the gate's current verdict OR contradicts the role assumption this
#      system would otherwise make (e.g. a confirmed backup/committee
#      member getting this system's starter-level projection), call
#      record_finding(..., contradicts_gate=True) with the real source and
#      quote/claim - this is what get_contradictions() surfaces loudly. Use
#      source="starter_status" for this category specifically, distinct
#      from "injury_news"/"inactive_list"/"weather", so a contradiction's
#      cause is legible from the stored row alone.
#   5. Also record a finding (contradicts_gate=False) for games/players
#      checked with nothing new, so a later run of get_contradictions() next
#      to summarize_checks() shows real coverage, not just silence.
#   6. Report get_contradictions(slate_id) results to the user prominently -
#      a contradiction found this close to lock is exactly the scenario this
#      check exists for.
# ---------------------------------------------------------------------------

# Thresholds for build_lineup_role_checklist's real, data-driven "does this
# player have a clear, current role" check - see that function's docstring
# for why this replaced a salary-proximity proxy. These look at actual
# recent player_weekly_stats usage, not price or position-tier within a team.
ROLE_CHECK_HISTORY_WEEKS = 4
MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE = 2
LOW_SNAP_PCT_THRESHOLD = 0.45
INCONSISTENT_SNAP_PCT_STDEV_THRESHOLD = 0.15
LOW_TARGET_SHARE_THRESHOLD = 0.15
LOW_CARRIES_PER_GAME_THRESHOLD = 8


def _load_recent_usage(gsis_id, engine, max_weeks=ROLE_CHECK_HISTORY_WEEKS):
    query = text(
        """
        SELECT season, week, snap_pct, target_share, carries, injury_status
        FROM player_weekly_stats
        WHERE player_id = :gsis_id
        ORDER BY season DESC, week DESC
        LIMIT :max_weeks
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"gsis_id": gsis_id, "max_weeks": max_weeks}).mappings().fetchall()
    return [dict(row) for row in rows]


def _role_check_reason(position, games):
    """Real, data-driven verdict on whether a player's recent usage shows a
    clear, current role - replacing the old "priced close to the team's top
    salary at that position" proxy, which missed cheap, deep-bench-priced
    players entirely (Kenny Gainwell at $5,100 - an uncertain committee back
    on a new team - and Jalen McMillan at $4,400 - returning from a real
    recent injury - both made it into a real lineup unflagged, since neither
    was priced near their team's top RB/WR salary). The right question is
    "does this player's own recent usage show a clear, current role,"
    independent of price entirely.

    `games` is recent player_weekly_stats rows for this player, most-recent
    first. Returns (needs_check: bool, reason: str | None).
    """
    if not games:
        return True, "no recent usage data on record - can't confirm a current role at all"

    recent_injury = next((g["injury_status"] for g in games if g["injury_status"]), None)
    if recent_injury:
        return True, f"recent injury designation on record ({recent_injury}) - confirm current practice/game status"

    if len(games) < MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE:
        return True, f"only {len(games)} recent game(s) of usage data on record - not enough to confirm a stable role"

    snap_pcts = [float(g["snap_pct"]) for g in games if g["snap_pct"] is not None]
    if not snap_pcts:
        return True, "no snap-share data recorded in recent games - can't confirm role from usage"

    mean_snap = sum(snap_pcts) / len(snap_pcts)
    if mean_snap < LOW_SNAP_PCT_THRESHOLD:
        return True, (
            f"low recent snap share (avg {mean_snap:.0%} over last {len(snap_pcts)} game(s)) - "
            "confirm current role, not just health"
        )
    if len(snap_pcts) >= 2:
        stdev = math.sqrt(sum((s - mean_snap) ** 2 for s in snap_pcts) / (len(snap_pcts) - 1))
        if stdev > INCONSISTENT_SNAP_PCT_STDEV_THRESHOLD:
            return True, (
                f"inconsistent recent snap share (week-to-week swings of {stdev:.0%}) - "
                "confirm this week's role isn't a committee split"
            )

    if position in ("WR", "TE"):
        target_shares = [float(g["target_share"]) for g in games if g["target_share"] is not None]
        if target_shares:
            mean_target = sum(target_shares) / len(target_shares)
            if mean_target < LOW_TARGET_SHARE_THRESHOLD:
                return True, (
                    f"low recent target share (avg {mean_target:.0%}) - confirm current role, not just health"
                )

    if position == "RB":
        carries = [g["carries"] for g in games if g["carries"] is not None]
        if carries:
            mean_carries = sum(carries) / len(carries)
            if mean_carries < LOW_CARRIES_PER_GAME_THRESHOLD:
                return True, (
                    f"low recent carries (avg {mean_carries:.1f}/game) - confirm current role, not just health"
                )

    return False, None


def build_lineup_role_checklist(slate_id, player_ids, engine=None):
    """The real, per-player role-confirmation checklist for one SPECIFIC
    generated lineup (pass the exact DK player_ids in it) - not the whole
    eligible pool. Runs unconditionally on every player given; price and
    position-tier within a team are never the trigger.

    This replaces an earlier design that only checked RB/WR priced within
    85% of their team's top salary at that position, on the theory that
    price was a proxy for "who this system currently treats as the lead."
    That proxy had a real, structural blind spot: a cheap, deep-bench-priced
    player can still end up in an actual lineup (a GPP leverage/value pick,
    a committee back priced low because the market doesn't trust the role
    either) without ever being priced near the top of their position - and
    those are exactly the players whose real, current role is least certain.
    The right trigger is simpler and has no such gap: if a player is
    actually in the lineup being entered, their role gets checked, full stop.

    QB is still an unconditional check regardless of usage data - only one
    player takes meaningful starter snaps most weeks, and this codebase's
    own data has no way to tell which one that is, only who's on the
    roster (see the Jameis Winston case in the module docstring above).
    DST is skipped - a team defense doesn't have an individual "role."
    Every other position is checked against real recent snap_pct/
    target_share/carries/injury_status from player_weekly_stats (see
    _role_check_reason) rather than any price signal.

    Returns a list of dicts: player_id, name, position, team, needs_check,
    reason.
    """
    engine = engine or get_engine()

    with engine.connect() as conn:
        players = conn.execute(
            text(
                "SELECT player_id, name, position, team FROM slate_player_pool "
                "WHERE slate_id = :slate_id AND player_id = ANY(:player_ids)"
            ),
            {"slate_id": slate_id, "player_ids": list(player_ids)},
        ).mappings().fetchall()
    players = [dict(row) for row in players]

    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(players, engine)

    results = []
    for p in players:
        if p["position"] == "DST":
            results.append({**p, "needs_check": False, "reason": None})
            continue
        if p["position"] == "QB":
            results.append(
                {
                    **p,
                    "needs_check": True,
                    "reason": "QB - confirm which player is this week's actual starter, not just who's rostered",
                }
            )
            continue

        gsis_id = gsis_by_dk_id.get(p["player_id"])
        if gsis_id is None:
            results.append(
                {
                    **p,
                    "needs_check": True,
                    "reason": "no crosswalk match to real usage history - can't confirm role from data at all",
                }
            )
            continue

        games = _load_recent_usage(gsis_id, engine)
        needs_check, reason = _role_check_reason(p["position"], games)
        results.append({**p, "needs_check": needs_check, "reason": reason})

    return results


def build_pre_lock_checklist(slate_id, engine=None):
    """The real, structured list of what a pre-lock check needs to verify:
    every currently-eligible (post-gate) player in `slate_id`, grouped by
    real game matchup, with the gate's own current verdict attached, so an
    agent doing the research step knows what it's trying to confirm or
    contradict for each one.

    This covers the injury/inactive/weather sweep only. The separate
    starter/current-role question ("is this player actually this week's
    starter/lead back/lead WR, not just healthy and rostered") is checked
    per actual generated lineup, not across the whole eligible pool - see
    build_lineup_role_checklist below, and step 3 of the module-level
    PROCEDURE comment above for why it's scoped that way.

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

    games = defaultdict(lambda: {"game_time": None, "players": []})
    for p in eligible:
        key = (p["team"], p["opponent"], p["game_time"])
        games[key]["game_time"] = p["game_time"]
        gate_status = flagged.get(p["player_id"], "CLEAR")
        games[key]["players"].append(
            {
                "player_id": p["player_id"],
                "name": p["name"],
                "position": p["position"],
                "team": p["team"],
                "salary": p["salary"],
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
