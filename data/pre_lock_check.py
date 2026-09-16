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
# hard_role_exclusions() is the one piece of this module that IS a hard gate,
# not an advisory cross-check: it's wired into models/optimizer.py's
# _load_player_pool() right alongside data/player_availability.py's
# roster/IR gate, so a player with a genuine structural role concern (real,
# low/inconsistent recent volume) never reaches the solver at all - the same
# standard as roster absence. This closes a real gap: before this existed,
# a bad role finding (Jameis Winston getting a starter-level projection
# despite being a real backup) only got excluded because a human manually
# passed excluded_player_ids on that one call, and reproduced itself in a
# fresh run the moment that manual step wasn't repeated. Everything else in
# this module - the injury/inactive/weather sweep and the lineup-scoped
# QB/MEDIUM/LOW findings - stays advisory, for the reasons below.
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
#      build_lineup_role_checklist(slate_id, player_ids). Every player it
#      returns with needs_check=True also carries a severity - HIGH (a real
#      structural volume concern, or an always-on QB check), MEDIUM (an
#      injury designation on an otherwise clearly-established player), or
#      LOW (thin data only - a rookie/recent trade, not a known problem).
#      Treat HIGH as the required short list to research every time; MEDIUM
#      is worth a same-day check close to lock; LOW is informational and
#      doesn't need the same urgency. For EVERY flagged player, search
#      specifically for current-week STARTER/ROLE status, not just health -
#      "is he playing"
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

# Severity tiers - a flat "needs_check" told you something was worth a look,
# but not whether it was a real problem or a thin-data footnote. Modeled on
# models/matchups.py's HIGH/MEDIUM/LOW confidence scale for the same reason:
# a flag only earns attention if its label tells you how much to give it.
#   HIGH   - a real, structural role concern: this player's own recent usage
#            (volume, consistency) doesn't look like a startable role, full
#            stop, independent of health. This is Kenny Gainwell's case
#            (6.5 carries/game) and the always-on QB check (this codebase
#            cannot tell rostered from starting at all - see the Jameis
#            Winston incident in the module docstring).
#   MEDIUM - an active injury designation (Questionable/Doubtful) on a
#            player whose own recent volume otherwise looks like a clear,
#            established role. Worth a same-day check close to lock, but
#            not, by itself, a reason to think the role has changed.
#   LOW    - thin data only: too few recent games, no snap data recorded, or
#            no crosswalk match at all - typically a rookie, a recent trade,
#            or a data gap. The honest read here is "we don't have much
#            signal," not "something is actually wrong."
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"


def _load_recent_usage_batch(gsis_ids, engine, max_weeks=ROLE_CHECK_HISTORY_WEEKS, before=None):
    """Recent player_weekly_stats usage rows for every id in gsis_ids, in one
    windowed query - the same pattern as models/projections.py's
    _load_recent_stats, and for the same reason: a per-player round trip is
    fine for a 9-player lineup checklist, but hard_role_exclusions below has
    to evaluate this across the WHOLE eligible pool (hundreds of players)
    before the optimizer ever runs, where an N+1 query pattern would be a
    real cost, not just a style nit.

    `before`, an optional (season, week) cutoff, exists for the same reason
    it does in _load_recent_stats: models/calibration.py's hard-exclude
    backtest needs to know what this function would have returned USING
    ONLY DATA AVAILABLE STRICTLY BEFORE the week under test, or it would be
    testing the rule against outcomes it could already see - lookahead bias
    that would make the backtest meaningless.

    Returns {gsis_id: [games most-recent-first]}.
    """
    cutoff_sql = ""
    params = {"gsis_ids": list(gsis_ids), "max_weeks": max_weeks}
    if before is not None:
        cutoff_sql = "AND (season < :before_season OR (season = :before_season AND week < :before_week))"
        params["before_season"], params["before_week"] = before

    query = text(
        f"""
        SELECT player_id, season, week, position, snap_pct, target_share, carries, injury_status, recency_rank FROM (
            SELECT player_id, season, week, position, snap_pct, target_share, carries, injury_status,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id ORDER BY season DESC, week DESC
                   ) AS recency_rank
            FROM player_weekly_stats
            WHERE player_id = ANY(:gsis_ids) {cutoff_sql}
        ) ranked
        WHERE recency_rank <= :max_weeks
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().fetchall()

    by_player = defaultdict(list)
    for row in rows:
        by_player[row["player_id"]].append(dict(row))
    for games in by_player.values():
        games.sort(key=lambda g: g["recency_rank"])
    return dict(by_player)


def _role_check_severity(position, games):
    """Real, data-driven verdict on whether a player's recent usage shows a
    clear, current role - replacing the old "priced close to the team's top
    salary at that position" proxy, which missed cheap, deep-bench-priced
    players entirely (Kenny Gainwell at $5,100 - an uncertain committee back
    on a new team - and Jalen McMillan at $4,400 - returning from a real
    recent injury - both made it into a real lineup unflagged, since neither
    was priced near their team's top RB/WR salary). The right question is
    "does this player's own recent usage show a clear, current role,"
    independent of price entirely.

    A structural volume problem (HIGH) is checked and reported ahead of a
    plain injury designation (MEDIUM): a player whose own recent carries/
    snaps/targets already look thin has a real role problem regardless of
    the injury report, so that's the more important fact to lead with -
    the injury note still gets folded into the HIGH reason when both are
    present, rather than silently dropped.

    `games` is recent player_weekly_stats rows for this player, most-recent
    first. Returns (needs_check: bool, severity: str | None, reason: str | None).
    """
    recent_injury = next((g["injury_status"] for g in games if g["injury_status"]), None)

    if not games:
        return True, SEVERITY_LOW, (
            "no recent usage data on record - likely a rookie/first appearance or recently "
            "traded; thin data, not a known problem"
        )

    if len(games) < MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE:
        reason = (
            f"only {len(games)} recent game(s) of usage data on record - likely a rookie/first "
            "appearance or recently traded; thin data, not a known problem"
        )
        if recent_injury:
            reason += f"; also carries a recent {recent_injury} designation on that limited sample"
            return True, SEVERITY_MEDIUM, reason
        return True, SEVERITY_LOW, reason

    snap_pcts = [float(g["snap_pct"]) for g in games if g["snap_pct"] is not None]
    if not snap_pcts:
        reason = "no snap-share data recorded in recent games - can't confirm role from usage; thin data, not a known problem"
        if recent_injury:
            reason += f"; also carries a recent {recent_injury} designation"
            return True, SEVERITY_MEDIUM, reason
        return True, SEVERITY_LOW, reason

    structural_reason = None
    mean_snap = sum(snap_pcts) / len(snap_pcts)
    if mean_snap < LOW_SNAP_PCT_THRESHOLD:
        structural_reason = f"low recent snap share (avg {mean_snap:.0%} over last {len(snap_pcts)} game(s))"
    elif len(snap_pcts) >= 2:
        stdev = math.sqrt(sum((s - mean_snap) ** 2 for s in snap_pcts) / (len(snap_pcts) - 1))
        if stdev > INCONSISTENT_SNAP_PCT_STDEV_THRESHOLD:
            structural_reason = (
                f"inconsistent recent snap share (week-to-week swings of {stdev:.0%}) - possible committee split"
            )

    if structural_reason is None and position in ("WR", "TE"):
        target_shares = [float(g["target_share"]) for g in games if g["target_share"] is not None]
        if target_shares:
            mean_target = sum(target_shares) / len(target_shares)
            if mean_target < LOW_TARGET_SHARE_THRESHOLD:
                structural_reason = f"low recent target share (avg {mean_target:.0%})"

    if structural_reason is None and position == "RB":
        carries = [g["carries"] for g in games if g["carries"] is not None]
        if carries:
            mean_carries = sum(carries) / len(carries)
            if mean_carries < LOW_CARRIES_PER_GAME_THRESHOLD:
                structural_reason = f"low recent carries (avg {mean_carries:.1f}/game)"

    if structural_reason:
        reason = f"{structural_reason} - real, structural role concern; confirm current role, not just health"
        if recent_injury:
            reason += f" (also carries a recent {recent_injury} designation)"
        return True, SEVERITY_HIGH, reason

    if recent_injury:
        return True, SEVERITY_MEDIUM, (
            f"recent injury designation on record ({recent_injury}) on an otherwise established, "
            "high-usage player - confirm current practice/game status close to lock; not inherently "
            "a reason to bench"
        )

    return False, None, None


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

    QB is still an unconditional HIGH-severity check regardless of usage
    data - only one player takes meaningful starter snaps most weeks, and
    this codebase's own data has no way to tell which one that is, only
    who's on the roster (see the Jameis Winston case in the module
    docstring above) - that's the same class of real, structural role
    concern as a committee back with thin volume, not a lesser one. DST is
    skipped - a team defense doesn't have an individual "role." Every other
    position is checked against real recent snap_pct/target_share/carries/
    injury_status from player_weekly_stats (see _role_check_severity)
    rather than any price signal, and comes back tagged HIGH/MEDIUM/LOW so
    a caller can pull out the short list of real problems (HIGH) instead of
    manually sorting every flag - see _role_check_severity's docstring for
    what each tier means.

    Returns a list of dicts: player_id, name, position, team, needs_check,
    severity, reason.

    Position-based branching below (QB unconditional, DST skipped, else
    _role_check_severity) can't just trust p["position"] - true on Classic,
    but a Showdown row is labeled "CPT"/"FLEX" regardless of the real
    player's position (data/player_crosswalk.py's SHOWDOWN_PSEUDO_
    POSITIONS), which silently made this function wrong for Showdown: a
    real QB would fall into the RB/WR/TE structural check instead of the
    unconditional QB flag, and a real DST would never get skipped. Same
    fix as hard_role_exclusions: resolve real position from the player's
    own history rather than the DK row when it's a pseudo-position.
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
    games_by_gsis = _load_recent_usage_batch(
        [gid for gid in gsis_by_dk_id.values() if gid is not None], engine
    )

    results = []
    for p in players:
        gsis_id = gsis_by_dk_id.get(p["player_id"])
        games = games_by_gsis.get(gsis_id, []) if gsis_id is not None else []
        real_position = p["position"] if p["position"] not in ("CPT", "FLEX") else (
            games[0]["position"] if games else None
        )

        if real_position == "DST":
            results.append({**p, "needs_check": False, "severity": None, "reason": None})
            continue
        if real_position == "QB":
            results.append(
                {
                    **p,
                    "needs_check": True,
                    "severity": SEVERITY_HIGH,
                    "reason": "QB - confirm which player is this week's actual starter, not just who's rostered",
                }
            )
            continue

        if gsis_id is None or real_position is None:
            results.append(
                {
                    **p,
                    "needs_check": True,
                    "severity": SEVERITY_LOW,
                    "reason": "no crosswalk match to real usage history - can't confirm role from data at all",
                }
            )
            continue

        needs_check, severity, reason = _role_check_severity(real_position, games)
        results.append({**p, "needs_check": needs_check, "severity": severity, "reason": reason})

    return results


# hard_role_exclusions below deliberately does NOT reuse _role_check_severity's
# thresholds. First attempt did, and it was a real, serious mistake caught by
# testing against the actual live pool before shipping it (not caught in
# review - only running it for real surfaced this): applied pool-wide, the
# advisory thresholds (avg snap share < 45%, or week-to-week snap swings >
# 15%) hard-excluded 317 of 1,066 real players, including Jonathan Taylor,
# CeeDee Lamb, Malik Nabers, and Jaylen Waddle - every one of them caught by
# the "inconsistent snap share" check reacting to one ordinary rest/blowout
# game (CeeDee Lamb: 0.81/0.45/0.87/0.86 across his 4 most recent - a single
# meaningless week-18 rest game once seeding was locked - which is completely
# normal variance, not evidence of "no role," but reads as a 20% stdev to a
# 15%-threshold check). A threshold tuned to be a useful "worth a look" nudge
# for a human reviewing 9 players is not automatically safe to silently and
# permanently exclude a player from an entire live pool with zero visibility
# - those are different bars, and conflating them here would have been a
# worse bug than the one this function exists to fix.
#
# The bar used here instead, for RB/WR/TE: hard-exclude only if snap_pct
# across EVERY one of a player's last MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE+
# recorded games (not the average - a max, so one real usage spike anywhere
# in the window rescues a player) stayed below HARD_EXCLUDE_MAX_SNAP_PCT.
# Verified against the live Thu-Mon pool: at 0.25, this catches 57 of 611
# RB/WR/TE candidates, every single one already priced at or near DK's
# roster-floor salary ($2,500-$4,800, true replacement-level deep bench) -
# zero false positives among real difference-makers, and it correctly leaves
# Kenny Gainwell ($5,100, snap_pct 0.46-0.68) and Jalen McMillan ($4,400)
# unexcluded here, since their own data doesn't clear this much higher bar -
# those stay correctly caught by the advisory, human-reviewed severity check
# instead (see build_lineup_role_checklist), the right layer for a genuinely
# borderline case, not a hard, silent block.
#
# BACKTESTED since (models/calibration.py::run_hard_exclude_backtest, 53 real
# historical weeks, no-lookahead as-of evaluation): RB/WR/TE is strongly
# validated as a real production-floor signal, not just a plausible-looking
# live-pool spot check. 979 real RB/WR/TE player-weeks would have been
# hard-excluded; they averaged 2.86 (RB) / 1.57 (WR) / 1.34 (TE) real PPR
# points, versus 9.47 for everyone NOT excluded (t=-37.2, p<0.0001) - and the
# MISS rate (a would-be-excluded player-week that nonetheless scored above
# 8.0 real PPR points - the real cost of this gate being wrong) was low:
# 9.4% (RB) / 4.8% (WR) / 4.6% (TE). This is what makes a hard, silent gate
# defensible - see MIN_SNAP_PCT_FOR_BENCHED below for the QB result, which
# is NOT this clean.
HARD_EXCLUDE_MAX_SNAP_PCT = 0.25

# QB needs a DIFFERENT signal entirely, not this same max-snap-pct rule: a
# real backup who starts when called upon plays close to 100% of snaps in
# the games he DOES start, same as any real QB1 - so "max recent snap_pct"
# can't tell them apart, and this is exactly why Jameis Winston (a real
# backup, given a real QB1-level projection by this system) still passes
# HARD_EXCLUDE_MAX_SNAP_PCT: his 4 most recent recorded games are
# [0.03, 1.00, 1.00, 0.77] - a high max, just like a real starter.
#
# What DOES separate a real starter from an intermittent backup in this
# data: a real starter's low games are rest/blowout dips (Lamb: 0.81/0.45/
# 0.87/0.86 - the low one is still 45%, a partial share, never near zero),
# while a backup who only plays when starters are hurt/benched swings
# between essentially NOT PLAYING (near-0%, a week he was inactive/DNP) and
# FULLY PLAYING (near-100%, the week he started) - a real, qualitatively
# different pattern, not just "more variance." Verified against the live
# Thu-Mon pool at MIN_SNAP_PCT_FOR_BENCHED=0.15/MAX_SNAP_PCT_FOR_FULL_GAME=
# 0.85 (both a game below 0.15 AND a game at/above 0.85 required): catches
# 21 of the live pool's QBs, every one of them a real backup/journeyman
# ($4,000-$5,400, at or near the QB salary floor - Sam Darnold, Cam Ward,
# Jameis Winston, Joe Flacco, Mason Rudolph, etc.), zero of the slate's real
# clear starters (Trevor Lawrence, Josh Allen, Jaxson Dart - none matched).
# Checked one skill-position case this pattern would ALSO catch if applied
# there before restricting it to QB: DeVonta Smith, a real, clearly-
# established Eagles WR1, shows [0.95, 0.14, 0.85, 0.86] - one single-game
# dip to 14% (almost certainly an in-game injury or early exit, not "no
# role") would have been a false positive. QB is different because a real
# starter essentially never sits below ~15% snaps in a game he isn't hurt or
# benched for (there's no "partial-share" version of starting QB the way a
# WR2 can play a reduced route share) - that clean binary doesn't hold at
# other positions, so this pattern check is QB-only.
#
# BACKTESTED since (models/calibration.py::run_hard_exclude_backtest, same
# 53-week no-lookahead run as HARD_EXCLUDE_MAX_SNAP_PCT above) - and this
# one is NOT clean the same way RB/WR/TE is. 141 real QB player-weeks would
# have been hard-excluded; they averaged 9.06 real PPR points when they
# happened (not "on average were near zero" - a real, usable-if-not-great
# QB score), and the miss rate (scored above 8.0 real PPR points anyway)
# was 46.8% - roughly a coin flip, nothing like RB/WR/TE's 4-9% miss rates.
# This isn't actually a contradiction of what this check is FOR, though:
# unlike the RB/WR/TE rule, which claims "this player has no real role,"
# this rule claims "this codebase can't confirm who's starting" - an
# intermittent backup who DOES get the start in a given week can
# legitimately put up a normal QB score, and that's expected, not a sign
# the pattern-detection itself is wrong. But a hard, silent gate is only
# defensible when it's very rarely wrong in the case that matters (production),
# and a coin-flip miss rate on real historical data is a genuinely weaker
# case than RB/WR/TE's - flagged here plainly rather than left looking
# equally validated. Not changed to advisory-only based on this result
# without that being a separate, explicit decision.
MIN_SNAP_PCT_FOR_BENCHED = 0.15
MAX_SNAP_PCT_FOR_FULL_GAME = 0.85


def hard_role_exclusions(players, engine=None):
    """Players with NO evidence of a real role in any recent game on record -
    hard-excluded from the eligible pool before the optimizer ever runs, the
    same way data/player_availability.py's roster-absence and IR checks are
    already hard gates rather than advisory flags.

    This is deliberately much narrower than "everything build_lineup_role_
    checklist would flag HIGH for a given lineup" - see the module-level
    comments above HARD_EXCLUDE_MAX_SNAP_PCT and MIN_SNAP_PCT_FOR_BENCHED for
    why a hard, silent, permanent exclusion needs a far more conservative,
    position-aware bar than an advisory flag a human is about to review
    anyway. It's not "always exclude QB" (that would leave no QB to roster
    at all) - QB gets its own real, data-driven pattern check instead of the
    RB/WR/TE volume check, precisely because Jameis Winston reappearing in
    lineup after lineup showed that raw snap-share volume alone can't tell a
    real starter from an intermittent backup at QB. It also skips every
    thin-data LOW/MEDIUM case (a rookie or recent trade isn't evidence of no
    role, just of not much data yet).

    This closes the real gap that let Jameis Winston reappear at 60%
    exposure in a freshly generated GPP set after being excluded once by
    hand - a genuinely no-role/intermittent-role player (this codebase's own
    data shows it, position-appropriately) now blocks structurally and
    permanently, without depending on a human remembering to re-apply a
    one-off exclusion every run. A merely borderline/committee case
    (Gainwell, McMillan) is intentionally left to the advisory lineup-scoped
    check instead, where a human reviews it before trusting the lineup -
    the honest trade-off of a hard gate that can't afford false positives.

    `players` is any list of dicts with player_id/name/position/team (the
    same shape models/optimizer.py's player pool uses) - not necessarily a
    whole slate; callers pass whatever pool they're about to hand to the
    solver. Returns {player_id: reason}.

    Position-aware branching below (QB pattern vs RB/WR/TE volume, and
    skipping DST entirely) can't just trust `p["position"]` - true on a
    Classic slate, but a Showdown row's position is "CPT" or "FLEX"
    regardless of what the real player plays (see
    data/player_crosswalk.py's SHOWDOWN_PSEUDO_POSITIONS), which silently
    made this whole function inert for Showdown: DST rows never got
    filtered out (never literally "DST"), and no real player ever hit the
    QB branch either. Real position is resolved the same way identity
    already is for these rows - from the player's own history - rather
    than trusted from the DK row.
    """
    engine = engine or get_engine()
    if not players:
        return {}

    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(players, engine)
    games_by_gsis = _load_recent_usage_batch(
        [gid for gid in gsis_by_dk_id.values() if gid is not None and not gid.startswith("DST_")], engine
    )

    excluded = {}
    for p in players:
        gsis_id = gsis_by_dk_id.get(p["player_id"])
        if gsis_id is None or gsis_id.startswith("DST_"):
            continue  # no crosswalk match (thin-data/LOW case), or a real defense - neither is a hard exclude here

        games = games_by_gsis.get(gsis_id, [])
        real_position = p["position"] if p["position"] not in ("CPT", "FLEX") else (
            games[0]["position"] if games else None
        )
        if real_position is None:
            continue  # a Showdown pseudo-position row with no history to recover a real position from

        is_excluded, reason = _would_be_hard_excluded(real_position, games)
        if is_excluded:
            excluded[p["player_id"]] = reason
    return excluded


def _would_be_hard_excluded(real_position, games):
    """The exact per-player decision hard_role_exclusions makes, given a
    real (not Showdown CPT/FLEX) position and that player's own recent
    usage history - factored out so models/calibration.py's backtest can
    apply this SAME rule against real historical data as-of each week
    (see _load_recent_usage_batch's `before` param), rather than risk a
    second, hand-copied implementation drifting out of sync with the one
    actually wired into the optimizer. Returns (excluded: bool, reason:
    str | None).
    """
    snap_pcts = [float(g["snap_pct"]) for g in games if g["snap_pct"] is not None]
    if len(snap_pcts) < MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE:
        return False, None  # too little data to be confident it's genuinely zero/unclear role, not just unrecorded

    if real_position == "QB":
        if min(snap_pcts) < MIN_SNAP_PCT_FOR_BENCHED and max(snap_pcts) >= MAX_SNAP_PCT_FOR_FULL_GAME:
            return True, (
                f"intermittent starter pattern in the last {len(snap_pcts)} recorded games "
                f"(snap share swings between {min(snap_pcts):.0%} and {max(snap_pcts):.0%}) - "
                "this is the real backup/spot-starter signature, not normal starter variance"
            )
        return False, None

    if max(snap_pcts) < HARD_EXCLUDE_MAX_SNAP_PCT:
        return True, (
            f"no meaningful snap share in any of the last {len(snap_pcts)} recorded games "
            f"(max {max(snap_pcts):.0%}) - no evidence of a real current role"
        )
    return False, None


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
