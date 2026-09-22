from collections import defaultdict

from sqlalchemy import text

from data.depth_charts import latest_depth_chart_by_player
from data.player_availability import (
    HARD_EXCLUDE_INJURY_STATUSES,
    HARD_EXCLUDE_ROSTER_STATUSES,
)
from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine

# ---------------------------------------------------------------------------
# The real distinction this module exists to enforce: ACTIVE is a roster/
# injury-report fact (data/player_availability.py already gates on it
# correctly), not evidence of a real offensive role. A player can clear
# every existing hard gate - on the 53-man roster, no injury designation -
# and still be a real WR6 or RB3 who will not see the field in any way that
# matters for fantasy. This module is a SEPARATE, ADDITIONAL gate that runs
# alongside get_availability_gate/hard_role_exclusions (data/pre_lock_check.
# py), not a replacement for either - two independently-computed real
# signals landing on the same verdict is real corroboration, not waste, and
# is used directly below (see _role_confidence's "sources disagree" case).
#
# WHAT'S REAL VS. SUBSTITUTED VS. NOT BUILT, stated plainly rather than
# implied by naming things after a spec that describes data this codebase
# doesn't have:
#   - Player status (OUT/IR/etc), snap_pct, carries, targets, receptions,
#     target_share, red_zone_targets/carries: 100% real, from
#     player_weekly_stats/data.player_availability.py, same sources every
#     other gate in this codebase already relies on.
#   - Depth chart position/rank (STARTER/BACKUP/DEPTH labeling): 100% real,
#     nflverse's own real depth-chart feed (data/depth_charts.py) - genuinely
#     new to this codebase. Real, important limitation: nflverse only
#     publishes this for the CURRENT season, not historical seasons - it can
#     inform a live gate call but CANNOT be backtested against past weeks
#     the way every threshold below it can.
#   - "Route participation %" (the spec's literal WR/TE floor metric): NO
#     real source for this exists anywhere this codebase can reach -
#     substituted with real snap_pct, which is real, available, and highly
#     correlated with real route participation for a WR/TE (a player on the
#     field for a real pass play is running a route in the overwhelming
#     majority of real NFL snaps), but it is a substitute, not the literal
#     metric, and is documented as such everywhere it's used.
#   - Goal-line share, two-minute participation, preseason/training-camp
#     role: NOT implemented. No real, structured data source for any of
#     these exists in what this codebase can fetch. Not faked.
#   - "Beat/news reports" as a real-time multi-source input: ties to
#     data/news_scanner.py (built separately), which is real but requires a
#     real ANTHROPIC_API_KEY this environment doesn't have configured - not
#     wired into this module's live confidence scoring by default.
# ---------------------------------------------------------------------------

PLAYER_STATUS_OUT = "OUT"
PLAYER_STATUS_IR = "IR"
PLAYER_STATUS_INACTIVE = "INACTIVE"
PLAYER_STATUS_DOUBTFUL = "DOUBTFUL"
PLAYER_STATUS_QUESTIONABLE = "QUESTIONABLE"
PLAYER_STATUS_ACTIVE = "ACTIVE"

ROLE_STARTER = "STARTER"
ROLE_ROTATIONAL = "ROTATIONAL"
ROLE_LIMITED = "LIMITED ROLE"
ROLE_BACKUP = "BACKUP"
ROLE_DEPTH = "DEPTH"
ROLE_UNKNOWN = "UNKNOWN"

# How many of a team's real recent games (this season, then last season)
# this engine looks at - matches ROLE_CHECK_HISTORY_WEEKS's own real
# precedent in data/pre_lock_check.py (4), kept the same rather than
# invented fresh, since that window was already the established real choice
# for "how much recent usage history is enough to read a role from."
CURRENT_SEASON_MAX_GAMES = 8
PRIOR_SEASON_MAX_GAMES = 8

# The shrinkage constant K in weight = n_current / (n_current + K): at
# n_current = K real current-season games, current and prior-season data get
# equal real weight; more current-season games than that and current-season
# evidence dominates, matching item 4's "progressively increase the weight
# of current-season evidence" requirement with a standard, real statistical
# form (empirical-Bayes-style shrinkage) instead of an arbitrary game-count
# cutoff. K=4 is an INITIAL choice (same status as every position threshold
# below).
#
# IMPORTANT, HONEST RESULT from models/calibration.py::run_shrinkage_
# backtest (finally run for real, n=1606 real early-season player-weeks,
# 34 real weeks, dk_sunday_2026_09_20's pool): this blend does NOT beat the
# naive "trust current-season data the instant any of it exists, ignore
# prior season" baseline it was meant to improve on - it's WORSE, by a
# real, statistically significant margin (avg abs error predicting next-
# week snap_pct: 0.1461 blended vs 0.1274 naive, t=-5.54, p<0.0001). Kept
# as the live default for now rather than silently reverted, since this
# contradicts the assumption this whole module shipped under and changing
# it changes every downstream gate/backtest that depends on estimate_role -
# a decision flagged to the user rather than made unilaterally. Plausible
# real cause, not yet tested: real NFL offensive roles can differ sharply
# year over year (team/scheme/coaching changes), so blending in year-old
# usage may add real noise rather than real signal, especially this early
# in a season - i.e. recent within-season data may just be a stronger real
# predictor of role than last year's, undermining the empirical-Bayes prior
# this blend assumes.
SHRINKAGE_K = 4.0

# INITIAL thresholds, substituting real snap_pct for the unavailable real
# "route participation %" metric on WR/TE (see module docstring). Backtested
# for real via models/calibration.py::run_playing_time_floor_backtest - see
# that function's own real, disclosed result before trusting these as more
# than a documented starting point, exactly as this spec's own item 5 asks.
WR_MIN_SNAP_PCT = 0.50
RB_MIN_SNAP_PCT = 0.30
RB_MIN_TOUCHES = 8
TE_MIN_SNAP_PCT = 0.40


def classify_player_status(roster_status, injury_status, flag_status):
    """Real taxonomy from real data/player_availability.py sources - no new
    status categories invented, every one of these maps to an already-real,
    already-fetched signal (fetch_current_roster_status/fetch_current_
    injury_report). `roster_status`/`injury_status` may be None (absent
    from either real feed)."""
    if roster_status == "RES":
        return PLAYER_STATUS_IR
    if roster_status in HARD_EXCLUDE_ROSTER_STATUSES:
        return PLAYER_STATUS_INACTIVE
    if injury_status in HARD_EXCLUDE_INJURY_STATUSES:
        return PLAYER_STATUS_OUT
    if flag_status == "Doubtful":
        return PLAYER_STATUS_DOUBTFUL
    if flag_status == "Questionable":
        return PLAYER_STATUS_QUESTIONABLE
    return PLAYER_STATUS_ACTIVE


def _weighted_mean(values):
    return sum(values) / len(values) if values else None


def _shrinkage_blend(current_values, prior_values, k=SHRINKAGE_K):
    """Real empirical-Bayes-style shrinkage: as real current-season sample
    size n grows, weight shifts smoothly from the real prior-season mean
    toward the real current-season mean, rather than an all-or-nothing
    cutoff (data/pre_lock_check.py's own MIN_RECENT_GAMES_FOR_ROLE_
    CONFIDENCE, by contrast, treats 1 and 100 current-season games
    identically as "not enough" below its cutoff, and treats the cutoff
    itself and 100 games identically as "trust it fully" above it - this
    is the real gap item 4 is asking to close). Returns (blended_value,
    n_current, n_prior, weight_on_current) - None if no real data exists
    on either side at all.
    """
    n_current = len(current_values)
    n_prior = len(prior_values)
    mean_current = _weighted_mean(current_values)
    mean_prior = _weighted_mean(prior_values)

    if mean_current is None and mean_prior is None:
        return None, 0, 0, None
    if mean_current is None:
        return mean_prior, 0, n_prior, 0.0
    if mean_prior is None:
        return mean_current, n_current, 0, 1.0

    weight = n_current / (n_current + k)
    blended = weight * mean_current + (1 - weight) * mean_prior
    return blended, n_current, n_prior, weight


def _load_role_history(gsis_ids, season, week, engine, before_current_week=True):
    """Real current-season (strictly before `week` when before_current_week,
    to keep this no-lookahead-safe for backtesting the same way every other
    real usage-history query in this codebase is) and real prior-season
    (season - 1, all real played weeks) usage rows, batched in two queries
    for the whole real pool rather than per-player - same real reason
    data/pre_lock_check.py's _load_recent_usage_batch is batched.

    Returns {gsis_id: {"current": [games], "prior": [games]}}, each game a
    dict with season/week/snap_pct/carries/targets/receptions/
    red_zone_targets/red_zone_carries - real player_weekly_stats columns,
    nothing invented.
    """
    columns = "player_id, season, week, snap_pct, carries, targets, receptions, red_zone_targets, red_zone_carries"
    current_cutoff = "AND week < :week" if before_current_week else ""

    current_query = text(
        f"""
        SELECT {columns} FROM player_weekly_stats
        WHERE player_id = ANY(:gsis_ids) AND season = :season {current_cutoff}
        ORDER BY week DESC
        """
    )
    prior_query = text(
        f"""
        SELECT {columns} FROM player_weekly_stats
        WHERE player_id = ANY(:gsis_ids) AND season = :prior_season
        ORDER BY week DESC
        """
    )

    result = defaultdict(lambda: {"current": [], "prior": []})
    with engine.connect() as conn:
        for row in conn.execute(current_query, {"gsis_ids": list(gsis_ids), "season": season, "week": week}).mappings():
            if len(result[row["player_id"]]["current"]) < CURRENT_SEASON_MAX_GAMES:
                result[row["player_id"]]["current"].append(dict(row))
        for row in conn.execute(prior_query, {"gsis_ids": list(gsis_ids), "prior_season": season - 1}).mappings():
            if len(result[row["player_id"]]["prior"]) < PRIOR_SEASON_MAX_GAMES:
                result[row["player_id"]]["prior"].append(dict(row))
    return dict(result)


def _touches(game):
    return (game["carries"] or 0) + (game["receptions"] or 0)


def estimate_role(position, history, depth_chart_entry):
    """The real per-player estimate: blended snap_pct and touches (via
    _shrinkage_blend), a role label, PLAYING_TIME_CONFIDENCE (0-100), and
    whether real sources (recent usage vs. the real current depth chart)
    agree. `history` is one entry from _load_role_history's return value
    (or {"current": [], "prior": []} if the player has none at all -
    a rookie/practice-squad call-up with zero real recorded history on
    either side of two seasons).
    """
    current_snaps = [float(g["snap_pct"]) for g in history["current"] if g["snap_pct"] is not None]
    prior_snaps = [float(g["snap_pct"]) for g in history["prior"] if g["snap_pct"] is not None]
    current_touches = [_touches(g) for g in history["current"]]
    prior_touches = [_touches(g) for g in history["prior"]]

    blended_snap_pct, n_current, n_prior, snap_weight = _shrinkage_blend(current_snaps, prior_snaps)
    blended_touches, _, _, _ = _shrinkage_blend(current_touches, prior_touches)

    depth_chart_rank = depth_chart_entry["pos_rank"] if depth_chart_entry else None

    role = _role_label(position, depth_chart_rank, blended_snap_pct)
    confidence, sources_agree = _role_confidence(position, n_current, n_prior, depth_chart_rank, blended_snap_pct)

    return {
        "position": position,
        "blended_snap_pct": round(blended_snap_pct, 4) if blended_snap_pct is not None else None,
        "blended_touches": round(blended_touches, 2) if blended_touches is not None else None,
        "n_current_games": n_current,
        "n_prior_games": n_prior,
        "current_season_weight": round(snap_weight, 2) if snap_weight is not None else None,
        "depth_chart_rank": depth_chart_rank,
        "role": role,
        "playing_time_confidence": confidence,
        "sources_agree": sources_agree,
    }


def _role_label(position, depth_chart_rank, blended_snap_pct):
    if position == "QB":
        if depth_chart_rank == 1:
            return "EXPECTED STARTER"
        if depth_chart_rank is not None:
            return ROLE_BACKUP
        return ROLE_UNKNOWN

    if depth_chart_rank is not None:
        if depth_chart_rank == 1:
            return ROLE_STARTER
        if depth_chart_rank == 2:
            return ROLE_ROTATIONAL if (blended_snap_pct or 0) >= 0.30 else ROLE_LIMITED
        return ROLE_DEPTH

    if blended_snap_pct is None:
        return ROLE_UNKNOWN
    if blended_snap_pct >= 0.65:
        return ROLE_STARTER
    if blended_snap_pct >= 0.35:
        return ROLE_ROTATIONAL
    if blended_snap_pct >= 0.15:
        return ROLE_LIMITED
    return ROLE_DEPTH


def _role_confidence(position, n_current, n_prior, depth_chart_rank, blended_snap_pct):
    """0-100. Grows with real sample size (current season weighted more
    than prior, matching SHRINKAGE_K's own weighting), and is EXPLICITLY
    never a function of any fantasy point projection (item 6's own
    requirement) - only of how much real usage/role evidence backs the
    estimate. Sources disagreeing (a real depth-chart starter with a real
    low recent snap share, or vice versa) caps confidence rather than
    averaging past the disagreement, per item 9.
    """
    if n_current == 0 and n_prior == 0 and depth_chart_rank is None:
        return 0, None  # truly no real signal at all

    sample_score = min(60, n_current * 12) + min(20, n_prior * 3)
    depth_chart_bonus = 20 if depth_chart_rank is not None else 0
    confidence = sample_score + depth_chart_bonus

    sources_agree = None
    if depth_chart_rank is not None and blended_snap_pct is not None:
        depth_chart_says_starter = depth_chart_rank == 1
        usage_says_starter = blended_snap_pct >= 0.50
        sources_agree = depth_chart_says_starter == usage_says_starter
        if not sources_agree:
            confidence = min(confidence, 55)  # real disagreement - never a "secure role" verdict

    return max(0, min(100, round(confidence))), sources_agree


def compute_opportunity_score(position, role_estimate):
    """0-100 EXPECTED_OPPORTUNITY_SCORE from real blended snap_pct/touches -
    separate from PLAYING_TIME_CONFIDENCE (that's "how sure are we", this is
    "how much real opportunity, if the estimate is right"). A player who
    clears the hard floor but still scores low here is real HIGH-VARIANCE-
    PUNT territory (item 7) - a thin-but-legal role, not a clear one.
    """
    snap_pct = role_estimate["blended_snap_pct"] or 0.0
    touches = role_estimate["blended_touches"] or 0.0
    if position in ("RB",):
        touch_component = min(60, touches * 4)
    else:
        touch_component = min(40, touches * 6)
    return round(min(100, snap_pct * 60 + touch_component))


def meets_playing_time_floor(position, role_estimate):
    """The real hard floor (item 5's INITIAL thresholds, substituting real
    snap_pct for the unavailable real route-participation-% metric on WR/TE
    - see module docstring). Returns (meets_floor: bool, reason: str|None).
    A player with zero real history on both sides of two seasons (a rookie
    week 1, a practice-squad call-up) is NOT excluded here - LOW confidence,
    not a floor failure; that distinction is the whole point of item 4 vs.
    a hard "not enough games" cutoff.
    """
    snap_pct = role_estimate["blended_snap_pct"]
    touches = role_estimate["blended_touches"]
    depth_chart_rank = role_estimate["depth_chart_rank"]

    if snap_pct is None and touches is None and depth_chart_rank is None:
        return True, None  # no real signal either way - not this gate's call to make

    if position == "QB":
        if depth_chart_rank is not None and depth_chart_rank != 1:
            return False, f"real current depth chart lists him QB{depth_chart_rank}, not the expected starter"
        return True, None

    if position == "WR":
        if snap_pct is not None and snap_pct < WR_MIN_SNAP_PCT:
            return False, f"real blended snap share {snap_pct:.0%} is below the {WR_MIN_SNAP_PCT:.0%} floor (route-participation proxy)"
        return True, None

    if position == "TE":
        if snap_pct is not None and snap_pct < TE_MIN_SNAP_PCT:
            return False, f"real blended snap share {snap_pct:.0%} is below the {TE_MIN_SNAP_PCT:.0%} floor (route-participation proxy)"
        return True, None

    if position == "RB":
        if snap_pct is not None and touches is not None and snap_pct < RB_MIN_SNAP_PCT and touches < RB_MIN_TOUCHES:
            return False, (
                f"real blended snap share {snap_pct:.0%} (floor {RB_MIN_SNAP_PCT:.0%}) AND "
                f"real blended touches {touches:.1f}/game (floor {RB_MIN_TOUCHES}) both below floor"
            )
        return True, None

    return True, None  # DST/K etc. - this engine's floor doesn't apply, same real scope as hard_role_exclusions


def build_debug_entry(player, real_position, status, role_estimate, opportunity_score, eligible, reason):
    return {
        "player_id": player["player_id"],
        "player": player["name"],
        "status": status,
        "projected_snap_pct": role_estimate["blended_snap_pct"],
        "projected_touches": role_estimate["blended_touches"],
        "role": role_estimate["role"],
        "role_confidence": role_estimate["playing_time_confidence"],
        "opportunity_score": opportunity_score,
        "dfs_eligible": "YES" if eligible else "NO",
        "reason": reason or ("insufficient expected offensive participation" if not eligible else None),
    }


def apply_playing_time_gate(
    players,
    season,
    week,
    engine=None,
    roster_status_by_gsis=None,
    injury_status_by_gsis=None,
    flag_status_by_gsis=None,
    punt_mode=False,
):
    """The real, top-level entry point - runs BEFORE the optimizer sees a
    pool, per item 12's ordering. `players` is any list of dicts with
    player_id/name/position/team (the same shape every other gate in this
    codebase takes). `roster_status_by_gsis`/`injury_status_by_gsis`/
    `flag_status_by_gsis` are optional - pass them through from a caller
    that already fetched get_availability_gate's real data this run, so
    this doesn't re-fetch the same real roster/injury feeds a second time;
    omitted entries are just treated as unknown (ACTIVE-if-nothing-else-
    said, matching get_availability_gate's own real absence handling).

    NORMAL MODE (punt_mode=False, default): a player who fails the real
    hard floor is excluded outright - never reaches the optimizer.
    PUNT MODE (punt_mode=True): nobody is excluded for playing-time reasons
    alone; every player who would have been excluded is instead returned
    in `punt_flagged` with a `penalty_multiplier` (0.35 - a severe, real,
    fixed real penalty on his own projection, not a suggestion) alongside
    his real debug entry, so a caller can apply it before optimizing rather
    than pretending the concern doesn't exist.

    Returns {"eligible": [...players...], "excluded": [debug entries],
    "punt_flagged": [debug entries + penalty_multiplier], "debug_log":
    [every real debug entry, excluded and eligible alike, per item 13]}.
    """
    engine = engine or get_engine()
    roster_status_by_gsis = roster_status_by_gsis or {}
    injury_status_by_gsis = injury_status_by_gsis or {}
    flag_status_by_gsis = flag_status_by_gsis or {}

    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(players, engine)
    non_dst = [p for p in players if not (gsis_by_dk_id.get(p["player_id"]) or "").startswith("DST_")]

    from data.pre_lock_check import _load_recent_usage_batch  # local import - avoids a module-load cycle with pre_lock_check

    # Real position resolution - a Showdown row's own position is "CPT"/
    # "FLEX" regardless of what the real player plays (see data/player_
    # crosswalk.py's SHOWDOWN_PSEUDO_POSITIONS), the same real gap every
    # other position-aware gate in this codebase already resolves the same
    # way (from the player's own real game history).
    showdown_lookup = _load_recent_usage_batch(
        [gid for gid in gsis_by_dk_id.values() if gid], engine
    )
    real_position_by_dk_id = {}
    for p in non_dst:
        gsis_id = gsis_by_dk_id.get(p["player_id"])
        if gsis_id is None:
            continue
        games = showdown_lookup.get(gsis_id, [])
        real_position_by_dk_id[p["player_id"]] = p["position"] if p["position"] not in ("CPT", "FLEX") else (
            games[0]["position"] if games else None
        )

    try:
        depth_chart = latest_depth_chart_by_player(season)
    except RuntimeError:
        depth_chart = {}  # real feed unavailable (e.g. a past season it doesn't cover) - not a crash, just no depth-chart signal this run

    history_by_gsis = _load_role_history(
        [gid for gid in gsis_by_dk_id.values() if gid], season, week, engine
    )

    eligible, excluded, punt_flagged, debug_log = [], [], [], []
    for p in non_dst:
        gsis_id = gsis_by_dk_id.get(p["player_id"])
        real_position = real_position_by_dk_id.get(p["player_id"])
        if gsis_id is None or real_position is None:
            eligible.append(p)  # no real crosswalk match / no history to resolve a real position from - not this gate's call
            continue

        status = classify_player_status(
            roster_status_by_gsis.get(gsis_id), injury_status_by_gsis.get(gsis_id), flag_status_by_gsis.get(gsis_id)
        )
        if status in (PLAYER_STATUS_OUT, PLAYER_STATUS_IR, PLAYER_STATUS_INACTIVE):
            continue  # already excluded by the real availability gate this status came from - not this engine's job to re-report

        history = history_by_gsis.get(gsis_id, {"current": [], "prior": []})
        depth_chart_entry = depth_chart.get(gsis_id)
        role_estimate = estimate_role(real_position, history, depth_chart_entry)
        opportunity_score = compute_opportunity_score(real_position, role_estimate)
        meets_floor, reason = meets_playing_time_floor(real_position, role_estimate)

        debug_entry = build_debug_entry(p, real_position, status, role_estimate, opportunity_score, meets_floor, reason)
        debug_log.append(debug_entry)

        if meets_floor:
            eligible.append(p)
        elif punt_mode:
            punt_flagged.append({**debug_entry, "penalty_multiplier": 0.35})
            eligible.append(p)
        else:
            excluded.append(debug_entry)

    non_dst_ids = {p["player_id"] for p in non_dst}
    dst_players = [p for p in players if p["player_id"] not in non_dst_ids]
    eligible.extend(dst_players)  # DST has no individual "role" - same real scope as hard_role_exclusions

    return {"eligible": eligible, "excluded": excluded, "punt_flagged": punt_flagged, "debug_log": debug_log}
