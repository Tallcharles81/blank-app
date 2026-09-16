from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import text

from data.nflverse_fetch import download_csv, fetch_schedules
from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine

# nflverse's schedule `gameday` is the game's real local date in US Eastern
# time (confirmed against real data: BUF@DET week 2 2026's gametime column
# reads "20:15", matching DK's own "08:15PM ET" for the same game exactly) -
# not UTC, and not each team's own local timezone.
_SCHEDULE_TZ = ZoneInfo("America/New_York")

# From nflverse's own dictionary (github.com/nflverse/nflreadr/data-raw/
# dictionary_roster_status.csv) - these are roster-level statuses, not weekly
# injury-report designations, so they're known well before Wed/Thu/Fri injury
# reports get filed. RES is the general "on the reserve list" bucket that
# covers IR; PUP and SUS are their own explicit statuses. The rest are players
# who are simply no longer on the team (cut/retired/waived/free agent) and
# obviously can't be rostered either.
HARD_EXCLUDE_ROSTER_STATUSES = {
    "RES",  # reserve list (this is where IR shows up)
    "PUP",  # physically unable to perform
    "SUS",  # suspended
    "RET",  # retired
    "CUT",  # cut from the roster
    "UFA",  # released, unrestricted free agent
    "TRC",
    "TRD",
    "TRT",  # released from the practice squad
    "NWT",  # waived
    "RFA",  # cut, restricted free agent
}

# The weekly injury report (separate source, separate cadence - see
# fetch_current_injury_report) uses these instead.
HARD_EXCLUDE_INJURY_STATUSES = {"Out"}
FLAG_INJURY_STATUSES = {"Questionable", "Doubtful"}


def fetch_current_roster_status(season, week):
    # Confirmed filename via nflreadr's own source (R/load_rosters_weekly.R):
    # weekly_rosters/roster_weekly_{season}.csv.gz - not guessed from the
    # release page's asset listing, which has previously been an unreliable,
    # truncating summary rather than a direct read.
    df = download_csv("weekly_rosters", f"roster_weekly_{season}.csv.gz")
    return df[df["week"] == week]


def fetch_current_injury_report(season, week):
    df = download_csv("injuries", f"injuries_{season}.csv.gz")
    return df[df["week"] == week]


def resolve_slate_season_week(slate_id, engine=None):
    """Derive (season, week) for a slate from its own game_time + team,
    matched against the real schedule - so the availability gate can't be
    skipped by a caller simply forgetting to pass season/week by hand.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        sample = conn.execute(
            text("SELECT team, game_time FROM slate_player_pool WHERE slate_id = :slate_id LIMIT 1"),
            {"slate_id": slate_id},
        ).fetchone()
    if sample is None:
        return None

    schedules_df = fetch_schedules()
    # game_time is stored UTC-aware (converted from the DK CSV's real ET
    # timestamp at parse time - see data/dk_salary_csv.py). Taking .date()
    # directly on that UTC value is wrong for any night game: an 8:15 PM ET
    # kickoff is 00:15 UTC the FOLLOWING calendar day, which would look for
    # a schedule row on a date that doesn't exist - a real bug, caught for
    # real on the Thursday-night game of a Thu-Mon slate (every game on the
    # earlier Sunday-only slate started by 4:25 PM ET, never late enough to
    # cross the UTC midnight boundary, so this never surfaced before).
    # Converting back to the schedule's own real timezone (US Eastern) before
    # extracting the date fixes this for every kickoff time, not just
    # Thursday's.
    game_date = sample.game_time.astimezone(_SCHEDULE_TZ).date().isoformat()
    match = schedules_df[
        (schedules_df["gameday"] == game_date)
        & ((schedules_df["home_team"] == sample.team) | (schedules_df["away_team"] == sample.team))
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return int(row["season"]), int(row["week"])


def get_availability_gate(dk_players, season, week, engine=None):
    """Resolve each DK player to their current roster/injury status for
    (season, week) and split them into excluded (hard blocker - must never
    appear in a generated lineup) vs flagged (Questionable/Doubtful - still
    eligible, surfaced as risk).

    Returns (excluded, flagged, injury_report_available):
    - excluded: {dk_player_id: reason string}
    - flagged: {dk_player_id: "Questionable" | "Doubtful"}
    - injury_report_available: whether nflverse had actually published a
      weekly injury report for this exact (season, week) yet - injury reports
      are filed Wed-Fri of game week, so for an upcoming week early in that
      cycle this can legitimately be False. Roster status (IR/PUP/Suspended)
      is unaffected by this - it's a different, earlier-available source.
    """
    engine = engine or get_engine()
    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(dk_players, engine)

    roster_df = fetch_current_roster_status(season, week)
    roster_status_by_gsis = dict(zip(roster_df["gsis_id"], roster_df["status"]))

    try:
        injuries_df = fetch_current_injury_report(season, week)
    except RuntimeError:
        injuries_df = None
    injury_report_available = injuries_df is not None and not injuries_df.empty
    injury_status_by_gsis = (
        dict(zip(injuries_df["gsis_id"], injuries_df["report_status"])) if injury_report_available else {}
    )

    excluded = {}
    flagged = {}
    for player in dk_players:
        dk_id = player["player_id"]
        gsis_id = gsis_by_dk_id.get(dk_id)
        if gsis_id is None:
            continue  # a crosswalk gap is data.player_crosswalk's concern, not this gate's

        # DST rows resolve to a synthetic "DST_{team}" id (see
        # data/player_crosswalk.py), never a real GSIS id - team defenses
        # aren't people, so they'd never legitimately appear in a per-player
        # roster feed. The absence check below is about real players only.
        if player["position"] != "DST" and gsis_id not in roster_status_by_gsis:
            # Absent from the current week's roster feed entirely - not even
            # an explicit status, just missing. A real, actively-employed NFL
            # player (53-man roster, practice squad, or reserve/IR) always
            # appears in this feed with SOME status; total absence means
            # retired, released-and-unsigned, or an edge-case designation this
            # feed doesn't capture under a normal code. Caught for real:
            # Brandon Aiyuk (real ACL/MCL/meniscus tear Oct 2024, hasn't
            # played since, his own GM has said he'll never play for the team
            # again, on a "Reserve/Left Squad" designation) is completely
            # absent from the 2026 week-2 roster feed - not listed as RES,
            # not listed at all - and was passing through this gate as
            # eligible before this check existed, since `.get()` on a missing
            # key returned None, which isn't in HARD_EXCLUDE_ROSTER_STATUSES
            # either. Absence is treated as MORE suspicious than an explicit
            # status, not less.
            excluded[dk_id] = "not found on any team's current roster"
            continue

        roster_status = roster_status_by_gsis.get(gsis_id)
        if roster_status in HARD_EXCLUDE_ROSTER_STATUSES:
            excluded[dk_id] = f"roster status: {roster_status}"
            continue

        injury_status = injury_status_by_gsis.get(gsis_id)
        if injury_status in HARD_EXCLUDE_INJURY_STATUSES:
            excluded[dk_id] = f"injury report: {injury_status}"
        elif injury_status in FLAG_INJURY_STATUSES:
            flagged[dk_id] = injury_status

    return excluded, flagged, injury_report_available


# ---------------------------------------------------------------------------
# Lock-time awareness.
#
# Every function above answers "is this player available" as of whenever the
# data was last fetched - none of it knows what time it actually is right
# now relative to any given game's real kickoff. That's never mattered on a
# single-slate-start slate (every game locks at the same moment, so either
# the whole slate is open or the whole thing's over), but a real Thu-Mon
# slate has games locking at meaningfully different real times across
# several days (this exact slate: BUF@DET locks Thu 2026-09-18 00:15 UTC,
# the Sunday early games lock 2026-09-20 17:00 UTC, Monday's game locks
# later still) - DraftKings itself only lets you edit a roster spot up
# until THAT player's own game starts (late swap), and there was previously
# no way for this codebase to know which of a lineup's spots were even
# still editable.
#
# game_locked() is a hard, deterministic fact (a game has started or it
# hasn't - no calibration risk the way the role-check thresholds had) - see
# models/optimizer.py's _load_player_pool, which hard-excludes any player
# whose game has already started from a freshly built lineup's pool, the
# same way an injury/role exclusion already does. lineup_lock_report()
# below is the reporting counterpart for an ALREADY-ENTERED lineup: which
# of its specific roster spots are locked (can't be touched) vs still open
# (still swappable before that player's own kickoff).
# ---------------------------------------------------------------------------


def game_locked(game_time, now=None):
    """Whether a single game_time (a real, UTC-aware kickoff timestamp from
    slate_player_pool) has already started as of `now` (defaults to the
    real current time). A game with no recorded game_time is never treated
    as locked - there's nothing to compare against, and silently excluding
    a player over a missing timestamp would be a data gap masquerading as a
    real finding, exactly what this project's conventions rule out.
    """
    if game_time is None:
        return False
    now = now or datetime.now(timezone.utc)
    return game_time <= now


def game_lock_status(dk_players, now=None):
    """{player_id: locked bool} for a list of player dicts that already
    carry game_time (e.g. models/optimizer.py's player pool once game_time
    is selected). Pure - no DB access - so a caller who already has
    game_time loaded doesn't pay for a second query just to check locks.
    """
    now = now or datetime.now(timezone.utc)
    return {p["player_id"]: game_locked(p.get("game_time"), now) for p in dk_players}


def lineup_lock_report(slate_id, player_ids, engine=None, now=None):
    """For a SPECIFIC lineup (the exact player_ids in it, e.g. an already-
    entered real lineup) - which roster spots are locked (that player's
    game has already started - DK will not let you touch this spot) vs
    still open (still swappable before kickoff), and how long until an open
    spot's own lock, so a caller checking in on a multi-day slate mid-week
    can see real remaining swap windows, not just a locked/open bit.

    Returns a list of dicts: player_id, name, position, team, game_time,
    locked, seconds_until_lock (None if already locked).
    """
    engine = engine or get_engine()
    now = now or datetime.now(timezone.utc)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT player_id, name, position, team, opponent, game_time FROM slate_player_pool "
                "WHERE slate_id = :slate_id AND player_id = ANY(:player_ids)"
            ),
            {"slate_id": slate_id, "player_ids": list(player_ids)},
        ).mappings().fetchall()

    report = []
    for row in rows:
        row = dict(row)
        locked = game_locked(row["game_time"], now)
        seconds_until_lock = None if locked or row["game_time"] is None else (row["game_time"] - now).total_seconds()
        report.append({**row, "locked": locked, "seconds_until_lock": seconds_until_lock})
    return report
