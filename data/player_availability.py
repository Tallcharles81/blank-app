from sqlalchemy import text

from data.nflverse_fetch import download_csv, fetch_schedules
from data.player_crosswalk import resolve_dk_players_to_gsis
from db.migrate import get_engine

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
    game_date = sample.game_time.date().isoformat()
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
