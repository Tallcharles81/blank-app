import csv
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text

from db.migrate import get_engine

PLAYER_POOL_HEADER = [
    "Position",
    "Name + ID",
    "Name",
    "ID",
    "Roster Position",
    "Salary",
    "Game Info",
    "TeamAbbrev",
    "AvgPointsPerGame",
]

CLASSIC_ROSTER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SHOWDOWN_ROSTER = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

GAME_INFO_RE = re.compile(
    r"^(?P<away>[A-Z]+)@(?P<home>[A-Z]+)\s+"
    r"(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<time>\d{1,2}:\d{2}(?:AM|PM))\s+ET$"
)

# DK's "Game Info" times are always ET but never say which offset (EST/EDT), and
# game_time is stored as TIMESTAMPTZ, so attach the zone explicitly rather than
# storing a naive datetime that would be misinterpreted as UTC or local server time.
EASTERN = ZoneInfo("America/New_York")


def parse_game_info(game_info):
    match = GAME_INFO_RE.match(game_info.strip())
    if not match:
        raise ValueError(f"Unrecognized Game Info format: {game_info!r}")
    kickoff_time = datetime.strptime(
        f"{match['date']} {match['time']}", "%m/%d/%Y %I:%M%p"
    ).replace(tzinfo=EASTERN)
    return match["away"], match["home"], kickoff_time


def _find_header_row(rows):
    # DK's real contest-export CSVs put the player pool to the right of the
    # entry-template columns (e.g. 10 blank leading fields for a Classic slate's
    # "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,," row), so the header isn't at column 0 -
    # search for it at any offset instead of assuming a fixed column.
    header_len = len(PLAYER_POOL_HEADER)
    for i, row in enumerate(rows):
        for offset in range(len(row) - header_len + 1):
            if row[offset : offset + header_len] == PLAYER_POOL_HEADER:
                return i, offset
    raise ValueError("Could not find DraftKings player pool header row")


def _detect_slate_type(rows, header_idx):
    # The player-pool header itself is identical for Classic and Showdown slates;
    # the roster shape only shows up in the entry-template row above it (e.g.
    # "QB,RB,RB,WR,WR,WR,TE,FLEX,DST" vs "CPT,FLEX,FLEX,FLEX,FLEX,FLEX"), so scan
    # the rows before the header rather than the header row or the player data.
    preamble = "\n".join(",".join(row) for row in rows[:header_idx])
    if ",".join(SHOWDOWN_ROSTER) in preamble or re.search(r"\bCPT\b", preamble):
        return "showdown"
    return "classic"


def parse_dk_salary_csv(file_path):
    with open(file_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header_idx, col_offset = _find_header_row(rows)
    slate_type = _detect_slate_type(rows, header_idx)

    players = []
    for row in rows[header_idx + 1 :]:
        fields = row[col_offset : col_offset + len(PLAYER_POOL_HEADER)]
        if len(fields) < len(PLAYER_POOL_HEADER) or not fields[0].strip():
            continue
        real_position, _name_and_id, name, player_id, roster_position, salary, game_info, team, avg_pts = fields
        away_team, home_team, kickoff_time = parse_game_info(game_info)
        team = team.strip()
        opponent = home_team if team == away_team else away_team
        # DraftKings' real "Position" column is the true football position
        # (QB/RB/WR/TE/DST) on EVERY slate type, including Showdown - it's
        # "Roster Position" that carries the CPT/FLEX slot label there
        # (confirmed against a real Showdown CSV export, e.g. the same real
        # player appearing as two rows, "Jahmyr Gibbs"/RB/CPT at 1.5x salary
        # and "Jahmyr Gibbs"/RB/FLEX at base salary - both with real
        # position "RB" in that column). The rest of this codebase's
        # Showdown handling (data/player_crosswalk.py's
        # SHOWDOWN_PSEUDO_POSITIONS, models/optimizer.py's _solve_showdown,
        # hard_role_exclusions, get_availability_gate, the correlation
        # model) was all built and tested against CPT/FLEX sitting in
        # slate_player_pool.position specifically (with real position
        # re-derived from game history when needed) - so a Showdown row
        # stores its roster slot label there instead of the real position,
        # to match that existing, already-tested contract; a Classic row
        # keeps using its real position as always (Classic's own "Roster
        # Position" is just each row's fixed slot template label, e.g. "RB"
        # or "FLEX", not something any existing code depends on).
        position = roster_position if slate_type == "showdown" else real_position
        players.append(
            {
                "player_id": player_id.strip(),
                "name": name.strip(),
                "position": position.strip(),
                "salary": int(salary),
                "team": team,
                "opponent": opponent,
                "game_time": kickoff_time,
                "avg_points_per_game": float(avg_pts) if avg_pts.strip() else None,
            }
        )

    return slate_type, players


def load_slate_player_pool(slate_id, file_path, engine=None):
    engine = engine or get_engine()
    slate_type, players = parse_dk_salary_csv(file_path)

    upsert_sql = text(
        """
        INSERT INTO slate_player_pool
            (slate_id, player_id, name, position, salary, team, opponent, game_time, avg_points_per_game)
        VALUES
            (:slate_id, :player_id, :name, :position, :salary, :team, :opponent, :game_time, :avg_points_per_game)
        ON CONFLICT (slate_id, player_id) DO UPDATE SET
            name = EXCLUDED.name,
            position = EXCLUDED.position,
            salary = EXCLUDED.salary,
            team = EXCLUDED.team,
            opponent = EXCLUDED.opponent,
            game_time = EXCLUDED.game_time,
            avg_points_per_game = EXCLUDED.avg_points_per_game
        """
    )

    with engine.begin() as conn:
        for player in players:
            conn.execute(upsert_sql, {"slate_id": slate_id, **player})

    return slate_type, len(players)


def write_dk_upload_csv(slate_id, player_ids_in_slot_order, output_path, engine=None):
    """`player_ids_in_slot_order`: either ONE lineup's flat list of ids in
    slot order, or a list of such lists for several lineups at once (DK's
    own real bulk-upload template is exactly this - one shared header row,
    then one data row per lineup, all uploaded together in a single file).
    """
    engine = engine or get_engine()

    # Normalize to "a list of lineups" either way, so the single-lineup case
    # is just the n=1 case of the loop below rather than separate code paths
    # that could drift apart.
    if player_ids_in_slot_order and isinstance(player_ids_in_slot_order[0], str):
        lineups_ids = [player_ids_in_slot_order]
    else:
        lineups_ids = list(player_ids_in_slot_order)
    if not lineups_ids:
        raise ValueError("No lineups given")

    roster = None
    for ids in lineups_ids:
        if len(ids) == len(CLASSIC_ROSTER):
            this_roster = CLASSIC_ROSTER
        elif len(ids) == len(SHOWDOWN_ROSTER):
            this_roster = SHOWDOWN_ROSTER
        else:
            raise ValueError(
                f"Expected {len(CLASSIC_ROSTER)} (classic) or {len(SHOWDOWN_ROSTER)} "
                f"(showdown) player IDs, got {len(ids)}"
            )
        if roster is None:
            roster = this_roster
        elif this_roster != roster:
            raise ValueError("All lineups in one upload file must be the same roster shape (classic vs showdown)")

    all_ids = {pid for ids in lineups_ids for pid in ids}
    select_sql = text(
        """
        SELECT player_id, name FROM slate_player_pool
        WHERE slate_id = :slate_id AND player_id = ANY(:player_ids)
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(select_sql, {"slate_id": slate_id, "player_ids": list(all_ids)}).fetchall()
    names_by_id = {row.player_id: row.name for row in rows}

    missing = sorted(all_ids - names_by_id.keys())
    if missing:
        raise ValueError(f"Player ID(s) not found in slate_player_pool for slate {slate_id}: {missing}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(roster)
        for ids in lineups_ids:
            writer.writerow([f"{names_by_id[pid]} ({pid})" for pid in ids])
