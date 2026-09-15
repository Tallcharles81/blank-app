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
    for i, row in enumerate(rows):
        if row[: len(PLAYER_POOL_HEADER)] == PLAYER_POOL_HEADER:
            return i
    raise ValueError("Could not find DraftKings player pool header row")


def _detect_slate_type(rows, header_idx):
    preamble = "\n".join(",".join(row) for row in rows[:header_idx])
    if ",".join(SHOWDOWN_ROSTER) in preamble or re.search(r"\bCPT\b", preamble):
        return "showdown"
    return "classic"


def parse_dk_salary_csv(file_path):
    with open(file_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header_idx = _find_header_row(rows)
    slate_type = _detect_slate_type(rows, header_idx)

    players = []
    for row in rows[header_idx + 1 :]:
        if len(row) < len(PLAYER_POOL_HEADER) or not row[0].strip():
            continue
        position, _name_and_id, name, player_id, _roster_position, salary, game_info, team, avg_pts = row[:9]
        away_team, home_team, kickoff_time = parse_game_info(game_info)
        team = team.strip()
        opponent = home_team if team == away_team else away_team
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
    engine = engine or get_engine()

    if len(player_ids_in_slot_order) == len(CLASSIC_ROSTER):
        roster = CLASSIC_ROSTER
    elif len(player_ids_in_slot_order) == len(SHOWDOWN_ROSTER):
        roster = SHOWDOWN_ROSTER
    else:
        raise ValueError(
            f"Expected {len(CLASSIC_ROSTER)} (classic) or {len(SHOWDOWN_ROSTER)} "
            f"(showdown) player IDs, got {len(player_ids_in_slot_order)}"
        )

    select_sql = text(
        """
        SELECT player_id, name FROM slate_player_pool
        WHERE slate_id = :slate_id AND player_id = ANY(:player_ids)
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            select_sql, {"slate_id": slate_id, "player_ids": list(player_ids_in_slot_order)}
        ).fetchall()
    names_by_id = {row.player_id: row.name for row in rows}

    missing = [pid for pid in player_ids_in_slot_order if pid not in names_by_id]
    if missing:
        raise ValueError(f"Player ID(s) not found in slate_player_pool for slate {slate_id}: {missing}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(roster)
        writer.writerow([f"{names_by_id[pid]} ({pid})" for pid in player_ids_in_slot_order])
