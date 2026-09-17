"""Core lineup-optimization logic for DraftKings NBA Classic contests.

Uses integer linear programming (PuLP) to build salary-cap-constrained,
position-eligible lineups that maximize projected fantasy points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

SALARY_CAP = 50_000
ROSTER_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

SLOT_ELIGIBILITY = {
    "PG": {"PG"},
    "SG": {"SG"},
    "SF": {"SF"},
    "PF": {"PF"},
    "C": {"C"},
    "G": {"PG", "SG"},
    "F": {"SF", "PF"},
    "UTIL": {"PG", "SG", "SF", "PF", "C"},
}

MIN_GAMES_REQUIRED = 2

REQUIRED_COLUMNS = ["Name", "Salary", "Position"]

_COLUMN_ALIASES = {
    "name": "Name",
    "player": "Name",
    "salary": "Salary",
    "position": "Position",
    "roster position": "Position",
    "pos": "Position",
    "team": "Team",
    "teamabbrev": "Team",
    "game info": "Game",
    "game": "Game",
    "avgpointspergame": "Projection",
    "projection": "Projection",
    "fppg": "Projection",
    "proj": "Projection",
}


def normalize_positions(pos_str: str) -> set:
    """Turn a DK-style position string ("PG/SG") into a set of positions."""
    if not isinstance(pos_str, str):
        return set()
    parts = pos_str.replace(",", "/").split("/")
    return {p.strip().upper() for p in parts if p.strip()}


def extract_game(game_info: str, team: str) -> str:
    """Derive a stable game identifier from a DK 'Game Info' field.

    Falls back to the team name when no game info is available, so every
    player still has *some* grouping key (single-game slates degrade to
    "everyone shares one game", which is handled by the caller).
    """
    if isinstance(game_info, str) and game_info.strip():
        # DK format looks like "BOS@NYK 07:30PM ET" - the matchup is the
        # part before the first space, and is unique per game.
        return game_info.strip().split(" ")[0]
    return str(team) if team else "UNKNOWN"


def load_player_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an arbitrary DK-ish CSV into the columns the optimizer needs.

    Produces: Name, Salary, Positions (set), Team, Game, Projection, PlayerId
    """
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[key]
    normalized = df.rename(columns=rename_map).copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in normalized.columns]
    if missing:
        raise ValueError(
            f"Player pool is missing required column(s): {', '.join(missing)}"
        )

    if "Team" not in normalized.columns:
        normalized["Team"] = ""
    if "Game" not in normalized.columns:
        normalized["Game"] = ""
    if "Projection" not in normalized.columns:
        normalized["Projection"] = 0.0

    normalized["Salary"] = (
        normalized["Salary"].astype(str).str.replace(r"[^0-9.-]", "", regex=True)
    )
    normalized["Salary"] = pd.to_numeric(normalized["Salary"], errors="coerce")
    normalized["Projection"] = pd.to_numeric(
        normalized["Projection"], errors="coerce"
    ).fillna(0.0)

    normalized["Positions"] = normalized["Position"].apply(normalize_positions)
    normalized["Game"] = [
        extract_game(g, t) for g, t in zip(normalized["Game"], normalized["Team"])
    ]

    normalized = normalized.dropna(subset=["Salary", "Name"])
    normalized = normalized[normalized["Positions"].map(len) > 0]
    normalized["Salary"] = normalized["Salary"].astype(int)
    normalized = normalized.reset_index(drop=True)
    normalized["PlayerId"] = normalized.index.astype(str)
    return normalized


@dataclass
class LineupResult:
    slots: dict = field(default_factory=dict)  # slot -> player row (dict)
    total_salary: int = 0
    total_projection: float = 0.0

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for slot in ROSTER_SLOTS:
            p = self.slots[slot]
            rows.append(
                {
                    "Slot": slot,
                    "Name": p["Name"],
                    "Team": p["Team"],
                    "Position": "/".join(sorted(p["Positions"])),
                    "Salary": p["Salary"],
                    "Projection": p["Projection"],
                }
            )
        return pd.DataFrame(rows)


def optimize_lineup(
    pool: pd.DataFrame,
    salary_cap: int = SALARY_CAP,
    locked_ids=None,
    excluded_ids=None,
    banned_player_sets=None,
    max_overlap=None,
) -> LineupResult | None:
    """Solve for a single optimal lineup.

    banned_player_sets / max_overlap: a list of previously-generated lineups
    (as sets of PlayerId). Each new lineup may share at most `max_overlap`
    players with any one of them, used to produce diverse multi-lineup output.
    """
    locked_ids = set(locked_ids or [])
    excluded_ids = set(excluded_ids or [])
    candidates = pool[~pool["PlayerId"].isin(excluded_ids)].reset_index(drop=True)

    if candidates.empty:
        return None

    prob = pulp.LpProblem("dk_nba_lineup", pulp.LpMaximize)

    x = {}
    for _, row in candidates.iterrows():
        pid = row["PlayerId"]
        for slot in ROSTER_SLOTS:
            if row["Positions"] & SLOT_ELIGIBILITY[slot]:
                x[(pid, slot)] = pulp.LpVariable(f"x_{pid}_{slot}", cat="Binary")

    # Objective: maximize total projected points.
    proj_by_id = dict(zip(candidates["PlayerId"], candidates["Projection"]))
    prob += pulp.lpSum(
        x[(pid, slot)] * proj_by_id[pid] for (pid, slot) in x
    )

    # Each slot filled by exactly one eligible player.
    for slot in ROSTER_SLOTS:
        prob += (
            pulp.lpSum(x[(pid, s)] for (pid, s) in x if s == slot) == 1,
            f"fill_{slot}",
        )

    # Each player used at most once across all slots.
    for pid in candidates["PlayerId"]:
        player_vars = [x[(pid, s)] for (p, s) in x if p == pid]
        if player_vars:
            prob += pulp.lpSum(player_vars) <= 1, f"once_{pid}"

    # Salary cap.
    salary_by_id = dict(zip(candidates["PlayerId"], candidates["Salary"]))
    prob += (
        pulp.lpSum(x[(pid, slot)] * salary_by_id[pid] for (pid, slot) in x)
        <= salary_cap
    )

    # Locked players must appear somewhere in the lineup.
    for pid in locked_ids:
        player_vars = [x[(p, s)] for (p, s) in x if p == pid]
        if not player_vars:
            # Locked player isn't a valid/eligible candidate; infeasible.
            return None
        prob += pulp.lpSum(player_vars) == 1, f"locked_{pid}"

    # At least two distinct games represented (DK Classic rule).
    games = candidates["Game"].unique().tolist()
    if len(games) > 1:
        game_indicator = {
            g: pulp.LpVariable(f"game_{g}", cat="Binary") for g in games
        }
        game_by_id = dict(zip(candidates["PlayerId"], candidates["Game"]))
        for g in games:
            players_in_game = [
                x[(pid, s)] for (pid, s) in x if game_by_id[pid] == g
            ]
            if players_in_game:
                prob += game_indicator[g] <= pulp.lpSum(players_in_game)
        prob += pulp.lpSum(game_indicator.values()) >= min(
            MIN_GAMES_REQUIRED, len(games)
        )

    # Diversity constraints against previously generated lineups.
    if banned_player_sets and max_overlap is not None:
        for i, prior in enumerate(banned_player_sets):
            overlap_vars = [x[(pid, s)] for (pid, s) in x if pid in prior]
            if overlap_vars:
                prob += pulp.lpSum(overlap_vars) <= max_overlap, f"diverse_{i}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    result = LineupResult()
    id_to_row = candidates.set_index("PlayerId").to_dict("index")
    for (pid, slot), var in x.items():
        if var.value() and var.value() > 0.5:
            row = dict(id_to_row[pid])
            row["PlayerId"] = pid
            result.slots[slot] = row
            result.total_salary += row["Salary"]
            result.total_projection += row["Projection"]

    return result


def generate_lineups(
    pool: pd.DataFrame,
    n_lineups: int,
    salary_cap: int = SALARY_CAP,
    locked_ids=None,
    excluded_ids=None,
    max_overlap: int = 5,
) -> list[LineupResult]:
    """Generate multiple diverse lineups, one at a time, each differing from
    every previous lineup by at least (8 - max_overlap) players."""
    lineups: list[LineupResult] = []
    prior_sets: list[set] = []

    for _ in range(n_lineups):
        result = optimize_lineup(
            pool,
            salary_cap=salary_cap,
            locked_ids=locked_ids,
            excluded_ids=excluded_ids,
            banned_player_sets=prior_sets,
            max_overlap=max_overlap,
        )
        if result is None:
            break
        lineups.append(result)
        player_ids = {row["PlayerId"] for row in result.slots.values()}
        prior_sets.append(player_ids)

    return lineups
