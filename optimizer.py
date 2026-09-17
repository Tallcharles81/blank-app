"""
optimizer.py
------------
Solves the DraftKings NBA classic-contest lineup problem as a linear
program: maximize total projected fantasy points subject to salary cap
and roster/position constraints.

DK Classic NBA roster:
    PG, SG, SF, PF, C, G (PG/SG), F (SF/PF), UTIL (any)
    Salary cap: $50,000

Usage:
    python optimizer.py --projections data/projections.csv --lineups 1

For multiple lineups (GPP tournament entries), use --lineups N and the
script will force diversity by excluding the top lineup's exact player
set from subsequent solves.
"""

import argparse
import pandas as pd
import pulp

SALARY_CAP = 50000
ROSTER_SLOTS = {
    "PG": ["PG"],
    "SG": ["SG"],
    "SF": ["SF"],
    "PF": ["PF"],
    "C": ["C"],
    "G": ["PG", "SG"],
    "F": ["SF", "PF"],
    "UTIL": ["PG", "SG", "SF", "PF", "C"],
}


def load_projections(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Name", "Salary", "Position", "PROJECTION"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"projections.csv is missing columns: {missing}")
    return df


def solve_lineup(df: pd.DataFrame, excluded_lineups=None) -> pd.DataFrame:
    excluded_lineups = excluded_lineups or []
    prob = pulp.LpProblem("DK_NBA_Lineup", pulp.LpMaximize)

    player_vars = {i: pulp.LpVariable(f"player_{i}", cat="Binary") for i in df.index}
    slot_vars = {
        (i, slot): pulp.LpVariable(f"slot_{i}_{slot}", cat="Binary")
        for i in df.index
        for slot in ROSTER_SLOTS
    }

    # Objective: maximize total projected points
    prob += pulp.lpSum(player_vars[i] * df.loc[i, "PROJECTION"] for i in df.index)

    # Salary cap
    prob += pulp.lpSum(player_vars[i] * df.loc[i, "Salary"] for i in df.index) <= SALARY_CAP

    # Exactly 8 players
    prob += pulp.lpSum(player_vars[i] for i in df.index) == 8

    # Each player assigned to at most one slot, and only if selected
    for i in df.index:
        prob += pulp.lpSum(slot_vars[(i, slot)] for slot in ROSTER_SLOTS) == player_vars[i]

    # Each slot filled exactly once, only by eligible positions
    for slot, eligible_positions in ROSTER_SLOTS.items():
        prob += pulp.lpSum(
            slot_vars[(i, slot)]
            for i in df.index
            if any(pos in str(df.loc[i, "Position"]).split("/") for pos in eligible_positions)
        ) == 1
        # Zero out ineligible assignments
        for i in df.index:
            if not any(pos in str(df.loc[i, "Position"]).split("/") for pos in eligible_positions):
                prob += slot_vars[(i, slot)] == 0

    # Exclude previously generated lineups (for multi-lineup diversity)
    for prev_lineup_indices in excluded_lineups:
        prob += pulp.lpSum(player_vars[i] for i in prev_lineup_indices) <= 7

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {pulp.LpStatus[status]}")

    selected_indices = [i for i in df.index if player_vars[i].value() == 1]
    lineup_slots = {}
    for i in selected_indices:
        for slot in ROSTER_SLOTS:
            if slot_vars[(i, slot)].value() == 1:
                lineup_slots[i] = slot

    result = df.loc[selected_indices].copy()
    result["SLOT"] = result.index.map(lineup_slots)
    slot_order = list(ROSTER_SLOTS.keys())
    result["SLOT_ORDER"] = result["SLOT"].map({s: n for n, s in enumerate(slot_order)})
    result = result.sort_values("SLOT_ORDER").drop(columns="SLOT_ORDER")
    return result, selected_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections", required=True)
    parser.add_argument("--lineups", type=int, default=1, help="Number of unique lineups to generate")
    args = parser.parse_args()

    df = load_projections(args.projections)
    excluded = []

    for n in range(args.lineups):
        lineup, indices = solve_lineup(df, excluded_lineups=excluded)
        excluded.append(indices)

        total_salary = lineup["Salary"].sum()
        total_projection = lineup["PROJECTION"].sum()

        print(f"\n=== Lineup {n + 1} ===")
        print(lineup[["SLOT", "Name", "Position", "Salary", "PROJECTION"]].to_string(index=False))
        print(f"Total Salary: ${total_salary:,} / ${SALARY_CAP:,}")
        print(f"Total Projection: {total_projection:.2f} DK points")


if __name__ == "__main__":
    main()
