from models.optimizer import _load_player_pool, build_lineups_from_pool, validate_lineup


def test_ownership_cap_is_off_by_default(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    lineups, _ = build_lineups_from_pool(players, num_lineups=1, salary_cap=50000)
    assert len(lineups) == 1
    validate_lineup(lineups[0])


def test_ownership_cap_forces_a_real_lower_ownership_roster(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    ownership_values_by_id = {p["player_id"]: p["points"] / (p["salary"] / 1000.0) for p in players}

    uncapped_lineups, _ = build_lineups_from_pool(players, num_lineups=1, salary_cap=50000)
    uncapped_total = sum(ownership_values_by_id[p["player_id"]] for _, p in uncapped_lineups[0]["roster"])

    # A cap set at 70% of the real, unconstrained-optimal lineup's own total
    # ownership must force a genuinely different, lower-ownership roster -
    # not just accept the same one anyway.
    cap = 0.7 * uncapped_total
    capped_lineups, _ = build_lineups_from_pool(
        players, num_lineups=1, salary_cap=50000, ownership_values_by_id=ownership_values_by_id, ownership_cap=cap
    )
    assert len(capped_lineups) == 1
    validate_lineup(capped_lineups[0])
    capped_total = sum(ownership_values_by_id[p["player_id"]] for _, p in capped_lineups[0]["roster"])
    assert capped_total <= cap + 1e-6
    assert capped_total < uncapped_total


def test_ownership_cap_too_tight_raises_value_error(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    ownership_values_by_id = {p["player_id"]: p["points"] / (p["salary"] / 1000.0) for p in players}

    # A cap tighter than any real 9-man roster (even the 9 lowest-ownership
    # players in the whole pool) can possibly satisfy - must raise, never
    # silently return an illegal/partial lineup.
    nine_lowest = sorted(ownership_values_by_id.values())[:9]
    impossible_cap = sum(nine_lowest) / 2
    try:
        build_lineups_from_pool(
            players,
            num_lineups=1,
            salary_cap=50000,
            ownership_values_by_id=ownership_values_by_id,
            ownership_cap=impossible_cap,
        )
        assert False, "expected ValueError for an infeasible ownership cap"
    except ValueError:
        pass
