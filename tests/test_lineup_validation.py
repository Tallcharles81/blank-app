import pytest

from models.optimizer import (
    LineupValidationError,
    generate_lineups,
    validate_lineup,
)

# Regression coverage for a real incident: a lineup reported to the user
# labeled a FLEX slot - correctly filled by a second TE, which is legal
# under DraftKings' own rules (FLEX accepts RB/WR/TE) - as a second
# dedicated "TE" slot in a human-written summary table. That made a fully
# legal 2 RB/3 WR/1 TE/1 FLEX-as-TE roster read as an illegal 2 RB/2 TE
# roster missing an RB. The underlying optimizer output was correct the
# whole time; nothing had ever independently re-verified the REPORTED
# lineup against real DK roster rules before this existed.


def _make_lineup(roster):
    return {"roster": roster, "total_salary": sum(p["salary"] for _, p in roster), "total_points": 0}


def test_a_real_legal_lineup_including_a_flex_as_te_case_passes(engine, stable_slate):
    # Uses live data so this is the same shape as the real incident, not a
    # hand-built stand-in. Locks two real, legitimately-eligible starting
    # TEs (George Kittle, Trey McBride) in explicitly - since models/
    # playing_time_engine.py's real floor now correctly excludes most
    # real backup/committee TEs, the ceiling-maximizer no longer reaches
    # for a second TE on its own the way it used to on this slate, which
    # made this test flaky against live data rather than testing what it's
    # actually meant to (a real FLEX-as-TE roster validates as legal).
    lineups, _, _ = generate_lineups(
        stable_slate,
        num_lineups=1,
        projection_field="proj_ceiling",
        locked_player_ids=["44221112", "44221104"],  # George Kittle, Trey McBride
        engine=engine,
    )
    lineup = lineups[0]
    positions = [p["position"] for _, p in lineup["roster"]]
    assert positions.count("TE") == 2, "this regression test needs a real FLEX-as-TE case to be meaningful"

    validate_lineup(lineup)  # must not raise


def test_flex_mislabeled_as_a_second_te_slot_is_rejected(engine, stable_slate):
    lineups, _, _ = generate_lineups(
        stable_slate,
        num_lineups=1,
        projection_field="proj_ceiling",
        engine=engine,
    )
    roster = lineups[0]["roster"]
    mislabeled = [("TE" if slot == "FLEX" else slot, p) for slot, p in roster]

    with pytest.raises(LineupValidationError, match="slot labels"):
        validate_lineup(_make_lineup(mislabeled))


def test_duplicate_player_is_rejected(engine, stable_slate):
    lineups, _, _ = generate_lineups(
        stable_slate, num_lineups=1, projection_field="proj_ceiling", engine=engine
    )
    roster = lineups[0]["roster"]
    duplicated = roster[:-1] + [roster[0]]

    with pytest.raises(LineupValidationError, match="duplicate player"):
        validate_lineup(_make_lineup(duplicated))


def test_over_salary_cap_is_rejected(engine, stable_slate):
    lineups, _, _ = generate_lineups(
        stable_slate, num_lineups=1, projection_field="proj_ceiling", engine=engine
    )
    roster = lineups[0]["roster"]
    over_cap = [(slot, {**p, "salary": p["salary"] + 100_000} if slot == "QB" else p) for slot, p in roster]

    with pytest.raises(LineupValidationError, match="exceeds cap"):
        validate_lineup(_make_lineup(over_cap))


def test_missing_a_required_position_is_rejected(engine, stable_slate):
    lineups, _, _ = generate_lineups(
        stable_slate, num_lineups=1, projection_field="proj_ceiling", engine=engine
    )
    roster = lineups[0]["roster"]
    # Filter by real position, not slot label: FLEX could legally be filled
    # by an RB in an unseeded run, and dropping only "RB"-labeled slots
    # wouldn't necessarily bring the real RB count below 2 in that case.
    # Keeping at most one real RB regardless of which slot it's in is what
    # makes this test unambiguously "missing a required position."
    non_rb = [t for t in roster if t[1]["position"] != "RB"]
    one_rb = [t for t in roster if t[1]["position"] == "RB"][:1]
    missing_rb = non_rb + one_rb

    with pytest.raises(LineupValidationError, match="expected at least 2 RB"):
        validate_lineup(_make_lineup(missing_rb))


def test_generate_lineups_only_ever_returns_lineups_that_already_passed_validation(engine, stable_slate):
    # build_lineups_from_pool calls validate_lineup on every lineup before
    # it's appended to the results - this just confirms that wiring holds
    # for a real multi-lineup GPP run with exposure caps and stacking on,
    # not just a single lineup.
    lineups, _, _ = generate_lineups(
        stable_slate,
        num_lineups=10,
        projection_field="proj_ceiling",
        max_exposure=0.6,
        min_uniques=3,
        engine=engine,
    )
    assert len(lineups) == 10
    for lineup in lineups:
        validate_lineup(lineup)  # must not raise for any of them
