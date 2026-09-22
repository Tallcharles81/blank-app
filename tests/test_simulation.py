import json

import pytest
from sqlalchemy import text

from models.simulation import (
    BRING_BACK_CORR,
    QB_PASS_CATCHER_CORR,
    SAME_TEAM_CORR,
    _load_players,
    _player_correlation,
)

# Regression coverage for a real gap found by a fresh audit late in this
# session: _player_correlation - the function every Monte Carlo simulation
# in this codebase (simulate_lineups, select_best_by_simulation,
# generate_simulation_selected_lineup's win-rate ranking, field_simulation's
# run_field_simulation/run_validation_comparison) depends on for real
# stacking correlation - read p["position"] directly. On a Showdown slate,
# DK labels every row's position "CPT"/"FLEX" regardless of the real
# player's football position (data/player_crosswalk.py's SHOWDOWN_PSEUDO_
# POSITIONS), which silently collapsed every real correlation (a QB and his
# own top pass-catcher, a real bring-back) to the generic same-team/no-
# correlation fallback - confirmed for real: a QB+same-team-WR pair labeled
# CPT/FLEX came back with correlation 0.0136 (SAME_TEAM_CORR) instead of the
# real 0.3284 (QB_PASS_CATCHER_CORR), a ~24x understatement, in the exact
# format (Showdown) where correlation is the single biggest strategic lever.
# Fixed by resolving each player's real_position from their own history
# before correlation is ever computed (_load_players/_resolve_real_
# positions), the same real-position-resolution pattern already used by
# data/pre_lock_check.py's hard_role_exclusions/build_lineup_role_checklist
# for the same underlying DK data quirk.


def _player(player_id, real_position, team, opponent="OPP"):
    return {"player_id": player_id, "real_position": real_position, "team": team, "opponent": opponent}


def test_player_correlation_uses_real_position_not_raw_position():
    # Two players whose real_position is QB/WR but whose (irrelevant here)
    # raw "position" field is never read by this function at all - proves
    # the function is driven by real_position, not by a "position" key that
    # happens to also be present.
    qb = {"player_id": "qb1", "real_position": "QB", "team": "JAX", "opponent": "DEN"}
    wr = {"player_id": "wr1", "real_position": "WR", "team": "JAX", "opponent": "DEN"}
    assert _player_correlation(qb, wr) == QB_PASS_CATCHER_CORR


def test_player_correlation_showdown_shaped_qb_and_pass_catcher_is_not_flattened():
    # The exact real regression: a QB and his own team's pass-catcher,
    # labeled the way a real Showdown row would be (real_position resolved
    # to QB/WR despite whatever raw DK position label existed), must still
    # get the real, strong QB_PASS_CATCHER_CORR - not silently fall through
    # to the generic same-team correlation the way raw "CPT"/"FLEX" would.
    qb = _player("SD_CPT_QB", "QB", "JAX")
    wr = _player("SD_FLEX_WR", "WR", "JAX")
    corr = _player_correlation(qb, wr)
    assert corr == QB_PASS_CATCHER_CORR
    assert corr != SAME_TEAM_CORR


def test_player_correlation_showdown_shaped_bring_back_is_not_erased():
    qb = _player("SD_CPT_QB", "QB", "JAX", opponent="DEN")
    opp_wr = _player("SD_FLEX_OPP_WR", "WR", "DEN", opponent="JAX")
    corr = _player_correlation(qb, opp_wr)
    assert corr == BRING_BACK_CORR
    assert corr != 0.0


def test_player_correlation_unresolvable_real_position_degrades_safely():
    # A player with no crosswalk match / no recorded history at all resolves
    # to real_position=None (see _resolve_real_positions) - must not crash,
    # and correctly falls through to the generic same-team/cross-team
    # fallback rather than matching any specific position-pair branch.
    known = _player("known1", "RB", "JAX")
    unknown = _player("unknown1", None, "JAX")
    assert _player_correlation(known, unknown) == SAME_TEAM_CORR


TEST_SLATE_ID = "TEST_SHOWDOWN_SIMULATION"


@pytest.fixture
def showdown_simulation_slate(engine):
    # Reuses two real players already loaded for dk_thu_mon_2026_09_17 with
    # real stored projections (Trevor Lawrence/QB and Parker Washington/WR,
    # a real same-team JAX pair already used in this session's own stacks),
    # duplicated under synthetic Showdown-labeled (CPT/FLEX) ids and a
    # dedicated test slate_id - the same insert/cleanup pattern
    # tests/test_showdown_gates.py already uses for this class of test.
    with engine.connect() as conn:
        source_rows = conn.execute(
            text(
                """
                SELECT sp.player_id, sp.name, sp.position, sp.team, sp.opponent, sp.salary,
                       proj.proj_floor, proj.proj_median, proj.proj_ceiling, proj.proj_percentiles
                FROM slate_player_pool sp
                JOIN projections proj ON proj.slate_id = sp.slate_id AND proj.player_id = sp.player_id
                WHERE sp.slate_id = 'dk_thu_mon_2026_09_17' AND sp.name IN ('Trevor Lawrence', 'Parker Washington')
                """
            )
        ).mappings().fetchall()
    assert len(source_rows) == 2, "expected real fixture data for these two players to still exist"

    ids_by_name = {}
    try:
        with engine.begin() as conn:
            for r in source_rows:
                percentiles = r["proj_percentiles"]
                percentiles_json = json.dumps(percentiles) if isinstance(percentiles, dict) else percentiles
                for prefix, pos in (("SD_CPT_", "CPT"), ("SD_FLEX_", "FLEX")):
                    pid = f"{prefix}{r['player_id']}"
                    ids_by_name.setdefault(r["name"], {})[pos] = pid
                    conn.execute(
                        text(
                            """
                            INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent)
                            VALUES (:slate_id, :player_id, :name, :position, :salary, :team, :opponent)
                            """
                        ),
                        {
                            "slate_id": TEST_SLATE_ID,
                            "player_id": pid,
                            "name": r["name"],
                            "position": pos,
                            "salary": r["salary"],
                            "team": r["team"],
                            "opponent": r["opponent"],
                        },
                    )
                    conn.execute(
                        text(
                            """
                            INSERT INTO projections
                                (slate_id, player_id, proj_floor, proj_median, proj_ceiling, proj_percentiles)
                            VALUES (:slate_id, :player_id, :proj_floor, :proj_median, :proj_ceiling, :proj_percentiles)
                            """
                        ),
                        {
                            "slate_id": TEST_SLATE_ID,
                            "player_id": pid,
                            "proj_floor": r["proj_floor"],
                            "proj_median": r["proj_median"],
                            "proj_ceiling": r["proj_ceiling"],
                            "proj_percentiles": percentiles_json,
                        },
                    )
        yield ids_by_name
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM projections WHERE slate_id = :s"), {"s": TEST_SLATE_ID})
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :s"), {"s": TEST_SLATE_ID})


def test_load_players_resolves_real_position_for_showdown_rows(engine, showdown_simulation_slate):
    ids_by_name = showdown_simulation_slate
    all_ids = [pid for positions in ids_by_name.values() for pid in positions.values()]

    players_by_id = _load_players(TEST_SLATE_ID, all_ids, engine)

    for pid in ids_by_name["Trevor Lawrence"].values():
        assert players_by_id[pid]["real_position"] == "QB"
    for pid in ids_by_name["Parker Washington"].values():
        assert players_by_id[pid]["real_position"] == "WR"


def test_full_pipeline_showdown_qb_pass_catcher_correlation_is_real(engine, showdown_simulation_slate):
    # End-to-end: through the actual _load_players function real
    # simulate_lineups calls, not just the isolated _player_correlation
    # unit test above - the real regression this session found.
    ids_by_name = showdown_simulation_slate
    all_ids = [pid for positions in ids_by_name.values() for pid in positions.values()]
    players_by_id = _load_players(TEST_SLATE_ID, all_ids, engine)

    qb = players_by_id[ids_by_name["Trevor Lawrence"]["CPT"]]
    wr = players_by_id[ids_by_name["Parker Washington"]["FLEX"]]
    assert _player_correlation(qb, wr) == QB_PASS_CATCHER_CORR
