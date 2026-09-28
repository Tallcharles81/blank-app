import json

import numpy as np
import pytest
from sqlalchemy import text

from models.simulation import (
    BRING_BACK_CORR,
    QB_PASS_CATCHER_CORR,
    SAME_TEAM_CORR,
    _load_players,
    _player_correlation,
    _simulate_player_scores,
    simulate_player_distributions,
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


def test_showdown_cpt_and_flex_rows_of_one_player_share_each_simulated_game(engine, showdown_simulation_slate):
    # Same player, two DK rows: every simulated world must give him ONE game.
    # This fixture stores identical ladders for both rows, so the draws must
    # match exactly (a real CPT row's ladder is 1.5x, scaling that one game).
    ids_by_name = showdown_simulation_slate
    lawrence = ids_by_name["Trevor Lawrence"]
    scores, _ = _simulate_player_scores([lawrence["CPT"], lawrence["FLEX"]], TEST_SLATE_ID, 2000, 7, engine)
    assert np.array_equal(scores[lawrence["CPT"]], scores[lawrence["FLEX"]])


# --- simulate_player_distributions -----------------------------------------

DISTRIBUTION_TEST_SLATE_ID = "TEST_SIMULATION_DISTRIBUTION"


@pytest.fixture
def distribution_test_slate(engine):
    # Same two real players/real projections as showdown_simulation_slate
    # above, reused under a dedicated Classic-shaped (plain QB/WR position
    # labels, no CPT/FLEX) test slate - simulate_player_distributions has no
    # Showdown-specific behavior to exercise, so a plain slate keeps this
    # fixture simpler (no crosswalk/real_position resolution needed).
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

    source_by_name = {r["name"]: dict(r) for r in source_rows}
    ids_by_name = {}
    try:
        with engine.begin() as conn:
            for r in source_rows:
                percentiles = r["proj_percentiles"]
                percentiles_json = json.dumps(percentiles) if isinstance(percentiles, dict) else percentiles
                pid = f"DIST_{r['player_id']}"
                ids_by_name[r["name"]] = pid
                conn.execute(
                    text(
                        """
                        INSERT INTO slate_player_pool (slate_id, player_id, name, position, salary, team, opponent)
                        VALUES (:slate_id, :player_id, :name, :position, :salary, :team, :opponent)
                        """
                    ),
                    {
                        "slate_id": DISTRIBUTION_TEST_SLATE_ID,
                        "player_id": pid,
                        "name": r["name"],
                        "position": r["position"],
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
                        "slate_id": DISTRIBUTION_TEST_SLATE_ID,
                        "player_id": pid,
                        "proj_floor": r["proj_floor"],
                        "proj_median": r["proj_median"],
                        "proj_ceiling": r["proj_ceiling"],
                        "proj_percentiles": percentiles_json,
                    },
                )
        yield ids_by_name, source_by_name
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM projections WHERE slate_id = :s"), {"s": DISTRIBUTION_TEST_SLATE_ID})
            conn.execute(text("DELETE FROM slate_player_pool WHERE slate_id = :s"), {"s": DISTRIBUTION_TEST_SLATE_ID})


def test_simulate_player_distributions_matches_own_stored_percentile_ladder(engine, distribution_test_slate):
    # The Gaussian copula preserves each player's own marginal distribution
    # exactly regardless of the joint correlation structure (see this
    # module's own docstring) - a single player's simulated p10/p50/p90 must
    # land close to their real, stored proj_percentiles ladder, not some
    # other number the correlated joint draw invented.
    ids_by_name, source_by_name = distribution_test_slate
    pid = ids_by_name["Trevor Lawrence"]
    ladder = source_by_name["Trevor Lawrence"]["proj_percentiles"]
    if isinstance(ladder, str):
        ladder = json.loads(ladder)

    result = simulate_player_distributions(
        DISTRIBUTION_TEST_SLATE_ID, player_ids=[pid], num_simulations=30000, seed=42, engine=engine
    )
    stats = result[pid]
    assert stats["p10"] == pytest.approx(float(ladder["10"]), abs=1.0)
    assert stats["p50"] == pytest.approx(float(ladder["50"]), abs=1.0)
    assert stats["p90"] == pytest.approx(float(ladder["90"]), abs=1.0)


def test_simulate_player_distributions_prob_at_least_is_monotonically_decreasing(engine, distribution_test_slate):
    ids_by_name, _ = distribution_test_slate
    pid = ids_by_name["Trevor Lawrence"]
    result = simulate_player_distributions(
        DISTRIBUTION_TEST_SLATE_ID, player_ids=[pid], num_simulations=5000, seed=7,
        thresholds=(10, 20, 30, 40), engine=engine,
    )
    probs = result[pid]["prob_at_least"]
    values = [probs[t] for t in sorted(probs)]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_simulate_player_distributions_reports_real_identity_fields(engine, distribution_test_slate):
    ids_by_name, source_by_name = distribution_test_slate
    pid = ids_by_name["Parker Washington"]
    result = simulate_player_distributions(
        DISTRIBUTION_TEST_SLATE_ID, player_ids=[pid], num_simulations=2000, seed=3, engine=engine
    )
    stats = result[pid]
    assert stats["name"] == "Parker Washington"
    assert stats["position"] == "WR"
    assert stats["salary"] == source_by_name["Parker Washington"]["salary"]


def test_simulate_player_distributions_defaults_to_every_player_with_a_projection(engine, distribution_test_slate):
    ids_by_name, _ = distribution_test_slate
    result = simulate_player_distributions(DISTRIBUTION_TEST_SLATE_ID, num_simulations=2000, seed=1, engine=engine)
    assert set(result.keys()) == set(ids_by_name.values())


def test_simulate_player_distributions_raises_for_a_slate_with_no_stored_projections(engine):
    with pytest.raises(ValueError):
        simulate_player_distributions("TEST_SLATE_DOES_NOT_EXIST_AT_ALL", engine=engine)
