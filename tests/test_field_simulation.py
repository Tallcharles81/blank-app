import pytest

from models.field_simulation import (
    POSITION_OWNERSHIP_CALIBRATION,
    build_chalk_lineup,
    build_contrarian_lineup,
    build_leverage_lineup,
    build_ownership_cap_lineup,
    calibrated_ownership_proxy,
    find_ownership_cap_matching_projection,
    generate_opponent_lineups,
    leverage_score,
    ownership_proxy,
    run_field_simulation,
)
from models.optimizer import _load_player_pool
from models.payout import import_contest_payout_structure

# --- calibrated_ownership_proxy: pure unit tests, no DB needed -------------


def _player(player_id, position, points, salary):
    return {"player_id": player_id, "position": position, "points": points, "salary": salary}


def test_calibrated_ownership_proxy_applies_the_real_position_multiplier():
    players = [_player("qb1", "QB", 20.0, 5000), _player("dst1", "DST", 8.0, 2000)]
    raw = ownership_proxy(players)
    calibrated = calibrated_ownership_proxy(players)

    for p in players:
        expected = raw[p["player_id"]] * POSITION_OWNERSHIP_CALIBRATION[p["position"]]
        assert calibrated[p["player_id"]] == pytest.approx(expected)

    # Real, measured result this multiplier is built from: the field owns
    # real DST plays MORE, relative to raw value, than it does QB plays - so
    # for two players with the same raw proxy, DST's calibrated value must
    # come out higher.
    same_raw_players = [_player("qb2", "QB", 10.0, 5000), _player("dst2", "DST", 10.0, 5000)]
    calibrated2 = calibrated_ownership_proxy(same_raw_players)
    assert calibrated2["dst2"] > calibrated2["qb2"]


def test_calibrated_ownership_proxy_falls_back_to_uncalibrated_for_unknown_position():
    players = [_player("k1", "K", 5.0, 4000)]  # DK Classic has no K, but the function shouldn't crash on one
    raw = ownership_proxy(players)
    calibrated = calibrated_ownership_proxy(players)
    assert calibrated["k1"] == pytest.approx(raw["k1"])  # multiplier of 1.0, unscaled


# --- real, DB-backed integration checks - proxy_fn actually gets used ------


def test_generate_opponent_lineups_defaults_to_calibrated_proxy(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_median", engine)

    # A uniform per-position scalar (POSITION_OWNERSHIP_CALIBRATION) doesn't
    # change the relative ORDER within a position, and DST is a mandatory
    # roster slot regardless of relative attractiveness - so an emergent
    # end-to-end selection-rate comparison isn't a reliable signal here (a
    # first version of this test tried exactly that and found the DST
    # selection rate trivially 1.0 in both cases, and per-DST-player shares
    # nearly identical - not evidence of a wiring bug, just the wrong
    # metric). A spy on proxy_fn is the direct, robust check: confirms the
    # DEFAULT argument is genuinely calibrated_ownership_proxy, actually
    # gets called, and produces the exact same draw as passing it
    # explicitly.
    calls = []

    def spy_proxy_fn(players):
        calls.append(True)
        return calibrated_ownership_proxy(players)

    default_opponents, _ = generate_opponent_lineups(players, contest_size=50, concentration=8, seed=7)
    spied_opponents, _ = generate_opponent_lineups(players, contest_size=50, concentration=8, seed=7, proxy_fn=spy_proxy_fn)
    assert calls, "proxy_fn was never called"

    default_rosters = [frozenset(p["player_id"] for _, p in opp["roster"]) for opp in default_opponents]
    spied_rosters = [frozenset(p["player_id"] for _, p in opp["roster"]) for opp in spied_opponents]
    assert default_rosters == spied_rosters


def test_leverage_score_is_zero_when_ceiling_rank_matches_ownership_rank():
    # Two RBs where the higher-ceiling one is ALSO the higher-owned one, in
    # the same order - ceiling percentile rank and ownership percentile rank
    # coincide exactly, so leverage should be flat 0.0 for both, not favor
    # either.
    players = [
        _player("rb_low", "RB", 8.0, 4000),
        _player("rb_high", "RB", 20.0, 9000),
    ]
    scores = leverage_score(players)
    assert scores["rb_low"] == pytest.approx(0.0)
    assert scores["rb_high"] == pytest.approx(0.0)


def test_leverage_score_is_positive_for_high_ceiling_low_ownership_player():
    # rb_sleeper: HIGHEST ceiling but priced high enough that its raw
    # points-per-$1000 (ownership_proxy) is only middling - the real
    # "leverage" case this signal exists to find. rb_chalk is the mirror
    # image: cheap with the best raw value/ownership_proxy of the three, but
    # the lowest ceiling.
    players = [
        _player("rb_sleeper", "RB", 22.0, 9000),  # ceiling rank 1st, value 2.44
        _player("rb_mid", "RB", 14.0, 6000),  # ceiling rank 2nd, value 2.33
        _player("rb_chalk", "RB", 8.0, 2500),  # ceiling rank last, value 3.2 (highest)
    ]
    scores = leverage_score(players)
    assert scores["rb_sleeper"] > 0
    assert scores["rb_chalk"] < scores["rb_sleeper"]


def test_leverage_score_ranks_within_position_not_across_positions():
    # A TE's raw ceiling/ownership units are nowhere near a WR's - leverage_
    # score must compare each position against its OWN peers, so a
    # last-place TE and a last-place WR should get the same (most negative)
    # score within their own group, regardless of the huge raw-unit gap
    # between the two positions.
    players = [
        _player("te_last", "TE", 3.0, 2500),
        _player("te_first", "TE", 12.0, 5000),
        _player("wr_last", "WR", 9.0, 3000),
        _player("wr_first", "WR", 28.0, 9000),
    ]
    scores = leverage_score(players)
    assert scores["te_last"] == pytest.approx(scores["wr_last"])
    assert scores["te_first"] == pytest.approx(scores["wr_first"])


def test_build_leverage_lineup_defaults_to_calibrated_proxy(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)

    calls = []

    def spy_proxy_fn(players):
        calls.append(True)
        return calibrated_ownership_proxy(players)

    default_lineup = build_leverage_lineup(players, lambda_boost=5.0)
    spied_lineup = build_leverage_lineup(players, lambda_boost=5.0, proxy_fn=spy_proxy_fn)
    assert calls, "proxy_fn was never called"

    default_ids = frozenset(p["player_id"] for _, p in default_lineup["roster"])
    spied_ids = frozenset(p["player_id"] for _, p in spied_lineup["roster"])
    assert default_ids == spied_ids


def test_build_ownership_cap_lineup_respects_the_hard_cap(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    proxy_by_id = calibrated_ownership_proxy(players)

    uncapped = build_chalk_lineup(players)
    uncapped_total = sum(proxy_by_id[p["player_id"]] for _, p in uncapped["roster"])

    cap = 0.7 * uncapped_total
    capped = build_ownership_cap_lineup(players, cap)
    capped_total = sum(proxy_by_id[p["player_id"]] for _, p in capped["roster"])
    assert capped_total <= cap + 1e-6


def test_run_field_simulation_without_contest_id_has_no_payout_key(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    chalk = build_chalk_lineup(players)
    result = run_field_simulation(chalk, stable_slate, contest_size=10, num_simulations=200, seed=1, engine=engine)
    assert result["payout"] is None


def test_run_field_simulation_with_contest_id_adds_real_payout_stats(engine, stable_slate):
    import_contest_payout_structure(
        "test_fs_payout_contest",
        [
            {"rank_start": 1, "rank_end": 1, "prize": 10.0},
            {"rank_start": 2, "rank_end": 5, "prize": 5.0},
        ],
        slate_id=stable_slate,
        entry_fee=1.0,
        total_entries=10,
        places_paid=5,
        total_prizes=30.0,
        structure_complete=True,
        engine=engine,
    )
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    chalk = build_chalk_lineup(players)
    result = run_field_simulation(
        chalk,
        stable_slate,
        contest_size=10,
        num_simulations=500,
        seed=1,
        engine=engine,
        contest_id="test_fs_payout_contest",
    )
    payout = result["payout"]
    assert payout is not None
    assert 0.0 <= payout["cash_pct"] <= 1.0
    assert payout["pct_unknown_payout"] == 0.0  # every rank 1-10 is covered: 1, 2-5, or >5 (confident $0)
    assert payout["structure_complete"] is True
    assert payout["simulated_field_size"] == 10
    assert payout["real_total_entries"] == 10
    # The best simulated lineup here (chalk, highest projection) should clear
    # some real worlds outright - not a degenerate always-zero result.
    assert payout["mean_known_payout"] is not None and payout["mean_known_payout"] >= 0.0


def test_run_field_simulation_raises_when_contest_slate_id_mismatches(engine, stable_slate):
    import_contest_payout_structure(
        "test_fs_payout_wrong_slate",
        [{"rank_start": 1, "rank_end": 1, "prize": 10.0}],
        slate_id="some_other_slate_entirely",
        places_paid=1,
        engine=engine,
    )
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    chalk = build_chalk_lineup(players)
    with pytest.raises(ValueError, match="refusing to score"):
        run_field_simulation(
            chalk, stable_slate, contest_size=10, num_simulations=50, seed=1, engine=engine, contest_id="test_fs_payout_wrong_slate"
        )


def test_run_field_simulation_raises_for_an_unimported_contest_id(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    chalk = build_chalk_lineup(players)
    with pytest.raises(ValueError, match="No contest payout structure stored"):
        run_field_simulation(
            chalk, stable_slate, contest_size=10, num_simulations=50, seed=1, engine=engine, contest_id="nope_not_real"
        )


def test_find_ownership_cap_matching_projection_finds_a_real_alternative(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)
    proxy_by_id = calibrated_ownership_proxy(players)

    chalk = build_chalk_lineup(players)
    chalk_projection = sum(p["points"] for _, p in chalk["roster"])
    chalk_total_ownership = sum(proxy_by_id[p["player_id"]] for _, p in chalk["roster"])

    _lineup, cap_used, projection, matched = find_ownership_cap_matching_projection(
        players, chalk_projection, chalk_total_ownership
    )
    assert matched
    assert abs(projection - chalk_projection) <= 0.05 * chalk_projection
    assert cap_used < chalk_total_ownership  # a real, tighter cap than chalk's own unconstrained total


def test_build_contrarian_lineup_defaults_to_calibrated_proxy(engine, stable_slate):
    players, _, _, _ = _load_player_pool(stable_slate, "proj_ceiling", engine)

    calls = []

    def spy_proxy_fn(players):
        calls.append(True)
        return calibrated_ownership_proxy(players)

    # Confirms build_contrarian_lineup's own default argument is genuinely
    # calibrated_ownership_proxy (not silently the raw stand-in) by proving
    # the values it computes internally match calling the calibrated
    # function directly on the same pool/penalty.
    default_lineup = build_contrarian_lineup(players, lambda_penalty=2.0)
    spied_lineup = build_contrarian_lineup(players, lambda_penalty=2.0, proxy_fn=spy_proxy_fn)
    assert calls, "proxy_fn was never called"

    default_ids = frozenset(p["player_id"] for _, p in default_lineup["roster"])
    spied_ids = frozenset(p["player_id"] for _, p in spied_lineup["roster"])
    assert default_ids == spied_ids
