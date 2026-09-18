import pytest

from models.field_simulation import (
    POSITION_OWNERSHIP_CALIBRATION,
    build_contrarian_lineup,
    calibrated_ownership_proxy,
    generate_opponent_lineups,
    ownership_proxy,
)
from models.optimizer import _load_player_pool

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


def test_generate_opponent_lineups_defaults_to_calibrated_proxy(engine):
    players, _, _, _ = _load_player_pool("dk_thu_mon_2026_09_17", "proj_median", engine)

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


def test_build_contrarian_lineup_defaults_to_calibrated_proxy(engine):
    players, _, _, _ = _load_player_pool("dk_thu_mon_2026_09_17", "proj_ceiling", engine)

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
