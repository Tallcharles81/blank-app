import numpy as np
import pytest

import models.contest_selection as cs


def _lu(*ids):
    return {"roster": [("FLEX", {"player_id": i, "name": i}) for i in ids]}


def test_contest_rates_counts_worlds_inside_the_paid_share(monkeypatch):
    # 4 worlds, field of 4. Candidate A beats everyone in worlds 0-1 only;
    # candidate B is in the middle of the field every time.
    def fake_sim(lineups, slate_id, num_simulations, seed, engine):
        rows = [
            [100, 100, 0, 0],  # A
            [55, 55, 55, 55],  # B
            [10, 10, 10, 10], [50, 50, 50, 50], [60, 60, 60, 60], [90, 90, 90, 90],  # field
        ]
        return [{"scores": np.array(r, dtype=float)} for r in rows]

    monkeypatch.setattr(cs, "simulate_lineups", fake_sim)
    rates = cs.contest_rates([_lu("a"), _lu("b")], [_lu("f")] * 4, "x", paid_share=0.25, num_simulations=4)
    assert rates[0]["cash_rate"] == 0.5 and rates[0]["top1_rate"] == 0.5
    # B has 2 of 4 opponents above it every world -> 50%, outside the 25% paid.
    assert rates[1]["cash_rate"] == 0.0


def test_select_by_contest_cash_ranks_and_drops_excluded_names_from_the_field(monkeypatch):
    pool = [{"player_id": "q_cpt", "name": "Q", "points": 1}, {"player_id": "q_flex", "name": "Q", "points": 1},
            {"player_id": "r", "name": "R", "points": 1}]
    seen = {}
    monkeypatch.setattr(cs, "_load_player_pool", lambda slate_id, field, engine: (pool, {}, True, []))

    def fake_field(players, n, sigma, seed):
        seen["names"] = {p["name"] for p in players}
        return [_lu("f")]

    monkeypatch.setattr(cs, "noise_field", fake_field)
    monkeypatch.setattr(cs, "contest_rates", lambda c, f, s, p, **k: [
        {"cash_rate": 0.1, "top1_rate": 0.0, "mean_score": 1.0}, {"cash_rate": 0.3, "top1_rate": 0.0, "mean_score": 1.0}])
    ranked = cs.select_by_contest_cash([_lu("a"), _lu("b")], "x", 0.2, excluded_player_ids=["q_flex"], engine=object())
    assert [r["lineup"]["roster"][0][1]["player_id"] for r in ranked] == ["b", "a"]
    assert seen["names"] == {"R"}  # both of Q's rows left the field


def test_select_needs_two_candidates():
    with pytest.raises(ValueError):
        cs.select_by_contest_cash([_lu("a")], "x", 0.2, engine=object())


def test_noise_field_skips_infeasible_draws(monkeypatch):
    calls = {"n": 0}

    def fake_build(pool, num_lineups):
        calls["n"] += 1
        if calls["n"] % 2:
            raise ValueError("infeasible")
        return [_lu("x")], {}

    monkeypatch.setattr(cs, "build_lineups_from_pool", fake_build)
    out = cs.noise_field([{"player_id": "x", "points": 5.0}], 6, seed=1)
    assert len(out) == 3
