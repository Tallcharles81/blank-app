import numpy as np
import pytest

from models.ownership_model import (
    FEATURE_NAMES,
    _implied_gap,
    backtest_ownership_model,
    fit_ownership_model,
    predict_group_shares,
    predict_ownership,
)


def _synthetic_group(rng, n, slate_id, position="WR"):
    # Ownership generated from a known rule (share proportional to
    # projection^1.5 * salary^0.5) so the fit has a real, recoverable signal.
    proj = rng.uniform(2, 25, n)
    salary = rng.integers(3000, 9000, n)
    weights = proj ** 1.5 * (salary / 1000) ** 0.5
    pct = 100 * weights / weights.sum()
    players = [
        {"player_id": f"{slate_id}_{i}", "proj": float(proj[i]), "salary": int(salary[i]), "implied_gap": 0.0, "pct": float(pct[i])}
        for i in range(n)
    ]
    return {"slate_id": slate_id, "season": 2026, "week": 1, "position": position, "players": players}


def test_fit_recovers_a_known_ownership_rule_on_held_out_groups():
    rng = np.random.default_rng(0)
    train = [_synthetic_group(rng, 30, f"train{i}") for i in range(8)]
    test = _synthetic_group(rng, 30, "test")

    model = fit_ownership_model(train, l2=0.0)
    assert set(model["beta"]) == set(FEATURE_NAMES)
    assert model["beta"]["log_proj"] > 0
    assert model["position_totals"]["WR"] == pytest.approx(100.0)

    predicted = predict_group_shares(model, test["players"])
    actual = np.array([p["pct"] for p in test["players"]]) / 100
    assert predicted.sum() == pytest.approx(1.0)
    assert np.max(np.abs(predicted - actual)) < 0.01


def test_fit_raises_with_no_training_data():
    with pytest.raises(ValueError):
        fit_ownership_model([])


def test_implied_gap_uses_own_team_for_skill_players_and_opponent_for_dst():
    totals = {"KC": 28.0, "BUF": 20.0}
    avg = 24.0
    assert _implied_gap("WR", "KC", "BUF", totals, avg) == pytest.approx(4.0)
    # A DST facing a weak (low implied total) offense is a good spot - positive gap.
    assert _implied_gap("DST", "KC", "BUF", totals, avg) == pytest.approx(4.0)
    assert _implied_gap("DST", "BUF", "KC", totals, avg) == pytest.approx(-4.0)


def test_implied_gap_is_neutral_when_lines_are_missing():
    assert _implied_gap("QB", "KC", "BUF", {}, None) == 0.0
    assert _implied_gap("QB", "NOPE", "BUF", {"KC": 28.0}, 24.0) == 0.0


def test_predict_ownership_refuses_a_showdown_slate():
    with pytest.raises(ValueError, match="Showdown"):
        predict_ownership("dk_showdown_ind_kc_2026_09_20", model={"beta": {}, "position_totals": {}})


def test_backtest_beats_the_points_per_dollar_proxy_on_every_held_out_week(engine):
    # Real-data test against the Classic contests imported in the dev DB
    # (weeks 2 and 3 of 2026 when written). Direction only, not exact
    # numbers - those move as more real weeks are imported. If a future
    # week makes this fail, that's a real finding about the model, not a
    # flaky test.
    result = backtest_ownership_model(engine=engine)
    assert len(result["folds"]) >= 2
    for fold in result["folds"]:
        model_score, proxy_score = fold["overall"]["model"], fold["overall"]["proxy"]
        assert model_score["spearman_rho"] > proxy_score["spearman_rho"]
        assert model_score["mae_pct_points"] < proxy_score["mae_pct_points"]


def test_predict_ownership_on_a_real_slate_matches_position_totals(engine):
    model = {
        "beta": {"log_proj": 0.8, "log_salary_k": 0.6, "value_rank": 0.1, "proj_rank": 0.2, "implied_gap": 0.08},
        "position_totals": {"QB": 99.0, "RB": 240.0, "WR": 330.0, "TE": 120.0, "DST": 99.0},
    }
    predicted = predict_ownership("dk_sunday_2026_09_27", model=model, engine=engine)
    assert predicted
    assert all(v >= 0 for v in predicted.values())
    # Every position's predictions sum back to that position's total, so the
    # slate as a whole sums to the total roster slots' worth of ownership.
    assert sum(predicted.values()) == pytest.approx(sum(model["position_totals"].values()), abs=0.5)
