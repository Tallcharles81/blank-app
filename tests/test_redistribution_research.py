import numpy as np
import pandas as pd
import pytest

from research.redistribution_roles import add_trailing_usage, assign_roles
from research.redistribution_study import (
    benjamini_hochberg,
    predict_shares,
    rolling_prior,
    variance_components,
)

# Offline tests for the defensive-redistribution research module - the
# properties its conclusions depend on: no lookahead, shares that add up,
# correct FDR adjustment, and a variance test that finds a planted signal
# but doesn't invent one.


def test_rolling_prior_uses_only_strictly_prior_games():
    df = pd.DataFrame({"defteam": ["A"] * 4, "game_order": [1, 2, 3, 4], "v": [10.0, 20.0, 30.0, 40.0]})
    est, n = rolling_prior(df, "v", ["defteam"], window=8, k=0)
    assert est.tolist() == [0.0, 10.0, 15.0, 20.0]  # game 1 sees nothing; game 3 sees games 1-2 only
    assert n.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_rolling_prior_shrinks_toward_zero_with_k():
    df = pd.DataFrame({"defteam": ["A"] * 3, "game_order": [1, 2, 3], "v": [1.0, 1.0, 1.0]})
    est, _ = rolling_prior(df, "v", ["defteam"], window=8, k=2)
    assert est.iloc[2] == pytest.approx(2 / (2 + 2))


def test_role_assignment_ignores_same_game_production():
    # WR "B" has a monster game (15 targets) but a small trailing share; WR
    # "A" has the bigger trailing share and must still be WR1 for that game.
    rows = []
    for week, (a_tgt, b_tgt) in enumerate([(10, 2), (10, 2), (1, 15)], start=1):
        for pid, tgt in (("A", a_tgt), ("B", b_tgt)):
            rows.append({"game_id": f"g{week}", "posteam": "X", "player_id": pid, "position": "WR", "game_order": 202500 + week,
                         "tgt": tgt, "team_tgt": a_tgt + b_tgt, "car": 0, "team_car": 20, "snaps": 50, "team_snaps": 60,
                         "db_snaps": 30, "team_db": 35, "db": 0, "designed": 0, "i5_car": 0, "rec_yds": 0, "dk_points": 0})
    usage = assign_roles(add_trailing_usage(pd.DataFrame(rows)))
    week3 = usage[usage["game_id"] == "g3"].set_index("player_id")["role"]
    assert week3["A"] == "WR1"
    assert week3["B"] == "WR2"


def _toy_roles():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(40):
        exp = {"WR1": 0.3, "WR2": 0.2, "TE1": 0.2, "RB1": 0.2, "OTHER": 0.1}
        act = np.array(list(exp.values())) + rng.normal(0, 0.03, 5)
        act = np.clip(act, 0.01, None)
        act = act / act.sum()
        for (role, e), a in zip(exp.items(), act):
            rows.append({"game_id": f"g{g}", "posteam": "O", "defteam": ["D1", "D2"][g % 2], "game_order": 202000 + g, "role": role,
                         "tgt_share": a, "exp_tgt_share": e, "tgt_resid": a - e, "split": "train",
                         "off_favored_by": 0.0, "total_line": 44.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("level", ["none", "group", "role"])
def test_predicted_shares_sum_to_one_per_team_game(level):
    roles = _toy_roles()
    pf = predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", ["WR1", "WR2", "TE1", "RB1", "OTHER"], level, 8, 4)
    sums = pf.groupby(["game_id", "posteam"])["pred"].sum()
    assert np.allclose(sums.values, 1.0)


def test_benjamini_hochberg_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted: .01,.03,.04,.20 -> p*m/rank = .04,.06,.0533,.20 -> monotone from the top: .04,.0533,.0533,.20
    q = benjamini_hochberg(p)
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], abs=1e-6)


def _panel(tau, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2020, 2021):
        effects = {f"D{i}": rng.normal(0, tau) for i in range(24)}
        for d, eff in effects.items():
            for week in range(1, 17):
                rows.append({"defteam": d, "season": season, "week": week, "resid": eff + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_variance_components_finds_a_planted_defense_effect():
    vc = variance_components(_panel(tau=0.08, seed=1), "resid")
    assert vc["p_perm"] < 0.01
    assert vc["tau"] == pytest.approx(0.08, abs=0.03)


def test_variance_components_does_not_invent_a_signal_from_noise():
    vc = variance_components(_panel(tau=0.0, seed=2), "resid")
    assert vc["p_perm"] > 0.05
    assert vc["reliability_17_games"] < 0.3


def test_centering_keeps_a_league_wide_bias_out_of_defense_estimates():
    # Every defense sees WR1 run 5 share points above expectation - a
    # league-wide expectation bias, not anything defense-specific. Once the
    # calibration intercept absorbs it (center=True), the role model must
    # predict exactly what the no-defense model predicts.
    roles = _toy_roles()
    roles.loc[roles["role"] == "WR1", "tgt_share"] += 0.05
    roles["tgt_resid"] = roles["tgt_share"] - roles["exp_tgt_share"]
    calib = {"intercept": roles.groupby("role")["tgt_resid"].mean().to_dict(), "favored": {}, "total": {}}
    role_list = ["WR1", "WR2", "TE1", "RB1", "OTHER"]
    none = predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", role_list, "none", 8, 0, calib)
    role = predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", role_list, "role", 8, 0, calib, center=True)
    uncentered = predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", role_list, "role", 8, 0, calib, center=False)
    wr1 = lambda f: f[(f["role"] == "WR1") & (f["game_order"] > 202010)]["pred"].mean()  # noqa: E731
    assert abs(wr1(role) - wr1(none)) < 0.01
    assert wr1(uncentered) > wr1(none) + 0.02  # without centering the bias is double-counted
