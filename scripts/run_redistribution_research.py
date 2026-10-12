"""End-to-end run of the defensive-redistribution research
(research/redistribution_study.py). Downloads/caches nflverse data, runs
every stage, writes research/output/redistribution_results.json and the
report research/output/DK_EDGE_Defensive_Production_Redistribution_Research.md.

    python scripts/run_redistribution_research.py

Research only - nothing here touches the live optimizer or the database.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import redistribution_study as rs
from research.redistribution_data import warm_cache

OUT_DIR = Path(__file__).resolve().parent.parent / "research" / "output"


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (obj is None or not math.isfinite(float(obj))) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return {"slope": None, "se": None, "p": None, "n": len(x)}
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - y.mean() - b * xc
    se = math.sqrt((resid ** 2).sum() / (len(x) - 2) / (xc ** 2).sum())
    return {"slope": b, "se": se, "p": rs.normal_two_sided_p(b / se), "n": len(x)}


def by_split(frame, fn):
    return {name: fn(frame[frame["season"].isin(seasons)]) for name, seasons in
            [("train", rs.TRAIN_SEASONS), ("validation", rs.VALIDATION_SEASONS), ("test", rs.TEST_SEASONS)]}


def classify(train_q, train_effect, val, test):
    """Evidence category for a league-level pattern from its train FDR
    q-value and whether validation/test reproduce the same sign at p<.05."""
    if train_q is None or train_effect is None:
        return "INSUFFICIENT DATA"
    rep = [v for v in (val, test) if v and v.get("slope") is not None and v.get("p") is not None
           and np.sign(v["slope"]) == np.sign(train_effect) and v["p"] < 0.05]
    if train_q < 0.05 and len(rep) == 2:
        return "STRONG REPEATABLE SIGNAL"
    if train_q < 0.05 and len(rep) == 1:
        return "MODERATE SIGNAL"
    if train_q < 0.10 or len(rep) >= 1:
        return "WEAK SIGNAL"
    return "NO RELIABLE SIGNAL"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"design": {
        "warmup": rs.WARMUP_SEASONS, "train": rs.TRAIN_SEASONS, "validation": rs.VALIDATION_SEASONS, "test": rs.TEST_SEASONS,
        "windows": rs.WINDOWS, "shrinkage_ks": rs.SHRINKAGE_KS, "permutations": rs.N_PERMUTATIONS, "bootstrap": rs.N_BOOTSTRAP,
    }}
    results["data_failures"] = warm_cache(rs.ALL_SEASONS)
    ds = rs.build_dataset()
    roles = rs.add_residuals(ds["roles"])
    pg = ds["player_games"]
    passing = ds["passing"]
    train = roles[roles["season"].isin(rs.TRAIN_SEASONS)].copy()
    results["sample"] = {
        "games_by_season": roles.groupby("season")["game_id"].nunique().to_dict(),
        "team_games": int(roles[["game_id", "posteam"]].drop_duplicates().shape[0]),
        "player_games": len(pg),
        "mean_role_shares_train": train.groupby("role")[["tgt_share", "exp_tgt_share", "car_share", "exp_car_share"]].mean().to_dict(),
    }

    # ---- 1. Is there a between-defense signal? (train) ----------------------
    signal = {"role_target_share": {}, "group_target_share": {}, "within_group_share": {}, "carry_share": {}, "environment": {}}
    tests = []
    for role in ["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2", "RB3+"]:
        d = train[train["role"] == role]
        signal["role_target_share"][role] = {**rs.variance_components(d, "tgt_resid"), **rs.split_half_and_yoy(d, "tgt_resid"),
                                             "mean_share": float(d["tgt_share"].mean())}
        t = rs.per_defense_season_tests(d, "tgt_resid")
        tests.append(t.assign(family="role_target_share", role=role))
    grp_map = {r: g for g, rs_ in rs.GROUPS.items() for r in rs_}
    train["grp"] = train["role"].map(grp_map)
    g = train.groupby(["game_id", "posteam", "grp"]).agg(resid=("tgt_resid", lambda s: s.sum(min_count=1)), share=("tgt_share", "sum"),
                                                        defteam=("defteam", "first"), season=("season", "first"),
                                                        week=("week", "first")).reset_index()
    for grp in ["WR", "TE", "RB"]:
        d = g[g["grp"] == grp]
        signal["group_target_share"][grp] = {**rs.variance_components(d, "resid"), **rs.split_half_and_yoy(d, "resid"),
                                             "mean_share": float(d["share"].mean())}
        tests.append(rs.per_defense_season_tests(d, "resid").assign(family="group_target_share", role=grp))
    gs = train.groupby(["game_id", "posteam", "grp"])[["tgt_share", "exp_tgt_share"]].transform("sum")
    train["within_resid"] = train["tgt_share"] / gs["tgt_share"].replace(0, np.nan) - train["exp_tgt_share"] / gs["exp_tgt_share"].replace(0, np.nan)
    for role in ["WR1", "WR2", "WR3+", "TE1", "RB1", "RB2"]:
        d = train[train["role"] == role]
        signal["within_group_share"][role] = {**rs.variance_components(d, "within_resid"), **rs.split_half_and_yoy(d, "within_resid")}
    for role in ["RB1", "RB2", "QB"]:
        d = train[train["role"] == role]
        signal["carry_share"][role] = {**rs.variance_components(d, "car_resid"), **rs.split_half_and_yoy(d, "car_resid"),
                                       "mean_share": float(d["car_share"].mean())}
    pr = passing.copy()
    for col in ["pressure_rate", "sack_rate", "adot", "deep_rate", "cmp_rate", "int_rate", "scramble_rate", "man_rate", "two_high_rate"]:
        est, _ = rs.rolling_prior(pr, col, ["posteam"], 8, 0)
        pr[col + "_resid"] = pr[col] - est.where(est != 0, pr.groupby("season")[col].transform("mean"))
    for col in ["pressure_rate", "sack_rate", "adot", "deep_rate", "cmp_rate", "int_rate", "scramble_rate", "man_rate", "two_high_rate"]:
        d = pr[pr["season"].isin(rs.TRAIN_SEASONS)]
        signal["environment"][col] = {**rs.variance_components(d, col + "_resid"), **rs.split_half_and_yoy(d, col + "_resid"),
                                      "league_mean": float(d[col].mean())}
    all_tests = pd.concat(tests, ignore_index=True)
    all_tests["q"] = rs.benjamini_hochberg(all_tests["p"].values)
    signal["per_defense_season_tests"] = {
        fam: {"n_tests": len(t), "p_lt_05": int((t["p"] < 0.05).sum()), "expected_false_positives_at_05": 0.05 * len(t),
              "bh_q_lt_10": int((t["q"] < 0.10).sum()), "bh_q_lt_05": int((t["q"] < 0.05).sum())}
        for fam, t in all_tests.groupby("family")
    }
    results["signal"] = signal

    # ---- 2. Redistribution: where does a suppressed role's share go? --------
    redis = {}
    for sup in ["WR1", "WR2", "TE1", "RB1"]:
        redis[sup] = rs.conditional_redistribution(train, sup)
    # RB rushing suppression -> RB receiving?  (defense-season halves)
    rb = train[train["grp"] == "RB"].groupby(["game_id", "posteam"]).agg(
        car_resid=("car_resid", lambda s: s.sum(min_count=1)), tgt_resid=("tgt_resid", lambda s: s.sum(min_count=1)),
        defteam=("defteam", "first"), season=("season", "first"), week=("week", "first")).reset_index()
    halves = {p: rb[rb["week"] % 2 == p].groupby(["defteam", "season"])[["car_resid", "tgt_resid"]].mean() for p in (0, 1)}
    j1 = halves[1][["car_resid"]].join(halves[0][["tgt_resid"]]).dropna()
    j2 = halves[0][["car_resid"]].join(halves[1][["tgt_resid"]]).dropna()
    r1, p1, n1 = rs.pearson(j1["car_resid"], j1["tgt_resid"])
    r2, p2, n2 = rs.pearson(j2["car_resid"], j2["tgt_resid"])
    redis["RB_rushing_to_RB_receiving"] = {"cross_half_r": float(np.nanmean([r1, r2])), "p_half1": p1, "p_half2": p2, "n": int(min(n1, n2))}
    # RB1 -> RB2 carry split
    rbs = train[train["role"].isin(["RB1", "RB2"])].pivot_table(index=["game_id", "posteam", "defteam", "season", "week"], columns="role",
                                                               values="car_resid").reset_index().dropna()
    halves = {p: rbs[rbs["week"] % 2 == p].groupby(["defteam", "season"])[["RB1", "RB2"]].mean() for p in (0, 1)}
    j1 = halves[1][["RB1"]].join(halves[0][["RB2"]]).dropna()
    j2 = halves[0][["RB1"]].join(halves[1][["RB2"]]).dropna()
    r1, p1, n1 = rs.pearson(j1["RB1"], j1["RB2"])
    r2, p2, n2 = rs.pearson(j2["RB1"], j2["RB2"])
    redis["RB1_carries_to_RB2_carries"] = {"cross_half_r": float(np.nanmean([r1, r2])), "p_half1": p1, "p_half2": p2, "n": int(min(n1, n2))}
    red_p = [v["p"] for sup in ["WR1", "WR2", "TE1", "RB1"] for v in redis[sup].values()]
    results["redistribution"] = redis
    results["redistribution_fdr"] = {"n_tests": len(red_p), "p_lt_05": int(sum(p < 0.05 for p in red_p)),
                                     "bh_q_lt_10": int(np.nansum(rs.benjamini_hochberg(red_p) < 0.10))}

    # ---- 3. Out-of-sample share prediction: none vs generic vs role ---------
    # Primary comparison is CALIBRATED: every model gets the same per-role
    # league calibration (train mean residual) and defense estimates are
    # built from residuals net of it, so a defense model can't win by
    # correcting a league-wide expectation bias. Uncalibrated runs are kept
    # to show how large that artifact is.
    calib = rs.fit_role_calibration(roles, "tgt_resid", rs.TARGET_ROLES)
    calib_car = rs.fit_role_calibration(roles, "car_resid", rs.CARRY_ROLES)
    script = rs.fit_script_terms(roles, "tgt_resid", rs.TARGET_ROLES)
    script_car = rs.fit_script_terms(roles, "car_resid", rs.CARRY_ROLES)
    raw_tgt_res, raw_tgt_frames = rs.tune_and_compare_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", rs.TARGET_ROLES)
    raw_car_res, raw_car_frames = rs.tune_and_compare_shares(roles, "car_share", "exp_car_share", "car_resid", rs.CARRY_ROLES)
    tgt_res, tgt_frames = rs.tune_and_compare_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", rs.TARGET_ROLES, calib, center=True)
    car_res, car_frames = rs.tune_and_compare_shares(roles, "car_share", "exp_car_share", "car_resid", rs.CARRY_ROLES, calib_car, center=True)
    c = tgt_res["chosen"]["role"]
    tgt_frames["script_role"] = rs.predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", rs.TARGET_ROLES, "role",
                                                  c["window"], c["k"], script, center=True)
    tgt_frames["script_none"] = rs.predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", rs.TARGET_ROLES, "none", 8, 0, script)
    cc = car_res["chosen"]["role"]
    car_frames["script_role"] = rs.predict_shares(roles, "car_share", "exp_car_share", "car_resid", rs.CARRY_ROLES, "role",
                                                  cc["window"], cc["k"], script_car, center=True)
    car_frames["script_none"] = rs.predict_shares(roles, "car_share", "exp_car_share", "car_resid", rs.CARRY_ROLES, "none", 8, 0, script_car)
    for split in ["train", "validation", "test"]:
        for name in ["script_role", "script_none"]:
            tgt_res["scores"][split][name] = rs.share_loss(tgt_frames[name], "tgt_share", split)[0]
            car_res["scores"][split][name] = rs.share_loss(car_frames[name], "car_share", split)[0]
    results["oos_shares"] = {"targets": {k: v for k, v in tgt_res.items() if k != "tuning"},
                             "carries": {k: v for k, v in car_res.items() if k != "tuning"},
                             "targets_uncalibrated": {k: v for k, v in raw_tgt_res.items() if k != "tuning"},
                             "carries_uncalibrated": {k: v for k, v in raw_car_res.items() if k != "tuning"},
                             "tuning_targets": tgt_res["tuning"], "calibration_targets": calib, "calibration_carries": calib_car,
                             "script_terms_targets": script, "script_terms_carries": script_car}

    # ---- 4. Pressure / coverage / Vegas -------------------------------------
    results["pressure_play_level"] = rs.pressure_play_level(pg, rs.ALL_SEASONS[1:])
    pre = rs.defense_pre_game_rates(passing)
    for col in ["def_pressure_pre", "def_man_pre", "def_two_high_pre"]:
        pre[col + "_z"] = pre.groupby("season")[col].transform(lambda s: (s - s.mean()) / s.std())
    env = passing.merge(pre[["game_id", "posteam", "def_pressure_pre_z", "def_man_pre_z", "def_two_high_pre_z", "def_pressure_pre_n"]],
                        on=["game_id", "posteam"])
    for col in ["adot", "sack_rate", "deep_rate", "cmp_rate", "int_rate", "scramble_rate"]:
        est, _ = rs.rolling_prior(env, col, ["posteam"], 8, 0)
        env[col + "_resid"] = env[col] - est
    env = env[env["def_pressure_pre_n"] >= 4]
    rr = roles.merge(pre[["game_id", "posteam", "def_pressure_pre_z", "def_man_pre_z", "def_two_high_pre_z", "def_pressure_pre_n"]],
                     on=["game_id", "posteam"])
    rr = rr[rr["def_pressure_pre_n"] >= 4]
    predictive = {}
    for z in ["def_pressure_pre_z", "def_man_pre_z", "def_two_high_pre_z"]:
        for y in ["adot_resid", "sack_rate_resid", "deep_rate_resid", "cmp_rate_resid", "int_rate_resid", "scramble_rate_resid"]:
            predictive[f"{z} -> {y}"] = by_split(env, lambda d, z=z, y=y: slope(d[z], d[y]))
        for role in ["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2"]:
            predictive[f"{z} -> {role} target share"] = by_split(rr[rr["role"] == role], lambda d, z=z: slope(d[z], d["tgt_resid"]))
    cov = rs.coverage_pre_game(rs.ALL_SEASONS)
    wr1 = pg[pg["role"] == "WR1"].copy().sort_values(["player_id", "game_order"])
    for col in ["rec_yds", "tgt"]:
        wr1[f"p_{col}"] = wr1.groupby("player_id")[col].transform(lambda x: x.shift(1).rolling(16, min_periods=1).sum())
    wr1["ypt_pre"] = (wr1["p_rec_yds"] + 20 * 8.0) / (wr1["p_tgt"] + 20)
    wr1["yds_resid_per_tgt"] = np.where(wr1["tgt"] > 0, (wr1["rec_yds"] - wr1["tgt"] * wr1["ypt_pre"]) / wr1["tgt"].replace(0, np.nan), np.nan)
    wr1 = wr1.merge(cov, on=["defteam", "game_order"], how="left")
    wr1 = wr1.merge(roles[roles["role"] == "WR1"][["game_id", "posteam", "tgt_resid"]], on=["game_id", "posteam"])
    wr1 = wr1.merge(pre[["game_id", "posteam", "def_pressure_pre_z", "def_man_pre_z"]], on=["game_id", "posteam"], how="left")
    wr1["top_cb_z"] = wr1.groupby("season")["top_cb_ypt_pre"].transform(lambda s: (s - s.mean()) / s.std())
    wr1["team_cov_z"] = wr1.groupby("season")["def_ypt_allowed_pre"].transform(lambda s: (s - s.mean()) / s.std())
    for x in ["top_cb_z", "team_cov_z"]:
        for y in ["tgt_resid", "yds_resid_per_tgt"]:
            predictive[f"{x} -> WR1 {y}"] = by_split(wr1, lambda d, x=x, y=y: slope(d[x], d[y]))
    # "all four signals agree" subgroup: defense historically suppresses WR1 (prior-games role estimate < 0),
    # good top CB (z <= -0.5), high pressure (z >= 0.5), man-heavy (z >= 0.5)
    ch = tgt_res["chosen"]["role"]
    wr1_est_frame = rs.predict_shares(roles, "tgt_share", "exp_tgt_share", "tgt_resid", rs.TARGET_ROLES, "role", ch["window"], ch["k"])
    wr1_est = wr1_est_frame[wr1_est_frame["role"] == "WR1"][["game_id", "posteam", "pred", "exp_tgt_share"]]
    wr1 = wr1.merge(wr1_est, on=["game_id", "posteam"], how="left")
    wr1["def_suppresses_wr1"] = wr1["pred"] < wr1["exp_tgt_share"]
    agree = wr1["def_suppresses_wr1"] & (wr1["top_cb_z"] <= -0.5) & (wr1["def_pressure_pre_z"] >= 0.5) & (wr1["def_man_pre_z"] >= 0.5)
    stack = {}
    for name, seasons in [("train", rs.TRAIN_SEASONS), ("validation", rs.VALIDATION_SEASONS), ("test", rs.TEST_SEASONS)]:
        d = wr1[wr1["season"].isin(seasons)]
        a, o = d[agree.loc[d.index]], d[~agree.loc[d.index]]
        stack[name] = {"n_agree": len(a), "wr1_tgt_resid_agree": float(a["tgt_resid"].mean()) if len(a) else None,
                       "wr1_tgt_resid_other": float(o["tgt_resid"].mean()),
                       "wr1_yds_resid_per_tgt_agree": float(a["yds_resid_per_tgt"].mean()) if len(a) else None,
                       "wr1_yds_resid_per_tgt_other": float(o["yds_resid_per_tgt"].mean()),
                       "wr1_dk_agree": float(a["dk_points"].mean()) if len(a) else None, "wr1_dk_other": float(o["dk_points"].mean())}
    results["cb_signal_stack"] = stack
    for x in ["off_favored_by", "total_line"]:
        for role in ["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2"]:
            predictive[f"{x} -> {role} target share"] = by_split(roles[roles["role"] == role], lambda d, x=x: slope(d[x], d["tgt_resid"]))
        for role in ["RB1", "RB2", "QB"]:
            predictive[f"{x} -> {role} carry share"] = by_split(roles[roles["role"] == role], lambda d, x=x: slope(d[x], d["car_resid"]))
    # defense x script: does a defense's group effect look the same when the offense is favored vs an underdog?
    consistency = {}
    for dim, cond in [("favored_vs_underdog", lambda d: d["off_favored_by"] > 0), ("home_vs_away", lambda d: d["is_home"] == True),  # noqa: E712
                      ("high_vs_low_total", lambda d: d["total_line"] >= d.groupby("season")["total_line"].transform("median"))]:
        gg = train.groupby(["game_id", "posteam", "grp"]).agg(resid=("tgt_resid", lambda s: s.sum(min_count=1)), defteam=("defteam", "first"),
                                                             season=("season", "first"), off_favored_by=("off_favored_by", "first"),
                                                             is_home=("is_home", "first"), total_line=("total_line", "first")).reset_index()
        m = cond(gg)
        consistency[dim] = {}
        for grp in ["WR", "TE", "RB"]:
            a = gg[m & (gg["grp"] == grp)].groupby(["defteam", "season"])["resid"].mean()
            b = gg[~m & (gg["grp"] == grp)].groupby(["defteam", "season"])["resid"].mean()
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            r, p, n = rs.pearson(j["a"], j["b"])
            consistency[dim][grp] = {"r": r, "p": p, "n": n}
    results["consistency_across_conditions"] = consistency
    train_p = [v["train"]["p"] for v in predictive.values() if v["train"]["p"] is not None]
    q = rs.benjamini_hochberg(train_p)
    qi = iter(q)
    for key, v in predictive.items():
        if v["train"]["p"] is not None:
            v["train"]["q"] = float(next(qi))
        v["category"] = classify(v["train"].get("q"), v["train"]["slope"], v["validation"], v["test"])
    results["predictive_tests"] = predictive
    results["predictive_fdr"] = {"n_tests": len(train_p), "p_lt_05": int(sum(p < 0.05 for p in train_p)), "bh_q_lt_05": int((q < 0.05).sum())}

    # ---- 5. DFS translation -------------------------------------------------
    vol = rs.team_expected_volume(roles)
    share_models = {"none": tgt_frames["none"], "group": tgt_frames["group"], "role": tgt_frames["role"],
                    "script_role": tgt_frames["script_role"], "script_none": tgt_frames["script_none"],
                    "none_uncal": raw_tgt_frames["none"], "role_uncal": raw_tgt_frames["role"]}
    carry_models = {"none": car_frames["none"], "group": car_frames["group"], "role": car_frames["role"],
                    "script_role": car_frames["script_role"], "script_none": car_frames["script_none"],
                    "none_uncal": raw_car_frames["none"], "role_uncal": raw_car_frames["role"]}
    dfs = rs.player_dfs_frame(pg, share_models, carry_models, vol)
    mu_tgt, mu_car = rs.league_efficiency_calibration(dfs)
    for name in ["none", "group", "role", "script_role", "script_none"]:
        dfs[f"pred_dk_{name}"] = dfs[f"pred_tgt_{name}"] * (dfs["dk_per_tgt"] + mu_tgt) + dfs[f"pred_car_{name}"] * (dfs["dk_per_car"] + mu_car)
    eff, eff_k = rs.defense_receiving_efficiency(dfs, "role", mu_tgt, mu_car)
    dfs = dfs.merge(eff, on=["game_id", "defteam"], how="left")
    dfs["eff"] = dfs["eff"].fillna(0)
    for base, new in [("role", "role_matchup"), ("script_role", "script_role_matchup")]:
        dfs[f"pred_tgt_{new}"] = dfs[f"pred_tgt_{base}"]
        dfs[f"pred_car_{new}"] = dfs[f"pred_car_{base}"]
        dfs[f"pred_dk_{new}"] = dfs[f"pred_tgt_{base}"] * (dfs["dk_per_tgt"] + mu_tgt + dfs["eff"]) + dfs[f"pred_car_{base}"] * (dfs["dk_per_car"] + mu_car)
        dfs[f"pred_td_{new}"] = dfs[f"pred_td_{base}"]
    dfs["pred_dk_trailing"] = dfs["tr_dk_pg"].fillna(0)
    dfs["pred_tgt_trailing"] = dfs["prior_tgt"] / dfs["prior_games"].replace(0, np.nan)
    dfs["pred_car_trailing"] = dfs["prior_car"] / dfs["prior_games"].replace(0, np.nan)
    dfs["pred_td_trailing"] = dfs["pred_td_none"]
    models = ["trailing", "none_uncal", "role_uncal", "none", "group", "role", "role_matchup", "script_none", "script_role_matchup"]
    for mname in models:
        dfs[f"pred_tgt_{mname}"] = dfs[f"pred_tgt_{mname}"].fillna(0)
        dfs[f"pred_car_{mname}"] = dfs[f"pred_car_{mname}"].fillna(0)
    results["dfs"] = {"model_key": {
        "trailing": "reference: player's trailing DK points per game (no opportunity model)",
        "none_uncal": "reference: opportunity model without league calibration",
        "role_uncal": "reference: role model without league calibration (shows the calibration artifact)",
        "none": "MODEL A - calibrated opportunity model, offense/player expectation only, no defense",
        "group": "generic defense-vs-position (group-level target/carry share adjustment)",
        "role": "MODEL B - role-level redistribution",
        "role_matchup": "MODEL C - B + defense receiving-efficiency (DK per target allowed)",
        "script_none": "Vegas/game-script role terms only, no defense (isolates D's Vegas part)",
        "script_role_matchup": "MODEL D - C + Vegas/game-script role terms",
    }, "efficiency_k": eff_k, "league_efficiency_calibration": {"dk_per_target": mu_tgt, "dk_per_carry": mu_car},
        "results": rs.evaluate_dfs(dfs, models, baseline="none")}

    # ---- 6. QB: generic QB DvP vs pressure/coverage model ----------------------
    results["qb"] = rs.qb_models(pg, passing, pre)

    # ---- 7. Fingerprints (end of 2025) --------------------------------------
    k_by = {}
    for role, v in signal["role_target_share"].items():
        k_by[role] = ("tgt_resid", v["optimal_k"], v["sigma"], v["mean_share"])
    for col in ["pressure_rate", "sack_rate", "adot", "deep_rate"]:
        v = signal["environment"][col]
        k_by[f"env:{col}"] = (col + "_resid", v["optimal_k"], v["sigma"], v["league_mean"])
    fp = rs.defensive_fingerprints(roles, pr, 2025, k_by)
    results["fingerprints_2025"] = fp

    OUT_DIR.joinpath("redistribution_results.json").write_text(json.dumps(_clean(results), indent=1))
    print("wrote", OUT_DIR / "redistribution_results.json")


if __name__ == "__main__":
    main()
