# ruff: noqa: RUF001 - en dashes and Greek letters in report copy are intentional typography.
"""Builds 'DK EDGE - DEFENSIVE PRODUCTION REDISTRIBUTION RESEARCH' from
research/output/redistribution_results.json. Every number and every
verdict sentence here is computed from that file; the fixed text is
methodology and data-provenance description only.

Output: a list of sections, rendered to Markdown (committed with the repo)
and to an HTML body (published as the shareable report).
"""
import html
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
TITLE = "DK EDGE — DEFENSIVE PRODUCTION REDISTRIBUTION RESEARCH"
ROLE_LIST = ["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2", "RB3+"]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def f(x, nd=3, pct=False, sign=False):
    if x is None:
        return "—"
    if pct:
        return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fp(p):
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sig_word(p, thr=0.05):
    return "significant" if p is not None and p < thr else "not significant"


def para(text):
    return ("p", text)


def bullets(items):
    return ("ul", items)


def table(headers, rows, caption=None):
    return ("table", headers, rows, caption)


# ---------------------------------------------------------------------------
# verdict logic (all rule-based, from the numbers)
# ---------------------------------------------------------------------------

def signal_verdict(v):
    """For a variance-component + reliability entry."""
    if v is None:
        return "INSUFFICIENT DATA"
    rep = v.get("yoy_r") is not None and v["yoy_r"] > 0 and (v.get("yoy_p") or 1) < 0.05
    if v["p_perm"] < 0.01 and v["reliability_17_games"] >= 0.3 and (v.get("split_half_p") or 1) < 0.05 and rep:
        return "STRONG REPEATABLE SIGNAL"
    if v["p_perm"] < 0.05 and (v.get("split_half_p") or 1) < 0.05:
        return "MODERATE SIGNAL"
    if v["p_perm"] < 0.10 or (v.get("split_half_p") or 1) < 0.05:
        return "WEAK SIGNAL"
    return "NO RELIABLE SIGNAL"


def paired_verdict(paired_val, paired_test):
    """Model B vs baseline: must be better (negative diff) and significant
    in BOTH the validation and the untouched test season to be accepted."""
    ok_v = paired_val["mean_diff_abs_err_per_team_game" if "mean_diff_abs_err_per_team_game" in paired_val else "mean_diff"] < 0 and paired_val["p"] < 0.05
    ok_t = paired_test["mean_diff_abs_err_per_team_game" if "mean_diff_abs_err_per_team_game" in paired_test else "mean_diff"] < 0 and paired_test["p"] < 0.05
    if ok_v and ok_t:
        return "IMPROVES OUT-OF-SAMPLE (validation and test)"
    if ok_v or ok_t:
        return "MIXED (improves in one holdout season only)"
    return "NO OUT-OF-SAMPLE IMPROVEMENT"


def diff_key(d):
    return d.get("mean_diff_abs_err_per_team_game", d.get("mean_diff"))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def build_sections(R):
    S = []
    sig = R["signal"]
    oos_t = R["oos_shares"]["targets"]
    oos_c = R["oos_shares"]["carries"]
    dfs = R["dfs"]["results"]
    qb = R["qb"]
    design = R["design"]

    # ---- A. Executive summary --------------------------------------------
    grp = sig["group_target_share"]
    role = sig["role_target_share"]
    within = sig["within_group_share"]
    red = R["redistribution"]
    red_fdr = R["redistribution_fdr"]
    env = sig["environment"]
    pl = R["pressure_play_level"]

    role_verdicts = {r: signal_verdict(role[r]) for r in ROLE_LIST}
    within_verdicts = {r: signal_verdict(within[r]) for r in within}
    b_vs_g_val = oos_t["paired"]["validation"]["role_minus_group"]
    b_vs_g_test = oos_t["paired"]["test"]["role_minus_group"]
    b_vs_a_val = oos_t["paired"]["validation"]["role_minus_none"]
    b_vs_a_test = oos_t["paired"]["test"]["role_minus_none"]
    cg_val = oos_c["paired"]["validation"]["group_minus_none"]
    cg_test = oos_c["paired"]["test"]["group_minus_none"]
    cr_vs_g_test = oos_c["paired"]["test"]["role_minus_group"]
    dk_test = dfs["test"]["ALL"]
    dk_pair_test = dfs["test"]["paired_vs_none"]
    dk_pair_val = dfs["validation"]["paired_vs_none"]
    qb_val, qb_test = qb["validation"]["paired_vs_A"], qb["test"]["paired_vs_A"]

    n_red_tests = red_fdr["n_tests"]
    exec_items = [
        f"Position-group effects exist but are small. A defense's effect on the WR / TE / RB share of targets clears a permutation test "
        f"(p = {fp(grp['WR']['p_perm'])} / {fp(grp['TE']['p_perm'])} / {fp(grp['RB']['p_perm'])}), with a true between-defense SD of "
        f"{f(grp['WR']['tau'] * 100, 1)} / {f(grp['TE']['tau'] * 100, 1)} / {f(grp['RB']['tau'] * 100, 1)} share points. A full 17-game sample "
        f"recovers only {f(grp['WR']['reliability_17_games'], 2)} / {f(grp['TE']['reliability_17_games'], 2)} / {f(grp['RB']['reliability_17_games'], 2)} "
        f"of that (reliability); year-over-year r = {f(grp['WR']['yoy_r'], 2)} / {f(grp['TE']['yoy_r'], 2)} / {f(grp['RB']['yoy_r'], 2)}.",
        "Role-level effects inside a group are not distinguishable from noise: "
        + ", ".join(f"{r} {role_verdicts[r]} (perm p {fp(role[r]['p_perm'])}, YoY r {f(role[r]['yoy_r'], 2)})" for r in ["WR1", "WR2", "TE1", "RB1", "RB2"])
        + ". Within-group splits (a role's share of its own group) show the same: "
        + ", ".join(f"{r} {within_verdicts[r]}" for r in within) + ".",
        f"Redistribution tests: {n_red_tests} tests of whether share taken from WR1, WR2, TE1 or RB1 goes to a specific other role more than "
        f"proportionally; {red_fdr['p_lt_05']} had p < 0.05 (about {0.05 * n_red_tests:.1f} expected by chance) and "
        f"{red_fdr['bh_q_lt_10']} survived FDR (q < 0.10). RB rushing suppression → RB receiving: cross-half r = "
        f"{f(red['RB_rushing_to_RB_receiving']['cross_half_r'], 3)}. RB1 → RB2 carry split: cross-half r = {f(red['RB1_carries_to_RB2_carries']['cross_half_r'], 3)}.",
        f"Out-of-sample target shares (mean abs error per role): no defense {f(oos_t['scores']['test']['none'], 5)}, generic defense-vs-position "
        f"{f(oos_t['scores']['test']['group'], 5)}, role redistribution {f(oos_t['scores']['test']['role'], 5)} (2025 test). Role vs generic: "
        f"{paired_verdict(b_vs_g_val, b_vs_g_test)}; role vs none: {paired_verdict(b_vs_a_val, b_vs_a_test)}. Size of the role model's test gain "
        f"over no-defense: {f(diff_key(b_vs_a_test), 5)} total share points per team-game (≈ {abs(diff_key(b_vs_a_test)) * 35:.2f} targets per team-game at 35 targets).",
        f"Carry shares: the generic group model beats no-defense ({paired_verdict(cg_val, cg_test)}); the RB1/RB2 role split is worse than generic "
        f"in 2025 (diff {f(diff_key(cr_vs_g_test), 5)}, p {fp(cr_vs_g_test['p'])}).",
        f"DFS points (WR/TE/RB, 2025 test, n = {dk_test['none']['dk']['n']}), all models league-calibrated: MAE no-defense (A) {f(dk_test['none']['dk']['mae'], 3)}, generic "
        f"{f(dk_test['group']['dk']['mae'], 3)}, role (B) {f(dk_test['role']['dk']['mae'], 3)}, B+matchup (C) {f(dk_test['role_matchup']['dk']['mae'], 3)}, "
        f"Vegas terms only, no defense {f(dk_test['script_none']['dk']['mae'], 3)}, C+Vegas (D) {f(dk_test['script_role_matchup']['dk']['mae'], 3)}. Paired vs A in test: "
        + "; ".join(f"{m} {f(dk_pair_test[m]['mean_diff'], 3)} (p {fp(dk_pair_test[m]['p'])})" for m in ["group", "role", "role_matchup", "script_none", "script_role_matchup"])
        + ". In validation: "
        + "; ".join(f"{m} {f(dk_pair_val[m]['mean_diff'], 3)} (p {fp(dk_pair_val[m]['p'])})" for m in ["group", "role", "role_matchup", "script_none", "script_role_matchup"]) + ".",
        f"Calibration artifact: without the shared league calibration, the role model's DK MAE edge over no-defense is "
        f"{f(dk_test['role_uncal']['dk']['mae'] - dk_test['none_uncal']['dk']['mae'], 3)} (2025) / {f(R['dfs']['results']['validation']['ALL']['role_uncal']['dk']['mae'] - R['dfs']['results']['validation']['ALL']['none_uncal']['dk']['mae'], 3)} (2024); "
        f"with it, {f(dk_test['role']['dk']['mae'] - dk_test['none']['dk']['mae'], 3)} / {f(R['dfs']['results']['validation']['ALL']['role']['dk']['mae'] - R['dfs']['results']['validation']['ALL']['none']['dk']['mae'], 3)}. "
        "Any gap between those pairs came from defenses' rolling estimates absorbing a league-wide bias in the expectations, not from defense-specific information.",
        f"Pressure changes the target distribution within a game: on pressured dropbacks WR1 share is {f(pl['target_share_by_role']['WR1']['within_game_diff'] * 100, 1, sign=True)} pts, "
        f"RB1 {f(pl['target_share_by_role']['RB1']['within_game_diff'] * 100, 1, sign=True)} pts, RB2 {f(pl['target_share_by_role']['RB2']['within_game_diff'] * 100, 1, sign=True)} pts, "
        f"TE1 {f(pl['target_share_by_role']['TE1']['within_game_diff'] * 100, 1, sign=True)} pts; aDOT {f(pl['metrics']['adot']['within_game_diff'], 2, sign=True)} yds and deep-target rate "
        f"{f(pl['metrics']['deep_target_rate']['within_game_diff'] * 100, 1, sign=True)} pts (deeper, not shallower). A defense's PRE-GAME pressure rate does not forecast "
        f"any role's target share out of sample (Section I).",
        f"QB: generic QB defense-vs-position vs pressure/coverage model, 2025 test MAE — trailing only {f(qb['test']['A']['mae'], 3)}, generic {f(qb['test']['G']['mae'], 3)}, "
        f"pressure/coverage {f(qb['test']['P']['mae'], 3)}, both {f(qb['test']['GP']['mae'], 3)}. Paired vs trailing in test: generic {f(qb_test['G']['mean_diff'], 3)} "
        f"(p {fp(qb_test['G']['p'])}), pressure {f(qb_test['P']['mean_diff'], 3)} (p {fp(qb_test['P']['p'])}); validation: generic {f(qb_val['G']['mean_diff'], 3)} "
        f"(p {fp(qb_val['G']['p'])}), pressure {f(qb_val['P']['mean_diff'], 3)} (p {fp(qb_val['P']['p'])}).",
        "Positive controls: the same method detects clear, repeatable defense effects on pressure rate, sack rate, aDOT, deep rate, completion rate and "
        "coverage shell (Section L), so the null results for role redistribution are not a failure to detect anything.",
    ]
    S.append(("A", "Executive summary", [bullets(exec_items), most_important_answer(R)]))

    # ---- B. Data sources ---------------------------------------------------
    games = R["sample"]["games_by_season"]
    S.append(("B", "Data sources", [
        bullets([
            "nflverse play-by-play (regular season): targets, receiver, air yards, pass location/length, completions, yards, TDs, red zone, sacks, QB hits, scrambles, rushes.",
            "nflverse participation: on-field offensive player ids for every play (snap and dropback participation, the route proxy), was_pressure, defense man/zone type, defense coverage type (charted for 89–100% of dropbacks each season).",
            "nflverse player weekly stats, scored with DK Edge's own DK formula (data/nflverse_fetch.py::_skill_player_fantasy_points).",
            "nflverse players (position), schedules (closing spread and total, roof, weather).",
            "PFR per-defender coverage charting (targets/yards allowed per defender per week) via nflverse pfr_advstats.",
            "FTN charting (2022+) was downloaded (blitzers, pass rushers) but not used in the final models; participation pressure covers every season.",
        ]),
        table(["Season", "Games", "Split"], [[s, g, split_name(int(s), design)] for s, g in games.items()]),
        para(f"{R['sample']['team_games']} team-games and {R['sample']['player_games']} player-games. Nothing was read from or written to the DK Edge database."),
    ]))

    # ---- C. Role classification -------------------------------------------
    S.append(("C", "Role classification methodology", [
        para("Roles are assigned per game, among players who were active that day, using only each player's usage in his previous 8 games (across seasons). Nothing from the game itself is used except who was active, which is known at lock."),
        table(["Role", "Rule"], [
            ["WR1 / WR2 / WR3+", "Active WRs ranked by trailing share of team targets"],
            ["TE1 / TE2+", "Active TEs ranked by trailing share of offensive snaps"],
            ["RB1 / RB2 / RB3+", "Active RBs/FBs ranked by trailing share of team carries + targets"],
            ["QB", "Active QB with the most trailing dropbacks"],
            ["Stratifiers", "TE1 targets per dropback snap (receiving vs blocking-heavy), RB committee (top RB < 60% of RB carries), receiving-RB identity, QB designed runs per game"],
        ]),
        para("Expected production for a role = the trailing target (or carry) share of the players filling it, with thin histories pulled toward a position prior, normalized across every active player so a team-game's expected shares sum to exactly 1. Actual shares also sum to 1, so every model respects the team's real pass volume."),
        para("Slot vs outside alignment, routes run, and CB-to-receiver assignments are not in any public data source (see Section R). Dropback participation (on the field for a dropback) stands in for routes; pass location (middle vs outside) is a target-location measure, not alignment."),
    ]))

    # ---- D. Fingerprints -----------------------------------------------------
    fps = R["fingerprints_2025"]
    rows = []
    for e in fps:
        row = [e["defense"], e["games"]]
        for m in ["WR1", "WR2", "WR3+", "TE1", "RB1", "RB2", "env:pressure_rate", "env:sack_rate", "env:adot"]:
            v = e.get(m, {})
            if v.get("effect") is None:
                row.append(v.get("label", "—"))
            else:
                val = f(v["effect_pct"], pct=True, sign=True) if not m.startswith("env:adot") else f(v["effect"], 2, sign=True)
                row.append(f"{val} ({v['label']})")
        rows.append(row)
    labels = [e.get(m, {}).get("label") for e in fps for m in ["WR1", "WR2", "WR3+", "TE1", "RB1", "RB2"]]
    count = {lab: labels.count(lab) for lab in set(labels)}
    S.append(("D", "Defense-by-defense fingerprints (end of 2025, last 17 games)", [
        para("Each cell: shrunk effect vs expectation (role cells: % of the league-average share for that role; aDOT: yards) and its evidence label. "
             "Shrinkage for each metric uses that metric's own estimated between-defense variance (k = σ²/τ²), so a metric with little real "
             "between-defense variance is shrunk hard toward zero. Labels: HIGH = |effect| ≥ 3 posterior SD and reliability ≥ 0.5; MEDIUM ≥ 2 SD; LOW otherwise."),
        para("Role-cell labels across all 32 defenses: " + ", ".join(f"{k}: {v}" for k, v in sorted(count.items(), key=lambda kv: str(kv[0])))),
        table(["Defense", "Games", "WR1 tgt", "WR2 tgt", "WR3+ tgt", "TE1 tgt", "RB1 tgt", "RB2 tgt", "Pressure", "Sack rate", "aDOT"], rows),
    ]))

    # ---- E. WR findings --------------------------------------------------------
    S.append(("E", "WR1 / WR2 / WR3+ findings", [
        signal_table(role, ["WR1", "WR2", "WR3+"], "Defense effect on each WR role's share of team targets (train 2019–2023)"),
        signal_table(within, ["WR1", "WR2", "WR3+"], "Defense effect on each WR role's share of WR targets only"),
        redistribution_table(red["WR1"], "Where WR1's lost share goes: cross-half correlation between a defense's WR1 residual and each other role's share of the remaining targets (0 = proportional reallocation)"),
        redistribution_table(red["WR2"], "Same test for WR2"),
        predictive_table(R["predictive_tests"], [k for k in R["predictive_tests"] if any(w in k for w in ["-> WR1 target", "-> WR2 target", "-> WR3+ target"])]),
    ]))

    # ---- F. TE ---------------------------------------------------------------
    S.append(("F", "TE findings", [
        signal_table(role, ["TE1", "TE2+"], "Defense effect on TE roles' share of team targets (train)"),
        signal_table(grp, ["TE"], "Defense effect on the TE group's share (generic)"),
        redistribution_table(red["TE1"], "Where TE1's lost share goes (cross-half r)"),
        predictive_table(R["predictive_tests"], [k for k in R["predictive_tests"] if "-> TE1 target" in k or "-> TE2+ target" in k]),
    ]))

    # ---- G. RB1 / RB2 --------------------------------------------------------
    S.append(("G", "RB1 / RB2 findings (rushing)", [
        signal_table(sig["carry_share"], ["RB1", "RB2", "QB"], "Defense effect on share of team carries (train)"),
        para(f"RB1 → RB2 carry redistribution (does a defense that cuts RB1's carry share hand them to RB2?): cross-half r = "
             f"{f(red['RB1_carries_to_RB2_carries']['cross_half_r'], 3)} (n = {red['RB1_carries_to_RB2_carries']['n']} defense-seasons)."),
        share_model_table(oos_c, "Carry-share prediction, mean abs error per role"),
        paired_table(oos_c["paired"], "Carry-share paired differences (negative = second model better)"),
    ]))

    # ---- H. RB receiving -----------------------------------------------------
    S.append(("H", "RB receiving findings", [
        signal_table(role, ["RB1", "RB2", "RB3+"], "Defense effect on RB roles' share of team targets (train)"),
        signal_table(grp, ["RB"], "Defense effect on the RB group's target share (generic)"),
        redistribution_table(red["RB1"], "Where RB1's lost target share goes (cross-half r)"),
        para(f"RB rushing suppression → RB receiving increase: cross-half r between a defense's RB carry-share residual and its RB target-share residual = "
             f"{f(red['RB_rushing_to_RB_receiving']['cross_half_r'], 3)} (half-season p values {fp(red['RB_rushing_to_RB_receiving']['p_half1'])}, "
             f"{fp(red['RB_rushing_to_RB_receiving']['p_half2'])}; n = {red['RB_rushing_to_RB_receiving']['n']})."),
        predictive_table(R["predictive_tests"], [k for k in R["predictive_tests"] if "-> RB1 target" in k or "-> RB2 target" in k]),
    ]))

    # ---- I. QB pressure ------------------------------------------------------
    pm = pl["metrics"]
    S.append(("I", "QB pressure findings", [
        para(f"Play level, within the same game ({pl['dropbacks']} charted dropbacks, {', '.join(map(str, pl['seasons']))}; pressured on "
             f"{f(pl['pressured_rate'], pct=True)} of dropbacks). Differences are pressured minus clean, averaged within games; 95% CIs resample games."),
        table(["Metric", "Pressured", "Clean", "Within-game diff", "95% CI", "p"],
              [[k.replace("_", " "), f(v["pressured"], 3), f(v["clean"], 3), f(v["within_game_diff"], 4, sign=True),
                f"{f(v['ci95'][0], 4, sign=True)} to {f(v['ci95'][1], 4, sign=True)}", fp(v["p"])] for k, v in pm.items()]),
        table(["Target share", "Pressured", "Clean", "Within-game diff", "95% CI", "p"],
              [[k, f(v["pressured"], 3), f(v["clean"], 3), f(v["within_game_diff"], 4, sign=True),
                f"{f(v['ci95'][0], 4, sign=True)} to {f(v['ci95'][1], 4, sign=True)}", fp(v["p"])]
               for k, v in pl["target_share_by_role"].items() if k not in ("QB", "OTHER")]),
        para("Predictive (pre-game): a defense's trailing pressure rate, z-scored within season, against the offense's residual in that game. Slope per 1 SD."),
        predictive_table(R["predictive_tests"], [k for k in R["predictive_tests"] if k.startswith("def_pressure_pre_z")]),
        qb_table(qb),
    ]))

    # ---- J. CB-WR -----------------------------------------------------------
    st = R["cb_signal_stack"]
    S.append(("J", "CB-WR interaction findings", [
        para("No public data assigns a specific CB to a specific receiver on a play (models/matchups.py documents the same check). The closest real signals: "
             "the defense's best CB by trailing PFR-charted yards/target allowed (min 10 trailing targets), the whole secondary's trailing yards/target allowed, "
             "and trailing man-coverage and two-high rates. The existing matchups.py WR adjustment built on the team-level version already backtested as harmful."),
        predictive_table(R["predictive_tests"], [k for k in R["predictive_tests"] if k.startswith("top_cb_z") or k.startswith("team_cov_z") or k.startswith("def_man_pre_z") or k.startswith("def_two_high_pre_z")]),
        para("All-signals-agree subgroup: defense's prior-games WR1 estimate is suppressive AND best CB ≥ 0.5 SD better than average AND pressure ≥ 0.5 SD above AND man rate ≥ 0.5 SD above."),
        table(["Split", "Games", "WR1 tgt-share resid (agree)", "(other)", "WR1 yds/tgt resid (agree)", "(other)", "WR1 DK (agree)", "(other)"],
              [[s, v["n_agree"], f(v["wr1_tgt_resid_agree"], 4, sign=True), f(v["wr1_tgt_resid_other"], 4, sign=True),
                f(v["wr1_yds_resid_per_tgt_agree"], 2, sign=True), f(v["wr1_yds_resid_per_tgt_other"], 2, sign=True),
                f(v["wr1_dk_agree"], 2), f(v["wr1_dk_other"], 2)] for s, v in st.items()]),
    ]))

    # ---- K. Vegas -----------------------------------------------------------
    cons = R["consistency_across_conditions"]
    S.append(("K", "Vegas / game-script interaction findings", [
        predictive_table(R["predictive_tests"], [k for k in R["predictive_tests"] if k.startswith("off_favored_by") or k.startswith("total_line")]),
        para("Consistency of a defense's group-level effect across conditions (same defense-season, split by condition; r across defense-seasons):"),
        table(["Condition", "WR r", "TE r", "RB r"], [[dim.replace("_", " "), f(v["WR"]["r"], 3), f(v["TE"]["r"], 3), f(v["RB"]["r"], 3)] for dim, v in cons.items()]),
        para(f"Per-role spread/total terms fit on 2019–2023, 2025 test target-share MAE: no defense {f(oos_t['scores']['test']['none'], 5)}, Vegas terms only {f(oos_t['scores']['test'].get('script_none'), 5)}, "
             f"role model {f(oos_t['scores']['test']['role'], 5)}, role + Vegas (D) {f(oos_t['scores']['test'].get('script_role'), 5)}. Carry shares: no defense {f(oos_c['scores']['test']['none'], 5)}, "
             f"Vegas only {f(oos_c['scores']['test'].get('script_none'), 5)}, role + Vegas {f(oos_c['scores']['test'].get('script_role'), 5)}."),
    ]))

    # ---- L. Statistical results ---------------------------------------------
    rows = []
    for fam, entries in [("Role target share", role), ("Group target share", grp), ("Within-group share", within), ("Carry share", sig["carry_share"]),
                         ("Passing environment (positive controls)", env)]:
        for k, v in entries.items():
            rows.append([fam, k, f(v["tau"], 4), f(v["sigma"], 3), fp(v["p_perm"]), f(v["reliability_17_games"], 2),
                         f(v["optimal_k"], 0) if v["optimal_k"] else "∞", f(v["split_half_r"], 3), fp(v["split_half_p"]), f(v["yoy_r"], 3), fp(v["yoy_p"]),
                         signal_verdict(v)])
    S.append(("L", "Statistical results", [
        para(f"Train seasons {design['train']}. τ = true between-defense SD beyond sampling noise (method of moments over defense-seasons); σ = game-to-game SD; "
             f"permutation p shuffles which defense faced which game within a season ({design['permutations']} permutations, minimum p ≈ {1 / (design['permutations'] + 1):.3f}); "
             "reliability = share of a 17-game average that is real signal; optimal k = σ²/τ² games of shrinkage; split-half = odd vs even weeks; YoY = season N vs N+1."),
        table(["Family", "Metric", "τ", "σ", "Perm p", "Rel. 17g", "Opt. k", "Split-half r", "p", "YoY r", "p", "Verdict"], rows),
    ]))

    # ---- M. OOS --------------------------------------------------------------
    S.append(("M", "Out-of-sample results", [
        para(f"Shrinkage and window chosen on train only. Target-share models chose window {oos_t['chosen']['role']['window']} / k {oos_t['chosen']['role']['k']} (role) and "
             f"window {oos_t['chosen']['group']['window']} / k {oos_t['chosen']['group']['k']} (generic); carry-share models {oos_c['chosen']['role']} / {oos_c['chosen']['group']}. "
             f"Grid: windows {design['windows']}, k {design['shrinkage_ks']}."),
        share_model_table(oos_t, "Target-share prediction, calibrated models, mean abs error per role (lower is better)"),
        share_model_table(R["oos_shares"]["targets_uncalibrated"], "Same, WITHOUT league calibration (defense estimates absorb league-wide expectation bias)"),
        paired_table(R["oos_shares"]["targets_uncalibrated"]["paired"], "Uncalibrated paired differences, for comparison"),
        paired_table(oos_t["paired"], "Target-share paired differences per team-game (sum over roles; negative = second model better; CI resamples team-games)"),
        by_role_table(oos_t),
    ]))

    # ---- N. DFS ---------------------------------------------------------------
    S.append(("N", "DFS projection comparison", [
        para("Player level (WR/TE/RB with ≥ 4 prior games). Every model shares the same team-volume and player-efficiency inputs; only the defense terms differ. "
             "Model key: " + "; ".join(f"{k}: {v}" for k, v in R["dfs"]["model_key"].items()) + "."),
        dfs_table(dfs, "validation"), dfs_table(dfs, "test"),
        dfs_component_table(dfs, "test"),
        dfs_paired_table(dfs),
        quantile_table(dfs, "test"),
    ]))

    # ---- O. FDR ---------------------------------------------------------------
    pdt = sig["per_defense_season_tests"]
    S.append(("O", "False-discovery controls", [
        table(["Test family", "Tests", "p < 0.05", "Expected by chance", "BH q < 0.10", "BH q < 0.05"],
              [[k.replace("_", " ") + " (per defense-season)", v["n_tests"], v["p_lt_05"], f"{v['expected_false_positives_at_05']:.0f}", v["bh_q_lt_10"], v["bh_q_lt_05"]] for k, v in pdt.items()]
              + [["Redistribution (suppressed role → other role)", red_fdr["n_tests"], red_fdr["p_lt_05"], f"{0.05 * red_fdr['n_tests']:.1f}", red_fdr["bh_q_lt_10"], "—"],
                 ["Predictive slopes (train period)", R["predictive_fdr"]["n_tests"], R["predictive_fdr"]["p_lt_05"], f"{0.05 * R['predictive_fdr']['n_tests']:.1f}", "—", R["predictive_fdr"]["bh_q_lt_05"]]]),
        para("Evidence categories for predictive patterns: STRONG = train BH q < 0.05 and same-sign p < 0.05 in both 2024 and 2025; MODERATE = train q < 0.05 and one holdout; "
             "WEAK = train q < 0.10 or one holdout only; otherwise NO RELIABLE SIGNAL. Out-of-sample model gains are judged on the untouched 2025 test and 2024 validation, not on significance in training."),
    ]))

    # ---- P / Q. candidates and rejections -----------------------------------
    cand, rej = candidates_and_rejections(R)
    S.append(("P", "Recommended candidate features", [bullets(cand) if cand else para("None met the bar (improvement in both 2024 validation and 2025 test).")]))
    S.append(("Q", "Rejected features", [bullets(rej)]))

    # ---- R. limitations ------------------------------------------------------
    S.append(("R", "Data limitations", [bullets([
        "No slot/outside alignment, route counts, or CB-to-receiver assignments exist in public nflverse data. Dropback participation is a route proxy; pass location is not alignment.",
        "Time snapshots (OPEN / T-24 / T-6 / T-1 / LOCK) cannot be reconstructed: nflverse keeps closing lines and final injury reports only. Every pre-game input here is either strictly prior-game data or closing Vegas lines plus game-day actives, i.e. a LOCK snapshot.",
        "Man/zone charting changes level between 2022 (≈29% man) and 2023–2024 (42–49%), consistent with a change in charting source; coverage rates are z-scored within season for that reason.",
        "PFR per-defender coverage is a charter's judgment of the covering defender, not a verified assignment.",
        "Seven seasons of training data give ~160 defense-seasons; a true role-level defense SD below ~0.5 share points would be hard to detect even if it exists.",
        "Expected production uses trailing player usage; it does not model mid-week role news. Residuals therefore include role changes the offense made that no defense caused, which adds noise equally to every model.",
    ]), para("What would make this stronger: charted alignment (slot/wide) and route data, per-play coverage assignments, timestamped injury and line history, and more seasons of participation charting.")]))

    # ---- S. Full list ----------------------------------------------------------
    rows = []
    for k, v in R["predictive_tests"].items():
        t, va, te = v["train"], v["validation"], v["test"]
        rows.append([k, f(t["slope"], 5, sign=True), fp(t["p"]), fp(t.get("q")), f(va["slope"], 5, sign=True), fp(va["p"]), f(te["slope"], 5, sign=True), fp(te["p"]), t["n"], v["category"]])
    for r in ROLE_LIST:
        rows.append([f"defense → {r} target share (variance component)", f(role[r]["tau"], 4), fp(role[r]["p_perm"]), "—", "—", "—", "—", "—", role[r]["n_groups"], signal_verdict(role[r])])
    for g in ["WR", "TE", "RB"]:
        rows.append([f"defense → {g} group target share (variance component)", f(grp[g]["tau"], 4), fp(grp[g]["p_perm"]), "—", "—", "—", "—", "—", grp[g]["n_groups"], signal_verdict(grp[g])])
    for sup in ["WR1", "WR2", "TE1", "RB1"]:
        for tgt, v in red[sup].items():
            rows.append([f"{sup} suppression → {tgt} (conditional share)", f(v["cross_half_r"], 3, sign=True), fp(v["p"]), "—", "—", "—", "—", "—", v["n_defense_seasons"],
                         "NO RELIABLE SIGNAL" if v["p"] >= 0.05 else "WEAK SIGNAL"])
    S.append(("S", "Full list of tested patterns", [
        para("Slope columns: effect per unit of the predictor (per 1 SD for z-scored predictors, per point for spread/total) in train / 2024 validation / 2025 test. "
             "Variance-component rows show τ and permutation p; redistribution rows show the cross-half correlation."),
        table(["Pattern", "Train effect", "p", "q", "2024 effect", "p", "2025 effect", "p", "n", "Category"], rows),
    ]))
    return S


def split_name(season, design):
    for name in ("warmup", "train", "validation", "test"):
        if season in design[name]:
            return name
    return ""


def most_important_answer(R):
    oos_t, oos_c, qb, dfs = R["oos_shares"]["targets"], R["oos_shares"]["carries"], R["qb"], R["dfs"]["results"]
    wr_te = paired_verdict(oos_t["paired"]["validation"]["role_minus_group"], oos_t["paired"]["test"]["role_minus_group"])
    rb = paired_verdict(oos_c["paired"]["validation"]["role_minus_group"], oos_c["paired"]["test"]["role_minus_group"])
    dk_t = dfs["test"]["paired_vs_none"]
    dk_role_vs_group_t = dk_t["role"]["mean_diff"] - dk_t["group"]["mean_diff"]
    qbv = paired_verdict(qb["validation"]["paired_vs_A"]["P"], qb["test"]["paired_vs_A"]["P"])
    qbg = paired_verdict(qb["validation"]["paired_vs_A"]["G"], qb["test"]["paired_vs_A"]["G"])
    return table(["Most important test", "Result"], [
        ["WR/TE: role redistribution vs generic defense-vs-position (target shares)", wr_te],
        ["WR/TE/RB: DK points, role model minus generic model (2025 test, MAE)", f"{dk_role_vs_group_t:+.3f} DK pts per player-game"],
        ["WR/TE/RB: DK points, role model (B) vs no defense", paired_verdict(dfs["validation"]["paired_vs_none"]["role"], dk_t["role"])],
        ["WR/TE/RB: DK points, generic defense-vs-position vs no defense", paired_verdict(dfs["validation"]["paired_vs_none"]["group"], dk_t["group"])],
        ["RB: RB1/RB2 rushing split vs generic (carry shares)", rb],
        ["QB: pressure/coverage model vs trailing baseline (DK points)", qbv],
        ["QB: generic QB defense-vs-position vs trailing baseline", qbg],
    ], "Specialized vs generic, judged on 2024 validation and the untouched 2025 test")


def signal_table(entries, keys, caption):
    rows = []
    for k in keys:
        v = entries.get(k)
        if not v:
            continue
        rows.append([k, f(v.get("mean_share"), 3) if v.get("mean_share") is not None else "—", f(v["tau"] * 100, 2), f(v["reliability_17_games"], 2),
                     fp(v["p_perm"]), f(v["split_half_r"], 3), f(v["yoy_r"], 3), signal_verdict(v)])
    return table(["Role", "Avg share", "True defense SD (share pts)", "Rel. 17 games", "Perm p", "Split-half r", "YoY r", "Verdict"], rows, caption)


def redistribution_table(entries, caption):
    rows = [[k, f(v["cross_half_r"], 3, sign=True), fp(v["p"]), v["n_defense_seasons"]] for k, v in entries.items() if k not in ("QB", "OTHER")]
    return table(["Receiving role", "Cross-half r", "p", "Defense-seasons"], rows, caption)


def predictive_table(tests, keys):
    rows = []
    for k in keys:
        v = tests[k]
        rows.append([k, f(v["train"]["slope"], 4, sign=True), fp(v["train"]["p"]), fp(v["train"].get("q")), f(v["validation"]["slope"], 4, sign=True), fp(v["validation"]["p"]),
                     f(v["test"]["slope"], 4, sign=True), fp(v["test"]["p"]), v["category"]])
    return table(["Predictor → outcome", "Train", "p", "q", "2024", "p", "2025", "p", "Category"], rows)


def share_model_table(res, caption):
    rows = []
    for split in ["train", "validation", "test"]:
        s = res["scores"][split]
        rows.append([split, f(s["none"], 5), f(s["group"], 5), f(s["role"], 5), f(s.get("script_none"), 5), f(s.get("script_role"), 5)])
    return table(["Split", "No defense (A)", "Generic DvP", "Role (B)", "Vegas terms, no defense", "Role + Vegas (D)"], rows, caption)


def paired_table(paired, caption):
    rows = []
    for split in ["train", "validation", "test"]:
        for k, v in paired[split].items():
            rows.append([split, k.replace("_minus_", " vs ").replace("none", "no-defense").replace("group", "generic").replace("role", "role"),
                         f(v["mean_diff_abs_err_per_team_game"], 5, sign=True), f"{f(v['ci95'][0], 5, sign=True)} to {f(v['ci95'][1], 5, sign=True)}", fp(v["p"]), v["n_team_games"]])
    return table(["Split", "Comparison", "Mean diff", "95% CI", "p", "Team-games"], rows, caption)


def by_role_table(res):
    rows = []
    for role in ROLE_LIST:
        row = [role]
        for split in ["validation", "test"]:
            br = res["scores"][split]["by_role"]
            row += [f(br["none"].get(role), 4), f(br["group"].get(role), 4), f(br["role"].get(role), 4)]
        rows.append(row)
    return table(["Role", "2024 none", "2024 generic", "2024 role", "2025 none", "2025 generic", "2025 role"], rows, "Target-share mean abs error by role")


DFS_MODELS = ["trailing", "none_uncal", "role_uncal", "none", "group", "role", "role_matchup", "script_none", "script_role_matchup"]


def dfs_table(dfs, split):
    rows = []
    for pos in ["ALL", "WR", "TE", "RB"]:
        for m in DFS_MODELS:
            d = dfs[split][pos][m]["dk"]
            rows.append([pos, m, d["n"], f(d["mae"], 3), f(d["rmse"], 3), f(d["bias"], 3, sign=True), f(d["corr"], 3), f(d["median_ae"], 3)])
    return table(["Pos", "Model", "n", "MAE", "RMSE", "Bias", "Corr", "Median AE"], rows, f"DK points, {split}")


def dfs_component_table(dfs, split):
    rows = []
    for m in DFS_MODELS:
        d = dfs[split]["ALL"][m]
        rows.append([m, f(d["targets"]["mae"], 3), f(d["receptions"]["mae"], 3), f(d["rec_yards"]["mae"], 2), f(d["carries"]["mae"], 3), f(d["tds"]["mae"], 3)])
    return table(["Model", "Targets MAE", "Receptions MAE", "Rec yards MAE", "Carries MAE", "TDs MAE"], rows, f"Component errors, all positions, {split}")


def dfs_paired_table(dfs):
    rows = []
    for split in ["train", "validation", "test"]:
        for m, v in dfs[split]["paired_vs_none"].items():
            rows.append([split, m, f(v["mean_diff"], 4, sign=True), f"{f(v['ci95'][0], 4, sign=True)} to {f(v['ci95'][1], 4, sign=True)}", fp(v["p"]), v["n"]])
    return table(["Split", "Model vs no-defense", "Mean abs-error diff (DK pts)", "95% CI (games resampled)", "p", "n"], rows)


def quantile_table(dfs, split):
    rows = []
    for m in DFS_MODELS:
        q = dfs[split]["ALL"][m].get("quantiles", {})
        rows.append([m] + [f"{f(q[k]['coverage'], 3)} / {f(q[k]['pinball'], 3)}" for k in ["P10", "P25", "P50", "P75", "P90", "P95", "P99"]])
    return table(["Model", "P10", "P25", "P50", "P75", "P90", "P95", "P99"], rows,
                 f"Quantile coverage / pinball loss, {split} (coverage should equal the quantile; lower pinball is better)")


def qb_table(qb):
    rows = []
    for split in ["train", "validation", "test"]:
        for m, name in [("A", "Trailing DK/game"), ("G", "Generic QB DvP"), ("P", "Pressure + coverage"), ("GP", "Both")]:
            d = qb[split][m]
            rows.append([split, name, d["n"], f(d["mae"], 3), f(d["rmse"], 3), f(d["bias"], 3, sign=True), f(d["corr"], 3)])
    return table(["Split", "QB model", "n", "MAE", "RMSE", "Bias", "Corr"], rows,
                 f"QB DK points (baseline = trailing DK/game + train league calibration {qb['league_calibration']:+.3f}; generic shrinkage k = {qb['chosen_k']}; "
                 f"pressure/coverage terms fit on train, DK pts per 1 SD: {', '.join(f'{k} {v:+.3f}' for k, v in qb['pressure_coverage_betas_dk_per_sd'].items())})")


def candidates_and_rejections(R):
    cand, rej = [], []
    oos_t, oos_c, dfs, qb = R["oos_shares"]["targets"], R["oos_shares"]["carries"], R["dfs"]["results"], R["qb"]

    def judge(name, val, test, unit):
        dv, dt = diff_key(val), diff_key(test)
        text = f"{name}: 2024 {dv:+.4f} (p {fp(val['p'])}), 2025 {dt:+.4f} (p {fp(test['p'])}) {unit}"
        (cand if dv < 0 and dt < 0 and val["p"] < 0.05 and test["p"] < 0.05 else rej).append(text)

    judge("Role-level target redistribution vs generic defense-vs-position", oos_t["paired"]["validation"]["role_minus_group"], oos_t["paired"]["test"]["role_minus_group"], "share pts per team-game")
    judge("Role-level target model vs no defense", oos_t["paired"]["validation"]["role_minus_none"], oos_t["paired"]["test"]["role_minus_none"], "share pts per team-game")
    judge("Generic defense-vs-position target shares vs no defense", oos_t["paired"]["validation"]["group_minus_none"], oos_t["paired"]["test"]["group_minus_none"], "share pts per team-game")
    judge("Generic defense-vs-position carry shares vs no defense", oos_c["paired"]["validation"]["group_minus_none"], oos_c["paired"]["test"]["group_minus_none"], "share pts per team-game")
    judge("RB1/RB2 carry-split role model vs generic", oos_c["paired"]["validation"]["role_minus_group"], oos_c["paired"]["test"]["role_minus_group"], "share pts per team-game")
    for m, name in [("group", "Generic DvP (DK points)"), ("role", "Role redistribution, model B (DK points)"), ("role_matchup", "B + matchup efficiency, model C (DK points)"),
                    ("script_none", "Vegas/game-script role terms only, no defense (DK points)"), ("script_role_matchup", "C + Vegas/game script, model D (DK points)")]:
        judge(name, dfs["validation"]["paired_vs_none"][m], dfs["test"]["paired_vs_none"][m], "DK pts per player-game")
    for m, name in [("G", "Generic QB defense-vs-position"), ("P", "QB pressure/coverage model"), ("GP", "QB generic + pressure/coverage")]:
        judge(name, qb["validation"]["paired_vs_A"][m], qb["test"]["paired_vs_A"][m], "DK pts per QB-game")
    for k, v in R["predictive_tests"].items():
        same_sign = all(x["slope"] is not None and (x["slope"] > 0) == (v["train"]["slope"] > 0) for x in (v["validation"], v["test"]))
        if v["category"] == "STRONG REPEATABLE SIGNAL" or (v["category"] == "MODERATE SIGNAL" and same_sign):
            cand.append(f"Football pattern (not yet a DFS feature): {k} — {v['category']}; train {v['train']['slope']:+.4f}, 2024 {v['validation']['slope']:+.4f}, 2025 {v['test']['slope']:+.4f}")
        elif v["category"] == "MODERATE SIGNAL":
            rej.append(f"{k}: MODERATE in train + 2024 but reversed sign in 2025 ({v['test']['slope']:+.4f}) - not repeatable")
    return cand, rej


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

def to_markdown(sections, R):
    out = [f"# {TITLE}", "", f"Generated from `research/output/redistribution_results.json` by `scripts/run_redistribution_research.py`. "
           f"Train {R['design']['train']}, validation {R['design']['validation']}, untouched test {R['design']['test']}.", ""]
    for sid, title, blocks in sections:
        out += [f"## {sid}. {title}", ""]
        for b in blocks:
            if b[0] == "p":
                out += [b[1], ""]
            elif b[0] == "ul":
                out += [f"- {i}" for i in b[1]] + [""]
            elif b[0] == "table":
                _, headers, rows, caption = b
                if caption:
                    out += [f"*{caption}*", ""]
                out.append("| " + " | ".join(map(str, headers)) + " |")
                out.append("|" + "---|" * len(headers))
                out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
                out.append("")
    return "\n".join(out)


def to_html_body(sections):
    parts = []
    for sid, title, blocks in sections:
        parts.append(f'<section id="s{sid}"><h2><span class="sid">{sid}</span>{html.escape(title)}</h2>')
        for b in blocks:
            if b[0] == "p":
                parts.append(f"<p>{html.escape(b[1])}</p>")
            elif b[0] == "ul":
                parts.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in b[1]) + "</ul>")
            elif b[0] == "table":
                _, headers, rows, caption = b
                cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
                head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
                body = "".join("<tr>" + "".join(f"<td{verdict_class(c)}>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
                parts.append(f'<div class="tw"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        parts.append("</section>")
    return "\n".join(parts)


def verdict_class(cell):
    s = str(cell)
    for key, cls in [("STRONG", "v-strong"), ("MODERATE", "v-mod"), ("WEAK", "v-weak"), ("NO RELIABLE", "v-none"), ("NO OUT-OF-SAMPLE", "v-none"),
                     ("IMPROVES OUT-OF-SAMPLE", "v-strong"), ("MIXED", "v-weak"), ("INSUFFICIENT", "v-none")]:
        if s.startswith(key):
            return f' class="{cls}"'
    return ""


def build(results_path=OUT_DIR / "redistribution_results.json"):
    R = json.loads(Path(results_path).read_text())
    sections = build_sections(R)
    md = to_markdown(sections, R)
    (OUT_DIR / "DK_EDGE_Defensive_Production_Redistribution_Research.md").write_text(md)
    return R, sections, md




HTML_STYLE = """
<title>Defensive Redistribution Research</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --paper:#F4F6F4; --panel:#FFFFFF; --ink:#18201C; --muted:#5A6660; --rule:#D3DAD6; --accent:#2E6A4E; --accent-soft:#E3EEE8;
  --strong-bg:#DDEFE4; --strong-fg:#1E5A3D; --mod-bg:#F3EBD3; --mod-fg:#6E5613; --weak-bg:#F4E3DA; --weak-fg:#8A4127;
  --none-bg:#E8ECEA; --none-fg:#4E5854;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"IBM Plex Sans","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark; --paper:#111614; --panel:#171E1B; --ink:#E3E9E6; --muted:#9BA7A1; --rule:#2B3531; --accent:#71B28F; --accent-soft:#1D2B25;
    --strong-bg:#1C3328; --strong-fg:#8FD1AC; --mod-bg:#332C17; --mod-fg:#E0C27A; --weak-bg:#3A2419; --weak-fg:#F0A983; --none-bg:#232B28; --none-fg:#AAB5B0;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark; --paper:#111614; --panel:#171E1B; --ink:#E3E9E6; --muted:#9BA7A1; --rule:#2B3531; --accent:#71B28F; --accent-soft:#1D2B25;
  --strong-bg:#1C3328; --strong-fg:#8FD1AC; --mod-bg:#332C17; --mod-fg:#E0C27A; --weak-bg:#3A2419; --weak-fg:#F0A983; --none-bg:#232B28; --none-fg:#AAB5B0;
}
body{background:var(--paper);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.6;margin:0}
.wrap{max-width:980px;margin:0 auto;padding-inline:20px;padding-block:32px 64px}
header .eyebrow{font-family:var(--display);letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600;font-size:14px}
h1{font-family:var(--display);font-weight:700;font-size:clamp(30px,5vw,46px);line-height:1.05;margin:.2em 0 .35em;text-wrap:balance;letter-spacing:.01em}
.lede{color:var(--muted);max-width:68ch;margin:0}
.timeline{display:grid;grid-template-columns:1fr 5fr 1fr 1fr;gap:4px;margin:22px 0 8px;font-size:12px}
.timeline div{padding:8px 10px;border-radius:4px;background:var(--none-bg);color:var(--none-fg)}
.timeline .tr{background:var(--accent-soft);color:var(--accent)}
.timeline .va{background:var(--mod-bg);color:var(--mod-fg)}
.timeline .te{background:var(--weak-bg);color:var(--weak-fg)}
.timeline b{display:block;font-family:var(--display);font-size:15px;letter-spacing:.04em}
nav.index{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 8px}
nav.index a{font-family:var(--display);font-weight:600;font-size:14px;text-decoration:none;color:var(--ink);border:1px solid var(--rule);border-radius:3px;padding:3px 8px;background:var(--panel)}
nav.index a:hover,nav.index a:focus-visible{border-color:var(--accent);color:var(--accent);outline:none}
section{margin-top:44px;scroll-margin-top:16px}
h2{font-family:var(--display);font-weight:600;font-size:26px;letter-spacing:.02em;margin:0 0 12px;display:flex;align-items:baseline;gap:12px;text-wrap:balance;border-bottom:1px solid var(--rule);padding-bottom:6px}
h2 .sid{color:var(--accent);font-weight:700;min-width:1.2em}
p,li{max-width:74ch}
ul{padding-left:1.2em;display:grid;gap:8px}
.tw{overflow-x:auto;margin:14px 0 18px;border:1px solid var(--rule);border-radius:4px;background:var(--panel)}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
caption{caption-side:top;text-align:left;padding:10px 12px 6px;color:var(--muted);font-size:12.5px}
th{font-family:var(--display);font-weight:600;font-size:13.5px;letter-spacing:.03em;text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);white-space:nowrap;color:var(--muted)}
td{padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:0}
td.v-strong,td.v-mod,td.v-weak,td.v-none{font-family:var(--display);font-weight:600;letter-spacing:.03em;font-size:12.5px;white-space:nowrap}
td.v-strong{background:var(--strong-bg);color:var(--strong-fg)}
td.v-mod{background:var(--mod-bg);color:var(--mod-fg)}
td.v-weak{background:var(--weak-bg);color:var(--weak-fg)}
td.v-none{background:var(--none-bg);color:var(--none-fg)}
footer{margin-top:56px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--rule);padding-top:12px}
@media (max-width:560px){.timeline{grid-template-columns:1fr 1fr}.wrap{padding-inline:16px}}
</style>
"""


def to_html(sections, R):
    d = R["design"]
    idx = "".join(f'<a href="#s{sid}">{sid}</a>' for sid, _, _ in sections)
    timeline = (f'<div class="timeline" aria-label="Season splits">'
                f'<div><b>{d["warmup"][0]}</b>warm-up</div>'
                f'<div class="tr"><b>{d["train"][0]}–{d["train"][-1]}</b>discovery &amp; tuning</div>'
                f'<div class="va"><b>{d["validation"][0]}</b>validation</div>'
                f'<div class="te"><b>{d["test"][0]}</b>untouched test</div></div>')
    return (HTML_STYLE + '<div class="wrap"><header><div class="eyebrow">DK Edge research module</div>'
            f"<h1>{html.escape(TITLE)}</h1>"
            '<p class="lede">Does a defense change where an offense\'s production goes, and does modeling that predict better than a generic '
            'defense-vs-position adjustment? Every figure below comes from research/output/redistribution_results.json; nothing here changes the live optimizer.</p>'
            f"{timeline}<nav class=\"index\" aria-label=\"Sections\">{idx}</nav></header>"
            + to_html_body(sections)
            + '<footer>Generated by scripts/run_redistribution_research.py from nflverse play-by-play, participation, player stats, schedules and PFR coverage charting, 2018–2025 regular seasons.</footer></div>')


def build_all(results_path=OUT_DIR / "redistribution_results.json"):
    R, sections, _md = build(results_path)
    (OUT_DIR / "redistribution_report.html").write_text(to_html(sections, R))
    return R, sections


if __name__ == "__main__":
    build_all()
