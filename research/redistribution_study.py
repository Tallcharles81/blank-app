"""DK Edge defensive production redistribution research.

Question: does a defense systematically change WHERE an opposing offense's
production goes (e.g. WR1 down, WR2/TE up), beyond a generic
defense-vs-position effect - and does modeling that improve out-of-sample
prediction?

Design (fixed before looking at results):
  - Seasons: 2018 warm-up (trailing history only), 2019-2023 discovery/
    hyperparameter selection, 2024 validation, 2025 untouched test.
  - Expected production for an offense-role comes from the specific
    players filling that role, via their own trailing (prior-game) usage
    (research/redistribution_roles.py), normalized so each team-game's
    expected target shares sum to 1. That's the player/offense control.
  - Residual = actual - expected. A defense's effect on a role, as of a
    game, is a shrunk mean of residuals from ITS PRIOR games only:
    sum(resid) / (n + k). k and the window are chosen on 2019-2023.
  - "Is it real" = between-defense variance beyond sampling noise
    (method of moments, permutation-tested by shuffling which defense
    faced which game within a season), split-half and year-over-year
    reliability.
  - "Redistribution" = where a suppressed role's share goes, measured as
    each other role's share of the REMAINING targets vs expectation; the
    null is proportional reallocation, which the sum-to-1 constraint
    produces mechanically.
  - Every family of tests reports its count and a Benjamini-Hochberg FDR
    adjustment.
Nothing in this module is used by the live optimizer.
"""
import math

import numpy as np
import pandas as pd

from research.redistribution_data import (
    load_participation,
    load_pbp,
    load_pfr_def,
    load_schedules,
)
from research.redistribution_roles import (
    build_player_games,
    team_game_flags,
    team_game_roles,
)

WARMUP_SEASONS = [2018]
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
VALIDATION_SEASONS = [2024]
TEST_SEASONS = [2025]
ALL_SEASONS = WARMUP_SEASONS + TRAIN_SEASONS + VALIDATION_SEASONS + TEST_SEASONS

TARGET_ROLES = ["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2", "RB3+", "QB", "OTHER"]
CARRY_ROLES = ["RB1", "RB2", "RB3+", "QB", "WR1", "WR2", "WR3+", "TE1", "TE2+", "OTHER"]
GROUPS = {"WR": ["WR1", "WR2", "WR3+"], "TE": ["TE1", "TE2+"], "RB": ["RB1", "RB2", "RB3+"], "QB": ["QB"], "OTHER": ["OTHER"]}
WINDOWS = [8, 17, 34]
SHRINKAGE_KS = [0, 2, 4, 8, 16, 32, 64, 128]
MIN_GAMES_FOR_FINGERPRINT = 8
N_PERMUTATIONS = 300
N_BOOTSTRAP = 500
RNG = np.random.default_rng(20260928)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


# ---------------------------------------------------------------------------
# Statistics helpers (numpy only - this project has no scipy dependency)
# ---------------------------------------------------------------------------

def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(len(x) - 3)
    return r, normal_two_sided_p(z), len(x)


def benjamini_hochberg(pvalues):
    """BH-adjusted q-values, same order as input; NaN stays NaN."""
    p = np.asarray(pvalues, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def bootstrap_mean_ci(values, clusters=None, n=N_BOOTSTRAP):
    """95% CI of a mean, resampling clusters (e.g. games) when given."""
    values = np.asarray(values, float)
    if clusters is None:
        draws = [values[RNG.integers(0, len(values), len(values))].mean() for _ in range(n)]
    else:
        clusters = np.asarray(clusters)
        uniq, inv = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inv, weights=values)
        counts = np.bincount(inv)
        draws = []
        for _ in range(n):
            pick = RNG.integers(0, len(uniq), len(uniq))
            draws.append(sums[pick].sum() / counts[pick].sum())
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def game_context():
    """Per (game_id, posteam): Vegas (closing) spread/total/implied total
    from the offense's perspective, plus weather/roof."""
    sched = load_schedules()
    sched = sched[sched["game_type"] == "REG"]
    rows = []
    for g in sched.itertuples():
        for team, is_home in ((g.home_team, True), (g.away_team, False)):
            # nflverse spread_line: points the HOME team is favored by.
            favored_by = g.spread_line if is_home else -g.spread_line
            rows.append({
                "game_id": g.game_id, "posteam": team, "is_home": is_home, "total_line": g.total_line,
                "off_favored_by": favored_by,
                "off_implied_total": (g.total_line + favored_by) / 2 if pd.notna(g.total_line) else np.nan,
                "roof": g.roof, "wind": g.wind, "temp": g.temp,
            })
    return pd.DataFrame(rows)


def team_game_passing(seasons):
    """Per (game_id, posteam) passing environment: dropbacks, sacks, QB
    hits, scrambles, INTs, completions, aDOT, deep/middle target rates and
    participation-charted pressure where available."""
    frames = []
    for s in seasons:
        pbp = load_pbp(s)
        db = pbp[(pbp["qb_dropback"] == 1) & (pbp["two_point_attempt"] != 1) & (pbp["qb_spike"] != 1)].copy()
        part = load_participation(s)[["game_id", "play_id", "was_pressure", "defense_man_zone_type", "defense_coverage_type"]]
        db = db.merge(part, on=["game_id", "play_id"], how="left")
        pressure = db["was_pressure"].astype(str).str.upper().map({"TRUE": 1.0, "FALSE": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
        db["pressure"] = pressure
        db["man"] = db["defense_man_zone_type"].map({"MAN_COVERAGE": 1.0, "ZONE_COVERAGE": 0.0})
        coverage = db["defense_coverage_type"].astype(str).str.upper()
        # COMBO/BLOWN/PREVENT don't say how many deep safeties there were.
        known_shell = db["defense_coverage_type"].notna() & ~coverage.isin(["COMBO", "BLOWN", "PREVENT"])
        db["two_high"] = np.where(known_shell, coverage.isin(["COVER_2", "COVER_4", "COVER_6", "COVER_9", "2_MAN"]).astype(float), np.nan)
        att = db[(db["pass_attempt"] == 1) & (db["sack"] != 1)]
        tg = att[att["receiver_player_id"].notna()]
        g = db.groupby(["game_id", "posteam", "defteam"])
        out = pd.DataFrame({
            "dropbacks": g.size(),
            "sacks": g["sack"].sum(),
            "qb_hits": g["qb_hit"].sum(),
            "scrambles": g["qb_scramble"].sum(),
            "ints": g["interception"].sum(),
            "pressure_known": g["pressure"].count(),
            "pressures": g["pressure"].sum(),
            "man_known": g["man"].count(),
            "man_plays": g["man"].sum(),
            "two_high_known": g["two_high"].count(),
            "two_high_plays": g["two_high"].sum(),
        })
        ga = att.groupby(["game_id", "posteam", "defteam"])
        out["attempts"] = ga.size()
        out["completions"] = ga["complete_pass"].sum()
        gt = tg.groupby(["game_id", "posteam", "defteam"])
        out["air_yards"] = gt["air_yards"].sum()
        out["targets_charted"] = gt["air_yards"].count()
        out["deep_targets"] = gt["air_yards"].apply(lambda a: (a >= 20).sum())
        out = out.fillna(0).reset_index()
        meta = pbp.groupby("game_id").agg(season=("season", "first"), week=("week", "first")).reset_index()
        frames.append(out.merge(meta, on="game_id"))
    df = pd.concat(frames, ignore_index=True)
    df["game_order"] = df["season"] * 100 + df["week"]
    df["sack_rate"] = df["sacks"] / df["dropbacks"].replace(0, np.nan)
    df["pressure_rate"] = np.where(df["pressure_known"] >= 0.5 * df["dropbacks"], df["pressures"] / df["pressure_known"].replace(0, np.nan), np.nan)
    df["hit_sack_rate"] = (df["sacks"] + df["qb_hits"]) / df["dropbacks"].replace(0, np.nan)
    df["int_rate"] = df["ints"] / df["attempts"].replace(0, np.nan)
    df["scramble_rate"] = df["scrambles"] / df["dropbacks"].replace(0, np.nan)
    df["cmp_rate"] = df["completions"] / df["attempts"].replace(0, np.nan)
    df["adot"] = df["air_yards"] / df["targets_charted"].replace(0, np.nan)
    df["deep_rate"] = df["deep_targets"] / df["targets_charted"].replace(0, np.nan)
    df["man_rate"] = df["man_plays"] / df["man_known"].replace(0, np.nan)
    df["two_high_rate"] = df["two_high_plays"] / df["two_high_known"].replace(0, np.nan)
    return df


def build_dataset():
    """Everything the study needs, built once. Returns dict of frames."""
    pg = build_player_games(ALL_SEASONS)
    tgr = team_game_roles(pg)
    flags = team_game_flags(pg)
    ctx = game_context()
    passing = team_game_passing(ALL_SEASONS)
    tgr = tgr.merge(ctx, on=["game_id", "posteam"], how="left")
    return {"player_games": pg, "roles": tgr, "flags": flags, "context": ctx, "passing": passing}


def split_of(season):
    if season in WARMUP_SEASONS:
        return "warmup"
    if season in TRAIN_SEASONS:
        return "train"
    if season in VALIDATION_SEASONS:
        return "validation"
    return "test"


# ---------------------------------------------------------------------------
# Rolling (prior-games-only) defense estimates
# ---------------------------------------------------------------------------

def rolling_prior(df, value_col, by, window, k):
    """Shrunk mean of `value_col` over each `by` group's previous `window`
    rows (strictly prior - shift(1)), as sum/(n+k). Rows must be one per
    game per group. Returns (estimate, n_prior)."""
    df = df.sort_values([*by, "game_order"])
    v = df[value_col].fillna(0.0)
    ind = df[value_col].notna().astype(float)
    grp = df[by].apply(tuple, axis=1) if len(by) > 1 else df[by[0]]
    s = v.groupby(grp.values).transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum()).fillna(0.0)
    n = ind.groupby(grp.values).transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum()).fillna(0.0)
    est = s / (n + k) if k > 0 else s / n.replace(0, np.nan)
    return est.reindex(df.index).fillna(0.0).sort_index(), n.reindex(df.index).sort_index()


def add_residuals(roles):
    r = roles.copy()
    r["split"] = r["season"].map(split_of)
    r["tgt_resid"] = r["tgt_share"] - r["exp_tgt_share"]
    r["car_resid"] = r["car_share"] - r["exp_car_share"]
    return r


def predict_shares(roles, share_col, exp_col, resid_col, role_list, level, window, k, script_terms=None, center=False):
    """Adds `pred` for one model: 'none' (offense expectation only), 'group'
    (generic defense-vs-position: the defense's group-level residual,
    spread proportionally within the group) or 'role' (role-level
    redistribution). Predictions are clipped and renormalized so each
    team-game sums to 1.

    `script_terms` (per-role intercept/favored/total, fit on train) shift
    the offense expectation for every level. With center=True the defense
    estimates are built from residuals net of those terms - otherwise a
    league-wide bias in the expectation (e.g. WR3+ expected share running
    high for everyone) gets absorbed into every defense's rolling estimate
    and masquerades as a defense-specific effect."""
    r = roles[roles["role"].isin(role_list)].copy()
    base = r[exp_col].copy()
    if script_terms is not None:
        script_pred = r["role"].map(script_terms["intercept"]).fillna(0) \
            + r["role"].map(script_terms["favored"]).fillna(0) * r["off_favored_by"].fillna(0) \
            + r["role"].map(script_terms["total"]).fillna(0) * (r["total_line"].fillna(44) - 44)
        base = base + script_pred
        if center:
            r[resid_col] = r[resid_col] - script_pred
    if level == "none":
        pred = base
    elif level == "role":
        est, _ = rolling_prior(r, resid_col, ["defteam", "role"], window, k)
        pred = base + est
    elif level == "group":
        r["grp"] = r["role"].map({role: g for g, rs in GROUPS.items() for role in rs})
        g = r.groupby(["game_id", "posteam", "grp"]).agg(resid=(resid_col, lambda s: s.sum(min_count=1)),
                                                         exp=(exp_col, "sum"), defteam=("defteam", "first"),
                                                         game_order=("game_order", "first")).reset_index()
        est, _ = rolling_prior(g, "resid", ["defteam", "grp"], window, k)
        g["est"] = est
        r = r.merge(g[["game_id", "posteam", "grp", "est", "exp"]], on=["game_id", "posteam", "grp"], how="left")
        base = base.values
        pred = pd.Series(np.where(r["exp"] > 0, base * (1 + r["est"] / r["exp"].replace(0, np.nan)), base), index=r.index).fillna(0)
    else:
        raise ValueError(level)
    r["pred"] = np.clip(np.asarray(pred, float), 0.001, None)
    r["pred"] = r["pred"] / r.groupby(["game_id", "posteam"])["pred"].transform("sum")
    return r


def share_loss(pred_frame, share_col, split):
    sub = pred_frame[(pred_frame["split"] == split) & pred_frame[share_col].notna()]
    err = (sub["pred"] - sub[share_col]).abs()
    per_game = err.groupby([sub["game_id"], sub["posteam"]]).sum()
    return float(err.mean()), per_game


def tune_and_compare_shares(roles, share_col, exp_col, resid_col, role_list, script_terms=None, center=False):
    """Pick (window, k) per model level on train, then report train/
    validation/test MAE for none/group/role, and paired differences with
    game-clustered bootstrap CIs."""
    results = {"tuning": [], "chosen": {}, "scores": {}, "paired": {}}
    for level in ["group", "role"]:
        best = None
        for w in WINDOWS:
            for k in SHRINKAGE_KS:
                pf = predict_shares(roles, share_col, exp_col, resid_col, role_list, level, w, k, script_terms, center)
                mae, _ = share_loss(pf, share_col, "train")
                results["tuning"].append({"level": level, "window": w, "k": k, "train_mae": mae})
                if best is None or mae < best[0]:
                    best = (mae, w, k)
        results["chosen"][level] = {"window": best[1], "k": best[2]}
    frames = {"none": predict_shares(roles, share_col, exp_col, resid_col, role_list, "none", 8, 0, script_terms, center)}
    for level in ["group", "role"]:
        c = results["chosen"][level]
        frames[level] = predict_shares(roles, share_col, exp_col, resid_col, role_list, level, c["window"], c["k"], script_terms, center)
    for split in ["train", "validation", "test"]:
        results["scores"][split] = {}
        per_game = {}
        for level, pf in frames.items():
            mae, pg = share_loss(pf, share_col, split)
            results["scores"][split][level] = mae
            per_game[level] = pg
        for a, b in [("none", "group"), ("none", "role"), ("group", "role")]:
            d = (per_game[b] - per_game[a]).dropna()
            lo, hi = bootstrap_mean_ci(d.values)
            z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else 0.0
            results["paired"].setdefault(split, {})[f"{b}_minus_{a}"] = {
                "mean_diff_abs_err_per_team_game": float(d.mean()), "ci95": [lo, hi], "p": normal_two_sided_p(z), "n_team_games": len(d),
            }
        # per-role MAE for the chosen models
        results["scores"][split]["by_role"] = {
            level: pf[(pf["split"] == split)].assign(e=lambda x: (x["pred"] - x[share_col]).abs()).groupby("role")["e"].mean().to_dict()
            for level, pf in frames.items()
        }
    return results, frames


# ---------------------------------------------------------------------------
# Is there a real between-defense signal?
# ---------------------------------------------------------------------------

def variance_components(df, value_col, group_cols=("defteam", "season")):
    """Method-of-moments between-group SD (tau) of a residual, beyond what
    sampling noise alone produces, plus the implied optimal shrinkage k =
    sigma^2 / tau^2 and a permutation p-value (defense labels shuffled
    within season)."""
    d = df[df[value_col].notna()][[*list(group_cols), value_col]]
    if d.empty:
        return None

    def stat(frame):
        g = frame.groupby(list(group_cols))[value_col]
        n, m, v = g.count(), g.mean(), g.var(ddof=1)
        keep = n >= 4
        n, m, v = n[keep], m[keep], v[keep]
        sigma2 = float(v.mean())
        between = float(m.var(ddof=1))
        tau2 = between - float((v / n).mean())
        return tau2, sigma2, float(n.mean()), len(n)

    tau2, sigma2, n_bar, n_groups = stat(d)
    null = []
    seasons = d[group_cols[1]].values
    for _ in range(N_PERMUTATIONS):
        shuffled = d.copy()
        for s in np.unique(seasons):
            mask = seasons == s
            shuffled.loc[mask, group_cols[0]] = RNG.permutation(shuffled.loc[mask, group_cols[0]].values)
        null.append(stat(shuffled)[0])
    p = (1 + sum(1 for x in null if x >= tau2)) / (1 + len(null))
    tau = math.sqrt(tau2) if tau2 > 0 else 0.0
    reliability_17 = tau2 / (tau2 + sigma2 / 17) if tau2 > 0 else 0.0
    return {
        "tau": tau, "sigma": math.sqrt(sigma2), "tau2": tau2, "n_groups": n_groups, "mean_games": n_bar,
        "p_perm": p, "optimal_k": (sigma2 / tau2) if tau2 > 0 else None, "reliability_17_games": reliability_17,
    }


def split_half_and_yoy(df, value_col):
    d = df[df[value_col].notna()]
    odd = d[d["week"] % 2 == 1].groupby(["defteam", "season"])[value_col].mean()
    even = d[d["week"] % 2 == 0].groupby(["defteam", "season"])[value_col].mean()
    j = pd.concat([odd.rename("a"), even.rename("b")], axis=1).dropna()
    r, p, n = pearson(j["a"], j["b"])
    sb = 2 * r / (1 + r) if not math.isnan(r) and r > -1 else float("nan")
    season_mean = d.groupby(["defteam", "season"])[value_col].mean().reset_index()
    nxt = season_mean.assign(season=season_mean["season"] - 1).rename(columns={value_col: "next"})
    yy = season_mean.merge(nxt, on=["defteam", "season"])
    ry, py, ny = pearson(yy[value_col], yy["next"])
    return {"split_half_r": r, "split_half_p": p, "split_half_n": n, "spearman_brown": sb,
            "yoy_r": ry, "yoy_p": py, "yoy_n": ny}


def per_defense_season_tests(df, value_col):
    """z-test of every defense-season's mean residual vs 0. Returns the
    frame of tests (for FDR accounting)."""
    g = df[df[value_col].notna()].groupby(["defteam", "season"])[value_col]
    t = pd.DataFrame({"n": g.count(), "mean": g.mean(), "sd": g.std(ddof=1)}).reset_index()
    t = t[t["n"] >= 4]
    t["z"] = t["mean"] / (t["sd"] / np.sqrt(t["n"]))
    t["p"] = [normal_two_sided_p(z) if np.isfinite(z) else np.nan for z in t["z"]]
    return t


def conditional_redistribution(roles, suppressed_role, share_col="tgt_share", exp_col="exp_tgt_share", role_list=TARGET_ROLES):
    """For each team-game: each other role's share of the targets NOT going
    to `suppressed_role`, minus its expected share of the same remainder.
    Then, across defense-seasons, correlate the defense's suppressed-role
    residual in odd weeks with the other roles' conditional residuals in
    even weeks (and vice versa, averaged). Proportional reallocation - the
    arithmetic consequence of shares summing to 1 - predicts ~0."""
    r = roles[roles["role"].isin(role_list)].copy()
    sup = r[r["role"] == suppressed_role].set_index(["game_id", "posteam"])
    r = r.join(sup[[share_col, exp_col]].rename(columns={share_col: "sup_act", exp_col: "sup_exp"}), on=["game_id", "posteam"])
    r["cond_act"] = r[share_col] / (1 - r["sup_act"]).replace(0, np.nan)
    r["cond_exp"] = r[exp_col] / (1 - r["sup_exp"]).replace(0, np.nan)
    r["cond_resid"] = r["cond_act"] - r["cond_exp"]
    r["sup_resid"] = r["sup_act"] - r["sup_exp"]
    out = {}
    for role in role_list:
        if role == suppressed_role:
            continue
        sub = r[r["role"] == role]
        halves = {}
        for parity in (0, 1):
            h = sub[sub["week"] % 2 == parity].groupby(["defteam", "season"]).agg(sup=("sup_resid", "mean"), cond=("cond_resid", "mean"))
            halves[parity] = h
        j1 = halves[1][["sup"]].join(halves[0][["cond"]], how="inner")
        j2 = halves[0][["sup"]].join(halves[1][["cond"]], how="inner")
        r1, _p1, n1 = pearson(j1["sup"], j1["cond"])
        r2, _p2, n2 = pearson(j2["sup"], j2["cond"])
        rr = np.nanmean([r1, r2])
        z = math.atanh(max(min(rr, 0.999), -0.999)) * math.sqrt(min(n1, n2) - 3) if min(n1, n2) > 3 else 0.0
        out[role] = {"cross_half_r": float(rr), "p": normal_two_sided_p(z), "n_defense_seasons": int(min(n1, n2))}
    return out


# ---------------------------------------------------------------------------
# Pressure study
# ---------------------------------------------------------------------------

def pressure_play_level(player_games, seasons):
    """Within-game comparison on dropbacks with charted pressure: where do
    targets go, and what happens to aDOT/deep rate/completion/sack/INT/
    scramble when the QB is pressured vs not?"""
    roles = player_games[["game_id", "posteam", "player_id", "role"]]
    frames = []
    for s in seasons:
        pbp = load_pbp(s)
        db = pbp[(pbp["qb_dropback"] == 1) & (pbp["two_point_attempt"] != 1) & (pbp["qb_spike"] != 1)]
        part = load_participation(s)[["game_id", "play_id", "was_pressure"]]
        db = db.merge(part, on=["game_id", "play_id"], how="left")
        db["pressure"] = db["was_pressure"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
        db = db[db["pressure"].notna()]
        if db.empty:
            continue
        db = db.merge(roles.rename(columns={"player_id": "receiver_player_id"}), on=["game_id", "posteam", "receiver_player_id"], how="left")
        db["season"] = s
        frames.append(db)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    out = {"seasons": sorted(d["season"].unique().tolist()), "dropbacks": len(d), "pressured_rate": float(d["pressure"].mean())}
    metrics = {}
    for name, frame, col in [
        ("sack_rate", d, "sack"), ("scramble_rate", d, "qb_scramble"),
        ("int_rate_per_attempt", d[(d["pass_attempt"] == 1) & (d["sack"] != 1)], "interception"),
        ("completion_rate", d[(d["pass_attempt"] == 1) & (d["sack"] != 1)], "complete_pass"),
        ("adot", d[d["receiver_player_id"].notna() & d["air_yards"].notna()], "air_yards"),
        ("deep_target_rate", d[d["receiver_player_id"].notna() & d["air_yards"].notna()].assign(deep=lambda x: (x["air_yards"] >= 20).astype(float)), "deep"),
    ]:
        metrics[name] = _pressured_vs_clean(frame, col)
    targets = d[d["receiver_player_id"].notna()].copy()
    targets["role"] = targets["role"].fillna("OTHER")
    role_shares = {}
    for role in TARGET_ROLES:
        targets["is_role"] = (targets["role"] == role).astype(float)
        role_shares[role] = _pressured_vs_clean(targets, "is_role")
    out["metrics"] = metrics
    out["target_share_by_role"] = role_shares
    return out


def _pressured_vs_clean(frame, col):
    frame = frame[frame[col].notna()]
    p = frame[frame["pressure"] == 1]
    c = frame[frame["pressure"] == 0]
    diff_by_game = (p.groupby("game_id")[col].mean() - c.groupby("game_id")[col].mean()).dropna()
    lo, hi = bootstrap_mean_ci(diff_by_game.values)
    z = diff_by_game.mean() / (diff_by_game.std(ddof=1) / math.sqrt(len(diff_by_game))) if len(diff_by_game) > 2 else 0.0
    return {"pressured": float(p[col].mean()), "clean": float(c[col].mean()), "n_pressured": len(p), "n_clean": len(c),
            "within_game_diff": float(diff_by_game.mean()), "ci95": [lo, hi], "p": normal_two_sided_p(z)}


def defense_pre_game_rates(passing, window=8):
    """Trailing (prior-games) defensive pressure/man/two-high rates and the
    offense's trailing pressure-allowed rate, merged onto each team-game."""
    p = passing.copy()
    p["press_for_def"] = p["pressure_rate"].fillna(p["hit_sack_rate"])
    out = p[["game_id", "posteam", "defteam", "season", "week", "game_order"]].copy()
    for col, by, name in [("press_for_def", "defteam", "def_pressure_pre"), ("press_for_def", "posteam", "off_pressure_allowed_pre"),
                          ("man_rate", "defteam", "def_man_pre"), ("two_high_rate", "defteam", "def_two_high_pre"),
                          ("adot", "posteam", "off_adot_pre"), ("sack_rate", "posteam", "off_sack_rate_pre")]:
        tmp = p[[by, "game_order", col]].rename(columns={by: "team"})
        est, n = rolling_prior(tmp.assign(game_order=tmp["game_order"]), col, ["team"], window, 0)
        out[name] = est.values
        out[name + "_n"] = n.values
    return out


# ---------------------------------------------------------------------------
# Coverage (PFR per-defender charting) - team-level and top-CB signals
# ---------------------------------------------------------------------------

def coverage_pre_game(seasons, window=8):
    """Trailing team coverage (yards/target allowed, all charted
    defenders) and the team's best CB's trailing yards/target allowed
    (min 10 trailing targets), both strictly prior-game. PFR charting is a
    charter's call about the covering defender, not a verified assignment."""
    frames = []
    for s in seasons:
        d = load_pfr_def(s)
        cols = {c: c for c in d.columns}
        tgt = "def_targets" if "def_targets" in cols else None
        yds = "def_yards_allowed" if "def_yards_allowed" in cols else None
        if not tgt or not yds:
            continue
        d = d[["season", "week", "team", "pfr_player_id", tgt, yds]].rename(columns={tgt: "t", yds: "y"})
        d["t"] = pd.to_numeric(d["t"], errors="coerce").fillna(0)
        d["y"] = pd.to_numeric(d["y"], errors="coerce").fillna(0)
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["game_order"] = d["season"] * 100 + d["week"]
    team = d.groupby(["team", "game_order"])[["t", "y"]].sum().reset_index().sort_values(["team", "game_order"])
    for col in ["t", "y"]:
        team[f"prior_{col}"] = team.groupby("team")[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    team["def_ypt_allowed_pre"] = team["prior_y"] / team["prior_t"].replace(0, np.nan)
    d = d.sort_values(["pfr_player_id", "game_order"])
    for col in ["t", "y"]:
        d[f"prior_{col}"] = d.groupby("pfr_player_id")[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    d["ypt_pre"] = d["prior_y"] / d["prior_t"].replace(0, np.nan)
    elig = d[d["prior_t"] >= 10]
    best = elig.groupby(["team", "game_order"])["ypt_pre"].min().rename("top_cb_ypt_pre").reset_index()
    out = team[["team", "game_order", "def_ypt_allowed_pre"]].merge(best, on=["team", "game_order"], how="left")
    return out.rename(columns={"team": "defteam"})


# ---------------------------------------------------------------------------
# DFS translation (player level)
# ---------------------------------------------------------------------------

DK_PER_REC = 1.0


def player_dfs_frame(player_games, share_frames, carry_frames, team_env):
    """Per skill player-game: opportunity-based DK prediction under each
    model. pred_targets = role predicted share x team expected targets x
    player's share within the role (trailing); likewise carries. DK per
    target/carry = player's trailing rate shrunk toward position means.
    Team expected targets/carries = team trailing per-game volume (the
    same for every model, so differences come only from the defense/
    redistribution terms)."""
    pg = player_games[player_games["position"].isin(["WR", "TE", "RB"])].copy()
    pg["rec_dk"] = pg["rec"] * DK_PER_REC + 0.1 * pg["rec_yds"] + 6 * pg["rec_td"]
    pg["rush_dk"] = 0.1 * pg["rush_yds"] + 6 * pg["rush_td"]
    pg = pg.sort_values(["player_id", "game_order"])
    for col in ["rec_dk", "rush_dk", "tgt", "car", "rec", "rec_yds", "rec_td", "rush_td"]:
        pg[f"pr_{col}"] = pg.groupby("player_id")[col].transform(lambda x: x.shift(1).rolling(16, min_periods=1).sum()).fillna(0)
    pos_rates = pg[pg["season"].isin(TRAIN_SEASONS)].groupby("position").agg(rec_dk=("rec_dk", "sum"), tgt=("tgt", "sum"),
                                                                              rush_dk=("rush_dk", "sum"), car=("car", "sum"))
    rec_rate = (pos_rates["rec_dk"] / pos_rates["tgt"]).to_dict()
    rush_rate = (pos_rates["rush_dk"] / pos_rates["car"]).to_dict()
    m = 20.0  # pseudo-opportunities pulling thin per-opportunity rates to the position mean
    pg["dk_per_tgt"] = (pg["pr_rec_dk"] + m * pg["position"].map(rec_rate)) / (pg["pr_tgt"] + m)
    pg["dk_per_car"] = (pg["pr_rush_dk"] + m * pg["position"].map(rush_rate)) / (pg["pr_car"] + m)
    pg["catch_rate"] = (pg["pr_rec"] + 10 * 0.65) / (pg["pr_tgt"] + 10)
    pg["ypt"] = (pg["pr_rec_yds"] + 20 * 7.5) / (pg["pr_tgt"] + 20)
    pg["td_per_tgt"] = (pg["pr_rec_td"] + 40 * 0.04) / (pg["pr_tgt"] + 40)
    pg["td_per_car"] = (pg["pr_rush_td"] + 40 * 0.03) / (pg["pr_car"] + 40)

    keys = ["game_id", "posteam"]
    w_tgt = pg["tr_tgt_share"].fillna(0).clip(lower=0)
    w_car = pg["tr_car_share"].fillna(0).clip(lower=0)
    pg["within_role_tgt"] = w_tgt / w_tgt.groupby([pg["game_id"], pg["posteam"], pg["role"]]).transform("sum").replace(0, np.nan)
    pg["within_role_car"] = w_car / w_car.groupby([pg["game_id"], pg["posteam"], pg["role"]]).transform("sum").replace(0, np.nan)
    pg = pg.merge(team_env[[*keys, "exp_team_tgt", "exp_team_car"]], on=keys, how="left")

    for name, sf in share_frames.items():
        pg = pg.merge(sf[[*keys, "role", "pred"]].rename(columns={"pred": f"tshare_{name}"}), on=[*keys, "role"], how="left")
    for name, cf in carry_frames.items():
        pg = pg.merge(cf[[*keys, "role", "pred"]].rename(columns={"pred": f"cshare_{name}"}), on=[*keys, "role"], how="left")
    for name in share_frames:
        cname = name if name in carry_frames else "none"
        pg[f"pred_tgt_{name}"] = pg[f"tshare_{name}"].fillna(0) * pg["exp_team_tgt"] * pg["within_role_tgt"].fillna(0)
        pg[f"pred_car_{name}"] = pg[f"cshare_{cname}"].fillna(0) * pg["exp_team_car"] * pg["within_role_car"].fillna(0)
        pg[f"pred_dk_{name}"] = pg[f"pred_tgt_{name}"] * pg["dk_per_tgt"] + pg[f"pred_car_{name}"] * pg["dk_per_car"]
        pg[f"pred_td_{name}"] = pg[f"pred_tgt_{name}"] * pg["td_per_tgt"] + pg[f"pred_car_{name}"] * pg["td_per_car"]
    pg["split"] = pg["season"].map(split_of)
    return pg


def team_expected_volume(roles, window=8):
    """Team trailing targets and carries per game (strictly prior)."""
    t = roles.drop_duplicates(["game_id", "posteam"])[["game_id", "posteam", "season", "week", "game_order", "team_tgt", "team_car"]].copy()
    for col in ["team_tgt", "team_car"]:
        est, _ = rolling_prior(t, col, ["posteam"], window, 0)
        t[f"exp_{col}"] = est
    return t.rename(columns={"exp_team_tgt": "exp_team_tgt", "exp_team_car": "exp_team_car"})


def point_metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = p - a
    r, _, _ = pearson(a, p)
    return {"n": len(a), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "corr": r, "median_ae": float(np.median(np.abs(e)))}


def quantile_metrics(train_actual, train_pred, test_actual, test_pred):
    """Multiplicative empirical residual quantiles from the training
    period (actual/pred), applied to test predictions. Reports coverage
    (share of actuals at or below each predicted quantile) and pinball
    loss per quantile."""
    ratio = np.asarray(train_actual, float) / np.clip(np.asarray(train_pred, float), 0.5, None)
    out = {}
    for q in QUANTILES:
        rq = float(np.quantile(ratio, q))
        pq = np.clip(np.asarray(test_pred, float), 0.5, None) * rq
        a = np.asarray(test_actual, float)
        diff = a - pq
        pinball = float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
        out[f"P{int(q * 100)}"] = {"coverage": float((a <= pq).mean()), "pinball": pinball}
    return out


# ---------------------------------------------------------------------------
# Script (Vegas) terms, defense receiving-efficiency term, DFS evaluation
# ---------------------------------------------------------------------------

def fit_script_terms(roles, resid_col, role_list):
    """Per-role linear fit on TRAIN seasons: resid ~ a + b*favored_by +
    c*(total-44). Used by Model D to condition the expected distribution on
    game script."""
    out = {"intercept": {}, "favored": {}, "total": {}}
    tr = roles[roles["season"].isin(TRAIN_SEASONS) & roles["role"].isin(role_list)]
    for role in role_list:
        d = tr[(tr["role"] == role)][[resid_col, "off_favored_by", "total_line"]].dropna()
        if len(d) < 50:
            continue
        X = np.column_stack([np.ones(len(d)), d["off_favored_by"], d["total_line"] - 44])
        beta, *_ = np.linalg.lstsq(X, d[resid_col].values, rcond=None)
        out["intercept"][role], out["favored"][role], out["total"][role] = map(float, beta)
    return out


def fit_role_calibration(roles, resid_col, role_list):
    """Intercept-only version of fit_script_terms: each role's mean residual
    on TRAIN seasons. Applied to every calibrated model alike, so the
    defense terms are judged net of league-wide expectation bias."""
    terms = {"intercept": {}, "favored": {}, "total": {}}
    tr = roles[roles["season"].isin(TRAIN_SEASONS) & roles["role"].isin(role_list)]
    for role, v in tr.groupby("role")[resid_col].mean().items():
        terms["intercept"][role] = float(v)
    return terms


def league_efficiency_calibration(dfs):
    """Train-period league mean of (actual - expected) DK per target and per
    carry, so per-opportunity rates carry no shared bias into any model."""
    tr = dfs[dfs["split"] == "train"]
    mu_tgt = float((tr["rec_dk"] - tr["tgt"] * tr["dk_per_tgt"]).sum() / tr["tgt"].sum())
    mu_car = float((tr["rush_dk"] - tr["car"] * tr["dk_per_car"]).sum() / tr["car"].sum())
    return mu_tgt, mu_car


def defense_receiving_efficiency(dfs, base_model="role", mu_tgt=0.0, mu_car=0.0, k_values=SHRINKAGE_KS, window=17):
    """Per-target DK residual allowed by each defense (actual receiving DK
    minus targets x the receiver's own league-calibrated trailing DK/target),
    rolled over the defense's prior games and shrunk; k tuned on train by DK
    MAE of `base_model`. Returns (per-game frame with `eff`, chosen k)."""
    d = dfs.copy()
    d["eff_resid"] = d["rec_dk"] - d["tgt"] * (d["dk_per_tgt"] + mu_tgt)
    per_game = d.groupby(["game_id", "defteam"]).agg(resid=("eff_resid", "sum"), tgt=("tgt", "sum"),
                                                     game_order=("game_order", "first")).reset_index()
    per_game = per_game.sort_values(["defteam", "game_order"])
    s = per_game.groupby("defteam")["resid"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    t = per_game.groupby("defteam")["tgt"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    best = None
    for k in k_values:
        k_t = k * 30  # k in games -> ~30 targets per game
        per_game["eff"] = s / (t + k_t) if k_t > 0 else s / t.replace(0, np.nan)
        m = d.merge(per_game[["game_id", "defteam", "eff"]], on=["game_id", "defteam"], how="left")
        pred = m[f"pred_tgt_{base_model}"] * (m["dk_per_tgt"] + mu_tgt + m["eff"].fillna(0)) + m[f"pred_car_{base_model}"] * (m["dk_per_car"] + mu_car)
        tr = m["split"] == "train"
        mae = float((pred[tr] - m.loc[tr, "dk_points"]).abs().mean())
        if best is None or mae < best[0]:
            best = (mae, k, per_game[["game_id", "defteam", "eff"]].copy())
    return best[2], best[1]


def evaluate_dfs(dfs, model_cols, min_prior_games=4, baseline="none"):
    """Point metrics per split x position x model for DK points, targets,
    receptions, receiving yards, carries; quantile coverage/pinball using
    train residual ratios."""
    d = dfs[(dfs["prior_games"] >= min_prior_games) & dfs["role"].isin(["WR1", "WR2", "WR3+", "TE1", "TE2+", "RB1", "RB2", "RB3+"])]
    out = {}
    for split in ["train", "validation", "test"]:
        s = d[d["split"] == split]
        out[split] = {}
        for pos in ["ALL", "WR", "TE", "RB"]:
            sp = s if pos == "ALL" else s[s["position"] == pos]
            res = {}
            for name in model_cols:
                res[name] = {
                    "dk": point_metrics(sp["dk_points"], sp[f"pred_dk_{name}"]),
                    "targets": point_metrics(sp["tgt"], sp[f"pred_tgt_{name}"]),
                    "receptions": point_metrics(sp["rec"], sp[f"pred_tgt_{name}"] * sp["catch_rate"]),
                    "rec_yards": point_metrics(sp["rec_yds"], sp[f"pred_tgt_{name}"] * sp["ypt"]),
                    "carries": point_metrics(sp["car"], sp[f"pred_car_{name}"]),
                    "tds": point_metrics(sp["rec_td"] + sp["rush_td"], sp[f"pred_td_{name}"]),
                }
                if split != "train":
                    tr = d[(d["split"] == "train") & ((d["position"] == pos) | (pos == "ALL"))]
                    res[name]["quantiles"] = quantile_metrics(tr["dk_points"], tr[f"pred_dk_{name}"], sp["dk_points"], sp[f"pred_dk_{name}"])
            out[split][pos] = res
        # paired DK abs-error differences vs the baseline model, clustered by game
        base = (s[f"pred_dk_{baseline}"] - s["dk_points"]).abs()
        out[split]["paired_vs_none"] = {}
        for name in model_cols:
            if name == baseline:
                continue
            diff = (s[f"pred_dk_{name}"] - s["dk_points"]).abs() - base
            lo, hi = bootstrap_mean_ci(diff.values, clusters=s["game_id"].values)
            z = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 2 and diff.std() > 0 else 0.0
            out[split]["paired_vs_none"][name] = {"mean_diff": float(diff.mean()), "ci95": [lo, hi], "p": normal_two_sided_p(z), "n": len(diff)}
    return out


def qb_models(player_games, passing, pre_rates, k_values=SHRINKAGE_KS, window=17):
    """QB DK points: A = trailing DK/game plus the train-period league mean
    residual (so no model gets credit for fixing a shared bias); G = A +
    defense's shrunk QB-DK residual (generic QB DvP); P = A + pressure/
    coverage terms fit on train; GP = both."""
    qb = player_games[(player_games["role"] == "QB") & (player_games["prior_games"] >= 4)].copy()
    qb["split"] = qb["season"].map(split_of)
    mu = float((qb.loc[qb["split"] == "train", "dk_points"] - qb.loc[qb["split"] == "train", "tr_dk_pg"]).mean())
    qb["pred_A"] = qb["tr_dk_pg"] + mu
    qb["resid"] = qb["dk_points"] - qb["pred_A"]
    qb = qb.merge(pre_rates[["game_id", "posteam", "def_pressure_pre_z", "def_man_pre_z", "def_two_high_pre_z", "off_pressure_allowed_pre"]],
                  on=["game_id", "posteam"], how="left")
    best = None
    for k in k_values:
        est, _ = rolling_prior(qb, "resid", ["defteam"], window, k)
        pred = qb["pred_A"] + est
        tr = qb["split"] == "train"
        mae = float((pred[tr] - qb.loc[tr, "dk_points"]).abs().mean())
        if best is None or mae < best[0]:
            best = (mae, k, est)
    qb["pred_G"] = qb["pred_A"] + best[2]
    tr = qb[(qb["split"] == "train")].dropna(subset=["def_pressure_pre_z", "def_man_pre_z", "def_two_high_pre_z"])
    X = np.column_stack([tr["def_pressure_pre_z"], tr["def_man_pre_z"], tr["def_two_high_pre_z"]])
    beta, *_ = np.linalg.lstsq(X, (tr["dk_points"] - tr["pred_A"]).values, rcond=None)
    adj = (qb["def_pressure_pre_z"].fillna(0) * beta[0] + qb["def_man_pre_z"].fillna(0) * beta[1] + qb["def_two_high_pre_z"].fillna(0) * beta[2])
    qb["pred_P"] = qb["pred_A"] + adj
    qb["pred_GP"] = qb["pred_G"] + adj
    out = {"chosen_k": best[1], "league_calibration": mu,
           "pressure_coverage_betas_dk_per_sd": {"pressure": float(beta[0]), "man": float(beta[1]), "two_high": float(beta[2])}}
    for split in ["train", "validation", "test"]:
        s = qb[qb["split"] == split]
        out[split] = {m: point_metrics(s["dk_points"], s[f"pred_{m}"]) for m in ["A", "G", "P", "GP"]}
        base = (s["pred_A"] - s["dk_points"]).abs()
        out[split]["paired_vs_A"] = {}
        for m in ["G", "P", "GP"]:
            diff = (s[f"pred_{m}"] - s["dk_points"]).abs() - base
            lo, hi = bootstrap_mean_ci(diff.values)
            z = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 2 and diff.std() > 0 else 0.0
            out[split]["paired_vs_A"][m] = {"mean_diff": float(diff.mean()), "ci95": [lo, hi], "p": normal_two_sided_p(z), "n": len(diff)}
    return out


def defensive_fingerprints(roles, passing_resid, season, k_by_role, window=17):
    """Every defense's shrunk role and environment effects as of the END of
    `season` (its last `window` games), with posterior SDs and an evidence
    label. k_by_role gives each metric's variance-component optimal k (or
    None when no between-defense variance was found -> effect reported as 0,
    'NO SIGNAL')."""
    rows = []
    last = roles[roles["season"] == season]
    for defteam in sorted(last["defteam"].dropna().unique()):
        entry = {"defense": defteam}
        d = roles[(roles["defteam"] == defteam) & (roles["season"] <= season)].sort_values("game_order")
        games = d["game_id"].unique()[-window:]
        d = d[d["game_id"].isin(games)]
        entry["games"] = len(games)
        for metric, (col, k, sigma, league_mean) in k_by_role.items():
            if metric.startswith("env:"):
                vals = passing_resid[(passing_resid["defteam"] == defteam) & passing_resid["game_id"].isin(games)][col].dropna()
            else:
                vals = d[d["role"] == metric][col].dropna()
            n = len(vals)
            if n < MIN_GAMES_FOR_FINGERPRINT:
                entry[metric] = {"effect": None, "label": "INSUFFICIENT SAMPLE", "n": n}
                continue
            if k is None:
                entry[metric] = {"effect": 0.0, "effect_pct": 0.0, "label": "NO SIGNAL (no between-defense variance)", "n": n}
                continue
            est = float(vals.sum() / (n + k))
            post_sd = math.sqrt(1.0 / (n / sigma ** 2 + k / sigma ** 2))
            z = abs(est) / post_sd
            rel = n / (n + k)
            label = "HIGH" if (z >= 3 and rel >= 0.5) else "MEDIUM" if z >= 2 else "LOW"
            entry[metric] = {"effect": est, "effect_pct": est / league_mean if league_mean else None, "post_sd": post_sd, "reliability": rel,
                             "label": label, "n": n}
        rows.append(entry)
    return rows
