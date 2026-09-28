from collections import defaultdict

import numpy as np
from sqlalchemy import text

from data.nflverse_fetch import fetch_team_implied_totals, to_nflverse_team
from data.ownership_calibration import _spearman
from data.player_availability import resolve_slate_season_week
from db.migrate import get_engine

# ---------------------------------------------------------------------------
# A fitted ownership model, replacing points-per-$1000 as the ownership
# estimate - trained on real DK contest-standings %Drafted this codebase has
# imported (data/ownership_calibration.py's contest_ownership table), not
# hand-tuned.
#
# Model: a conditional logit over each (slate, position) group. Every player
# at a position on a slate competes for that position's share of the field's
# roster slots; a player's share is softmax(features . beta) within the
# group, and predicted %Drafted = share * that position's expected total
# (QB/DST ~100%, RB/WR/TE more because of FLEX). Modeling SHARE within a
# group, not raw %, is what lets one set of coefficients transfer across
# slates of different sizes: a 14-game slate spreads the same 100% of QB
# ownership over more QBs than a 10-game one.
#
# Features (see _feature_matrix): log projection, log salary, value rank and
# projection rank within the group, and the team's Vegas implied-total gap
# from the slate average (for DST, the OPPONENT's implied total, negated -
# a defense is attractive when the offense it faces is projected to
# struggle). The positive log-salary coefficient this fits is the real
# finding behind the old proxy's biggest miss: at equal projection, the
# field rosters the more EXPENSIVE player more, not less.
#
# Real, disclosed limits of the training data, as of this module's first
# fit: 3 Classic slates across only 2 real NFL weeks (dk_thu_mon_2026_09_17
# and dk_sunday_2026_09_20 are the same week 2 - the Sunday slate's games
# are a subset of the Thu-Mon slate's). Held-out validation is therefore
# leave-one-WEEK-out (2 folds), not leave-one-slate-out, so the same real
# player outcomes never appear on both sides of a split. Showdown slates are
# excluded entirely - see POSITION_OWNERSHIP_CALIBRATION's comment in
# models/field_simulation.py for why Showdown ownership isn't comparable.
# Re-run backtest_ownership_model() as more weeks are imported.
# ---------------------------------------------------------------------------

FEATURE_NAMES = ("log_proj", "log_salary_k", "value_rank", "proj_rank", "implied_gap")
DEFAULT_L2 = 0.1
_MIN_PROJ = 0.1  # floor so a zero-projected player doesn't produce log(0)


def _is_showdown_slate(slate_id):
    return "showdown" in slate_id.lower()


def _classic_slate_ids_with_ownership(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT slate_id FROM contest_ownership "
                "WHERE ownership_proxy IS NOT NULL AND slate_id IS NOT NULL"
            )
        ).fetchall()
    return sorted(r[0] for r in rows if not _is_showdown_slate(r[0]))


def _implied_totals_for_slate(slate_id, engine):
    """({nflverse_team: implied_total}, (season, week) or None). Best-effort
    per this project's convention for external fetches: a failed schedule
    fetch returns empty totals (every implied_gap becomes 0, a neutral
    value) rather than failing the whole fit/prediction.
    """
    try:
        season_week = resolve_slate_season_week(slate_id, engine)
        if season_week is None:
            return {}, None
        return fetch_team_implied_totals(*season_week), season_week
    except Exception:
        return {}, None


def _implied_gap(position, team, opponent, totals, league_avg):
    if not totals or league_avg is None:
        return 0.0
    if position == "DST":
        opp_total = totals.get(to_nflverse_team(opponent)) if opponent else None
        return float(league_avg - opp_total) if opp_total is not None else 0.0
    own_total = totals.get(to_nflverse_team(team)) if team else None
    return float(own_total - league_avg) if own_total is not None else 0.0


def _feature_matrix(players):
    proj = np.array([max(float(p["proj"]), _MIN_PROJ) for p in players])
    salary_k = np.array([float(p["salary"]) / 1000.0 for p in players])
    value = proj / salary_k
    denom = max(len(players) - 1, 1)
    # Rank features are 0 for the best in the group and -1 for the worst, so
    # a positive coefficient always means "better-ranked -> more owned".
    value_rank = -np.argsort(np.argsort(-value)) / denom
    proj_rank = -np.argsort(np.argsort(-proj)) / denom
    implied_gap = np.array([float(p["implied_gap"]) for p in players])
    return np.column_stack([np.log(proj), np.log(salary_k), value_rank, proj_rank, implied_gap])


def _softmax(u):
    e = np.exp(u - u.max())
    return e / e.sum()


def load_training_groups(engine=None, slate_ids=None):
    """Real training data: one group per (Classic slate, position), each
    player's %Drafted averaged across every imported contest on that slate
    (several contests on one slate share the same projections and outcomes,
    so they're one observation of the field, not several independent ones).

    Uses proj_median_at_import - the projection actually stored for the
    slate when the contest was imported - rather than today's projections
    row, which may have been regenerated since.

    Returns [{"slate_id", "season", "week", "position", "players": [{
    "player_id", "proj", "salary", "implied_gap", "pct"}]}].
    """
    engine = engine or get_engine()
    slate_ids = slate_ids or _classic_slate_ids_with_ownership(engine)

    groups = []
    for slate_id in slate_ids:
        totals, season_week = _implied_totals_for_slate(slate_id, engine)
        league_avg = sum(totals.values()) / len(totals) if totals else None
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT co.player_id, co.position, co.salary, sp.team, sp.opponent,
                           avg(co.pct_drafted) AS pct, avg(co.proj_median_at_import) AS proj
                    FROM contest_ownership co
                    JOIN slate_player_pool sp ON sp.slate_id = co.slate_id AND sp.player_id = co.player_id
                    WHERE co.slate_id = :s AND co.ownership_proxy IS NOT NULL
                    GROUP BY co.player_id, co.position, co.salary, sp.team, sp.opponent
                    """
                ),
                {"s": slate_id},
            ).mappings().fetchall()

        by_position = defaultdict(list)
        for r in rows:
            by_position[r["position"]].append(
                {
                    "player_id": r["player_id"],
                    "proj": float(r["proj"]),
                    "salary": r["salary"],
                    "implied_gap": _implied_gap(r["position"], r["team"], r["opponent"], totals, league_avg),
                    "pct": float(r["pct"]),
                }
            )
        for position, players in by_position.items():
            if len(players) < 2 or sum(p["pct"] for p in players) <= 0:
                continue
            groups.append(
                {
                    "slate_id": slate_id,
                    "season": season_week[0] if season_week else None,
                    "week": season_week[1] if season_week else None,
                    "position": position,
                    "players": players,
                }
            )
    return groups


def fit_ownership_model(groups, l2=DEFAULT_L2, max_iters=20000, lr=0.05, tol=1e-7):
    """Fit the conditional-logit coefficients by gradient descent on the
    cross-entropy between each group's real ownership shares and its
    predicted softmax shares (plain numpy - this codebase deliberately has
    no scipy/sklearn dependency; see data/ownership_calibration.py's
    _spearman for the same choice). Small L2 penalty because the training
    set is only a few slates.

    Also records each position's mean total %Drafted across training slates
    (the scale predict_ownership multiplies shares by) - real and stable so
    far: QB/DST ~99%, RB ~233-252%, WR ~326-332%, TE ~115-130%.
    """
    if not groups:
        raise ValueError("No training groups - import real Classic contest-standings first")

    Xs = [_feature_matrix(g["players"]) for g in groups]
    ys = []
    for g in groups:
        pct = np.array([p["pct"] for p in g["players"]])
        ys.append(pct / pct.sum())

    beta = np.zeros(len(FEATURE_NAMES))
    for _ in range(max_iters):
        grad = l2 * beta
        for X, y in zip(Xs, ys):
            grad = grad - X.T @ (y - _softmax(X @ beta)) / len(groups)
        beta = beta - lr * grad
        if float(np.linalg.norm(grad)) < tol:
            break

    totals_by_position = defaultdict(list)
    for g in groups:
        totals_by_position[g["position"]].append(sum(p["pct"] for p in g["players"]))

    return {
        "beta": dict(zip(FEATURE_NAMES, (round(float(b), 4) for b in beta))),
        "position_totals": {pos: round(sum(t) / len(t), 2) for pos, t in totals_by_position.items()},
        "training_slates": sorted({g["slate_id"] for g in groups}),
        "training_weeks": sorted({(g["season"], g["week"]) for g in groups if g["week"] is not None}),
        "n_players": sum(len(g["players"]) for g in groups),
    }


def predict_group_shares(model, players):
    beta = np.array([model["beta"][f] for f in FEATURE_NAMES])
    return _softmax(_feature_matrix(players) @ beta)


def predict_ownership(slate_id, model=None, engine=None, player_ids=None):
    """Predicted %Drafted for every player on a Classic `slate_id` with a
    stored projection - {player_id: predicted_pct}. `model` defaults to a
    fresh fit on every real Classic contest currently imported, so the
    estimate improves automatically as more contests are imported.

    `player_ids` restricts prediction to that subset, and each position's
    total ownership is spread over only those players - pass the
    gate-eligible pool so players already ruled out (OUT, no role) don't
    soak up share the real field would never give them.

    Still a MODEL of ownership, not a live feed: every downstream use should
    say "predicted ownership", never present it as the field's actual
    ownership.
    """
    engine = engine or get_engine()
    if _is_showdown_slate(slate_id):
        raise ValueError(f"{slate_id} is a Showdown slate - this model is fit on Classic slates only")
    model = model or fit_ownership_model(load_training_groups(engine))

    totals, _ = _implied_totals_for_slate(slate_id, engine)
    league_avg = sum(totals.values()) / len(totals) if totals else None
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT sp.player_id, sp.position, sp.salary, sp.team, sp.opponent, proj.proj_median
                FROM slate_player_pool sp
                JOIN projections proj ON proj.slate_id = sp.slate_id AND proj.player_id = sp.player_id
                WHERE sp.slate_id = :s AND proj.proj_median IS NOT NULL
                """
            ),
            {"s": slate_id},
        ).mappings().fetchall()
    if player_ids is not None:
        wanted = set(player_ids)
        rows = [r for r in rows if r["player_id"] in wanted]
    if not rows:
        raise ValueError(f"No players with a stored projection for slate {slate_id}")

    by_position = defaultdict(list)
    for r in rows:
        by_position[r["position"]].append(
            {
                "player_id": r["player_id"],
                "proj": float(r["proj_median"]),
                "salary": r["salary"],
                "implied_gap": _implied_gap(r["position"], r["team"], r["opponent"], totals, league_avg),
            }
        )

    predicted = {}
    for position, players in by_position.items():
        position_total = model["position_totals"].get(position)
        if position_total is None:
            continue
        shares = predict_group_shares(model, players)
        for p, share in zip(players, shares):
            predicted[p["player_id"]] = round(float(share * position_total), 3)
    return predicted


# Floor for field-simulation weights: models/field_simulation.py samples
# opponents on log(weight), and a softmax share can underflow to exactly 0
# for a deep-bench player.
_MIN_FIELD_WEIGHT = 1e-4


def fitted_ownership_fn(slate_id, model=None, engine=None):
    """A proxy_fn for models/field_simulation.py: takes the simulator's
    already-gated player list and returns {player_id: predicted %Drafted}
    over exactly those players. The model is fit once here, not on every
    call.
    """
    engine = engine or get_engine()
    model = model or fit_ownership_model(load_training_groups(engine))

    def proxy_fn(players):
        predicted = predict_ownership(slate_id, model=model, engine=engine, player_ids=[p["player_id"] for p in players])
        return {p["player_id"]: max(predicted.get(p["player_id"], 0.0), _MIN_FIELD_WEIGHT) for p in players}

    proxy_fn.model = model
    return proxy_fn


def _proxy_shares(players):
    value = np.array([max(p["proj"], _MIN_PROJ) / (p["salary"] / 1000.0) for p in players])
    return value / value.sum()


def _score(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    rho, _ = _spearman(list(actual), list(predicted))
    return {
        "n": len(actual),
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "mae_pct_points": round(float(np.mean(np.abs(actual - predicted))), 3),
    }


def backtest_ownership_model(engine=None, l2=DEFAULT_L2):
    """Leave-one-week-out validation against the current stand-in: for each
    real NFL week, fit on every OTHER week's real contests, then predict the
    held-out week's real %Drafted and score both this model and the raw
    points-per-$1000 proxy on the same players, overall and per position.

    Both predictions are scaled to the TRAINING weeks' mean position totals
    (not the held-out week's actual totals), which is what a real pre-lock
    prediction would have. The proxy gets the same within-position
    normalization, so POSITION_OWNERSHIP_CALIBRATION's per-position scalars
    would not change its score - this compares ranking/allocation quality
    within a position, the part the per-position scalars can't fix.

    Returns {"folds": [{"held_out_week", "model", "overall": {"model",
    "proxy"}, "by_position": {...}}], "note": str}.
    """
    engine = engine or get_engine()
    groups = load_training_groups(engine)
    weeks = sorted({(g["season"], g["week"]) for g in groups if g["week"] is not None})
    if len(weeks) < 2:
        raise ValueError(f"Need real contests from at least 2 NFL weeks for a held-out test, have {weeks}")

    folds = []
    for held in weeks:
        train = [g for g in groups if (g["season"], g["week"]) != held]
        test = [g for g in groups if (g["season"], g["week"]) == held]
        model = fit_ownership_model(train, l2=l2)

        actual_all, model_all, proxy_all = [], [], []
        by_position = defaultdict(lambda: ([], [], []))
        for g in test:
            scale = model["position_totals"].get(g["position"])
            if scale is None:
                continue
            actual = [p["pct"] for p in g["players"]]
            model_pred = list(predict_group_shares(model, g["players"]) * scale)
            proxy_pred = list(_proxy_shares(g["players"]) * scale)
            actual_all += actual
            model_all += model_pred
            proxy_all += proxy_pred
            a, m, x = by_position[g["position"]]
            a += actual
            m += model_pred
            x += proxy_pred

        folds.append(
            {
                "held_out_week": held,
                "model": model,
                "overall": {"model": _score(actual_all, model_all), "proxy": _score(actual_all, proxy_all)},
                "by_position": {
                    pos: {"model": _score(a, m), "proxy": _score(a, x)} for pos, (a, m, x) in sorted(by_position.items())
                },
            }
        )

    return {
        "folds": folds,
        "note": (
            f"{len(weeks)} real NFL week(s) of Classic contest data - a small sample; "
            "re-run as more weeks are imported before treating these numbers as settled"
        ),
    }


def ownership_error_report(slate_id, contest_id=None, model=None, engine=None, top_n=10):
    """The weekly "predicted vs actual, where did we miss" loop, for one
    real slate that already has imported contest ownership: every player's
    predicted %Drafted next to the real number, the error, and the biggest
    misses in each direction. `model` should be fit WITHOUT this slate's
    week (e.g. a backtest fold's model, or a fit made before the contest
    was imported) - otherwise this is scoring the model on its own
    training data and the errors will look smaller than they really are.

    `contest_id` restricts the actual side to one contest; default averages
    every imported contest on the slate, same as training.
    """
    engine = engine or get_engine()
    if model is None:
        _, season_week = _implied_totals_for_slate(slate_id, engine)
        groups = [
            g for g in load_training_groups(engine)
            if season_week is None or (g["season"], g["week"]) != tuple(season_week)
        ]
        model = fit_ownership_model(groups)

    predicted = predict_ownership(slate_id, model=model, engine=engine)
    query = (
        "SELECT player_id, name, position, salary, avg(pct_drafted) AS pct FROM contest_ownership "
        "WHERE slate_id = :s AND player_id IS NOT NULL"
        + (" AND contest_id = :c" if contest_id else "")
        + " GROUP BY player_id, name, position, salary"
    )
    with engine.connect() as conn:
        rows = conn.execute(text(query), {"s": slate_id, "c": contest_id}).mappings().fetchall()

    players = []
    for r in rows:
        if r["player_id"] not in predicted:
            continue
        actual = float(r["pct"])
        pred = predicted[r["player_id"]]
        players.append(
            {
                "name": r["name"],
                "position": r["position"],
                "salary": r["salary"],
                "predicted_pct": pred,
                "actual_pct": round(actual, 3),
                "error": round(pred - actual, 3),  # positive: we over-predicted
            }
        )
    if not players:
        raise ValueError(f"No real imported ownership overlaps predictions for slate {slate_id}")

    return {
        "slate_id": slate_id,
        "model_training_weeks": model["training_weeks"],
        "score": _score([p["actual_pct"] for p in players], [p["predicted_pct"] for p in players]),
        "most_under_predicted": sorted(players, key=lambda p: p["error"])[:top_n],
        "most_over_predicted": sorted(players, key=lambda p: -p["error"])[:top_n],
        "players": players,
    }
