import json

import numpy as np
from sqlalchemy import text

from data.player_crosswalk import SHOWDOWN_PSEUDO_POSITIONS, resolve_dk_players_to_gsis
from data.pre_lock_check import _load_recent_usage_batch
from db.migrate import get_engine

DEFAULT_NUM_SIMULATIONS = 10000

# Real, empirically-fitted correlations - replaces an earlier hand-picked set
# of well-known GPP-stacking rules of thumb ("QB correlates strongly (~0.55)
# with his own pass-catchers... weakly/negatively (~-0.05) with his own RB...")
# that were never actually fit to data, because at the time this module was
# built there wasn't enough real joint (same-game, same-week) player outcome
# data available to fit from. That data exists now - see
# models/calibration.py::fit_real_correlation_matrix for the fitting function
# (real Pearson correlation across every real (team, season, week) in
# player_weekly_stats, restricted to players with meaningful usage -
# target_share >= 0.10 or carries >= 5 - so the fit measures the same
# population of "players who'd actually end up in a real generated lineup"
# that this correlation matrix gets applied to, not deep-bench/garbage-time
# noise). Several of the real numbers below are notably different from the
# original hand-picked guesses, in both direction and size:
#   - QB_PASS_CATCHER_CORR: guessed 0.55, real ~0.33 (n=5,249) - real,
#     positive, and strong, but the hand-picked value overstated it.
#   - QB_RB_CORR: guessed -0.05 (assumed competing game-script), real ~0.05
#     (n=2,516) - near zero and, if anything, slightly POSITIVE: a QB's own
#     RB's receiving work and shared positive game script outweigh any
#     rush-vs-pass tradeoff in the real data.
#   - SAME_TEAM_CORR: guessed 0.15 (assumed shared pace/Vegas total lifts
#     everyone), real ~0.01 (n=16,222) - essentially zero. A real, offsetting
#     effect was missing from the original assumption: teammates who aren't
#     directly connected by a QB's throws are also competing for the same
#     limited touches/targets, which cancels out most of the shared-total
#     lift.
#   - BRING_BACK_CORR: guessed 0.20, real ~0.09 (n=5,245) - real and
#     positive (the shootout effect is real), but weaker than assumed.
#   - DST_VS_OPPONENT_OFFENSE_CORR: guessed -0.30, real ~-0.51 (n=1,664) -
#     real, negative, and considerably STRONGER than assumed: a defense
#     suppressing its opponent's offensive scoring is a bigger real effect
#     than the original guess gave it credit for.
# All five real sample sizes clear MIN_PAIRS_FOR_FITTED_CORRELATION (200) by
# a wide margin. Re-run fit_real_correlation_matrix() periodically as more
# real seasons accumulate rather than treating these as permanently fixed.
QB_PASS_CATCHER_CORR = 0.3284
QB_RB_CORR = 0.0499
SAME_TEAM_CORR = 0.0136
BRING_BACK_CORR = 0.0872
DST_VS_OPPONENT_OFFENSE_CORR = -0.5073

PERCENTILE_POINTS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])


def _player_correlation(a, b):
    if a["player_id"] == b["player_id"]:
        return 1.0

    same_team = a["team"] == b["team"]
    opponents = a["team"] == b["opponent"] or b["team"] == a["opponent"]
    # real_position, not position: on a Showdown slate DK labels every row's
    # position "CPT" or "FLEX" regardless of the real player's football
    # position (data/player_crosswalk.py's SHOWDOWN_PSEUDO_POSITIONS) - see
    # _load_players, which resolves this from each player's own history
    # before this function ever sees them. Trusting raw "position" here
    # would silently collapse every Showdown pair to the same-team/no-
    # correlation fallback below, exactly the failure this real, fitted
    # correlation matrix exists to avoid.
    positions = {a["real_position"], b["real_position"]}

    if same_team:
        if "QB" in positions and positions & {"WR", "TE"}:
            return QB_PASS_CATCHER_CORR
        if positions == {"QB", "RB"}:
            return QB_RB_CORR
        if "DST" in positions:
            return 0.0
        return SAME_TEAM_CORR

    if opponents:
        if "QB" in positions and positions & {"WR", "TE"}:
            return BRING_BACK_CORR
        if "DST" in positions and len(positions) > 1:
            return DST_VS_OPPONENT_OFFENSE_CORR

    return 0.0


def _standard_normal_cdf(x):
    # Abramowitz & Stegun 7.1.26 erf approximation (max error ~1.5e-7), vectorized
    # over numpy arrays - used instead of adding scipy as a dependency for one
    # function.
    sign = np.sign(x)
    x_abs = np.abs(x) / np.sqrt(2)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(x_abs**2))
    return 0.5 * (1.0 + sign * y)


def _correlated_percentile_ranks(players, num_simulations, seed):
    n = len(players)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = _player_correlation(players[i], players[j])

    rng = np.random.default_rng(seed)
    try:
        z = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=num_simulations)
    except np.linalg.LinAlgError:
        # The rule-based matrix above isn't guaranteed positive semi-definite for
        # every possible player combination - fall back to independent draws
        # rather than fail the whole simulation over a modeling edge case.
        z = rng.standard_normal((num_simulations, n))
    # Gaussian copula: correlated standard normals -> uniform(0,1) percentile
    # ranks via the standard normal CDF, so each player's own marginal
    # distribution (their stored percentile ladder) is preserved while the
    # *joint* draws carry the stacking correlations above.
    return _standard_normal_cdf(z)


def _quantile_to_scores(percentile_ranks, percentiles):
    ys = np.array([percentiles[str(int(p * 100))] for p in PERCENTILE_POINTS])
    scores = np.interp(percentile_ranks, PERCENTILE_POINTS, ys)
    # np.interp flat-clamps outside [0.10, 0.90]; extend the outermost segment's
    # slope instead so extreme boom/bust draws aren't all clipped to one score.
    low_slope = (ys[1] - ys[0]) / (PERCENTILE_POINTS[1] - PERCENTILE_POINTS[0])
    high_slope = (ys[-1] - ys[-2]) / (PERCENTILE_POINTS[-1] - PERCENTILE_POINTS[-2])
    below = percentile_ranks < PERCENTILE_POINTS[0]
    above = percentile_ranks > PERCENTILE_POINTS[-1]
    scores = np.where(below, ys[0] + low_slope * (percentile_ranks - PERCENTILE_POINTS[0]), scores)
    scores = np.where(above, ys[-1] + high_slope * (percentile_ranks - PERCENTILE_POINTS[-1]), scores)
    return np.clip(scores, 0.0, None)


def _resolve_real_positions(rows, engine):
    """{player_id: real_position} for every row - the actual football
    position for a Showdown row (DK labels these "CPT"/"FLEX" regardless of
    the real player's position, see SHOWDOWN_PSEUDO_POSITIONS), resolved
    from that player's own recent history the same way data/pre_lock_check.py's
    hard_role_exclusions/build_lineup_role_checklist already do - reusing
    that exact machinery rather than a third, potentially-drifting
    reimplementation. A Classic row's position is already real and passes
    through unchanged, no history lookup needed.
    """
    pseudo_rows = [dict(row) for row in rows if row["position"] in SHOWDOWN_PSEUDO_POSITIONS]
    real_position_by_id = {row["player_id"]: row["position"] for row in rows if row["position"] not in SHOWDOWN_PSEUDO_POSITIONS}
    if not pseudo_rows:
        return real_position_by_id

    # resolve_dk_players_to_gsis needs name/team for its Showdown DST-
    # nickname and name-only matching - not selected by _load_players'
    # own query (which only needs position/team/opponent/percentiles for
    # everything else), so fetched separately here, only for the rows that
    # actually need it.
    with engine.connect() as conn:
        name_rows = conn.execute(
            text("SELECT player_id, name, team FROM slate_player_pool WHERE player_id = ANY(:ids)"),
            {"ids": [row["player_id"] for row in pseudo_rows]},
        ).mappings().fetchall()
    name_by_id = {row["player_id"]: row for row in name_rows}
    for row in pseudo_rows:
        row.update(name_by_id.get(row["player_id"], {}))

    gsis_by_dk_id, _, _ = resolve_dk_players_to_gsis(pseudo_rows, engine)
    games_by_gsis = _load_recent_usage_batch(
        [gid for gid in gsis_by_dk_id.values() if gid is not None], engine
    )
    for row in pseudo_rows:
        gsis_id = gsis_by_dk_id.get(row["player_id"])
        games = games_by_gsis.get(gsis_id, []) if gsis_id is not None else []
        real_position_by_id[row["player_id"]] = games[0]["position"] if games else None

    return real_position_by_id


def _load_players(slate_id, player_ids, engine):
    query = text(
        """
        SELECT sp.player_id, sp.position, sp.team, sp.opponent, proj.proj_percentiles
        FROM slate_player_pool sp
        JOIN projections proj ON proj.slate_id = sp.slate_id AND proj.player_id = sp.player_id
        WHERE sp.slate_id = :slate_id AND sp.player_id = ANY(:player_ids)
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"slate_id": slate_id, "player_ids": list(player_ids)}).mappings().fetchall()

    real_position_by_id = _resolve_real_positions(rows, engine)

    players_by_id = {}
    for row in rows:
        percentiles = row["proj_percentiles"]
        if isinstance(percentiles, str):
            percentiles = json.loads(percentiles)
        players_by_id[row["player_id"]] = {
            "player_id": row["player_id"],
            "position": row["position"],
            "real_position": real_position_by_id.get(row["player_id"]),
            "team": row["team"],
            "opponent": row["opponent"],
            "percentiles": percentiles,
        }

    missing = [pid for pid in player_ids if pid not in players_by_id]
    if missing:
        raise ValueError(f"Missing projections for player(s): {missing}")
    return players_by_id


def simulate_lineups(lineups, slate_id, num_simulations=DEFAULT_NUM_SIMULATIONS, seed=None, engine=None):
    engine = engine or get_engine()

    player_ids = sorted({p["player_id"] for lu in lineups for _, p in lu["roster"]})
    if not player_ids:
        raise ValueError("No players found across the given lineups")

    players_by_id = _load_players(slate_id, player_ids, engine)
    ordered_players = [players_by_id[pid] for pid in player_ids]

    percentile_ranks = _correlated_percentile_ranks(ordered_players, num_simulations, seed)
    scores_by_player_id = {
        player["player_id"]: _quantile_to_scores(percentile_ranks[:, j], player["percentiles"])
        for j, player in enumerate(ordered_players)
    }

    results = []
    for lu in lineups:
        ids = [p["player_id"] for _, p in lu["roster"]]
        lineup_scores = sum(scores_by_player_id[pid] for pid in ids)
        results.append(
            {
                "roster": lu["roster"],
                "scores": lineup_scores,
                "mean_score": float(np.mean(lineup_scores)),
                "stdev_score": float(np.std(lineup_scores, ddof=1)),
                "p10": float(np.percentile(lineup_scores, 10)),
                "p50": float(np.percentile(lineup_scores, 50)),
                "p90": float(np.percentile(lineup_scores, 90)),
            }
        )
    return results


def win_rates(sim_results):
    # Fraction of simulated outcomes where each lineup scored the best among the
    # given candidates - a relative equity metric across your own lineups, not a
    # true field-of-hundreds contest simulation (we don't have opponent lineup
    # data), but it directly answers "which of these lineups gives me the best
    # shot" the same way GPP-focused tools rank candidate lineups.
    stacked = np.vstack([r["scores"] for r in sim_results])
    winner_idx = np.argmax(stacked, axis=0)
    counts = np.bincount(winner_idx, minlength=len(sim_results))
    return [count / stacked.shape[1] for count in counts]


def select_best_by_simulation(candidate_lineups, slate_id, num_simulations=DEFAULT_NUM_SIMULATIONS, seed=None, engine=None):
    """Rank several already-built, already-legal candidate lineups by real
    simulated relative win rate instead of raw linear-objective ceiling
    sum - the actual selection mechanism a real simulation-driven
    optimizer uses (score correlated outcomes, then pick whichever lineup
    wins the most simulated worlds), layered on top of this codebase's
    MILP-generated candidates rather than requiring a full simulation-
    native rebuild of the optimizer itself.

    Why this matters: models/optimizer.py's solver picks the single
    highest-ceiling-sum roster under a LINEAR objective - it has no way to
    directly compare two similarly-projected but differently-stacked
    lineups on which one actually wins more often once real player
    correlation is accounted for (that needs the quadratic/joint-outcome
    reasoning a MILP can't do, which is exactly what simulate_lineups
    already provides). This is the missing selection step: build a few
    diverse legal candidates (e.g. via build_lineups_from_pool's
    min_uniques diversity, so they're meaningfully different bets, not
    near-duplicates), simulate them all against the SAME correlated
    worlds, and let the real win rate - not the raw projection sum -
    decide which one you actually enter.

    `candidate_lineups` needs at least 2 entries - ranking a single lineup
    against itself isn't a real selection. Returns candidates sorted
    best-to-worst by win rate, each annotated with the real numbers the
    ranking was based on (win_rate, mean_score, p10/p50/p90) - not a
    silent re-order, so a caller can see why one lineup beat another.
    """
    if len(candidate_lineups) < 2:
        raise ValueError(
            f"Need at least 2 candidate lineups to rank by simulation, got {len(candidate_lineups)}"
        )

    sim_results = simulate_lineups(
        candidate_lineups, slate_id, num_simulations=num_simulations, seed=seed, engine=engine
    )
    rates = win_rates(sim_results)

    ranked = sorted(zip(candidate_lineups, sim_results, rates), key=lambda triple: triple[2], reverse=True)
    return [
        {
            "lineup": lineup,
            "win_rate": round(rate, 4),
            "mean_score": round(result["mean_score"], 2),
            "p10": round(result["p10"], 2),
            "p50": round(result["p50"], 2),
            "p90": round(result["p90"], 2),
        }
        for lineup, result, rate in ranked
    ]
