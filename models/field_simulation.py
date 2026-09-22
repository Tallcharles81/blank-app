import math
from collections import Counter

import numpy as np

from models.optimizer import SALARY_CAP, _load_player_pool, build_lineups_from_pool
from models.payout import estimate_payout_from_ranks, load_contest_payout
from models.simulation import DEFAULT_NUM_SIMULATIONS, simulate_lineups

# ---------------------------------------------------------------------------
# This is new and unvalidated - section 5 of the original spec, never built or
# tested before now. Two things in here are explicitly NOT real data and must
# stay labeled as such everywhere they surface:
#
#   1. Ownership. We have no live ownership feed. OWNERSHIP_PROXY below is
#      "projected points per $1,000 salary" - a standard, real DFS value
#      heuristic (this exact ratio is what most public DFS tools call
#      "value"), used here as a STAND-IN for ownership because real GPP
#      ownership does correlate with value plays. It is not measured
#      ownership and every output built from it is labeled
#      "ownership_proxy", never "ownership".
#
#   2. The field itself. There is no real historical record of what lineups
#      actual opponents build. Opponent rosters are synthetic, sampled from
#      the same live, availability-gated player pool real opponents would
#      also be drawing from, weighted toward the ownership proxy via the
#      Gumbel-max trick (see _draw_opponent_lineup) so they cluster around
#      likely-popular builds without every simulated opponent being
#      identical. This is a real, principled sampling technique - argmax of
#      (log-weight + i.i.d. Gumbel noise) samples from softmax(log-weight)
#      for an unconstrained categorical choice - but under DK's salary-cap/
#      roster-construction constraints it's an approximation, not an exact
#      draw from a true joint field distribution (no such distribution exists
#      without real ownership data to fit it against). CONCENTRATION controls
#      how much the sampling clusters toward high-value players; there is no
#      "right" value for it without real ownership data to calibrate against.
#
# VALIDATION RESULT (run_validation_comparison, live pool, seed=7): the
# overlap mechanism is real but concentration-dependent in a way that matters
# a lot. A chalk lineup and a contrarian lineup matched to within 5% of the
# same total projection (170.8 vs chalk's 178.6, 6/9 players different) were
# compared at concentration 1/3/8/15/25 and contest_size 100/500/2000:
#   - At concentration 1-8 (this module's original default), chalk's field
#     overlap (avg_shared_players) WAS consistently higher than contrarian's,
#     as expected - but chalk's win_pct came back HIGHER than contrarian's at
#     every one of those settings and every contest_size tried, the OPPOSITE
#     of the expected "chalk loses to duplication" signature. A ~4.6%
#     projection edge simply dominated the overlap penalty at those settings.
#   - At concentration 15-25, the field became concentrated enough
#     (avg_shared_players ~5-5.3 of 9, max_shared up to 8 of 9) that the
#     ordering flipped correctly: contrarian's win_pct exceeded chalk's
#     (0.0234 vs 0.0129 at concentration=15; 0.0747 vs 0.0616 at 25) - the
#     expected result, confirming the underlying mechanism (shared players ->
#     correlated scores -> fewer worlds where "my" score is the sole max) is
#     implemented correctly.
#   - EXACT full-roster duplication_rate was 0.0 in every single test run
#     (concentration 1 through 25, contest_size up to 2000) - a 9-slot exact
#     match essentially never happens by chance even in a fairly concentrated
#     field. avg_shared_players/max_shared_players is what actually carries
#     the signal; duplication_rate as literally specified should be expected
#     to read ~0 in most realistic runs and must not be read as "no overlap."
#
# Conclusion: the mechanism is validated, the calibration is not. Without a
# real ownership feed there is no principled way to know what concentration
# value corresponds to an actual DK slate's real field behavior - this
# implementation can demonstrate the effect exists but cannot yet tell you
# whether today's actual field looks more like concentration=1 (where chalk's
# edge wins) or concentration=25 (where it doesn't). DEFAULT_CONCENTRATION is
# left at 1.0 deliberately - bumping it to "make the effect show up" would be
# tuning a free parameter to produce a desired conclusion, exactly what this
# project's conventions rule out.
# ---------------------------------------------------------------------------

DEFAULT_CONTEST_SIZE = 100
DEFAULT_CONCENTRATION = 1.0  # UNCALIBRATED - see VALIDATION RESULT above
_VALUE_FLOOR = 0.01  # avoids log(0) for a zero-projected player without meaningfully affecting sampling


def ownership_proxy(players):
    """projected points per $1,000 salary - a real, standard DFS "value"
    metric, used here as a stand-in for ownership because we have no live
    ownership feed. NOT real ownership - label every downstream use of this
    as "ownership_proxy", never "ownership". Expects players loaded via
    optimizer._load_player_pool(..., "proj_median", ...), whose query aliases
    the requested projection field to "points".
    """
    return {p["player_id"]: max(p["points"] / (p["salary"] / 1000.0), _VALUE_FLOOR) for p in players}


# Real, measured correction to the raw stand-in above - see
# data/ownership_calibration.py for the full real-data pipeline this comes
# from. UPDATED with genuine cross-week validation: the previous version of
# this table was fit from real contests on dk_thu_mon_2026_09_17 alone,
# explicitly flagged as "not yet cross-week validated." Two more real
# contests (195661326, 195661349) on a DIFFERENT real slate/week
# (dk_sunday_2026_09_20) have since been played and imported - this is that
# update. For each real matched player, mean(real %Drafted) / mean(raw
# ownership_proxy) per position, by real slate:
#   dk_thu_mon_2026_09_17 (n=76-262/position): QB 1.18 RB 2.45 WR 2.05 TE
#     1.26 DST 2.37 (the prior table, reproduced exactly from this slate's
#     own real data)
#   dk_sunday_2026_09_20 (n=52-255/position, genuinely different week):
#     QB 1.15 RB 2.52 WR 1.88 TE 1.25 DST 2.08
# QB/TE landed almost identically across two independent real weeks - real
# cross-week confirmation, not just within-week replication. RB/WR/DST held
# the same real direction and rough magnitude (all still badly under-
# predicted by raw points-per-$1000) but moved enough between weeks that
# pooling both real slates, not trusting either alone, is the better
# estimate now that both exist.
#
# UPDATED AGAIN with 2 more real Classic contests on the same
# dk_sunday_2026_09_20 slate (195866131, 195658397) - a third real,
# independent data point, not just more of the same contest. Pooled ratio
# across ALL real Classic contests to date (n=200-730/position): QB 1.19
# RB 2.55 WR 2.00 TE 1.28 DST 2.17 - all within a few hundredths of the
# prior table, real convergence rather than drift, the sign this estimate
# is stabilizing as more real weeks accumulate rather than still hunting
# for its true value.
#
# REAL METHODOLOGICAL FINDING while doing this update, worth preserving:
# a real Showdown contest (195910196, dk_showdown_ind_kc_2026_09_20) was
# ALSO imported alongside these, and naively pooling its real ratios in
# with the Classic ones distorted every position badly (e.g. WR's pooled
# ratio jumping from 2.00 to 2.13, RB from 2.55 to 2.78) despite Showdown
# contributing only 2-11 real matched players per position. A Showdown
# pool's real ownership_proxy scale is NOT comparable to a Classic pool's
# (far fewer real players compete for the same %Drafted share, and
# CPT/FLEX pricing changes the raw points-per-$1000 math entirely) - this
# table must only ever be fit from real Classic-slate contests. A future
# Showdown-specific calibration, if ever built, needs its own separate
# table and enough real Showdown contests to trust it, not a blend with
# this one.
#
# This table is expected to keep moving as more real Classic weeks
# accumulate, per this module's own accumulating-dataset design; re-run
# data/ownership_calibration.py's summarize_ownership_calibration (Classic
# slates only) before assuming these are still current.
POSITION_OWNERSHIP_CALIBRATION = {
    "QB": 1.19,
    "RB": 2.55,
    "WR": 2.00,
    "TE": 1.28,
    "DST": 2.17,
}


def calibrated_ownership_proxy(players):
    """ownership_proxy(), scaled per-position by POSITION_OWNERSHIP_
    CALIBRATION - still NOT real ownership (still a stand-in, just a real-
    data-corrected one), and every downstream use should still say
    "ownership_proxy" / "calibrated_ownership_proxy", never "ownership".
    A position missing from the calibration table (shouldn't happen for a
    real DK Classic pool - all 5 real positions are covered) falls back to
    an unscaled 1.0 rather than raising, so a genuinely novel position label
    degrades to the uncalibrated estimate instead of breaking lineup
    construction.
    """
    raw = ownership_proxy(players)
    return {
        p["player_id"]: raw[p["player_id"]] * POSITION_OWNERSHIP_CALIBRATION.get(p["position"], 1.0) for p in players
    }


def _draw_opponent_lineup(players, proxy_by_id, concentration, rng):
    # Gumbel-max trick: argmax(theta_i + Gumbel(0,1)_i) samples from
    # softmax(theta). theta_i = concentration * log(value_score_i) so a
    # higher concentration sharpens the draw toward the highest-value
    # players (more field agreement, more duplication) and a lower one
    # flattens it toward uniform (more spread, less duplication) - see the
    # module docstring for why this is an approximation once the salary cap/
    # roster constraints are applied, not an exact joint field draw.
    gumbel_noise = rng.gumbel(size=len(players))
    perturbed = [dict(p) for p in players]
    for p, noise in zip(perturbed, gumbel_noise):
        p["points"] = concentration * math.log(proxy_by_id[p["player_id"]]) + noise

    lineups, _ = build_lineups_from_pool(perturbed, num_lineups=1, salary_cap=SALARY_CAP)
    return lineups[0]


def generate_opponent_lineups(players, contest_size, concentration=DEFAULT_CONCENTRATION, seed=None, proxy_fn=calibrated_ownership_proxy):
    """contest_size independently-drawn, salary-cap/roster-feasible opponent
    lineups from the same player pool, weighted toward proxy_fn's output via
    _draw_opponent_lineup. Returns (lineups, failed_draws) - a draw can fail
    if that particular random perturbation makes the ILP infeasible for some
    edge-case combination; those are skipped and counted, mirroring how
    models/backtest.py's random-field generation already handles ValueError
    from an infeasible solve rather than failing the whole run over one draw.

    proxy_fn defaults to calibrated_ownership_proxy (the real, position-
    corrected estimate) rather than the raw ownership_proxy stand-in - pass
    proxy_fn=ownership_proxy explicitly to compare against the uncalibrated
    behavior.
    """
    proxy_by_id = proxy_fn(players)
    rng = np.random.default_rng(seed)

    opponents = []
    failed = 0
    for _ in range(contest_size):
        try:
            opponents.append(_draw_opponent_lineup(players, proxy_by_id, concentration, rng))
        except ValueError:
            failed += 1
    return opponents, failed


def _roster_id_set(lineup):
    return frozenset(p["player_id"] for _, p in lineup["roster"])


def run_field_simulation(
    my_lineup,
    slate_id,
    contest_size=DEFAULT_CONTEST_SIZE,
    concentration=DEFAULT_CONCENTRATION,
    num_simulations=DEFAULT_NUM_SIMULATIONS,
    seed=None,
    engine=None,
    projection_field="proj_median",
    proxy_fn=calibrated_ownership_proxy,
    contest_id=None,
):
    """Simulate a `contest_size`-lineup GPP field around `my_lineup` and score
    everyone against the SAME num_simulations simulated worlds (one call to
    simulate_lineups over [my_lineup] + opponents, not independently re-
    simulated per lineup - required so a duplicate opponent actually produces
    identical scores in every world, which is what makes the duplication
    effect on win% real rather than coincidental).

    win_pct: fraction of worlds where my_lineup's score is STRICTLY greater
    than every opponent's score that world. A world where an exact-duplicate
    opponent ties my_lineup's score counts as tie_pct, not a win - the
    mechanism that, in principle, should make a heavily-overlapping "chalk"
    lineup's win_pct come in lower than an equally-projected, less-owned one.
    In practice this only shows up once the simulated field is concentrated
    enough (see the module docstring's VALIDATION RESULT) - at low/moderate
    concentration a modest real projection edge can dominate it, so a chalk
    lineup coming back with a HIGHER win_pct than a lower-owned one is not
    automatically a bug; check avg_shared_players/max_shared_players and the
    concentration used before concluding either way.

    Returns a dict including ownership_proxy_top (the field's own realized
    selection frequency per player - a direct, measured output of the
    sampling, not a separately asserted ownership number) so the sampling
    itself can be sanity-checked.

    `contest_id`, if given, adds a "payout" key: real per-world finish rank
    (1 + how many opponents strictly outscored my_lineup that world) run
    through models.payout's stored curve for that real contest (raises
    ValueError via load_contest_payout if it was never imported). This is a
    genuinely different, complementary question from win_pct/tie_pct/loss_pct
    above: those answer "do I beat the WHOLE field" (a single-winner framing
    that only strictly matters for a winner-take-all contest), while `payout`
    answers "how does this lineup do under this contest's REAL multi-place
    payout structure," including honestly reporting when that structure is
    only partially known (see models/payout.py's module docstring).

    One real approximation this carries that a caller must weigh, not just
    take on faith: the simulated field here has `contest_size` opponents,
    which may be far smaller than the real contest's own total_entries
    (large single-entry GPPs commonly run into the thousands) - a rank
    computed against a much smaller simulated field is an easier bar to
    clear than the real one, so payout.cash_pct will read optimistic unless
    `contest_size` is set close to the real contest's own total_entries. See
    payout.field_size_note.
    """
    players, _, _, _ = _load_player_pool(slate_id, projection_field, engine=engine)
    opponents, failed_draws = generate_opponent_lineups(players, contest_size, concentration, seed, proxy_fn=proxy_fn)
    if not opponents:
        raise RuntimeError(f"Could not generate any feasible opponent lineups (all {contest_size} draws failed)")

    my_ids = _roster_id_set(my_lineup)
    exact_duplicates = sum(1 for opp in opponents if _roster_id_set(opp) == my_ids)
    duplication_rate = exact_duplicates / len(opponents)
    # Exact full-9-slot duplication is a high bar (roughly the product of each
    # slot's individual match probability) and can legitimately be ~0 for a
    # 100-lineup field even when the sampling is working correctly - shared-
    # player overlap (not requiring every slot to match) is tracked too,
    # since that's the softer, more common real mechanism behind correlated
    # scores between "my" lineup and an opponent's.
    overlap_counts = [len(my_ids & _roster_id_set(opp)) for opp in opponents]
    avg_shared_players = sum(overlap_counts) / len(overlap_counts)
    max_shared_players = max(overlap_counts)

    sim_results = simulate_lineups([my_lineup] + opponents, slate_id, num_simulations=num_simulations, seed=seed, engine=engine)
    my_scores = sim_results[0]["scores"]
    opponent_scores = np.vstack([r["scores"] for r in sim_results[1:]])
    opponent_best = opponent_scores.max(axis=0)

    win_pct = float(np.mean(my_scores > opponent_best))
    tie_pct = float(np.mean(my_scores == opponent_best))
    loss_pct = 1.0 - win_pct - tie_pct

    realized_selection_counts = Counter()
    for opp in opponents:
        for _, p in opp["roster"]:
            realized_selection_counts[p["player_id"]] += 1
    names_by_id = {p["player_id"]: p["name"] for p in players}
    ownership_proxy_top = [
        {"player_id": pid, "name": names_by_id.get(pid, pid), "field_selection_rate": count / len(opponents)}
        for pid, count in realized_selection_counts.most_common(10)
    ]

    payout = None
    if contest_id is not None:
        contest = load_contest_payout(contest_id, engine=engine)
        if contest["slate_id"] is not None and contest["slate_id"] != slate_id:
            raise ValueError(
                f"contest_id={contest_id} was imported for slate_id={contest['slate_id']!r}, not {slate_id!r} - "
                "refusing to score a real contest's payout curve against a different slate's simulated field"
            )
        my_rank_per_world = 1 + np.sum(opponent_scores > my_scores, axis=0)
        payout = estimate_payout_from_ranks(
            my_rank_per_world.tolist(),
            contest["tiers"],
            places_paid=contest["places_paid"],
            entry_fee=contest["entry_fee"],
        )
        payout["structure_complete"] = contest["structure_complete"]
        payout["simulated_field_size"] = len(opponents)
        payout["real_total_entries"] = contest["total_entries"]
        payout["field_size_note"] = (
            "simulated_field_size vs real_total_entries - cash_pct is only a faithful estimate of "
            "the real contest once these are close; a much smaller simulated field makes cashing look "
            "easier than it really is."
        )

    return {
        "contest_size_requested": contest_size,
        "contest_size_actual": len(opponents),
        "failed_draws": failed_draws,
        "concentration": concentration,
        "exact_duplicate_opponents": exact_duplicates,
        "duplication_rate": round(duplication_rate, 4),
        "avg_shared_players": round(avg_shared_players, 3),
        "max_shared_players": max_shared_players,
        "win_pct": round(win_pct, 4),
        "tie_pct": round(tie_pct, 4),
        "loss_pct": round(loss_pct, 4),
        "my_median_score": round(float(np.median(my_scores)), 2),
        "ownership_proxy_top": ownership_proxy_top,
        "payout": payout,
    }


def build_chalk_lineup(players):
    """The single highest-proj_median roster achievable under DK's real
    salary-cap/roster rules (players already carry proj_median as "points"
    from _load_player_pool) - the "obvious best play" build a real GPP field
    gravitates toward. This was originally (wrongly) built by maximizing
    total ownership_proxy instead of total points, which doesn't need to
    spend anywhere near the cap - it produced a $12,650, 33-point dart-throw
    roster, not a real chalk lineup, and made the validation comparison
    meaningless (it would have "won" less just for being a worse lineup, not
    for being more duplicated). Fixed to maximize proj_median directly, same
    as the live GPP pipeline would.
    """
    lineups, _ = build_lineups_from_pool(list(players), num_lineups=1, salary_cap=SALARY_CAP)
    return lineups[0]


def build_contrarian_lineup(players, lambda_penalty, proxy_fn=calibrated_ownership_proxy):
    """Maximize (proj_median - lambda_penalty * proxy_fn(players)) under the
    same cap/roster rules - a continuously tunable projection-vs-ownership
    trade-off, not a hard top-N cutoff. A hard cutoff (the first version of
    this function, exclude_top_n) was tried first and rejected: removing the
    top 15 ownership_proxy players cost ~12% of total projection (178.6 ->
    157.2), which meant chalk wasn't just more-owned than contrarian, it was
    also a meaningfully BETTER lineup - a real confound that made chalk look
    like it won more DESPITE overlap, when the comparison wasn't actually
    equally-projected in the first place. The penalty form lets
    _match_contrarian_projection search for a lambda that keeps projection
    close to chalk's while still shifting composition toward lower-owned
    players.

    proxy_fn defaults to calibrated_ownership_proxy so "lower-owned" means
    real, position-corrected leverage (a real cheap DST/RB/WR the raw
    formula underrates ownership for is worth less penalty here than the raw
    stand-in would give it; a real "best-value" QB is worth less penalty
    than its raw proxy rank alone would suggest, since the real field
    doesn't concentrate QB ownership the way it does RB/WR/DST) rather than
    the uncalibrated points-per-dollar heuristic.
    """
    proxy_by_id = proxy_fn(players)
    pool = [{**p, "points": p["points"] - lambda_penalty * proxy_by_id[p["player_id"]]} for p in players]
    lineups, _ = build_lineups_from_pool(pool, num_lineups=1, salary_cap=SALARY_CAP)
    return lineups[0]


# Fine increments, not a handful of round numbers: build_contrarian_lineup's
# objective is piecewise-constant in lambda (an ILP's optimal choice only
# switches at specific threshold values), confirmed by hand on the live pool -
# 178.6/178.6/178.0/178.0/.../170.8/160.4 as lambda stepped 1.6->3.0 in 0.1
# increments, with the roster itself jumping 9/9/7/7/.../6/6 players shared
# with chalk at the same points. A coarse grid (the first version of this:
# [0.25, 0.5, 1, 1.5, 2, 3, ...]) jumped clean over the one useful plateau
# (~96% of chalk's projection, 6/9 shared) straight from "identical to chalk"
# to "10% below chalk" - not wrong, just too coarse to find the interesting
# middle ground.
_LAMBDA_SEARCH_GRID = [round(0.1 * i, 2) for i in range(1, 100)]


def _raw_projection(lineup, raw_points_by_id):
    # build_contrarian_lineup's returned roster carries the PENALIZED "points"
    # (proj_median - lambda*proxy), not the true projection - this looks up
    # each selected player's real proj_median instead, from the pool as
    # originally loaded (before any penalty was applied).
    return sum(raw_points_by_id[p["player_id"]] for _, p in lineup["roster"])


def find_contrarian_matching_projection(players, target_projection, tolerance=0.05, proxy_fn=calibrated_ownership_proxy):
    """Search _LAMBDA_SEARCH_GRID for a contrarian lineup within `tolerance`
    of target_projection (chalk's own total) that is ALSO meaningfully
    different in composition - not just the smallest lambda that reproduces
    chalk exactly (gap=0 there, but zero ownership divergence, which defeats
    the point of the comparison). Among every lambda whose projection falls
    within tolerance, picks the one with the LOWEST average proxy_fn value
    (the most contrarian option that still respects the projection
    constraint). Returns (lineup, lambda_used, achieved_projection,
    within_tolerance) - within_tolerance is False only if NOTHING in the grid
    landed within tolerance at all, which must be reported honestly rather
    than silently forcing a comparison that isn't actually apples-to-apples.
    """
    raw_points_by_id = {p["player_id"]: p["points"] for p in players}
    proxy_by_id = proxy_fn(players)

    candidates = []
    closest = None
    for lam in _LAMBDA_SEARCH_GRID:
        lineup = build_contrarian_lineup(players, lam, proxy_fn=proxy_fn)
        projection = _raw_projection(lineup, raw_points_by_id)
        gap = abs(projection - target_projection)
        avg_proxy = sum(proxy_by_id[p["player_id"]] for _, p in lineup["roster"]) / len(lineup["roster"])
        if closest is None or gap < closest[3]:
            closest = (lineup, lam, projection, gap)
        if gap <= tolerance * target_projection:
            candidates.append((lineup, lam, projection, avg_proxy))

    if not candidates:
        lineup, lam, projection, gap = closest
        return lineup, lam, projection, False

    lineup, lam, projection, _ = min(candidates, key=lambda c: c[3])
    return lineup, lam, projection, True


def _percentile_rank_within_position(players, value_by_id):
    """0-1 percentile rank of value_by_id, computed SEPARATELY within each
    position group (average rank for ties, midpoint 0.5 for a singleton
    group). Positions can't be compared on raw ceiling or raw ownership_proxy
    units - a top-5 TE's proj_ceiling is nowhere near a top-5 WR's in
    absolute points, and DST's calibrated ownership_proxy sits on its own
    scale entirely - so leverage_score below needs "highest ceiling for a
    RB" and "highest ceiling for a TE" to mean the same 1.0, not compare raw
    magnitudes across positions.
    """
    by_position = {}
    for p in players:
        by_position.setdefault(p["position"], []).append(p["player_id"])

    pct_by_id = {}
    for pids in by_position.values():
        n = len(pids)
        if n == 1:
            pct_by_id[pids[0]] = 0.5
            continue
        order = sorted(pids, key=lambda pid: value_by_id[pid])
        ranks = {}
        i = 0
        while i < n:
            j = i
            while j + 1 < n and value_by_id[order[j + 1]] == value_by_id[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        for pid in pids:
            pct_by_id[pid] = ranks[pid] / (n - 1)
    return pct_by_id


def leverage_score(players, proxy_fn=calibrated_ownership_proxy):
    """Per-player leverage rating: (within-position percentile rank of
    ceiling upside) minus (within-position percentile rank of ownership_
    proxy). Range [-1, 1]. Positive = this player's ceiling ranks higher at
    their position than their ownership_proxy does (real, testable
    "leverage": upside the field is under-rostering relative to); negative =
    the reverse (a chalky, ceiling-thin play).

    "Ceiling upside" is whatever projection field the caller loaded players
    with (players[i]["points"], per _load_player_pool's field-aliasing
    convention) - pass players loaded with projection_field="proj_ceiling"
    to make this a real ceiling-vs-ownership leverage score; loading with
    proj_median instead would make it a median-vs-ownership score, a
    different (weaker) claim. run_leverage_validation_comparison always
    loads with proj_ceiling for this reason.

    This is a SCORING signal only - it does not by itself change lineup
    construction. See build_leverage_lineup for the one place it gets turned
    into a selection bias, and run_leverage_validation_comparison for the
    backtest gating whether that bias is ever used as a default.
    """
    ceiling_by_id = {p["player_id"]: p["points"] for p in players}
    proxy_by_id = proxy_fn(players)
    ceiling_pct = _percentile_rank_within_position(players, ceiling_by_id)
    proxy_pct = _percentile_rank_within_position(players, proxy_by_id)
    return {p["player_id"]: ceiling_pct[p["player_id"]] - proxy_pct[p["player_id"]] for p in players}


def build_leverage_lineup(players, lambda_boost, proxy_fn=calibrated_ownership_proxy):
    """Maximize (points + lambda_boost * leverage_score(players)) under the
    same cap/roster rules as build_chalk_lineup/build_contrarian_lineup -
    directly REWARDS real leverage (high ceiling rank relative to low
    ownership rank, within position) rather than build_contrarian_lineup's
    approach of penalizing raw ownership_proxy regardless of ceiling. The two
    are not the same lineup in general: a player can have below-average
    ownership_proxy but also below-average ceiling (contrarian likes them,
    leverage doesn't), or above-average ownership with an even
    higher-ranking ceiling (leverage tolerates them more than contrarian
    would, if their upside outranks their ownership by enough).
    """
    scores = leverage_score(players, proxy_fn=proxy_fn)
    pool = [{**p, "points": p["points"] + lambda_boost * scores[p["player_id"]]} for p in players]
    lineups, _ = build_lineups_from_pool(pool, num_lineups=1, salary_cap=SALARY_CAP)
    return lineups[0]


# Same fine-grained grid as _LAMBDA_SEARCH_GRID, for the same reason -
# build_leverage_lineup's objective is piecewise-constant in lambda_boost, so
# a coarse grid can jump clean over the useful plateau between "identical to
# chalk" and "wildly under-projected."
_LEVERAGE_LAMBDA_SEARCH_GRID = [round(0.5 * i, 2) for i in range(1, 100)]


def find_leverage_matching_projection(players, target_projection, tolerance=0.05, proxy_fn=calibrated_ownership_proxy):
    """find_contrarian_matching_projection's counterpart for build_leverage_
    lineup: search _LEVERAGE_LAMBDA_SEARCH_GRID for a leverage-biased lineup
    within `tolerance` of target_projection that is ALSO meaningfully
    different in composition. Among every lambda_boost whose real projection
    (raw points, ignoring the leverage boost) falls within tolerance, picks
    the one with the HIGHEST average leverage_score (the most leverage-
    tilted option that still respects the projection constraint) - the
    mirror image of find_contrarian_matching_projection's "lowest average
    proxy" selection.
    """
    raw_points_by_id = {p["player_id"]: p["points"] for p in players}
    scores = leverage_score(players, proxy_fn=proxy_fn)

    candidates = []
    closest = None
    for lam in _LEVERAGE_LAMBDA_SEARCH_GRID:
        lineup = build_leverage_lineup(players, lam, proxy_fn=proxy_fn)
        projection = _raw_projection(lineup, raw_points_by_id)
        gap = abs(projection - target_projection)
        avg_score = sum(scores[p["player_id"]] for _, p in lineup["roster"]) / len(lineup["roster"])
        if closest is None or gap < closest[3]:
            closest = (lineup, lam, projection, gap)
        if gap <= tolerance * target_projection:
            candidates.append((lineup, lam, projection, avg_score))

    if not candidates:
        lineup, lam, projection, gap = closest
        return lineup, lam, projection, False

    lineup, lam, projection, _ = max(candidates, key=lambda c: c[3])
    return lineup, lam, projection, True


def run_leverage_validation_comparison(
    slate_id,
    contest_size=DEFAULT_CONTEST_SIZE,
    concentration=DEFAULT_CONCENTRATION,
    num_simulations=DEFAULT_NUM_SIMULATIONS,
    seed=None,
    engine=None,
    projection_field="proj_ceiling",
    proxy_fn=calibrated_ownership_proxy,
):
    """run_validation_comparison's counterpart for the leverage-score-biased
    lineup: build chalk, the EXISTING ownership-penalty contrarian lineup,
    and the NEW leverage-biased lineup, all matched to chalk's own total
    projection, then run all three through the SAME opponent field (same
    seed) so the comparison isn't confounded by different simulated fields.

    projection_field defaults to proj_ceiling (not proj_median, unlike
    run_validation_comparison's own default) - a leverage score defined
    against a ceiling field only means what it claims to mean ("real ceiling
    upside relative to ownership") when the lineups being compared were
    actually built to chase ceiling in the first place, which is also the
    real GPP path's own objective (see run_validation_comparison's
    docstring).

    This is a DIAGNOSTIC/backtest function, same as run_validation_
    comparison - it does not change what generate_lineups builds. See
    build_leverage_lineup's own callers (none yet in the live optimizer path)
    for confirmation nothing wires this into construction.
    """
    players, _, _, _ = _load_player_pool(slate_id, projection_field, engine=engine)
    raw_points_by_id = {p["player_id"]: p["points"] for p in players}

    chalk = build_chalk_lineup(players)
    chalk_projection = _raw_projection(chalk, raw_points_by_id)
    contrarian, contrarian_lambda, contrarian_projection, contrarian_matched = find_contrarian_matching_projection(
        players, chalk_projection, proxy_fn=proxy_fn
    )
    leverage, leverage_lambda, leverage_projection, leverage_matched = find_leverage_matching_projection(
        players, chalk_projection, proxy_fn=proxy_fn
    )

    chalk_result = run_field_simulation(
        chalk, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )
    contrarian_result = run_field_simulation(
        contrarian, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )
    leverage_result = run_field_simulation(
        leverage, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )

    chalk_result["total_projection"] = round(chalk_projection, 2)
    chalk_result["roster"] = [(slot, p["name"]) for slot, p in chalk["roster"]]
    contrarian_result["total_projection"] = round(contrarian_projection, 2)
    contrarian_result["roster"] = [(slot, p["name"]) for slot, p in contrarian["roster"]]
    contrarian_result["lambda_used"] = contrarian_lambda
    contrarian_result["projection_matched_within_tolerance"] = contrarian_matched
    leverage_result["total_projection"] = round(leverage_projection, 2)
    leverage_result["roster"] = [(slot, p["name"]) for slot, p in leverage["roster"]]
    leverage_result["lambda_used"] = leverage_lambda
    leverage_result["projection_matched_within_tolerance"] = leverage_matched

    return {
        "chalk": chalk_result,
        "contrarian": contrarian_result,
        "leverage": leverage_result,
    }


def build_ownership_cap_lineup(players, cap, proxy_fn=calibrated_ownership_proxy):
    """Maximize points under the same cap/roster rules as build_chalk_lineup,
    plus one HARD constraint: total proxy_fn(players) across the 9 selected
    players must not exceed `cap` (models/optimizer.py's
    _apply_ownership_cap_constraint, threaded through build_lineups_from_pool
    as ownership_values_by_id/ownership_cap - both default None/no-op, so
    every existing caller is unaffected).

    Structurally different from build_contrarian_lineup/build_leverage_
    lineup: those trade points off against ownership CONTINUOUSLY (a penalty
    or reward term the solver can balance against), so a high-enough-upside
    chalk play can still get picked even at a real ownership cost. A hard
    cap is a genuine constraint - the solver can pick ANY combination under
    the cap, including one that leaves easy points on the table rather than
    touch a single over-cap chalk play - a structurally different real
    question from either existing lever: does directly bounding total
    lineup-level ownership (as opposed to per-player exposure caps across a
    portfolio, or a continuous penalty) change real GPP win_pct.

    Raises ValueError (from build_lineups_from_pool/_solve) if `cap` is too
    tight for ANY legal 9-man roster to fit under it - a real, expected
    outcome at the tight end of a cap sweep, not a bug; callers doing a
    sweep (find_ownership_cap_matching_projection) must catch this.
    """
    ownership_values_by_id = proxy_fn(players)
    lineups, _ = build_lineups_from_pool(
        list(players),
        num_lineups=1,
        salary_cap=SALARY_CAP,
        ownership_values_by_id=ownership_values_by_id,
        ownership_cap=cap,
    )
    return lineups[0]


# Cap expressed as a FRACTION of the chalk lineup's own total ownership
# (find_ownership_cap_matching_projection scales this by the caller's actual
# chalk_total_ownership), not an absolute unit - unlike _LAMBDA_SEARCH_GRID/
# _LEVERAGE_LAMBDA_SEARCH_GRID (both hand-tuned to one slate's raw
# ownership_proxy scale by inspection), a fraction-of-chalk grid is scale-
# invariant across slates of very different size/projection totals without
# per-slate re-tuning. 10%-100% of chalk's own total ownership, in 2% steps.
_OWNERSHIP_CAP_FRACTIONS = [round(0.02 * i, 2) for i in range(5, 51)]


def find_ownership_cap_matching_projection(
    players, target_projection, chalk_total_ownership, tolerance=0.05, proxy_fn=calibrated_ownership_proxy
):
    """find_contrarian_matching_projection's counterpart for build_ownership_
    cap_lineup: search _OWNERSHIP_CAP_FRACTIONS (scaled by chalk_total_
    ownership) for a capped lineup within `tolerance` of target_projection.
    Among every cap whose projection falls within tolerance, picks the
    TIGHTEST cap (most constrained composition) - the mirror of find_
    contrarian_matching_projection's "lowest average proxy" pick.

    A cap tight enough to make the ILP infeasible (ValueError from
    build_ownership_cap_lineup) is skipped, same as a failed opponent draw in
    generate_opponent_lineups - expected at the tight end of the sweep, not
    an error to propagate.
    """
    raw_points_by_id = {p["player_id"]: p["points"] for p in players}
    ownership_values_by_id = proxy_fn(players)

    candidates = []
    closest = None
    for frac in _OWNERSHIP_CAP_FRACTIONS:
        cap = frac * chalk_total_ownership
        try:
            lineup = build_ownership_cap_lineup(players, cap, proxy_fn=proxy_fn)
        except ValueError:
            continue
        projection = _raw_projection(lineup, raw_points_by_id)
        gap = abs(projection - target_projection)
        if closest is None or gap < closest[3]:
            closest = (lineup, cap, projection, gap)
        if gap <= tolerance * target_projection:
            candidates.append((lineup, cap, projection))

    if not candidates:
        if closest is None:
            raise RuntimeError("no feasible lineup found at any cap in _OWNERSHIP_CAP_FRACTIONS")
        lineup, cap, projection, gap = closest
        return lineup, cap, projection, False

    lineup, cap, projection = min(candidates, key=lambda c: c[1])
    return lineup, cap, projection, True


def run_ownership_cap_validation_comparison(
    slate_id,
    contest_size=DEFAULT_CONTEST_SIZE,
    concentration=DEFAULT_CONCENTRATION,
    num_simulations=DEFAULT_NUM_SIMULATIONS,
    seed=None,
    engine=None,
    projection_field="proj_ceiling",
    proxy_fn=calibrated_ownership_proxy,
):
    """run_validation_comparison's counterpart for the hard ownership-cap
    lineup: build chalk, the EXISTING ownership-penalty contrarian lineup,
    and the NEW hard-capped lineup, all matched to chalk's own total
    projection, then run all three through the SAME opponent field (same
    seed).

    This is a DIAGNOSTIC/backtest function, same as run_validation_
    comparison and run_leverage_validation_comparison - it does not change
    what generate_lineups builds. build_ownership_cap_lineup's only caller is
    this function and its own tests; nothing in the live optimizer path
    passes ownership_values_by_id/ownership_cap.
    """
    players, _, _, _ = _load_player_pool(slate_id, projection_field, engine=engine)
    raw_points_by_id = {p["player_id"]: p["points"] for p in players}
    ownership_values_by_id = proxy_fn(players)

    chalk = build_chalk_lineup(players)
    chalk_projection = _raw_projection(chalk, raw_points_by_id)
    chalk_total_ownership = sum(ownership_values_by_id[p["player_id"]] for _, p in chalk["roster"])

    contrarian, contrarian_lambda, contrarian_projection, contrarian_matched = find_contrarian_matching_projection(
        players, chalk_projection, proxy_fn=proxy_fn
    )
    capped, cap_used, capped_projection, capped_matched = find_ownership_cap_matching_projection(
        players, chalk_projection, chalk_total_ownership, proxy_fn=proxy_fn
    )

    chalk_result = run_field_simulation(
        chalk, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )
    contrarian_result = run_field_simulation(
        contrarian, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )
    capped_result = run_field_simulation(
        capped, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )

    chalk_result["total_projection"] = round(chalk_projection, 2)
    chalk_result["total_ownership"] = round(chalk_total_ownership, 3)
    chalk_result["roster"] = [(slot, p["name"]) for slot, p in chalk["roster"]]

    contrarian_result["total_projection"] = round(contrarian_projection, 2)
    contrarian_result["roster"] = [(slot, p["name"]) for slot, p in contrarian["roster"]]
    contrarian_result["lambda_used"] = contrarian_lambda
    contrarian_result["projection_matched_within_tolerance"] = contrarian_matched

    capped_result["total_projection"] = round(capped_projection, 2)
    capped_result["roster"] = [(slot, p["name"]) for slot, p in capped["roster"]]
    capped_result["ownership_cap_used"] = round(cap_used, 3)
    capped_result["ownership_cap_fraction_of_chalk"] = round(cap_used / chalk_total_ownership, 3)
    capped_result["projection_matched_within_tolerance"] = capped_matched

    return {
        "chalk": chalk_result,
        "contrarian": contrarian_result,
        "ownership_cap": capped_result,
    }


def run_validation_comparison(
    slate_id,
    contest_size=DEFAULT_CONTEST_SIZE,
    concentration=DEFAULT_CONCENTRATION,
    num_simulations=DEFAULT_NUM_SIMULATIONS,
    seed=None,
    engine=None,
    projection_field="proj_median",
    proxy_fn=calibrated_ownership_proxy,
):
    """The validation check this module needs before being trusted: build a
    chalk lineup and a similarly-projected contrarian lineup from the SAME
    live pool, run both through the SAME opponent field (same seed -> same
    generate_opponent_lineups draw for both, so the field itself isn't a
    confound), and report win_pct/duplication_rate/total projection side by
    side.

    projection_field picks what both the lineup-builders AND ownership_proxy
    treat as "points" - default proj_median was this module's own original
    synthetic test. Passing projection_field="proj_ceiling" instead builds
    both lineups from the SAME field models/optimizer.generate_lineups uses
    for real GPP mode, so this stops being a synthetic median-based stand-in
    and starts testing the actual product's own construction logic.

    Ran this at the caller's `concentration` AND separately across a sweep
    (1/3/8/15/25) during development - see the module docstring's VALIDATION
    RESULT. Headline: chalk's field overlap (avg_shared_players) is
    consistently higher than contrarian's at every concentration, but that
    only translates into a LOWER win_pct for chalk once concentration is high
    enough (~15+ in this parameterization) that overlap dominates a modest
    real projection edge - at this function's own default concentration
    (1.0), chalk's win_pct came back HIGHER, not lower. A single call at the
    default concentration is therefore not, by itself, a pass/fail
    validation - the mechanism needs a concentration sweep (or real ownership
    data to pick one concentration honestly) to interpret correctly. Do not
    report a single run's win_pct ordering as proof either way without
    checking avg_shared_players and the concentration it was run at.
    """
    players, _, _, _ = _load_player_pool(slate_id, projection_field, engine=engine)
    raw_points_by_id = {p["player_id"]: p["points"] for p in players}

    chalk = build_chalk_lineup(players)
    chalk_projection = _raw_projection(chalk, raw_points_by_id)
    contrarian, lambda_used, contrarian_projection, projection_matched = find_contrarian_matching_projection(
        players, chalk_projection, proxy_fn=proxy_fn
    )

    chalk_result = run_field_simulation(
        chalk, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )
    contrarian_result = run_field_simulation(
        contrarian, slate_id, contest_size, concentration, num_simulations, seed, engine, projection_field, proxy_fn=proxy_fn
    )

    chalk_result["total_median_projection"] = round(chalk_projection, 2)
    chalk_result["roster"] = [(slot, p["name"]) for slot, p in chalk["roster"]]
    contrarian_result["total_median_projection"] = round(contrarian_projection, 2)
    contrarian_result["roster"] = [(slot, p["name"]) for slot, p in contrarian["roster"]]
    contrarian_result["lambda_used"] = lambda_used

    return {
        "chalk": chalk_result,
        "contrarian": contrarian_result,
        "projection_matched_within_tolerance": projection_matched,
        "projection_gap": round(abs(chalk_projection - contrarian_projection), 2),
    }
