"""Single-entry pick by simulated contest cash rate.

Each candidate is scored by how often it finishes inside the paid places
against a simulated field, in the same correlated worlds
(models/simulation.py). The field is built by opponents optimizing our own
median projections after multiplying every player's points by
exp(sigma * N(0,1)), a stand-in for the field disagreeing with our numbers.
Duplicates are kept, because real fields duplicate popular builds and that
matters for a single entry's finish.

Why this replaced simulator #1 (win rate among our own candidates) for
single entries: on 9 past slates (6 Showdown, 3 Classic Sundays; stored
projections, injury-report Out/Doubtful removed, 60 candidates each),
the win-rate #1 pick finished at the 70th percentile on average in Classic,
against 51% for the average candidate, and win rate was the least
predictive score there (Spearman -0.25 vs real points). The cash-rate pick
averaged top 43% across all 9 slates, tied for best and never the worst.
No score predicted much (every Spearman was within +/-0.06 on all 9 slates),
so this is the least-bad choice, not a strong edge, and it stays under
compare_top_pick_to_pool tracking like the rule before it.
"""
import numpy as np

from db.migrate import get_engine
from models.optimizer import _load_player_pool, build_lineups_from_pool, generate_lineups
from models.simulation import simulate_lineups

# 0.8 was calibrated so the noise field's projected strength matches real
# Showdown fields; the 9-slate study used it for Classic too.
FIELD_SIGMA = 0.8
DEFAULT_FIELD_SIZE = 800
DEFAULT_NUM_SIMULATIONS = 4000


def noise_field(players, n, sigma=FIELD_SIGMA, seed=None):
    """`n` opponent lineups, each the optimum of the pool's points scaled by
    independent lognormal noise. Infeasible draws are skipped."""
    rng = np.random.default_rng(seed)
    base = np.array([p["points"] for p in players], dtype=float)
    out = []
    for _ in range(n):
        noisy = base * np.exp(sigma * rng.standard_normal(len(players)))
        pool = [{**p, "points": float(v)} for p, v in zip(players, noisy)]
        try:
            out.append(build_lineups_from_pool(pool, num_lineups=1)[0][0])
        except ValueError:
            continue
    return out


def contest_rates(candidates, field, slate_id, paid_share, num_simulations=DEFAULT_NUM_SIMULATIONS, seed=None, engine=None):
    """[{cash_rate, top1_rate, mean_score}] per candidate: the share of
    simulated worlds in which it beats all but `paid_share` (and 1%) of the
    field. Candidates and field are simulated together so they share worlds."""
    if not field:
        raise ValueError("Empty opponent field - nothing to rank against")
    sims = simulate_lineups(list(candidates) + list(field), slate_id, num_simulations=num_simulations, seed=seed, engine=engine)
    mine = np.vstack([s["scores"] for s in sims[: len(candidates)]])
    opp = np.sort(np.vstack([s["scores"] for s in sims[len(candidates):]]), axis=0)
    above = np.empty_like(mine)
    for w in range(mine.shape[1]):
        above[:, w] = opp.shape[0] - np.searchsorted(opp[:, w], mine[:, w], side="right")
    share = above / opp.shape[0]
    return [
        {"cash_rate": float(c), "top1_rate": float(t), "mean_score": float(m)}
        for c, t, m in zip((share <= paid_share).mean(1), (share <= 0.01).mean(1), mine.mean(1))
    ]


def select_by_contest_cash(candidates, slate_id, paid_share, excluded_player_ids=None, field_size=DEFAULT_FIELD_SIZE,
                           sigma=FIELD_SIGMA, num_simulations=DEFAULT_NUM_SIMULATIONS, seed=None, engine=None):
    """Candidates ranked best-first by simulated cash rate (ties broken by
    top-1% rate), each annotated with the numbers the ranking used."""
    if len(candidates) < 2:
        raise ValueError(f"Need at least 2 candidate lineups to rank, got {len(candidates)}")
    engine = engine or get_engine()
    pool = _load_player_pool(slate_id, "proj_median", engine)[0]
    # Excluded by name, so a Showdown player's CPT and FLEX rows both go.
    by_id = {p["player_id"]: p["name"] for p in pool}
    for lu in candidates:
        for _, p in lu["roster"]:
            by_id.setdefault(p["player_id"], p.get("name"))
    out_names = {by_id.get(pid) for pid in (excluded_player_ids or [])} - {None}
    pool = [p for p in pool if p["name"] not in out_names]
    field = noise_field(pool, field_size, sigma, seed)
    rates = contest_rates(candidates, field, slate_id, paid_share, num_simulations=num_simulations, seed=seed, engine=engine)
    ranked = sorted(zip(candidates, rates), key=lambda cr: (cr[1]["cash_rate"], cr[1]["top1_rate"]), reverse=True)
    return [
        {"lineup": lu, "cash_rate": round(r["cash_rate"], 4), "top1_rate": round(r["top1_rate"], 4), "mean_score": round(r["mean_score"], 2)}
        for lu, r in ranked
    ]


def generate_contest_selected_lineup(slate_id, paid_share, num_candidates=60, projection_field="proj_ceiling",
                                     field_size=DEFAULT_FIELD_SIZE, sigma=FIELD_SIGMA, num_simulations=DEFAULT_NUM_SIMULATIONS,
                                     seed=None, engine=None, **generate_lineups_kwargs):
    """Same candidate pool as generate_simulation_selected_lineup (diverse,
    gated, stacked lineups from generate_lineups), ranked by contest cash
    rate. `paid_share` is the contest's paid places / entries (e.g. 0.22).
    Returns (ranked, exposure_report, availability_report); ranked[0] is the
    single-entry pick."""
    engine = engine or get_engine()
    generate_lineups_kwargs.setdefault("min_uniques", 3)
    lineups, exposure_report, availability_report = generate_lineups(
        slate_id, num_lineups=num_candidates, projection_field=projection_field, engine=engine, **generate_lineups_kwargs
    )
    ranked = select_by_contest_cash(
        lineups, slate_id, paid_share, excluded_player_ids=generate_lineups_kwargs.get("excluded_player_ids"),
        field_size=field_size, sigma=sigma, num_simulations=num_simulations, seed=seed, engine=engine,
    )
    return ranked, exposure_report, availability_report
