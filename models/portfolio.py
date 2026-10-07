"""Spread a Classic multi-lineup build across games, QBs and players.

Set by the user after 2026-10-04: 20 lineups used 3 QBs and up to 12 of 20
on the same players, so one bust (Sadiq 0.0 in 10, Raiders DST 1.0 in 11)
sank most of the file, and the #2 game (DAL@HOU) got 4 of 180 slots.

The plan: walk the slate's games best-first (models.game_environment) and
give each top game's QBs their own stacked lineup, then build each lineup
with every player already at his cap excluded, so no player or DST can
pile up. `choose` picks among a QB's candidate lineups - the first one for
the optimizer group, the best simulated one for the simulator group.
"""
from collections import Counter

from models.optimizer import build_lineups_from_pool

MAX_PLAYER_SHARE = 0.4
MAX_DST_SHARE = 0.3
MAX_LINEUPS_PER_QB = 2


def qb_plan(ranked_games, pool, num_lineups, top_games=5):
    """QB player_ids in build order: both QBs of each top game (best projected
    first), game by game, then the rest of the slate's QBs by projection,
    until `num_lineups` slots are filled (a QB may repeat up to
    MAX_LINEUPS_PER_QB times if the slate runs short)."""
    qbs_by_team = {}
    for p in pool:
        if p["position"] == "QB":
            qbs_by_team.setdefault(p["team"], []).append(p)
    best_qb = {t: max(ps, key=lambda p: p["points"]) for t, ps in qbs_by_team.items()}
    order = []
    for _, g in ranked_games.head(top_games).iterrows():
        pair = [best_qb[t] for t in (g["home"], g["away"]) if t in best_qb]
        order += [q["player_id"] for q in sorted(pair, key=lambda p: -p["points"])]
    rest = sorted((q for q in best_qb.values() if q["player_id"] not in order), key=lambda p: -p["points"])
    order += [q["player_id"] for q in rest]
    plan = []
    for rep in range(MAX_LINEUPS_PER_QB):
        for qb in order:
            if len(plan) < num_lineups:
                plan.append(qb)
    return plan


def build_spread_portfolio(pool, plan, choose=None, candidates_per_qb=1, max_player_share=MAX_PLAYER_SHARE,
                           max_dst_share=MAX_DST_SHARE, require_bring_back=False, excluded_player_ids=None,
                           max_count_by_player=None):
    """One lineup per QB in `plan`, each a QB stack built with every capped
    player excluded. Returns the lineups in plan order (a QB whose stack
    can't be built under the caps is skipped and reported in `skipped`).

    max_count_by_player: {player_id: n} tighter caps for single players, e.g.
    a Questionable player whose status can't be confirmed before the build
    (at most 2 lineups - ATL@NO 2026-10-05, where an unconfirmed
    Questionable TE turned out inactive in 6 of 20 lineups)."""
    n = len(plan)
    player_cap = max(1, int(max_player_share * n))
    dst_cap = max(1, int(max_dst_share * n))
    counts, lineups, skipped, seen = Counter(), [], [], set()
    base_excluded = set(excluded_player_ids or [])
    positions = {p["player_id"]: p["position"] for p in pool}
    for qb in plan:
        tight = max_count_by_player or {}
        capped = {pid for pid, c in counts.items()
                  if pid != qb and c >= min(tight.get(pid, 10**9), dst_cap if positions.get(pid) == "DST" else player_cap)}
        try:
            cands, _ = build_lineups_from_pool(pool, num_lineups=candidates_per_qb, locked_player_ids=[qb],
                                               excluded_player_ids=list(base_excluded | capped), min_uniques=2,
                                               require_qb_stack=True, require_bring_back=require_bring_back)
        except ValueError:
            skipped.append(qb)
            continue
        cands = [c for c in cands if frozenset(p["player_id"] for _, p in c["roster"]) not in seen]
        if not cands:
            skipped.append(qb)
            continue
        lu = choose(cands) if choose else cands[0]
        seen.add(frozenset(p["player_id"] for _, p in lu["roster"]))
        counts.update(p["player_id"] for _, p in lu["roster"])
        lineups.append(lu)
    return lineups, skipped
