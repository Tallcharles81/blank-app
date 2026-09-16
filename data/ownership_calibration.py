import csv
import json
import math
from collections import defaultdict

from sqlalchemy import text

from db.migrate import get_engine
from models.field_simulation import ownership_proxy
from models.simulation import _standard_normal_cdf

# ---------------------------------------------------------------------------
# Real, measured post-contest ownership calibration.
#
# models/field_simulation.py's ownership_proxy() has always been a labeled
# STAND-IN (projected points per $1,000 salary) because this codebase has no
# live ownership feed. The one real check ever run against it before this
# module existed used an external, unrelated 2017 DK contest CSV matched
# against external historical salary data (not our own projection engine at
# all) - real, but a stand-in validation of a stand-in metric, one step
# removed from the thing actually being shipped. Real result from that test:
# Spearman rho=0.328 (p=0.0068, n=67) against real %Drafted - positive and
# significant, but weak (~11% of rank variance explained), with two
# systematic misses: the proxy badly overrated cheap players who happened to
# have one big realized game the real field never saw coming (that test only
# had REALIZED box-score points to work with, not real pre-game
# projections), and badly underrated expensive, safe role/opportunity chalk
# the field rostered on reputation rather than that week's box score.
#
# This module is the first-party version: a real DK contest-standings export
# the user actually entered, matched against THIS codebase's own real salary
# and real pre-game proj_median for the matching slate - testing the actual
# mechanism this project ships, not a historical analogue of it. See
# import_contest_standings() and run_ownership_correlation_test().
#
# Every future contest-standings export the user provides accumulates here
# (contest_ownership, keyed by (contest_id, name), re-importing the same
# contest_id upserts rather than duplicates) so this becomes a real, growing
# multi-contest calibration dataset over time, not a one-off check.
# ---------------------------------------------------------------------------

# Below this real matched sample size, a Spearman rho is too easily swung by
# one or two outlier players to trust - same reasoning as
# MIN_RECENT_GAMES_FOR_ROLE_CONFIDENCE/MIN_PAIRS_FOR_FITTED_CORRELATION
# elsewhere in this codebase, applied to this new real-data source.
MIN_MATCHED_PLAYERS_FOR_CORRELATION = 30


def parse_contest_standings_csv(csv_path):
    """DraftKings' real contest-standings export format: one row per contest
    ENTRY (Rank/EntryId/EntryName/Points/Lineup), with a SEPARATE, unrelated
    per-player ownership side-table bolted onto the same rows via extra
    trailing columns (Player/Roster Position/%Drafted/FPTS) - populated only
    for as many leading rows as there are distinct (player, roster slot)
    combinations in the whole contest, then blank for the rest. The row
    index a player's ownership happens to appear on has nothing to do with
    that row's own entry/rank.

    A player who was eligible for more than one roster slot (a WR/RB/TE who
    some entries started in their natural slot and others started in FLEX)
    gets a SEPARATE row per slot, each with that slot's own %Drafted (real
    total ownership is the sum across all of a player's slot rows) - real,
    DST names in this side-table carry a trailing space DK's export doesn't
    put on skill-position names, stripped here during parsing rather than
    left as a silent matching failure downstream.

    Returns {player_name: {"pct_drafted": float, "fpts_contest": float |
    None}} - already deduplicated/summed across roster-slot rows.
    """
    totals = defaultdict(lambda: {"pct_drafted": 0.0, "fpts_contest": None})
    skipped = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Player") or "").strip()
            if not name:
                continue
            pct_raw = (row.get("%Drafted") or "").strip().rstrip("%")
            try:
                pct = float(pct_raw)
            except ValueError:
                skipped += 1
                continue
            fpts_raw = (row.get("FPTS") or "").strip()
            fpts = None
            if fpts_raw:
                try:
                    fpts = float(fpts_raw)
                except ValueError:
                    fpts = None
            totals[name]["pct_drafted"] += pct
            # FPTS is the player's real total, identical across every one of
            # their roster-slot rows (verified against this contest's own
            # real data - a player's FPTS never differs between their
            # natural-position row and their FLEX row), so the last row seen
            # is as good as any other, not an average or a sum.
            if fpts is not None:
                totals[name]["fpts_contest"] = fpts

    return dict(totals), skipped


def import_contest_standings(csv_path, slate_id, contest_id, engine=None):
    """Match a real, parsed contest-standings export against slate_id's real
    slate_player_pool (by name - both sides are DK's own naming, so an exact
    match after stripping whitespace is the right bar here, not the fuzzy
    nflverse-name matching data/player_crosswalk.py does for a completely
    different real problem) and this codebase's own real, stored projections
    for that slate, then upserts one row per real player into
    contest_ownership (see schema.sql) - re-running this for the same
    contest_id updates rather than duplicates.

    A player with no slate_player_pool match, or a real pool match with no
    stored projection for this slate, is still stored (unmatched_reason
    explains why) rather than silently dropped - an honest accounting of
    real coverage gaps, not just the players that happened to work.

    Returns {"total_players": int, "matched_with_projection": int,
    "matched_no_projection": int, "unmatched": int, "csv_rows_skipped": int}.
    """
    engine = engine or get_engine()
    ownership_by_name, skipped = parse_contest_standings_csv(csv_path)

    with engine.connect() as conn:
        pool_rows = conn.execute(
            text("SELECT player_id, name, position, salary FROM slate_player_pool WHERE slate_id = :slate_id"),
            {"slate_id": slate_id},
        ).mappings().fetchall()
        pool_by_name = {row["name"].strip(): dict(row) for row in pool_rows}

        proj_rows = conn.execute(
            text("SELECT player_id, proj_median FROM projections WHERE slate_id = :slate_id"),
            {"slate_id": slate_id},
        ).mappings().fetchall()
        proj_by_player_id = {row["player_id"]: float(row["proj_median"]) for row in proj_rows}

    to_upsert = []
    matched_with_projection = matched_no_projection = unmatched = 0
    for name, stats in ownership_by_name.items():
        pool_match = pool_by_name.get(name)
        if pool_match is None:
            unmatched += 1
            to_upsert.append(
                {
                    "contest_id": contest_id,
                    "slate_id": slate_id,
                    "player_id": None,
                    "name": name,
                    "position": None,
                    "salary": None,
                    "pct_drafted": stats["pct_drafted"],
                    "fpts_contest": stats["fpts_contest"],
                    "proj_median_at_import": None,
                    "ownership_proxy": None,
                    "unmatched_reason": "no slate_player_pool match on name for this slate",
                }
            )
            continue

        player_id = pool_match["player_id"]
        proj_median = proj_by_player_id.get(player_id)
        proxy_value = None
        unmatched_reason = None
        if proj_median is None:
            matched_no_projection += 1
            unmatched_reason = "matched real player, but no stored projection for this slate"
        else:
            matched_with_projection += 1
            proxy_value = ownership_proxy([{"player_id": player_id, "points": proj_median, "salary": pool_match["salary"]}])[
                player_id
            ]

        to_upsert.append(
            {
                "contest_id": contest_id,
                "slate_id": slate_id,
                "player_id": player_id,
                "name": name,
                "position": pool_match["position"],
                "salary": pool_match["salary"],
                "pct_drafted": stats["pct_drafted"],
                "fpts_contest": stats["fpts_contest"],
                "proj_median_at_import": proj_median,
                "ownership_proxy": proxy_value,
                "unmatched_reason": unmatched_reason,
            }
        )

    with engine.begin() as conn:
        for row in to_upsert:
            conn.execute(
                text(
                    """
                    INSERT INTO contest_ownership
                        (contest_id, slate_id, player_id, name, position, salary, pct_drafted,
                         fpts_contest, proj_median_at_import, ownership_proxy, unmatched_reason)
                    VALUES
                        (:contest_id, :slate_id, :player_id, :name, :position, :salary, :pct_drafted,
                         :fpts_contest, :proj_median_at_import, :ownership_proxy, :unmatched_reason)
                    ON CONFLICT (contest_id, name) DO UPDATE SET
                        slate_id = EXCLUDED.slate_id,
                        player_id = EXCLUDED.player_id,
                        position = EXCLUDED.position,
                        salary = EXCLUDED.salary,
                        pct_drafted = EXCLUDED.pct_drafted,
                        fpts_contest = EXCLUDED.fpts_contest,
                        proj_median_at_import = EXCLUDED.proj_median_at_import,
                        ownership_proxy = EXCLUDED.ownership_proxy,
                        unmatched_reason = EXCLUDED.unmatched_reason,
                        imported_at = now()
                    """
                ),
                row,
            )

    return {
        "total_players": len(to_upsert),
        "matched_with_projection": matched_with_projection,
        "matched_no_projection": matched_no_projection,
        "unmatched": unmatched,
        "csv_rows_skipped": skipped,
    }


def _rank(values):
    """1-indexed ranks, best (highest value) = rank 1, with the standard
    average-rank tie correction (two tied values both get the mean of the
    rank positions they span) - the same tie handling scipy.stats.spearmanr
    uses by default, needed here since real %Drafted and real proxy values
    both contain exact ties (multiple players at the DK salary floor with
    proxy exactly 0, multiple players at identical %Drafted).
    """
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs, ys):
    """Manual Spearman rank correlation - no scipy dependency, same
    reasoning as models/simulation.py's _standard_normal_cdf (this codebase
    deliberately hasn't added scipy as a requirement for one statistical
    test). rho is Pearson correlation of the two rank arrays (mathematically
    identical to scipy's default tie-corrected Spearman). Significance uses
    the standard large-n normal approximation z = rho * sqrt(n - 1); with
    matched samples in the hundreds (this module's real use case), that
    approximation is very close to the exact t-distribution result and
    avoids needing a t-distribution CDF implementation for one test.
    """
    n = len(xs)
    rx, ry = _rank(xs), _rank(ys)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    std_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    std_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if std_x == 0 or std_y == 0:
        return None, None
    rho = cov / (std_x * std_y)
    if n < 3 or abs(rho) >= 1.0:
        p_value = 0.0 if abs(rho) >= 1.0 else None
    else:
        z = rho * math.sqrt(n - 1)
        p_value = float(2 * (1 - _standard_normal_cdf(abs(z))))
    return rho, p_value


def _systematic_miss_check(rows, proxy_key):
    """The two real failure patterns the 2017 stand-in test found: does the
    proxy overrate cheap players the real field ignored, and underrate
    expensive players the real field rostered on role/opportunity rather
    than that check's own basis (proj_median or realized fpts)? Ranks every
    matched player on real %Drafted and on the proxy basis under test, then
    reports both the biggest individual outliers in each direction (for a
    concrete, named example the way the 2017 test found Anthony Sherman/Joe
    Mixon) and a quantitative summary: mean (proxy_rank - ownership_rank)
    for the cheapest real salary tercile vs the most expensive tercile - a
    systematic pattern shows up as a real gap between those two means, not
    just a couple of anecdotes.

    Also breaks the same mean rank_diff out BY POSITION - found on this
    contest's own real data, not assumed going in: DST was strongly
    UNDERrated by the proxy (real field ownership for a defense tracks real
    matchup quality/game-environment reputation, not a raw points-per-salary
    ratio the way skill-position value does) and QB was strongly OVERrated
    (a large real field spreads its QB ownership across several viable
    "consensus" options rather than concentrating on whichever one raw
    points-per-dollar ranks highest, the way it does for a single true
    "best value" skill-position play). This is a real, position-specific
    bias beyond the cheap-vs-expensive framing alone, worth surfacing on
    every future import rather than only when someone happens to slice the
    data this way by hand.
    """
    ownership_rank = _rank([r["pct_drafted"] for r in rows])
    proxy_rank = _rank([r[proxy_key] for r in rows])
    for r, o_rank, p_rank in zip(rows, ownership_rank, proxy_rank):
        r["ownership_rank"] = o_rank
        r["proxy_rank_for_check"] = p_rank
        r["rank_diff"] = p_rank - o_rank  # negative: proxy overrates; positive: proxy underrates

    by_salary = sorted(rows, key=lambda r: r["salary"])
    tercile_size = len(by_salary) // 3
    cheap_tercile = by_salary[:tercile_size]
    expensive_tercile = by_salary[-tercile_size:] if tercile_size else []

    def mean_rank_diff(group):
        return round(sum(r["rank_diff"] for r in group) / len(group), 2) if group else None

    most_overrated = sorted(rows, key=lambda r: r["rank_diff"])[:5]
    most_underrated = sorted(rows, key=lambda r: -r["rank_diff"])[:5]

    by_position = defaultdict(list)
    for r in rows:
        if r.get("position"):
            by_position[r["position"]].append(r)
    mean_rank_diff_by_position = {
        pos: {"n": len(group), "mean_rank_diff": mean_rank_diff(group)} for pos, group in by_position.items()
    }

    return {
        "cheap_tercile_size": len(cheap_tercile),
        "cheap_tercile_mean_rank_diff": mean_rank_diff(cheap_tercile),
        "expensive_tercile_size": len(expensive_tercile),
        "expensive_tercile_mean_rank_diff": mean_rank_diff(expensive_tercile),
        "mean_rank_diff_by_position": mean_rank_diff_by_position,
        "most_overrated_by_proxy": [
            {"name": r["name"], "salary": r["salary"], "pct_drafted": r["pct_drafted"], "rank_diff": r["rank_diff"]}
            for r in most_overrated
        ],
        "most_underrated_by_proxy": [
            {"name": r["name"], "salary": r["salary"], "pct_drafted": r["pct_drafted"], "rank_diff": r["rank_diff"]}
            for r in most_underrated
        ],
    }


def run_ownership_correlation_test(contest_id, slate_id=None, engine=None, store=True):
    """The real, first-party version of the 2017 stand-in test: real
    ownership_proxy (computed at import time from THIS codebase's own real
    salary + real pre-game proj_median) vs real %Drafted, for a contest
    already imported via import_contest_standings().

    Runs the comparison on TWO bases, since the 2017 test was structurally
    stuck using realized box-score points (it had no access to that slate's
    real PRE-GAME projections) and this module can now check whether that
    was actually costing it accuracy:
      - "proj_median": ownership_proxy computed from real pre-game
        proj_median (stored at import time) - the actual mechanism this
        product ships, and what a bettor deciding who to roster would have
        known BEFORE the game.
      - "realized_fpts": the same proxy formula, but computed from the
        contest's own real realized FPTS instead - mimicking the 2017 test's
        forced methodology.

    Both bases are computed over the SAME intersection of players (real
    salary + real proj_median + real fpts_contest all present) so the two
    numbers are a genuine head-to-head comparison, not an artifact of
    different sample sizes - an earlier version of this function ran
    realized_fpts over every matched row with salary+fpts regardless of
    whether proj_median existed (a real, larger, but DIFFERENT sample, ~494
    vs ~355 players), which would have made "realized_fpts scored higher"
    potentially just mean "easier sample," not "better basis." Caught before
    reporting a result on it. A separate "realized_fpts_full_sample" key
    reports that larger, non-intersected sample too - real and worth having,
    just clearly labeled as not an apples-to-apples comparison to
    proj_median.

    When store=True (default), records each basis's result as its own row
    in ownership_calibration_runs (proxy_basis distinguishes them) so this
    accumulates into a real multi-contest, multi-basis calibration history.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM contest_ownership WHERE contest_id = :contest_id"),
            {"contest_id": contest_id},
        ).mappings().fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError(f"No contest_ownership rows found for contest_id={contest_id} - import it first")
    if slate_id is None:
        slate_ids = {r["slate_id"] for r in rows if r["slate_id"]}
        slate_id = slate_ids.pop() if len(slate_ids) == 1 else None

    results = {}

    intersection = [
        {**r, "pct_drafted": float(r["pct_drafted"]), "salary": r["salary"]}
        for r in rows
        if r["ownership_proxy"] is not None and r["salary"] is not None and r["fpts_contest"] is not None
    ]
    results["proj_median"] = _run_one_basis(
        intersection, proxy_key="ownership_proxy", proxy_value_getter=lambda r: float(r["ownership_proxy"])
    )

    for r in intersection:
        r["realized_proxy"] = ownership_proxy(
            [{"player_id": r["name"], "points": float(r["fpts_contest"]), "salary": r["salary"]}]
        )[r["name"]]
    results["realized_fpts"] = _run_one_basis(
        intersection, proxy_key="realized_proxy", proxy_value_getter=lambda r: r["realized_proxy"]
    )

    full_realized_rows = []
    for r in rows:
        if r["salary"] is None or r["fpts_contest"] is None:
            continue
        realized_proxy = ownership_proxy(
            [{"player_id": r["name"], "points": float(r["fpts_contest"]), "salary": r["salary"]}]
        )[r["name"]]
        full_realized_rows.append(
            {**r, "pct_drafted": float(r["pct_drafted"]), "salary": r["salary"], "realized_proxy": realized_proxy}
        )
    results["realized_fpts_full_sample"] = _run_one_basis(
        full_realized_rows, proxy_key="realized_proxy", proxy_value_getter=lambda r: r["realized_proxy"]
    )

    if store:
        with engine.begin() as conn:
            for basis, result in results.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO ownership_calibration_runs
                            (contest_id, slate_id, n_matched, spearman_rho, p_value, proxy_basis, notes)
                        VALUES (:contest_id, :slate_id, :n_matched, :spearman_rho, :p_value, :proxy_basis, :notes)
                        """
                    ),
                    {
                        "contest_id": contest_id,
                        "slate_id": slate_id,
                        "n_matched": result["n"],
                        "spearman_rho": result["spearman_rho"],
                        "p_value": result["p_value"],
                        "proxy_basis": basis,
                        "notes": json.dumps(result.get("systematic_miss_check"))
                        if result.get("systematic_miss_check") is not None
                        else None,
                    },
                )

    return results


def _run_one_basis(rows, proxy_key, proxy_value_getter):
    n = len(rows)
    if n < MIN_MATCHED_PLAYERS_FOR_CORRELATION:
        return {
            "n": n,
            "spearman_rho": None,
            "p_value": None,
            "systematic_miss_check": None,
            "note": f"only {n} real matched players - below MIN_MATCHED_PLAYERS_FOR_CORRELATION "
            f"({MIN_MATCHED_PLAYERS_FOR_CORRELATION}), not enough to trust a correlation",
        }

    proxy_values = [proxy_value_getter(r) for r in rows]
    pct_values = [r["pct_drafted"] for r in rows]
    rho, p_value = _spearman(pct_values, proxy_values)

    for r, v in zip(rows, proxy_values):
        r[proxy_key] = v
    systematic_miss_check = _systematic_miss_check(rows, proxy_key)

    return {
        "n": n,
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "p_value": round(p_value, 6) if p_value is not None else None,
        "systematic_miss_check": systematic_miss_check,
    }
