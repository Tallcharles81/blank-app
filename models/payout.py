from sqlalchemy import text

from db.migrate import get_engine

# ---------------------------------------------------------------------------
# Real per-contest payout curves, keyed by contest_id (schema.sql's contests/
# contest_payout_tiers) - the "expected payout" / "cash rate" counterpart to
# models/field_simulation.py's win_pct (which only answers "do I beat the
# whole simulated field," a single-winner framing that doesn't match a real
# multi-place-paid GPP).
#
# The one real, confirmed payout structure this codebase has ever been given
# (contest 195693597, pasted directly off DK's own contest page) is genuinely
# PARTIAL: 1st ($100) and 2nd ($70) are confirmed exact numbers, but the text
# for 3rd place ("EQ" - DK's own marker for a tied/equal-payout rank range)
# got cut off before its upper rank bound, and DK's contest-standings CSV
# export (see data/ownership_calibration.py) has no Prize/Winnings column at
# all to cross-check against - confirmed by inspecting every real contest-
# standings export already downloaded this session (data/ownership_
# calibration.py's own header docstring). There is no reliable way to
# recover ranks 3-250's real per-tier amounts from what was actually given,
# so this module is built to keep that gap explicit end-to-end (a tier with
# rank_end=None, an `unknown`/`known` split on every computed statistic)
# rather than silently interpolating or assuming an equal split down to
# places_paid - CLAUDE.md's "if unsure about an exact file format or API
# response shape, check it first rather than guessing" applies exactly as
# much to a real payout table as to a file format.
# ---------------------------------------------------------------------------


def _validate_tiers(tiers):
    if not tiers:
        raise ValueError("tiers must be a non-empty list")

    ordered = sorted(tiers, key=lambda t: t["rank_start"])
    open_ended_seen = False
    prev_end = 0
    for tier in ordered:
        rank_start = tier["rank_start"]
        rank_end = tier.get("rank_end")
        prize = tier["prize"]

        if open_ended_seen:
            raise ValueError(
                "an open-ended tier (rank_end=None) must be the last tier by rank_start - "
                "its own real upper bound is unconfirmed, so nothing can be known to come after it"
            )
        if rank_start < 1:
            raise ValueError(f"rank_start must be >= 1, got {rank_start}")
        if rank_start <= prev_end:
            raise ValueError(f"tier ranks overlap: rank_start={rank_start} falls inside the previous tier (ends at {prev_end})")
        if prize < 0:
            raise ValueError(f"prize must be >= 0, got {prize}")

        if rank_end is None:
            open_ended_seen = True
        else:
            if rank_end < rank_start:
                raise ValueError(f"rank_end ({rank_end}) must be >= rank_start ({rank_start})")
            prev_end = rank_end

    return ordered


def import_contest_payout_structure(
    contest_id,
    tiers,
    slate_id=None,
    entry_fee=None,
    total_entries=None,
    places_paid=None,
    total_prizes=None,
    structure_complete=False,
    engine=None,
):
    """Store a real contest's payout curve, keyed by contest_id - re-running
    for the same contest_id replaces its tiers (not merges/appends), so a
    later, more complete real read of the same contest's payout page cleanly
    supersedes an earlier partial one rather than accumulating stale rows.

    `tiers`: list of {"rank_start": int, "rank_end": int | None, "prize":
    float}, non-overlapping, sorted internally by rank_start. At most one
    tier may have rank_end=None (an unconfirmed upper bound), and if present
    it must be the last (highest rank_start) tier given - see
    _validate_tiers.

    `structure_complete` defaults False: only set True when every rank from 1
    through places_paid is actually covered by a closed (non-None rank_end)
    tier - true for a real full contest-standings-with-prizes export, false
    for a partial page-scrape like contest 195693597's. This is trusted as
    given (not re-derived from the tiers themselves) since the caller is the
    one who knows whether what they read off DK's page was actually the
    whole table or got cut off.
    """
    engine = engine or get_engine()
    ordered = _validate_tiers(tiers)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO contests
                    (contest_id, slate_id, entry_fee, total_entries, places_paid, total_prizes, structure_complete)
                VALUES
                    (:contest_id, :slate_id, :entry_fee, :total_entries, :places_paid, :total_prizes, :structure_complete)
                ON CONFLICT (contest_id) DO UPDATE SET
                    slate_id = EXCLUDED.slate_id,
                    entry_fee = EXCLUDED.entry_fee,
                    total_entries = EXCLUDED.total_entries,
                    places_paid = EXCLUDED.places_paid,
                    total_prizes = EXCLUDED.total_prizes,
                    structure_complete = EXCLUDED.structure_complete,
                    imported_at = now()
                """
            ),
            {
                "contest_id": contest_id,
                "slate_id": slate_id,
                "entry_fee": entry_fee,
                "total_entries": total_entries,
                "places_paid": places_paid,
                "total_prizes": total_prizes,
                "structure_complete": structure_complete,
            },
        )
        conn.execute(text("DELETE FROM contest_payout_tiers WHERE contest_id = :contest_id"), {"contest_id": contest_id})
        for tier in ordered:
            conn.execute(
                text(
                    """
                    INSERT INTO contest_payout_tiers (contest_id, rank_start, rank_end, prize)
                    VALUES (:contest_id, :rank_start, :rank_end, :prize)
                    """
                ),
                {
                    "contest_id": contest_id,
                    "rank_start": tier["rank_start"],
                    "rank_end": tier.get("rank_end"),
                    "prize": tier["prize"],
                },
            )

    known_ranks = sum((t["rank_end"] - t["rank_start"] + 1) for t in ordered if t.get("rank_end") is not None)
    return {
        "contest_id": contest_id,
        "n_tiers": len(ordered),
        "known_ranks_covered": known_ranks,
        "places_paid": places_paid,
        "coverage_of_places_paid": round(known_ranks / places_paid, 4) if places_paid else None,
    }


def load_contest_payout(contest_id, engine=None):
    """Real, stored payout curve for contest_id - raises ValueError if this
    contest was never imported (never silently returns an empty/zeroed
    structure that would look like "no prizes" rather than "no data")."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        contest_row = conn.execute(
            text("SELECT * FROM contests WHERE contest_id = :contest_id"), {"contest_id": contest_id}
        ).mappings().fetchone()
        if contest_row is None:
            raise ValueError(f"No contest payout structure stored for contest_id={contest_id} - import it first")

        tier_rows = conn.execute(
            text(
                "SELECT rank_start, rank_end, prize FROM contest_payout_tiers "
                "WHERE contest_id = :contest_id ORDER BY rank_start"
            ),
            {"contest_id": contest_id},
        ).mappings().fetchall()

    return {
        "contest_id": contest_id,
        "slate_id": contest_row["slate_id"],
        "entry_fee": float(contest_row["entry_fee"]) if contest_row["entry_fee"] is not None else None,
        "total_entries": contest_row["total_entries"],
        "places_paid": contest_row["places_paid"],
        "total_prizes": float(contest_row["total_prizes"]) if contest_row["total_prizes"] is not None else None,
        "structure_complete": contest_row["structure_complete"],
        "tiers": [
            {
                "rank_start": t["rank_start"],
                "rank_end": t["rank_end"],
                "prize": float(t["prize"]),
            }
            for t in tier_rows
        ],
    }


def prize_for_rank(rank, tiers, places_paid=None):
    """(prize, known) for one finish `rank` against a payout curve's `tiers`.

    known=True means the returned prize is a real, confirmed number (either
    a closed tier the rank falls in, or a confident $0 because `places_paid`
    confirms this rank isn't paid at all). known=False (prize=None) means
    this rank genuinely can't be priced from what's on record - either an
    open-ended (rank_end=None) tier whose value at this rank isn't
    confirmed, or a paid-but-untiered gap in the curve - and callers must
    not treat that None as $0.
    """
    for tier in tiers:
        if tier["rank_end"] is not None and tier["rank_start"] <= rank <= tier["rank_end"]:
            return tier["prize"], True

    # A confident $0 (rank confirmed outside the paid field) takes priority
    # over an open-ended tier match below - an open-ended tier's real upper
    # bound is unconfirmed, but places_paid, when given, is always a real,
    # confirmed number regardless of how much of the dollar curve is known.
    if places_paid is not None and rank > places_paid:
        return 0.0, True

    for tier in tiers:
        if tier["rank_end"] is None and rank >= tier["rank_start"]:
            return None, False

    return None, False


def estimate_payout_from_ranks(ranks, tiers, places_paid=None, entry_fee=None):
    """Real payout statistics over a collection of simulated/implied finish
    ranks (one per simulated world, or one per backtested week) against a
    stored payout curve.

    cash_pct: fraction of ranks <= places_paid - well-defined and exact
    whenever places_paid is known, independent of how much of the curve's
    dollar amounts are actually confirmed (a rank being "paid" and "how
    much it pays" are separate facts, and DK's own places_paid number is
    always confirmed even when the fine-grained curve isn't).

    mean_known_payout: mean prize across only the ranks that resolved to a
    KNOWN dollar figure (prize_for_rank's known=True) - a real conditional
    expectation, not the full unconditional EV, and explicitly NOT scaled
    back up to "cover" the unknown ranks (that would be a guess). pct_
    unknown reports how much of the distribution that conditioning drops,
    so a caller can see whether mean_known_payout is resting on most of the
    real distribution or a small, uninformative slice of it.
    """
    if not ranks:
        raise ValueError("ranks must be non-empty")

    priced = [prize_for_rank(r, tiers, places_paid=places_paid) for r in ranks]
    known_prizes = [p for p, known in priced if known]
    n = len(ranks)

    cash_pct = round(sum(1 for r in ranks if r <= places_paid) / n, 4) if places_paid is not None else None
    mean_known_payout = round(sum(known_prizes) / len(known_prizes), 4) if known_prizes else None
    pct_unknown = round(1 - len(known_prizes) / n, 4)

    roi_known = None
    if mean_known_payout is not None and entry_fee is not None:
        roi_known = round(mean_known_payout - entry_fee, 4)

    return {
        "n_worlds": n,
        "cash_pct": cash_pct,
        "mean_known_payout": mean_known_payout,
        "pct_unknown_payout": pct_unknown,
        "roi_known": roi_known,
    }
