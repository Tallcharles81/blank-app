import pytest

from models.payout import (
    estimate_payout_from_ranks,
    import_contest_payout_structure,
    load_contest_payout,
    prize_for_rank,
)

# Real, confirmed tiers pasted directly off DK's own contest page for
# contest 195693597 during tonight's session - 1st/2nd exact, 3rd's real "EQ"
# marker present but its own upper rank bound got cut off in what was
# pasted, so it's stored open-ended (rank_end=None) rather than guessed.
REAL_PARTIAL_TIERS = [
    {"rank_start": 1, "rank_end": 1, "prize": 100.0},
    {"rank_start": 2, "rank_end": 2, "prize": 70.0},
    {"rank_start": 3, "rank_end": None, "prize": 50.0},
]


def test_prize_for_rank_returns_known_prize_for_a_closed_tier():
    prize, known = prize_for_rank(1, REAL_PARTIAL_TIERS, places_paid=250)
    assert known
    assert prize == 100.0

    prize, known = prize_for_rank(2, REAL_PARTIAL_TIERS, places_paid=250)
    assert known
    assert prize == 70.0


def test_prize_for_rank_open_ended_tier_is_unknown_not_guessed():
    # Rank 3 is real and confirmed PAID (the "EQ $50" marker), but this
    # module must not assume $50 covers every rank from 3 through 250 - only
    # the ranks explicitly closed off by a real rank_end are "known".
    prize, known = prize_for_rank(3, REAL_PARTIAL_TIERS, places_paid=250)
    assert not known
    assert prize is None

    prize, known = prize_for_rank(200, REAL_PARTIAL_TIERS, places_paid=250)
    assert not known
    assert prize is None


def test_prize_for_rank_beyond_places_paid_is_a_confident_zero():
    prize, known = prize_for_rank(251, REAL_PARTIAL_TIERS, places_paid=250)
    assert known
    assert prize == 0.0


def test_prize_for_rank_with_no_places_paid_and_no_matching_tier_is_unknown():
    prize, known = prize_for_rank(999, REAL_PARTIAL_TIERS, places_paid=None)
    assert not known
    assert prize is None


def test_validate_tiers_rejects_overlapping_ranks():
    with pytest.raises(ValueError, match="overlap"):
        import_contest_payout_structure(
            "fake_overlap",
            [
                {"rank_start": 1, "rank_end": 5, "prize": 10.0},
                {"rank_start": 3, "rank_end": 8, "prize": 5.0},
            ],
        )


def test_validate_tiers_rejects_open_ended_tier_not_last():
    with pytest.raises(ValueError, match="last tier"):
        import_contest_payout_structure(
            "fake_bad_order",
            [
                {"rank_start": 1, "rank_end": None, "prize": 10.0},
                {"rank_start": 50, "rank_end": 100, "prize": 5.0},
            ],
        )


def test_import_and_load_contest_payout_roundtrips(engine):
    result = import_contest_payout_structure(
        "test_contest_roundtrip",
        REAL_PARTIAL_TIERS,
        entry_fee=0.0,
        total_entries=6246,
        places_paid=250,
        total_prizes=2000.0,
        structure_complete=False,
        engine=engine,
    )
    assert result["n_tiers"] == 3
    assert result["known_ranks_covered"] == 2  # only ranks 1 and 2 are closed tiers
    assert result["coverage_of_places_paid"] == pytest.approx(2 / 250)

    loaded = load_contest_payout("test_contest_roundtrip", engine=engine)
    assert loaded["places_paid"] == 250
    assert loaded["total_prizes"] == 2000.0
    assert loaded["structure_complete"] is False
    assert len(loaded["tiers"]) == 3
    assert loaded["tiers"][0] == {"rank_start": 1, "rank_end": 1, "prize": 100.0}


def test_import_replaces_rather_than_merges_tiers(engine):
    import_contest_payout_structure(
        "test_contest_replace",
        [{"rank_start": 1, "rank_end": 1, "prize": 999.0}],
        engine=engine,
    )
    import_contest_payout_structure(
        "test_contest_replace",
        REAL_PARTIAL_TIERS,
        engine=engine,
    )
    loaded = load_contest_payout("test_contest_replace", engine=engine)
    assert len(loaded["tiers"]) == 3
    assert loaded["tiers"][0]["prize"] == 100.0  # old $999 tier is gone, not merged in


def test_load_contest_payout_raises_for_unimported_contest(engine):
    with pytest.raises(ValueError, match="No contest payout structure stored"):
        load_contest_payout("definitely_not_a_real_contest_id", engine=engine)


def test_estimate_payout_from_ranks_cash_pct_is_exact_even_with_partial_curve():
    # cash_pct only needs places_paid, which IS fully confirmed here - must
    # be exact regardless of how much of the dollar curve is unknown.
    ranks = [1, 2, 3, 100, 251, 300]
    stats = estimate_payout_from_ranks(ranks, REAL_PARTIAL_TIERS, places_paid=250)
    assert stats["cash_pct"] == pytest.approx(4 / 6, abs=1e-4)  # ranks 1,2,3,100 <= 250


def test_estimate_payout_from_ranks_reports_unknown_share_honestly():
    # Every one of these worlds lands in the real but unconfirmed "3rd EQ"
    # gap - mean_known_payout must come from confirmed-zero worlds only
    # (there are none here), not from silently treating unknown as $0 or
    # averaging in a guessed $50.
    ranks = [10, 20, 30]
    stats = estimate_payout_from_ranks(ranks, REAL_PARTIAL_TIERS, places_paid=250)
    assert stats["pct_unknown_payout"] == 1.0
    assert stats["mean_known_payout"] is None
    assert stats["cash_pct"] == 1.0


def test_estimate_payout_from_ranks_mixes_known_and_unknown_correctly():
    ranks = [1, 1, 300, 300]  # two wins at $100, two confirmed misses at $0
    stats = estimate_payout_from_ranks(ranks, REAL_PARTIAL_TIERS, places_paid=250, entry_fee=0.0)
    assert stats["pct_unknown_payout"] == 0.0
    assert stats["mean_known_payout"] == pytest.approx((100.0 + 100.0 + 0.0 + 0.0) / 4)
    assert stats["roi_known"] == pytest.approx(stats["mean_known_payout"])
