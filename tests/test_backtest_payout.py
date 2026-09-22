import pytest

from models.backtest import backtest_slate
from models.payout import import_contest_payout_structure

# Real, already-elapsed real slate/season/week combo used by
# test_calibration_backtests.py for the same reason: real actuals exist for
# this (season, week) and the pipeline can build a real as-of lineup for it.
SOURCE_SLATE = "dk_thu_mon_2026_09_17"
SEASON = 2026
WEEK = 2


def test_backtest_slate_without_contest_id_has_no_payout_estimate_key(engine):
    result = backtest_slate(SOURCE_SLATE, season=SEASON, week=WEEK, random_field_size=20, engine=engine)
    assert result["lineups"][0]["payout_estimate"] is None


def test_backtest_slate_with_contest_id_adds_a_real_payout_estimate(engine):
    import_contest_payout_structure(
        "test_backtest_payout_contest",
        [
            {"rank_start": 1, "rank_end": 1, "prize": 100.0},
            {"rank_start": 2, "rank_end": 10, "prize": 20.0},
        ],
        slate_id=SOURCE_SLATE,
        entry_fee=5.0,
        total_entries=100,
        places_paid=25,
        total_prizes=500.0,
        structure_complete=False,
        engine=engine,
    )
    result = backtest_slate(
        SOURCE_SLATE, season=SEASON, week=WEEK, random_field_size=20, engine=engine, contest_id="test_backtest_payout_contest"
    )
    lineup = result["lineups"][0]
    estimate = lineup["payout_estimate"]
    assert estimate is not None
    assert estimate["real_total_entries"] == 100
    assert 1 <= estimate["implied_rank"] <= 100
    # implied_rank is derived straight from field_percentile against total_entries -
    # this must be internally consistent, not two independently-drifting numbers.
    expected_rank = max(1, round((1 - lineup["field_percentile"]) * 100))
    assert estimate["implied_rank"] == expected_rank
    assert estimate["cashed"] == (estimate["implied_rank"] <= 25)


def test_backtest_slate_raises_for_an_unimported_contest_id(engine):
    with pytest.raises(ValueError, match="No contest payout structure stored"):
        backtest_slate(SOURCE_SLATE, season=SEASON, week=WEEK, random_field_size=20, engine=engine, contest_id="nope_not_real")


def test_backtest_slate_raises_when_contest_slate_id_mismatches(engine):
    import_contest_payout_structure(
        "test_backtest_payout_wrong_slate",
        [{"rank_start": 1, "rank_end": 1, "prize": 10.0}],
        slate_id="some_other_slate_entirely",
        places_paid=1,
        engine=engine,
    )
    with pytest.raises(ValueError, match="refusing to score"):
        backtest_slate(
            SOURCE_SLATE,
            season=SEASON,
            week=WEEK,
            random_field_size=20,
            engine=engine,
            contest_id="test_backtest_payout_wrong_slate",
        )
