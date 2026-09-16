import pytest

from models.optimizer import generate_simulation_selected_lineup, validate_lineup
from models.simulation import select_best_by_simulation


def test_generate_simulation_selected_lineup_ranks_real_diverse_candidates(engine):
    ranked, _exposure_report, _availability_report = generate_simulation_selected_lineup(
        "dk_thu_mon_2026_09_17",
        num_candidates=8,
        excluded_player_ids=["44137054"],
        seed=7,
        engine=engine,
    )

    assert len(ranked) == 8
    # Every candidate is still a fully legal roster - simulation ranking
    # never bypasses the hard gates or the roster-validity check.
    for candidate in ranked:
        validate_lineup(candidate["lineup"])
        assert 0.0 <= candidate["win_rate"] <= 1.0

    # Sorted best-to-worst by win rate.
    win_rates = [c["win_rate"] for c in ranked]
    assert win_rates == sorted(win_rates, reverse=True)


def test_recommended_lineup_is_not_just_the_highest_raw_mean_score(engine):
    # The real point of this feature: simulated win rate is a different
    # signal than raw mean/ceiling, not just a re-sort of the same number -
    # otherwise it wouldn't be adding anything over the existing MILP
    # objective. Not asserting it's ALWAYS different (that would be too
    # strict a claim about a specific random draw), just that the field
    # exists and the ranking is genuinely driven by win_rate, not mean_score.
    ranked, _, _ = generate_simulation_selected_lineup(
        "dk_thu_mon_2026_09_17",
        num_candidates=8,
        excluded_player_ids=["44137054"],
        seed=7,
        engine=engine,
    )
    mean_scores = [c["mean_score"] for c in ranked]
    win_rates = [c["win_rate"] for c in ranked]
    # The ranking is by win_rate, not by mean_score - if it were a pure
    # re-sort of mean_score, these two orderings would always match, which
    # isn't the real relationship being tested here.
    assert win_rates == sorted(win_rates, reverse=True)
    assert mean_scores != sorted(mean_scores, reverse=True) or len(set(mean_scores)) == 1


def test_select_best_by_simulation_requires_at_least_two_candidates(engine):
    ranked, _, _ = generate_simulation_selected_lineup(
        "dk_thu_mon_2026_09_17", num_candidates=3, engine=engine
    )
    with pytest.raises(ValueError, match="at least 2"):
        select_best_by_simulation([ranked[0]["lineup"]], "dk_thu_mon_2026_09_17", engine=engine)


def test_generate_simulation_selected_lineup_is_deterministic_with_a_seed(engine):
    r1, _, _ = generate_simulation_selected_lineup(
        "dk_thu_mon_2026_09_17", num_candidates=6, excluded_player_ids=["44137054"], seed=42, engine=engine
    )
    r2, _, _ = generate_simulation_selected_lineup(
        "dk_thu_mon_2026_09_17", num_candidates=6, excluded_player_ids=["44137054"], seed=42, engine=engine
    )
    assert [c["win_rate"] for c in r1] == [c["win_rate"] for c in r2]
