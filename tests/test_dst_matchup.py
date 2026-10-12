from models.dst_matchup import load_dst_history, matchup_ladder, matchup_projection


def _history():
    # Weak offenses (implied 16) give defenses more points than strong ones (implied 28).
    return [(10.0 + i % 5, 16.0) for i in range(200)] + [(2.0 + i % 5, 28.0) for i in range(200)]


def test_weak_opponent_gives_a_higher_ladder_than_a_strong_one():
    weak, strong = matchup_ladder(16.0, _history()), matchup_ladder(28.0, _history())
    assert set(weak) == {"10", "25", "50", "75", "90"}
    assert all(weak[k] > strong[k] for k in weak)


def test_window_widens_until_enough_games():
    # Nothing within 1.5 points of 22 - the window must widen rather than fail.
    ladder = matchup_ladder(22.0, _history(), min_games=150)
    assert ladder["10"] <= ladder["50"] <= ladder["90"]


def test_projection_fields_come_from_the_ladder():
    proj = matchup_projection(16.0, _history())
    assert proj["proj_median"] == proj["proj_percentiles"]["50"]
    assert proj["proj_ceiling"] == proj["proj_percentiles"]["90"]
    assert proj["proj_floor"] == proj["proj_percentiles"]["25"]


def test_history_before_a_week_excludes_that_week(engine):
    everything = load_dst_history(engine=engine)
    before = load_dst_history(before=(2025, 1), engine=engine)
    assert everything and before
    assert len(before) < len(everything)
