from models.optimizer import generate_lineups, validate_lineup


def _bring_back_present(lineup):
    qb = next(p for slot, p in lineup["roster"] if slot == "QB")
    return any(p["position"] in ("WR", "TE") and p["team"] == qb["opponent"] for _, p in lineup["roster"])


def test_bring_back_is_off_by_default(engine):
    # require_bring_back defaults to False even in generate_lineups (unlike
    # require_qb_stack, which defaults True) - dedicating a third roster
    # slot to one game's correlation is a real cost, opt-in only.
    lineups, _, _ = generate_lineups(
        "dk_thu_mon_2026_09_17", num_lineups=5, projection_field="proj_ceiling", engine=engine
    )
    assert len(lineups) == 5
    # Not asserting zero bring-backs (one could happen to occur naturally) -
    # just that it isn't forced the way it is below.
    for lu in lineups:
        validate_lineup(lu)


def test_require_bring_back_produces_a_real_opponent_pass_catcher(engine):
    lineups, _, _ = generate_lineups(
        "dk_thu_mon_2026_09_17",
        num_lineups=5,
        projection_field="proj_ceiling",
        require_bring_back=True,
        engine=engine,
    )
    assert len(lineups) == 5
    for lu in lineups:
        validate_lineup(lu)  # still a fully legal roster
        assert _bring_back_present(lu), f"lineup missing a real bring-back: {lu['roster']}"


def test_require_bring_back_holds_alongside_exposure_caps_and_uniqueness(engine):
    lineups, _, _ = generate_lineups(
        "dk_thu_mon_2026_09_17",
        num_lineups=10,
        projection_field="proj_ceiling",
        require_bring_back=True,
        max_exposure=0.6,
        min_uniques=3,
        engine=engine,
    )
    assert len(lineups) == 10
    for lu in lineups:
        validate_lineup(lu)
        assert _bring_back_present(lu)
