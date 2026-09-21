from data.player_availability import get_availability_gate, resolve_slate_season_week

# These are real integration tests (real DB, and get_availability_gate/
# resolve_slate_season_week also do real nflverse network fetches) rather
# than mocked unit tests - this project has no mocking infrastructure
# anywhere yet, and the whole point of both regressions below was a real,
# live-data bug, not a logic bug a mock would have caught.


def test_resolve_slate_season_week_handles_a_late_night_kickoff(engine):
    # Regression test for the real timezone bug: a night kickoff's game_time
    # is stored UTC-aware (e.g. 8:15 PM ET -> 00:15 UTC the FOLLOWING
    # calendar day). Taking .date() directly on that UTC value looks up the
    # wrong schedule date for any game crossing the UTC midnight boundary -
    # never surfaced on a Sunday-only slate (every kickoff there was well
    # before the boundary), caught for real on this Thu-Mon slate's Thursday
    # night game.
    season_week = resolve_slate_season_week("dk_thu_mon_2026_09_17", engine)
    assert season_week == (2026, 2), (
        "a real Thursday-night (UTC-crossing) kickoff must resolve to the correct real schedule "
        "week, not None and not the wrong week from an unconverted UTC date"
    )


def test_resolve_slate_season_week_handles_dk_rams_team_code(engine):
    # Regression test for a real, silent bug: DK exports the Rams as "LAR",
    # nflverse's own schedule (and every other nflverse table) uses "LA" -
    # resolve_slate_season_week joined sample.team directly against
    # schedules_df's home_team/away_team with no normalization, so this came
    # back None for a real Rams slate (NYG@LAR Showdown, 2026-09-21) rather
    # than the correct (season, week). Caught for real building that slate's
    # actual lineups, not by a mock - see data/nflverse_fetch.py's
    # DK_TO_NFLVERSE_TEAM for the same divergence's other two real fallout
    # sites (the Vegas implied-total lookup and the Rams DST's synthetic id).
    season_week = resolve_slate_season_week("dk_showdown_nyg_lar_2026_09_21", engine)
    assert season_week is not None, (
        "DK's 'LAR' team code must resolve against nflverse's own 'LA' schedule rows, "
        "not silently come back with no match"
    )


def test_roster_absence_is_a_hard_exclude():
    # Regression test for the real gate gap: a player completely absent from
    # the current roster feed (not even an explicit status - Brandon Aiyuk,
    # real ACL/MCL/meniscus tear, hasn't played since, absent from the
    # roster feed entirely) used to pass the gate as eligible, since .get()
    # on a missing key returned None, which wasn't in
    # HARD_EXCLUDE_ROSTER_STATUSES either. Absence must be treated as MORE
    # suspicious than an explicit status, not less.
    dk_players = [{"player_id": "TEST_DK_AIYUK", "name": "Brandon Aiyuk", "position": "WR", "team": "SF"}]
    excluded, _flagged, _injury_report_available = get_availability_gate(dk_players, 2026, 2)
    assert "TEST_DK_AIYUK" in excluded
    assert "not found on any team's current roster" in excluded["TEST_DK_AIYUK"]


def test_dst_is_exempt_from_the_roster_absence_check():
    # DST rows resolve to a synthetic DST_{team} id, never a real GSIS id -
    # team defenses aren't people, so they'd never legitimately appear in a
    # per-player roster feed. The absence check is about real players only.
    dk_players = [{"player_id": "TEST_DK_DST", "name": "Seahawks", "position": "DST", "team": "SEA"}]
    excluded, _flagged, _injury_report_available = get_availability_gate(dk_players, 2026, 2)
    assert "TEST_DK_DST" not in excluded
