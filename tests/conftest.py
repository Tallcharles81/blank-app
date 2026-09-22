import pytest

from db.migrate import get_engine


@pytest.fixture(scope="session")
def engine():
    return get_engine()


@pytest.fixture
def stable_slate():
    """A real, currently-imported slate_id whose live gate stays buildable
    regardless of real calendar time - for tests exercising construction/
    validation/simulation logic that merely NEEDS a real, legally-buildable
    player pool, not the two real-time-dependent gates themselves.

    Why this exists: a fixed real slate_id eventually elapses (dk_thu_mon_
    2026_09_17 broke every test using it, all at once, the moment its last
    real game kicked off - confirmed by checking game_lock_status directly:
    every player came back locked=True). Swapping to another currently-
    upcoming real slate doesn't durably fix this - dk_sunday_2026_09_27 is
    real and upcoming, but its own real nflverse roster-status feed isn't
    published yet this far out (confirmed empty, still real 416/442 "not on
    roster" exclusions as of when this fixture was written) - and it too
    will eventually elapse in real time. Both real-time-dependent gates
    (models.optimizer.game_lock_status, resolve_slate_season_week) are
    bypassed here - lock-time forced to "nothing is locked", season/week
    forced to (2026, 2), a real week with real, confirmed-populated roster/
    injury data - while every other real gate (hard role exclusions,
    playing-time floor) stays genuinely real and content-based, since
    that's what these tests actually exercise.

    Tests that are ABOUT gate timing itself (test_player_availability.py's
    own tests, the calibration backtests that deliberately use elapsed real
    slates) must NOT use this fixture - they need the real, unpatched gate
    behavior to mean anything.
    """
    import models.optimizer as opt

    real_lock = opt.game_lock_status
    real_resolve = opt.resolve_slate_season_week
    opt.game_lock_status = lambda players: {p["player_id"]: False for p in players}
    opt.resolve_slate_season_week = lambda slate_id, engine: (2026, 2)
    try:
        yield "dk_sunday_2026_09_27"
    finally:
        opt.game_lock_status = real_lock
        opt.resolve_slate_season_week = real_resolve
