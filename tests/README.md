# Regression tests

Real integration tests against this project's real local dev Postgres
(`db/migrate.py::get_engine()`), not mocked unit tests - there's no mocking
infrastructure in this codebase, and every bug these cover was a real,
live-data bug, not a pure logic bug a mock would have caught.

Run with:

```
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

Requires:
- A running local Postgres with this project's schema applied (`db/migrate.py`).
- Real history in `player_weekly_stats` (the staleness and severity tests
  need enough real weeks to run against - they read the real week sequence
  rather than assuming fixed season/week numbers, so they stay correct as
  real data accumulates).
- Network access to nflverse's real data releases, for
  `test_player_availability.py` specifically (`get_availability_gate`/
  `resolve_slate_season_week` do real fetches - see `data/nflverse_fetch.py`).
- `test_hard_role_exclusions_survives_real_stars_and_catches_real_winston`
  depends on this session's real `dk_thu_mon_2026_09_17` fixture slate still
  being loaded; the synthetic test alongside it covers the same logic
  without that dependency, so this one can be deleted if that slate is
  ever cleared.

Any row a test inserts (always a `TEST_`-prefixed `player_id`) is deleted
in a `finally` block, so a failed assertion still leaves the DB clean -
verified by hand after every test run while writing these.

What's covered, and which real bug each one guards against:
- `test_projections_staleness.py` - `MAX_STALENESS_WEEKS`: a player whose
  most recent game is stale beyond the threshold must be dropped, not
  silently projected from old data (the real Christian McCaffrey/Brandon
  Aiyuk case).
- `test_player_availability.py` - the real timezone bug (a night kickoff's
  UTC game_time resolving to the wrong schedule date) and the real
  roster-absence gate gap (Brandon Aiyuk passing the gate because absence
  from the feed wasn't itself a hard exclude).
- `test_pre_lock_check.py` - the HIGH/MEDIUM/LOW severity tiers, and
  `hard_role_exclusions`'s two-signal design (RB/WR/TE max-snap-share vs.
  QB intermittent-starter pattern) - including a direct regression test for
  the near-miss where the first version of that hard gate would have
  wrongly excluded real stars like Jonathan Taylor and CeeDee Lamb.
