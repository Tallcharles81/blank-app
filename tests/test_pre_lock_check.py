from sqlalchemy import text

from data.pre_lock_check import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    _role_check_severity,
    hard_role_exclusions,
)
from models.projections import _real_week_sequence


def _game(season, week, snap_pct=None, target_share=None, carries=None, injury_status=None):
    return {
        "season": season,
        "week": week,
        "snap_pct": snap_pct,
        "target_share": target_share,
        "carries": carries,
        "injury_status": injury_status,
    }


# --- _role_check_severity: the advisory HIGH/MEDIUM/LOW tiers -------------


def test_severity_high_for_a_real_structural_low_volume_pattern():
    # Kenny Gainwell's real shape: moderate snap share, but genuinely low
    # carries every recent game - a real committee/complementary role, not
    # a lead back. This is the exact case the old salary-proximity check
    # missed entirely (see data/pre_lock_check.py's module docstring).
    games = [
        _game(2026, 1, snap_pct=0.46, carries=5),
        _game(2025, 18, snap_pct=0.67, carries=5),
        _game(2025, 17, snap_pct=0.68, carries=7),
        _game(2025, 16, snap_pct=0.54, carries=9),
    ]
    needs_check, severity, reason = _role_check_severity("RB", games)
    assert needs_check is True
    assert severity == SEVERITY_HIGH
    assert "carries" in reason


def test_severity_medium_for_injury_on_an_established_player():
    games = [
        _game(2026, 1, snap_pct=0.81, target_share=0.27, injury_status="Questionable"),
        _game(2025, 18, snap_pct=0.87, target_share=0.28),
        _game(2025, 17, snap_pct=0.86, target_share=0.22),
        _game(2025, 16, snap_pct=0.85, target_share=0.31),
    ]
    needs_check, severity, reason = _role_check_severity("WR", games)
    assert needs_check is True
    assert severity == SEVERITY_MEDIUM
    assert "Questionable" in reason


def test_severity_low_for_thin_rookie_data_not_treated_as_a_known_problem():
    games = [_game(2025, 18, snap_pct=0.55, target_share=0.05)]
    needs_check, severity, reason = _role_check_severity("WR", games)
    assert needs_check is True
    assert severity == SEVERITY_LOW
    assert "not a known problem" in reason


def test_severity_clear_for_a_real_stable_established_role():
    games = [
        _game(2026, 1, snap_pct=0.85, target_share=0.25),
        _game(2025, 18, snap_pct=0.88, target_share=0.22),
        _game(2025, 17, snap_pct=0.84, target_share=0.28),
        _game(2025, 16, snap_pct=0.87, target_share=0.24),
    ]
    needs_check, severity, reason = _role_check_severity("WR", games)
    assert needs_check is False
    assert severity is None
    assert reason is None


def test_severity_clear_for_a_real_stable_star_with_one_ordinary_rest_game_dip():
    # Regression test for a confirmed real false positive: CeeDee Lamb's real
    # last-4 snap_pct was [0.81, 0.45, 0.87, 0.86] - the 0.45 a real, ordinary
    # week-18 rest game once seeding was locked, not evidence of a role
    # change. Raw stdev across all 4 games (~0.20) cleared the old threshold
    # and flagged this as "possible committee split", which is wrong - one
    # low game, once, isn't a recurring pattern. See the "inconsistent recent
    # snap share" branch's comment in data/pre_lock_check.py for the fix
    # (set aside the single lowest game before checking for a recurring
    # swing) and the real pool-wide check behind it (630 real WR/RB/TE
    # players: rescues 57/104 one-dip cases like this one, still catches 47
    # real recurring-swing cases).
    games = [
        _game(2026, 1, snap_pct=0.81, target_share=0.27),
        _game(2025, 18, snap_pct=0.45, target_share=0.04),
        _game(2025, 17, snap_pct=0.87, target_share=0.28),
        _game(2025, 16, snap_pct=0.86, target_share=0.22),
    ]
    needs_check, severity, reason = _role_check_severity("WR", games)
    assert needs_check is False, f"a single ordinary rest-game dip must not read as a committee split, got: {reason}"
    assert severity is None


def test_severity_high_for_a_real_recurring_alternating_role_not_just_one_dip():
    # The fix above must not become "inconsistency can never fire" - a
    # genuine recurring alternating role (unlike Lamb's single dip, this
    # swings low/high/low/high, so setting aside just the single lowest game
    # still leaves real week-to-week volatility behind) must still be caught.
    games = [
        _game(2026, 1, snap_pct=0.49, target_share=0.15),
        _game(2025, 18, snap_pct=0.28, target_share=0.08),
        _game(2025, 17, snap_pct=0.93, target_share=0.30),
        _game(2025, 16, snap_pct=0.91, target_share=0.29),
    ]
    needs_check, severity, reason = _role_check_severity("WR", games)
    assert needs_check is True
    assert severity == SEVERITY_HIGH
    assert "inconsistent" in reason


# --- hard_role_exclusions: the pool-wide HARD gate wired into the --------
# --- optimizer - deliberately much stricter, see its module docstring ----


TEST_INTERMITTENT_QB_NAME = "Test Intermittent Qb Regression"
TEST_STABLE_QB_NAME = "Test Stable Qb Regression"


def _insert_usage_row(conn, player_id, player_name, position, season, week, snap_pct):
    conn.execute(
        text(
            """
            INSERT INTO player_weekly_stats
                (player_id, player_name, position, team, season, week, snap_pct, fantasy_points_ppr)
            VALUES (:player_id, :player_name, :position, 'ZZ', :season, :week, :snap_pct, 10.0)
            """
        ),
        {
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "season": season,
            "week": week,
            "snap_pct": snap_pct,
        },
    )


def test_hard_role_exclusions_catches_the_intermittent_backup_qb_pattern(engine):
    # Regression test for the real near-miss this session: the FIRST attempt
    # at this hard gate reused the advisory thresholds pool-wide and wrongly
    # excluded 317 of 1,066 real players including Jonathan Taylor and
    # CeeDee Lamb, while ALSO failing to catch the real motivating case
    # (Jameis Winston), because max snap share alone can't distinguish a
    # real starter from a backup who plays ~100% of snaps in the games he
    # DOES start. This exercises the fixed, position-aware design directly -
    # see data/pre_lock_check.py's HARD_EXCLUDE_MAX_SNAP_PCT/
    # MIN_SNAP_PCT_FOR_BENCHED module comments for the full story.
    week_sequence = _real_week_sequence(engine)
    weeks = week_sequence[-4:]
    assert len(weeks) == 4, "not enough real week history in this DB to run this test"

    intermittent_id, stable_id = "TEST_DK_INTERMITTENT_QB", "TEST_DK_STABLE_QB"
    try:
        with engine.begin() as conn:
            # Real Jameis Winston shape: toggles between near-0% (didn't play) and ~100% (started).
            for (season, week), snap in zip(weeks, [0.03, 1.0, 1.0, 0.77]):
                _insert_usage_row(conn, intermittent_id, TEST_INTERMITTENT_QB_NAME, "QB", season, week, snap)
            # A real starter: consistently near-full snaps every week.
            for season, week in weeks:
                _insert_usage_row(conn, stable_id, TEST_STABLE_QB_NAME, "QB", season, week, 0.99)

        dk_players = [
            {"player_id": "TEST_DK1", "name": TEST_INTERMITTENT_QB_NAME, "position": "QB", "team": "ZZ"},
            {"player_id": "TEST_DK2", "name": TEST_STABLE_QB_NAME, "position": "QB", "team": "ZZ"},
        ]
        excluded = hard_role_exclusions(dk_players, engine)

        assert "TEST_DK1" in excluded, "an intermittent near-0%/near-100% snap pattern is the real backup signature"
        assert "TEST_DK2" not in excluded, "a consistently near-full-snap QB must never be hard-excluded"
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM player_weekly_stats WHERE player_id IN (:a, :b)"),
                {"a": intermittent_id, "b": stable_id},
            )


def test_hard_role_exclusions_survives_real_stars_and_catches_real_winston(engine):
    # Same regression, against real production fixture data already loaded
    # for this session's live Thu-Mon slate - if that slate is ever removed
    # from this dev DB, this test needs updating; the synthetic test above
    # covers the same logic without that dependency.
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT player_id, name, position, team FROM slate_player_pool "
                    "WHERE slate_id = 'dk_thu_mon_2026_09_17' AND name = ANY(:names)"
                ),
                {"names": ["Jonathan Taylor", "CeeDee Lamb", "Malik Nabers", "Jameis Winston", "Trevor Lawrence"]},
            )
            .mappings()
            .fetchall()
        )
    dk_players = [dict(r) for r in rows]
    assert len(dk_players) == 5, "expected real slate fixture data to still contain these players"

    excluded = hard_role_exclusions(dk_players, engine)
    by_name = {p["name"]: p["player_id"] for p in dk_players}

    for star in ("Jonathan Taylor", "CeeDee Lamb", "Malik Nabers", "Trevor Lawrence"):
        assert by_name[star] not in excluded, f"{star} is a real, clear starter and must never be hard-excluded"
    assert by_name["Jameis Winston"] in excluded, "Winston's real intermittent-starter pattern must be caught"
