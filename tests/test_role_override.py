import pytest
from sqlalchemy import text

from models.optimizer import _load_player_pool

SLATE = "dk_thu_mon_2026_10_01"


def _zay_flex_id(engine):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT player_id FROM slate_player_pool WHERE slate_id = :s AND name = 'Zay Flowers'"), {"s": SLATE}
        ).scalar()


def test_role_override_bypasses_only_the_role_checks(engine, monkeypatch):
    # Zay Flowers returned from a hamstring on a snap count (29%/33%), so the
    # snap-share floor excludes him; news says the role is intact.
    import models.optimizer as opt

    zay = _zay_flex_id(engine)
    if zay is None:
        pytest.skip("2026-10-01 slate not loaded")
    # This is about the role override, not lock timing: once the slate's
    # games kicked off, the lock gate excluded him on the override call and
    # the test failed for a reason it doesn't cover.
    monkeypatch.setattr(opt, "game_lock_status", lambda players: {p["player_id"]: False for p in players})
    _, excluded, _, _ = _load_player_pool(SLATE, "proj_ceiling", engine)
    assert "snap share" in excluded[zay]

    players, excluded, _, _ = _load_player_pool(SLATE, "proj_ceiling", engine, role_override_ids={zay})
    assert zay in {p["player_id"] for p in players}
    assert zay not in excluded


def test_role_override_does_not_bypass_the_injury_gate(engine):
    with engine.connect() as conn:
        achane = conn.execute(
            text("SELECT player_id FROM slate_player_pool WHERE slate_id = :s AND name = 'De''Von Achane'"), {"s": SLATE}
        ).scalar()
    if achane is None:
        pytest.skip("2026-10-01 slate not loaded")
    players, excluded, _, _ = _load_player_pool(SLATE, "proj_ceiling", engine, role_override_ids={achane})
    assert achane not in {p["player_id"] for p in players}
    assert achane in excluded
