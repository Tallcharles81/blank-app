import pytest
from sqlalchemy import text

from models.optimizer import _load_player_pool

SLATE = "dk_thu_mon_2026_10_01"


def _zay_flex_id(engine):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT player_id FROM slate_player_pool WHERE slate_id = :s AND name = 'Zay Flowers'"), {"s": SLATE}
        ).scalar()


def test_role_override_bypasses_only_the_role_checks(engine):
    # Zay Flowers returned from a hamstring on a snap count (29%/33%), so the
    # snap-share floor excludes him; news says the role is intact.
    zay = _zay_flex_id(engine)
    if zay is None:
        pytest.skip("2026-10-01 slate not loaded")
    _, excluded, _, _ = _load_player_pool(SLATE, "proj_ceiling", engine)
    if excluded.get(zay) == "game already started":
        pytest.skip("slate already locked")
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
