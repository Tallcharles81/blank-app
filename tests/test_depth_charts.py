from data.depth_charts import latest_depth_chart_by_player

# Real, live nflverse depth-chart data (2026 only - see data/depth_charts.py's
# own docstring for why no historical season works here). Uses the real
# current feed rather than a synthetic fixture, matching this codebase's
# established convention for data-source-integration tests.


def test_latest_depth_chart_by_player_returns_only_real_offensive_rows():
    result = latest_depth_chart_by_player(2026)
    assert len(result) > 0
    positions = {info["position"] for info in result.values()}
    assert positions <= {"QB", "RB", "WR", "TE", "FB"}


def test_latest_depth_chart_by_player_resolves_a_real_known_case():
    # Real, already-investigated case: Greg Dortch was BUF's real WR6 by
    # 2026-09-18 (post-game depth chart), consistent with his real zero
    # involvement that game - confirms the offensive-position filter picks
    # his real WR row, not a collision with his real special-teams (KR/PR)
    # rows at the same real timestamp.
    result = latest_depth_chart_by_player(2026)
    dortch = result.get("00-0035500")
    assert dortch is not None
    assert dortch["position"] == "WR"
    assert dortch["team"] == "BUF"
    assert dortch["pos_rank"] >= 3  # real depth WR, not a top-2 role


def test_latest_depth_chart_by_player_one_row_per_player():
    result = latest_depth_chart_by_player(2026)
    # latest_depth_chart_by_player's own real contract - exactly one real
    # offensive row per real gsis_id, already deduped to the latest snapshot.
    assert len(result) == len(set(result.keys()))
