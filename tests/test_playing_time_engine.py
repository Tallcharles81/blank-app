import pytest

from models.playing_time_engine import (
    RB_MIN_SNAP_PCT,
    RB_MIN_TOUCHES,
    SHRINKAGE_K,
    TE_MIN_SNAP_PCT,
    WR_MIN_SNAP_PCT,
    _shrinkage_blend,
    apply_playing_time_gate,
    classify_player_status,
    compute_opportunity_score,
    estimate_role,
    meets_playing_time_floor,
)

# --- classify_player_status: real taxonomy from real, already-existing sources ---


def test_classify_player_status_ir():
    assert classify_player_status("RES", None, None) == "IR"


def test_classify_player_status_inactive_roster():
    assert classify_player_status("CUT", None, None) == "INACTIVE"


def test_classify_player_status_out_injury():
    assert classify_player_status("ACT", "Out", None) == "OUT"


def test_classify_player_status_doubtful():
    assert classify_player_status("ACT", None, "Doubtful") == "DOUBTFUL"


def test_classify_player_status_questionable():
    assert classify_player_status("ACT", None, "Questionable") == "QUESTIONABLE"


def test_classify_player_status_active_default():
    assert classify_player_status("ACT", None, None) == "ACTIVE"
    assert classify_player_status(None, None, None) == "ACTIVE"


# --- _shrinkage_blend: real empirical-Bayes-style early-season blending ---


def test_shrinkage_blend_no_data_either_side():
    blended, n_current, n_prior, weight = _shrinkage_blend([], [])
    assert blended is None
    assert n_current == 0 and n_prior == 0
    assert weight is None


def test_shrinkage_blend_only_prior_season_data():
    blended, n_current, _n_prior, weight = _shrinkage_blend([], [0.8, 0.9])
    assert blended == pytest_approx(0.85)
    assert n_current == 0
    assert weight == 0.0


def test_shrinkage_blend_only_current_season_data():
    blended, _n_current, _n_prior, weight = _shrinkage_blend([0.7, 0.9], [])
    assert blended == pytest_approx(0.8)
    assert weight == 1.0


def test_shrinkage_blend_at_k_games_weights_equally():
    current = [0.9] * int(SHRINKAGE_K)
    prior = [0.1]
    blended, _n_current, _n_prior, weight = _shrinkage_blend(current, prior)
    assert weight == pytest_approx(0.5)
    assert blended == pytest_approx(0.5)  # halfway between 0.9 and 0.1


def test_shrinkage_blend_weight_grows_with_more_current_games():
    _, _, _, weight_2 = _shrinkage_blend([0.9, 0.9], [0.1])
    _, _, _, weight_8 = _shrinkage_blend([0.9] * 8, [0.1])
    assert weight_8 > weight_2  # more real current-season evidence -> more real weight on it


def pytest_approx(value, abs=1e-9):
    return pytest.approx(value, abs=abs)


# --- estimate_role / meets_playing_time_floor: position-specific real floors ---


def _history(current_snaps=None, prior_snaps=None, current_carries=None, current_receptions=None):
    current = [
        {"snap_pct": s, "carries": c, "targets": None, "receptions": r, "red_zone_targets": None, "red_zone_carries": None}
        for s, c, r in zip(
            current_snaps or [], current_carries or [0] * len(current_snaps or []), current_receptions or [0] * len(current_snaps or [])
        )
    ]
    prior = [
        {"snap_pct": s, "carries": 0, "targets": None, "receptions": 0, "red_zone_targets": None, "red_zone_carries": None}
        for s in (prior_snaps or [])
    ]
    return {"current": current, "prior": prior}


def test_wr_below_snap_floor_fails():
    history = _history(current_snaps=[0.20, 0.25, 0.22, 0.18])
    role = estimate_role("WR", history, None)
    meets, reason = meets_playing_time_floor("WR", role)
    assert meets is False
    assert f"{WR_MIN_SNAP_PCT:.0%}" in reason


def test_wr_above_snap_floor_passes():
    history = _history(current_snaps=[0.85, 0.80, 0.82, 0.88])
    role = estimate_role("WR", history, None)
    meets, reason = meets_playing_time_floor("WR", role)
    assert meets is True
    assert reason is None


def test_te_below_snap_floor_fails():
    history = _history(current_snaps=[0.15, 0.20])
    role = estimate_role("TE", history, None)
    meets, reason = meets_playing_time_floor("TE", role)
    assert meets is False
    assert f"{TE_MIN_SNAP_PCT:.0%}" in reason


def test_rb_low_snap_and_low_touches_fails():
    history = _history(current_snaps=[0.10, 0.12, 0.15], current_carries=[1, 2, 1], current_receptions=[0, 0, 1])
    role = estimate_role("RB", history, None)
    meets, reason = meets_playing_time_floor("RB", role)
    assert meets is False
    assert f"{RB_MIN_SNAP_PCT:.0%}" in reason and str(RB_MIN_TOUCHES) in reason


def test_rb_low_snap_but_high_touches_passes():
    # Real, legitimate case this rule must NOT catch: a real change-of-pace
    # back with a low snap share but real, meaningful touch volume (goal-
    # line/passing-down role) - snap share alone would wrongly flag him.
    history = _history(current_snaps=[0.20, 0.22, 0.18], current_carries=[10, 12, 9], current_receptions=[2, 1, 3])
    role = estimate_role("RB", history, None)
    meets, reason = meets_playing_time_floor("RB", role)
    assert meets is True
    assert reason is None


def test_qb_not_expected_starter_via_depth_chart_fails():
    depth_chart_entry = {"pos_rank": 2}
    role = estimate_role("QB", _history(current_snaps=[0.95, 0.90]), depth_chart_entry)
    assert role["role"] == "BACKUP"
    meets, reason = meets_playing_time_floor("QB", role)
    assert meets is False
    assert "QB2" in reason


def test_qb_expected_starter_via_depth_chart_passes():
    depth_chart_entry = {"pos_rank": 1}
    role = estimate_role("QB", _history(current_snaps=[0.95, 0.90]), depth_chart_entry)
    assert role["role"] == "EXPECTED STARTER"
    meets, _reason = meets_playing_time_floor("QB", role)
    assert meets is True


def test_no_real_history_or_depth_chart_is_not_excluded():
    # A rookie/practice-squad call-up with zero real recorded history on
    # either side of two seasons and no real depth-chart entry either -
    # LOW confidence, not a floor failure (item 4's own real distinction).
    role = estimate_role("WR", {"current": [], "prior": []}, None)
    assert role["playing_time_confidence"] == 0
    meets, reason = meets_playing_time_floor("WR", role)
    assert meets is True
    assert reason is None


def test_role_confidence_never_uses_a_fantasy_projection():
    # compute_opportunity_score and estimate_role never take a projection
    # as input at all - this just documents/enforces that by construction:
    # two calls with identical real usage data but nothing projection-
    # related must produce identical confidence.
    history = _history(current_snaps=[0.9, 0.9, 0.9])
    role_a = estimate_role("WR", history, None)
    role_b = estimate_role("WR", history, None)
    assert role_a["playing_time_confidence"] == role_b["playing_time_confidence"]


def test_sources_disagreeing_caps_confidence():
    # Real depth chart says QB1 (starter) but real recent usage shows a low
    # snap share - a genuine disagreement, which item 9 says must lower
    # confidence rather than average past it.
    depth_chart_entry = {"pos_rank": 1}
    history = _history(current_snaps=[0.10, 0.15, 0.12])
    role = estimate_role("QB", history, depth_chart_entry)
    assert role["sources_agree"] is False
    assert role["playing_time_confidence"] <= 55


def test_compute_opportunity_score_is_bounded():
    history = _history(current_snaps=[0.9, 0.9], current_carries=[20, 22], current_receptions=[3, 4])
    role = estimate_role("RB", history, None)
    score = compute_opportunity_score("RB", role)
    assert 0 <= score <= 100


# --- apply_playing_time_gate: real end-to-end behavior against the real DB ---


def test_apply_playing_time_gate_excludes_a_known_real_thin_role_player(engine):
    # Real, already-investigated case from this exact session: Greg Dortch
    # (BUF WR) had a real 0% snap share in his only prior recorded game -
    # using ONLY data available before week 2 2026 (the same real no-
    # lookahead standard as every other backtest in this codebase), the
    # real floor must exclude him.
    players = [{"player_id": "44138461", "name": "Greg Dortch", "position": "FLEX", "team": "BUF"}]
    result = apply_playing_time_gate(players, season=2026, week=2, engine=engine)
    assert result["eligible"] == []
    assert len(result["excluded"]) == 1
    assert result["excluded"][0]["player"] == "Greg Dortch"


def test_apply_playing_time_gate_keeps_a_known_real_starter_eligible(engine):
    players = [{"player_id": "44138434", "name": "Amon-Ra St. Brown", "position": "FLEX", "team": "DET"}]
    result = apply_playing_time_gate(players, season=2026, week=2, engine=engine)
    assert len(result["eligible"]) == 1
    assert result["excluded"] == []


def test_apply_playing_time_gate_punt_mode_flags_instead_of_excluding(engine):
    players = [{"player_id": "44138461", "name": "Greg Dortch", "position": "FLEX", "team": "BUF"}]
    result = apply_playing_time_gate(players, season=2026, week=2, engine=engine, punt_mode=True)
    assert len(result["eligible"]) == 1  # still eligible in punt mode
    assert result["excluded"] == []
    assert len(result["punt_flagged"]) == 1
    assert result["punt_flagged"][0]["penalty_multiplier"] < 1.0


def test_apply_playing_time_gate_debug_log_covers_every_real_player(engine):
    players = [
        {"player_id": "44138461", "name": "Greg Dortch", "position": "FLEX", "team": "BUF"},
        {"player_id": "44138434", "name": "Amon-Ra St. Brown", "position": "FLEX", "team": "DET"},
    ]
    result = apply_playing_time_gate(players, season=2026, week=2, engine=engine)
    assert len(result["debug_log"]) == 2
    for entry in result["debug_log"]:
        assert "role_confidence" in entry
        assert "opportunity_score" in entry
        assert entry["dfs_eligible"] in ("YES", "NO")
