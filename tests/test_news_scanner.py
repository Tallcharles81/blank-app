from sqlalchemy import text

from data.news_scanner import _extract_json_array, scan_game_news, scan_slate_news

# Real players/pool from dk_thu_mon_2026_09_17 (already used elsewhere in this
# suite - see tests/test_ownership_calibration.py), reused here rather than
# synthetic names so the (name, position, team) shape matches a real
# build_pre_lock_checklist row exactly.
GIBBS = {"player_id": "44137072", "name": "Jahmyr Gibbs", "position": "RB", "team": "DET", "salary": 8500, "gate_status": "CLEAR"}
ROBINSON = {"player_id": "44137074", "name": "Bijan Robinson", "position": "RB", "team": "ATL", "salary": 8200, "gate_status": "CLEAR"}


class _FakeTextBlock:
    type = "text"

    def __init__(self, text_value):
        self.text = text_value


class _FakeResponse:
    def __init__(self, text_value):
        self.content = [_FakeTextBlock(text_value)]


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_extract_json_array_handles_plain_array():
    assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_array_handles_prose_wrapper_and_code_fence():
    wrapped = 'Here is what I found:\n```json\n[{"a": 1}]\n```\nDone.'
    assert _extract_json_array(wrapped) == [{"a": 1}]


def test_extract_json_array_handles_empty_array():
    assert _extract_json_array("[]") == []


def test_scan_game_news_matches_real_finding_to_player_id():
    response_json = (
        '[{"player_name": "Jahmyr Gibbs", "source": "injury_news", '
        '"finding": "Ruled out per team site.", "contradicts_gate": true}]'
    )
    client = _FakeClient([response_json])
    findings, unmatched = scan_game_news(
        client, "claude-opus-5", "DET", "ATL", [GIBBS, ROBINSON], "2026-09-20T17:00:00Z"
    )
    assert unmatched == []
    assert len(findings) == 1
    assert findings[0]["player_id"] == GIBBS["player_id"]
    assert findings[0]["source"] == "injury_news"
    assert findings[0]["contradicts_gate"] is True
    assert findings[0]["gate_status_at_check"] == "CLEAR"


def test_scan_game_news_reports_unmatched_names_separately():
    response_json = (
        '[{"player_name": "Someone Not On This Roster", "source": "weather", '
        '"finding": "Heavy rain expected.", "contradicts_gate": false}]'
    )
    client = _FakeClient([response_json])
    findings, unmatched = scan_game_news(
        client, "claude-opus-5", "DET", "ATL", [GIBBS], "2026-09-20T17:00:00Z"
    )
    assert findings == []
    assert len(unmatched) == 1
    assert unmatched[0]["player_name"] == "Someone Not On This Roster"


def test_scan_game_news_skips_malformed_entries():
    response_json = '[{"player_name": "Jahmyr Gibbs", "source": "not_a_real_category", "finding": "x"}]'
    client = _FakeClient([response_json])
    findings, unmatched = scan_game_news(
        client, "claude-opus-5", "DET", "ATL", [GIBBS], "2026-09-20T17:00:00Z"
    )
    assert findings == []
    assert unmatched == []


class _ContentAwareFakeMessages:
    """Returns the real Gibbs finding only for the one real call whose game
    roster actually includes Jahmyr Gibbs, and raises for every other real
    call - deterministic regardless of which order build_pre_lock_checklist
    happens to iterate its real games in (dk_thu_mon_2026_09_17 spans many
    real games, not just DET's)."""

    def create(self, **kwargs):
        user_content = kwargs["messages"][0]["content"]
        if "Jahmyr Gibbs" in user_content:
            return _FakeResponse(
                '[{"player_name": "Jahmyr Gibbs", "source": "injury_news", '
                '"finding": "Real finding.", "contradicts_gate": false}]'
            )
        raise RuntimeError("simulated web_search failure")


class _ContentAwareFakeClient:
    messages = _ContentAwareFakeMessages()


def test_scan_slate_news_records_real_findings_and_isolates_one_game_failure(engine):
    # Confirms one game's failure doesn't abort the whole slate's scan (same
    # convention as every other external-data fetch in this codebase) - every
    # real game on dk_thu_mon_2026_09_17 except DET's own raises here.
    client = _ContentAwareFakeClient()

    try:
        result = scan_slate_news("dk_thu_mon_2026_09_17", engine=engine, client=client)

        assert result["games_scanned"] == 1
        assert len(result["games_failed"]) >= 1
        assert result["findings_recorded"] == 1

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM pre_lock_checks WHERE finding = 'Real finding.'")
            ).mappings().fetchall()
        assert len(rows) == 1
        assert rows[0]["player_id"] == GIBBS["player_id"]
        assert rows[0]["contradicts_gate"] is False
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM pre_lock_checks WHERE finding = 'Real finding.'"))
