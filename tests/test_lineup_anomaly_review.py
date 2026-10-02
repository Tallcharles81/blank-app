import json

from models.lineup_anomaly_review import review_lineup_for_anomalies

# Real, already-projected Showdown players from dk_showdown_det_buf_2026_09_17
# (see tests/test_showdown_projections.py) - a real, legal-shaped roster, not
# a synthetic stand-in, so build_lineup_role_checklist resolves real
# positions/history the same way it would for an actual generated lineup.
CPT_ST_BROWN = {"player_id": "44138480", "name": "Amon-Ra St. Brown", "position": "CPT", "team": "DET", "salary": 15600}
FLEX_GIBBS = {"player_id": "44138432", "name": "Jahmyr Gibbs", "position": "FLEX", "team": "DET", "salary": 12000}
FLEX_ALLEN = {"player_id": "44138433", "name": "Josh Allen", "position": "FLEX", "team": "BUF", "salary": 11400}

REAL_LINEUP = {
    "total_salary": 39000,
    "roster": [
        ("CPT", CPT_ST_BROWN),
        ("FLEX", FLEX_GIBBS),
        ("FLEX", FLEX_ALLEN),
    ],
}


class _FakeTextBlock:
    type = "text"

    def __init__(self, text_value):
        self.text = text_value


class _FakeResponse:
    def __init__(self, text_value):
        self.content = [_FakeTextBlock(text_value)]


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


def test_review_lineup_for_anomalies_parses_real_structured_flags(engine):
    flags_json = json.dumps(
        {
            "flags": [
                {
                    "severity": "MEDIUM",
                    "concern": "All three of the highest-salaried players are rostered together with no cheap complementary pieces.",
                    "players_involved": ["Amon-Ra St. Brown", "Jahmyr Gibbs", "Josh Allen"],
                }
            ]
        }
    )
    client = _FakeClient(flags_json)

    result = review_lineup_for_anomalies(REAL_LINEUP, "dk_showdown_det_buf_2026_09_17", engine=engine, client=client)

    assert len(result["flags"]) == 1
    assert result["flags"][0]["severity"] == "MEDIUM"
    assert "Josh Allen" in result["flags"][0]["players_involved"]
    assert "existing_role_check_flags" in result


def test_review_lineup_for_anomalies_returns_empty_flags_when_nothing_looks_off(engine):
    client = _FakeClient('{"flags": []}')
    result = review_lineup_for_anomalies(REAL_LINEUP, "dk_showdown_det_buf_2026_09_17", engine=engine, client=client)
    assert result["flags"] == []


def test_review_lineup_for_anomalies_includes_real_salary_and_projection_context(engine):
    client = _FakeClient('{"flags": []}')
    review_lineup_for_anomalies(REAL_LINEUP, "dk_showdown_det_buf_2026_09_17", engine=engine, client=client)

    sent_message = client.messages.last_call_kwargs["messages"][0]["content"]
    assert "$39000" in sent_message
    assert "Amon-Ra St. Brown" in sent_message
    assert "Josh Allen" in sent_message
    # Real stored proj_ceiling for St. Brown's CPT row (see
    # tests/test_showdown_projections.py's 1.5x-multiplier fix) must appear -
    # confirms real projection data was actually looked up, not omitted.
    assert "proj_ceiling" in sent_message
