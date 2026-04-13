from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from helpers import BaseLLMStub, _ZERO_STATS
    from src.llm_client import OpenRouterLLMClient
    from src.pipeline import AnalyticsPipeline
    from src.types import AnswerGroundingJudgeOutput, SQLGenerationOutput
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc

# LLM response payloads — no "stage" field (that is added by the judge method)
_GROUNDED_LLM_RESPONSE = {
    "verdict": True,
    "grade": "pass",
    "issues": [],
    "reason": "Answer correctly reflects the query results.",
}

_UNGROUNDED_LLM_RESPONSE = {
    "verdict": False,
    "grade": "fail",
    "issues": ["states value 99 not present in rows"],
    "reason": "Answer overstates the data.",
}

_SAMPLE_ROWS = [{"gender": "Male", "avg_playtime": 12.5}, {"gender": "Female", "avg_playtime": 10.2}]
_SAMPLE_SQL = "SELECT gender, AVG(playtime_hours) AS avg_playtime FROM gaming_mental_health GROUP BY gender"
_SAMPLE_ANSWER = "Males average 12.5 hours of playtime; Females average 10.2 hours."


def _make_client() -> OpenRouterLLMClient:
    client = OpenRouterLLMClient.__new__(OpenRouterLLMClient)
    client.model = "stub"
    client._stats = dict(_ZERO_STATS)
    client._client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# judge_answer_grounding — unit tests on the method itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chat_kwargs,answer,expected",
    [
        # grounded verdict — all fields present and correct
        ({"return_value": json.dumps(_GROUNDED_LLM_RESPONSE)}, _SAMPLE_ANSWER, {"verdict": True, "grade": "pass", "issues": []}),
        # ungrounded verdict — issues reference the hallucinated value "99"
        (
            {"return_value": json.dumps(_UNGROUNDED_LLM_RESPONSE)},
            "Males average 99 hours which is very high.",
            {"verdict": False, "issue_contains": "99"},
        ),
        # malformed JSON → error dict returned, pipeline does not crash
        ({"return_value": "not valid json"}, "any answer", {"error": True}),
        # API exception → same error dict shape
        ({"side_effect": RuntimeError("timeout")}, "any answer", {"error": True}),
    ],
)
def test_judge_answer_grounding(chat_kwargs, answer, expected):
    client = _make_client()
    with patch.object(client, "_chat", **chat_kwargs) as mock_chat:
        verdict = client.judge_answer_grounding(
            question="What is the average playtime by gender?",
            sql=_SAMPLE_SQL,
            rows=_SAMPLE_ROWS,
            answer=answer,
        )
    assert verdict.stage == "answer_grounding_judge"
    assert mock_chat.call_count == 1
    if expected.get("error"):
        assert verdict.error is not None
        return
    assert verdict.verdict == expected["verdict"]
    if "grade" in expected:
        assert verdict.grade == expected["grade"]
    if "issues" in expected:
        assert verdict.issues == expected["issues"]
    if "issue_contains" in expected:
        assert any(expected["issue_contains"] in issue for issue in verdict.issues)


# ---------------------------------------------------------------------------
# pipeline wiring — judge appended to AnswerGenerationOutput.intermediate_outputs
# ---------------------------------------------------------------------------


class _StubLLM(BaseLLMStub):
    """Stub LLM that records judge_answer_grounding calls."""

    def __init__(self) -> None:
        self.grounding_calls: list = []

    def generate_sql(self, question, context):
        return SQLGenerationOutput(
            sql="SELECT gender, AVG(playtime_hours) AS avg FROM gaming_mental_health GROUP BY gender",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_STATS),
        )

    def judge_answer_grounding(self, question, sql, rows, answer):
        verdict = AnswerGroundingJudgeOutput(**_GROUNDED_LLM_RESPONSE)
        self.grounding_calls.append(verdict)
        return verdict


def test_judge_appended_to_intermediate_outputs_when_enabled(analytics_db_with_data, schema_description_db, monkeypatch):
    # WHY: use a DB with rows — judge is skipped when rows == [] (tested separately)
    monkeypatch.setenv("ANSWER_GROUNDING_JUDGE_ENABLED", "true")
    llm = _StubLLM()
    pipeline = AnalyticsPipeline(
        db_path=analytics_db_with_data,
        llm_client=llm,
        metadata_db_path=schema_description_db,
    )
    output = pipeline.run("What is the average playtime by gender?")
    assert any(d.get("stage") == "answer_grounding_judge" for d in output.answer_generation.intermediate_outputs)


def test_judge_skipped_when_disabled(analytics_db, schema_description_db, monkeypatch):
    monkeypatch.setenv("ANSWER_GROUNDING_JUDGE_ENABLED", "false")
    llm = _StubLLM()
    pipeline = AnalyticsPipeline(db_path=analytics_db, llm_client=llm, metadata_db_path=schema_description_db)
    output = pipeline.run("any question")
    assert not any(d.get("stage") == "answer_grounding_judge" for d in output.answer_generation.intermediate_outputs)
    assert llm.grounding_calls == []


@pytest.mark.parametrize(
    "sql_value,error,use_empty_filter",
    [
        (None, "no sql", False),  # LLM produced no SQL → fallback path, skip judge
        (None, None, True),  # valid SQL but returns no rows → skip judge
    ],
)
def test_judge_skipped_when_no_data(analytics_db, schema_description_db, monkeypatch, sql_value, error, use_empty_filter):
    monkeypatch.setenv("ANSWER_GROUNDING_JUDGE_ENABLED", "true")

    class _LLM(_StubLLM):
        def generate_sql(self, question, context):
            if use_empty_filter:
                # WHY: filter guaranteed to return no rows on the empty test DB
                return SQLGenerationOutput(
                    sql="SELECT age FROM gaming_mental_health WHERE age < 0",
                    timing_ms=0.0,
                    llm_stats=dict(_ZERO_STATS),
                )
            return SQLGenerationOutput(
                sql=sql_value,
                timing_ms=0.0,
                llm_stats=dict(_ZERO_STATS),
                error=error,
            )

    llm = _LLM()
    pipeline = AnalyticsPipeline(db_path=analytics_db, llm_client=llm, metadata_db_path=schema_description_db)
    pipeline.run("any question")
    assert llm.grounding_calls == []
