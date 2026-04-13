from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from helpers import BaseLLMStub, _ZERO_STATS, make_llm_client
    from src.config import PipelineConfig
    from src.llm_client import OpenRouterLLMClient
    from src.pipeline import AnalyticsPipeline
    from src.types import SQLAnalyticsJudgeOutput, SQLGenerationOutput
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc

_SCHEMA_CTX = {
    "ddl": "CREATE TABLE gaming_mental_health (age INTEGER, gender TEXT, playtime_hours REAL)",
    "tables": {"gaming_mental_health"},
    "columns": {"gaming_mental_health": {"age", "gender", "playtime_hours"}},
    "column_types": {"gaming_mental_health": {"age": "INTEGER", "gender": "TEXT", "playtime_hours": "REAL"}},
}

# LLM response payloads — no "stage" field (that is added by the judge method)
_CORRECT_LLM_RESPONSE = {
    "verdict": True,
    "grade": "pass",
    "issues": [],
    "reason": "Aggregation and GROUP BY correctly match the question.",
}

_INCORRECT_LLM_RESPONSE = {
    "verdict": False,
    "grade": "fail",
    "issues": ["missing GROUP BY", "no aggregation function"],
    "reason": "Query returns raw rows instead of aggregated results.",
}


# ---------------------------------------------------------------------------
# judge_sql_analytics — unit tests on the method itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chat_kwargs,sql,expected",
    [
        # correct verdict — all fields present and correct
        (
            {"return_value": json.dumps(_CORRECT_LLM_RESPONSE)},
            "SELECT gender, AVG(playtime_hours) FROM gaming_mental_health GROUP BY gender",
            {"verdict": True, "grade": "pass", "issues": []},
        ),
        # incorrect verdict — issues list identifies the analytical problems
        (
            {"return_value": json.dumps(_INCORRECT_LLM_RESPONSE)},
            "SELECT * FROM gaming_mental_health",
            {"verdict": False, "issue_contains": "missing GROUP BY"},
        ),
        # malformed JSON → error dict returned, pipeline does not crash
        ({"return_value": "not valid json at all"}, "SELECT 1", {"error": True}),
        # API exception → same error dict shape
        ({"side_effect": RuntimeError("API down")}, "SELECT 1", {"error": True}),
    ],
)
def test_judge_sql_analytics(chat_kwargs, sql, expected):
    client = make_llm_client()
    with patch.object(client, "_chat", **chat_kwargs) as mock_chat:
        verdict = client.judge_sql_analytics(
            question="What is the average playtime by gender?",
            sql=sql,
            schema_context=_SCHEMA_CTX,
        )
    assert verdict.stage == "sql_analytics_judge"
    # WHY: verify _chat was called — in production (unpatched) this increments
    # _stats["llm_calls"]. Checking call_count is more reliable than checking
    # _stats after a patched call (patching _chat bypasses the real stats logic).
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
        assert expected["issue_contains"] in verdict.issues


# ---------------------------------------------------------------------------
# pipeline wiring via AnalyticsPipeline
# ---------------------------------------------------------------------------


class _StubLLM(BaseLLMStub):
    """Stub that records when judge_sql_analytics is called."""

    def __init__(self) -> None:
        self.judge_calls: list = []

    def generate_sql(self, question, context):
        return SQLGenerationOutput(
            sql="SELECT AVG(playtime_hours) FROM gaming_mental_health",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_STATS),
        )

    def judge_sql_analytics(self, question, sql, schema_context):
        verdict = SQLAnalyticsJudgeOutput(**_CORRECT_LLM_RESPONSE)
        self.judge_calls.append(verdict)
        return verdict


def test_judge_appended_to_intermediate_outputs_when_enabled(analytics_db, schema_description_db, monkeypatch):
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_ENABLED", "true")
    llm = _StubLLM()
    cfg = PipelineConfig(openrouter_api_key="test-key", db_path=analytics_db, metadata_db_path=schema_description_db)
    pipeline = AnalyticsPipeline(config=cfg, llm_client=llm)
    output = pipeline.run("What is the average playtime by gender?")
    assert any(d.get("stage") == "sql_analytics_judge" for d in output.sql_generation.intermediate_outputs)


def test_judge_skipped_when_disabled(analytics_db, schema_description_db, monkeypatch):
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_ENABLED", "false")
    llm = _StubLLM()
    cfg = PipelineConfig(openrouter_api_key="test-key", db_path=analytics_db, metadata_db_path=schema_description_db)
    pipeline = AnalyticsPipeline(config=cfg, llm_client=llm)
    output = pipeline.run("any question")
    assert not any(d.get("stage") == "sql_analytics_judge" for d in output.sql_generation.intermediate_outputs)
    assert llm.judge_calls == []


@pytest.mark.parametrize(
    "sql_value,error",
    [
        # rule-based validation rejects → sql set to None
        ("DELETE FROM gaming_mental_health", None),
        (None, "LLM returned no SQL"),  # LLM produced no SQL
    ],
)
def test_judge_skipped_when_no_valid_sql(analytics_db, schema_description_db, monkeypatch, sql_value, error):
    monkeypatch.setenv("SQL_ANALYTICS_JUDGE_ENABLED", "true")

    class _LLM(_StubLLM):
        def generate_sql(self, question, context):
            return SQLGenerationOutput(
                sql=sql_value,
                timing_ms=0.0,
                llm_stats=dict(_ZERO_STATS),
                error=error,
            )

    llm = _LLM()
    cfg = PipelineConfig(openrouter_api_key="test-key", db_path=analytics_db, metadata_db_path=schema_description_db)
    pipeline = AnalyticsPipeline(config=cfg, llm_client=llm)
    pipeline.run("any question")
    assert llm.judge_calls == []
