from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.pipeline import AnalyticsPipeline
    from src.types import AnswerGenerationOutput, SQLGenerationOutput
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc

_ZERO_LLM_STATS: dict[str, Any] = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "model": "stub",
}


class _CapturingLLM:
    """Stub LLM client that captures the context passed to generate_sql."""

    model = "stub"
    received_contexts: list[dict]

    def __init__(self) -> None:
        self.received_contexts = []

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        self.received_contexts.append(context)
        # WHY: return a harmless SELECT so pipeline execution succeeds
        return SQLGenerationOutput(
            sql="SELECT 1 AS x",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_LLM_STATS),
            error=None,
        )

    def generate_answer(
        self, question: str, sql: str | None, rows: list[dict]
    ) -> AnswerGenerationOutput:
        return AnswerGenerationOutput(
            answer="stub answer",
            timing_ms=0.0,
            llm_stats=dict(_ZERO_LLM_STATS),
            error=None,
        )

    def pop_stats(self) -> dict[str, Any]:
        return dict(_ZERO_LLM_STATS)


def test_pipeline_passes_ddl_context_to_generate_sql(analytics_db, schema_description_db):
    llm = _CapturingLLM()
    pipeline = AnalyticsPipeline(db_path=analytics_db, llm_client=llm, metadata_db_path=schema_description_db)
    pipeline.run("anything")
    assert len(llm.received_contexts) == 1
    ctx = llm.received_contexts[0]
    assert "ddl" in ctx
    assert ctx["ddl"].strip().startswith("CREATE TABLE gaming_mental_health")


def test_pipeline_context_built_once_and_reused(analytics_db, schema_description_db):
    """Same context object must be reused across runs — built once in __init__."""
    llm = _CapturingLLM()
    pipeline = AnalyticsPipeline(db_path=analytics_db, llm_client=llm, metadata_db_path=schema_description_db)
    pipeline.run("question one")
    pipeline.run("question two")
    pipeline.run("question three")
    assert llm.received_contexts[0] is llm.received_contexts[1]
    assert llm.received_contexts[1] is llm.received_contexts[2]


@pytest.mark.parametrize("env_value,expected_in_ddl", [
    ("true", "-- Age of the participant"),
    ("false", None),
])
def test_pipeline_context_respects_schema_include_description_env(
    analytics_db, schema_description_db, monkeypatch, env_value, expected_in_ddl
):
    monkeypatch.setenv("SCHEMA_INCLUDE_DESCRIPTION", env_value)
    llm = _CapturingLLM()
    pipeline = AnalyticsPipeline(db_path=analytics_db, llm_client=llm, metadata_db_path=schema_description_db)
    pipeline.run("anything")
    ddl = llm.received_contexts[0]["ddl"]
    if expected_in_ddl:
        assert expected_in_ddl in ddl
    else:
        assert "--" not in ddl
