from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pydantic import ValidationError
    from src.config import PipelineConfig
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults():
    # WHY: test field-level defaults via model_fields rather than instantiating —
    # instantiation picks up OS env vars (loaded from .env by src/__init__.py),
    # which would mask the code defaults we actually want to verify here.
    f = PipelineConfig.model_fields
    # LLM / API
    assert f["openrouter_model"].default == "openai/gpt-5-nano"
    assert f["llm_timeout_seconds"].default == 60
    assert f["llm_max_retries"].default == 3
    # LLM generation params
    assert f["sql_max_tokens"].default == 10000
    assert f["answer_max_tokens"].default == 220
    assert f["judge_max_tokens"].default == 20000
    assert f["sql_temperature"].default == 0.0
    assert f["answer_temperature"].default == 0.2
    assert f["answer_rows_sample"].default == 30
    # DB / execution
    assert f["sql_row_limit"].default == 100
    # Schema
    assert f["schema_include_description"].default is True
    assert f["table_name"].default == "gaming_mental_health"
    # SQL correction
    assert f["sql_correction_enabled"].default is False
    assert f["max_sql_correction_retries"].default == 3
    assert f["sql_analytics_judge_enabled"].default is False
    assert f["sql_analytics_judge_correction_enabled"].default is False
    assert f["max_sql_analytics_correction_retries"].default == 3
    # Result validation
    assert f["result_validation_enabled"].default is True
    assert f["result_validation_correction_enabled"].default is False
    assert f["max_result_validation_correction_retries"].default == 3
    # Answer grounding
    assert f["answer_grounding_judge_enabled"].default is False
    assert f["answer_grounding_judge_correction_enabled"].default is False
    assert f["max_answer_grounding_correction_retries"].default == 3


# ---------------------------------------------------------------------------
# Env override
# ---------------------------------------------------------------------------


def test_env_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-4")
    cfg = PipelineConfig(openrouter_api_key="test-key", _env_file=None)
    assert cfg.openrouter_model == "anthropic/claude-opus-4"


def test_bool_field_env_override(monkeypatch):
    monkeypatch.setenv("SQL_CORRECTION_ENABLED", "true")
    cfg = PipelineConfig(openrouter_api_key="test-key", _env_file=None)
    assert cfg.sql_correction_enabled is True


def test_int_field_env_override(monkeypatch):
    monkeypatch.setenv("MAX_SQL_CORRECTION_RETRIES", "7")
    cfg = PipelineConfig(openrouter_api_key="test-key", _env_file=None)
    assert cfg.max_sql_correction_retries == 7


# ---------------------------------------------------------------------------
# Missing / invalid API key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        PipelineConfig(_env_file=None)


def test_empty_api_key_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="", _env_file=None)


def test_whitespace_api_key_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="   ", _env_file=None)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def test_bool_fields_accept_true_string(monkeypatch):
    # WHY: pydantic-settings coerces "true" / "false" string env vars to bool
    monkeypatch.setenv("RESULT_VALIDATION_ENABLED", "false")
    cfg = PipelineConfig(openrouter_api_key="test-key", _env_file=None)
    assert cfg.result_validation_enabled is False


def test_path_fields_are_path_objects():
    cfg = PipelineConfig(openrouter_api_key="test-key", _env_file=None)
    assert isinstance(cfg.db_path, Path)
    assert isinstance(cfg.metadata_db_path, Path)


def test_path_fields_accept_string_override(tmp_path):
    db = tmp_path / "test.sqlite"
    cfg = PipelineConfig(openrouter_api_key="test-key", db_path=str(db), _env_file=None)
    assert cfg.db_path == db


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------


def test_temperature_out_of_range_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="test-key", sql_temperature=3.0, _env_file=None)


def test_temperature_negative_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="test-key", answer_temperature=-0.1, _env_file=None)


def test_temperature_boundary_values_accepted():
    cfg = PipelineConfig(openrouter_api_key="test-key", sql_temperature=0.0, answer_temperature=2.0, _env_file=None)
    assert cfg.sql_temperature == 0.0
    assert cfg.answer_temperature == 2.0


def test_retries_zero_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="test-key", max_sql_correction_retries=0, _env_file=None)


def test_sql_row_limit_zero_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="test-key", sql_row_limit=0, _env_file=None)


def test_llm_timeout_zero_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(openrouter_api_key="test-key", llm_timeout_seconds=0, _env_file=None)


# ---------------------------------------------------------------------------
# Constructor kwargs override env
# ---------------------------------------------------------------------------


def test_constructor_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "from-env")
    cfg = PipelineConfig(openrouter_api_key="test-key", openrouter_model="from-kwarg", _env_file=None)
    assert cfg.openrouter_model == "from-kwarg"
