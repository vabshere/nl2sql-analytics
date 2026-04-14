from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# WHY: resolved at import time so db_path / metadata_db_path defaults are
# always absolute — avoids relative-path ambiguity when CWD ≠ project root
_BASE_DIR = Path(__file__).resolve().parents[1]


class PipelineConfig(BaseSettings):
    """Centralised, type-safe configuration for the analytics pipeline.

    All fields are read from environment variables (uppercase, e.g.
    OPENROUTER_API_KEY → openrouter_api_key). Constructor kwargs take the
    highest priority and override both env vars and the .env file — use this
    in tests to inject specific values without mutating the environment.

    LOG_LEVEL and LOG_FORMAT are intentionally absent — logging is configured
    before any PipelineConfig is instantiated (src/__init__.py), so those two
    vars are read directly by configure_logging().
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # WHY: unknown env vars (e.g. PATH, HOME) are silently ignored so
        # adding new fields later never breaks existing deployments
        extra="ignore",
    )

    # ── LLM / API ─────────────────────────────────────────────────────────────
    openrouter_api_key: str
    openrouter_model: str = "openai/gpt-5-nano"
    # Maximum seconds to wait for a single LLM API call
    llm_timeout_seconds: int = 60
    # Retry cap for transient LLM errors (429, 500, 503)
    llm_max_retries: int = 3

    # ── LLM generation params ─────────────────────────────────────────────────
    # token budgets per stage
    sql_max_tokens: int = 10000
    answer_max_tokens: int = 220
    sql_judge_max_tokens: int = 20000
    answer_judge_max_tokens: int = 20000
    # sampling temperatures — judges default to 0.0 (deterministic grading)
    sql_temperature: float = 0.0
    answer_temperature: float = 0.2
    sql_judge_temperature: float = 0.0
    answer_judge_temperature: float = 0.0
    # reasoning effort per stage — omitted from API call when None
    # valid values: xhigh | high | medium | low | minimal | none
    sql_reasoning_effort: str | None = None
    answer_reasoning_effort: str | None = None
    sql_judge_reasoning_effort: str | None = None
    answer_judge_reasoning_effort: str | None = None
    # max rows serialised into the answer-generation prompt
    answer_rows_sample: int = 30

    # ── DB / execution ────────────────────────────────────────────────────────
    db_path: Path = _BASE_DIR / "data" / "gaming_mental_health.sqlite"
    metadata_db_path: Path = _BASE_DIR / "data" / "schema_metadata.sqlite"
    # fetchmany(N) cap — keeps result payloads bounded
    sql_row_limit: int = 100

    # ── Schema ────────────────────────────────────────────────────────────────
    # Include human-readable column descriptions as DDL comments in LLM prompts
    schema_include_description: bool = True
    table_name: str = "gaming_mental_health"

    # ── SQL generation & correction ───────────────────────────────────────────
    sql_correction_enabled: bool = False
    max_sql_correction_retries: int = 3
    sql_analytics_judge_enabled: bool = False
    sql_analytics_judge_correction_enabled: bool = False
    max_sql_analytics_correction_retries: int = 3

    # ── Result validation ─────────────────────────────────────────────────────
    result_validation_enabled: bool = True
    result_validation_correction_enabled: bool = False
    max_result_validation_correction_retries: int = 3

    # ── Answer grounding ──────────────────────────────────────────────────────
    answer_grounding_judge_enabled: bool = False
    answer_grounding_judge_correction_enabled: bool = False
    max_answer_grounding_correction_retries: int = 3

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator(
        "sql_reasoning_effort",
        "answer_reasoning_effort",
        "sql_judge_reasoning_effort",
        "answer_judge_reasoning_effort",
    )
    @classmethod
    def effort_must_be_valid(cls, v: str | None) -> str | None:
        _VALID_EFFORTS = {"xhigh", "high", "medium", "low", "minimal", "none"}
        if v is not None and v not in _VALID_EFFORTS:
            raise ValueError(f"reasoning effort must be one of {sorted(_VALID_EFFORTS)}, got {v!r}")
        return v

    @field_validator("openrouter_api_key")
    @classmethod
    def api_key_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("OPENROUTER_API_KEY must not be empty or whitespace")
        return v

    @field_validator("sql_temperature", "answer_temperature", "sql_judge_temperature", "answer_judge_temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator(
        "sql_row_limit",
        "answer_rows_sample",
        "llm_timeout_seconds",
        "llm_max_retries",
        "max_sql_correction_retries",
        "max_sql_analytics_correction_retries",
        "max_result_validation_correction_retries",
        "max_answer_grounding_correction_retries",
        "sql_max_tokens",
        "answer_max_tokens",
        "sql_judge_max_tokens",
        "answer_judge_max_tokens",
        mode="before",
    )
    @classmethod
    def must_be_positive(cls, v: object) -> object:
        # WHY: mode="before" so the raw value (may be a string from env) is
        # checked before pydantic casts it to int — avoids a confusing
        # int-cast success followed by a domain error later
        if int(v) < 1:
            raise ValueError(f"must be ≥ 1, got {v!r}")
        return v
