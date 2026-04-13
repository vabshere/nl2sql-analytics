# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `README.md` for setup, data download, OpenRouter configuration, commands, benchmark reference numbers, and deliverable requirements.

## Architecture

The pipeline is a linear chain with optional quality sub-stages, invoked via `AnalyticsPipeline.run(question)`:

```
Question
  → [SQL Generation]
  → [SQL Validation]          (rule-based: syntax, type, table refs, tautological WHERE)
  → [SQL Analytics Judge]     (opt-in LLM: analytical correctness — aggregation, GROUP BY, granularity)
  → [SQL Execution]
  → [Result Validation]       (opt-in rule-based: empty result, all-null/all-zero column signals)
  → [Answer Generation]
  → [Answer Grounding Judge]  (opt-in LLM: is the answer grounded in the rows?)
  → PipelineOutput
```

**`src/pipeline.py`** — orchestrator. `AnalyticsPipeline` owns `SQLiteExecutor` and `OpenRouterLLMClient`. Schema context is built once at `__init__` (static schema, no per-request DB round-trip). `SQLValidator` and `ResultValidator` are stateless class-method chains. The two LLM judges are wired directly in `run()` and both write to existing `intermediate_outputs` fields — no new blocking stages.

**`src/llm_client.py`** — `OpenRouterLLMClient` wraps the OpenRouter SDK. `generate_sql` uses structured JSON output (`SQLResponse` Pydantic model) to enforce `{"sql": "..."}` schema. `generate_answer` is a free-form chat call. `judge_sql_analytics` and `judge_answer_grounding` are non-blocking judge calls — both return typed `SQLAnalyticsJudgeOutput` / `AnswerGroundingJudgeOutput` objects, catch and log errors gracefully, and never raise. Each judge drains `pop_stats()` internally and stores the result in its `llm_stats` field; the pipeline then adds those stats to `total_llm_stats`. Per-call token stats accumulate in `self._stats` and are drained via `pop_stats()` after each stage (stats are per-stage, not global). `_chat()` applies exponential-backoff retry (via `tenacity`) on transient errors (429, 500, 503); non-retryable errors fail immediately.

**`src/schema_context.py`** — builds the schema context dict from two sources: (1) `PRAGMA table_info` on the main DB for column names/types, (2) `data/schema_metadata.sqlite` for human-readable descriptions. Returns `{ddl, tables, columns, column_types}`. `ddl` goes into the SQL generation prompt; `tables`/`columns` are used by `SQLValidator` to catch unknown table references. `_parse_bool_env` is imported by `pipeline.py` for env var parsing.

**`src/types.py`** — dataclass contracts for each stage output and the top-level `PipelineOutput`. Do not change field names or types on `PipelineOutput` or its stage outputs — automated evaluation depends on this exact contract. `JudgeResponse` (Pydantic) uses `grade: Literal["fail", "borderline", "pass"]` (not `score`) plus `Field(description=...)` on every field to guide structured output. `JudgeOutput` adds `llm_stats: dict[str, Any]` (defaults to `{}`) so per-judge token usage is accessible to the pipeline. `ResultValidationOutput` is informational flags only. `PipelineOutput.result_validation` is a new optional field appended at the end with a default factory. `SQLExecutionOutput.rows_truncated: bool` (default `False`) signals when `fetchmany(100)` silently dropped rows.

**`data/schema_metadata.sqlite`** — two tables: `table_descriptions` and `column_descriptions`. Seeded by `scripts/seed_schema_metadata.py`. Used to annotate DDL with human-readable column descriptions for better prompt context.

## Constraints

- `AnalyticsPipeline.run()` must return a `PipelineOutput` — automated evaluation depends on this.
- `total_llm_stats` integer fields (`llm_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`) must be `int`, not `float` — the SDK returns floats; cast explicitly.
- Do not modify `tests/test_public.py`.
- Only `SELECT` queries are permitted — `SQLValidator` rejects all other statement types.
- `SELECT 1` (no table reference) is rejected — queries must reference a real table.
- Always-true `WHERE` clauses (`WHERE 1=1`, `WHERE TRUE`) are rejected via `sqlglot.optimizer.simplify`.

## Test file conventions

Project imports in test files must come after `sys.path.insert` and be wrapped in a `try-except ImportError` block so isort/ruff do not reorder them above the path manipulation:

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.foo import Bar
except ImportError:
    raise
```

## Env Vars

See `.env.example` for the full list with descriptions.

## Key Design Decisions

- **Structured output for SQL**: `response_format=json_schema` is passed to the LLM so the response is always `{"sql": "..."}`. `_extract_sql` only does `json.loads` — no regex fallback. Non-JSON surfaces as an error, not silently swallowed.
- **Schema context built at init**: `load_schema_context` is called once in `__init__`, not per request.
- **`pop_stats()` drains per-stage**: the pipeline aggregates per-stage stats into `total_llm_stats`. This includes judge stats — each judge drains `pop_stats()` internally and the pipeline adds `verdict.llm_stats` for both judges to the total.
- **`SQLValidationError` is unused**: validation failures are communicated via `SQLValidationOutput.is_valid=False`, not raised exceptions, so the pipeline continues to the answer stage with a graceful "unanswerable" response.
- **LLM judges are non-blocking**: both `judge_sql_analytics` and `judge_answer_grounding` catch all exceptions internally, log a warning, and return a typed output object (`SQLAnalyticsJudgeOutput` / `AnswerGroundingJudgeOutput`) with `error` set. The pipeline never fails due to a judge error. Verdicts are stored in `intermediate_outputs` only — they do not affect `status`, `sql`, `rows`, or `answer`.
- **`ResultValidator` flags are informational**: `empty_result` / `all_null_column` / `all_zero_column` are monitoring signals only. They do not block the pipeline. The empty-rows answer path is already handled by `generate_answer`'s fallback message.
- **Status determination covers all stages**: `pipeline.run()` checks errors in pipeline order — sql generation → validation → execution → answer generation. `answer_output.error` is explicitly mapped to `status="error"`; earlier errors take precedence.
- **Row truncation probe is conditional**: `SQLiteExecutor` calls `cur.fetchone()` only when `len(rows) == 100` to detect a 101st row. Fewer than 100 rows skip the probe entirely. Result surfaces as `SQLExecutionOutput.rows_truncated`.
- **LLM timeout is bounded**: `_chat()` passes `timeout=self._timeout` (default 60s, env `LLM_TIMEOUT_SECONDS`) to every SDK call so a hung API call cannot block the pipeline indefinitely.
- **LLM retries are transient-only**: `_is_retryable_llm_error()` gates `tenacity` retries to 429/500/503 only. Auth and bad-request errors fail immediately. Attempt cap is `LLM_MAX_RETRIES` (default 3).
