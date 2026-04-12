# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `README.md` for setup, data download, OpenRouter configuration, commands, benchmark reference numbers, and deliverable requirements.

## Architecture

The pipeline is a linear 4-stage chain invoked via `AnalyticsPipeline.run(question)`:

```
Question → [SQL Generation] → [SQL Validation] → [SQL Execution] → [Answer Generation] → PipelineOutput
```

**`src/pipeline.py`** — orchestrator. `AnalyticsPipeline` owns `SQLiteExecutor` and `OpenRouterLLMClient`. Schema context is built once at `__init__` (static schema, no per-request DB round-trip). `SQLValidator` is a stateless class-method chain.

**`src/llm_client.py`** — `OpenRouterLLMClient` wraps the OpenRouter SDK. `generate_sql` uses structured JSON output (`SQLResponse` Pydantic model) to enforce `{"sql": "..."}` schema. `generate_answer` is a free-form chat call. Per-call token stats accumulate in `self._stats` and are drained via `pop_stats()` after each stage (stats are per-stage, not global).

**`src/schema_context.py`** — builds the schema context dict from two sources: (1) `PRAGMA table_info` on the main DB for column names/types, (2) `data/schema_metadata.sqlite` for human-readable descriptions. Returns `{ddl, tables, columns, column_types}`. `ddl` goes into the SQL generation prompt; `tables`/`columns` are used by `SQLValidator` to catch unknown table references.

**`src/types.py`** — dataclass contracts for each stage output and the top-level `PipelineOutput`. Do not change field names or types — automated evaluation depends on this exact contract.

**`data/schema_metadata.sqlite`** — two tables: `table_descriptions` and `column_descriptions`. Seeded by `scripts/seed_schema_metadata.py`. Used to annotate DDL with human-readable column descriptions for better prompt context.

## Constraints

- `AnalyticsPipeline.run()` must return a `PipelineOutput` — automated evaluation depends on this.
- `total_llm_stats` integer fields (`llm_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`) must be `int`, not `float` — the SDK returns floats; cast explicitly.
- Do not modify `tests/test_public.py`.
- Only `SELECT` queries are permitted — `SQLValidator` rejects all other statement types.
- `SELECT 1` (no table reference) is rejected — queries must reference a real table.
- Always-true `WHERE` clauses (`WHERE 1=1`, `WHERE TRUE`) are rejected via `sqlglot.optimizer.simplify`.

## Key Design Decisions

- **Structured output for SQL**: `response_format=json_schema` is passed to the LLM so the response is always `{"sql": "..."}`. `_extract_sql` only does `json.loads` — no regex fallback. Non-JSON surfaces as an error, not silently swallowed.
- **Schema context built at init**: `load_schema_context` is called once in `__init__`, not per request.
- **`pop_stats()` drains per-stage**: the pipeline aggregates per-stage stats into `total_llm_stats`.
- **`SQLValidationError` is unused**: validation failures are communicated via `SQLValidationOutput.is_valid=False`, not raised exceptions, so the pipeline continues to the answer stage with a graceful "unanswerable" response.
