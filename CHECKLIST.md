# Production Readiness Checklist

**Instructions:** Complete all sections below. Check the box when an item is implemented, and provide descriptions where requested. This checklist is a required deliverable.

---

## Approach

Describe how you approached this assignment and what key problems you identified and solved.

- [x] **System works correctly end-to-end**

**What were the main challenges you identified?**
```
1. Baseline benchmark script accessed PipelineOutput as a dict (subscript) — fixed by using attribute access.
2. Module import errors in scripts caused by isort reordering sys.path setup above project imports — fixed with try/except ImportError guards.
3. LLM returned unparsable responses: max_tokens was too small causing truncated responses, and JSON schema was not enforced — fixed by increasing max_tokens and enforcing response_format=json_schema via Pydantic.
4. The baseline had no schema context in prompts, causing the LLM to hallucinate column names and produce invalid SQL.
5. A DELETE statement in the public test wiped all rows before read-only enforcement was in place — discovered during debugging why queries returned empty results.
6. OpenRouter Python SDK has no published documentation, requiring inspection of source to understand structured output, retry, and timeout support.
```

**What was your approach?**
```
Built a linear pipeline with opt-in quality and correction layers around two core LLM calls (SQL generation and answer generation):

1. Schema enrichment: injected DDL with column descriptions from a project-owned metadata SQLite DB into the SQL generation prompt so the LLM uses real column names.
2. Structured output: enforced response_format=json_schema on the SQL generation call (Pydantic SQLResponse model) so the response is always parseable — no regex fallback needed.
3. SQL validation: 4-layer rule-based validator (parse, statement-type, schema reference, tautological WHERE) runs before execution and rejects invalid SQL without an LLM call.
4. Correction loops: a unified while-loop re-generates SQL on three independent triggers — execution error, LLM analytics judge failure, result validation signals — each with its own retry budget and correction history passed to the next attempt.
5. Answer quality: opt-in LLM-as-judge verifies answers are factually grounded in returned rows; a correction loop re-generates the answer (not the SQL) when the judge rejects it.
6. Observability: structlog structured JSON logging with per-request contextvars; optional OpenTelemetry tracing with Arize Phoenix integration; per-stage token counting.
7. Centralised config: PipelineConfig (pydantic-settings) is the single source of truth for all env vars; all non-deterministic components are behind feature flags (disabled by default).
```

---

## Observability

- [x] **Logging**
  - Description: Structured JSON logs emitted to stderr via structlog. Configured once at package import via `src/logging_config.py`. `structlog.contextvars` bind `pipeline_run_id`, `model`, `request_id`, and `stage` for the duration of each request so every log line carries correlation context automatically. `logger.exception()` used in all except blocks to include full tracebacks. Pretty-print format available via `LOG_FORMAT=pretty`. Key events: SQL/answer generation completed (DEBUG), pipeline run completed (INFO with per-stage timings and token totals), correction triggers (DEBUG/WARNING).

- [x] **Metrics**
  - Description: Per-stage token usage (prompt_tokens, completion_tokens, total_tokens) and llm_calls collected after each LLM call via `pop_stats()`, then aggregated into `PipelineOutput.total_llm_stats`. Judges drain their own stats internally. Stage wall-clock timing captured in `timing_ms` on each stage output and in `PipelineOutput.timings`. All fields available to the benchmark script and callers for efficiency evaluation.

- [x] **Tracing**
  - Description: OpenTelemetry with Arize Phoenix exporter. Controlled by `OTLP_ENABLED` env var (default: false — zero-cost no-op when disabled). Spans cover every pipeline stage, each LLM judge call, and each conversation method. FastAPIInstrumentor auto-instruments HTTP spans and propagates W3C traceparent headers so HTTP spans are parents of pipeline spans. `OTLP_INCLUDE_SQL` (default: true) controls whether SQL text is recorded as a span attribute.

---

## Validation & Quality Assurance

- [x] **SQL validation**
  - Description: 4-layer rule-based validator using sqlglot. Layer 1: syntax parse (sqlglot.parse_one raises on malformed SQL). Layer 2: statement type — only SELECT is allowed; DELETE/INSERT/UPDATE/DROP are rejected immediately. Layer 3: schema reference check — table names are verified against the live schema extracted at pipeline init; column name validation is not performed. Layer 4: tautological WHERE detection — `WHERE 1=1` / `WHERE TRUE` patterns are caught via sqlglot.optimizer.simplify and rejected. Validation failures produce `is_valid=False`; the pipeline continues to the answer stage with an "unanswerable" status rather than raising an exception.

- [x] **Answer quality**
  - Description: Opt-in LLM-as-judge controlled by `ANSWER_GROUNDING_JUDGE_ENABLED` (default: false). Returns a typed `AnswerGroundingJudgeOutput` with `verdict` (bool), `grade` (fail/borderline/pass), `issues`, and `reason`. Non-blocking — errors in the judge are caught and logged without failing the pipeline. An opt-in correction loop (`ANSWER_GROUNDING_JUDGE_CORRECTION_ENABLED`) re-calls `generate_answer()` with a `correction_hint` containing the prior answer and judge issues; budget controlled by `MAX_ANSWER_GROUNDING_CORRECTION_RETRIES`.

- [x] **Result consistency**
  - Description: Rule-based `ResultValidator` runs after every SQL execution. Detects: `empty_result` (zero rows returned), `all_null_column` (a column is entirely NULL), `all_zero_column` (a column is entirely 0), `rows_truncated` (fetchmany(100) hit the cap and a 101st row exists). Flags are informational only and stored in `PipelineOutput.result_validation` — they do not block answer generation. An opt-in correction loop (`RESULT_VALIDATION_CORRECTION_ENABLED`) re-generates SQL when flags are raised.

- [x] **Error handling**
  - Description: Errors handled explicitly at every stage. The unified correction loop in `_run_impl()` handles all SQL failure modes with priority ordering: execution error → analytics judge failure → result validation flags. Each trigger has an independent retry counter and max-retries config. Full correction history (prior SQL + reason) is accumulated and injected into each subsequent generation call. Judges never raise — they catch all exceptions internally and return an output with `error` set. LLM retries via tenacity target only transient errors (429/500/503); auth errors fail immediately.

---

## Maintainability

- [x] **Code organization**
  - Description: `src/` module split by concern: `pipeline.py` (orchestrator), `llm_client.py` (LLM calls), `config.py` (settings), `conversation.py` (multi-turn), `server.py` (HTTP), `schema_context.py` (schema introspection), `types.py` (output contracts), `tracing.py` (OTel setup), `logging_config.py` (structlog setup). Files are 150–500 lines. Feature/domain split rather than type split. `AnalyticsPipeline` is stateless with respect to conversation; `ConversationSession` wraps it — clear separation of concerns.

- [x] **Configuration**
  - Description: `PipelineConfig(BaseSettings)` from pydantic-settings is the single source of truth for all configuration. Reads from `.env` file and OS environment; constructor kwargs override both (used in tests for DI). Field validators enforce non-empty API key, temperature range (0.0–2.0), valid reasoning effort enum, and positive numeric bounds. All env vars documented in `.env.example`. `LOG_LEVEL` and `LOG_FORMAT` are the only env vars read directly (before any PipelineConfig is instantiated) in `logging_config.py`.

- [x] **Error handling**
  - Description: HTTP input validation at the server boundary (blank question → 422, question exceeds `MAX_QUESTION_LENGTH` → 422 with limit in message). SQLiteExecutor opened with URI `mode=ro` — OS-level read-only enforcement that fails immediately if the file doesn't exist. Graceful shutdown via FastAPI lifespan: `pipeline.close()` and OTel `provider.shutdown()` flush spans on exit.

- [x] **Documentation**
  - Description: `CLAUDE.md` documents every module's purpose, key design decisions, constraints, and full env var list. `README.md` provides setup instructions, available commands, and benchmark reference numbers. Inline `# WHY:` comments explain non-obvious decisions throughout source. `.env.example` provides a complete annotated env var reference. `SOLUTION_NOTES.md` summarises what changed and why.

---

## LLM Efficiency

- [x] **Token usage optimization**
  - Description: Schema context (DDL + column descriptions) in the SQL generation prompt reduces hallucinated column names and avoids correction-loop retries, which would cost more tokens overall. Answer rows capped at `ANSWER_ROWS_SAMPLE` (default 30) before serialisation to prevent oversized prompts. Temperature set to 0.0 for SQL generation and judges (deterministic), 0.2 for answer generation. Structured output via json_schema response_format eliminates follow-up extraction calls. tiktoken (cl100k_base) used for token-accurate conversation history windowing. Reasoning effort is configurable per stage (`SQL_REASONING_EFFORT`, `ANSWER_REASONING_EFFORT`, `SQL_JUDGE_REASONING_EFFORT`, `ANSWER_JUDGE_REASONING_EFFORT`) and omitted from the API call when unset, allowing non-reasoning models to ignore it.

- [x] **Efficient LLM requests**
  - Description: All optional LLM calls — analytics judge, answer grounding judge, intent classification, conversation summarisation — are disabled by default and incur zero cost when their feature flags are off. Schema context is built once at `AnalyticsPipeline.__init__` (no per-request DB round-trips). When conversation history is enabled, only turns exceeding the token limit trigger an LLM summarisation call; recent turns are injected verbatim. A 60-second timeout (`LLM_TIMEOUT_SECONDS`) bounds all API calls.

---

## Testing

- [x] **Unit tests**
  - Description: 15 test files covering each pipeline component in isolation using `BaseLLMStub` (deterministic LLM mock). Key modules: `test_sql_validator.py`, `test_sqlite_executor.py`, `test_result_validator.py`, `test_llm_client.py`, `test_pipeline_error_recovery.py`, `test_answer_grounding_judge.py`, `test_tracing.py`, `test_config.py`, `test_server.py`.

- [x] **Integration tests**
  - Description: Marked `@pytest.mark.integration`; require a real `OPENROUTER_API_KEY`. `tests/test_public.py` (protected, unmodified) tests the full single-turn pipeline end-to-end. `tests/test_integration_conversation.py` covers multi-turn conversation flows with real LLM calls. Run with `pytest -m integration`.

- [x] **Performance tests**
  - Description: `scripts/benchmark.py` runs N iterations of all public prompts reporting avg/p50/p95 latency (ms), success rate, avg prompt/completion/total tokens, and avg LLM calls per request. Provides the reference harness for before/after comparison.

- [x] **Edge case coverage**
  - Description: Unanswerable questions (answerable=False from LLM), DELETE/mutating requests (rejected by validator → unanswerable status), empty result sets, all-null and all-zero columns, row truncation at 100-row cap, questions exceeding max_question_length, malformed SQL, conversation summarisation trigger at token limit, data_question intent that bypasses SQL entirely, correction loop budget exhaustion.

---

## Optional: Multi-Turn Conversation Support

**Only complete this section if you implemented the optional follow-up questions feature.**

- [x] **Intent detection for follow-ups**
  - Description: `classify_intent()` makes a structured LLM call (via `_IntentResponse` Pydantic model) that returns one of three intents: `new_query` (no context injected), `follow_up` (context injected into SQL generation), `data_question` (answer from context, bypass SQL). Controlled by `INTENT_PREDICTION_ENABLED` (only active when `CONVERSATION_HISTORY_ENABLED=true`).

- [x] **Context-aware SQL generation**
  - Description: For `follow_up` intent, a formatted CONVERSATION CONTEXT section is prepended to the SQL generation user prompt containing: the LLM-generated summary of older turns (if any), then the N most recent verbatim turns (question + SQL + answer). The LLM resolves ambiguous references using the prior SQL and results visible in this context.

- [x] **Context persistence**
  - Description: `ConversationSession` owns a `Conversation` instance per HTTP session (keyed by session_id in `app.state.sessions`). `Conversation` stores `ConversationTurn` dataclasses (immutable records). Token counting via tiktoken. When verbatim history exceeds `CONVERSATION_CONTEXT_TOKEN_LIMIT` (default 2000 tokens), older turns beyond `CONVERSATION_HISTORY_WINDOW` (default 5) are LLM-summarised into a running summary string. Sessions are in-memory, lost on server restart.

- [x] **Ambiguity resolution**
  - Description: Ambiguous follow-up references are resolved implicitly by the SQL generation LLM. The conversation context injected into the prompt includes the previous SQL query and the answer text, giving the model sufficient information to infer what "it", "them", "males", or "the highest value" refers to without an explicit disambiguation step.

**Approach summary:**
```
Implemented as a stateless pipeline (AnalyticsPipeline) wrapped by a stateful ConversationSession layer. This separation ensures the core pipeline remains independently testable and the multi-turn logic is fully opt-in.

Each request to ConversationSession.run() goes through three steps:
1. Proactive summarisation: if verbatim history is over the token limit, old turns are compressed via an LLM summarise call and stored as a running summary.
2. Intent classification: an LLM structured-output call classifies the question as new_query, follow_up, or data_question.
3. Dispatch: new_query → pipeline with no context; follow_up → pipeline with context injected into the SQL prompt; data_question → answer_from_context() bypasses SQL entirely and answers from the summary + recent turns.

Every turn (including errors) is stored so the history is accurate for subsequent model calls.
```

---

## Production Readiness Summary

**What makes your solution production-ready?**
```
- Centralised validated configuration (PipelineConfig) with documented env vars and sane defaults for all optional features
- Multi-layer SQL validation that rejects invalid, destructive, and logically trivial queries before execution
- Read-only database access enforced at the OS level (SQLite URI mode=ro)
- Explicit error handling at every stage; no silent failures; correction budgets prevent infinite loops
- Structured logging with per-request correlation IDs and stage context; optional OTel tracing with Phoenix integration
- HTTP server with input validation (blank/oversized questions rejected at the boundary) and session management
- Feature flags for all non-deterministic, high-cost components — disabled by default so the baseline cost profile is preserved
- Exponential-backoff retry on transient LLM errors; LLM timeout bounded to 60s
- Comprehensive test suite: unit tests with deterministic LLM stubs + integration tests with real API calls
```

**Key improvements over baseline:**
```
- Schema context (DDL + column descriptions) injected into SQL generation prompt — eliminates hallucinated column names
- Structured JSON output enforced (response_format=json_schema + Pydantic) — eliminates unparseable LLM responses
- Token counting implemented — per-stage and total, required for efficiency evaluation
- 4-layer SQL validator (sqlglot-based) — catches parse errors, destructive statements, bad table/column refs, tautological WHERE clauses
- Unified correction loop — handles execution errors, analytics judge failures, and result validation signals with shared history
- LLM-as-judge for SQL analytics and answer grounding — opt-in quality floor
- Result validator — detects empty, null-column, zero-column, truncated results as monitoring signals
- Structlog structured logging + OpenTelemetry tracing — production-grade observability
- FastAPI HTTP server with session management — enables real client integration
- Multi-turn conversation with intent classification and token-aware summarisation — optional but production-ready
```

**Known limitations or future work:**
```
- Sessions are in-memory only — lost on server restart; a persistent session store would be needed for production
- Benchmark runs sequentially — not representative of concurrent production traffic
- Single LLM for all tasks — separate models optimised per task would reduce cost and latency
- More data context needed — the LLM makes assumptions about value scales that could be wrong; enriching column descriptions with domain context would improve SQL quality
- finish_reason="length" not handled as continuation — partial LLM output surfaces as an error rather than triggering a continuation call
- No mechanism for updating schema descriptions — changes to the source table schema or column semantics require manual re-seeding of the metadata DB and a server restart
- No indexes on the target DB table — query latency is O(n) full scans; outside this project's ownership boundary
- Few-shot examples not included in the SQL generation prompt — adding validated question/SQL pairs could improve generation quality for edge cases
- No persistence of input/output data — storing request-response pairs would enable future feedback loops, fine-tuning, and regression testing
- No semantic caching — similar or identical questions re-run the full pipeline; a vector store of past validated outputs could serve cached answers and reduce latency and cost for repeated queries
```

---

## Benchmark Results

Include your before/after benchmark results here.

Model: `openai/gpt-5-nano`

**Baseline (if you measured):**

_Schema context as dict, max_tokens increased:_
- Average latency: `21425 ms`
- p50 latency: `22131 ms`
- p95 latency: `29582 ms`
- Success rate: `25 %`
- Average tokens per request: `2691`
- Average LLM calls per request: `1.25`

_DDL schema context + increased max_tokens + Pydantic json_schema enforcement:_
- Average latency: `23376 ms`
- p50 latency: `23355 ms`
- p95 latency: `29782 ms`
- Success rate: `92 %`
- Average tokens per request: `3123`
- Average LLM calls per request: `1.92`

**Your solution:**

_Judges off:_
- Average latency: `4274 ms`
- p50 latency: `4100 ms`
- p95 latency: `5938 ms`
- Success rate: `94.4 %`

_All judges on, 3 max retries each:_
- Average latency: `10539 ms`
- p50 latency: `9632 ms`
- p95 latency: `17882 ms`
- Success rate: `100 %`

**LLM efficiency:**

_Judges off:_
- Average tokens per request: `1256`
- Average LLM calls per request: `1.94`

_All judges on, 3 max retries each:_
- Average tokens per request: `3848`
- Average LLM calls per request: `4.72`

---

**Completed by:** Vaibhav Mishra
**Date:** 14/04/2026
**Time spent:** 12–17 hours (10–13h core pipeline, 2–4h multi-turn optional)
