# Solution Notes

## What changed

- **Baseline fixes**: benchmark script used dict subscript on `PipelineOutput` (attribute access fix); `isort` reordered `sys.path` manipulation above project imports on every save (`try/except ImportError` guard fix); `max_tokens` too small causing truncated LLM responses
- **Schema enrichment**: project-owned `data/schema_metadata.sqlite` with table and column descriptions; DDL annotated with descriptions injected into SQL generation prompt at init
- **Structured output**: `response_format=json_schema` + Pydantic `SQLResponse` enforced on SQL generation; `_extract_sql` does plain `json.loads` — no regex fallback
- **SQL validation**: 4-layer sqlglot validator — syntax parse, SELECT-only enforcement, table name reference check, tautological WHERE detection
- **Correction loops**: unified loop in `_run_impl()` for three independent triggers (execution error, analytics judge failure, result validation flags), each with own retry budget and accumulated `(sql, reason)` history passed to the next attempt
- **Answer quality**: opt-in LLM judges for SQL analytics and answer grounding; opt-in answer correction loop; rule-based result validator for empty/null/zero/truncated signals
- **Observability**: structlog structured JSON logging with per-request contextvars; per-stage token counting aggregated into `total_llm_stats`; optional OpenTelemetry tracing with Arize Phoenix
- **Config**: `PipelineConfig(BaseSettings)` as single source of truth; feature flags for all optional components defaulting off
- **HTTP server**: FastAPI `POST /run` with input validation and in-memory session routing via `ConversationSession`
- **Multi-turn (optional)**: intent classification (new_query / follow_up / data_question), token-aware LLM summarisation of older turns via tiktoken

## Why

- **Schema context + json_schema**: `gpt-5-nano` produced 0% success without both — schema context gave the LLM real column names; json_schema was the only reliable way to get parseable output
- **Feature flags default off**: callers get baseline cost/latency by default and opt into quality checks progressively; always-on judges would add cost to every request including straightforward ones
- **Unified correction loop**: feeding accumulated error history prevents re-generating the same broken SQL on subsequent attempts
- **Single LLM for all tasks**: avoids per-task model config complexity; per-task model optimisation left as future work

## Measured impact

Model: `openai/gpt-5-nano`

| Step | Success | Avg ms | Avg tokens | LLM calls |
|------|---------|--------|------------|-----------|
| Baseline (no context, max_tokens too small) | 0% | 9,119 | 256 | 1.0 |
| + Schema context + increased max_tokens | 25% | 21,425 | 2,691 | 1.25 |
| + Pydantic json_schema enforcement | 92% | 23,376 | 3,123 | 1.92 |
| Final — judges off | 94% | 4,274 | 1,256 | 1.94 |
| **Final — all judges on, 3 retries** | **100%** | **10,539** | **3,848** | **4.72** |

With judges off the pipeline runs at 4.3s avg and 94% success — schema context and correction loops alone get most of the way there. Enabling all judges adds ~6s and ~2.5k tokens to reach 100%.

## Tradeoffs and next steps

**Tradeoffs:**
- Judges and correction loops default off — callers opt into quality checks progressively rather than paying judge cost on every request
- Single LLM for all tasks — simpler config; a smaller model for intent classification and a stronger one for SQL generation would be more cost-efficient in production
- Column name validation excluded from SQL validator — avoids false positives on aliased/computed columns at the cost of catching fewer invalid references before execution
- Stateless pipeline + stateful session wrapper — keeps `AnalyticsPipeline` independently testable; session state lives only in the wrapper layer

**Next steps:**
- Schema description update mechanism without server restart
- Few-shot examples in the SQL generation prompt for domain-specific edge cases
- Persistent input/output store for feedback loops, fine-tuning, and regression testing
- Semantic cache over past validated question→SQL→answer triples to reduce cost on repeated queries
- Concurrent load benchmark to measure realistic production throughput
