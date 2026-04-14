"""
Tracing tests — verify OTel span creation without a live Phoenix instance.

Uses InMemorySpanExporter + SimpleSpanProcessor so every span is captured
synchronously.  The span_exporter fixture installs a fresh TracerProvider
for each test and resets to NoOp on teardown, keeping tests isolated.

Test organisation:
  - Pipeline-level spans: pipeline.run, pipeline.sql_cycle, db.execute,
    pipeline.sql_validate — exercised via BaseLLMStub (no real LLM).
  - LLM-level spans: llm.generate_sql, llm.chat (GenAI semconv) — exercised
    by patching _chat() directly on OpenRouterLLMClient.
  - Log bridge: trace_id/span_id injected into structlog events.
  - Server: POST /run endpoint, lifespan shutdown.
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode
    from src.config import PipelineConfig
    from src.pipeline import AnalyticsPipeline, SQLiteExecutor, SQLValidator
    from src.tracing import configure_tracing, get_tracer
    from tests.helpers import BaseLLMStub, _ZERO_STATS
except ImportError:
    raise


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _reset_tracer_provider() -> None:
    """Reset the OTel global TracerProvider so it can be re-installed.

    WHY: trace.set_tracer_provider() is intentionally a one-shot operation in
    production, guarded by a Once lock.  Tests need to install fresh providers
    per-test, so we reset the internal state directly.  This is the standard
    approach used by the OTel Python SDK's own test suite.

    We also clear _real_tracer on any module-level ProxyTracer instances
    (pipeline._tracer, llm_client._tracer).  ProxyTracer caches the resolved
    real tracer the first time it is used; after a provider swap the cached
    tracer still points to the old provider, so spans land in the wrong
    exporter.  Setting _real_tracer = None forces re-resolution against the
    newly installed provider on the next span creation.
    """
    import opentelemetry.trace as _t
    _t._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    _t._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    # Reset cached real tracers on instrumented modules
    try:
        from src import pipeline as _pipeline_mod
        from src import llm_client as _llm_mod
        for _mod in [_pipeline_mod, _llm_mod]:
            _proxy = getattr(_mod, "_tracer", None)
            if _proxy is not None and hasattr(_proxy, "_real_tracer"):
                _proxy._real_tracer = None
    except ImportError:
        pass


@pytest.fixture
def span_exporter():
    """Install an in-memory OTel provider; yield the exporter; reset to NoOp."""
    _reset_tracer_provider()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    # WHY: reset fully so the next test gets a clean slate
    _reset_tracer_provider()


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    spans: dict[str, list] = {}
    for s in exporter.get_finished_spans():
        spans.setdefault(s.name, []).append(s)
    return spans


# ── Helper: minimal pipeline config wired to a test DB ───────────────────────

@pytest.fixture
def tracing_pipeline(analytics_db_with_data, schema_description_db, span_exporter):
    """AnalyticsPipeline with stub LLM and in-memory exporter active."""
    config = PipelineConfig(
        openrouter_api_key="test-key",
        db_path=analytics_db_with_data,
        metadata_db_path=schema_description_db,
        otlp_enabled=True,   # does not start Phoenix — provider is already set by fixture
        otlp_include_sql=True,
    )
    stub = BaseLLMStub()
    pipeline = AnalyticsPipeline(config=config, llm_client=stub)
    yield pipeline, span_exporter
    pipeline.close()


# ── Phase 1: foundation ───────────────────────────────────────────────────────

class TestTracingConfig:
    def test_otlp_disabled_by_default(self, analytics_db, schema_description_db):
        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db,
            metadata_db_path=schema_description_db,
        )
        assert config.otlp_enabled is False

    def test_otlp_fields_are_read_from_config(self, analytics_db, schema_description_db):
        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db,
            metadata_db_path=schema_description_db,
            otlp_enabled=True,
            phoenix_endpoint="http://localhost:6006/v1/traces",
            phoenix_project_name="my-project",
            otlp_include_sql=False,
        )
        assert config.phoenix_endpoint == "http://localhost:6006/v1/traces"
        assert config.phoenix_project_name == "my-project"
        assert config.otlp_include_sql is False

    def test_configure_tracing_disabled_installs_noop(self, analytics_db, schema_description_db):
        _reset_tracer_provider()
        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db,
            metadata_db_path=schema_description_db,
            otlp_enabled=False,
        )
        configure_tracing(config)
        provider = trace.get_tracer_provider()
        assert isinstance(provider, trace.NoOpTracerProvider)
        _reset_tracer_provider()  # clean up for next test


# ── Phase 2: pipeline spans ───────────────────────────────────────────────────

class TestPipelineSpans:
    def test_root_span_created(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        by_name = _spans_by_name(exporter)
        assert "pipeline.run" in by_name, "expected pipeline.run span"

    def test_root_span_has_pipeline_run_id(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        root = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.run")
        assert "pipeline.run_id" in root.attributes
        assert root.attributes["pipeline.run_id"]  # non-empty UUID string

    def test_root_span_has_model_attribute(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        root = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.run")
        assert "pipeline.model" in root.attributes

    def test_root_span_has_status_attribute(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        root = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.run")
        assert "pipeline.status" in root.attributes
        assert root.attributes["pipeline.status"] == "success"

    def test_sql_cycle_span_created(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        by_name = _spans_by_name(exporter)
        assert "pipeline.sql_cycle" in by_name, "expected pipeline.sql_cycle span"

    def test_sql_cycle_span_is_child_of_root(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        root = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.run")
        cycle = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.sql_cycle")

        assert cycle.parent is not None
        assert cycle.parent.span_id == root.context.span_id

    def test_sql_cycle_has_cycle_attempt_zero_initially(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        cycle = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.sql_cycle")
        assert cycle.attributes.get("pipeline.cycle_attempt") == 0

    def test_sql_validate_span_created(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        by_name = _spans_by_name(exporter)
        assert "pipeline.sql_validate" in by_name, "expected pipeline.sql_validate span"

    def test_sql_validate_span_has_is_valid_attribute(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        validate_span = next(s for s in exporter.get_finished_spans() if s.name == "pipeline.sql_validate")
        assert "sql.is_valid" in validate_span.attributes

    def test_db_execute_span_created(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        by_name = _spans_by_name(exporter)
        assert "db.execute" in by_name, "expected db.execute span"

    def test_db_execute_has_system_attribute(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        db_span = next(s for s in exporter.get_finished_spans() if s.name == "db.execute")
        assert db_span.attributes.get("db.system") == "sqlite"

    def test_db_execute_includes_sql_when_configured(self, analytics_db_with_data, schema_description_db, span_exporter):
        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
            otlp_include_sql=True,
        )
        stub = BaseLLMStub()
        with AnalyticsPipeline(config=config, llm_client=stub) as pipeline:
            pipeline.run("What is the average age?")

        db_span = next(s for s in span_exporter.get_finished_spans() if s.name == "db.execute")
        assert "db.statement" in db_span.attributes

    def test_db_execute_omits_sql_when_configured(self, analytics_db_with_data, schema_description_db, span_exporter):
        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
            otlp_include_sql=False,
        )
        stub = BaseLLMStub()
        with AnalyticsPipeline(config=config, llm_client=stub) as pipeline:
            pipeline.run("What is the average age?")

        db_span = next(s for s in span_exporter.get_finished_spans() if s.name == "db.execute")
        assert "db.statement" not in db_span.attributes

    def test_db_execute_has_row_count(self, tracing_pipeline):
        pipeline, exporter = tracing_pipeline
        pipeline.run("What is the average age?")

        db_span = next(s for s in exporter.get_finished_spans() if s.name == "db.execute")
        assert "db.row_count" in db_span.attributes

    def test_tracing_disabled_no_pipeline_spans(self, analytics_db_with_data, schema_description_db):
        # Install a fresh in-memory provider first so we can verify nothing lands in it
        _reset_tracer_provider()
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
            otlp_enabled=False,
        )
        # WHY: configure_tracing resets provider to NoOp, so spans from the
        # pipeline are never emitted — the in-memory exporter stays empty
        _reset_tracer_provider()
        configure_tracing(config)

        stub = BaseLLMStub()
        with AnalyticsPipeline(config=config, llm_client=stub) as pipeline:
            pipeline.run("What is the average age?")

        assert exporter.get_finished_spans() == ()
        _reset_tracer_provider()  # clean up

    def test_error_recorded_on_root_span_when_pipeline_errors(
        self, analytics_db_with_data, schema_description_db, span_exporter
    ):
        from src.types import SQLGenerationOutput

        class ErrorLLMStub(BaseLLMStub):
            def generate_sql(self, question, context):
                return SQLGenerationOutput(
                    sql=None,
                    answerable=False,
                    timing_ms=0.0,
                    llm_stats=dict(_ZERO_STATS),
                    error="simulated failure",
                )

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
        )
        with AnalyticsPipeline(config=config, llm_client=ErrorLLMStub()) as pipeline:
            result = pipeline.run("trigger error")

        root = next(s for s in span_exporter.get_finished_spans() if s.name == "pipeline.run")
        assert root.status.status_code == StatusCode.ERROR
        assert result.status == "unanswerable"

    def test_correction_loop_produces_multiple_sql_cycle_spans(
        self, analytics_db_with_data, schema_description_db, span_exporter
    ):
        """When the first SQL attempt fails validation and correction is enabled,
        a second pipeline.sql_cycle span should be created for the correction."""
        from src.types import SQLGenerationOutput
        from unittest.mock import MagicMock

        call_count = 0

        class _FirstFailStub(BaseLLMStub):
            def generate_sql(self, question, context):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: return an invalid SQL (bad table name triggers validation)
                    return SQLGenerationOutput(
                        sql="SELECT * FROM nonexistent_table",
                        answerable=True,
                        timing_ms=0.0,
                        llm_stats=dict(_ZERO_STATS),
                    )
                # Subsequent calls: return valid SQL
                return SQLGenerationOutput(
                    sql="SELECT age FROM gaming_mental_health",
                    answerable=True,
                    timing_ms=0.0,
                    llm_stats=dict(_ZERO_STATS),
                )

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
            sql_correction_enabled=True,
            max_sql_correction_retries=1,
        )
        with AnalyticsPipeline(config=config, llm_client=_FirstFailStub()) as pipeline:
            pipeline.run("What is the average age?")

        cycle_spans = [s for s in span_exporter.get_finished_spans() if s.name == "pipeline.sql_cycle"]
        assert len(cycle_spans) >= 2, f"expected ≥2 sql_cycle spans, got {len(cycle_spans)}"

        # Second span should have cycle_attempt=1
        attempts = sorted(s.attributes.get("pipeline.cycle_attempt", -1) for s in cycle_spans)
        assert 0 in attempts
        assert 1 in attempts


# ── Phase 3: LLM spans ────────────────────────────────────────────────────────

class TestLLMSpans:
    def test_generate_sql_span_created(self, analytics_db_with_data, schema_description_db, span_exporter):
        """generate_sql() should produce an llm.generate_sql span when the real client is used."""
        from unittest.mock import patch
        from src.llm_client import OpenRouterLLMClient
        from src.types import SQLResponse

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
        )
        client = OpenRouterLLMClient(config=config)

        sql_response = json.dumps({"sql": "SELECT age FROM gaming_mental_health", "answerable": True})
        with patch.object(client, "_chat", return_value=sql_response):
            result = client.generate_sql("What is the average age?", {"ddl": "", "tables": set()})

        by_name = _spans_by_name(span_exporter)
        assert "llm.generate_sql" in by_name

    def test_chat_span_created_with_genai_attributes(self, analytics_db_with_data, schema_description_db, span_exporter):
        """_chat() should produce an llm.chat span with GenAI semantic convention attributes."""
        from unittest.mock import MagicMock, patch
        from src.llm_client import OpenRouterLLMClient

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
        )
        client = OpenRouterLLMClient(config=config)

        # Patch the internal _client (SDK) so no real HTTP call is made
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = json.dumps({"sql": "SELECT 1", "answerable": True})

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100.0
        mock_usage.completion_tokens = 50.0
        mock_usage.total_tokens = 150.0

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        client._client = MagicMock()
        client._client.chat.send.return_value = mock_response

        client._chat(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=100,
        )

        chat_spans = [s for s in span_exporter.get_finished_spans() if s.name == "llm.chat"]
        assert chat_spans, "expected an llm.chat span"

        attrs = chat_spans[0].attributes
        assert attrs.get("gen_ai.system") == "openrouter"
        assert attrs.get("gen_ai.request.model") == config.openrouter_model
        assert attrs.get("gen_ai.request.max_tokens") == 100
        assert attrs.get("gen_ai.usage.input_tokens") == 100
        assert attrs.get("gen_ai.usage.output_tokens") == 50

    def test_generate_sql_span_is_parent_of_chat_span(self, analytics_db_with_data, schema_description_db, span_exporter):
        """llm.chat should be a child of llm.generate_sql."""
        from unittest.mock import MagicMock
        from src.llm_client import OpenRouterLLMClient
        import json

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
        )
        client = OpenRouterLLMClient(config=config)

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = json.dumps({"sql": "SELECT age FROM gaming_mental_health", "answerable": True})
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        client._client = MagicMock()
        client._client.chat.send.return_value = mock_response

        client.generate_sql("What is the average age?", {"ddl": "", "tables": set()})

        generate_span = next(s for s in span_exporter.get_finished_spans() if s.name == "llm.generate_sql")
        chat_span = next(s for s in span_exporter.get_finished_spans() if s.name == "llm.chat")

        assert chat_span.parent is not None
        assert chat_span.parent.span_id == generate_span.context.span_id

    def test_judge_span_records_verdict(self, analytics_db_with_data, schema_description_db, span_exporter):
        """judge_sql_analytics span should include pipeline.judge_verdict and judge_grade."""
        from unittest.mock import MagicMock
        from src.llm_client import OpenRouterLLMClient
        from src.types import JudgeResponse
        import json

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
        )
        client = OpenRouterLLMClient(config=config)

        verdict_json = json.dumps({
            "verdict": True,
            "grade": "pass",
            "issues": [],
            "reason": "looks good",
        })
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = verdict_json
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        client._client = MagicMock()
        client._client.chat.send.return_value = mock_response

        client.judge_sql_analytics("q", "SELECT 1", {})

        judge_spans = [s for s in span_exporter.get_finished_spans() if s.name == "llm.judge_sql_analytics"]
        assert judge_spans, "expected llm.judge_sql_analytics span"
        attrs = judge_spans[0].attributes
        assert attrs.get("pipeline.judge_verdict") is True
        assert attrs.get("pipeline.judge_grade") == "pass"

    def test_judge_error_recorded_on_span_not_raised(self, analytics_db_with_data, schema_description_db, span_exporter):
        """When the judge LLM call fails, the error is on the span but not raised."""
        from unittest.mock import MagicMock
        from src.llm_client import OpenRouterLLMClient

        config = PipelineConfig(
            openrouter_api_key="test-key",
            db_path=analytics_db_with_data,
            metadata_db_path=schema_description_db,
        )
        client = OpenRouterLLMClient(config=config)

        client._client = MagicMock()
        client._client.chat.send.side_effect = RuntimeError("network error")

        # Must not raise
        result = client.judge_sql_analytics("q", "SELECT 1", {})
        assert result.error is not None

        judge_spans = [s for s in span_exporter.get_finished_spans() if s.name == "llm.judge_sql_analytics"]
        assert judge_spans, "expected llm.judge_sql_analytics span even on error"
        assert judge_spans[0].status.status_code == StatusCode.ERROR


# ── Phase 4: Log bridge ────────────────────────────────────────────────────────

class TestLogBridge:
    def test_trace_id_injected_into_structlog_event(self, span_exporter):
        """While inside an active span, structlog events should contain trace_id."""
        import structlog
        import io

        # Capture structlog output so we can inspect it
        captured_events: list[dict] = []

        def _capture_processor(logger_arg, method, event_dict):
            captured_events.append(dict(event_dict))
            raise structlog.DropEvent()  # suppress output

        from src.logging_config import configure_logging
        # Reconfigure logging to inject our capture processor before the renderer
        # NOTE: this is a test-only reconfiguration
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # inject_otel_context is added by configure_logging; we replicate it:
                lambda l, m, e: (
                    {**e, "trace_id": format(trace.get_current_span().get_span_context().trace_id, "032x"),
                     "span_id": format(trace.get_current_span().get_span_context().span_id, "016x")}
                    if trace.get_current_span().get_span_context().is_valid else e
                ),
                _capture_processor,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(10),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=io.StringIO()),
            cache_logger_on_first_use=False,
        )

        test_tracer = get_tracer("test.log_bridge")
        log = structlog.get_logger()

        with test_tracer.start_as_current_span("test.span"):
            log.info("hello from inside span")

        # At least one event should have trace_id
        events_with_trace = [e for e in captured_events if "trace_id" in e]
        assert events_with_trace, "expected trace_id in structlog event inside span"
        trace_id = events_with_trace[0]["trace_id"]
        assert len(trace_id) == 32  # 128-bit hex

    def test_no_trace_id_outside_span(self):
        """Outside an active span, trace_id should NOT appear in log events."""
        import structlog
        import io

        captured_events: list[dict] = []

        def _capture_processor(logger_arg, method, event_dict):
            captured_events.append(dict(event_dict))
            raise structlog.DropEvent()

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                lambda l, m, e: (
                    {**e, "trace_id": format(trace.get_current_span().get_span_context().trace_id, "032x")}
                    if trace.get_current_span().get_span_context().is_valid else e
                ),
                _capture_processor,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(10),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=io.StringIO()),
            cache_logger_on_first_use=False,
        )

        # Ensure no span is active (NoOp provider → invalid span context)
        trace.set_tracer_provider(trace.NoOpTracerProvider())

        log = structlog.get_logger()
        log.info("hello from outside span")

        events_with_trace = [e for e in captured_events if "trace_id" in e]
        assert not events_with_trace, "trace_id should not appear when no span is active"


# ── Phase 5: Server ───────────────────────────────────────────────────────────

class TestServer:
    def test_run_endpoint_returns_200(self, analytics_db_with_data, schema_description_db, monkeypatch, span_exporter):
        """POST /run with a valid question returns HTTP 200 with status and answer fields."""
        from fastapi.testclient import TestClient
        from tests.helpers import BaseLLMStub

        # Point config at test DBs via env vars
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("DB_PATH", str(analytics_db_with_data))
        monkeypatch.setenv("METADATA_DB_PATH", str(schema_description_db))

        # Import server AFTER monkeypatching env so PipelineConfig picks them up
        import importlib
        import src.server as server_module
        importlib.reload(server_module)

        # Inject stub LLM so no real API key is needed
        from src.server import app

        # Override the pipeline inside the app after lifespan runs
        with TestClient(app, raise_server_exceptions=True) as client:
            # Patch the pipeline's llm attribute to use a stub
            app.state.pipeline.llm = BaseLLMStub()
            response = client.post("/run", json={"question": "What is the average age?"})

        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "answer" in body

    def test_run_endpoint_returns_error_on_unanswerable(
        self, analytics_db_with_data, schema_description_db, monkeypatch, span_exporter
    ):
        """POST /run returns 200 with status=unanswerable when generate_sql reports an error.

        WHY: the pipeline's status logic maps sql=None + error set → 'unanswerable'.
        A real LLM client sets error when the call fails (not when answerable=False),
        so the stub mirrors that by providing an error message.
        """
        from fastapi.testclient import TestClient
        from src.types import SQLGenerationOutput

        class _UnanswerableStub(BaseLLMStub):
            def generate_sql(self, question, context):
                return SQLGenerationOutput(
                    sql=None,
                    answerable=False,
                    timing_ms=0.0,
                    llm_stats=dict(_ZERO_STATS),
                    error="question cannot be answered from available schema",
                )

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("DB_PATH", str(analytics_db_with_data))
        monkeypatch.setenv("METADATA_DB_PATH", str(schema_description_db))

        import importlib
        import src.server as server_module
        importlib.reload(server_module)

        from src.server import app

        with TestClient(app) as client:
            app.state.pipeline.llm = _UnanswerableStub()
            response = client.post("/run", json={"question": "delete all data"})

        assert response.status_code == 200
        assert response.json()["status"] == "unanswerable"
