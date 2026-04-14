"""
OpenTelemetry + Phoenix tracing initialisation.

Call configure_tracing() once at startup.  All modules obtain a tracer via
get_tracer(__name__); those tracers are ProxyTracers that forward to whatever
provider is installed at span-creation time, so module-level get_tracer()
calls are safe even before configure_tracing() runs.

When otlp_enabled=False (the default) a NoOpTracerProvider is installed — all
start_as_current_span calls become zero-cost no-ops with no conditional
checks required in pipeline code.

configure_tracing() is called with no arguments from src/__init__.py so that
every import path (library mode, benchmark, scripts) gets the correct provider
automatically — the same pattern configure_logging() uses.  Callers that have
a PipelineConfig available (server lifespan, tests) may pass it explicitly;
constructor kwargs always take the highest priority.
"""
from __future__ import annotations

import logging
import os

from opentelemetry import trace

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"
_DEFAULT_PROJECT = "analytics-pipeline"


def configure_tracing(config=None) -> None:
    """Initialise the global OTel TracerProvider.

    When called with no arguments (e.g. from src/__init__.py), reads
    OTLP_ENABLED, PHOENIX_ENDPOINT, and PHOENIX_PROJECT_NAME directly from
    the environment — no PipelineConfig or API key required.

    When called with a PipelineConfig (e.g. from server lifespan or tests),
    config values take highest priority over env vars.

    Safe to call multiple times: each call overwrites the global provider.
    """
    if config is not None:
        otlp_enabled = config.otlp_enabled
        phoenix_endpoint = config.phoenix_endpoint
        phoenix_project_name = config.phoenix_project_name
    else:
        # WHY: read env vars directly so configure_tracing() can be called
        # before PipelineConfig is instantiated (src/__init__.py runs before
        # any config object exists and must not require OPENROUTER_API_KEY)
        otlp_enabled = os.environ.get("OTLP_ENABLED", "false").lower() == "true"
        phoenix_endpoint = os.environ.get("PHOENIX_ENDPOINT", _DEFAULT_ENDPOINT)
        phoenix_project_name = os.environ.get("PHOENIX_PROJECT_NAME", _DEFAULT_PROJECT)

    if not otlp_enabled:
        trace.set_tracer_provider(trace.NoOpTracerProvider())
        return

    try:
        from phoenix.otel import register  # type: ignore[import-untyped]

        register(
            project_name=phoenix_project_name,
            endpoint=phoenix_endpoint,
        )
        # WHY: phoenix.otel.register() sets the global TracerProvider internally;
        # we do not need to call trace.set_tracer_provider() ourselves.
    except Exception:
        logger.exception("Phoenix tracing init failed; falling back to NoOp")
        trace.set_tracer_provider(trace.NoOpTracerProvider())


def get_tracer(name: str) -> trace.Tracer:
    """Return a ProxyTracer for name, regardless of what provider is installed.

    WHY: trace.get_tracer(name) delegates to the currently installed provider.
    If called after configure_tracing() installs a NoOpTracerProvider (e.g.
    from src/__init__.py), it returns a frozen NoOpTracer — tests cannot swap
    it out later.  Constructing ProxyTracer(name) directly always produces a
    lazy tracer that re-resolves against whatever provider is active at
    span-creation time, so import order and configure_tracing() call order do
    not matter.
    """
    from opentelemetry.trace import ProxyTracer
    return ProxyTracer(name)


# NOTE — HTTP API trace context propagation:
# opentelemetry-instrumentation-fastapi extracts the incoming W3C traceparent
# header and makes it the active context before the route handler runs.
# This means pipeline.run() spans are automatically children of the HTTP span
# with no changes to pipeline code.
# For other HTTP frameworks, call opentelemetry.propagate.extract(headers) at
# the handler entry point and pass the resulting context to
# tracer.start_as_current_span(..., context=ctx).
