"""
Thin FastAPI HTTP server for the analytics pipeline.

Single endpoint: POST /run — wraps AnalyticsPipeline.run().

Startup (lifespan):
  1. Construct PipelineConfig from env / .env file
  2. configure_tracing(config) — installs Phoenix or NoOp provider
  3. Instrument FastAPI for automatic HTTP spans (W3C traceparent propagation)
  4. Construct AnalyticsPipeline

Shutdown (lifespan):
  1. pipeline.close() — closes SQLite connection
  2. provider.shutdown() — flushes BatchSpanProcessor before exit

WHY lifespan over @app.on_event: lifespan is the recommended FastAPI pattern
and guarantees cleanup runs even when the server exits abnormally.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import PipelineConfig
from src.pipeline import AnalyticsPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    config = PipelineConfig()

    # WHY: instrument AFTER src import (which called configure_tracing) so the FastAPIInstrumentor uses
    # the real provider; instrumenting before would capture the NoOp provider.
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi not installed; HTTP spans disabled")

    pipeline = AnalyticsPipeline(config=config)
    app.state.pipeline = pipeline

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    # WHY: close SQLite connection before flushing spans so any in-flight
    # db.execute spans complete and are recorded before the exporter shuts down
    pipeline.close()

    from opentelemetry import trace as _trace
    provider = _trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()


app = FastAPI(lifespan=lifespan)


class RunRequest(BaseModel):
    question: str
    request_id: str | None = None


class RunResponse(BaseModel):
    status: str
    answer: str
    sql: str | None = None


@app.post("/run", response_model=RunResponse)
async def run_pipeline(body: RunRequest) -> RunResponse:
    """Run the analytics pipeline for a natural-language question.

    The incoming W3C traceparent header (if present) is extracted by the
    FastAPIInstrumentor, so pipeline.run() spans are automatically children
    of the HTTP span — no manual context propagation required.
    """
    result = app.state.pipeline.run(body.question, body.request_id)
    return RunResponse(
        status=result.status,
        answer=result.answer,
        sql=result.sql,
    )


if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
