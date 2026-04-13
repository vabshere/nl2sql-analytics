"""Smoke test for configure_logging.

WHY: structlog.testing.capture_logs() replaces the processor chain, so custom
processors don't run inside it. We test configure_logging() directly to verify
the setup doesn't raise.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.logging_config import configure_logging
except ImportError:
    raise


def test_configure_logging(monkeypatch):
    """configure_logging() runs without error in both supported formats."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()

    monkeypatch.setenv("LOG_FORMAT", "pretty")
    configure_logging()
