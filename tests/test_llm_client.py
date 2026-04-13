from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.llm_client import OpenRouterLLMClient, _is_retryable_llm_error
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


def _make_client(monkeypatch=None, timeout_seconds: int | None = None) -> OpenRouterLLMClient:
    """Build a client with a stubbed OpenRouter dependency."""
    if monkeypatch and timeout_seconds is not None:
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", str(timeout_seconds))
    client = OpenRouterLLMClient.__new__(OpenRouterLLMClient)
    client.model = "stub"
    client._stats = {
        "llm_calls": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "total_tokens": 0,
    }
    client._client = MagicMock()
    import os
    client._timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    client._max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    return client


# ---------------------------------------------------------------------------
# Phase 3 — timeout passed to SDK
# ---------------------------------------------------------------------------


class TestChatTimeout:
    def _stub_response(self):
        """Minimal object that _chat() can parse without errors."""
        res = MagicMock()
        res.usage = None
        choice = MagicMock()
        choice.message.content = "stub content"
        res.choices = [choice]
        return res

    def test_timeout_passed_to_sdk(self, monkeypatch):
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
        client = _make_client(monkeypatch, timeout_seconds=30)
        captured_kwargs = {}

        def _fake_send(**kwargs):
            captured_kwargs.update(kwargs)
            return self._stub_response()

        client._client.chat.send = _fake_send
        client._chat(messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=100)
        assert "timeout_ms" in captured_kwargs
        assert captured_kwargs["timeout_ms"] == 30_000  # LLM_TIMEOUT_SECONDS=30 → 30_000ms

    def test_default_timeout_is_60(self, monkeypatch):
        monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
        client = _make_client()
        assert client._timeout == 60


# ---------------------------------------------------------------------------
# Retry on transient LLM errors
# ---------------------------------------------------------------------------


class TestRetryableErrorClassification:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("429 rate limit exceeded", True),
            ("500 internal server error", True),
            ("503 overloaded", True),
            ("401 unauthorized", False),
            ("JSON decode error", False),
            ("400 bad request", False),
        ],
    )
    def test_is_retryable_error_classification(self, message, expected):
        exc = RuntimeError(message)
        assert _is_retryable_llm_error(exc) is expected


class TestChatRetry:
    def _stub_response(self):
        res = MagicMock()
        res.usage = None
        choice = MagicMock()
        choice.message.content = "good response"
        res.choices = [choice]
        return res

    def test_retries_on_retryable_error_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_RETRIES", "3")
        client = _make_client(monkeypatch)
        call_count = 0

        def _flaky_send(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("429 rate limit exceeded")
            return self._stub_response()

        client._client.chat.send = _flaky_send
        result = client._chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
        )
        assert call_count == 3
        assert result == "good response"

    def test_no_retry_on_non_retryable_error(self):
        client = _make_client()
        call_count = 0

        def _auth_error(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("401 unauthorized")

        client._client.chat.send = _auth_error
        with pytest.raises(RuntimeError, match="401 unauthorized"):
            client._chat(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                max_tokens=100,
            )
        assert call_count == 1


# ---------------------------------------------------------------------------
# _extract_sql
# ---------------------------------------------------------------------------


class TestExtractSQL:
    def test_json_with_sql_key(self):
        text = json.dumps({"sql": "SELECT age FROM gaming_mental_health"})
        assert OpenRouterLLMClient._extract_sql(text) == "SELECT age FROM gaming_mental_health"

    @pytest.mark.parametrize(
        "text",
        [
            "SELECT age FROM gaming_mental_health",
            "",
        ],
    )
    def test_invalid_json_raises(self, text: str):
        # WHY: structured output guarantees JSON; non-JSON is an unexpected
        # API failure that must surface as an error, not be silently swallowed
        with pytest.raises(json.JSONDecodeError):
            OpenRouterLLMClient._extract_sql(text)
