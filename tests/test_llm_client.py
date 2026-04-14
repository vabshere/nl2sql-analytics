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
    import time as _time

    from helpers import make_llm_client
    from src.conversation import Conversation, ConversationTurn
    from src.llm_client import LLMTokenLimitError, OpenRouterLLMClient, _is_retryable_llm_error
    from src.types import IntentClassificationOutput, SummarizationOutput
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


# ---------------------------------------------------------------------------
# Shared conversation helpers
# ---------------------------------------------------------------------------


def _conv_turn(question: str = "How many users?", sql: str = "SELECT COUNT(*) FROM t", answer: str = "42") -> ConversationTurn:
    return ConversationTurn(question=question, sql=sql, answer=answer, status="success", timestamp=_time.time())


def _conv_with_one_turn() -> Conversation:
    conv = Conversation()
    conv.add_turn(_conv_turn())
    return conv


# ---------------------------------------------------------------------------
# Prompt builder tests — Phase 1: SQL generation
# ---------------------------------------------------------------------------


class TestBuildSqlGenerationMessages:
    def test_message_structure(self):
        # DDL must appear in user message; validation-only keys must not leak in.
        ctx = {
            "ddl": "CREATE TABLE t (x INT)",
            "tables": {"t"},
            "columns": {"t": {"x"}},
            "column_types": {"t": {"x": "INT"}},
        }
        msgs = OpenRouterLLMClient._build_sql_generation_messages("how many rows?", ctx)
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        assert "CREATE TABLE t" in msgs[1]["content"]
        assert "column_types" not in msgs[1]["content"]

    def test_correction_hint_present_when_set(self):
        ctx = {"ddl": "CREATE TABLE t (x INT)", "tables": {"t"},
               "correction_hint": "no such column y"}
        msgs = OpenRouterLLMClient._build_sql_generation_messages("q", ctx)
        assert "no such column y" in msgs[1]["content"]

    def test_empty_context_does_not_raise(self):
        msgs = OpenRouterLLMClient._build_sql_generation_messages("q", {})
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# Prompt builder tests — Phase 2: Answer generation
# ---------------------------------------------------------------------------


class TestBuildAnswerGenerationMessages:
    def test_columns_rows_and_count(self):
        # Columns from first row listed; total row count shown;
        # up to 30 rows serialised in prompt body
        rows = [{"Gender": "M", "Score": i} for i in range(40)]
        msgs = OpenRouterLLMClient._build_answer_generation_messages("q", "SELECT ...", rows)
        user = msgs[1]["content"]
        assert "Gender" in user and "Score" in user   # columns present
        assert "40" in user                            # total count
        rows_section = user.split("ROWS:\n")[1].split("\n\nROW COUNT")[0]
        assert len(json.loads(rows_section)) == 30     # capped at 30

    def test_correction_hint_present_when_set(self):
        rows = [{"x": 1}]
        msgs = OpenRouterLLMClient._build_answer_generation_messages("q", "SELECT ...", rows, correction_hint="foo bar")
        assert "foo bar" in msgs[1]["content"]



# ---------------------------------------------------------------------------
# Prompt builder tests — Phase 3: Judge prompts
# ---------------------------------------------------------------------------


class TestBuildSqlJudgeMessages:
    def test_schema_question_sql_ordering(self):
        # User message must present context in SCHEMA → QUESTION → SQL order.
        ctx = {"ddl": "CREATE TABLE t (x INT)"}
        msgs = OpenRouterLLMClient._build_sql_judge_messages("q", "SELECT x FROM t", ctx)
        user_content = msgs[1]["content"]
        assert user_content.index("SCHEMA") < user_content.index("QUESTION") < user_content.index("SQL")


class TestBuildGroundingJudgeMessages:
    def test_sql_excluded_all_rows_passed(self):
        # SQL must not appear in user prompt (grounding judge checks answer vs rows only).
        # All rows passed — no cap — so the judge sees the full result set.
        rows = [{"x": i} for i in range(20)]
        sql = "SELECT x FROM t"
        msgs = OpenRouterLLMClient._build_grounding_judge_messages("q", rows, "The answer")
        user_content = msgs[1]["content"]
        assert sql not in user_content
        rows_section = user_content.split("ROWS:\n")[1].split("\n\nANSWER")[0]
        assert len(json.loads(rows_section)) == 20


def _make_client(monkeypatch=None, timeout_seconds: int | None = None) -> OpenRouterLLMClient:
    """Thin wrapper around make_llm_client that honours timeout env overrides."""
    import os
    if monkeypatch and timeout_seconds is not None:
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", str(timeout_seconds))
    return make_llm_client(
        timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
    )


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
# Token limit (finish_reason == "length") handling
# ---------------------------------------------------------------------------


def _stub_response_with_finish_reason(finish_reason: str = "stop") -> MagicMock:
    """Build a minimal SDK response with a configurable finish_reason."""
    res = MagicMock()
    res.usage = None
    choice = MagicMock()
    choice.message.content = "some content"
    choice.finish_reason = finish_reason
    res.choices = [choice]
    return res


class TestTokenLimitHandling:
    def test_raises_llm_token_limit_error_when_finish_reason_is_length(self):
        # WHY: a truncated response will fail JSON parsing downstream; raising a
        # distinct error here gives a clear signal instead of a confusing parse error
        client = _make_client()
        client._client.chat.send = lambda **kw: _stub_response_with_finish_reason("length")
        with pytest.raises(LLMTokenLimitError):
            client._chat(messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=100)

    def test_does_not_raise_when_finish_reason_is_stop(self):
        client = _make_client()
        client._client.chat.send = lambda **kw: _stub_response_with_finish_reason("stop")
        result = client._chat(messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=100)
        assert result == "some content"

    def test_token_limit_error_is_not_retried(self):
        # WHY: retrying with same params produces the same truncated output — wasteful
        client = _make_client()
        call_count = 0

        def _length_response(**kw):
            nonlocal call_count
            call_count += 1
            return _stub_response_with_finish_reason("length")

        client._client.chat.send = _length_response
        with pytest.raises(LLMTokenLimitError):
            client._chat(messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=100)
        assert call_count == 1  # no retries

    def test_token_limit_error_is_not_retryable(self):
        assert _is_retryable_llm_error(LLMTokenLimitError("token cap hit")) is False


# ---------------------------------------------------------------------------
# Conversation context — SQL generation prompt builder
# ---------------------------------------------------------------------------


class TestBuildSqlGenMessagesConversationContext:
    def test_with_conversation_context_injects_block(self):
        ctx = {
            "ddl": "CREATE TABLE t (x INT)",
            "tables": {"t"},
            "conversation_context": "RECENT TURNS:\nTurn 1:\n  Question: Q1\n  SQL: SELECT x FROM t\n  Answer: 5",
        }
        msgs = OpenRouterLLMClient._build_sql_generation_messages("follow up?", ctx)
        user = msgs[1]["content"]
        assert "CONVERSATION CONTEXT" in user
        assert "RECENT TURNS" in user

    def test_context_appears_before_user_question(self):
        ctx = {
            "ddl": "CREATE TABLE t (x INT)",
            "tables": {"t"},
            "conversation_context": "some history",
        }
        msgs = OpenRouterLLMClient._build_sql_generation_messages("my question", ctx)
        user = msgs[1]["content"]
        assert user.index("CONVERSATION CONTEXT") < user.index("USER QUESTION")


# ---------------------------------------------------------------------------
# Conversation context — Answer generation prompt builder
# ---------------------------------------------------------------------------


class TestBuildAnswerGenMessagesConversationContext:
    def test_with_conversation_context_injects_block(self):
        rows = [{"x": 1}]
        msgs = OpenRouterLLMClient._build_answer_generation_messages(
            "q", "SELECT x FROM t", rows, conversation_context="Turn 1: Q1"
        )
        assert "CONVERSATION CONTEXT" in msgs[1]["content"]
        assert "Turn 1: Q1" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Summarization prompt builder and summarize_turns()
# ---------------------------------------------------------------------------


class TestBuildSummarizationMessages:
    def test_formats_turns_into_user_message(self):
        turns = [_conv_turn(question="How many users?", sql="SELECT COUNT(*) FROM t", answer="42")]
        msgs = OpenRouterLLMClient._build_summarization_messages(turns)
        user = msgs[1]["content"]
        assert "How many users?" in user
        assert "SELECT COUNT(*)" in user
        assert "42" in user

    def test_includes_null_sql_as_none_text(self):
        turns = [ConversationTurn(question="Delete all", sql=None, answer="Cannot answer", status="unanswerable", timestamp=_time.time())]
        msgs = OpenRouterLLMClient._build_summarization_messages(turns)
        assert "(none)" in msgs[1]["content"]

    def test_two_messages_system_then_user(self):
        turns = [_conv_turn()]
        msgs = OpenRouterLLMClient._build_summarization_messages(turns)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestSummarizeTurns:
    def test_returns_summary_on_success(self):
        client = make_llm_client()
        turns = [_conv_turn()]
        with patch.object(client, "_chat", return_value="- Q1 asked for count; answer 42"):
            result = client.summarize_turns(turns)
        assert result.summary == "- Q1 asked for count; answer 42"
        assert result.error is None

    def test_non_blocking_on_error(self):
        client = make_llm_client()
        turns = [_conv_turn()]
        with patch.object(client, "_chat", side_effect=RuntimeError("LLM down")):
            result = client.summarize_turns(turns)
        assert result.summary == ""
        assert result.error is not None
        assert "LLM down" in result.error

    def test_returns_summarization_output_type(self):
        client = make_llm_client()
        turns = [_conv_turn()]
        with patch.object(client, "_chat", return_value="summary text"):
            result = client.summarize_turns(turns)
        assert isinstance(result, SummarizationOutput)


# ---------------------------------------------------------------------------
# Intent classification prompt builder and classify_intent()
# ---------------------------------------------------------------------------


class TestBuildIntentMessages:
    def test_includes_conversation_history(self):
        conv = _conv_with_one_turn()
        msgs = OpenRouterLLMClient._build_intent_messages("break it down by gender", conv)
        user = msgs[1]["content"]
        assert "How many users?" in user  # from conv history
        assert "break it down by gender" in user

    def test_two_messages_system_then_user(self):
        msgs = OpenRouterLLMClient._build_intent_messages("q", _conv_with_one_turn())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestClassifyIntent:
    def test_returns_follow_up(self):
        client = make_llm_client()
        conv = _conv_with_one_turn()
        response = json.dumps({"intent": "follow_up", "reason": "refs prior result"})
        with patch.object(client, "_chat", return_value=response):
            result = client.classify_intent("What about women?", conv)
        assert result.intent == "follow_up"
        assert result.error is None

    def test_returns_new_query(self):
        client = make_llm_client()
        conv = _conv_with_one_turn()
        response = json.dumps({"intent": "new_query", "reason": "unrelated topic"})
        with patch.object(client, "_chat", return_value=response):
            result = client.classify_intent("What is the most popular game?", conv)
        assert result.intent == "new_query"
        assert result.error is None

    def test_fallback_follow_up_on_error(self):
        client = make_llm_client()
        conv = _conv_with_one_turn()
        with patch.object(client, "_chat", side_effect=RuntimeError("API error")):
            result = client.classify_intent("another question", conv)
        assert result.intent == "follow_up"  # WHY: safe fallback — over-inject context
        assert result.error is not None
        assert "API error" in result.error

    def test_returns_intent_classification_output_type(self):
        client = make_llm_client()
        conv = _conv_with_one_turn()
        response = json.dumps({"intent": "follow_up", "reason": "ref"})
        with patch.object(client, "_chat", return_value=response):
            result = client.classify_intent("q", conv)
        assert isinstance(result, IntentClassificationOutput)

    def test_returns_data_question(self):
        client = make_llm_client()
        conv = _conv_with_one_turn()
        response = json.dumps({"intent": "data_question", "reason": "asking about prior rows"})
        with patch.object(client, "_chat", return_value=response):
            result = client.classify_intent("What was the max score?", conv)
        assert result.intent == "data_question"
        assert result.error is None


# ---------------------------------------------------------------------------
# answer_from_context prompt builder and method
# ---------------------------------------------------------------------------


class TestBuildContextAnswerMessages:
    def test_includes_question_and_context(self):
        msgs = OpenRouterLLMClient._build_context_answer_messages(
            "What was the max?", "RECENT TURNS:\nTurn 1:\n  Question: Q1\n  Answer: 42"
        )
        user = msgs[1]["content"]
        assert "What was the max?" in user
        assert "RECENT TURNS" in user

    def test_two_messages_system_then_user(self):
        msgs = OpenRouterLLMClient._build_context_answer_messages("q", "some context")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestAnswerFromContext:
    def test_returns_answer_on_success(self):
        client = make_llm_client()
        with patch.object(client, "_chat", return_value="The max score was 10."):
            result = client.answer_from_context("What was the max?", "Turn 1: Q answered 10")
        assert result.answer == "The max score was 10."
        assert result.error is None

    def test_non_blocking_on_error(self):
        client = make_llm_client()
        with patch.object(client, "_chat", side_effect=RuntimeError("API down")):
            result = client.answer_from_context("q", "some context")
        assert result.error is not None
        assert "API down" in result.error

    def test_returns_answer_generation_output_type(self):
        from src.types import AnswerGenerationOutput
        client = make_llm_client()
        with patch.object(client, "_chat", return_value="answer"):
            result = client.answer_from_context("q", "context")
        assert isinstance(result, AnswerGenerationOutput)
