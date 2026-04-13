from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.llm_client import OpenRouterLLMClient
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


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
