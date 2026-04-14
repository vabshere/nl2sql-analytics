# Analytics Pipeline

Natural language to SQL analytics pipeline over a single SQLite table, with multi-layer validation, structured logging, optional OpenTelemetry tracing, LLM-as-judge quality checks, and optional multi-turn conversation support.

---

## Setup

**Requirements:** Python 3.13+

```bash
pip install -r requirements.txt
```

**Data** — download `gaming_mental_health_10M_40features.csv` from [Kaggle](https://www.kaggle.com/datasets/sharmajicoder/gaming-and-mental-health?select=gaming_mental_health_10M_40features.csv), place in `data/`, then:

```bash
python3 scripts/gaming_csv_to_db.py
python3 scripts/seed_schema_metadata.py
```

**API key:**

```bash
cp .env.example .env
# set OPENROUTER_API_KEY in .env
# or: export OPENROUTER_API_KEY=<your_key>
```

---

## Running

**Single question:**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.pipeline import AnalyticsPipeline
with AnalyticsPipeline() as p:
    print(p.run('What is the average addiction level by gender?').answer)
"
```

**HTTP server:**

```bash
uvicorn src.server:app --reload
```

```
POST /run
{"question": "...", "session_id": "optional-for-multi-turn"}
```

**Benchmark:**

```bash
python3 scripts/benchmark.py --runs 3
```

Reference (judges off): avg ~4274ms, p50 ~4100ms, p95 ~5938ms, 94% success, ~1256 tokens/req.

---

## Tests

```bash
# Unit tests — no API key required
pytest tests/ -m "not integration" --ignore=tests/test_public.py

# Integration tests — requires OPENROUTER_API_KEY
pytest -m integration
```

---

## Configuration

See `.env.example` for the full list. Key variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required |
| `OPENROUTER_MODEL` | `openai/gpt-5-nano` | LLM model |
| `LOG_FORMAT` | `json` | `json` or `pretty` |
| `OTLP_ENABLED` | `false` | Enable Phoenix tracing |
| `PHOENIX_ENDPOINT` | `http://localhost:6006/v1/traces` | OTel collector |

**Feature flags** — all default off, enable to add quality layers:

| Flag | What it enables |
|------|-----------------|
| `SQL_CORRECTION_ENABLED` | Re-generate SQL on execution error |
| `SQL_ANALYTICS_JUDGE_ENABLED` | LLM judge for SQL analytical correctness |
| `SQL_ANALYTICS_JUDGE_CORRECTION_ENABLED` | Re-generate SQL when analytics judge fails |
| `RESULT_VALIDATION_CORRECTION_ENABLED` | Re-generate SQL on empty/null/zero result signals |
| `ANSWER_GROUNDING_JUDGE_ENABLED` | LLM judge for answer factual grounding |
| `ANSWER_GROUNDING_JUDGE_CORRECTION_ENABLED` | Re-generate answer when grounding judge fails |
| `CONVERSATION_HISTORY_ENABLED` | Multi-turn conversation support |

---

## Docs

| File | Contents |
|------|----------|
| `CLAUDE.md` | Architecture, design decisions, module reference |
| `CHECKLIST.md` | Production readiness checklist |
| `SOLUTION_NOTES.md` | Engineering notes: what changed, why, benchmark impact |
| `ORIGINAL_README.md` | Original assignment brief |
