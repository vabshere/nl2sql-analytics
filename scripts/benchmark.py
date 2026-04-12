from __future__ import annotations
import statistics
import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from scripts.gaming_csv_to_db import (
        DEFAULT_CSV_PATH,
        DEFAULT_DB_PATH,
        DEFAULT_TABLE_NAME,
        csv_to_sqlite,
    )
    from src.pipeline import AnalyticsPipeline
except ImportError as exc:
    raise RuntimeError("Could not import project modules. Run from project root.") from exc


def _ensure_gaming_db() -> Path:
    """Ensure gaming mental health DB exists; create from CSV if missing."""
    if not DEFAULT_DB_PATH.exists():
        csv_to_sqlite(DEFAULT_CSV_PATH, DEFAULT_DB_PATH, DEFAULT_TABLE_NAME, if_exists="replace")
    return DEFAULT_DB_PATH


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of full prompt-set repetitions.")
    args = parser.parse_args()

    db_path = _ensure_gaming_db()
    root = Path(__file__).resolve().parents[1]
    prompts_path = root / "tests" / "public_prompts.json"

    pipeline = AnalyticsPipeline(db_path=db_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    totals: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []
    llm_calls: list[int] = []
    success = 0
    count = 0

    for _ in range(args.runs):
        for prompt in prompts:
            result = pipeline.run(prompt)
            totals.append(result.timings["total_ms"])
            success += int(result.status == "success")
            count += 1
            stats = result.total_llm_stats
            prompt_tokens.append(stats.get("prompt_tokens", 0))
            completion_tokens.append(stats.get("completion_tokens", 0))
            total_tokens.append(stats.get("total_tokens", 0))
            llm_calls.append(stats.get("llm_calls", 0))

    summary = {
        "runs": args.runs,
        "samples": count,
        "success_rate": round(success / count, 4) if count else 0.0,
        "avg_ms": round(statistics.fmean(totals), 2) if totals else 0.0,
        "p50_ms": round(percentile(totals, 50), 2),
        "p95_ms": round(percentile(totals, 95), 2),
        "avg_prompt_tokens": round(statistics.fmean(prompt_tokens), 1) if prompt_tokens else 0.0,
        "avg_completion_tokens": round(statistics.fmean(completion_tokens), 1) if completion_tokens else 0.0,
        "avg_total_tokens": round(statistics.fmean(total_tokens), 1) if total_tokens else 0.0,
        "avg_llm_calls": round(statistics.fmean(llm_calls), 2) if llm_calls else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
