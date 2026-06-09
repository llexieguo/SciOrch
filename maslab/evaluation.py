from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def accuracy_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key) or "unknown")
        totals[value] += 1
        correct[value] += float(row.get("mca") or 0.0)

    return {
        value: {
            "total": total,
            "correct": correct[value],
            "accuracy": correct[value] / total if total else 0.0,
        }
        for value, total in sorted(totals.items())
    }


def accuracy_by_image_warning(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    labeled_rows = []
    for row in rows:
        label = "warning" if int(row.get("image_load_warning_count") or 0) > 0 else "clean"
        labeled = dict(row)
        labeled["image_warning_group"] = label
        labeled_rows.append(labeled)
    return accuracy_breakdown(labeled_rows, "image_warning_group")


def write_summary(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    method: str,
    model: str,
    input_path: Path,
    image_root: Path,
    started_at: datetime,
    finished_at: datetime,
    runtime_seconds: float,
    env: dict[str, Any],
    config: dict[str, Any],
) -> None:
    total = len(rows)
    error_rows = [row for row in rows if row.get("sample_status") == "error"]
    completed_rows = [row for row in rows if row.get("sample_status") != "error"]
    correct = sum(1 for row in rows if float(row.get("mca") or 0.0) >= 0.5)
    completed_correct = sum(1 for row in completed_rows if float(row.get("mca") or 0.0) >= 0.5)
    total_cost = sum(float(row.get("metrics", {}).get("cost") or 0.0) for row in rows)
    total_tokens = sum(int(row.get("metrics", {}).get("total_tokens") or 0) for row in rows)
    image_warning_rows = [row for row in rows if row.get("image_load_warnings")]
    image_warning_count = sum(len(row.get("image_load_warnings") or []) for row in rows)
    sample_latencies = [
        float(row.get("metrics", {}).get("latency_seconds") or 0.0)
        for row in rows
    ]
    summary = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "method": method,
        "model": model,
        "total": total,
        "completed": len(completed_rows),
        "failed": len(error_rows),
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "completed_accuracy": completed_correct / len(completed_rows) if completed_rows else 0.0,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "sample_latency_seconds_sum": sum(sample_latencies),
        "sample_latency_seconds_avg": sum(sample_latencies) / len(sample_latencies) if sample_latencies else 0.0,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(output_path.with_suffix(".summary.json")),
        "image_root": str(image_root),
        "image_load_warning_samples": len(image_warning_rows),
        "image_load_warning_count": image_warning_count,
        "accuracy_by_source": accuracy_breakdown(rows, "source"),
        "accuracy_by_image_warning": accuracy_by_image_warning(rows),
        "image_load_warnings": [
            {
                "sample_index": row.get("sample_index"),
                "id": row.get("id"),
                "warnings": row.get("image_load_warnings"),
            }
            for row in image_warning_rows
        ],
        "env": env,
        "errors": [
            {
                "sample_index": row.get("sample_index"),
                "id": row.get("id"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in error_rows
        ],
        "config": config,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
